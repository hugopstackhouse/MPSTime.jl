function make_objective(
    folds::AbstractVector,
    objective::TuningLoss, 
    distribute_folds::Bool,
    opts0::AbstractMPSOptions, 
    fields::Vector{Symbol}, 
    types::Vector{Type},
    X::AbstractMatrix, 
    y::AbstractVector, 
    windows::Union{Nothing, AbstractVector};
    logspace_eta::Bool=false
    )

    fieldnames = Tuple(fields)
    cache = Dict{Tuple{types...}, Float64}()

    function safe_paramlist(optslist::AbstractVector; output=false)
        optslist_safe = Vector{Union{AbstractFloat, Integer}}(undef, length(optslist))
        for (i, field) in enumerate(optslist)
            t = types[i]
            if t <: Integer
                rounded = round(Int,field)
                optslist_safe[i] = rounded
                if output && ~isapprox(field, rounded)
                    println("Integer parameter $(fieldnames[i])=$field rounded to $(rounded)!")
                end
            elseif logspace_eta && fieldnames[i] == :eta
                optslist_safe[i] = convert(t, 10^field)
            else
                optslist_safe[i] = convert(t, field)
            end

        end
        return optslist_safe
    end

    function tr_objective(optslist::AbstractVector, p)
        verbosity, tstart, nfolds = p

        optslist_safe = safe_paramlist(optslist; output=verbosity>=3)
        
        key = tuple(optslist_safe...)
        if haskey(cache, key )
            verbosity >= 1 && println("Cache hit!")
            loss = cache[key]
        else
            hparams = NamedTuple{fieldnames}(Tuple(optslist_safe))
            opts = _set_options(opts0; hparams...)
            
            if distribute_folds
                loss = @distributed (+) for fold in eachindex(folds)
                    (train_inds, val_inds) = folds[fold]
                    X_train, y_train, X_val, y_val = X[train_inds,:], y[train_inds], X[val_inds,:], y[val_inds]
                    # X_val, y_val = X_val[1:2, :], y_val[1:2]

                    verbosity >= 1 && println("cvfold $fold: t=$(rtime(tstart)): training MPS with $(hparams)... ")
                    mps, _... = fitMPS(X_train, y_train, opts);
                    # println(" done")

                    mean(eval_loss(objective, mps, X_val, y_val, windows; p_fold=(verbosity, tstart, fold, nfolds))) # eval_loss always returns an array
                end
            else
                loss = 0
                for (fold, (train_inds, val_inds)) in enumerate(folds)
                    X_train, y_train, X_val, y_val = X[train_inds,:], y[train_inds], X[val_inds,:], y[val_inds]
                    # X_val, y_val = X_val[1:2, :], y_val[1:2]

                    verbosity >= 1 && println("cvfold $fold: t=$(rtime(tstart)): training MPS with $(hparams)... ")
                    mps, _... = fitMPS(X_train, y_train, opts);
                    # println(" done")

                    loss += mean(eval_loss(objective, mps, X_val, y_val, windows; p_fold=(verbosity, tstart, fold, nfolds))) # eval_loss always returns an array
                end
            end
            loss /= nfolds
            
            
            cache[key] = loss
            verbosity >= 1 && println("t=$(rtime(tstart)): Mean CV Loss: $loss")
        end
        return loss
    end

    return tr_objective, cache, safe_paramlist
end

function tune_across_folds(
    folds::AbstractVector, 
    parameter_info::Tuple,
    tuning_settings::Tuple,
    X::AbstractMatrix,
    y::AbstractVector, 
    tstart::Real
    )
    x0, opts0, lb, ub, is_disc, fields, types = parameter_info
    objective, method, distribute_folds, nfolds, windows, abstol, maxiters, verbosity, provide_x0, logspace_eta = tuning_settings 

    tr_objective, cache, safe_params = make_objective(folds, objective, distribute_folds, opts0, fields, types, X, y, windows; logspace_eta=logspace_eta)
    p = (verbosity, tstart, nfolds)

    # for rapid debugging
    # tr_objective = (x,u...) -> begin @show x; return sum(x.^2) end

    if nfolds <= 1
        optslist_safe = safe_params(x0)

        return NamedTuple{Tuple(fields)}(Tuple(optslist_safe))
    end


    x0_adj = provide_x0 ? x0 : nothing
    obj = OptimizationFunction(tr_objective, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(obj, x0_adj, p; int=is_disc, lb=lb, ub=ub)
    sol = solve(prob, method; abstol=abstol, maxiters=maxiters)

    verbosity >= 5 && print(sol)
    optslist_safe = safe_params(sol.u)
    best_params = NamedTuple{Tuple(fields)}(Tuple(optslist_safe))
    return best_params

end

"""
    tune(
        X::AbstractMatrix, 
        y::AbstractVector, 
        parameters::NamedTuple,
        method=SAMIN(); # bounded simulated annealing from Optim
        opts0::AbstractMPSOptions=MPSOptions(; verbosity=-5, log_level=-1),
        objective::TuningLoss=ImputationLoss(), 
        nfolds::Integer=5,
        rng::Union{Integer, AbstractRNG}=1,
        foldmethod::Union{Function, Vector}=make_stratified_folds, 
        pms::Union{Nothing, AbstractVector}=collect(0.05:0.15:0.95),
        verbosity::Integer=1,
        abstol::Float64=1e-3,
        maxiters::Integer=500
    )

Train a hyperparameter tuned MPS model on X,y with 
# Arguments
...
- `method::TuningMethod`: either GridSearch or an algorithm from Optim (including Pswarm, simulatedAnnealing, grad desc methods if we want to be fancy )
- `objective::TuningLoss`: I'll setup a predetermined list including classification losses ('inaccuracy') or imputation metrics (MAE)
- `distributed::Bool`: I can use all the work I put in to optimise imputation with the distributed library here


"""
function tune(
        X::AbstractMatrix, 
        y::AbstractVector, 
        parameters::NamedTuple,
        method=SAMIN(); # bounded simulated annealing from Optim
        opts0::AbstractMPSOptions=MPSOptions(; verbosity=-5, log_level=-1),
        input_supertype::Type=Float64,
        objective::TuningLoss=ImputationLoss(), 
        nfolds::Integer=5,
        rng::Union{Integer, AbstractRNG}=1,
        foldmethod::Union{Function, Vector}=make_stratified_cvfolds, 
        pms::Union{Nothing, AbstractVector}=nothing, #TODO make default behaviour a bit better
        windows::Union{Nothing, AbstractVector, Dict}=nothing,
        verbosity::Integer=1,
        provide_x0::Bool=true,
        logspace_eta::Bool=false,
        abstol::Float64=1e-3,
        maxiters::Integer=500,
        distribute_folds::Bool=false,
        disable_nondistributed_threading::Bool=false,

    )
    # basic checks    
    abs_rng = rng isa Integer ? Xoshiro(rng) : rng

    if !(length(unique(keys(parameters))) == length(keys(parameters)))
       throw(ArgumentError("The 'parameters' argument contains duplicates!")) 
    end
    if objective isa ImputationLoss
        windows = make_windows(windows, pms, X, abs_rng)
    end

    
    is_disc = Vector{Bool}(undef, length(parameters))
    lb = Vector{input_supertype}(undef, length(parameters))
    ub = Vector{input_supertype}(undef, length(parameters))
    x0 = Vector{input_supertype}(undef, length(parameters))
    types = Vector{Type}(undef, length(parameters))

    


    # setup tuned hyperparameters
    for (i, (key, val)) in enumerate(pairs(parameters))
        startx = getproperty(opts0, key)
        param_type = typeof(startx)
        if !( param_type <: Number)
            throw(ArgumentError("Cannot tune '$key', only numeric types can be hyperoptimised."))
        end
        is_disc[i] = param_type <: Integer

        if !isempty(val)
            lb[i], ub[i] = convert(param_type, val[1]), convert(param_type, val[2])
        else
            lb[i], ub[i] = one(param_type), typemax(param_type)
        end

        if startx < lb[i] || startx > ub[i]
            startx = lb[i]
        end
        x0[i] = startx
        types[i] = param_type

    end

  
    # not super necessary, but its nice to have the result be independent of the order of the paramters vector
    fields = [keys(parameters)...]
    perm = sortperm(fields)

    for vec in [fields, types, x0, is_disc, lb, ub]
        permute!(vec, perm)
    end


    parameter_info = x0, opts0, lb, ub, is_disc, fields, types
    tuning_settings = objective, method, distribute_folds, nfolds, windows, abstol, maxiters, verbosity, provide_x0, logspace_eta

    if nfolds <= 1
        folds = []
    else
        # println("Generating Folds")
        folds::Vector = foldmethod isa Function ? foldmethod(X,y, nfolds; rng=abs_rng) : foldmethod
    end
    tstart = time()

    if disable_nondistributed_threading 
        
        GenericLinearAlgebra.LinearAlgebra.BLAS.set_num_threads(1)
        ITensors.Strided.disable_threads()
        @warn "Threading may still be active, if it is, try setting the environment variable OMP_NUM_THREADS=1 before launching julia. Alternatively, you can sidestep this issue by calling tune() with distribute_folds=true, num_procs=1"

    end


    return tune_across_folds(folds, parameter_info, tuning_settings, X, y, tstart)

end

#eval_loss returns an array of loss scores. This is either a singleton or imputation loss scores indexed by percentage missing



