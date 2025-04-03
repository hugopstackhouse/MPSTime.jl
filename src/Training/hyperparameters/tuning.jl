function make_objective(
    folds::AbstractVector,
    objective::TuningLoss, 
    workers::AbstractVector,
    distribute_folds::Bool,
    opts0::AbstractMPSOptions, 
    fields::AbstractVector{Symbol}, 
    types::AbstractVector{<:Type},
    X::AbstractMatrix, 
    y::AbstractVector, 
    windows::Union{Nothing, AbstractVector};
    logspace_eta::Bool=false,
    caching::Bool=true
    )

    fieldnames = Tuple(fields)
    cache = Dict{Tuple{types...}, Float64}()
    pool = distribute_folds ? CachingPool(workers) : []
    iters = 0

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
                # @show field
                optslist_safe[i] = convert(t, 10^field)
                if output
                    println("Logspace eta $field -> $(10^field)")
                end
            else
                optslist_safe[i] = convert(t, field)
            end

        end
        return optslist_safe
    end

    function cvloss(fold, hparams, opts, p)
        verbosity, pre_string, tstart, nfolds = p
        (train_inds, val_inds) = folds[fold]
        X_train, y_train, X_val, y_val = X[train_inds,:], y[train_inds], X[val_inds,:], y[val_inds]

        ti=time()
        verbosity >= 1 && println(pre_string, "iter $iters, cvfold $fold: training MPS with $(hparams)...)")
        loss = 0
        try
            mps, _... = fitMPS(X_train, y_train, opts);
            train_time = time()

            loss = mean(eval_loss(objective, mps, X_val, y_val, windows; p_fold=(verbosity-1, pre_string * "iter $iters, ", tstart, fold, nfolds))) # eval_loss always returns an array
            verbosity >= 1 && println(pre_string, "iter $iters, cvfold $fold: finished. MPS $(hparams) finished in $(rtime(ti))s (train=$(rtime(ti, train_time))s, loss=$(rtime(train_time))s))")
        
        catch e # handle scd algorithm diverging
            if e isa LoadError || e isa ArgumentError
                if opts.svd_alg == "recursive"
                    loss = Inf64
                else
                    println(pre_string, "iter $iters, cvfold $fold: MPS $(hparams)...) diverged, retrying with slower SVD algorithm")
                    loss = cvloss(fold, hparams, _set_options(opts; svd_alg="recursive"), p)
                end
            else
                throw(e)
            end
        end
        return loss
    end

    function tr_objective(optslist::AbstractVector, p)
        verbosity, pre_string, tstart, nfolds = p
        iters += 1

        optslist_safe = safe_paramlist(optslist; output=verbosity>=3)
        
        key = tuple(optslist_safe...)
        if caching && haskey(cache, key )
            verbosity >= 1 && println(pre_string, "iter $iters:Cache hit at $(optslist) -> $(optslist_safe)!")
            loss = cache[key]
        else
            hparams = NamedTuple{fieldnames}(Tuple(optslist_safe))
            opts = _set_options(opts0; hparams...)
            
            if distribute_folds
                losses = pmap(fold->cvloss(fold, hparams, opts, p), pool, 1:nfolds)
            else
                losses = map(fold->cvloss(fold, hparams, opts, p), 1:nfolds)
            end
            loss = mean(losses)
            
            if caching
                cache[key] = loss
            end
            verbosity >= 1 && println(pre_string, "iter $iters, t=$(rtime(tstart)): Mean CV Loss: $loss")
        end
        return loss
    end

    return tr_objective, cache, safe_paramlist
end

function tune_across_folds(
    rng::AbstractRNG,
    folds::AbstractVector, 
    parameter_info::Tuple,
    tuning_settings::Tuple,
    X::AbstractMatrix,
    y::AbstractVector, 
    tstart::Real
    )
    x0, opts0, bounded, lb, ub, is_disc, fields, types = parameter_info
    objective, method, workers, distribute_folds, distribute_iters, nfolds, windows, pre_string, abstol, maxiters, verbosity, provide_x0, logspace_eta = tuning_settings 

    tr_objective, cache, safe_params = make_objective(folds, objective, workers, distribute_folds, opts0, fields, types, X, y, windows; logspace_eta=logspace_eta, caching=~distribute_iters)
    p = (verbosity, pre_string, tstart, nfolds)

    # for rapid debugging
    # tr_objective = (x,u...) -> begin @show x; return sum(x.^2) end

    if nfolds <= 1
        optslist_safe = safe_params(x0)

        return NamedTuple{Tuple(fields)}(Tuple(optslist_safe))
    end

    if method isa MPSRandomSearch
        sol = grid_search(rng, x-> tr_objective(x,p), method, lb, ub, is_disc, types, fields, maxiters, distribute_iters)
        optslist_safe = safe_params(sol)
    else
        if distribute_iters
            throw(ArgumentError("Can only distribute iterations when using an MPSRandomSearch"))
        end
        x0_adj = provide_x0 ? x0 : nothing
        obj = OptimizationFunction(tr_objective, Optimization.AutoForwardDiff())
        if bounded
            prob = OptimizationProblem(obj, x0_adj, p; int=is_disc, lb=lb, ub=ub)
        else
            prob = OptimizationProblem(obj, x0_adj, p; int=is_disc)
        end
        sol = solve(prob, method; abstol=abstol, maxiters=maxiters)
        verbosity >= 5 && print(sol)
        optslist_safe = safe_params(sol.u)
    end



    best_params = NamedTuple{Tuple(fields)}(Tuple(optslist_safe))
    return best_params, cache

end

"""
```
function tune(
    X::AbstractMatrix, 
    [y::AbstractVector], 
    nfolds::Integer,
    parameters::NamedTuple,
    method=MPSRandomSearch(:LatinHypercubeSampling)) -> best_parameters::NamedTuple, cache::Dictionary
```
Perform `nfolds`-fold hyperparameter tuning of an MPS on the timeseries data `X`, optionally specifying the data classes `y`. Returns a named\
tuple containing the optimal hyperparameters, and a cache dictionary that saves the mean loss of every tested hyperparameter combination.

`parameters` specifies the hyperparameters to tune and their upper and lower bounds. Currently, every numeric field of [`MPSOptions`](@ref) is supported.\
E.g. to tune the maximum bond dimension (`chi_max`) over the range [15,45], and the physical dimension (`d`) over the range [2,15]:
```Julia
parameters = (d=(2,15), chi_max=(15,45))
```
Note that the upper and lower bounds are passed to the hyperparameter tuning algorithm, but may not be strictly enforced depending on your choice of algorithm.\
alternatively, use the `enforce_bounds=false` keyword argument to disable bounds checking completely.

# Example:

# Hyperparameter Tuning Methods
The hyperparameter tuning algorithm can be specified with the `method` argument. This supports the builtin [`MPSRandomSearch`](@ref) methods, as well\
as (in theory) any solver that is supported by the [`Optimization.jl interface`](https://docs.sciml.ai/Optimization/stable). Note that many of these solvers\
struggle with discrete inputs, some of them require initial conditions (`provide_x0=true`), and some require no initial conditions (`provide_x0=false`),\
so your mileage may vary.

# Keyword Arguments
## Hyperparameter Options
opts0::AbstractMPSOptions=MPSOptions(; verbosity=-5, log_level=-1),
        input_supertype::Type=Float64,
        enforce_bounds::Bool=true,
        logspace_eta::Bool=false,
## Tuning algorithm
abstol::Float64=1e-3,
        maxiters::Integer=250,
        provide_x0::Bool=true,
        rng::Union{Integer, AbstractRNG}=1,
        foldmethod::Union{Function, Vector}=make_stratified_cvfolds, 
## Loss and Windowing
- `objective::TuningLoss=ImputationLoss()`:
- `pms::Union{Nothing, AbstractVector}=nothing`:
- `windows::Union{Nothing, AbstractVector, Dict}=nothing`:

## Folds and Cross validation

## Logging
    `verbosity::Integer=1`:
            pre_string::String=""

## Distributed Computing
- distribute_folds::Bool=false,
        distribute_iters::Bool=false,
        workers::AbstractVector{Int}=distribute_folds ? workers() : Int[],
        disable_nondistributed_threading::Bool=false
"""
function tune(
        X::AbstractMatrix, 
        y::AbstractVector, 
        nfolds::Integer,
        parameters::NamedTuple,
        method=MPSRandomSearch(); # A latin hypercube based random search
        opts0::AbstractMPSOptions=MPSOptions(; verbosity=-5, log_level=-1),
        input_supertype::Type=Float64,
        enforce_bounds::Bool=true,
        objective::TuningLoss=ImputationLoss(), 
        rng::Union{Integer, AbstractRNG}=1,
        foldmethod::Union{Function, Vector}=make_stratified_cvfolds, 
        pms::Union{Nothing, AbstractVector}=nothing, #TODO make default behaviour a bit better
        windows::Union{Nothing, AbstractVector, Dict}=nothing,
        verbosity::Integer=1,
        provide_x0::Bool=true,
        logspace_eta::Bool=false,
        abstol::Float64=1e-3,
        maxiters::Integer=250,
        distribute_folds::Bool=false,
        distribute_iters::Bool=false,
        workers::AbstractVector{Int}=distribute_folds ? workers() : Int[],
        disable_nondistributed_threading::Bool=false,
        pre_string::String=""

    )
    if isempty(parameters) || nfolds == 0 || maxiters == 0
        return opts0, Dict()
    end

    if objective isa ImputationLoss && opts0.sigmoid_transform
        @warn pre_string * "Using sigmoid_transform preprocessing on an imputation-style problem generally leads to worse performance."
    end
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


    parameter_info = x0, opts0, enforce_bounds, lb, ub, is_disc, fields, types
    tuning_settings = objective, method, workers, distribute_folds, distribute_iters, nfolds, windows, pre_string, abstol, maxiters, verbosity, provide_x0, logspace_eta

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


    return tune_across_folds(abs_rng, folds, parameter_info, tuning_settings, X, y, tstart)

end

# no class version
tune(X::AbstractMatrix, nfolds::Integer, args...; kwargs...) = tune(X, zeros(Int, size(X, 1)), args...; kwargs...)


