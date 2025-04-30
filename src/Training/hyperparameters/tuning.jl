function make_objective(
    folds::AbstractVector,
    objective::TuningLoss, 
    workers::AbstractVector,
    distribute_folds::Bool,
    opts0::AbstractMPSOptions, 
    fields::AbstractVector{Symbol}, 
    types::AbstractVector{<:Type},
    value_map::AbstractVector,
    Xs::AbstractMatrix, 
    ys::AbstractVector, 
    maxiters::Integer,
    windows::Union{Nothing, AbstractVector};
    logspace_eta::Bool=false,
    caching::Bool=true,
    max_cache_hits::Integer=100
    )

    fieldnames = Tuple(fields)
    cache = Dict{Tuple{types...}, Float64}()
    pool = distribute_folds ? CachingPool(workers) : []
    iters = Ref{Int}(0)
    hits = Ref{Int}(0)

    function safe_paramlist(optslist::AbstractVector; output=false)
        optslist_mapped = Vector{Union{AbstractFloat, Integer}}(undef, length(optslist)) # after mapping using value_map but before rounding. For logging cache hits
        optslist_safe = Vector{Union{AbstractFloat, Integer}}(undef, length(optslist))
        for (i, field) in enumerate(optslist)
            if ~isempty(value_map[i])
                idx = round(Int,field)
                field_mapped = value_map[i][idx]
            else
                field_mapped = field
            end
            optslist_mapped[i] = field_mapped

            t = types[i]
            if t <: Integer
                rounded = round(Int, field_mapped)
                optslist_safe[i] = rounded
                if output && ~isapprox(field_mapped, rounded)
                    println("Integer parameter $(fieldnames[i])=$field_mapped rounded to $(rounded)!")
                end
            elseif logspace_eta && fieldnames[i] == :eta
                # @show field
                optslist_safe[i] = convert(t, 10^field_mapped)
                if output
                    println("Logspace eta $field_mapped -> $(10^field_mapped)")
                end
            else
                optslist_safe[i] = convert(t, field_mapped)
            end

        end
        return optslist_mapped, optslist_safe
    end

    function cvloss(fold, hparams, opts, p)
        verbosity, pre_string, tstart, nfolds = p
        (train_inds, val_inds) = folds[fold]
        X_train, y_train, X_val, y_val = Xs[train_inds,:], ys[train_inds], Xs[val_inds,:], ys[val_inds]

        ti=time()
        verbosity >= 1 && println(pre_string, "iter $(iters[]), cvfold $fold: training MPS with $(hparams)...")
        loss = 0
        try
            mps, _... = fitMPS(X_train, y_train, opts);
            train_time = time()

            loss = mean(eval_loss(objective, mps, X_val, y_val, windows; p_fold=(verbosity-1, pre_string * "iter $(iters[]), ", tstart, fold, nfolds))) # eval_loss always returns an array
            verbosity >= 1 && println(pre_string, "iter $(iters[]), cvfold $fold: finished. MPS $(hparams) finished in $(rtime(ti))s (train=$(rtime(ti, train_time))s, loss=$(rtime(train_time))s)")
        
        catch e # handle scd algorithm diverging
            if e isa LoadError || e isa ArgumentError
                if opts.svd_alg == "recursive"
                    loss = Inf64
                else
                    println(pre_string, "iter $(iters[]), cvfold $fold: MPS $(hparams)...) diverged, retrying with slower SVD algorithm")
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

        optslist_mapped, optslist_safe = safe_paramlist(optslist; output=verbosity>=3)
        
        key = tuple(optslist_safe...)
        if caching && haskey(cache, key )
            loss = cache[key]
            hits[] += 1
            if verbosity >= 1
                if verbosity >= 5 || hits[] <=3
                    println(pre_string, "iter $(iters[]): Cache hit at $(optslist_mapped) -> $(optslist_safe)!")
                elseif hits[] == 4
                    println(pre_string, "iter $(iters[]): Too many cache hits, suppressing notifications!")
                end

            end
        else
            hits[] = 0
            iters[] += 1
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
            verbosity >= 1 && println(pre_string, "iter $(iters[]), t=$(rtime(tstart)): Mean CV Loss: $loss")
        end
        return loss
    end

    function enforce_maxiters_callback(state, l)
        if iters[] >= maxiters
            println("Manually stopping tune() due to max iterations hit! Optimization.jl will give a warning.")
            stop = true
        elseif hits[] > max_cache_hits
            println("Manually stopping tune() because there were too many cache hits! (Is your search space too small?) Optimization.jl will give a warning.")
            stop = true
        else

            stop = false
        end
        return stop
    end

    return tr_objective, enforce_maxiters_callback, cache, safe_paramlist
end

function tune_across_folds(
    rng::AbstractRNG,
    folds::AbstractVector, 
    parameter_info::Tuple,
    tuning_settings::Tuple,
    Xs::AbstractMatrix,
    ys::AbstractVector, 
    tstart::Real;
    kwargs...
    )
    x0, opts0, bounded, lb, ub, is_disc, fields, types, value_map = parameter_info
    objective, method, workers, distribute_folds, distribute_iters, nfolds, windows, pre_string, abstol, maxiters, verbosity, provide_x0, logspace_eta = tuning_settings 

    tr_objective, callback, cache, safe_params = make_objective(
        folds, 
        objective, 
        workers, 
        distribute_folds,
        opts0, 
        fields, 
        types, 
        value_map,
        Xs, 
        ys, 
        maxiters,
        windows; 
        logspace_eta=logspace_eta, 
        caching=~distribute_iters
    )
    p = (verbosity, pre_string, tstart, nfolds)

    # for rapid debugging
    # tr_objective = (x,u...) -> begin @show x; return sum(x.^2) end

    if nfolds <= 1
        optslist_mapped, optslist_safe = safe_params(x0)

        return NamedTuple{Tuple(fields)}(Tuple(optslist_safe))
    end

    if method isa MPSRandomSearch
        sol = grid_search(rng, x-> tr_objective(x,p), method, lb, ub, is_disc, types, fields, maxiters, distribute_iters)
        optslist_mapped, optslist_safe = safe_params(sol)
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
        sol = solve(prob, method; abstol=abstol, callback=callback, kwargs...) # enforcing maxiters is handled by the callback
        verbosity >= 5 && print(sol)
        optslist_mapped, optslist_safe = safe_params(sol.u)
    end



    best_params = NamedTuple{Tuple(fields)}(Tuple(optslist_safe))
    return best_params, cache

end

@doc """
```julia
function tune(
    Xs::AbstractMatrix, 
    [ys::AbstractVector], 
    nfolds::Integer,
    parameters::NamedTuple,
    optimiser=MPSRandomSearch(:LatinHypercubeSampling);
    <Keyword Arguments>) -> best_parameters::NamedTuple, cache::Dictionary
```
Perform `nfolds`-fold hyperparameter tuning of an MPS on the timeseries data `Xs`, optionally specifying the data classes `ys`. Returns a NamedTuple
containing the optimal hyperparameters, and a cache Dictionary that saves the loss of every tested hyperparameter combination.

`parameters` specifies the hyperparameters to tune and (optionally) a search space. Currently, every numeric field of [`MPSOptions`](@ref) is supported. `parameters` 
are specified as named tuples, with the key being the name of the hyperparameter (See Example below). There are a couple of options for specifying the bounds:
- The preferred option is `Tuple` of lower/upper bounds: "`params=(eta=(upper_bound,lower_bound), ...)`", which allows the optimiser to choose any value in the \
interval [upper_bound, lower_bound]. You can also pass an empty tuple: "`params=(eta=(), ...)`" which allows the parameter to take any non-negative value. 
- As a `Vector` of possible values, e.g. `params = (d=[1,3,6,7,8], ...)`. For convenience, you can also use the `Tuple` syntax "`params=(d=(start,step,stop), ...)`", \
which is equivalent to "`params = (d=start:step:stop, ...)`"

Note that if you use the first method, the upper and lower bounds are passed to the hyperparameter tuning algorithm, but may not be strictly enforced depending on your choice of algorithm. 
Alternatively, use the `enforce_bounds=false` keyword argument to disable bounds checking completely (for compatible optimisers).

# Example:
To tune a classification problem by searching the hyperparameter space η ∈ [0.001, 0.1], d ∈ {5,6,7}, and χmax ∈ {20,21,...,25}:
```julia-repl
julia> params = (
    eta=(1e-3, 1e-1), 
    d=(5,7), 
    chi_max=(20,25)
); # configure the search space

julia> nfolds = 5;

julia> best_params, cache = tune(
    X_train, # Training data as a matrix, rows are time series
    y_train, # Vector of time series class labels
    nfolds, # Number of cross validation folds
    params,
    MPSRandomSearch(); # Tuning algorithm
    objective=MisclassificationRate(), # Type of loss to use
    maxiters=20, # Maximum number of tuning iterations
    logspace_eta=true # When true, the eta search space [0.001, 0.1] is sampled logarithmically
)
[...]

julia> best_opts = MPSOptions(best_params); # convert to an MPSOptions object
[...]
```
Other problem classes are available, see the MPSTime documentation.


# Hyperparameter Tuning Methods
The hyperparameter tuning algorithm can be specified with the `optimiser` argument. This supports the default builtin [`MPSRandomSearch`](@ref) methods, as well
as (in theory) any solver that is supported by the [`Optimization.jl interface`](https://docs.sciml.ai/Optimization/stable). Note that many of these solvers
struggle with discrete search spaces, such as tuning the integer valued `chi_max` and `d`, or tuning an `eta` specified with a vector. Some of them require initial conditions (set `provide_x0=true`), 
and some require no initial conditions (set `provide_x0=false`), so your mileage may vary. By default, `tune()` handles optimisers attempting to evaluate 
discrete hyperparameters at a non-integer value by rounding, and using its own cache to avoid rounding based cache misses.


There are a lot of keyword arguments... Extended help is avaliable with \`??tune\`

See also: [`evaluate`](@ref)

# Extended Help

# Keyword Arguments
## Hyperparameter Options
- `opts0::AbstractMPSOptions=MPSOptions(; verbosity=-5, log_level=-1, sigmoid_transform=objective isa ClassificationLoss)`: Default hyperparamaters to pass to `fitMPS`. \
Hyperparameter candidates are generated by modifying `opts0` with the values in the search space specified by `parameters`.
- `enforce_bounds::Bool=true`: Whether to pass the constraints given in params to the optimisation algorithm.
- `logspace_eta::Bool=false`: Whether to treat the `eta` parameterspace as logarithmic. E.g. setting `parameters=(eta=(10^-3,10^-1) )` and `logspace_eta=true` \
will sample each `eta_candidate` from the log10 search space [-3.,-1.], and then pass `eta = 10^(eta_candidate)` to `MPSOptions`. 
- `input_supertype::Type=Float64`: A numeric type that can represent the types of each hyperparameter being tuned as well as their upper and lower bounds. \
Typically, `Float64` is sufficient, but it can be set to `Int` for purely discrete optimisation problems etc. This is necessary for mixed integer / Float  \
hyperparameter tuning because certain solvers in `Optimization.jl` require variables in the search space to all be the same type.


## Loss and Windowing
- `objective::TuningLoss=ImputationLoss()`: The objective of the hyperparameter optimisation. This comes in two categories:
######  Imputation Loss
`obvjective=ImputationLoss()` Uses the mean of the mean absolute error to measure the performance of an MPS on imputing unseen data. First, it generates 'corrupted' \
time series data by applying missing data windows to the validation set, using one of the following options:
* `pms::Union{Nothing, AbstractVector}=nothing`: Stands for 'percentage missings'. Will remove a randomly selected contiguous \
blocks from each time series in the validation set, according to the percentages missing listed in the `pms` vector. For example, \
`pms=[0.05, 0.05, 0.6, 0.95]` will generate four windows, two with 5% missing, and one each with 60% and 95% missing
* `windows::Union{Nothing, AbstractVector, Dict}=nothing`: Manually input missing windows. Expects a vector of missing windows, or a dictionary where \
`values(windows)` is a vector of missing windows.

The tuning loss is the average of computing the mean absolute error of imputing every window on every element of the validation set.

###### Classification Loss
Classification type problems can be hyperparameter tuned to minimise either `MisclassificationRate()` (1 - classification accuracy), \
or `BalancedMisclassificationRate()` (1 - balanced accuracy).

###### Custom Loss Functions
Custom losses can be used by implementing a custom loss value type (`CustomLoss <: TuningLoss`) and extending the definition of [`MPSTime.eval_loss`](@ref) \
with the signature
```
eval_loss(
    CustomLoss(), 
    mps::TrainedMPS, 
    X_validation::AbstractMatrix, 
    y_validation::AbstractVector, 
    windows; 
    p_fold=nothing, 
    distribute::Bool=false
) -> Union{Float64, Vector{Float64}
```
if `eval_loss` returns a vector, then `tune()` will optimise `mean(eval_loss(...))`. For concrete examples, see the documentation

## Tuning algorithm
- `abstol::Float64=1e-3`: Passed directly to `Optimization.jl`: Absolute tolerance in changes to the objective (loss) function
- `maxiters::Integer=250`: Maximum number of iterations allowed when solving
- `provide_x0::Bool=true`: Whether to provide initial conditions to the solve, ignored by [`MPSRandomSearch`](@ref). The initial condition will be `opts0`, \
unless it contains a hyperparameter outside the range specified by `parameters`, in which case the lower bound of that hyperparameter will be used.
- `rng::Union{Integer, AbstractRNG}=1`: An Integer or RNG object used to seed any randomness in imputation window or search space generation.
- `kwargs...`: Any further keyword arguments to passed through to `Optimization.jl` through the [`Optimization.solve`](@extref CommonSolve.solve) function

## Folds and Cross validation
- `foldmethod::Union{Function, AbstractVector}=make_stratified_cvfolds`: The method used to generate the train/validation folds from `Xs`. \
Can either be an `nfolds`-long Vector of `[train_indices::Vector, validation_indices::Vector]` pairs, or a function that produces them, with the signature `foldmethod(Xs,ys, nfolds; rng::AbstractRNG)` \
To clarify, the `tune` function determines the train/validation splits for the ith fold in the following way:
```
Julia> folds::Vector = foldmethod isa Function ? foldmethod(Xs,ys, nfolds; rng=rng) : foldmethod;
Julia> train_inds, validation_inds = folds[i];
Julia> X_train, y_train = Xs[train_inds, :], ys[train_inds];
Julia> X_validation, y_validation = Xs[validation_inds, :], ys[validation_inds];
```

## Logging 
- `verbosity::Integer=1`: Controls how explicit the logging is. 0 for none, 5 for maximimum. This is separate to the verbosity in MPSOptions.
- `pre_string::String=""`: Prints this string on the same line before logging messages are printed. Useful for logging when calling `tune()` multiple times in parallel.

## Distributed Computing
Parallel processing is available using processors added via Distributed.jl's [`addprocs`](@extref Distributed.addprocs) function.
- `distribute_iters::Bool=false`: When using an `MPSRandomSearch`, distribute the search grid across all available processors. For thread safety, using `distribute_iters` disables caching.
- `distribute_folds::Bool=false`: Distribute each fold to its own processor. Scales up to at most `nfolds` processors. Not very compatible with `distribute_iters`.
- `workers::AbstractVector{Int}=distribute_folds ? workers() : Int[]`: Workers that may be used to distribute folds, does not affect `distribute_iters`. \
This can be used to run multiple instances of `tune()` on different sets of workers.
- `disable_nondistributed_threading::Bool=false`: Attempts to disable threading using `BLAS.set_num_threads(1)` and `Strided.disable_threads()` (May not work if using the MKL.jl linear algebra backend).
"""
function tune(
        Xs::AbstractMatrix, 
        ys::AbstractVector, 
        nfolds::Integer,
        parameters::NamedTuple,
        method=MPSRandomSearch(); 
        input_supertype::Type=Float64,
        enforce_bounds::Bool=true,
        objective::TuningLoss=ImputationLoss(), 
        opts0::AbstractMPSOptions=MPSOptions(; verbosity=-5, log_level=-1, sigmoid_transform=(objective isa ClassificationLoss)),
        rng::Union{Integer, AbstractRNG}=1,
        foldmethod::Union{Function, AbstractVector}=make_stratified_cvfolds, 
        pms::Union{Nothing, AbstractVector}=nothing,
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
        pre_string::String="",
        kwargs...

    )
    if isempty(parameters) || nfolds == 0 || maxiters == 0
        return opts0, Dict()
    end

    if objective isa ImputationLoss && opts0.sigmoid_transform
        @warn pre_string * "Using sigmoid_transform preprocessing on an imputation-style problem generally leads to worse performance."

    elseif objective isa ClassificationLoss && ~opts0.sigmoid_transform
        @warn pre_string * "Disabling sigmoid_transform preprocessing on an imputation-style problem may lead to worse performance."
    end
    # basic checks    
    abs_rng = rng isa Integer ? Xoshiro(rng) : rng


    if !(length(unique(keys(parameters))) == length(keys(parameters)))
       throw(ArgumentError("The 'parameters' argument contains duplicates!")) 
    end
    if objective isa ImputationLoss
        windows = make_windows(windows, pms, Xs, abs_rng)
    end

    
    is_disc = Vector{Bool}(undef, length(parameters))
    lb = Vector{input_supertype}(undef, length(parameters))
    ub = Vector{input_supertype}(undef, length(parameters))
    x0 = Vector{input_supertype}(undef, length(parameters))
    types = Vector{Type}(undef, length(parameters))
    value_map = Vector[ [] for _ in 1:length(parameters)]

    


    # setup tuned hyperparameters
    for (i, (key, val)) in enumerate(pairs(parameters))
        startx = getproperty(opts0, key)
        param_type = typeof(startx)
        if !( param_type <: Number)
            throw(ArgumentError("Cannot tune '$key', only numeric types can be hyperoptimised."))
        end

        if logspace_eta && key == :eta
            if val[1] <= 0
                throw(ArgumentError("Lower and upper bounds on eta must be positive!"))
            end
            if val isa AbstractVector || length(val) == 3
                throw(ArgumentError("logspace_eta doesn't make sense with this method of specifying eta values"))

            end
            val = log10.(val)
        end

        if val isa AbstractVector # [vals]
            is_disc[i] = true # override default type
            sorted = sort(val)
            value_map[i] = sorted
            lb[i], ub[i] = 1, length(val)

            if enforce_bounds == false
                @warn "Not enforcing bounds when specifying `params` as a Vector can lead to odd/undefined interactions between MPSTime and Optimization.jl."
            end

            
        elseif val isa Tuple
            if length(val) == 3 # (lb, step, ub)
                is_disc[i] = true # override default type
                value_map[i] = range(val[1], val[end]; step=val[2])
                lb[i], ub[i] = 1, length(value_map[i])
                if enforce_bounds == false
                    @warn "Not enforcing bounds when specifying `params` as a range (`key=(start,step,step)`) can lead to odd/undefined interactions between MPSTime and Optimization.jl."
                end
                
            elseif length(val) == 2 # (lower bound, upper bound)
                is_disc[i] = param_type <: Integer
                lb[i], ub[i] = convert(param_type, val[1]), convert(param_type, val[2])

            elseif length(val) == 0 # no bounds
                is_disc[i] = param_type <: Integer

                if is_disc
                    lb[i], ub[i] = one(param_type), typemax(param_type)
                else
                    lb[i], ub[i] = eps(param_type), typemax(param_type)
                end
            else
                throw(ArgumentError("Unknown parameter format. Options are key=[vals], key=(), key=(lb,ub), key=(lb,step,ub)"))
            end
        else
            throw(ArgumentError("Unknown parameter format. Options are key=[vals], key=(), key=(lb,ub), key=(lb,step,ub)"))
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

    for vec in [fields, types, x0, is_disc, lb, ub, value_map]
        permute!(vec, perm)
    end


    parameter_info = x0, opts0, enforce_bounds, lb, ub, is_disc, fields, types, value_map
    tuning_settings = objective, method, workers, distribute_folds, distribute_iters, nfolds, windows, pre_string, abstol, maxiters, verbosity, provide_x0, logspace_eta

    if nfolds <= 1
        folds = []
    else
        # println("Generating Folds")
        folds::Vector = foldmethod isa Function ? foldmethod(Xs,ys, nfolds; rng=abs_rng) : foldmethod
    end
    tstart = time()

    if disable_nondistributed_threading 
        
        GenericLinearAlgebra.LinearAlgebra.BLAS.set_num_threads(1)
        ITensors.Strided.disable_threads()
        @warn "Threading may still be active, if it is, try setting the environment variable OMP_NUM_THREADS=1 before launching julia."

    end


    return tune_across_folds(abs_rng, folds, parameter_info, tuning_settings, Xs, ys, tstart, kwargs...)

end

# no class version
tune(Xs::AbstractMatrix, nfolds::Integer, args...; kwargs...) = tune(Xs, zeros(Int, size(Xs, 1)), args...; kwargs...)


