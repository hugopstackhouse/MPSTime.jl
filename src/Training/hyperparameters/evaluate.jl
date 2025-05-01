"""
```Julia
function evaluate(
    Xs::AbstractMatrix, 
    [ys::AbstractVector], 
    nfolds::Integer,
    tuning_parameters::NamedTuple,
    tuning_optimiser=MPSRandomSearch();
    <Keyword Arguments>) -> results::Vector{Dictionary}
```
Evaluate the performance of MPSTime by [`hyperparameter tuning`](@ref tune) on `nfolds` resampled folds of the timeseries dataset `Xs` with classes `ys`.

`tuning_parameters` controls the hyperparamters to tune over, and `tuning_optimiser` specifies the hyperparameter tuning algorithm.
These are passed directly to [`tune`](@ref), refer to its documentation for details. 

# Example
To evaluate a classification problem by searching the hyperparameter space η ∈ [0.001, 0.1], d ∈ {5,6,7}, and χmax ∈ {20,21,...,25},
```julia-repl
julia> params = (
    eta=(1e-3, 1e-1), 
    d=(5,7), 
    chi_max=(20,25)
);
julia> nfolds = 30;

julia> results = evaluate(
    X_train, # training data as a matrix, rows are time series
    y_train, # Vector of time series class labels
    nfolds, # number of resample folds
    params, # search space
    MPSRandomSearch(); # Hyperparameter search method, see Extended help
    objective=MisclassificationRate(), # Type of lsso to use
    maxiters=20, # Maximum number of tuning iterations
    logspace_eta=true # the eta search space [0.001, 0.1] is sampled logarithmically
)
[...]
```

# Return value
A length `nfolds` vector of dictionaries that contain detailed informatation about each fold. Each dictionary has the following keys:
- `"fold"=>Integer`: The fold index.
- `"objective"=>String`: The objective (loss function) this fold was trained on.
- `"train_inds"=>Vector`: The indices (rows of Xs) this fold was tuned/trained on.
- `"test_inds"=>Vector`: The indices (rows of Xs) this fold was tested on.
- `"optimiser"=>String`: Name of the optimiser used to hyperparameter tune each fold.
- `"tuning_windows"=>Vector`: The windows used to hyperparameter tune this fold.
- `"tuning_pms"=>Vector`: The 'Percentages Missing' used to hyperparameter tune this fold (possibly used to generate tuning_windows).
- `"eval_windows"=>Vector`: The windows used to evaluate the test loss.
- `"eval_pms"=>eval_pms`: The 'Percentages Missing' used to evaluate the test loss (possibly used to generate `eval_windows`).
- `"time"=>Vector`: Total time to tune and test this fold in seconds.
- `"opts"=>MPSOptions`: Optimal options for this fold as determined by tune(). Used to compute the test loss.
- `"cache"=>Dict`: Cache of the validation losses of every set of hyperparameters evaluated on this fold. Disabled if `distribute_iters` is true.
- `"loss"=>Union{Vector{Float64}, Float64}`. The test loss of this fold. If `objective` is an `ImputationLoss()`, this is a vector with each entry corresponding to a window \
in `results[fold]["eval_windows"]`.

There are a lot of keyword arguments... Extended help is avaliable with \`??evaluate\`

# Extended Help
# Keyword Arguments
## Loss and Windowing
- `objective::TuningLoss=ImputationLoss()`: The loss used to evaluate and tune the MPS. If its an ImputationLoss, then either `pms` or `windows` \
must be specified for each of evaluation and tuning. See the [`tune`](@ref) extended documentation for more details.
- `eval_pms::Union{Nothing, AbstractVector}=nothing`: 'Percentage MissingS' used to evaluate the test loss.
- `eval_windows::Union{Nothing, AbstractVector, Dict}=nothing`: Windows used to evaluate the test loss.
- `tuning_pms::Union{Nothing, AbstractVector}=eval_pms`: 'Percentage MissingS' passed to tune, and used to compute validation loss.
- `tuning_windows::Union{Nothing, AbstractVector, Dict}=eval_windows`: Windows passed to tune, and used to compute validation loss.
    
- `rng::Union{Integer, AbstractRNG}=1`: An integer or RNG object used to seed any randomness in imputation window or search space generation. `Random.seed!(fold)`` is called \
prior to tuning each fold, so that any optimization algorithms that are random but don't take rng objects should still be deterministic.
- `tuning_rng::AbstractVector{<:Union{Integer, AbstractRNG}=collect(1:nfolds)`: Passed through to `tune`. An integer or RNG object used to seed any randomness in \
tuning imputation window generation or hyperparameter searching.

- `opts0::AbstractMPSOptions=MPSOptions(; verbosity=-5, log_level=-1, sigmoid_transform=(objective isa ClassificationLoss))`: \
Options that are modified by the best options returned by tune. Used to train the MPS which evalutates the test loss. 

- `tuning_opts0::AbstractMPSOptions=opts0`: Initial guess passed as `opts0` to [`tune`](@ref) that sets the values of the non-tuned hyperparameters. \
Should generally always be the same as `opts0`, but can be specified separately in case you wish to make the final mps train with more verbosity etc. 


## Resampling and Cross Validation
- `foldmethod::Union{Function, Vector}=make_stratified_cvfolds`: Can either be an `nfolds`-long Vector of `[train_indices::Vector, test_indices::Vector]` pairs, or a function that produces them, with the signature `foldmethod(Xs,ys, nfolds; rng::AbstractRNG)` \
To clarify, the `tune` function determines the train/test splits for the ith fold in the following way:
```
julia> folds::Vector = foldmethod isa Function ? foldmethod(Xs,ys, nfolds; rng=rng) : foldmethod;
julia> train_inds, test_inds = folds[i];
julia> X_train, y_train = Xs[train_inds, :], ys[train_inds];
julia> X_test, y_test = Xs[test_inds, :], ys[test_inds];
```
`foldmethod` defaults to `nfold`-fold cross validation.
- `tuning_foldmethod::Union{Function, Vector}=make_stratified_cvfolds`: Same as above, although it is passed to `tune` and used to split the training set into \
hyperparameter train/validation sets. The fold number specified by the `n_cvfolds` keyword.
- `fold_inds::Vector{<:Integer}=collect(1:nfolds)`: A vector of the fold indices to evaluate. This can be used to split large training runs into batches, or to resume a halted benchmark.


## Distributed Computing
Several parallel processing paradigms are availble for different use cases, implented using processors added via Distributed.jl's [`addprocs``](@extref Distributed.addprocs) function.
- `distribute_iters::Bool=false`: When using an `MPSRandomSearch`, for each fold, distribute the search grid across all available processors. \
For thread safety, using `distributed_iters` disables caching.
- `distribute_folds::Bool=false`: Allocate one processor to each fold.
- `distribute_cvfolds::Bool=false`: Equivalent to passing `distribute_folds` to `tune`. Allocates a processor to each hyperparameter train/val split.
- `distribute_final_eval::Bool=false`: Allocate a processor to each test timeseries when computing the test loss. Useful when the test set is very large. \
The only option compatible with the others. 

## Saving and Resuming
If `write` is enabled, `evaluate` will automatically resume if it finds a partially complete run. 
!!! danger "Only the filename is checked when comparing save data, so it is possible to accidentally merge incompatible evaluations or overwrite complete ones if they are named the same thing!"
- `write::Bool=false`: Whether to write output to files. If true, it will save temporary files, saving each completed fold inside `"\$writedir/\$(simname)_temp/"`, and the \
final result to `"\$writedir/\$(simname).jld2"`.
- `writedir::String="evals"`: The directory to save data to.
- `simname::String="\$(objective)_\$(tuning_optimiser)_f=\$(nfolds)_cv\$(n_cvfolds)_iters=\$(tuning_maxiters)"`: The simulation name. Used to determine save location.
- `delete_tmps::Bool=length(fold_inds)==nfolds`: Whether to delete the temp directory at the end.

## Logging
- `verbosity::Integer=1`: Controls how explicit the logging is. 0 for none, 5 for maximimum. This is separate to the verbosity in MPSOptions.

## Hyperparameter Tuning Options
These options are passed directly to their corresponding keywords in [`tune`](@ref)
- `n_cvfolds::Integer=5`: Corresponds to `nfolds` in tune, number of train/val splits.
- `logspace_eta::Bool=false`: Whether to treat the `eta` parameterspace as logarithmic. E.g. setting `parameters=(eta=(10^-3,10^-1) )` and `logspace_eta=true` \
will sample each `eta_candidate` from the log10 search space [-3.,-1.], and then pass `eta = 10^(eta_candidate)` to `MPSOptions`. 
- `input_supertype::Type=Float64`: A numeric type that can represent the types of each hyperparameter being tuned as well as their upper and lower bounds. \
Typically, `Float64` is sufficient, but it can be set to `Int` for purely discrete optimisation problems etc. This is necessary for mixed Integer / Float \
hyperparameter tuning because certain solvers in `Optimization.jl` require variables in the search space to all be the same type.
- `tuning_abstol::Float64=1e-3`: Passed directly to `Optimization.jl`: Absolute tolerance in changes to the objective (loss) function. 
- `tuning_maxiters::Integer=250`: Maximum number of iterations allowed when solving.
- `provide_x0::Bool=true`: Whether to provide initial conditions to the solve, ignored by [`MPSRandomSearch`](@ref). The initial condition will be `opts0`, \
unless it contains a hyperparameter outside the range specified by `parameters`, in which case the lower bound of that hyperparameter will be used.


Further keyword arguments to `evaluate` are passed through to [`tune`](@ref), and then `Optimization.jl` through the [`Optimization.solve`](@extref CommonSolve.solve) function.

"""
function evaluate(
    Xs::AbstractMatrix, 
    ys::AbstractVector, 
    nfolds::Integer,
    tuning_parameters::NamedTuple,
    tuning_optimiser=MPSRandomSearch();
    objective::TuningLoss=ImputationLoss(), 
    verbosity::Integer=1,
    opts0::AbstractMPSOptions=MPSOptions(; verbosity=-5, log_level=-1, sigmoid_transform=(objective isa ClassificationLoss)),
    input_supertype::Type=Float64,
    tuning_opts0::AbstractMPSOptions=opts0,
    n_cvfolds::Integer=5,
    fold_inds::Vector{<:Integer}=collect(1:nfolds),
    provide_x0::Bool=true,
    logspace_eta::Bool=false,
    rng::Union{Integer, AbstractRNG}=1,
    tuning_rng::AbstractVector{<:Union{Integer, AbstractRNG}}=collect(1:nfolds),
    foldmethod::Union{Function, Vector}=make_stratified_folds, 
    tuning_foldmethod::Union{Function, Vector}=make_stratified_cvfolds, 
    eval_pms::Union{Nothing, AbstractVector}=nothing,
    eval_windows::Union{Nothing, AbstractVector, Dict}=nothing,
    tuning_pms::Union{Nothing, AbstractVector}=eval_pms,
    tuning_windows::Union{Nothing, AbstractVector, Dict}=tuing_windows,
    tuning_abstol::Float64=1e-3,
    tuning_maxiters::Integer=250,
    distribute_folds::Bool=false,   
    distribute_cvfolds::Bool=false,
    distribute_final_eval::Bool=false,
    write::Bool=false,
    writedir::String="evals",
    simname::String="$(objective)_$(tuning_optimiser)_f=$(nfolds)_cv=$(n_cvfolds)_iters=$(tuning_maxiters)",
    overwrite::Bool=false,
    delete_tmps::Bool=length(fold_inds)==nfolds, # default to delete only if the entire eval is done at once
    kwargs... # further kwargs passed to tune()
    )
    abs_rng = rng isa Integer ? Xoshiro(rng) : rng

    if objective isa ImputationLoss
        eval_windows = make_windows(eval_windows, eval_pms, Xs, abs_rng)
        # tuning_windows = make_windows(tuning_windows, tuning_pms, Xs, abs_rng)
    end

    folds::Vector = foldmethod isa Function ? foldmethod(Xs,ys, nfolds; rng=abs_rng) : foldmethod

    outfile = strip(writedir, '/') * "/" * strip(simname, '/') * ".jld2"
    tmpdir = strip(writedir, '/') * "/" * strip(simname, '/') * "_tmp/"

    if write
        mkpath(tmpdir)
    end

    tstart = time()
    function _eval_fold(fold, cv_workers=Int[])
        Random.seed!(fold)
        fname = tmpdir * "f" * string(fold) * ".jld2"

        if write
            if isfile(fname)
                if overwrite
                    println("Fold " * string(fold) * " already exists, overwriting...")
                else
                    println("Fold " * string(fold) * " already exists, skipping...")
                    JLD2.@load fname res_iter
                    return res_iter
                end
            end
        end

        println("Beginning fold $fold:")
        tbeg = time()
        (train_inds, test_inds) = folds[fold]
        X_train, y_train, X_test, y_test = Xs[train_inds,:], ys[train_inds], Xs[test_inds,:], ys[test_inds]
    
        abs_rng_inner = tuning_rng[fold] isa Integer ? Xoshiro(tuning_rng[fold]) : tuning_rng[fold]
        tuning_windows_inner = nothing
        if objective isa ImputationLoss
            tuning_windows_inner = make_windows(tuning_windows, tuning_pms, Xs, abs_rng_inner)
        end
        best_params, cache = tune(
            X_train, 
            y_train,
            n_cvfolds, 
            tuning_parameters,
            tuning_optimiser; 
            objective=objective, 
            opts0=tuning_opts0,
            input_supertype=input_supertype,
            provide_x0=provide_x0,
            logspace_eta=logspace_eta,
            pms=nothing,
            windows=tuning_windows_inner,
            abstol=tuning_abstol, 
            maxiters=tuning_maxiters,
            verbosity=verbosity,
            rng=abs_rng_inner,
            foldmethod=tuning_foldmethod,
            distribute_folds=distribute_cvfolds,
            workers=cv_workers,
            pre_string="Fold $fold: ",
            kwargs...
        )
        # @show best_params
        if best_params isa AbstractMPSOptions
            opts = best_params
        else
            opts = _set_options(opts0; best_params...)
        end
        verbosity >= 1 && print("fold $fold: t=$(rtime(tstart)): training MPS with $(best_params)... ")
        mps, _... = fitMPS(X_train, y_train, opts);
        println(" done")
        p_fold = verbosity, "Fold $fold: ", tstart, nothing, nfolds
        res_iter = Dict(
            "fold"=>fold,
            "objective"=>string(objective),
            "train_inds"=>train_inds, 
            "test_inds"=>test_inds, 
            "optimiser"=>string(tuning_optimiser),
            "tuning_windows"=>tuning_windows, 
            "tuning_pms"=>tuning_pms,
            "eval_windows"=>eval_windows,
            "eval_pms"=>eval_pms,
            "time"=>time() - tbeg,
            "opts"=>opts, 
            "cache"=>cache,
            "loss"=>eval_loss(objective, mps, X_test, y_test, eval_windows; p_fold=p_fold, distribute=distribute_final_eval)
        )
        mps, X_train, X_test, y_train, y_test = nothing, nothing, nothing, nothing, nothing # attempt to force garbage collection - probably does nothing
        if write
            @save fname res_iter
            println("saved fold at $fname")
        end
        return res_iter
    end

    if distribute_folds
        if nprocs() == 1
            println("No workers")
        end
        threading = pmap(i -> is_omp_threading(), 1:nworkers())

        if ~all(threading)
            @warn "Using both threading and multiprocessing at the same time is not advised, set OMP_NUM_THREADS=1 when adding a new process to disable this messaage"
        end

        if ~distribute_cvfolds || nworkers() <= nfolds
            res = pmap(_eval_fold, fold_inds)

        else
            pools = divide_procs(workers(), nfolds)
            # @show pools
            res = pmap(_eval_fold, fold_inds, pools)
        end

    else
        if distribute_cvfolds
            pool = workers()
            res = map(f->_eval_fold(f, pool), fold_inds)
        else
            res = map(_eval_fold, fold_inds)
        end

    end
    if write 
        @save outfile res
        println("Results saved to $outfile")
        if delete_tmps
            rm(tmpdir; recursive=true)
        end
    end
    return res
end


# class free version
evaluate(X::AbstractMatrix, nfolds::Integer) = evaluate(X, zeros(Int, size(X,1)), nfolds, args...; kwargs...)