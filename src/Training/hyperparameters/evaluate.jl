function is_omp_threading()
    return "OMP_NUM_THREADS" in keys(ENV) && ENV["OMP_NUM_THREADS"] == "1"
end




function evaluate(
    X::AbstractMatrix, 
    y::AbstractVector, 
    tuning_parameters::NamedTuple,
    tuning_optimiser=SAMIN(); # bounded simulated annealing from Optim
    objective::TuningLoss=ImputationLoss(), 
    verbosity::Integer=1,
    opts0::AbstractMPSOptions=MPSOptions(; verbosity=-5, log_level=-1),
    input_supertype::Type=Float64,
    tuning_opts0::AbstractMPSOptions=opts0,
    nfolds::Integer=30,
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
    tuning_pms::Union{Nothing, AbstractVector}= nothing,
    tuning_windows::Union{Nothing, AbstractVector, Dict}= nothing,
    tuning_abstol::Float64=1e-3,
    tuning_maxiters::Integer=500,
    distribute_folds::Bool=false,   
    distribute_cvfolds::Bool=false,
    write::Bool=false,
    writedir::String="evals",
    simname::String="",
    collect_tmps::Bool=length(fold_inds)==nfolds # default to delete only if the entire eval is done at once
    )
    abs_rng = rng isa Integer ? Xoshiro(rng) : rng

    if objective isa ImputationLoss
        eval_windows = make_windows(eval_windows, eval_pms, X, abs_rng)
        # tuning_windows = make_windows(tuning_windows, tuning_pms, X, abs_rng)
    end

    folds::Vector = foldmethod isa Function ? foldmethod(X,y, nfolds; rng=abs_rng) : foldmethod

    if isempty(simname)
        simname = string(objective) * "_" * string(tuning_optimiser) * "_f=" * string(nfolds) * "_cv=" * string(n_cvfolds) * "_iters=" * string(tuning_maxiters)
    end
    outfile = strip(writedir, '/') * "/" * strip(simname, '/') * ".jld2"
    tmpdir = strip(writedir, '/') * "/" * strip(simname, '/') * "_tmp/"
    mkpath(tmpdir)
    tstart = time()
    function _eval_fold(fold, cv_workers=Int[])
        Random.seed!(fold)
        println("Beginning fold $fold:")
        tbeg = time()
        (train_inds, test_inds) = folds[fold]
        X_train, y_train, X_test, y_test = X[train_inds,:], y[train_inds], X[test_inds,:], y[test_inds]
    
        abs_rng_inner = tuning_rng[fold] isa Integer ? Xoshiro(tuning_rng[fold]) : tuning_rng[fold]
        tuning_windows_inner = make_windows(tuning_windows, tuning_pms, X, abs_rng_inner)
        best_params = tune(
            X_train, 
            y_train, 
            tuning_parameters,
            tuning_optimiser; 
            objective=objective, 
            opts0=tuning_opts0,
            input_supertype=input_supertype,
            provide_x0=provide_x0,
            logspace_eta=logspace_eta,
            nfolds=n_cvfolds, 
            pms=nothing,
            windows=tuning_windows_inner,
            abstol=tuning_abstol, 
            maxiters=tuning_maxiters,
            verbosity=verbosity,
            rng=abs_rng_inner,
            foldmethod=tuning_foldmethod,
            distribute_folds=distribute_cvfolds,
            workers=cv_workers,
            pre_string="Fold $fold: "
        )
        # @show best_params
        opts = _set_options(opts0; best_params...)
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
            "loss"=>eval_loss(objective, mps, X_test, y_test, eval_windows; p_fold=p_fold, distribute=distribute_folds)
        )
        mps, X_train, X_test, y_train, y_test = nothing, nothing, nothing, nothing, nothing # attempt to force garbage collection - probably does nothing
        if write
            @save (tmpdir * "f" * string(fold) * ".jld2") res_iter
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
        res = map(_eval_fold, fold_inds)

    end
    if write && collect_tmps
        @save outfile res
        rm(tmpdir; recursive=true)
    end
    return res
end