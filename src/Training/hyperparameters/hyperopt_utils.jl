abstract type TuningLoss end
abstract type ClassificationLoss <: TuningLoss end
struct MisclassificationRate <: ClassificationLoss end
struct BalancedMisclassificationRate <: ClassificationLoss end

struct ImputationLoss <: TuningLoss end 

"""
```Julia
    MPSRandomSearch(sampling::Symbol=:LatinHypercube)
```
Value type used to specify a random search algorithm for [`hyperparameter tuning`](@ref MPSTime.tune) an MPS. 

`Sampling` Specifies the method used to determine the search space. The supported sampling methods are 
- `:LatinHypercube`: An implementation of [`LatinHypercubeSampling.jl`](https://www.juliapackages.com/p/latinhypercubesampling)'s random (pseudo-) Latin Hypercube\
search space generator. Supports both discrete and continuous hyperparameters.
- `:UniformRandom`: Generate a search space by randomly sampling from the interval [lower bound, upper bound] for each hyperparameter. Supports both discrete and\
continuous hyperparameters.
- `Exhaustive`: Perform an exhaustive gridsearch of all hyperparameters within the lower and upper bounds. Only supports discrete hyperparameters. Not actually a random search.  
"""
struct MPSRandomSearch
    sampling::Symbol

    function MPSRandomSearch(sampling::Symbol=:LatinHypercube)
        if sampling in [:LatinHypercube, :UniformRandom, :Exhaustive]
            return new(sampling)
        else
            throw(ArgumentError("Unknown sampling type, expected :LatinHypercube, :UniformRandom, or :Exhaustive"))
        end
    end
end


function rtime(tstart::Float64)

    return round(time() - tstart; digits=2)
end

function rtime(tstart::Float64, tend::Float64)

    return round(tend - tstart; digits=2)
end

function is_omp_threading()
    return "OMP_NUM_THREADS" in keys(ENV) && ENV["OMP_NUM_THREADS"] == "1"
end


function divide_procs(workers, nfolds)
    nprocs = length(workers)
    i = 1
    j = 0
    split = [Int[] for _ in 1:nfolds]
    while i <= nprocs
        push!(split[j % nfolds + 1], workers[i])
        i += 1
        j += 1
    end
    return split
end

#TODO fix
# function resample_folds(X::AbstractMatrix, k::Integer, train_ratio::Float64; shuffle_first=false, rng::Union{Nothing, AbstractRNG}=nothing)
#     if isnothing(rng)
#         rng = Xoshiro()
#     end
#     folds = fill(Vector{Vector{Int}}(undef, 2), k)
#     h1 = MLJ.Holdout(; fraction_train=train_ratio, shuffle=shuffle_first, rng=rng)
#     folds[1] .= MLJBase.train_test_pairs(h1, 1:size(X,1))[1]

#     for i in 2:k
#         h = MLJ.Holdout(; fraction_train=train_ratio, shuffle=true, rng=rng)
#         folds[i] .= MLJBase.train_test_pairs(h, 1:size(X,1))[1]
#     end
#     return folds
# end

# function resample_folds(X::AbstractMatrix, y::AbstractVector, k::Integer, train_ratio::Float64; kwargs...)
#     return resample_folds(X, k, train_ratio; kwargs...)
# end

"""
```
make_stratified_cvfolds(
    Xs::AbstractMatrix, 
    ys::AbstractVector, 
    nfolds::Integer; 
    rng=Union{Integer, AbstractRNG}, 
    shuffle::Bool=true
) -> folds::Vector{Vector{Vector{Int}}}
 
Creates `nfold`-fold stratified cross validation train/validation splits for hyperparameter tuning, with the form:

```
julia> train_indices_fold_i, validation_indices_fold_i = folds[i]
```
Uses MLJs [`StratifiedCV()`](@ref MLJ.StratifiedCV) method. 
 ```

"""
function make_stratified_cvfolds(Xs::AbstractMatrix, ys::AbstractVector, nfolds::Integer; rng=Union{Integer, AbstractRNG}, shuffle::Bool=true)
    stratified_cv = MLJ.StratifiedCV(; nfolds=nfolds,shuffle=shuffle, rng=rng)

    return MLJBase.train_test_pairs(stratified_cv, 1:size(Xs,1), ys)
end

function make_windows(windows::Union{Nothing, AbstractVector, Dict}, pms::Union{Nothing, AbstractVector}, X::AbstractMatrix, rng::AbstractRNG=Random.GLOBAL_RNG)

    if ~isnothing(windows) 
        if ~isnothing(pms)
            throw(ArgumentError("Cannot specifiy both windows and pms!"))
        end

        if windows isa Dict
            return vcat([windows[key] for key in sort(collect(keys(windows)))]...)

        else
            @assert all(isa.(windows, AbstractVector)) "Elements of windows must be window vectors!"
        end
        return windows
    elseif ~isnothing(pms) 
        ts_length = size(X, 2)
        if eltype(pms) <: Integer
            pms ./ 100
        end
        return [mar(collect(1.:ts_length), pm; rng=rng)[2] for pm in pms]
    else
        throw(ArgumentError("Must specifiy either windows or pms when measuring Imputation Loss!"))
        return []
    end
end
"""
```Julia
eval_loss(
    ::TuningLoss, 
    mps::TrainedMPS, 
    X_val::AbstractMatrix, 
    y_val::AbstractVector, 
    windows::Union{Nothing, AbstractVector}=nothing;
    p_fold::Union{Nothing, Tuple}=nothing,
    distribute::Bool=false,
    )
```

Evaluate the `TuningLoss` of `mps` on the validation time-series dataset specified by `X_val`, `y_val`.

`p_fold` is to allow verbose logging during runs of [`tune`](@ref) and [`evaluate`](@ref). When computing an imputation loss, \
`windows` are used to compute imputation losses, as specified in [`tune`](@ref), and `distribute` will distribute the loss calculation across each time-series instance.

"""
function eval_loss end
function eval_loss(::BalancedMisclassificationRate, mps::TrainedMPS, X_val::AbstractMatrix, y_val::AbstractVector, windows; p_fold=nothing, distribute::Bool=false)
    recall_sum = 0
    y_pred = classify(mps, X_val)
    classes = unique(vcat(y_val, y_pred))

    for cls in classes
        tp = sum((y_val .== cls) .& (y_pred .== cls))
        fn = sum((y_val .== cls) .& (y_pred .!= cls))
        recall = tp / (tp + fn + eps())
        recall_sum += recall
    end

    bal_acc = recall_sum / length(classes)
    
    return [1. - bal_acc]

end

function eval_loss(::MisclassificationRate, mps::TrainedMPS, X_val::AbstractMatrix, y_val::AbstractVector, windows; p_fold=nothing, distribute::Bool=false)
    return [1. - mean(classify(mps, X_val) .== y_val)] # misclassification rate, vector for type stability
end

function eval_loss(
    ::ImputationLoss, 
    mps::TrainedMPS, 
    X_val::AbstractMatrix, 
    y_val::AbstractVector, 
    windows::Union{Nothing, AbstractVector}=nothing;
    p_fold::Union{Nothing, Tuple}=nothing,
    distribute::Bool=false,
    method::Symbol=:median
    )
    
    if ~isnothing(p_fold)
        verbosity, pre_string, tstart, fold, nfolds = p_fold
        logging = verbosity >= 2
        foldstr = isnothing(fold) ? "" : "cvfold $fold:"
    else
        logging = false
    end
    imp = init_imputation_problem(mps, X_val, y_val, verbosity=-5);
    numval = size(X_val, 1)
    # conversion from inst to something MPS_impute understands. #TODO This is awful, should fix

    cmap = countmap(y_val)
    classes= vcat([fill(k,v) for (k,v) in pairs(cmap)]...)
    class_ind = vcat([1:v for v in values(cmap)]...)

    if distribute
        loss_by_window = @sync @distributed (+) for inst in 1:numval
            logging && print(pre_string, "$foldstr Evaluating instance $inst/$numval...")
            t = time()
            ws = Vector{Float64}(undef, length(windows))
            for (iw, impute_sites) in enumerate(windows)
                # impute_sites = mar(X_val[inst, :], p)[2]
                stats = MPS_impute(imp, classes[inst], class_ind[inst], impute_sites, method; NN_baseline=false, plot_fits=false)[4]
                ws[iw] = stats[1][:MAE]
            end
            logging && println("done ($(rtime(t))s)")
            ws
        end
        loss_by_window /= numval
    else
        instance_scores = Matrix{Float64}(undef, numval, length(windows)) # score for each instance across all % missing
        for inst in 1:numval
            logging && print(pre_string, "$foldstr Evaluating instance $inst/$numval...")
            t = time()
            for (iw, impute_sites) in enumerate(windows)
                # impute_sites = mar(X_val[inst, :], p)[2]
                stats = MPS_impute(imp, classes[inst], class_ind[inst], impute_sites, method; NN_baseline=false, plot_fits=false)[4]
                instance_scores[inst, iw] = stats[1][:MAE]
            end
            logging && println("done ($(rtime(t))s)")
        end
        loss_by_window = mean(instance_scores; dims=1)[:]
    end
    

    return loss_by_window
end
