abstract type TuningLoss end
struct MisClassificationRate <: TuningLoss end
struct ImputationLoss <: TuningLoss end 
struct BalancedMisclassificationRate <: TuningLoss end

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

function make_folds(X::AbstractMatrix, k::Int; rng::Union{Nothing, AbstractRNG}=nothing)
    if isnothing(rng)
        rng = Xoshiro()
    end
    ninstances = size(X, 1)
    X_idxs = randperm(rng, ninstances)
    # split into k folds
    fold_size = ceil(Int, ninstances/k)
    all_folds = [X_idxs[(i-1)*fold_size+1 : min(i*fold_size, ninstances)] for i in 1:k]
    # build pairs
    X_train_idxs = Vector{Vector{Int}}(undef, k)
    X_val_idxs = Vector{Vector{Int}}(undef, k)
    for i in 1:k
        X_val_idxs[i] = all_folds[i]
        X_train_idxs[i] = vcat(all_folds[1:i-1]..., all_folds[i+1:end]...)
    end
    return zip(X_train_idxs, X_val_idxs)
end

function make_stratified_cvfolds(X::AbstractMatrix, y::AbstractVector, nfolds::Integer; rng=Union{Integer, AbstractRNG}, shuffle::Bool=true)
    stratified_cv = MLJ.StratifiedCV(; nfolds=nfolds,shuffle=shuffle, rng=rng)

    return MLJBase.train_test_pairs(stratified_cv, 1:size(X,1), y)
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
        throw(ArgumentError("Cannot specifiy both windows and pms!"))
        return []
    end
end


function rtime(tstart::Float64)

    return round(time() - tstart; digits=2)
end

function rtime(tstart::Float64, tend::Float64)

    return round(tend - tstart; digits=2)
end

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
