# [Hyperparameter Tuning](@id hyperparameters_top)

This tutorial for MPSTime will take you through tuning the hyperparameters of the [`fitMPS`](@ref) algorithm to maximise either imputation or classification performance.

## Setup

For this tutorial, we'll be solving a classification hyperoptimisation problem and an imputation hyperoptimisation problem use the same noisy trendy sinusoid dataset from the [`Classification`](@ref Classification_top) and [`Imputation`](@ref Imputation_top) sections.

```julia hyperopt
using MPSTime, Random
rng = Xoshiro(1); # fix rng seed
ntimepoints = 100; # specify number of samples per instance
ntrain_instances = 300; # specify num training instances
ntest_instances = 200; # specify num test instances
X_train = vcat(
    trendy_sine(ntimepoints, ntrain_instances ÷ 2; sigma=0.1, slope=[-3,0,3], period=(12,15), rng=rng)[1],
    trendy_sine(ntimepoints, ntrain_instances ÷ 2; sigma=0.1, slope=[-3,0,3], period=(16,19), rng=rng)[1]
);
y_train = vcat(
    fill(1, ntrain_instances ÷ 2),
    fill(2, ntrain_instances ÷ 2)
);
X_test = vcat(
    trendy_sine(ntimepoints, ntest_instances ÷ 2; sigma=0.2, slope=[-3,0,3], period=(12,15), rng=rng)[1],
    trendy_sine(ntimepoints, ntest_instances ÷ 2; sigma=0.2, slope=[-3,0,3], period=(16,19), rng=rng)[1]
);
y_test = vcat(
    fill(1, ntest_instances ÷ 2),
    fill(2, ntest_instances ÷ 2)
);
```
!!! info
    Given how computationally intensive hyperparameter tuning can be, if you're running these examples yourself it's a good idea to take advantage of the multiprocessing built into the MPSTime library (see the [Distributed Computing](@ref distributed_computing) section). 


## Hyperoptimising classification 
The hyperparameter tuning algorithms supported by MPSTime supports every numerical hyperparameter that may be specified by [`MPSOptions`](@ref). For this problem, we'll generate a small search space over the three most important hyperparameters: the maximum MPS bond dimension `chi_max`, the physical dimension `d`, and the learning rate `eta`. Every other hyperparamter will be left at its default value.

The variables to optimise, along with their upper and lower bounds are specified with the syntax `params = (<variable_name_1>=(<lower_bound_1>, <upper_bound_1>), ...)`, e.g.

```julia
params = (
    eta=(1e-3, 1e-1), 
    d=(2,8), 
    chi_max=(20,40)
) 
```
When solving real-world problems, it's a good idea to explore a larger `d` and `chi_max` search space, but for this example it will serve well enough.

### K-fold cross validation with tune()
To optimise the hyperparameters on your dataset, simply call [`tune`](@ref):

```julia-repl
julia> nfolds = 5;
julia> best_params, cache = tune(
    X_train, 
    y_train, 
    nfolds,
    params,
    MPSRandomSearch(); 
    objective=MisclassificationRate(), 
    maxiters=20, # for demonstration purposes only, typically this should be much larger
    logspace_eta=true
);
iter 1, cvfold 1: training MPS with (chi_max = 37, d = 8, eta = 0.023357214690901226)...
iter 1, cvfold 2: training MPS with (chi_max = 37, d = 8, eta = 0.023357214690901226)...
iter 1, cvfold 3: training MPS with (chi_max = 37, d = 8, eta = 0.023357214690901226)...
iter 1, cvfold 4: training MPS with (chi_max = 37, d = 8, eta = 0.023357214690901226)...
iter 1, cvfold 5: training MPS with (chi_max = 37, d = 8, eta = 0.023357214690901226)...
iter 1, cvfold 1: finished. MPS (chi_max = 37, d = 8, eta = 0.023357214690901226) finished in 128.25s (train=126.38s, loss=1.87s)
iter 1, cvfold 2: finished. MPS (chi_max = 37, d = 8, eta = 0.023357214690901226) finished in 129.0s (train=127.11s, loss=1.89s)
iter 1, cvfold 3: finished. MPS (chi_max = 37, d = 8, eta = 0.023357214690901226) finished in 128.57s (train=126.74s, loss=1.84s)
iter 1, cvfold 4: finished. MPS (chi_max = 37, d = 8, eta = 0.023357214690901226) finished in 128.78s (train=126.96s, loss=1.82s)
iter 1, cvfold 5: finished. MPS (chi_max = 37, d = 8, eta = 0.023357214690901226) finished in 127.77s (train=125.94s, loss=1.83s)
iter 1, t=139.39: Mean CV Loss: 0.040000000000000015
iter 2, cvfold 1: training MPS with (chi_max = 39, d = 6, eta = 0.0379269019073225)...
  
[...]

iter 20, cvfold 4: finished. MPS (chi_max = 21, d = 2, eta = 0.018329807108324356) finished in 13.52s (train=13.45s, loss=0.08s)
iter 20, cvfold 5: finished. MPS (chi_max = 21, d = 2, eta = 0.018329807108324356) finished in 14.98s (train=14.47s, loss=0.5s)
iter 20, t=843.55: Mean CV Loss: 0.08

julia> best_params
(chi_max = 31,
 d = 4,
 eta = 0.07847599703514611,)

julia> cache[values(best_params)] # retrieve loss from the cache
0.0033333333333333435 # equivalent to 99.67% accuracy
```
which returns `best params`: a named tuple containing the optimised hyperparameters, and `cache`: a dictionary that saves the mean loss of every tested hyperparameter combination.

The arguments used here are:
- `X_train`: timeseries data in a matrix, time series are rows.
- `y_train`: Vector of time series class labels.
- `nfolds`: Number of cross validiation folds to use. Folding type can be specified with the `foldmethod` keyword.
- `params`: Hyperparameters to tune and their upper and lower bounds, see previous section.
- `MPSRandomSearch()`: The tuning algorithm to use, see the [`tuning algorithm`](@ref hyper_algs) section.
- `objective=MisclassificationRate()`: Optimise the raw misclassification rate (1 - accuracy). The other options are `BalancedMisclassificationRate` (optimises balanced accuracy), or `ImputationLoss()` (see [imputation hyperoptimisation](@ref imputation_hyper)).
- `maxiters=20`: The maximum number of solver iterations. Here we use a very small number for demonstration reasons.
- `logspace_eta=true`: A useful option that tells the tuning algorithm to sample the eta search space logarithmically.

There are many more customisation options for [`tune`](@ref), see the docstring and extended help for more information / advanced usecases.

### Evaluating model performance with evaluate()
If you want to estimate the performance of MPSTime on a dataset, you can call the [`evaluate`](@ref) function, which resamples your data into train/test splits using a provided resampling strategy (default is k-fold cross validation), tunes each split on the "training" set, and evaluates the test set. It can be called with the following syntax:

```julia-repl
julia> nresamples = 3; # number of outer "resampling" folds - usually 30

julia> Xs = vcat(X_train, X_test);

julia> ys = vcat(y_train, y_test);

julia> results = evaluate(
    Xs,
    ys,
    nresamples,
    params,
    MPSRandomSearch(); 
    n_cvfolds=nfolds, # the number of folds used by tune()
    objective=MisclassificationRate(),
    tuning_maxiters=20
);
Beginning fold 1:
Fold 1: iter 1, cvfold 1: training MPS with (chi_max = 40, d = 8, eta = 0.09478947368421053)...
Fold 1: iter 1, cvfold 2: training MPS with (chi_max = 40, d = 8, eta = 0.09478947368421053)...
Fold 1: iter 1, cvfold 3: training MPS with (chi_max = 40, d = 8, eta = 0.09478947368421053)...
Fold 1: iter 1, cvfold 4: training MPS with (chi_max = 40, d = 8, eta = 0.09478947368421053)...
Fold 1: iter 1, cvfold 5: training MPS with (chi_max = 40, d = 8, eta = 0.09478947368421053)...

[...]

Fold 3: iter 20, cvfold 4: finished. MPS (chi_max = 28, d = 2, eta = 0.06873684210526317) finished in 19.26s (train=19.14s, loss=0.12s)
Fold 3: iter 20, cvfold 5: finished. MPS (chi_max = 28, d = 2, eta = 0.06873684210526317) finished in 19.86s (train=19.78s, loss=0.08s)
Fold 3: iter 20, t=875.4: Mean CV Loss: 0.011985526910900046
fold 3: t=2899.6: training MPS with (chi_max = 39, d = 3, eta = 0.016631578947368424)...  done

julia> results[1]
Dict{String, Any} with 13 entries:
  "time"           => 1055.61
  "objective"      => "MisclassificationRate()"
  "train_inds"     => [340, 445, 262, 132, 89, 379, 225, 59, 224, 57  …  495, 484, 355, 322, 284, 363, …
  "optimiser"      => "MPSRandomSearch(:LatinHypercube)"
  "fold"           => 1
  "test_inds"      => [133, 477, 112, 148, 13, 453, 342, 83, 252, 455  …  151, 483, 74, 380, 61, 297, 1…
  "tuning_windows" => nothing
  "eval_windows"   => nothing
  "cache"          => Dict((39, 6, 0.0166316)=>0.0149706, (33, 7, 0.0426842)=>0.0300769, (36, 7, 0.0635…
  "tuning_pms"     => nothing
  "loss"           => [0.00598802]
  "eval_pms"       => nothing
  "opts"           => MPSOptions(-5, 10, 32, 0.0531053, 4, :Legendre_No_Norm, false, 2, 1.0e-10, 1, Flo…

julia> losses = getindex.(results, "loss")
3-element Vector{Vector{Float64}}:
 [0.005988023952095856]
 [0.005988023952095856]
 [0.0060240963855421326]

```

Which outputs a results dictionary, containing the losses on each resample fold, as well as a lot of other useful information. See the [`docstring`](@ref evaluate) for more detail as well as a plethora of customistation options.

A very common extension of `evaluate` is to customise the resampling strategy. The simplest way to do this is to pass a vector of `(training_indices, testing_indices)` to the `foldmethod` keyword. For example, to use scikit-learn's [`StratifiedShuffleSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html#sklearn.model_selection.StratifiedShuffleSplit) to generate the train/test splits:


```julia-repl
julia> using PyCall

julia> nresamples = 3;

julia> py"""
    from sklearn.model_selection import StratifiedShuffleSplit # requires a python environment with sklearn installed
    sp = StratifiedShuffleSplit(n_splits=$nresamples, test_size=$(length(y_test)), random_state=1)
    folds_py = sp.split($Xs, $ys)
    """

julia> folds = [(tr_inds .+ 1, te_inds .+ 1) for (tr_inds, te_inds) in py"folds_py"] # shift indices up by 1
3-element Vector{Tuple{Vector{Int64}, Vector{Int64}}}:
 ([197, 472, 462, 108, 133, 258, 179, 57, 149, 373  …  279, 94, 234, 473, 319, 378, 387, 92, 359, 35], [354, 313, 137, 239, 316, 479, 274, 145, 134, 485  …  110, 417, 346, 141, 165, 50, 77, 23, 347, 130])
 ([399, 105, 322, 289, 281, 187, 131, 18, 56, 231  …  463, 38, 491, 288, 408, 430, 330, 185, 481, 353], [345, 469, 396, 10, 96, 452, 245, 76, 367, 84  …  202, 153, 94, 446, 372, 152, 79, 387, 51, 301])

[...]

julia> results = evaluate(
    Xs,
    ys,
    nresamples,
    params,
    MPSRandomSearch(); 
    objective=MisclassificationRate(),
    tuning_maxiters=20,
    foldmethod=folds
);
[...]
```


## [Hyperoptimising imputation](@id imputation_hyper)
The [`tune`](@ref) and [`evaluate`](@ref) methods may both be used to minimise imputation loss, with a small amount of extra setup. Setting `objective=ImputationLoss()` will optimise an MPS for imputation performance by minimising the mean absolute error (MAE) between predicted and unseen data. To accomplish this, MPSTime takes data from the test (or validation) set, corrupts a portion of it, and then predicts what the corrupted data should be based on the uncorrupted values. There are two methods for how the test (or validation) data can be corrupted.
1) Setting the `windows` (or `eval_windows`) keyword in [`tune`](@ref) (or [`evaluate`](@ref), respectively) to a vector of 'windows'. Each window is a vector of missing/corrupted data indices, for example
```julia
windows = [[1,3,7],[4,5,6]]
```
will take each timeseries in the test set, and create two 'corrupted' test series, missing the 1st, 3rd, and 7th; and the 4th, 5th, and 6th values respectively.
2) Setting the `pms` (or `eval_pms`) keyword in [`tune`](@ref) (or [`evaluate`](@ref), respectively) to a vector of 'percentage missings'. This generates corrupted time series by removing randomly selected contiguous blocks that make up a specified percentage of the data. For example, 
```iulia
pms = [0.05, 0.05, 0.6, 0.95]
```
will generate four corrupted time series from each element of the test (or validation) set. Two will have missing blocks that make up 5% of their length, and one each will have blocks with 60% and 95% missing. The imputation tuning loss is the average MAE of imputing every window on every element of the test (or validation) set.

**Example: Calling tune with percentages missing**
Tune the MPS on an imputation problem with randomly selected 5%, 15%, 25%, ... , 95% long missing blocks.
```julia
params = (
    d=(8,12), 
    chi_max=(30,50)
)

best_params, cache = tune(
    X_train, 
    y_train, 
    nfolds,
    params,
    MPSRandomSearch(); 
    objective=ImputationLoss(), 
    pms=collect(0.05:0.1:0.95),
    maxiters=20,
    logspace_eta=true
)
```

```
iter 1, cvfold 1: training MPS with (chi_max = 47, d = 12)...
iter 1, cvfold 2: training MPS with (chi_max = 47, d = 12)...
iter 1, cvfold 3: training MPS with (chi_max = 47, d = 12)...
iter 1, cvfold 4: training MPS with (chi_max = 47, d = 12)...
iter 1, cvfold 5: training MPS with (chi_max = 47, d = 12)...
iter 1, cvfold 1: finished. MPS (chi_max = 47, d = 12) finished in 515.58s (train=402.83s, loss=112.75s)
iter 1, cvfold 2: finished. MPS (chi_max = 47, d = 12) finished in 511.0s (train=399.52s, loss=111.48s)
iter 1, cvfold 3: finished. MPS (chi_max = 47, d = 12) finished in 516.33s (train=406.92s, loss=109.41s)
iter 1, cvfold 4: finished. MPS (chi_max = 47, d = 12) finished in 519.06s (train=405.13s, loss=113.93s)
iter 1, cvfold 5: finished. MPS (chi_max = 47, d = 12) finished in 522.12s (train=406.87s, loss=115.25s)
iter 1, t=534.28: Mean CV Loss: 0.44716868140737487
iter 2, cvfold 1: training MPS with (chi_max = 50, d = 11)...

[...]

iter 20, cvfold 4: finished. MPS (chi_max = 31, d = 8) finished in 112.81s (train=66.33s, loss=46.48s)
iter 20, cvfold 5: finished. MPS (chi_max = 31, d = 8) finished in 113.9s (train=66.32s, loss=47.58s)
iter 20, t=5223.05: Mean CV Loss: 0.4755302875557089
```


```julia-repl
julia> best_params
(chi_max = 48,
 d = 9,)

julia> cache[values(best_params)]
0.39402101779354365
```

**Example: Using evaluate with the Missing Completely At Random tool**
Tune the MPS on an imputation problem by completely randomly corrupting 5%, 15%, 25%, ... , or 95% of each test (or validation) time series. See the [`Missing Completely at Random`](@ref mcar) tool.

```julia
inds = collect(1:size(Xs,2))
rng = Xoshiro(42)
pms = 0.05:0.1:0.95
mcar_windows = [mcar(inds, pm; rng=rng)[2] for pm in pms]
results = evaluate(
    Xs,
    ys,
    nresamples,
    params,
    MPSRandomSearch(); 
    objective=ImputationLoss(),
    eval_windows=mcar_windows,
    tuning_maxiters=20,
)
```
```
Fold 1: iter 1, cvfold 1: training MPS with (chi_max = 50, d = 12)...
     
[...]

Fold 3: iter 20, cvfold 5: finished. MPS (chi_max = 35, d = 8) finished in 635.05s (train=255.05s, loss=380.0s)
Fold 3: iter 20, t=20328.2: Mean CV Loss: 0.24440015696620948
fold 3: t=42210.43: training MPS with (chi_max = 48, d = 12)...  done
```

```julia-repl
julia> losses = getindex.(results, "opts");

julia> mean.(losses)
3-element Vector{Float64}:
 0.21739511880650658
 0.21129840885487566
 0.21441462843171047

julia> getindex.(results, "opts") # print out the MPSOptions objects
3-element Vector{MPSOptions}:
 MPSOptions(-5, 10, 47, 0.01, 11, :Legendre_No_Norm, false, 2, 1.0e-10, 1, Float64, :KLD, :TSGO, false,
(false, true), false, false, false, true, false, false, 1234, 4, -1, (0.0, 1.0), false, "divide_and_conquer")
 MPSOptions(-5, 10, 50, 0.01, 12, :Legendre_No_Norm, false, 2, 1.0e-10, 1, Float64, :KLD, :TSGO, false,
(false, true), false, false, false, true, false, false, 1234, 4, -1, (0.0, 1.0), false, "divide_and_conquer")
 MPSOptions(-5, 10, 48, 0.01, 12, :Legendre_No_Norm, false, 2, 1.0e-10, 1, Float64, :KLD, :TSGO, false,
(false, true), false, false, false, true, false, false, 1234, 4, -1, (0.0, 1.0), false, "divide_and_conquer")

julia> results[1]
Dict{String, Any} with 13 entries:
  "time"           => 6152.08
  "objective"      => "ImputationLoss()"
  "train_inds"     => [340, 445, 262, 132, 89, 379, 225, 59, 224, 57  …  495, 484, 355, 322, 284, 363, …
  "optimiser"      => "MPSRandomSearch(:LatinHypercube)"
  "fold"           => 1
  "test_inds"      => [133, 477, 112, 148, 13, 453, 342, 83, 252, 455  …  151, 483, 74, 380, 61, 297, 1…
  "tuning_windows" => [[35, 97], [10, 36, 38, 49, 60, 62, 65, 72, 81, 82, 88, 92, 93, 97], [9, 16, 24, …
  "eval_windows"   => [[35, 97], [10, 36, 38, 49, 60, 62, 65, 72, 81, 82, 88, 92, 93, 97], [9, 16, 24, …
  "cache"          => Dict((49, 10)=>0.217511, (41, 10)=>0.233205, (45, 8)=>0.231089, (31, 8)=>0.259456…
  "tuning_pms"     => nothing
  "loss"           => [0.192093, 0.193801, 0.183081, 0.189859, 0.194329, 0.19409, 0.211727, 0.235688, 0…
  "eval_pms"       => nothing
  "opts"           => MPSOptions(-5, 10, 47, 0.01, 11, :Legendre_No_Norm, false, 2, 1.0e-10, 1, Float64…


```

## Minimising a custom loss function
Custom objectives can be used by implementing a custom loss value type (`CustomLoss <: MPSTime.TuningLoss`) and extending the definition of [`MPSTime.eval_loss`](@ref) with the signature
```julia
eval_loss(
    CustomLoss(), 
    mps::TrainedMPS, 
    X_validation::AbstractMatrix, 
    y_validation::AbstractVector, 
    windows; 
    p_fold=nothing, 
    distribute::Bool=false
) -> Union{Float64, Vector{Float64}}
```

As a simple example, we could implement a custom misclassification rate with the following:
```julia-repl
julia> import MPSTime: TuningLoss, eval_loss;
julia> struct CustomMisclassificationRate <: TuningLoss end;

julia> function eval_loss(
        ::CustomMisclassificationRate, 
        mps::TrainedMPS, 
        X_val::AbstractMatrix, 
        y_val::AbstractVector, 
        windows; 
        p_fold=nothing,
        distribute::Bool=false
    )
    return [1. - mean(classify(mps, X_val) .== y_val)] # misclassification rate, vector for type stability
end;

julia> results = evaluate(
    Xs,
    ys,
    nfolds,
    params,
    MPSRandomSearch(); 
    objective=CustomMisclassificationRate(),
    tuning_maxiters=20
)
[...]
```
Since imputation type losses have multiple windows, the general output type of the `eval_loss` function is a vector. Because of this, the `tune()` function always optimises `mean(eval_loss(...))`

The `p_fold` variable is a tuple containing information used for logging. The `distribute` flag is enabled by `evaluate` when `distribute_final_eval` is true. As an example of implementing both of these, here's MPSTime's implementation of `eval_loss(::ImputationLoss, ...)`:

!!! details "`eval_loss` source code"
    ```julia
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

        # conversion from instance index to something MPS_impute understands. 
        cmap = countmap(y_val) # from StatsBase
        classes = vcat([fill(k,v) for (k,v) in pairs(cmap)]...)
        class_ind = vcat([1:v for v in values(cmap)]...)

        if distribute
            loss_by_window = @sync @distributed (+) for inst in 1:numval
                logging && print(pre_string, "$foldstr Evaluating instance $inst/$numval...")
                t = time()
                ws = Vector{Float64}(undef, length(windows))
                for (iw, impute_sites) in enumerate(windows)
                    stats = MPS_impute(imp, classes[inst], class_ind[inst], impute_sites, method; NN_baseline=false, plot_fits=false)[4]
                    ws[iw] = stats[1][:MAE]
                end
                logging && println("done ($(MPSTime.rtime(t))s)") # rtime just nicely prints time elapsed since $t in seconds
                ws
            end
            loss_by_window /= numval
        else
            instance_scores = Matrix{Float64}(undef, numval, length(windows)) # score for each instance across all % missing
            for inst in 1:numval
                logging && print(pre_string, "$foldstr Evaluating instance $inst/$numval...")
                t = time()
                for (iw, impute_sites) in enumerate(windows)
                    stats = MPS_impute(imp, classes[inst], class_ind[inst], impute_sites, method; NN_baseline=false, plot_fits=false)[4]
                    instance_scores[inst, iw] = stats[1][:MAE]
                end
                logging && println("done ($(MPSTime.rtime(t))s)") # rtime just nicely prints time elapsed since $t in seconds
            end
            loss_by_window = mean(instance_scores; dims=1)[:]
        end
        

        return loss_by_window
    end

    ```


## [Choice of tuning algorithm](@id hyper_algs)
The hyperparameter tuning algorithm used by [`tune`](@ref) (or [`evaluate`](@ref)) can be specified with the `optimiser` argument. This supports the default builtin [`MPSRandomSearch`](@ref) methods, as well as (in theory) any solver that is supported by the [`Optimization.jl interface`](https://docs.sciml.ai/Optimization/stable). Note that many of these solvers struggle with discrete search spaces, such as tuning the integer valued `chi_max` and `d`. Some of them require initial conditions (set `provide_x0=true`), and some require no initial conditions (set `provide_x0=false`), so your mileage may vary. By default, `tune()` handles optimisers attempting to evaluate discrete hyperparameters at a non-integer value by rounding and using its own cache to avoid rounding based cache misses. This is effective, but has the downside of causing `maxiters` to be inaccurate (as repeated hyperparameter evaluations caused by rounding result in a 'skipped' iteration).

```@docs
MPSRandomSearch
```


## [Distributed computing](@id distributed_computing)
Both tune [`tune`](@ref) and [`evaluate`](@ref) support several different parallel processing paradigms for different use cases, compatible with processors added via Distributed.jl's [`addprocs`](@extref Distributed.addprocs) function. 
For example, to distribute each fold of the classification style evaluation above, run:

```julia-repl
using Distributed
nfolds = 30;

e = copy(ENV);
e["OMP_NUM_THREADS"] = "1"; # attempt to prevent threading
e["JULIA_NUM_THREADS"] = "1"; # attempt to prevent threading

addprocs(nfolds; env=e);
@everywhere using MPSTime

Xs = vcat(X_train, X_test);
ys = vcat(y_train, y_test);

results = evaluate(
    Xs,
    ys,
    nfolds,
    params,
    MPSRandomSearch(); 
    objective=MisclassificationRate(),
    tuning_maxiters=20,
    distribute_folds=true
)

[...]

```

See the respective docstrings for more information.


## Docstrings
### Hyperparameter tuning
```@docs
tune
evaluate
```

### Hyperparameter tuning utilities
```@docs
MPSTime.make_stratified_cvfolds
eval_loss
```
```@docs; canonical=false
MPSRandomSearch
```