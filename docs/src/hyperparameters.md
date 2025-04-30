# [Hyperparameter Tuning](@id hyperparameters_top)

This tutorial for MPSTime will take you through tuning the hyperparameters of the [`fitMPS`](@ref) algorithm to maximise either imputation or classification performance.

## Setup

For this tutorial, we'll be solving a classification hyperoptimisation problem and an imputation hyperoptimisation problem use the same noisy trendy sinusoid dataset from the [`Classification`](@ref Classification_top) and [`Imputation`](@ref Imputation_top) sections.

```@repl hyperopt
using MPSTime, Random
rng = Xoshiro(1); # define trendy sine function
ntimepoints = 100; # specify number of samples per instance
ntrain_instances = 300; # specify num training instances
ntest_instances = 200; # specify num test instances
X_train = vcat(
    trendy_sine(ntimepoints, ntrain_instances ÷ 2; sigma=0.1, rng=rng)[1],
    trendy_sine(ntimepoints, ntrain_instances ÷ 2; sigma=0.9, rng=rng)[1]
);
y_train = vcat(
    fill(1, ntrain_instances ÷ 2),
    fill(2, ntrain_instances ÷ 2)
);
X_test = vcat(
    trendy_sine(ntimepoints, ntest_instances ÷ 2; sigma=0.1, rng=rng)[1],
    trendy_sine(ntimepoints, ntest_instances ÷ 2; sigma=0.9, rng=rng)[1]
);
y_test = vcat(
    fill(1, ntest_instances ÷ 2),
    fill(2, ntest_instances ÷ 2)
);
```
!!! info
    Given how computationally intensive hyperparameter tuning can be, if you're running these examples yourself it's a good idea to take advantage of the multiprocessing built into the MPSTime library (see the [Distributed Computing](@ref distributed_computing) section). 


## Hyperoptimising classification 
The hyperparameter tuning algorithms supported by MPSTime support every numerical hyperparameter that may be specified by [`MPSOptions`](@ref). For this problem, we'll generate a small search space over the three most important hyperparameters: the maximum MPS bond dimension `chi_max`, the physical dimension `d`, and the learning rate `eta`. Every other hyperparamter will be left at its default value.

The variables to optimise, along with their upper and lower bounds are specified with the syntax `params = (<variable_name_1>=(<lower bound>, <upper bound>))`, e.g.

```@repl hyperopt
params = (
    eta=(1e-3, 1e-1), 
    d=(5,7), 
    chi_max=(20,25)
) 
```
To solve real-world problems, the upper bounds on `d` and `chi_max` should be set much higher (e.g. 10 and 40), however the small search space will serve well enough for this example.

### K-fold cross validation with tune()
To optimise the hyperparameters on your dataset, simply call tune():

```julia-repl
julia> nfolds = 5
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
# iter 1, cvfold 1: training MPS with (chi_max = 25, d = 7, eta = 0.0379269019073225)...
# iter 1, cvfold 1: training MPS with (chi_max = 25, d = 7, eta = 0.0379269019073225)...
# iter 1, cvfold 1: finished. MPS (chi_max = 25, d = 7, eta = 0.0379269019073225) finished in 68.38s (train=66.85s, loss=1.53s)
# iter 1, cvfold 2: training MPS with (chi_max = 25, d = 7, eta = 0.0379269019073225)...
# iter 1, cvfold 2: finished. MPS (chi_max = 25, d = 7, eta = 0.0379269019073225) finished in 39.89s (train=39.81s, loss=0.08s)
# iter 1, cvfold 3: training MPS with (chi_max = 25, d = 7, eta = 0.0379269019073225)...
# iter 1, cvfold 3: finished. MPS (chi_max = 25, d = 7, eta = 0.0379269019073225) finished in 38.94s (train=38.86s, loss=0.08s)
# iter 1, cvfold 4: training MPS with (chi_max = 25, d = 7, eta = 0.0379269019073225)...
# iter 1, cvfold 4: finished. MPS (chi_max = 25, d = 7, eta = 0.0379269019073225) finished in 39.12s (train=39.04s, loss=0.08s)
# iter 1, cvfold 5: training MPS with (chi_max = 25, d = 7, eta = 0.0379269019073225)...
# iter 1, cvfold 5: finished. MPS (chi_max = 25, d = 7, eta = 0.0379269019073225) finished in 39.07s (train=38.99s, loss=0.08s)
# iter 1, t=232.71: Mean CV Loss: 0.19666666666666666
# iter 2, cvfold 1: training MPS with (chi_max = 24, d = 7, eta = 0.023357214690901226)...
# [...]
# iter 20, cvfold 5: finished. MPS (chi_max = 20, d = 5, eta = 0.018329807108324356) finished in 23.75s (train=23.68s, loss=0.07s))
# iter 20, t=2797.13: Mean CV Loss: 0.32666666666666666

best_params
# (chi_max = 23,
#  d = 7,
#  eta = 0.008858667904100823,)

```
which returns `best params`: a named tuple containing the optimised hyperparameters, and `cache`: a dictionary that saves the mean loss of every tested hyperparameter combination.

The arguments used here are:
- `X_train`: timeseries data in a matrix, time series are rows.
- `y_train`: Vector of time series class labels.
- `nfolds`: Number of cross validiation folds to use. Folding type can be specified with the `foldmethod` keyword.
- `params`: Hyperparameters to tune and their upper and lower bounds, see previous section.
- `MPSRandomSearch()`: The tuning algorithm to use, see [`the tuning algorithm`](@ref hyper_algs) section.
- `objective=MisclassificationRate()`: Optimise the raw classification accuracy. Other options are balanced classification accuracy `BalancedMisclassificationRate`, or `ImputationLoss()` (see [imputation hyperoptimisation](@ref imputation_hyper)).
- `maxiters=20`: The maximum number of solver iterations. Here we use a very small number for demonstration reasons.
- `logspace_eta=true`: A useful option that tells the tuning algorithm to sample the eta search space logarithmically.

There are many more customisation options for [`tune`](@ref), see the docstring and extended help for more information / advanced usecases.

### Evaluating model performance with evaluate()
If you want to estimate the performance of MPSTime on a dataset, you can call the [`evaluate`](@ref) function, which resamples your data into train/test splits using a provided resampling strategy (default is k-fold cross validation), tunes each split on the "training" set, and evaluates the test set. It can be called with the following syntax:

```julia-repl
julia> nfolds = 30;

julia> Xs = vcat(X_train, X_test);

julia> ys = vcat(y_train, y_test);

julia> results = evaluate(
    Xs,
    ys,
    nfolds,
    params,
    MPSRandomSearch(); 
    objective=MisclassificationRate(),
    tuning_maxiters=20
);

julia> results[1] # displays results for fold 1.

julia> losses = getindex.(res, "loss");

julia> using StatsBase # for the mean function

julia> println("The mean classification loss on the resampled noisy trendy sine data is: $(mean(losses))")

```

Which outputs a results dictionary, containing the losses on each resample fold, as well as a lot of other useful information. See the [`docstring`](@ref evaluate) for more detail as well as a plethora of customistation options.

A very common extension of `evaluate` is to customise the resampling strategy. The simplest way to do this is to pass a vector of `(training_indices, testing_indices)` to the `foldmethod` keyword. For example, to use scikit-learn's [`StratifiedShuffleSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html#sklearn.model_selection.StratifiedShuffleSplit) to generate the train/test splits:


```julia-repl
julia> using PyCall

julia> nfolds = 30

julia> py"""
from sklearn.model_selection import StratifiedShuffleSplit # requires a python environment with sklearn installed
sp = StratifiedShuffleSplit(n_splits=$nfolds, test_size=$(length(y_test)), random_state=1)
folds_py = sp.split($Xs, $ys)
"""
julia> folds = collect(py"folds_py")
30-element Vector{Any}:
 ([29, 85, 93, 56, 41, 59, 98, 24, 78, 96  …  35, 32, 77, 65, 23, 30, 91, 10, 19, 17], [9, 94, 5, 1, 67, 69, 95, 48, 63, 38  …  52, 60, 16, 57, 37, 12, 66, 11, 18, 15])
 ([26, 74, 86, 43, 78, 14, 76, 3, 22, 7  …  5, 71, 91, 66, 63, 64, 61, 34, 35, 89], [55, 40, 68, 73, 84, 59, 79, 58, 54, 15  …  75, 8, 32, 10, 81, 60, 90, 87, 44, 85])
[...]

julia> results = evaluate(
    Xs,
    ys,
    nfolds,
    params,
    MPSRandomSearch(); 
    objective=MisclassificationRate(),
    tuning_maxiters=20,
    foldmethod=folds
)
[...]
```


## [Hyperoptimising imputation](@id imputation_hyper)
The [`tune`](@ref) and [`evaluate`](@ref) methods may both be used to minimise imputation loss, with a small amount of extra setup. Setting `objective=ImputationLoss()` will optimise an MPS for imputation performance by minimising the mean absolute error between predicted and unseen data. To accomplish this, MPSTime takes data from the test (or validation) set, corrupts a portion of it, and then predicts what the corrupted data should be based on the uncorrupted values. There are two methods for how the test (or validation) data can be corrupted.
1) Setting the `windows` (or `eval_windows`) keyword in [`tune`](@ref) (or [`evaluate`](@ref), respectively) to a vector of 'windows'. Each window is a vector of missing/corrupted data indices, for example
```julia
windows = [[1,3,7],[4,5,6]]
```
will take each timeseries in the test set, and create two 'corrupted' test series, missing the 1st, 3rd, and 7th; and the 4th, 5th, and 6th values respectively.
2) Setting the `pms` (or `eval_pms`) keyword in [`tune`](@ref) (or [`evaluate`](@ref), respectively) to a vector of 'percentage missings'. This generates corrupted time series by removing randomly selected contiguous blocks that make up a specified percentage of the data. For example, 
```Julia
pms=[0.05, 0.05, 0.6, 0.95]
```
will generate four corrupted time series from each element of the test (or validation) set. Two will have missing blocks that make up 5% of their length, and one each will have blocks with 60% and 95% missing.
        
The imputation tuning loss is the average of computing the mean absolute error of imputing every window on every element of the test (or validation) set.

**Example: Calling tune with percentages missing**
Tune the MPS on an imputation problem with randomly selected 5%, 15%, 25%, ... , 95% long missing blocks.
```julia-repl
julia> nfolds = 5
best_params, cache = tune(
    X_train, 
    y_train, 
    nfolds,
    params,
    MPSRandomSearch(); 
    objective=ImputationLoss(), 
    pms=collect(0.05:0.1:0.95)
    maxiters=20,
    logspace_eta=true
)
```

**Example: Using evaluate with the Missing Completely At Random tool**
Tune the MPS on an imputation problem by completely randomly corrupting 5%, 15%, 25%, ... , or 95% of each test (or validation) time series. See the [`Missing Completely at Random`](@ref mcar) tool.

```julia-repl
julia> inds = collect(1:size(Xs,2));
julia> rng = Xoshiro(42);
julia> pms=0.05:0.1:0.95;
julia> mcar_windows = [mcar(inds, pm; rng=rng)[2] for pm in pms]
julia> results = evaluate(
    Xs,
    ys,
    nfolds,
    params,
    MPSRandomSearch(); 
    objective=ImputationLoss(),
    eval_windows=mcar_windows,
    tuning_maxiters=20
)

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
) -> Union{Float64, Vector{Float64}
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
The hyperparameter tuning algorithm used by [`tune`](@ref) (or [`evaluate`](@ref)) can be specified with the `optimiser` argument. This supports the default builtin [`MPSRandomSearch`](@ref) methods, as well as (in theory) any solver that is supported by the [`Optimization.jl interface`](https://docs.sciml.ai/Optimization/stable). Note that many of these solvers struggle with discrete search spaces, such as tuning the integer valued `chi_max` and `d`. Some of them require initial conditions (set `provide_x0=true`), and some require no initial conditions (set `provide_x0=false`), so your mileage may vary. By default, `tune()` handles optimisers attempting to evaluate discrete hyperparameters at a non-integer value by rounding,  and using its own cache to avoid rounding based cache misses. This is effective, but has the downside of causing `maxiters` to be inaccurate (as repeated hyperparameter evaluations caused by rounding result in a 'skipped' iteration).

```@docs
MPSRandomSearch
```


## [Distributed computing](@id distributed_computing)
Both tune [`tune`](@ref) and [`evaluate`](@ref) support several different parallel processing paradigms for different use cases, compatible with processors added via Distributed.jl's [`addprocs`](@extref Distributed.addprocs) function. 
For example, to distribute each fold of the classification style evalutation above, run:
```julia-repl
julia> using Distributed

julia> nfolds = 30;

julia> e = copy(ENV);

julia> e["OMP_NUM_THREADS"] = "1"; # attempt to prevent threading

julia> e["JULIA_NUM_THREADS"] = "1"; # attempt to prevent threading

julia> addprocs(nfolds; env=e);

julia> @everywhere using MPSTime

julia> Xs = vcat(X_train, X_test);

julia> ys = vcat(y_train, y_test);

julia> results = evaluate(
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