# [Classification](@id Classification_top)

This tutorial for MPSTime will take you through the basic steps needed to fit an MPS to a time-series dataset.

## [Demo dataset](@id nts_demo)

First, import or generate your data. Here, we generate a two class "noisy trendy sine" dataset for the sake of demonstration, but if you have a dataset in mind, you can skip to the next section. Our demonstration dataset consists of a sine function with a randomised phase, plus a linear trend, plus some normally distributed noise. Each ``T``-length time series in class ``c`` at time ``t`` is given by:

```math
x^c_t = \sin{\left(\frac{2\pi}{\tau}t + \psi\right)} + \frac{mt}{T} + \sigma_c n_t\,,
```

where ``\tau`` is the period, ``m`` is the slope of a linear trend, ``\psi \in [0, 2\pi)`` is a uniformly random phase offset, ``\sigma_c`` is the noise scale, and ``n_t \sim \mathcal{N}(0,1)`` are  normally distributed random variables. 

For the demonstration dataset, the two classes will be generated with different distributions of periods. The class one time series ``x^1`` have ``\tau \in[12, 15]``, and the class two time series ``x^2`` will have``\tau \in[16, 19]``. We'll use ``\sigma_c = 0.2``, and the slope ``m`` will be randomly selected from ``\{-3,0,3\}``.


We'll set up this dataset using the [`trendy_sine`](@ref) function from MPSTime.
 
```jldoctest classification; output=false
using MPSTime, Random
rng = Xoshiro(1); # fix rng seed
ntimepoints = 100; # specify number of samples per instance
ntrain_instances = 600; # specify num training instances
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

# output

200-element Vector{Int64}:
 1
 1
 1
 1
 1
 1
 1
 1
 1
 1
 ⋮
 2
 2
 2
 2
 2
 2
 2
 2
 2

```

```@example classification
using Plots
p1 = plot(X_train[1:20,:]'; colour="blue", alpha=0.5, legend=:none);
p2 = plot(X_train[end-20:end,:]'; colour="blue", alpha=0.5, legend=:none);
plot(p1,p2)
```

## Training an MPS
For the most basic use of fitMPS, select your hyperparameters, and run the [`fitMPS`](@ref) function. 
Some (truncated) output from our noisy trendy sine datam with default hyperparameters is given below. 

```jldoctest classification; filter=[r"\[1\/10\](.*)MPS normalised"s => "[1/10]\n\n[...]\n\nMPS normalised"]
julia> opts = MPSOptions(); # calling this with no arguments gives default hyperparameters

julia> mps, info, test_states = fitMPS(X_train, y_train, X_test, y_test, opts);
Generating initial weight MPS with bond dimension χ_init = 4
        using random state 1234.
The test set couldn't be perfectly rescaled by the training set normalization, 5 additional rescaling operations had to be performed!
Initialising train states.
Initialising test states.
Using 1 iterations per update.
Training KL Div. 114.94918174702372 | Training acc. 0.5016666666666667.
Test KL Div. 116.97189034446042 | Testing acc. 0.54.

Test conf: [52 48; 44 56].
Using optimiser CustomGD with the "TSGO" algorithm
Starting backward sweeep: [1/10]
Backward sweep finished.
Starting forward sweep: [1/10]

[...]

MPS normalised!

Training KL Div. -46.44916444847569 | Training acc. 0.9883333333333333.
Test KL Div. -41.03275906157264 | Testing acc. 0.98.

Test conf: [99 1; 3 97].

```

[`fitMPS`](@ref) doesn't use `X_test` or `y_test` for anything except printing performance evaluations, so it is safe to leave them blank. For unsupervised learning, input a dataset with only one class, or only pass `X_train` ( `y_train` has a default value of `zeros(Int, size(X_train, 1))` ).

The `mps::TrainedMPS` can be passed directly to [`classify`](@ref) for classification, or [`init_imputation_problem`](@ref) to set up an imputation problem. `info` provides a short training summary, which can be pretty-printed with the [`sweep_summary`](@ref) function.

You can use also `test_states` to print a summary of the MPS performance on the test set.
```jldoctest classification
julia> get_training_summary(mps, test_states; print_stats=true);   

         Overlap Matrix
┌──────┬───────────┬───────────┐
│      │   |ψ1⟩    │   |ψ2⟩    │
├──────┼───────────┼───────────┤
│ ⟨ψ1| │ 1.000e+00 │ 9.391e-03 │
├──────┼───────────┼───────────┤
│ ⟨ψ2| │ 9.391e-03 │ 1.000e+00 │
└──────┴───────────┴───────────┘
          Confusion Matrix
┌──────────┬───────────┬───────────┐
│          │ Pred. |1⟩ │ Pred. |2⟩ │
├──────────┼───────────┼───────────┤
│ True |1⟩ │        99 │         1 │
├──────────┼───────────┼───────────┤
│ True |2⟩ │         3 │        97 │
└──────────┴───────────┴───────────┘
┌───────────────────┬───────────┬──────────┬──────────┬─────────────┬─────────┬───────────┐
│ test_balanced_acc │ train_acc │ test_acc │ f1_score │ specificity │  recall │ precision │
│           Float64 │   Float64 │  Float64 │  Float64 │     Float64 │ Float64 │   Float64 │
├───────────────────┼───────────┼──────────┼──────────┼─────────────┼─────────┼───────────┤
│              0.98 │  0.988333 │     0.98 │ 0.979998 │        0.98 │    0.98 │  0.980192 │
└───────────────────┴───────────┴──────────┴──────────┴─────────────┴─────────┴───────────┘

```

## Hyperparameters

There are number of hyperparameters and data preprocessing options that can be specified using `MPSOptions(; key=value)`


```@docs
MPSOptions
```

You can also print a formatted table of options with [`print_opts`](@ref) (beware long output)

```jldoctest classification
julia> print_opts(opts; long=false);
┌─────────┬──────────────────┬─────────┬─────────┬───────────────────┬───────────┬───────┐
│ nsweeps │         encoding │     eta │ chi_max │ sigmoid_transform │ loss_grad │     d │
│   Int64 │           Symbol │ Float64 │   Int64 │              Bool │    Symbol │ Int64 │
├─────────┼──────────────────┼─────────┼─────────┼───────────────────┼───────────┼───────┤
│      10 │ Legendre_No_Norm │    0.01 │      25 │              true │       KLD │     5 │
└─────────┴──────────────────┴─────────┴─────────┴───────────────────┴───────────┴───────┘

```

## Classification
To predict the class of unseen data, use the [`classify`](@ref) function.

```@docs
classify(::TrainedMPS, ::AbstractMatrix)
```

For example, for the noisy trendy sine from earlier:
```jldoctest classification
julia> predictions = classify(mps, X_test);

julia> using StatsBase

julia> mean(predictions .== y_test)
0.98
```

## Training with a custom basis
To train with a custom basis, first, declare a custom basis with [`function_basis`](@ref), and pass it in as the last argument to [`fitMPS`](@ref). For this to work, the encoding hyperparameter must be set to `:Custom` in `MPSOptions`

```jldoctest classification; filter=[r"random state 1234(.*)"s => "\n\n[...]"], setup=:(X_train=X_train[1:3,1:5]; y_train=y_train[1:3])
using LegendrePolynomials
function legendre_encode(x::Float64, d::Int)
    # default legendre encoding: choose the first n-1 legendre polynomials

    leg_basis = [Pl(x, i; norm = Val(:normalized)) for i in 0:(d-1)] 
    
    return leg_basis
end
custom_basis = function_basis(legendre_encode, false, (-1., 1.))
fitMPS(X_train, y_train, X_test, y_test, MPSOptions(; encoding=:Custom), custom_basis)
# output
Generating initial weight MPS with bond dimension χ_init = 4
        using random state 1234.

[...]
```

## Docstrings

```@docs
fitMPS(::Matrix, ::Vector, ::Matrix, ::Vector, ::MPSOptions, ::Nothing)
sweep_summary(info)
get_training_summary(mps::TrainedMPS, test_states::EncodedTimeSeriesSet)
print_opts
```
