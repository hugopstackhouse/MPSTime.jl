import Pkg
Pkg.activate(".")
using MPSTime
using JLD2
using Distributed
using Optimization
using OptimizationBBO
using Random
# using OptimizationMetaheuristics
# using OptimizationOptimJL
# using OptimizationNLopt
# using OptimizationOptimisers


which_nts = 2
nts_loaded = jldopen("Folds/NTS/NTS$(which_nts)_dataset.jld2")
    X_train = read(nts_loaded, "X_train")
    train_info = read(nts_loaded, "train_info")
    y_train = read(nts_loaded, "y_train")
    X_test = read(nts_loaded, "X_test")
    test_info = read(nts_loaded, "test_info")
    y_test = read(nts_loaded, "y_test")
    eval_windows = read(nts_loaded, "eval_windows")
close(nts_loaded)

chi_lower = 20
chi_upper = 160
params = (
    eta=logrange(10^-3, 0.5; length=10), 
    d=(5,15), 
    chi_max=(chi_lower, 20, chi_upper)) # adjust as needed 
    
folds = [(collect(1:1:size(X_train, 1)), collect(size(X_train, 1)+1:1:size(X_train, 1)+size(X_test, 1)))]
nfolds = length(folds)

e = copy(ENV)
e["OMP_NUM_THREADS"] = "1"
e["JULIA_NUM_THREADS"] = "1"

if nprocs() == 1
    addprocs(109; env=e, exeflags="--heap-size-hint=2G", enable_threaded_blas=false)
end

@everywhere using MPSTime, Distributed, Optimization, OptimizationBBO

max_iters = 250
sweeps = 10

res = evaluate(
    vcat(X_train, X_test), 
    vcat(y_train, y_test), 
    nfolds,
    params,
    MPSRandomSearch(); 
    objective=ImputationLoss(), 
    opts0=MPSOptions(; verbosity=-5, log_level=-1, nsweeps=sweeps, sigmoid_transform=false), 
    n_cvfolds=5,
    eval_windows=eval_windows,
    tuning_windows=nothing,
    tuning_pms=collect(5:10:95) ./100,
    tuning_abstol=1e-8, 
    tuning_maxiters=max_iters,
    verbosity=2,
    foldmethod=folds,
    input_supertype=Float64,
    provide_x0=false,
    distribute_iters=true)

@save "NTS$(which_nts)_rand_$(max_iters)_$(sweeps)sw_chi_$(chi_lower)_$(chi_upper).jld2" res
