# using Revise
using MPSTime
using JLD2
using Distributed
# using Optimization
# using OptimizationBBO
using Random
# using OptimizationMetaheuristics
# using OptimizationOptimJL
# using OptimizationNLopt
# using OptimizationOptimisers

Random.seed!(1)
@load "test/Data/italypower/datasets/ItalyPowerDemandOrig.jld2" X_train y_train X_test y_test

params = (
    eta=(-3,log10(0.5)), 
    d=(5,15), 
    chi_max=(20,40),
) 
e = copy(ENV)
e["OMP_NUM_THREADS"] = "1"
e["JULIA_NUM_THREADS"] = "1"

if nprocs() == 1
    addprocs(10; env=e, exeflags="--heap-size-hint=2G", enable_threaded_blas=false)
end

@everywhere using MPSTime, Distributed, Optimization

rs_f = jldopen("Folds/IPD/ipd_resample_folds_julia_idx.jld2", "r");
fold_idxs = read(rs_f, "rs_folds_julia");
close(rs_f)

@load "Folds/IPD/ipd_windows_julia_idx.jld2" windows_julia
folds = [(fold_idxs[i-1]["train"], fold_idxs[i-1]["test"]) for i in 1:30]

Xs = vcat(X_train, X_test)
ys = zeros(Int, size(Xs, 1))
res = evaluate(
    Xs,
    ys,
    2,
    params,
    MPSRandomSearch(); 
    objective=ImputationLoss(), 
    opts0=MPSOptions(; verbosity=-5, log_level=-1, nsweeps=10, sigmoid_transform=false), 
    n_cvfolds=5,
    eval_windows=windows_julia,
    eval_pms=nothing,#collect(5:20:95) ./100,
    tuning_windows = nothing,
    tuning_pms=collect(5:10:95) ./100,
    tuning_abstol=1e-9, 
    tuning_maxiters=14,
    verbosity=2,
    foldmethod=folds,
    input_supertype=Float64,
    provide_x0=false,
    logspace_eta=true,
    distribute_folds=true,
    distribute_cvfolds=false,
    distribute_iters=true,
    distribute_final_eval=true,
    write=false,
    writedir="IPD_final"
)
