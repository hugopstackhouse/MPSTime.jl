using MPSTime
using JLD2
using Distributed
using Optimization
using OptimizationBBO
using Random
# using OptimizationMetaheuristics
using OptimizationOptimJL
# using OptimizationNLopt
# using OptimizationOptimisers

Random.seed!(1)
e = copy(ENV)
e["OMP_NUM_THREADS"] = "1"
e["JULIA_NUM_THREADS"] = "1"

if nprocs() == 1
    addprocs(30; env=e, exeflags="--heap-size-hint=2.4G", enable_threaded_blas=false)
end
@everywhere using MPSTime, Distributed, Optimization, OptimizationBBO, OptimizationOptimJL

@load "test/Data/ecg200/datasets/ecg200.jld2" X_train y_train X_test y_test

params = (
    eta=(-3,log10(0.5)), 
    d=(5,15), 
    chi_max=(20,40),
) 


rs_f = jldopen("Folds/ECG200/resample_folds_julia_idx.jld2", "r");
fold_idxs = read(rs_f, "rs_folds_julia");
close(rs_f)

@load "Folds/ECG200/windows_julia_idx.jld2" windows_julia
folds = [(fold_idxs[i-1]["train"], fold_idxs[i-1]["test"]) for i in 1:30]

Xs = vcat(X_train, X_test)
ys = zeros(Int, size(Xs, 1))
nfolds=30
res = evaluate(
    Xs,
    ys,
    nfolds,
    params,
    MPSRandomSearch(); 
    objective=ImputationLoss(), 
    opts0=MPSOptions(; verbosity=-5, log_level=-1, nsweeps=10, sigmoid_transform=false), 
    n_cvfolds=5,
    eval_windows=windows_julia,
    eval_pms = nothing,#collect(5:10:95) ./100,
    tuning_windows = nothing,
    tuning_pms=collect(5:10:95) ./100,
    tuning_abstol=1e-9, 
    tuning_maxiters=250,
    verbosity=1,
    foldmethod=folds,
    input_supertype=Float64,
    provide_x0=false,
    logspace_eta=true,
    distribute_folds=true,
    writedir="ECG_final",
    write=true)

@save "ECG_rand_50_ns_final.jld2" res
# 20 iter benchmarks 


# SA()
# t=697.29: training MPS with ((chi_max = 22, d = 2, eta = 1.7493084678516522))...  done
# t=697.97: Loss 0.17647058823529416

#t=2896, chim 30, d=2, eta=1.26
# 0.0303

# PSO()
# retcode: Default
# u: [20.456448963449063, 9.853361431316502, 3.6032277648473126]
# Final objective value:     0.02941176470588236

#  retcode: Default
# u: [20.5526603176487, 9.815468744903043, 3.5211517956867766]
# Final objective value:     0.02941176470588236


# BBO_adaptive_de_rand_1_bin()
# t=81.99: training MPS with ((chi_max = 29, d = 10, eta = 2.431916965383183))...  done
# t=88.85: Loss 0.02941176470588236

# u: [20.68733736874209, 7.282024263293762, 3.3227249898786315]
# Final objective value:     0.02941176470588236

#  retcode: MaxIters
# u: [21.794199781220755, 8.945887613181625, 0.8267693406424581]
# Final objective value:     0.030303030303030276
