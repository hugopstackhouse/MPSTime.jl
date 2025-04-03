# using Revise
using MPSTime
using JLD2
# using Distributed
using Optimization
using OptimizationBBO
using Random
# using OptimizationMetaheuristics
# using OptimizationOptimJL
# using OptimizationNLopt
# using OptimizationOptimisers

fold = parse(Int, ARGS[1])

@assert ENV["OMP_NUM_THREADS"] == "1" "Disable MKL Threading in the job array!"

Random.seed!(1)

params = (
    eta=(-3,log10(0.5)), 
    d=(5, 15), 
    chi_max=(20, 40),
) 
nfolds = 30


rs_f = jldopen("Folds/Kepler/c4_folds_flat.jld2", "r");
folds = read(rs_f, "folds");
Xs = read(rs_f, "Xs");
close(rs_f)
ys = ones(Int, size(Xs, 1))

@load "Folds/Kepler/kepler_windows_julia_idx.jld2" windows_per_percentage
windows_julia = windows_per_percentage
res = evaluate(
    Xs,
    ys,
    nfolds,
    params,
    MPSRandomSearch(); 
    objective=ImputationLoss(), 
    opts0=MPSOptions(; verbosity=-5, log_level=-1, nsweeps=10), 
    n_cvfolds=5,
    fold_inds=[fold],
    eval_windows=windows_julia,
    eval_pms=nothing,#collect(5:20:95) ./100,
    tuning_windows = nothing,
    tuning_pms=collect(5:10:95) ./100,
    tuning_abstol=1e-9, 
    tuning_maxiters=250,
    verbosity=1,
    foldmethod=folds,
    input_supertype=Float64,
    provide_x0=false,
    logspace_eta=true,
    distribute_folds=false,
    distribute_cvfolds=false,
    writedir="KC4_evals_final",
    write=true,
    collect_tmps=false
)

# 0.20072699080538697
# 0.22382542363624128
# 0.1972986806310512
# @save "IPD_rand_50_full.jld2" res
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
