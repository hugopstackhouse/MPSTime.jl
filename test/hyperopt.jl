using JLD2
using Distributed
using Optimization
using OptimizationBBO
using OptimizationMetaheuristics
using OptimizationOptimJL
using OptimizationNLopt
# using OptimizationOptimisers

@load "test/Data/italypower/datasets/ItalyPowerDemandOrig.jld2" X_train y_train X_test y_test

params = (
    eta=(1e-5,4), 
    d=(2,15), 
    chi_max=(15,45),
    nsweeps=(1,10)
,) 
nfolds = 5


res = tune(
    X_train, 
    y_train, 
    nfolds,
    params,
    SAMIN(); 
    objective=ImputationLoss(), 
    opts0=MPSOptions(; verbosity=-5, log_level=-1, nsweeps=5), 
    pms=[0.05, 0.9],
    abstol=1e-3, 
    maxiters=3,
    verbosity=2,
)


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