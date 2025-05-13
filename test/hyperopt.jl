using JLD2
using Distributed
using Statistics


@load "Data/italypower/datasets/ItalyPowerDemandOrig.jld2" X_train y_train X_test y_test

params = (
    eta=(1e-5,1), 
    d=(2,8), 
    chi_max=(15,30),
    nsweeps=(1,10)
,) 
nfolds = 5


e = copy(ENV)
e["OMP_NUM_THREADS"] = "1"
e["JULIA_NUM_THREADS"] = "1"

if nprocs() == 1
    addprocs(nfolds; env=e, exeflags="--heap-size-hint=2G", enable_threaded_blas=false)
end

@everywhere using MPSTime, Distributed


res, cache = tune(
    X_train, 
    y_train, 
    nfolds,
    params,
    MPSRandomSearch(),
    objective=ImputationLoss(), 
    opts0=MPSOptions(; verbosity=-5, log_level=-1, nsweeps=5, sigmoid_transform=false), 
    pms=[0.05, 0.9],
    abstol=1e-3, 
    maxiters=3,
    verbosity=-1,
    logspace_eta=true,
    distribute_folds=true
)

@test res.chi_max == 30 && res.d == 8 && isapprox(res.eta, 0.0031622776601683794) && res.nsweeps == 10
# @test isapprox(cache[values(res)], 0.13815086627287926) #TODO integer/precision arithmetic operations?