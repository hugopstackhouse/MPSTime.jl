using JLD2
using Distributed
using Random
using Statistics


Random.seed!(1)
@load "Data/italypower/datasets/ItalyPowerDemandOrig.jld2" X_train y_train X_test y_test

params = (
    eta=logrange(10^-3,0.5; length=10), 
    d=[2,5,7,8,15],#5,15), 
    chi_max=(5,5,10)#20,40),
) 
nfolds = 5

e = copy(ENV)
e["OMP_NUM_THREADS"] = "1"
e["JULIA_NUM_THREADS"] = "1"

if nprocs() == 1
    addprocs(nfolds; env=e, exeflags="--heap-size-hint=2G", enable_threaded_blas=false)
end

@everywhere using MPSTime, Distributed


Xs = vcat(X_train, X_test)
ys = vcat(y_train, y_test)
res = evaluate(
    Xs,
    ys,
    nfolds,
    params,
    MPSRandomSearch(); 
    objective=MisclassificationRate(), 
    opts0=MPSOptions(; verbosity=-5, log_level=-1, nsweeps=10, sigmoid_transform=true), 
    n_cvfolds=2,
    tuning_maxiters=5,
    verbosity=-1,
    input_supertype=Float64,
    provide_x0=false,
    distribute_folds=true,
    distribute_cvfolds=false,
    distribute_iters=false,
    distribute_final_eval=false,
    write=false,
    writedir="IPD_final"
)

@load "Data/eval_results.jld2" res_baseline

for i in eachindex(res)
    # opts
    local opts = res[i]["opts"] 
    local opts_bl = res_baseline[i]["opts"]
    # @test opts.d == opts_bl.d && opts.chi_max == opts_bl.chi_max && isapprox(opts.eta, opts_bl.eta) #TODO integer/precision arithmetic operations?

    @test res[i]["train_inds"] == res_baseline[i]["train_inds"] && res[i]["test_inds"] == res_baseline[i]["test_inds"]
end

mean_loss = mean(getindex.(res, "loss"))[1]
ml_baseline = mean(getindex.(res_baseline, "loss"))[1]

# @test isapprox(mean_loss, ml_baseline) #TODO integer/precision arithmetic operations?