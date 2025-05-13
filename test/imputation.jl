using JLD2
using Random

# ecg200 dataset
@load "Data/ecg200/datasets/ecg200.jld2" X_train y_train X_test y_test

# opts = MPSOptions(verbosity=3, log_level=0, dtype=BigFloat, nsweeps=3)
# setprecision(60)
# mps, info, test_states = fitMPS(X_train, y_train, X_test, y_test, opts)
@load "Data/ecg200/mps_saves/60_prec_ecg.jld2" mps info test_states


imp = init_imputation_problem(mps, X_test, y_test; verbosity=-10);
imp_rtol = 0.0001;

# median test
class = 1
pm = 0.8 # 80% missing data
instance_idx = 20 # time series instance in test set
_, impute_sites_pm80 = mar(X_test[instance_idx, :], pm; rng=Xoshiro(123)) # simulate MAR mechanism
method = :median

_,_,_, stats_pm80, plots_pm80 = MPS_impute(
    imp,
    class, 
    instance_idx, 
    impute_sites_pm80, 
    method; 
    NN_baseline=true, # whether to also do a baseline imputation using 1-NN
    plot_fits=true, # whether to plot the fits
)

# we don't want to be _that_ precise here becase the fperror can really add up depending on what architecture this is running on
@test isapprox(stats_pm80[1][:MAPE], 0.3830051169633825; rtol=imp_rtol)
@test isapprox(stats_pm80[1][:NN_MAPE], 0.5319691385738694; rtol=imp_rtol)

pm = 0.2 # a quick version

rng = Xoshiro(1)
imp_methods = [:median, :mean, :mode, :ITS, :kNearestNeighbour]

# ecg200 has two class, 0 and 1
nc1s = sum(y_test)
ncs = [length(y_test) - nc1s, nc1s]

expected_maes = [
    0.36581457566749176 0.2204382247157053;
    0.17783806685365627 0.1876692173707628;
    0.33536170154856404 0.3271278413457546;
    0.7649367656713248 0.8727296222420307;
    0.3877101919863158 0.2120027330331579;
]
# maes BigFloat-256
# 5×2 Matrix{Float64}:
#  0.347927  0.21203
#  0.17893   0.196088
#  0.35399   0.351822
#  0.738057  0.857018
#  0.38771   0.212003

# cluster version
# maes BigFloat-256
# 5×2 Matrix{Float64}:
#  0.347927  0.21203
#  0.17893   0.196088
#  0.35399   0.351822
#  0.738057  0.857018
#  0.38771   0.212003

maes = Matrix{Float64}(undef, size(expected_maes)...)
for (i, method) in enumerate(imp_methods)
    # println("method = $(string(method))")
   
    for (ci, class) in enumerate([0,1])
        # println("class = $class")
        if method == :mean && class == 1
            # println("Expecting exactly one warning:")
        end
        ns = ncs[ci]
        idxs = randperm(rng, ns)[1:10]
        mae = 0.
        for instance_idx in idxs
            # println("idx=$(instance_idx)")
            _, impute_sites_pm20 = mar(X_test[instance_idx, :], pm; rng=rng) 
            _, _, _, stats_pm20, plots_pm20 = MPS_impute(
                imp,
                class, 
                instance_idx, 
                impute_sites_pm20, 
                method; 
                NN_baseline=false, # whether to also do a baseline imputation using 1-NNI
                plot_fits=false, # whether to plot the fits
            )
            mae += stats_pm20[1][:MAE]
        end
        #@show mae/length(idxs)
        maes[i,ci] = mae / length(idxs)
        @test isapprox(expected_maes[i, ci], maes[i,ci]; rtol=imp_rtol)
    end
end
