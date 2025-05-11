using Pkg
Pkg.activate(".")
include("../../../LogLoss/RealRealHighDimension.jl")
include("../../../Interpolation/imputation.jl");
using JLD2
using DelimitedFiles
using Plots

# load the original ECG200 split
dloc = "Data/italypower/datasets/ItalyPowerDemandOrig.jld2"
f = jldopen(dloc, "r")
    X_train = read(f, "X_train")
    y_train = read(f, "y_train")
    X_test = read(f, "X_test")
    y_test = read(f, "y_test")
close(f)

# recombine the original train/test splits
Xs = vcat(X_train, X_test)
ys = vcat(y_train, y_test)

# load the resample indices
rs_f = jldopen("FinalBenchmarks/ItalyPower/Julia/ipd_resample_folds_julia_idx.jld2", "r");
rs_fold_idxs = read(rs_f, "rs_folds_julia");
close(rs_f)

# load the window indices
windows_f = jldopen("FinalBenchmarks/ItalyPower/Julia/ipd_windows_julia_idx.jld2", "r");
window_idxs = read(windows_f, "windows_julia")
close(windows_f)

# define structs for the results
struct WindowScores
    mps_scores::Vector{Float64}
    nn_scores::Vector{Float64}
end

struct InstanceScores
    pm_scores::Vector{WindowScores}
end

struct FoldResults 
    fold_scores::Vector{InstanceScores}
end

function run_folds(Xs::Matrix{Float64}, window_idxs::Dict, fold_idxs::Dict, which_folds::UnitRange=0:29)
    
    # training related parameters
    Rdtype = Float64
    verbosity = -10
    test_run = false
    track_cost = false
    encoding = :legendre_no_norm
    encode_classes_separately = false
    train_classes_separately = false

    d = 10
    chi_max = 20 # 
    eta = 0.5 # 0.1
    nsweeps = 5 # 3
    opts=MPSOptions(; nsweeps=nsweeps, chi_max=chi_max, update_iters=1, verbosity=verbosity, loss_grad=:KLD,
            bbopt=:TSGO, track_cost=track_cost, eta=eta, rescale = (false, true), d=d, aux_basis_dim=2, encoding=encoding, 
            encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, 
            exit_early=false, sigmoid_transform=false, init_rng=4567, chi_init=4, log_level=0)
    opts_safe, _... = safe_options(opts, nothing, nothing)

    dx = 5e-3
    mode_range=(-1,1)
    xvals=collect(range(mode_range...; step=dx))
    mode_index=Index(opts_safe.d)
    pms = 5:10:95
    xvals_enc= [get_state(x, opts_safe) for x in xvals]
    xvals_enc_it=[ITensor(s, mode_index) for s in xvals_enc];

    num_folds = length(which_folds) # specified by the user
    if num_folds > length(fold_idxs)
        error("Fold range specified must be samller than the max number of folds ($length(fold_idxs)).")
    end
    # main loop
    fold_scores = Vector{FoldResults}(undef, num_folds)
    for (i, fold_idx) in enumerate(which_folds)
        # imputation related parameters
        dx = 5e-3
        mode_range = (-1, 1)
        xvals = collect(range(mode_range...; step=dx))
        mode_index = Index(opts_safe.d)
        pms = 5:10:95 
        xvals_enc = [get_state(x, opts_safe) for x in xvals] # encode the xvals form mode imputation 
        xvals_enc_it = [ITensor(s, mode_index) for s in xvals_enc] # convert encoded vals to ITensors for mode imputation
        fold_time = @elapsed begin
            fold_train_idxs = fold_idxs[fold_idx]["train"]
            fold_test_idxs = fold_idxs[fold_idx]["test"]
            X_train_fold = Xs[fold_train_idxs, :]
            X_test_fold = Xs[fold_test_idxs, :]
            # all of the instances go into the same class
            y_train_fold = zeros(Int64, size(X_train_fold, 1))
            y_test_fold = zeros(Int64, size(X_test_fold, 1))
            println(size(X_train_fold))
            println(size(y_train_fold))
            println(size(X_test_fold))
            println(size(y_test_fold))
            W, _, _, _ = fitMPS(X_train_fold, y_train_fold, X_test_fold, y_test_fold, chi_init=4, opts=opts, test_run=false)
            # begin imputation
            fc = load_forecasting_info_variables(W, X_train_fold, y_train_fold, X_test_fold, y_test_fold, opts_safe; verbosity=0)
            println("Finished training, beginning evaluation of imputed values...")
            num_instances = size(X_test_fold, 1)
            mps_scores = zeros(Float64, num_instances)
            nn_scores = zeros(Float64, num_instances)
            instance_scores = Vector{InstanceScores}(undef, num_instances)
            for instance in 1:num_instances
                println("Evaluating instance $instance")
                # loop over windows
                pm_scores = Vector{WindowScores}(undef, length(pms))
                for (ipm, pm) in enumerate(pms)
                    # loop over window iterations
                    num_wins = length(window_idxs[pm])
                    mps_scores = Vector{Float64}(undef, num_wins)
                    nn_scores = Vector{Float64}(undef, num_wins)
                    @threads for it in 1:num_wins
                        interp_sites = window_idxs[pm][it]
                        stats, _ = any_impute_single_timeseries(fc, 0, instance, interp_sites, :directMedian; 
                            invert_transform=true, 
                                    NN_baseline=true, X_train=X_train_fold, y_train=y_train_fold, 
                                    n_baselines=1, plot_fits=false, dx=dx, mode_range=mode_range, xvals=xvals, 
                                    mode_index=mode_index, xvals_enc=xvals_enc, xvals_enc_it=xvals_enc_it)
                        mps_scores[it] = stats[:MAE]
                        nn_scores[it] = stats[:NN_MAE]
                    end
                    pm_scores[ipm] = WindowScores(mps_scores, nn_scores)
                end
                instance_scores[instance] = InstanceScores(pm_scores)
            end
            fold_scores[i] = FoldResults(instance_scores)
        end
        println("Fold $fold_idx took $fold_time seconds.")
    end
    return fold_scores, opts_safe
end

results, opts_safe = run_folds(Xs, window_idxs, rs_fold_idxs, 0:29)

mps_results = Dict()
nn_results = Dict()

for pm in 1:10
    per_pm_res_mps = Dict()
    per_pm_res_nn = Dict()
    for f in 1:30
        total_instances = length(results[f].fold_scores)
        per_pm_res_mps[f] = [results[f].fold_scores[inst].pm_scores[pm].mps_scores for inst in 1:total_instances]
        per_pm_res_nn[f] = [results[f].fold_scores[inst].pm_scores[pm].nn_scores for inst in 1:total_instances]
    end
    mps_results[pm] = per_pm_res_mps
    nn_results[pm] = per_pm_res_nn
end
mps_results
nn_results

JLD2.@save "IPD_ImputationFinalResults_30Fold.jld2" mps_results nn_results opts_safe


mpsres = [mean([mean([mean(mps_results[pm][f][inst]) for inst in 1:1029]) for f in 1:30]) for pm in 1:10]
nnres = [mean([mean([mean(nn_results[pm][f][inst]) for inst in 1:1029]) for f in 1:30]) for pm in 1:10]
