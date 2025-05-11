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

# load the resample ind ices
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

function run_folds(Xs::Matrix{Float64}, ys::Vector{Int64}, window_idxs::Dict,
        fold_idxs::Dict)

        setprecision(BigFloat, 128)
        Rdtype = Float64

        # training related stuff
        verbosity = -10
        test_run = false
        track_cost = false
        encoding = :legendre_no_norm
        encode_classes_separately = false
        train_classes_separately = false

        d = 10 #10
        chi_max=20 #20
        nsweeps = 5 #5
        eta = 0.5
        opts=MPSOptions(; nsweeps=nsweeps, chi_max=chi_max,  update_iters=1, verbosity=verbosity, loss_grad=:KLD,
            bbopt=:TSGO, track_cost=track_cost, eta=eta, rescale = (false, true), d=d, aux_basis_dim=2, encoding=encoding, 
            encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, 
            exit_early=false, sigmoid_transform=false, init_rng=4567, chi_init=4, log_level=0)
        opts_safe, _... = safe_options(opts, nothing, nothing)

        num_folds = length(fold_idxs)
        dx = 5e-3
        mode_range=(-1,1)
        xvals=collect(range(mode_range...; step=dx))
        mode_index=Index(opts_safe.d)
        pms = 5:10:95
        xvals_enc= [get_state(x, opts_safe) for x in xvals]
        xvals_enc_it=[ITensor(s, mode_index) for s in xvals_enc];

        # main loop
        fold_scores = Vector{FoldResults}(undef, num_folds)
        for fold_idx in 0:(num_folds-1)
            fold_time = @elapsed begin
                fold_train_idxs = fold_idxs[fold_idx]["train"]
                fold_test_idxs = fold_idxs[fold_idx]["test"]
                X_train_fold = Xs[fold_train_idxs, :]
                y_train_fold = ys[fold_train_idxs]
                X_test_fold = Xs[fold_test_idxs, :]
                y_test_fold = ys[fold_test_idxs]

                W, _, _, _ = fitMPS(X_train_fold, y_train_fold, X_test_fold, y_test_fold; chi_init=4, opts=opts, test_run=false)
                fc = load_forecasting_info_variables(W, X_train_fold, y_train_fold, X_test_fold, y_test_fold, opts_safe; verbosity=0)

                println("Finished training, beginning evaluation of imputed values...")
                samps_per_class = [size(f.test_samples, 1) for f in fc]
                all_instances = Vector{Vector{InstanceScores}}(undef, 2)
                for (i, s) in enumerate(samps_per_class)
                    # each class instances
                    per_class_instances = Vector{InstanceScores}(undef, s)
                    for inst in 1:s
                        println("Evaluating class $i, instance $inst")
                        # loop over windows
                        pm_scores = Vector{WindowScores}(undef, length(pms))
                        for (ipm, pm) in enumerate(pms)
                            # loop over window iterations
                            num_wins = length(window_idxs[pm])
                            mps_scores = Vector{Float64}(undef, num_wins)
                            nn_scores = Vector{Float64}(undef, num_wins)
                            @threads for it in 1:num_wins
                                interp_sites = window_idxs[pm][it]
                                stats, _ = any_impute_single_timeseries(fc, (i-1), inst, interp_sites, :directMedian; invert_transform=true, 
                                    NN_baseline=true, X_train=X_train_fold, y_train=y_train_fold, 
                                    n_baselines=1, plot_fits=false, dx=dx, mode_range=mode_range, xvals=xvals, 
                                    mode_index=mode_index, xvals_enc=xvals_enc, xvals_enc_it=xvals_enc_it)
                                    mps_scores[it] = stats[:MAE]
                                    nn_scores[it] = stats[:NN_MAE]
                            end
                            pm_scores[ipm] = WindowScores(mps_scores, nn_scores)
                        end
                        #instance_scores = InstanceScores(pm_scores)
                        per_class_instances[inst] = InstanceScores(pm_scores)
                        #push!(all_instances, instance_scores)
                    end
                    all_instances[i] = per_class_instances
                end
                all_instances = vcat(all_instances[1], all_instances[2])
                fold_scores[fold_idx+1] = FoldResults(all_instances)
            end
            println("Fold $fold_idx took $fold_time seconds.")
        end

        return fold_scores, opts_safe
end

results, opts_safe = run_folds(Xs, ys, window_idxs, rs_fold_idxs)

JLD2.@save "ipd_30_fold_imputation_results_mac5sweep.jld2" results opts_safe

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

[mean([mean([mean(mps_results[pm][f][inst]) for inst in 1:1029]) for f in 1:30]) for pm in 1:10]
[mean([mean([mean(nn_results[pm][f][inst]) for inst in 1:1029]) for f in 1:2]) for pm in 1:10]

jldopen("ipd_30fold_imputation_results_mac5sweep.jld2", "w") do f
    f["mps_results"] = mps_results
    f["nn_results"] = nn_results
    f["opts"] = opts_safe
end

#[mps_results[5][1][inst] for inst in 1:100] # pm/fold/inst
mps_per_pm_30fold = [mean([mean([mean(mps_results[pm][f][inst]) for inst in 1:100]) for f in 1:30]) for pm in 1:10]
nn_per_pm_30fold = [mean([mean([mean(nn_results[pm][f][inst]) for inst in 1:100]) for f in 1:30]) for pm in 1:10]

mps_per_pm_30fold_std_err = 1.96 * [std([mean([mean(mps_results[pm][f][inst]) for inst in 1:100]) for f in 1:30]) for pm in 1:10]/sqrt(30)
nn_per_pm_30fold_std_err = 1.96 * [std([mean([mean(nn_results[pm][f][inst]) for inst in 1:100]) for f in 1:30]) for pm in 1:10]/sqrt(30)

    #mps_results[pm] = 
#t = [results[1].fold_scores[inst].pm_scores[5].mps_scores for inst in 1:100]


#old_means_5pt_mps = mean([mean([mean(results[fold].fold_scores[inst].pm_scores[5].mps_scores) for inst in 1:100]) for fold in 1:30])
#fold_std_err_5pt_mps = 1.96 * std([mean([mean(results[fold].fold_scores[inst].pm_scores[65].mps_scores) for inst in 1:100]) for fold in 1:30])/sqrt(30)

# fold_means_5pt_nn = mean([mean([mean(results[fold].fold_scores[inst].pm_scores[5].nn_scores) for inst in 1:100]) for fold in 1:30])
# fold_std_err_5pt_nn = 1.96 * std([mean([mean(results[fold].fold_scores[inst].pm_scores[5].nn_scores) for inst in 1:100]) for fold in 1:30])/sqrt(30)

# mps_all_pm = [mean([mean([mean(results[fold].fold_scores[inst].pm_scores[pm].mps_scores) for inst in 1:100]) for fold in 1:30]) for pm in 5:10:95]
# nn_all_pm = [mean([mean([mean(results[fold].fold_scores[inst].pm_scores[pm].nn_scores) for inst in 1:100]) for fold in 1:30]) for pm in 5:10:95]

groupedbar([mps_per_pm_30fold nn_per_pm_30fold],
    yerr=[mps_per_pm_30fold_std_err nn_per_pm_30fold_std_err],
    label=["MPS" "NN"]);
xflip!(true)