using MPSTime
using JLD2
# using Optimization
# using OptimizationBBO
# using OptimizationMetaheuristics
using OptimizationOptimJL
# using OptimizationNLopt
# using StatsBase
using DelimitedFiles
using Distributed



# load in the published data
train_f_ecg = "Folds/ECG200/ECG200_TRAIN.txt"
test_f_ecg = "Folds/ECG200/ECG200_TEST.txt"
ecg_X_train_original = readdlm(train_f_ecg);
ecg_X_test_original = readdlm(test_f_ecg);
ecg_X_train = ecg_X_train_original[:, 2:end]
ecg_X_test = ecg_X_test_original[:, 2:end]
ecg_label_remap = Dict(-1 => 0, 1 => 1)
ecg_y_train = [ecg_label_remap[label] for label in Int.(ecg_X_train_original[:, 1])] # remap labels to MPS labels 0 and 1
ecg_y_test = [ecg_label_remap[label] for label in Int.(ecg_X_test_original[:, 1])];
# sanity checks
@assert size(ecg_X_train, 1) == 100
@assert size(ecg_y_train, 1) == 100
@assert size(ecg_X_test, 1) == 100
@assert size(ecg_y_test, 1) == 100

# recombine the original datasets
ecg_X = vcat(ecg_X_train, ecg_X_test);
ecg_y = vcat(ecg_y_train, ecg_y_test);

JLD2.@load "bakeoff_folds/ecg200_bakeoff_folds.jld2"

folds = [(fold_idxs_ecg[k]["train"], fold_idxs_ecg[k]["test"]) for k in keys(fold_idxs_ecg)]
@assert length(folds) == 30
nfolds = length(folds)

params = (
    eta=(1e-3,0.5), 
    d=(2, 15), 
    chi_max=(20, 40)
) 
# fold_idx = parse(Int, ARGS[1])
# fold_map = [1,2,5,6,9,10,11,12,15,17,19,25,26,27,29]
# fold= fold_map[fold_idx]
# fold = fold_idx

e = copy(ENV)
e["OMP_NUM_THREADS"] = "1"
e["JULIA_NUM_THREADS"] = "1"

if nprocs() == 1
    addprocs(30; env=e, exeflags="--heap-size-hint=2.4G", enable_threaded_blas=false)
end

@everywhere using MPSTime, Distributed, Optimization, OptimizationOptimJL

res = MPSTime.evaluate(
    ecg_X, 
    ecg_y,
    nfolds,
    params,
    MPSRandomSearch(); 
    # fold_inds = [fold],
    objective=MisclassificationRate(), 
    opts0=MPSOptions(; d=5, chi_max=30, verbosity=-5, log_level=-1, sigmoid_transform=true, nsweeps=10),
    tuning_abstol=1e-5, 
    tuning_maxiters=0,
    tuning_windows=nothing,
    verbosity=2,
    logspace_eta=true,
    pms=nothing,
    foldmethod=folds,
    input_supertype=Float64,
    distribute_folds=true, 
    distribute_cvfolds=false,
    write=true,
    writedir="ECG_classification_timing",
    provide_x0=true)

