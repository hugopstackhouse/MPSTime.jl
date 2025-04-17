import json 
import numpy as np
import matplotlib.pyplot as plt
from pypots.optim import Adam
from pypots.imputation import CSDI, BRITS
from pypots.utils.random import set_random_seed
from pypots.utils.metrics import calc_mae
import pickle
import sys
from time import time
set_random_seed(1234)
# check that GPU acceleration is enabled
import torch
torch.cuda.device_count()
# print(f"GPU: {torch.cuda.get_device_name()}")
print(f"CUDA ENABLED: {torch.cuda.is_available()}")


def evaluate_folds(model, nfolds, Xs, folds, windows_by_pm):
    # make the splits
    errs = {}
    n_tr_samples = len(folds[0][0])
    nsamples =  len(folds[1][0])

    for pm, windows in windows_by_pm.items():
        errs[pm] = -np.ones((nfolds, len(windows), nsamples))

    tstart = time()
    for fold in folds:
        tr_inds, te_inds = fold
        # Xs[fold][train/test][instance]
        X_train_fold = np.array(Xs[tr_inds, :]).reshape(n_tr_samples, 100, 1)
        X_test_fold = np.array(Xs[te_inds, :]).reshape(nsamples, 100,1)
        # rescale
        # merged_data = np.vstack([X_train_fold, X_test_fold])
        # d_range = np.max(merged_data) - np.min(merged_data)



        print(f"(t={round(time() - tstart,2)}s) Training model on fold {fold}/{nfolds}...")
        model.fit(train_set={'X':X_train_fold})

        # loop over % missing
        for pm, windows in windows_by_pm.items():
            print(f"(t={round(time() - tstart,2)}s) fold {fold}: Testing model on {pm}% missing")
            for (idx, widx) in enumerate(windows):
                X_test_corrupted = X_test_fold.copy()
                X_test_corrupted[:, widx] = np.nan
                mask = np.isnan(X_test_corrupted) # mask ensures only misisng values are imputed
                imputed = model.impute(test_set={'X': X_test_corrupted})
                # get individual errors for uncertainty quantification
                errs[pm][fold, idx, :] = np.fromiter(map(calc_mae, imputed, X_test_fold, mask), dtype=np.float64) # no scaling / d_range 

            
    return errs



with open("../c6_folds_per_inst.json") as f:
    kepler_C6 = json.load(f)
# Xs[fold][train/test][instance]
Xs_per_inst_C6 = kepler_C6["Xs_per_inst"]

folds_C6 = kepler_C6["folds"]
for (i,inst) in enumerate(folds_C6):
    for (j,fold) in enumerate(inst):
        for (k,tr_te) in enumerate(fold):
            for l in range(len(tr_te)):
                folds_C6[i][j][k][l] -= 1 # julia v python indexing

with open("../GD_folds_per_inst.json") as f:
    kepler_GD = json.load(f)
# Xs[instance][fold]::Vector
# folds[instance][train/test]::Vector
Xs_per_inst_GD = kepler_GD["Xs_per_inst"]

folds_GD = kepler_GD["folds"]
for (i,inst) in enumerate(folds_GD):
    for (j,fold) in enumerate(inst):
        for (k,tr_te) in enumerate(fold):
            for l in range(len(tr_te)):
                folds_GD[i][j][k][l] -= 1 # julia v python indexing




# load imputation window indices
with open("../kepler_windows_julia_idx.json") as f:
    window_idxs_f = json.load(f)
window_idxs_kepler = {int(float(k)*100): np.array(v) - 1 for k, v in window_idxs_f.items()} # subtract one because julia indexing
# print(window_idxs.keys())


n_steps = 100
n_features = 1
rnn_hidden_size=128
batch_size=32
epochs=250
optimizer=Adam(lr=1e-3)
num_workers=0
device=None # infer the best device to use
model_saving_strategy=None

britsi = BRITS(
    n_steps=n_steps,
    n_features=n_features,
    rnn_hidden_size=rnn_hidden_size,
    batch_size=batch_size,
    use_BRITSI=True,
    epochs=epochs,
    optimizer=optimizer,
    num_workers=num_workers,
    device=device, 
    model_saving_strategy=model_saving_strategy
)


inst = int(sys.argv[1])
fold_scores_britsi_kepler_GD = evaluate_folds(britsi, 30, Xs_per_inst_GD[inst], window_idxs_kepler)
print("Kepler GD BRITS-I Mean MAE:")
for pm in window_idxs_kepler:
    print(f"{pm}%:", np.mean(fold_scores_britsi_kepler_GD[pm]))

with open("KEPC4_britsi_results_{}.pkl".format(inst), "wb") as f:
    pickle.dump(fold_scores_britsi_kepler_GD, f)


fold_scores_britsi_kepler_C6 = evaluate_folds(britsi, 30, Xs_per_inst_C6[inst], folds, window_idxs_kepler)
print("Kepler c6 BRITS-I Mean MAE:")
for pm in window_idxs_kepler:
    print(f"{pm}%:", np.mean(fold_scores_britsi_kepler_C6[pm]))

with open("KEPC6_britsi_results_{}.pkl".format(inst), "wb") as f:
    pickle.dump(fold_scores_britsi_kepler_C6, f)