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

from imputation_functions import evaluate_folds_pypots, evaluate_folds_cdrec, restack_Xdata, sub_one_from_folds

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


csdi = CSDI(
    n_steps=n_steps,
    n_features=n_features, # univariate time series, so num features is equal to one
    n_layers=6,
    n_heads=2,
    n_channels=128,
    d_time_embedding=64,
    d_feature_embedding=32,
    d_diffusion_embedding=128,
    target_strategy="random",
    n_diffusion_steps=50,
    batch_size=batch_size,
    epochs=epochs,
    patience=None,
    optimizer=optimizer,
    num_workers=num_workers,
    device=device,
    model_saving_strategy=model_saving_strategy
)

######### load data
with open("../c6_folds_per_inst.json") as f:
    kepler_C6 = json.load(f)
# Xs[fold][train/test][instance]
Xs_per_inst_C6 = restack_Xdata(kepler_C6["Xs_per_inst"]) # col major -> row major order
folds_C6 = sub_one_from_folds(kepler_C6["folds"]) # julia indexing starts at one


with open("../kepler_windows_python_idx.json") as f:
    window_idxs_f = json.load(f)
window_idxs_kepler = {int(float(k)*100): np.array(v) for k, v in window_idxs_f.items()}



inst = int(sys.argv[1])
if len(sys.argv) == 3:
    nfolds = int(sys.argv[2])
else:
    nfolds = len(folds_C6[0])

folds = folds_C6[inst][0:nfolds]
Xs = Xs_per_inst_C6[inst]

#### Britsi
fold_scores_britsi_kepler_C6 = evaluate_folds_pypots(britsi, Xs, folds, window_idxs_kepler)
print("Kepler c6 BRITS-I Mean MAE:")
for pm in window_idxs_kepler:
    print(f"{pm}%:", np.mean(fold_scores_britsi_kepler_C6[pm]))

with open("results/KEPC6_britsi_results_{}.pkl".format(inst), "wb") as f:
    pickle.dump(fold_scores_britsi_kepler_C6, f)


# #### CSDI
# fold_scores_CSDI_kepler_C6 = evaluate_folds_pypots(csdi, Xs_per_inst_C6[inst], folds_C6[inst], window_idxs_kepler)
# print("Kepler c6 CSDI Mean MAE:")
# for pm in window_idxs_kepler:
#     print(f"{pm}%:", np.mean(fold_scores_CSDI_kepler_C6[pm]))

# with open("results/KEPC6_CSDI_results_{}.pkl".format(inst), "wb") as f:
#     pickle.dump(fold_scores_CSDI_kepler_C6, f)


# # CDREC
# fold_scores_CDrec_kepler_C6 = evaluate_folds_cdrec(Xs_per_inst_C6[inst], folds_C6[inst], window_idxs_kepler)
# print("Kepler c6 CDrec Mean MAE:")
# for pm in window_idxs_kepler:
#     print(f"{pm}%:", np.mean(fold_scores_CDrec_kepler_C6[pm]))

# with open("results/KEPC6_CDREC_results_{}.pkl".format(inst), "wb") as f:
#     pickle.dump(fold_scores_CDrec_kepler_C6, f)