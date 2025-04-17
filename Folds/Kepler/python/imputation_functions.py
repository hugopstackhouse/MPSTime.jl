import numpy as np
from pypots.utils.random import set_random_seed
from pypots.utils.metrics import calc_mae
import sys
from cdrec.python.recovery import centroid_recovery as CDrec
from time import time
import torch

def evaluate_folds_pypots(model, Xs, folds, windows_by_pm):
    # make the splits
    nfolds = len(folds)
    errs = {}
    n_tr_samples = len(folds[0][0])
    nsamples =  len(folds[0][1])

    for pm, windows in windows_by_pm.items():
        errs[pm] = -np.ones((nfolds, len(windows), nsamples))

    tstart = time()
    for f_idx, fold in enumerate(folds):
        tr_inds, te_inds = fold
        X_train_fold = np.array(Xs[tr_inds, :]).reshape(n_tr_samples, 100, 1)
        X_test_fold = np.array(Xs[te_inds, :]).reshape(nsamples, 100, 1)


        print(f"(t={round(time() - tstart,2)}s) Training model on fold {f_idx}/{nfolds}...")
        model.fit(train_set={'X':X_train_fold})

        # loop over % missing
        for pm, windows in windows_by_pm.items():
            print(f"(t={round(time() - tstart,2)}s) fold {f_idx}: Testing model on {pm}% missing")
            for (idx, widx) in enumerate(windows):
                X_test_corrupted = X_test_fold.copy()
                X_test_corrupted[:, widx] = np.nan
                mask = np.isnan(X_test_corrupted) # mask ensures only misisng values are imputed
                imputed = model.impute(test_set={'X': X_test_corrupted})
                if imputed.shape[1] == 1:
                    imputed = imputed.squeeze(axis=1) # squeeze out the extra axis csdi adds
                # get individual errors for uncertainty quantification
                errs[pm][f_idx, idx, :] = np.fromiter(map(calc_mae, imputed, X_test_fold, mask), dtype=np.float64) 
    return errs



def evaluate_folds_cdrec(Xs, folds, windows_by_pm):
    errs = {}
    nfolds = len(folds)
    n_tr_samples = len(folds[0][0])
    nsamples =  len(folds[0][1])

    for pm, windows in windows_by_pm.items():
        errs[pm] = -np.ones((nfolds, len(windows), nsamples))

    tstart = time()
    for f_idx, fold in enumerate(folds):
        tr_inds, te_inds = fold
        # Xs[fold][train/test][instance]
        X_train_fold = np.array(Xs[tr_inds, :]).reshape(n_tr_samples, 100, 1)
        X_test_fold = np.array(Xs[te_inds, :]).reshape(nsamples, 100, 1)
        # check class distributions
        print(f"Computing CDrec on fold {f_idx}...")
        # loop over % missing
        for pm, windows in windows_by_pm.items():
            print(f"(t={round(time() - tstart,2)}s) fold {f_idx}: Testing model on {pm}% missing")
            for (idx, widx) in enumerate(windows):
                X_test_corrupted = X_test_fold.copy()
                X_test_corrupted[:, widx] = np.nan
                mask = np.isnan(X_test_corrupted) # mask ensures only missing values are imputed
                Xdata = np.concatenate([X_train_fold.squeeze(), X_test_corrupted.squeeze()])
                cdrec_imputed_raw = CDrec(matrix=Xdata) # using default paramss
                cdrec_imputed = cdrec_imputed_raw[X_train_fold.shape[0]:][:].reshape([-1, 100, 1]) # only the test data from the concatenated matrix            
                # get individual errors for uncertainty quantification
                errs[pm][f_idx, idx, :] = np.fromiter(map(calc_mae, cdrec_imputed, X_test_fold, mask), dtype=np.float64) 
    return errs

def restack_Xdata(Xs):
    for (i,Xdata) in enumerate(Xs):
        Xs[i] = np.transpose(np.vstack(Xdata))
    return Xs

def sub_one_from_folds(folds):
    for (i,inst) in enumerate(folds):
        for (j,fold) in enumerate(inst):
            for (k,tr_te) in enumerate(fold):
                for l in range(len(tr_te)):
                    folds[i][j][k][l] -= 1 # julia v python indexing
    return folds