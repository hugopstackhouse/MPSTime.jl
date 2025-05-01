import aeon
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
import tensorflow as tf
import numpy as np
import pickle as pkl
from scipy.stats import zscore
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV

train_data = np.loadtxt("Astronomy/KeplerBinaryOrigBal_TRAIN.txt")
test_data = np.loadtxt("Astronomy/KeplerBinaryOrigBal_TEST.txt")
X_train = train_data[:, 1:][:, :100]
y_train = train_data[:, 0].astype(int)
X_test = test_data[:, 1:][:, :100]
y_test = test_data[:, 0].astype(int)

# recombine data
Xs = np.vstack([X_train, X_test])
print(Xs.shape)
ys = np.concatenate([y_train, y_test])
print(ys.shape)

# load the fold idxs
fold_idxs = {}
for fold in range(30):
    train_idxs = np.loadtxt(f"Astronomy/resample{fold}Indices_TRAIN.txt")
    test_idxs = np.loadtxt(f"Astronomy/resample{fold}Indices_TEST.txt")
    fold_idxs[fold] = {"train": train_idxs.astype(int), "test": test_idxs.astype(int)}

knn = KNeighborsTimeSeriesClassifier(distance='dtw')
fold_scores = {}
for fold in range(30):
    print(f"Evaluating fold {fold}")
    # get the training folds
    train_idxs = fold_idxs[fold]["train"]
    test_idxs = fold_idxs[fold]["test"]
    X_train = Xs[train_idxs, :]
    y_train = ys[train_idxs]
    X_test = Xs[test_idxs, :]
    y_test = ys[test_idxs]

    X_train = zscore(X_train, axis=1)
    X_test = zscore(X_test, axis=1)

    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test) # score the best estimator
    fold_scores[fold] = score
    print(f"fold {fold} score: {score}")

fold_accs = [fold_scores[fold] for fold in fold_scores]
print(np.mean(fold_accs))
np.savetxt("1nn_folds_astroBench_Zscore.txt", fold_accs)
