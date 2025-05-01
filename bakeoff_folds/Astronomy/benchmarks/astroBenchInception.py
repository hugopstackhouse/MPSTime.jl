import aeon
from aeon.classification.deep_learning import InceptionTimeClassifier
import tensorflow as tf
import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV
from tensorflow.keras.callbacks import EarlyStopping
import pickle as pkl
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


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

#params = {
#    "depth": [2, 4, 6],
#    "n_filters": [8, 16, 32],
#    "kernel_size": [8, 16, 32, 64],
#}

inceptionTime = InceptionTimeClassifier(verbose=False, batch_size=16, random_state=0)
#cv = KFold(n_splits=5, random_state=42, shuffle=True)
#inceptionCV = RandomizedSearchCV(inceptionTime, params, n_iter=15, cv=cv, verbose=1)

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

    inceptionTime.fit(X_train, y_train)
    score = inceptionTime.score(X_test, y_test) # score the best estimator
    fold_scores[fold] = score
    print(f"Finished fold {fold}, Score: {score}")

fold_accs = [fold_scores[fold] for fold in fold_scores]
np.savetxt("inceptionTime_folds_astroBench.txt", fold_accs)
