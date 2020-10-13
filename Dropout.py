import numpy as np
import os
import sys
from time import time
import net


# pass the name of the UCI Dataset directory as 1st argument,
# the number of hidden layers as 2nd.
try:
    data_directory = sys.argv[1]
except:
    data_directory = 'bostonHousing'
try:
    n_layers = int(sys.argv[2])
except:
    n_layers = 1

epochs = 40
epochs_multiplier = 100


_UCI_DIRECTORY_PATH = "DropoutUncertaintyExps-master/UCI_Datasets/"
subfolders = [f.name for f in os.scandir(_UCI_DIRECTORY_PATH) if f.is_dir()]
subfolders.sort()

if data_directory not in subfolders:
    raise ValueError("data directory must be one of the following " +
                     repr(subfolders) + " but was " + data_directory)

_DATA_DIRECTORY_PATH = _UCI_DIRECTORY_PATH + data_directory + "/data/"

data = np.loadtxt(_DATA_DIRECTORY_PATH + "data.txt")
index_features = np.loadtxt(_DATA_DIRECTORY_PATH + "index_features.txt")
index_target = np.loadtxt(_DATA_DIRECTORY_PATH + "index_target.txt")
X = data[:, [int(i) for i in index_features.tolist()]]
y = data[:, int(index_target.tolist())]

n_splits = int(np.loadtxt(_DATA_DIRECTORY_PATH + 'n_splits.txt'))
n_hidden = int(np.loadtxt(_DATA_DIRECTORY_PATH + "n_hidden.txt"))

validation_ll = np.genfromtxt(_UCI_DIRECTORY_PATH + data_directory + "/results/validation_ll_" +
                              str(epochs_multiplier) + "_xepochs_1_hidden_layers.txt")


def _get_index_train_test_path(split_num, train=True):
    """
         Method to generate the path containing the training/test split for the given
         split number (generally from 1 to 20).
         @param split_num      Split number for which the data has to be generated
         @param train          Is true if the data is training data. Else false.
         @return path          Path of the file containing the requried data
    """
    if train:
        return _DATA_DIRECTORY_PATH + "index_train_" + str(split_num) + ".txt"
    else:
        return _DATA_DIRECTORY_PATH + "index_test_" + str(split_num) + ".txt"


perf = np.nan * np.zeros((n_splits, 6, 4))

np.random.seed(1)
for split in range(n_splits):
    # We load the indexes of the training and test sets
    print('Loading file: ' + _get_index_train_test_path(split, train=True))
    print('Loading file: ' + _get_index_train_test_path(split, train=False))
    index_train = np.loadtxt(_get_index_train_test_path(split, train=True))
    index_test = np.loadtxt(_get_index_train_test_path(split, train=False))

    X_train = X[[int(i) for i in index_train.tolist()]]
    y_train = y[[int(i) for i in index_train.tolist()]]
    X_test = X[[int(i) for i in index_test.tolist()]]
    y_test = y[[int(i) for i in index_test.tolist()]]

    tmp = validation_ll[split * len(validation_ll) // n_splits:
                        (split + 1) * len(validation_ll) // n_splits, 1::2]
    best_dropout, best_tau = tmp[np.argmax(tmp[:, 2]), :2]
    best_network = net.net(X_train, y_train, ([n_hidden] * n_layers), normalize=True,
                           n_epochs=int(epochs * epochs_multiplier), tau=best_tau,
                           dropout=best_dropout)

    for j, T in enumerate((1, 1, 10, 100, 1000, 10000)): 
    # there's apparenlty some overhead for the first prediction, hence it's done twice
        t = -time()
        error, MC_error, ll = best_network.predict(X_test, y_test, T)
        t += time()
        perf[split, j] = error, MC_error, ll, t

np.save('results/Dropout/%s_%glayers' % (data_directory, n_layers), perf)
