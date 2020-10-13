from GPnet import logpdf
import math
import numpy as np
import os
import sys
from time import time
sys.path.append('Probabilistic-Backpropagation-master/c/PBP_net')
import PBP_net

# pass the name of the UCI Dataset directory as argument
try:
    data_directory = sys.argv[1]
except:
    data_directory = 'bostonHousing'

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
n_epochs = int(np.loadtxt(_DATA_DIRECTORY_PATH + 'n_epochs.txt'))


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


perf = np.nan * np.zeros((n_splits, 2, 3))

for n_layers in (1, 2):
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

        net = PBP_net.PBP_net(X_train, y_train, [n_hidden] * n_layers,
                              normalize=True, n_epochs=n_epochs)

        # We make predictions for the test set
        t = -time()
        m, v, v_noise = net.predict(X_test)
        t += time()
        # We compute the test RMSE and ll
        perf[split, n_layers - 1, :2] = (np.sqrt(np.mean((y_test - m)**2)),
                                         logpdf(y_test - m, v + v_noise).mean())
        perf[split, n_layers - 1, 2] = t

np.save('results/PBP/' + data_directory, perf)
