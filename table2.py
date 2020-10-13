import h5py
import GPy
import numpy as np
import os
from GPnet import NN, KL


def KLs(data_directory):
    _DATA_DIRECTORY_PATH = "DropoutUncertaintyExps-master/UCI_Datasets/" + \
        data_directory + "/data/"

    data = np.loadtxt(_DATA_DIRECTORY_PATH + "data.txt")
    index_features = np.loadtxt(_DATA_DIRECTORY_PATH + "index_features.txt")
    index_target = np.loadtxt(_DATA_DIRECTORY_PATH + "index_target.txt")
    X = data[:, [int(i) for i in index_features.tolist()]]
    y = data[:, int(index_target.tolist())]

    n_splits = int(np.loadtxt(_DATA_DIRECTORY_PATH + 'n_splits.txt'))
    n_hidden = int(np.loadtxt(_DATA_DIRECTORY_PATH + "n_hidden.txt"))

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

    normalize = True
    KLpq = np.nan * np.zeros((n_splits, 3))
    KLqp = np.nan * np.zeros((n_splits, 3))

    for split in range(n_splits):
        index_train = np.loadtxt(_get_index_train_test_path(split, train=True))
        index_test = np.loadtxt(_get_index_train_test_path(split, train=False))

        X_train = X[[int(i) for i in index_train.tolist()]]
        y_train = y[[int(i) for i in index_train.tolist()]]
        X_test = X[[int(i) for i in index_test.tolist()]]

        if normalize:
            mean_X_train = np.mean(X_train, 0)
            std_X_train = np.std(X_train, 0)
            std_X_train[std_X_train == 0] = 1
            X_train_normalized = (X_train - mean_X_train) / std_X_train
            X_test_normalized = (X_test - mean_X_train) / std_X_train
            mean_y_train = np.mean(y_train)
            std_y_train = np.std(y_train)
            y_train_normalized = (y_train - mean_y_train) / std_y_train
        else:
            X_train_normalized = X_train
            y_train_normalized = y_train

        gp = GPy.models.GPRegression(X_train_normalized, y_train_normalized[:, None],
                                     GPy.kern.RBF(X_train.shape[1], ARD=True))
        gp[:] = h5py.File('results/GP/%s_split%g.hdf5' %
                          (data_directory, split), 'r')['param_array']

        vfe = GPy.models.SparseGPRegression(X_train_normalized, y_train_normalized[:, None],
                                            GPy.kern.RBF(X_train.shape[1], ARD=True),
                                            num_inducing=n_hidden)
        vfe[:] = h5py.File('results/VFE/%s_split%g.hdf5' %
                           (data_directory, split), 'r')['param_array']

        fitc = GPy.models.SparseGPRegression(X_train_normalized, y_train_normalized[:, None],
                                             GPy.kern.RBF(X_train.shape[1], ARD=True),
                                             num_inducing=n_hidden)
        fitc[:] = h5py.File('results/FITC/%s_split%g.hdf5' %
                            (data_directory, split), 'r')['param_array']

        nn = NN(X_train_normalized, y_train_normalized[:, None],
                vfe.inducing_inputs, vfe.kern.lengthscale)

        KLpq[split] = [KL(gp, m, X_test_normalized) for m in (vfe, nn, fitc)]
        KLqp[split] = [KL(m, gp, X_test_normalized) for m in (vfe, nn, fitc)]

    return KLpq, KLqp


_UCI_DIRECTORY_PATH = "DropoutUncertaintyExps-master/UCI_Datasets"
subfolders = [f.name for f in os.scandir(_UCI_DIRECTORY_PATH) if f.is_dir()]
subfolders.sort()


KLpq = {}
KLqp = {}
for f in subfolders:
    try:
        np.load('results/GP/%s.npy' % f)
        KLpq[f], KLqp[f] = KLs(f)
    except:
        continue

for j, kl in enumerate((KLpq, KLqp)):
    print(('KL(P|Q)', 'KL(Q|P)')[j])
    print(' ' * 26 + 'VFE           BioNN              FITC')
    for f in kl.keys():
        print('%19s' % f, end='  ')
        try:
            perf = kl[f]
            n_splits = int(np.loadtxt(_UCI_DIRECTORY_PATH + '/' + f + '/data/n_splits.txt'))
            for k in range(3):
                print((('%9.2f+-%8.2f' if k == 2 else '%6.2f+-%5.2f') % tuple(map(
                    lambda x: x if x < 1e50 else np.inf, (perf.mean(0)[k],
                                                          perf.std(0)[k] / np.sqrt(n_splits - 1))))
                       ).replace('nan', ' nan '), end='  ')
        except:
            print('ERROR', end='  ')
        print()
