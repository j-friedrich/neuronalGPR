import gpflow
from GPnet import BioNN, logpdf
import GPy
import h5py
import numpy as np
import os
from sys import argv
from time import time

import tensorflow as tf
from tensorflow.keras import layers

# pass the name of the UCI Dataset directory as argument
data_directory = argv[1] if len(argv) > 1 else 'bostonHousing'

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


def _split_data(split, normalize=True):
    # We load the indexes of the training and test sets
    print('Loading file: ' + _get_index_train_test_path(split, train=True))
    print('Loading file: ' + _get_index_train_test_path(split, train=False))
    index_train = np.loadtxt(_get_index_train_test_path(split, train=True))
    index_test = np.loadtxt(_get_index_train_test_path(split, train=False))

    X_train = X[[int(i) for i in index_train.tolist()]]
    y_train = y[[int(i) for i in index_train.tolist()]]
    X_test = X[[int(i) for i in index_test.tolist()]]
    y_test = y[[int(i) for i in index_test.tolist()]]

    if normalize:
        mean_X_train = np.mean(X_train, 0)
        std_X_train = np.std(X_train, 0)
        std_X_train[std_X_train == 0] = 1
        X_train_normalized = (X_train - mean_X_train) / std_X_train
        X_test_normalized = (X_test - mean_X_train) / std_X_train
        mean_y_train = np.mean(y_train)
        std_y_train = np.std(y_train)
        y_train_normalized = (y_train - mean_y_train) / std_y_train
        return (X_train_normalized, y_train_normalized, y_train,
                X_test_normalized, X_test, y_test,
                mean_X_train, std_X_train, mean_y_train, std_y_train)
    else:
        return (X_train, y_train, y_train,
                X_test, X_test, y_test,
                None, None, None, None)


def fitGP(method, normalize=True):
    assert method in ('VFE', 'FITC', 'GP')
    if data_directory == 'year-prediction-MSD' and method == 'FITC':
        return  # big data, thus using only SVGP, no FITC

    perf = np.nan * np.zeros((n_splits, 2))

    np.random.seed(1)
    for split in range(n_splits):
        (X_train_normalized, y_train_normalized, y_train,
         X_test_normalized, X_test, y_test,
         mean_X_train, std_X_train, mean_y_train, std_y_train) = _split_data(split, normalize)

        if data_directory != 'year-prediction-MSD':
            if method == 'GP':
                gp = GPy.models.GPRegression(X_train_normalized, y_train_normalized[:, None],
                                             GPy.kern.RBF(X_train_normalized.shape[1], ARD=True))
            else:
                gp = GPy.models.SparseGPRegression(X_train_normalized, y_train_normalized[:, None],
                                                   GPy.kern.RBF(
                                                       X_train_normalized.shape[1], ARD=True),
                                                   num_inducing=n_hidden)
            if method == 'FITC':
                gp.inference_method = GPy.inference.latent_function_inference.FITC()
            success = False
            for _ in range(10):
                try:
                    gp.optimize_restarts(robust=True)
                    success = True
                    break
                except:
                    pass
            if success:
                gp.save('results/%s/%s_split%g.hdf5' % (method, data_directory, split))
            else:
                continue
        else:
            gpflow.reset_default_graph_and_session()
            Z = X_train_normalized[np.random.choice(
                np.arange(len(X_train_normalized)), n_hidden, replace=False)].copy()
            gp = gpflow.models.SVGP(X_train_normalized, y_train_normalized[:, None],
                                    gpflow.kernels.RBF(X_train_normalized.shape[1], ARD=True),
                                    gpflow.likelihoods.Gaussian(), Z, minibatch_size=1000)
            adam = gpflow.train.AdamOptimizer().make_optimize_action(gp)
            gpflow.actions.Loop(adam, stop=30000)()
            gp.anchor(gp.enquire_session())
            saver = gpflow.saver.Saver()
            saver.save('results/%s/%s_split%g' % (method, data_directory, split), gp)

        if data_directory != 'year-prediction-MSD':
            m, v = np.squeeze(gp.predict(X_test_normalized))
        else:
            m, v = np.squeeze(gp.predict_y(X_test_normalized))
        if normalize:
            v *= std_y_train**2
            m = m * std_y_train + mean_y_train
        perf[split] = np.sqrt(np.mean((y_test - m)**2)), -logpdf(y_test - m, v).mean()

    np.save('results/%s/%s' % (method, data_directory), perf)


def fitBioNN(normalize=True):
    perf = np.nan * np.zeros((n_splits, 8))

    np.random.seed(1)
    for split in range(n_splits):
        (X_train_normalized, y_train_normalized, y_train,
         X_test_normalized, X_test, y_test,
         mean_X_train, std_X_train, mean_y_train, std_y_train) = _split_data(split, normalize)

        if data_directory != 'year-prediction-MSD':
            vfe = GPy.models.SparseGPRegression(
                X_train_normalized, y_train_normalized[:, None],
                GPy.kern.RBF(X_train_normalized.shape[1], ARD=True), num_inducing=n_hidden)
            vfe[:] = h5py.File('results/VFE/%s_split%g.hdf5' %
                               (data_directory, split), 'r')['param_array']
            nn = BioNN(X_train_normalized, y_train_normalized[:, None],
                       vfe.inducing_inputs, vfe.kern.lengthscale)
        else:
            gpflow.reset_default_graph_and_session()
            saver = gpflow.saver.Saver()
            vfe = saver.load('results/VFE/%s_split%g' % (data_directory, split))
            nn = BioNN(X_train_normalized, y_train_normalized[:, None],
                       vfe.feature.Z.value, vfe.kern.lengthscales.value)

        m, v = np.squeeze(nn.predict(X_test_normalized))
        if normalize:
            m = m * std_y_train + mean_y_train
            v = v * std_y_train**2
        perf[split, :2] = np.sqrt(np.mean((y_test - m)**2)), -logpdf(y_test - m, v).mean()
        perf[split, 2] = v.var()

        # measure prediction time
        if normalize:
            def predict(X_test):
                X_test_normalized = (X_test - mean_X_train) / std_X_train
                K = nn.kern.K(X_test_normalized, nn.inducing_inputs)
                m = K.dot(nn.w_mean)
                SNRinv = np.maximum(1 - np.sum(K**2, 1), 0)
                v = np.vstack([SNRinv, np.ones(len(m))]).T.dot(nn.wb_var)
                return np.concatenate([m * std_y_train + mean_y_train, v * std_y_train**2], 1)
        else:
            def predict(X_test):
                K = nn.kern.K(X_test, nn.inducing_inputs)
                m = K.dot(nn.w_mean)
                SNRinv = np.maximum(1 - np.sum(K**2, 1), 0)
                v = np.vstack([SNRinv, np.ones(len(m))]).T.dot(nn.wb_var)
                return np.concatenate([m, v], 1)

        for i in range(5):
            t = -time()
            _ = predict(X_test)
            t += time()
            perf[split, 3 + i] = t

    np.save('results/BioNN/' + data_directory, perf)


def fitANN(normalize=True):
    etas = np.array([1,  2,  5, 10, 20, 50, 100]) * {'bostonHousing': 1e-7,
                                                     'concrete': 1e-7,
                                                     'energy': 1e-10,
                                                     'kin8nm': 1e-5,
                                                     'naval-propulsion-plant': 1e-9,
                                                     'power-plant': 1e-6,
                                                     'protein-tertiary-structure': 1e-5,
                                                     'wine-quality-red': 1e-6,
                                                     'yacht': 1e-9,
                                                     'year-prediction-MSD': 1e-4}[data_directory]

    perf = np.nan * np.zeros((n_splits, 8))

    np.random.seed(1)
    for split in range(n_splits):
        (X_train_normalized, y_train_normalized, y_train,
         X_test_normalized, X_test, y_test,
         mean_X_train, std_X_train, mean_y_train, std_y_train) = _split_data(split, normalize)

        if data_directory != 'year-prediction-MSD':
            gp = GPy.models.SparseGPRegression(X_train_normalized, y_train_normalized[:, None],
                                               GPy.kern.RBF(X_train_normalized.shape[1], ARD=True),
                                               num_inducing=n_hidden)
            gp[:] = h5py.File('results/VFE/%s_split%g.hdf5' %
                              (data_directory, split), 'r')['param_array']
            var = gp.Gaussian_noise.variance
            varK = gp.kern.variance
            Kfu = gp.kern.K(X_train_normalized, gp.inducing_inputs)
            Kfu_test = gp.kern.K(X_test_normalized, gp.inducing_inputs)
            w = gp.posterior.woodbury_vector.ravel()
            woodbury_inv = gp.posterior.woodbury_inv
        else:
            gpflow.reset_default_graph_and_session()
            saver = gpflow.saver.Saver()
            gp = saver.load('results/VFE/%s_split%g' % (data_directory, split))
            var = gp.likelihood.variance.value
            varK = gp.kern.variance.value
            Kfu = gp.kern.compute_K(X_train_normalized, gp.feature.Z.value)
            Kuu = gp.kern.compute_K(gp.feature.Z.value, gp.feature.Z.value)
            Kfu_test = gp.kern.compute_K(X_test_normalized, gp.feature.Z.value)
            Sigma = np.linalg.inv(Kfu.T.dot(Kfu) + var*Kuu)
            w = Sigma.dot(Kfu.T.dot(y_train_normalized))
            woodbury_inv = np.linalg.inv(Kuu) - var*Sigma

        def custom_loss():  # neg loglikelihood
            def loss(y_true, y_pred):
                return tf.divide(tf.square(y_pred[..., 0] - y_true[..., 0]), y_pred[..., 1]) + \
                    tf.math.log(y_pred[..., 1])
            return loss

        def build_model(eta):
            u, s, v = np.linalg.svd(woodbury_inv)
            U = (u + v.T).dot(np.diag(np.sqrt(s))) / 2
            inputs = layers.Input(shape=(n_hidden,))
            m = layers.Dense(1, kernel_initializer=tf.constant_initializer(w),
                             trainable=False)(inputs)
            x = layers.Dense(n_hidden, kernel_initializer=tf.constant_initializer(U),
                             activation=tf.square)(inputs)

            def act(a): return tf.math.softplus(a / var / 2) * var * 2
            v = layers.Dense(1, kernel_initializer=tf.constant_initializer(-np.ones((1, n_hidden))),
                             bias_initializer=tf.constant_initializer(var + varK),
                             activation=act)(x)
            outputs = layers.concatenate([m, v])
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            model.compile(loss=custom_loss(), optimizer=tf.keras.optimizers.Adam(eta))
            return model

        # find best learning rate using 5-fold cross validation
        best_loss = np.inf
        best_eta = etas[0]
        for eta in etas:
            loss = 0
            for fold in range(5):
                model = build_model(eta)
                train_idx = np.ones(X_train_normalized.shape[0], dtype=bool)
                train_idx[fold::5] = False
                history = model.fit(
                    Kfu[train_idx], y_train_normalized[train_idx], epochs=n_epochs,
                    validation_data=(Kfu[~train_idx], y_train_normalized[~train_idx]), verbose=0)
                loss += history.history['val_loss'][-1]
            if loss < best_loss:
                best_loss = loss
                best_eta = eta

        model = build_model(best_eta)
        history = model.fit(Kfu, y_train_normalized, epochs=n_epochs, verbose=0)
        if data_directory != 'year-prediction-MSD':
            m = np.squeeze(gp.predict(X_test_normalized))[0]
        else:
            m = np.squeeze(gp.predict_y(X_test_normalized))[0]
        v = np.squeeze(model.predict(Kfu_test)).T[1]
        if normalize:
            m = m * std_y_train + mean_y_train
            v = v * std_y_train**2
        perf[split, :2] = np.sqrt(np.mean((y_test - m)**2)), -logpdf(y_test - m, v).mean()
        perf[split, 2] = best_eta

        # measure prediction time
        if data_directory != 'year-prediction-MSD':
            U, Ub, _, _, w, wb = model.get_weights()
            m = gp.posterior.woodbury_vector
            var = 2 * gp.Gaussian_noise.variance

            def act(a): return np.log(1 + np.exp(a/var)) * var
            if normalize:
                def predict(X_test):
                    X_test_normalized = (X_test - mean_X_train) / std_X_train
                    K = gp.kern.K(X_test_normalized, gp.inducing_inputs)
                    return np.concatenate([K.dot(m) * std_y_train + mean_y_train,
                                           act(((K.dot(U) + Ub)**2).dot(w) + wb) * std_y_train**2], 1)
            else:
                def predict(X_test):
                    K = gp.kern.K(X_test, gp.inducing_inputs)
                    return np.concatenate([K.dot(m),
                                           act(((K.dot(U) + Ub)**2).dot(w) + wb)], 1)
        else:
            if normalize:
                def predict(X_test):
                    X_test_normalized = (X_test - mean_X_train) / std_X_train
                    K = gp.kern.compute_K(X_test_normalized, gp.feature.Z.value)
                    m, v = np.squeeze(model.predict(K)).T
                    return np.array([m * std_y_train + mean_y_train, v * std_y_train**2])
            else:
                def predict(X_test):
                    K = gp.kern.compute_K(X_test, gp.feature.Z.value)
                    m, v = np.squeeze(model.predict(K)).T
                    return np.array([m, v])

        for i in range(5):
            t = -time()
            _ = predict(X_test)
            t += time()
            perf[split, 3 + i] = t

    np.save('results/ANN/' + data_directory, perf)


# fit all the different methods on the dataset
fitGP('VFE')
fitGP('FITC')
if len(np.loadtxt(_get_index_train_test_path(0))) < 2000:
    fitGP('GP')  # fit GP on small enough datasets
fitBioNN()
fitANN()
