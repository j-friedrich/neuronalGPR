from copy import deepcopy
from GPnet import RMSE, NLPD, logpdf
import GPy
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append('streaming_sparse_gp-master/code')


plt.rc('font', size=18)
plt.rc('legend', **{'fontsize': 12})
plt.rc('lines', linewidth=3)
plt.rc('pdf', fonttype=42)

# create figure directory if not existent yet
os.makedirs('fig', exist_ok=True)
os.makedirs('results/streamingGP', exist_ok=True)

# Ed Snelson's example data
X = np.genfromtxt('snelson_data/train_inputs')[:, None]
Y = np.genfromtxt('snelson_data/train_outputs')[:, None]
N = len(X)


# full GP
kernel = GPy.kern.RBF(1)
truth = GPy.models.GPRegression(X, Y, kernel)
truth.optimize()

np.random.seed(0)
Xstream, Ystream = [], []
for _ in range(100):
    x = np.random.rand(N // 2)[:, None] * 6
    y = truth.posterior_samples_f(x, 1)[..., 0] + \
        np.sqrt(truth.Gaussian_noise.variance) * np.random.randn(N // 2, 1)
    Xstream.append(x)
    Ystream.append(y)
Xstream = np.ravel(Xstream)[:, None]
Ystream = np.ravel(Ystream)[:, None]

# np.savez_compressed('stream.npz', Xstream=Xstream, Ystream=Ystream)
# Xstream = np.load('stream.npz')['Xstream']
# Ystream = np.load('stream.npz')['Ystream']


def d(x, pm=-1):
    return np.mean(x, 0) + pm * np.std(x, 0) / np.sqrt(len(x) - 1)


# sparse GP
def sparseSGD(run, eta, etaV, T):
    np.random.seed(run)
    idx_train = np.sort(np.random.choice(range(N), N // 2, False))
    idx_test = np.setdiff1d(range(N), idx_train)
    Xtest = X[idx_test]
    Ytest = Y[idx_test]

    vfe = GPy.models.SparseGPRegression(Xstream, Ystream, GPy.kern.RBF(1), num_inducing=6)
    vfe.Gaussian_noise.variance = truth.Gaussian_noise.variance
    vfe.optimize()

    """Computation of weights w for mean prediction and weigths w^Sigma & bias
    b^Sigma (wV & bV) for variance prediction using stochastic gradient decent"""
    K_uf_test = vfe.kern.K(vfe.Z, Xtest)
    w = np.zeros((6, 1))
    wV, bV = 1, truth.Gaussian_noise.variance[0]
    rmse = np.zeros(T)
    nlpd = np.zeros(T)
    for t in range(T):
        genX = Xstream[t * 100:(t + 1) * 100]
        genY = Ystream[t * 100:(t + 1) * 100]
        K_uf = vfe.kern.K(vfe.Z, genX)
        for i in range(len(genX)):
            delta = (K_uf.T[i].dot(w) - genY[i])[0]
            w -= eta * K_uf[:, i:i + 1] * delta
            rho = np.maximum(1 - np.sum(K_uf[:, i]**2, 0), 0)
            deltaV = wV * rho + bV - delta**2
            wV -= etaV * rho * deltaV
            bV -= etaV * deltaV
        mu = K_uf_test.T.dot(w)
        rho = np.maximum(1 - np.sum(K_uf_test**2, 0), 0)[:, None]
        Sigma = wV * rho + bV
        rmse[t] = np.sqrt(np.mean((mu - Ytest)**2))
        nlpd[t] = - logpdf(mu - Ytest, Sigma).mean()
    return (rmse, nlpd), [RMSE(vfe, Xtest, Ytest), NLPD(vfe, Xtest, Ytest)]


def init_Z(cur_Z, new_X, use_old_Z=True):
    if use_old_Z:
        Z = np.copy(cur_Z)
    else:
        M = cur_Z.shape[0]
        M_old = int(0.7 * M)
        M_new = M - M_old
        old_Z = cur_Z[np.random.permutation(M)[0:M_old], :]
        new_Z = new_X[np.random.permutation(new_X.shape[0])[0:M_new], :]
        Z = np.vstack((old_Z, new_Z))
    return Z


def streamingGP(run, M=6, use_old_Z=True):
    # N.B.: need to run in a different environment with e.g.
    # python 2.7, gpflow=0.5 and tensorflow=1.4.1
    import tensorflow as tf
    import gpflow as GPflow
    import osgpr
    np.random.seed(run)
    idx_train = np.sort(np.random.choice(range(N), N // 2, False))
    idx_test = np.setdiff1d(range(N), idx_train)
    Xtest = X[idx_test]
    Ytest = Y[idx_test]

    rmse = np.zeros(T)
    nlpd = np.zeros(T)

    # get the first portion and call sparse GP regression
    X1 = Xstream[:100]
    y1 = Ystream[:100]
    Z1 = X1[np.random.permutation(X1.shape[0])[0:M], :]

    tf.reset_default_graph()
    model1 = GPflow.sgpr.SGPR(X1, y1, GPflow.kernels.RBF(1), Z=Z1)
    model1.likelihood.variance = 0.1
    model1.kern.variance = .3
    model1.kern.lengthscales = 0.6
    model1.optimize(disp=1)

    mu, Sigma = model1.predict_y(Xtest)
    rmse[0] = np.sqrt(np.mean((mu - Ytest)**2))
    nlpd[0] = - logpdf(mu - Ytest, Sigma).mean()

    Zopt = model1.Z.value
    mu1, Su1 = model1.predict_f_full_cov(Zopt)
    if len(Su1.shape) == 3:
        Su1 = Su1[:, :, 0]

    # now call online method on the other portions of the data
    for t in range(1, T):
        X2 = Xstream[t * 100:(t + 1) * 100]
        y2 = Ystream[t * 100:(t + 1) * 100]

        x_free = tf.placeholder('float64')
        model1.kern.make_tf_array(x_free)
        X_tf = tf.placeholder('float64')
        with model1.kern.tf_mode():
            Kaa1 = tf.Session().run(
                model1.kern.K(X_tf),
                feed_dict={x_free: model1.kern.get_free_state(), X_tf: model1.Z.value})

        Zinit = init_Z(Zopt, X2, use_old_Z)
        model2 = osgpr.OSGPR_VFE(X2, y2, GPflow.kernels.RBF(1), mu1, Su1, Kaa1,
                                 Zopt, Zinit)
        model2.likelihood.variance = model1.likelihood.variance.value
        model2.kern.variance = model1.kern.variance.value
        model2.kern.lengthscales = model1.kern.lengthscales.value
        model2.optimize(disp=1)

        model1 = deepcopy(model2)
        Zopt = model1.Z.value
        mu1, Su1 = model1.predict_f_full_cov(Zopt)
        if len(Su1.shape) == 3:
            Su1 = Su1[:, :, 0]

        mu, Sigma = model1.predict_y(Xtest)
        rmse[t] = np.sqrt(np.mean((mu - Ytest)**2))
        nlpd[t] = - logpdf(mu - Ytest, Sigma).mean()
    np.savez_compressed('results/streamingGP/%g.npz' % run, rmse=rmse, nlpd=nlpd)
    return rmse, nlpd


runs, T = 10, 100
perf = np.empty((runs, 2, T))
VFE = np.empty((runs, 2))
for run in range(runs):
    perf[run], VFE[run] = sparseSGD(run, 250, .005, T)


stream = np.empty((runs, 2, T))
for run in range(runs):
    try:  # load saved result
        stream[run] = (np.load('results/streamingGP/%g.npz' % run)['rmse'],
                       np.load('results/streamingGP/%g.npz' % run)['nlpd'])
    except:
        stream[run] = streamingGP(run)


def plot(typ):
    plt.figure(figsize=(6, 4))
    T = perf.shape[-1]
    j = ('RMSE', 'NLPD').index(typ)
    data = perf[:, j]
    for c, label in ((1, 'VFE'),):
        plt.axhline(np.mean(VFE[:, j]), c='C{}'.format(c), label=label)
        plt.fill_between((0, T), [d(VFE[:, j], -1)] * 2, [d(VFE[:, j], +1)] * 2,
                         color='C{}'.format(c), alpha=.3)
    plt.plot(range(1, 1 + T), np.mean(data, 0), c='C2', label='BioNN')
    plt.fill_between(range(1, 1 + T), d(data, -1), d(data, +1),
                     color='C2', alpha=.3)
    plt.plot(range(1, 1 + T), np.mean(stream, 0)[j], c='C3', label='streamingGP')
    plt.fill_between(range(1, 1 + T), d(stream[:, j], -1), d(stream[:, j], +1),
                     color='C3', alpha=.3)
    plt.xticks(range(0, 100, 20), range(0, 10000, 2000))
    plt.xlabel('Samples')
    plt.ylabel(typ)
    plt.xlim(0, T)
    plt.legend()
    plt.tight_layout(.05)


plot('RMSE')
plt.savefig('fig/snelson_stream-RMSE.pdf', transparent=True)
plot('NLPD')
plt.savefig('fig/snelson_stream-NLPD.pdf', transparent=True)
