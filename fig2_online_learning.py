from GPnet import RMSE, NLPD, logpdf
import GPy
import matplotlib.pyplot as plt
import numpy as np
import os


plt.rc('font', size=18)
plt.rc('legend', **{'fontsize': 12})
plt.rc('lines', linewidth=3)
plt.rc('pdf', fonttype=42)

# create figure directory if not existent yet
os.makedirs('fig', exist_ok=True)

# Ed Snelson's example data
X = np.genfromtxt('snelson_data/train_inputs')[:, None]
Y = np.genfromtxt('snelson_data/train_outputs')[:, None]
N = len(X)


# full GP
def CD(run, eta, T):
    np.random.seed(run)
    idx_train = np.sort(np.random.choice(range(N), N // 2, False))
    idx_test = np.setdiff1d(range(N), idx_train)
    Xtrain = X[idx_train]
    Ytrain = Y[idx_train]
    Xtest = X[idx_test]
    Ytest = Y[idx_test]

    kernel = GPy.kern.RBF(1)
    full = GPy.models.GPRegression(Xtrain, Ytrain, kernel)
    full.optimize()

    var = full.Gaussian_noise.variance
    K_ff = kernel.K(Xtrain)
    K_ff_test = kernel.K(Xtest, Xtrain)

    # computation of w using coordinate decent
    w = np.zeros((len(Xtrain), 1))
    rmse = np.zeros(T)
    for t in range(T):
        for i in range(len(Xtrain)):
            pre = (K_ff[i] >= kernel.variance.values)[:, None]
            post = (K_ff[i].dot(w) - Ytrain[i]) * np.ones((len(Xtrain), 1))
            w -= eta * pre * (post + var * w)
        mu = K_ff_test.dot(w)
        rmse[t] = np.sqrt(np.mean((mu - Ytest)**2))
    return rmse, RMSE(full, Xtest, Ytest)


def d(x, pm=-1):
    return np.mean(x, 0) + pm * np.std(x, 0) / np.sqrt(len(x) - 1)


runs, T = 10, 50
data = np.empty((runs, T))
GP = np.empty(runs)
for run in range(runs):
    data[run], GP[run] = CD(run, .05, T=T)

plt.figure(figsize=(6, 4))
plt.axhline(np.mean(GP), c='C0', label='GP')
plt.fill_between((0, T), [d(GP, -1)] * 2, [d(GP, +1)] * 2,
                 color='C0', alpha=.3)
plt.plot(range(1, 1+T), np.mean(data, 0), c='C2', label='BioNN')
plt.fill_between(range(1, 1+T), d(data, -1), d(data, +1),
                 color='C2', alpha=.3)
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.xlim(0, T)
plt.legend()
plt.tight_layout(.05)
plt.savefig('fig/snelson_onlineFull-RMSE.pdf', transparent=True)


# sparse GP
def sparseSGD(run, eta, etaV, T):
    np.random.seed(run)
    idx_train = np.sort(np.random.choice(range(N), N // 2, False))
    idx_test = np.setdiff1d(range(N), idx_train)
    Xtrain = X[idx_train]
    Ytrain = Y[idx_train]
    Xtest = X[idx_test]
    Ytest = Y[idx_test]

    full = GPy.models.GPRegression(Xtrain, Ytrain, GPy.kern.RBF(1))
    full.optimize()
    var = full.Gaussian_noise.variance
    vfe = GPy.models.SparseGPRegression(Xtrain, Ytrain, GPy.kern.RBF(1), num_inducing=6)
    vfe.Gaussian_noise.variance = var
    vfe.optimize()

    """Computation of weights w for mean prediction and weigths w^Sigma & bias
    b^Sigma (wV & bV) for variance prediction using stochastic gradient decent"""
    K_uf = vfe.kern.K(vfe.Z, Xtrain)
    K_uf_test = vfe.kern.K(vfe.Z, Xtest)
    w = np.zeros((6, 1))
    wV, bV = 1, var[0]
    rmse = np.zeros(T)
    nlpd = np.zeros(T)
    for t in range(T):
        for i in range(len(Xtrain)):
            delta = (K_uf.T[i].dot(w) - Ytrain[i])[0]
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
    return (rmse, nlpd), [[RMSE(m, Xtest, Ytest), NLPD(m, Xtest, Ytest)] for m in (full, vfe)]


runs, T = 10, 100
perf = np.empty((runs, 2, T))
GPandVFE = np.empty((runs, 2, 2))
for run in range(runs):
    perf[run], GPandVFE[run] = sparseSGD(run, 1, .001, T)


def plot(typ):
    plt.figure(figsize=(6, 4))
    T = perf.shape[-1]
    j = ('RMSE', 'NLPD').index(typ)
    data = perf[:, j]
    for c, label in ((0, 'GP'), (1, 'VFE')):
        plt.axhline(np.mean(GPandVFE[:, c, j]), c='C{}'.format(c), label=label)
        plt.fill_between((0, T), [d(GPandVFE[:, c, j], -1)] * 2, [d(GPandVFE[:, c, j], +1)] * 2,
                         color='C{}'.format(c), alpha=.3)
    plt.plot(range(1, 1+T), np.mean(data, 0), c='C2', label='BioNN')
    plt.fill_between(range(1, 1+T), d(data, -1), d(data, +1),
                     color='C2', alpha=.3)
    plt.xlabel('Epochs')
    plt.ylabel(typ)
    plt.xlim(0, T)
    plt.legend()
    plt.tight_layout(.05)


plot('RMSE')
plt.savefig('fig/snelson_onlineSparse-RMSE.pdf', transparent=True)
plot('NLPD')
plt.savefig('fig/snelson_onlineSparse-NLPD.pdf', transparent=True)
