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


# Full GP
kernel = GPy.kern.RBF(1)
full = GPy.models.GPRegression(X, Y, kernel)
full.optimize()

var = full.Gaussian_noise.variance
K_ff = kernel.K(X)


# computation of w using coordinate decent
def CD(eta, iters):
    w = np.zeros((N, 1))
    rmse = np.zeros(iters)
    for t in range(iters):
        for i in range(N):
            pre = (K_ff[i] >= kernel.variance.values)[:, None]
            post = (K_ff[i].dot(w) - Y[i]) * np.ones((N, 1))
            w -= eta * pre * (post + var * w)
        mu = K_ff.dot(w)
        rmse[t] = np.sqrt(np.mean((mu - Y)**2))
    return rmse


def plot(data, typ='RMSE', vfe=None):
    plt.figure(figsize=(6, 4))
    iters = len(data)
    plt.xlabel('Epochs')
    plt.ylabel(typ)
    fun = dict(RMSE=RMSE, NLPD=NLPD)[typ]  # fun = eval(typ)
    plt.axhline(fun(full, X, Y), label='GP')
    if vfe is not None:
        plt.axhline(fun(vfe, X, Y), c='C1', label='VFE')
    plt.plot(range(1, 1 + iters), data, c='C2', label='BioNN')
    plt.xlim(0, iters)
    plt.legend()
    plt.tight_layout(.05)


rmse = CD(.04, 50)
plot(rmse)
plt.savefig('fig/snelson_onlineFull-RMSE.pdf', transparent=True)


# sparse GP
np.random.seed(0)
kernel = GPy.kern.RBF(1)
vfe = GPy.models.SparseGPRegression(X, Y, kernel, num_inducing=6)
vfe.Gaussian_noise.variance = full.Gaussian_noise.variance.values
vfe.optimize_restarts()

K_uf = kernel.K(vfe.Z, X)


def sparseSGD(eta, etaV, iters):
    """Computation of weights w for mean prediction and weigths w^Sigma & bias
    b^Sigma (wV & bV) for variance prediction using stochastic gradient decent"""
    w = np.zeros((6, 1))
    wV, bV = 1, var[0]
    rmse = np.zeros(iters)
    nlpd = np.zeros(iters)
    for t in range(iters):
        for i in range(N):
            delta = (K_uf.T[i].dot(w) - Y[i])[0]
            w -= eta * K_uf[:, i:i + 1] * delta
            rho = np.maximum(1 - np.sum(K_uf[:, i]**2, 0), 0)
            deltaV = wV * rho + bV - delta**2
            wV -= etaV * rho * deltaV
            bV -= etaV * deltaV
        mu = K_uf.T.dot(w)
        Sigma = wV * rho + bV
        rmse[t] = np.sqrt(np.mean((mu - Y)**2))
        nlpd[t] = - logpdf(mu - Y, Sigma).mean()
    return rmse, nlpd


rmse, nlpd = sparseSGD(.4, .15, 200)
plot(rmse, vfe=vfe)
plt.savefig('fig/snelson_onlineSparse-RMSE.pdf', transparent=True)
plot(nlpd, 'NLPD', vfe=vfe)
plt.savefig('fig/snelson_onlineSparse-NLPD.pdf', transparent=True)
