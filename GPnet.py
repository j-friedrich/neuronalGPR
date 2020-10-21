import GPy
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.linalg.lapack import dpotrf, dpotrs


class Foo:
    pass


class BioNN:
    """Biological plausible neural network (BioNN)

    Parameters
    ----------
    X : np.ndarray (num_data x input_dim)
        inputs
    Y : np.ndarray (num_data x output_dim)
        outputs
    inducing inputs : np.ndarray (num_inducing x input_dim)
    lengthscale : array-like (input_dim)
        lengthscales of the ARD RBF-kernel
    """
    def __init__(self, X, Y, inducing_inputs, lengthscale):
        self.X = X
        self.Y = Y
        self.inducing_inputs = inducing_inputs
        self.lengthscale = lengthscale
        self.rectify = rectify
        self.kern = GPy.kern.RBF(input_dim=X.shape[1],
                                 variance=1., lengthscale=lengthscale, ARD=True)
        K_uf = self.kern.K(inducing_inputs, X)
        KK = K_uf.dot(K_uf.T)
        Ky = K_uf.dot(Y)
        self.w_mean = dpotrs(dpotrf(KK)[0], Ky)[0]  # faster than np.linalg.solve(KK, Ky)
        if np.any(np.isnan(self.w_mean)):
            try:
                self.w_mean = np.linalg.solve(KK, Ky)
            except:
                jitter = np.diag(KK).mean() * 1e-6
                num_tries = 1
                while num_tries <= 5 and np.isfinite(jitter):
                    try:
                        self.w_mean = np.linalg.solve(KK + np.eye(KK.shape[0]) * jitter, Ky)
                    except:
                        jitter *= 10
                    finally:
                        num_tries += 1
        self.wb_var = np.transpose([scipy.optimize.nnls(
            np.vstack([self._SNRinv(X), np.ones(len(X))]).T,
            (self.mean(X) - Y)[:, a]**2)[0] for a in range(Y.shape[1])])
        self.Gaussian_noise = Foo()
        self.variance, self.Gaussian_noise.variance = self.wb_var

    def mean(self, x):
        return self.kern.K(np.atleast_2d(x), self.inducing_inputs).dot(self.w_mean)

    def _SNRinv(self, x):
        return np.maximum(1 - np.sum(self.kern.K(self.inducing_inputs, np.atleast_2d(x))**2, 0), 0)

    def var(self, x):
        return np.maximum(np.vstack([self._SNRinv(x), np.ones(len(np.atleast_2d(x)))]).T.dot(
            self.wb_var), 1e-6)

    def predict(self, x):
        """Predict the function(s) at the new point(s) x. This includes the
           likelihood variance added to the predicted underlying function
           (usually referred to as f)."""
        return self.mean(x), self.var(x)

    def plot(self, ax=None):
        """Convenience function for plotting the fit of a BioNN"""
        if ax is None:
            plt.figure(figsize=(6.7, 4.6))
        xmin, xmax = self.X.min(0), self.X.max(0)
        xmin, xmax = xmin - 0.25 * (xmax - xmin), xmax + 0.25 * (xmax - xmin)
        test = np.linspace(xmin, xmax, 200)
        plt.scatter(self.X, self.Y, marker='x', c='k')
        plt.plot(test, self.mean(test), lw=2)
        plt.fill_between(test.ravel(), (self.mean(test) - 2 * np.sqrt(self.var(test))).ravel(),
                         (self.mean(test) + 2 * np.sqrt(self.var(test))).ravel(), alpha=.15)
        GPy.plotting.matplot_dep.plot_definitions.MatplotlibPlots().plot_axis_lines(
            plt.gca(), self.inducing_inputs, s=450)


def logpdf(d, v):
    return -np.log(2 * math.pi * v) / 2 - d**2 / 2 / v


def smse(Ypredicted, Y):
    return np.mean((Ypredicted - Y)**2 / np.var(Y, 0))


def RMSE(m, X, Y):
    """Root mean square error between the predictions of
       model m on input data X and the true labels Y"""
    assert(X.ndim == 2)
    assert(Y.ndim == 2)
    return np.sqrt(np.mean((m.predict(X)[0] - Y)**2))


def NLPD(m, X, Y):
    """Negative log predictive desity for model m to predict Y on input data X"""
    assert(X.ndim == 2)
    assert(Y.ndim == 2)
    return - logpdf(Y - m.predict(X)[0], m.predict(X)[1]).mean()


def SMSE(m, X, Y):
    """Standardized mean square error between the predictions of
       model m on input data X and the true labels Y"""
    assert(X.ndim == 2)
    assert(Y.ndim == 2)
    return smse(m.predict(X)[0], Y)


def KL(m0, m1, Xtest):
    """Caclulate the KL divergence between model m0 and model m1 on input data Xtest"""
    def predict(m):
        try:
            mu, sig = m.predict(Xtest, full_cov=True)
        except:
            mu, sig = m.predict(Xtest)
            sig = np.diag(sig.ravel())
        return mu, sig
    mu0, sig0 = predict(m0)
    mu1, sig1 = predict(m1)
    sign0, logdet0 = np.linalg.slogdet(sig0)
    sign1, logdet1 = np.linalg.slogdet(sig1)
    if sign0 != 1:
        sig0 += 1e-12 * np.eye(len(sig0))
        sign0, logdet0 = np.linalg.slogdet(sig0)
    if sign1 != 1:
        sig1 += 1e-12 * np.eye(len(sig1))
        sign1, logdet1 = np.linalg.slogdet(sig1)
    return (np.trace(np.linalg.solve(sig1, sig0)) +
            (mu1 - mu0).ravel().dot(np.linalg.solve(sig1, mu1 - mu0).ravel()) -
            len(mu0) + logdet1 - logdet0) / 2
