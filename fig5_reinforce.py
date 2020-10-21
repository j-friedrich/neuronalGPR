from GPnet import logpdf, SMSE
import GPy
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
from scipy.linalg.lapack import dpotrf, dpotrs
from scipy.optimize import minimize


# create results and figure directories if not existent yet
os.makedirs('fig', exist_ok=True)
os.makedirs('results', exist_ok=True)

plt.rc('font', size=18)
plt.rc('legend', **{'fontsize': 12})
plt.rc('lines', linewidth=2.5)
plt.rc('pdf', fonttype=42)


# definitions
class Foo:
    pass


class BioNN:
    """Biological plausible neural network (BioNN)

    Similar to the one defined in GPnet.py, but supporting more flexible tuning functions

    Parameters
    ----------
    X : np.ndarray (num_data x input_dim)
        inputs
    Y : np.ndarray (num_data x output_dim)
        outputs
    tuning : function
        tuning function phi(x) returning activity phi of all neurons as function of input x
    """
    def __init__(self, X, Y, tuning):
        self.X = X
        self.Y = Y
        self.tuning = tuning
        self.K_uf = tuning(X).T
        KK = self.K_uf.dot(self.K_uf.T)
        Ky = self.K_uf.dot(Y)
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
        return self.tuning(np.atleast_2d(x)).dot(self.w_mean)

    def _SNRinv(self, x):
        return np.maximum(1 - np.sum(self.tuning(np.atleast_2d(x))**2, 1), 0)

    def var(self, x):
        return np.vstack([self._SNRinv(x), np.ones(len(np.atleast_2d(x)))]).T.dot(self.wb_var)

    def predict(self, x):
        """Predict the function(s) at the new point(s) x. This includes the
           likelihood variance added to the predicted underlying function
           (usually referred to as f)."""
        return self.mean(x), self.var(x)

    @property
    def inducing_inputs(self):
        init = self.X[np.argmax(self.K_uf, 1)]
        return np.array([minimize(lambda x: -self.tuning(np.array([x]))[0][i],
                                  init[i]).x for i in range(len(init))])

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


def reinforce(Xtrain, Ytrain, Xtest, Ytest, T=1100, eta_u=.2, eta_l=0, eta_r=.1,
              kern=None, return_nn=False):
    l = kern.lengthscale[0] * np.ones(6)
    u = np.linspace(.5, 5.5, 6)[:, None]
    r0 = -SMSE(BioNN(Xtrain, Ytrain, lambda x: kern.K(x, u)), Xtrain, Ytrain)
    perf = np.empty((T, 2))
    for t in range(T):
        perturb_u = eta_u * np.random.randn(*u.shape)
        if eta_l == 0:
            def tuning(x): return kern.K(x, u)

            def tuningP(x): return kern.K(x, u + perturb_u)
        else:
            perturb_l = eta_l * np.random.randn(6)

            def tuning(x):
                return np.transpose([GPy.kern.RBF(1, lengthscale=l[i]).K(
                    x, u[i:i + 1]) for i in range(6)])[0]

            def tuningP(x):
                return np.transpose([GPy.kern.RBF(1, lengthscale=(l + perturb_l)[i]).K(
                    x, (u + perturb_u)[i:i + 1]) for i in range(6)])[0]
        nn = BioNN(Xtrain, Ytrain, tuning)
        nnp = BioNN(Xtrain, Ytrain, tuningP)
        m, v = nn.predict(Xtest)
        perf[t] = np.sqrt(np.mean((Ytest - m)**2)), -logpdf(Ytest - m, v).mean()
        delta = -SMSE(nnp, Xtrain, Ytrain) - r0
        u += delta * perturb_u
        if eta_l != 0:
            l += delta * perturb_l
        r0 += eta_r * delta
    if return_nn:
        return perf, u, l, nn
    else:
        return perf, u, l


def sim(run, eta_u, eta_l, eta_r, return_nn=False):
    np.random.seed(run)
    idx_train = np.sort(np.random.choice(range(N), N // 2, False))
    idx_test = np.sort(np.setdiff1d(range(N), idx_train))
    Xtrain = X[idx_train]
    Ytrain = Y[idx_train]
    Xtest = X[idx_test]
    Ytest = Y[idx_test]
    vfe = GPy.models.SparseGPRegression(Xtrain, Ytrain, GPy.kern.RBF(1), num_inducing=6)
    vfe.optimize()
    kern = GPy.kern.RBF(1, lengthscale=vfe.kern.lengthscale.values)
    return reinforce(Xtrain, Ytrain, Xtest, Ytest, 1100, eta_u, eta_l, eta_r, kern, return_nn)


if __name__ == "__main__":
    # Ed Snelson's example data
    X = np.genfromtxt('snelson_data/train_inputs')[:, None]
    Y = np.genfromtxt('snelson_data/train_outputs')[:, None]
    N = len(X)


    # Run Reinforce or load results
    try:
        opt_z = np.load('results/reinforce.npz', allow_pickle=True)['opt_z']
        opt_zl = np.load('results/reinforce.npz', allow_pickle=True)['opt_zl']
    except FileNotFoundError:
        opt_z = np.array([sim(run, .12, .0, .1) for run in range(10)])
        opt_zl = np.array([sim(run, .1, .1, .1) for run in range(10)])
        np.savez('results/reinforce.npz', opt_z=opt_z, opt_zl=opt_zl)


    # Plot results

    def plot(typ, opt_z, opt_zl, T=1000):
        j = ('RMSE', 'NLPD').index(typ)
        try:
            perf = np.load('results/performance_snelson.npz')['perf'][:, 3, :, 1 + j]
        except FileNotFoundError:
            print('please run fig3_snelson.py first')

        def d(x, pm=-1):
            return np.mean(x, 0) + pm * np.std(x, 0) / np.sqrt(len(x) - 1)

        for c, label in ((0, 'GP'), (1, 'VFE')):
            plt.axhline(np.mean(perf[:, c]), c='C{}'.format(c), label=label)
            plt.fill_between((0, T), [d(perf[:, c], -1)] * 2, [d(perf[:, c], +1)] * 2,
                             color='C{}'.format(c), alpha=.3)

        for k, data in enumerate((opt_z[:, :, j], opt_zl[:, :, j])):
            plt.plot(np.mean(data, 0), c='C{}'.format(4 + k),
                     label=(r'BioNN optimize $z$', r'BioNN optimize $z$ & $l$')[k])
            plt.fill_between(range(data.shape[1]), d(data, -1), d(data, +1),
                             color='C{}'.format(4 + k), alpha=.3)

        plt.xlabel('Iterations')
        plt.ylabel(typ)
        plt.xlim(0, T)
        plt.legend()


    for typ in ('RMSE', 'NLPD'):
        plt.figure(figsize=(4.5, 4))
        plot(typ, np.array([o[0] for o in opt_z]), np.array([o[0] for o in opt_zl]), T=1000)
        plt.tight_layout(.05)
        plt.savefig('fig/reinforce-' + typ + '.pdf', transparent=True)


    # Plot fit for an example 50:50 train/test split
    np.random.seed(10)
    idx_train = np.sort(np.random.choice(range(N), N // 2, False))
    idx_test = np.sort(np.setdiff1d(range(N), idx_train))
    Xtrain = X[idx_train]
    Ytrain = Y[idx_train]
    Xtest = X[idx_test]
    Ytest = Y[idx_test]

    # treat full GP on full data as ground truth
    truth = GPy.models.GPRegression(X, Y, GPy.kern.RBF(1))
    truth.optimize()

    full = GPy.models.GPRegression(Xtrain, Ytrain, GPy.kern.RBF(1))
    full.optimize()
    fitc = GPy.models.SparseGPRegression(Xtrain, Ytrain, GPy.kern.RBF(1), num_inducing=6)
    fitc.inference_method = GPy.inference.latent_function_inference.FITC()
    fitc.optimize()
    vfe = GPy.models.SparseGPRegression(Xtrain, Ytrain, GPy.kern.RBF(1), Z=fitc.inducing_inputs)
    vfe.Gaussian_noise.variance = full.Gaussian_noise.variance.values
    vfe.optimize()


    kern = GPy.kern.RBF(1, lengthscale=vfe.kern.lengthscale.values)
    nn_vfe = BioNN(Xtrain, Ytrain, lambda x: kern.K(x, vfe.Z))
    nn_z = sim(10, .12, .0, .1, True)[-1]
    nn_zl = sim(10, .1, .1, .1, True)[-1]


    plt.rc('lines', linewidth=1.5)

    plt.figure(figsize=(9, 4))
    xmin, xmax = X.min(0), X.max(0)
    test = np.linspace(xmin - .05 * (xmax - xmin), xmax + .05 * (xmax - xmin), 200)
    plt.scatter(Xtrain, Ytrain, marker='x', c='k', s=18, label='Training Data')
    plt.scatter(Xtest, Ytest, marker='x', c='r', s=18, label='Test Data')
    for m, label, c in ((nn_vfe, 'BioNN', 'C2'), (nn_z, 'BioNN optimize $z$', 'C4'),
                        (nn_zl, 'BioNN optimize $z$ & $l$', 'C5'), (truth, '"Truth"', 'k')):
        plt.plot(test, m.predict(test)[0], c=c, lw=2.5, label=label, zorder=-10 if c == 'k' else 0)
        for k in (-2, 2):
            plt.plot(test, m.predict(test)[0] +
                     k * np.sqrt(m.predict(test)[1]), c=c, lw=1.5, zorder=-10 if c == 'k' else 0)
    vfe.plot_inducing(ax=plt.gca(), label=None, color='C2', s=700)
    tmp = GPy.models.SparseGPRegression(X, Y, GPy.kern.RBF(1), Z=nn_z.inducing_inputs)
    tmp.plot_inducing(ax=plt.gca(), label=None, color='C4', s=500)
    tmp = GPy.models.SparseGPRegression(X, Y, GPy.kern.RBF(1), Z=nn_zl.inducing_inputs)
    tmp.plot_inducing(ax=plt.gca(), label=None, color='C5', s=300)
    plt.xlim(test[0], test[-1])
    plt.yticks([])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(ncol=2, loc=(.36, .08))
    plt.tight_layout(.05)
    plt.savefig('fig/reinforce-fit.pdf')
