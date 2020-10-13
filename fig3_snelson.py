from GPnet import BioNN, RMSE, NLPD, KL
import GPy
import matplotlib.pyplot as plt
import numpy as np
import os


# create results and figure directories if not existent yet
os.makedirs('fig', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Ed Snelson's example data
X = np.genfromtxt('snelson_data/train_inputs')[:, None]
Y = np.genfromtxt('snelson_data/train_outputs')[:, None]
N = len(X)

# treat full GP on full data as ground truth
truth = GPy.models.GPRegression(X, Y, GPy.kern.RBF(1))
truth.optimize()


# Vary number of inducing points,  50:50 train/test split
inducing = list(range(3, 16))
runs = 10
try:
    perf = np.load('results/performance_snelson.npz')['perf']
    kl = np.load('results/performance_snelson.npz')['kl']
except FileNotFoundError:
    perf = np.empty((runs, len(inducing), 4, 3))
    kl = np.empty((runs, len(inducing), 3))

    for run in range(runs):
        np.random.seed(run)
        idx_train = np.sort(np.random.choice(range(N), N // 2, False))
        idx_test = np.setdiff1d(range(N), idx_train)
        Xtrain = X[idx_train]
        Ytrain = Y[idx_train]
        Xtest = X[idx_test]
        Ytest = Y[idx_test]

        full = GPy.models.GPRegression(Xtrain, Ytrain, GPy.kern.RBF(1))
        full.optimize()
        for n, num_inducing in enumerate(inducing):
            fitc = GPy.models.SparseGPRegression(
                Xtrain, Ytrain, GPy.kern.RBF(1), num_inducing=num_inducing)
            fitc.inference_method = GPy.inference.latent_function_inference.FITC()
            fitc.optimize()
            vfe = GPy.models.SparseGPRegression(
                Xtrain, Ytrain, GPy.kern.RBF(1), Z=fitc.inducing_inputs)
            vfe.Gaussian_noise.variance = full.Gaussian_noise.variance.values
            vfe.optimize()
            nn = BioNN(Xtrain, Ytrain, vfe.inducing_inputs, vfe.kern.lengthscale)

            perf[run, n] = [[m.Gaussian_noise.variance[0],
                             RMSE(m, Xtest, Ytest), NLPD(m, Xtest, Ytest)]
                            for m in (full, vfe, nn, fitc)]
            kl[run, n] = [KL(full, m, Xtest) for m in (vfe, nn, fitc)]


# Vary noise level,  50:50 train/test split
res = Y - truth.predict(X)[0]
runs = 10
noises = (.01, .1, 1, 10, 100)
try:
    perfN = np.load('results/performance_snelson.npz')['perfN']
except FileNotFoundError:
    perfN = np.zeros((runs, len(noises), 4, 3))

    for run in range(runs):
        np.random.seed(run)
        idx_train = np.sort(np.random.choice(range(N), N // 2, False))
        idx_test = np.setdiff1d(range(N), idx_train)
        Xtrain = X[idx_train]
        Xtest = X[idx_test]
        for j, noise in enumerate(np.sqrt(noises)):
            Ytrain = truth.predict(Xtrain)[0] + noise * res[idx_train]
            Ytest = truth.predict(Xtest)[0] + noise * res[idx_test]

            full = GPy.models.GPRegression(Xtrain, Ytrain, GPy.kern.RBF(1))
            full.optimize()
            fitc = GPy.models.SparseGPRegression(Xtrain, Ytrain, GPy.kern.RBF(1), num_inducing=6)
            fitc.Gaussian_noise.variance = full.Gaussian_noise.variance.values
            fitc.inference_method = GPy.inference.latent_function_inference.FITC()
            fitc.optimize()
            vfe = GPy.models.SparseGPRegression(Xtrain, Ytrain, GPy.kern.RBF(1), num_inducing=6)
            vfe.Gaussian_noise.variance = full.Gaussian_noise.variance.values
            vfe.optimize()
            nn = BioNN(Xtrain, Ytrain, vfe.inducing_inputs, vfe.kern.lengthscale)

            perfN[run, j] = [[m.Gaussian_noise.variance[0],
                              RMSE(m, Xtest, Ytest), NLPD(m, Xtest, Ytest)]
                             for m in (full, vfe, nn, fitc)]

    np.savez('results/performance_snelson.npz', perf=perf, kl=kl, perfN=perfN)


# plot
np.random.seed(10)
idx_train = np.sort(np.random.choice(range(N), N // 2, False))
idx_test = np.setdiff1d(range(N), idx_train)
Xtrain = X[idx_train]
Ytrain = Y[idx_train]
Xtest = X[idx_test]
Ytest = Y[idx_test]

full = GPy.models.GPRegression(Xtrain, Ytrain, GPy.kern.RBF(1))
full.optimize()
fitc = GPy.models.SparseGPRegression(Xtrain, Ytrain, GPy.kern.RBF(1), num_inducing=6)
fitc.inference_method = GPy.inference.latent_function_inference.FITC()
fitc.optimize()
vfe = GPy.models.SparseGPRegression(Xtrain, Ytrain, GPy.kern.RBF(1), Z=fitc.inducing_inputs)
vfe.Gaussian_noise.variance = full.Gaussian_noise.variance.values
vfe.optimize()
nn = BioNN(Xtrain, Ytrain, vfe.inducing_inputs, vfe.kern.lengthscale)


plt.rc('font', size=18)
plt.rc('legend', **{'fontsize': 12})
fig = plt.figure(figsize=(18, 8))
fig.add_axes([.045, .595, .44, .4])
xmin, xmax = X.min(0), X.max(0)
test = np.linspace(xmin - .05 * (xmax - xmin), xmax + .05 * (xmax - xmin), 200)
plt.scatter(Xtrain, Ytrain, marker='x', c='k', s=18, label='Training Data')
plt.scatter(Xtest, Ytest, marker='x', c='r', s=18, label='Test Data')
for m, label, c in ((full, 'GP', 'C0'), (vfe, 'VFE', 'C1'), (nn, 'BioNN', 'C2')):
    plt.plot(test, m.predict(test)[0], c=c, lw=2.5, label=label)
    for k in (-2, 2):
        plt.plot(test, m.predict(test)[0] +
                 k * np.sqrt(m.predict(test)[1]), c=c, lw=1.5)
vfe.plot_inducing(ax=plt.gca())
plt.xlim(test[0], test[-1])
plt.xticks([])
plt.yticks([])
plt.xlabel('x')
plt.ylabel('y')
plt.legend(ncol=2, loc=(.36, .05))

for j in (0, 2):
    fig.add_axes([(.555, .805)[j // 2], .595, .19, .4])
    for i in (0, 1, 2, 3):
        bp = plt.boxplot((np.log(perfN[:, :, i, j] /
                                 (np.array(noises) * truth.Gaussian_noise.variance)) /
                          np.log(10)) if j == 0 else perfN[:, :, i, j],
                         boxprops={'color': 'C{}'.format(i)},
                         flierprops={'markeredgecolor': 'C{}'.format(i)},
                         whiskerprops={'color': 'C{}'.format(i)},
                         medianprops={'color': 'C{}'.format(i), 'linewidth': 3},
                         capprops={'color': 'C{}'.format(i)},
                         showfliers=False, widths=.7 - .15 * i)
    plt.xlabel("True Noise Variance")
    plt.ylabel(["Normalized Noise Variance  ", "RMSE", "NLPD"][j])
    plt.xticks(range(1, 1 + len(noises)), noises)
    if j == 0:
        lim = plt.ylim()
        for c, label in (('C0', 'GP'), ('C1', 'VFE'), ('C3', 'FITC'), ('C2', 'BioNN')):
            plt.axhline(-1000, lw=2, c=c, label=label)
        plt.ylim(-1.5, lim[1])
        plt.yticks([-1, 0, 1], [0.1, 1, 10])
        plt.legend()

for j in (0, 2, 3):
    fig.add_axes([(.045, .378, .384, .72)[j], .09, .275, .4])
    if j < 3:
        plt.axhline(np.median(perf[:, 0, 0, j]), lw=2, c='C0', label='GP')
        plt.fill_between((0, 14),
                         [np.percentile(perf[:, 0, 0, j], 25)] * 2,
                         [np.percentile(perf[:, 0, 0, j], 75)] * 2, color='C0', alpha=.3)
    for i in (1, 2, 3):
        bp = plt.boxplot(perf[:, :, i, j] if j < 3 else kl[:, :, i - 1],
                         boxprops={'color': 'C{}'.format(i)},
                         flierprops={'markeredgecolor': 'C{}'.format(i)},
                         whiskerprops={'color': 'C{}'.format(i)},
                         medianprops={'color': 'C{}'.format(i), 'linewidth': 3},
                         capprops={'color': 'C{}'.format(i)}, showfliers=False,
                         widths=.7 - .15 * i, patch_artist=True)

        for box in bp['boxes']:
            # change fill color
            box.set(facecolor=(1, 1, 1, 0))
    plt.xlabel("# inducing points")
    plt.ylabel(["Noise Variance", "RMSE", "NLPD", "KL(p|q)"][j])
    plt.xticks(range(1, 1 + len(inducing)), inducing)
    plt.xlim(.5, 13.5)
    if j == 3:
        plt.ylim(-3, 75)
plt.savefig('fig/snelson.pdf')
