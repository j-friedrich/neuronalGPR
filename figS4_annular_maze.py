import numpy as np
import matplotlib.pyplot as plt
import GPy
from GPnet import RMSE, NLPD
from fig5_reinforce import BioNN


# data
M = 20

noise = .05
N1 = 64
N2 = 72

np.random.seed(0)
angle = np.concatenate([np.arange(N1) / N1 * 2.5 / 6 * 2 * np.pi,
                        np.arange(N2) / N2 * 1. / 6 * 2 * np.pi + 2.5 / 6 * 2 * np.pi,
                        np.arange(N1) / N1 * 2.5 / 6 * 2 * np.pi + 3.5 / 6 * 2 * np.pi])
X = np.transpose([np.cos(angle), np.sin(angle)])
N = len(X)

F = 0 + np.exp(-np.arange(-N // 2, N // 2)**2 * 100. / N**2)[:, None]
F = 0 + np.minimum(1.5 * np.exp(-np.abs(np.arange(-N // 2, N // 2)) * 5 / N)[:, None], 1)
F -= F.mean()
Y = F + np.random.randn(N, 1) * noise


# full GP, VFE and BioNN with tuning curves from VFE
full = GPy.models.GPRegression(X, Y, GPy.kern.RBF(2))
full.optimize()

ang = np.arange(M) * 2 * np.pi / M
vfe = GPy.models.SparseGPRegression(
    X, Y, full.kern.copy(), Z=np.transpose([np.cos(ang), np.sin(ang)]))
vfe.Gaussian_noise.variance = noise**2
vfe.optimize()

nn_vfe = BioNN(X, Y, lambda x: vfe.kern.K(x, vfe.Z))


# optimize positions of inducing points using REINFORCE gradient estimates
kern = GPy.kern.RBF(2, lengthscale=.4)
np.random.seed(0)
T = 10000
eta_u = .1
eta_r = .1
ang = np.arange(M) * 2 * np.pi / M
u = np.transpose([np.cos(ang), np.sin(ang)])
nn = BioNN(X, Y, lambda x: kern.K(x, u))
r0 = -np.sum((nn.mean(X) - Y)**2)
for t in range(T):
    perturb_u = eta_u * np.random.randn(*u.shape)
    nn = BioNN(X, Y, lambda x: kern.K(x, u + perturb_u))
    delta = -np.sum((nn.mean(X) - Y)**2) - r0
    u += delta * perturb_u
    r0 += eta_r * delta
ang = np.arctan2(*u.T[::-1])
u = np.transpose([np.cos(ang), np.sin(ang)])
nn = BioNN(X, Y, lambda x: kern.K(x, u))


# plot
segment = (angle - 2 * np.pi / 12) % (2 * np.pi) * 6 // (2 * np.pi) - 2

fig = plt.figure(figsize=(1.8, 4))
fig.add_axes([.4, .16, .54, .835])
plt.scatter(range(-2, 4), np.histogram(segment, np.arange(-2.5, 4.5))[0], c='k')
plt.axhline(N / 6., c='k')
plt.ylim(0, 87)
plt.xlim(-3, 4)
plt.xticks([0, 3])
plt.xlabel('Segment no', x=.34)
plt.ylabel('Number of data points')
plt.savefig('fig/maze_time.pdf')


Longall = np.linspace(-np.pi, np.pi, 360)
Xall = np.transpose([np.sin(Longall), np.cos(Longall)])

R = np.array([[np.cos(-np.pi / 3), np.sin(-np.pi / 3)], [-np.sin(-np.pi / 3), np.cos(-np.pi / 3)]])

plt.figure(figsize=(4, 4))
for i in range(4):
    ax = plt.subplot(2, 2, 1 + i)
    ax.scatter(*R.dot(Xall.T), c=vfe.kern.K(Xall, vfe.Z)[:, 3 * i], s=120, cmap='hot')
    ax.scatter(*R.dot(vfe.Z.T), c=[[0, 1, 0]])
    ax.scatter(*R.dot(vfe.Z[3 * i]), c=[[0, 0, 1]])
    plt.xticks([])
    plt.yticks([])
plt.suptitle('VFE', y=.08)
plt.tight_layout(0, rect=(0, .08, 1, 1))
plt.savefig('fig/maze_Kvfe.pdf')


plt.figure(figsize=(4, 4))
for i in range(4):
    ax = plt.subplot(2, 2, 1 + i)
    ax.scatter(*R.dot(Xall.T), c=nn.tuning(Xall)[:, 3 * i], s=120, cmap='hot')
    ax.scatter(*R.dot(nn.inducing_inputs.T), c=[[0, 1, 0]])
    ax.scatter(*R.dot(nn.inducing_inputs[3 * i]), c=[[0, 0, 1]])
    plt.xticks([])
    plt.yticks([])
plt.tight_layout(0, rect=(0, .08, 1, 1))
plt.suptitle('BioNN optimize z', y=.08)
plt.savefig('fig/maze_Kopt.pdf')


angle_nn = np.arctan2(*nn.inducing_inputs.T[::-1])
segment_nn = (angle_nn - 2 * np.pi / 12) % (2 * np.pi) * 6 // (2 * np.pi) - 2
angle_vfe = np.arctan2(*vfe.Z.T[::-1])
segment_vfe = (angle_vfe - 2 * np.pi / 12) % (2 * np.pi) * 6 // (2 * np.pi) - 2
plt.figure(figsize=(4, 4))
plt.hist((segment_vfe, segment_nn), np.arange(-2.5, 4),
         density=True, color=('C1', 'C4'), label=('VFE', 'BioNN opt. z'))
plt.axhline(1 / 6., c='k')
plt.xticks(range(-2, 4))
plt.yticks([0, .1, .2, .3], [0, 10, 20, 30])
plt.xlabel('Segment no')
plt.ylabel('Percentage of cells')
plt.legend()
plt.tight_layout(0)
plt.savefig('fig/maze_count.pdf')


idx = np.concatenate([np.arange(N), np.arange(N // 10)])
angle2 = np.concatenate([angle, angle[:N // 10] + 2 * np.pi])
plt.figure(figsize=(6, 4))
plt.scatter(angle2, Y[idx], label='data', c='k', s=12)
for c, m in enumerate((full, vfe, nn_vfe, nn)):
    plt.plot(angle2, m.predict(X)[0][idx], '--' if c == 1 else '-',
             c='C{}'.format(c if c < 3 else 4),
             label=('GP', 'VFE', 'BioNN', 'BioNN opt. z')[c], lw=3)
    for i in (-2, 2):
        plt.plot(angle2, m.predict(X)[0][idx] + i * np.sqrt(m.predict(X)[1][idx]),
                 '--' if c == 1 else '-',
                 c='C{}'.format(c if c < 3 else 4), lw=1.5)
plt.plot(angle2, F[idx], c='k', label='Truth', lw=3, zorder=-5)
for (m, c, s) in ((vfe, 'C2', 700), (nn, 'C4', 500)):
    GPy.plotting.matplot_dep.plot_definitions.MatplotlibPlots().plot_axis_lines(
        plt.gca(), np.pi / 6 + (
            (np.arctan2(*m.inducing_inputs.T[::-1])[:, None] - np.pi / 6) % (2 * np.pi)),
        s=s, color=c, label=None)
for i in range(1, 12, 2):
    plt.axvline(i / 6 * np.pi, lw=1, c='gray')
    plt.text((i + .25) / 6 * np.pi, .63, i // 2 - 2, c='gray')
plt.yticks([])
plt.xticks(np.arange(5) / 2 * np.pi, ['-180°', '-90°', '0°', '90°', '180°'])
plt.xlabel('Angular position')
plt.ylabel('Value')
plt.xlim(np.pi / 6, 13 / 6 * np.pi)
plt.legend(loc=(.69, .45))
plt.tight_layout(0)
plt.savefig('fig/maze_fit.pdf')


# predictive performance using 50:50 train/test splits

def performance(run):
    np.random.seed(run)
    idx_train = np.sort(np.random.choice(range(N), N // 2, False))
    idx_test = np.sort(np.setdiff1d(range(N), idx_train))
    Xtrain = X[idx_train]
    Ytrain = Y[idx_train]
    Xtest = X[idx_test]
    Ytest = Y[idx_test]

    full = GPy.models.GPRegression(Xtrain, Ytrain, GPy.kern.RBF(2))
    full.optimize()

    ang = np.arange(M) * 2 * np.pi / M
    vfe = GPy.models.SparseGPRegression(
        Xtrain, Ytrain, full.kern.copy(), Z=np.transpose([np.cos(ang), np.sin(ang)]))
    vfe.Gaussian_noise.variance = noise**2
    vfe.optimize()

    nn_vfe = BioNN(Xtrain, Ytrain, lambda x: vfe.kern.K(x, vfe.Z))

    kern = GPy.kern.RBF(2, lengthscale=.4)
    np.random.seed(0)
    T = 10000
    eta_u = .1
    eta_r = .1
    ang = np.arange(M) * 2 * np.pi / M
    u = np.transpose([np.cos(ang), np.sin(ang)])
    nn = BioNN(Xtrain, Ytrain, lambda x: kern.K(x, u))
    r0 = -np.sum((nn.mean(Xtrain) - Ytrain)**2)
    for t in range(T):
        perturb_u = eta_u * np.random.randn(*u.shape)
        nn = BioNN(Xtrain, Ytrain, lambda x: kern.K(x, u + perturb_u))
        delta = -np.sum((nn.mean(Xtrain) - Ytrain)**2) - r0
        u += delta * perturb_u
        r0 += eta_r * delta
    ang = np.arctan2(*u.T[::-1])
    u = np.transpose([np.cos(ang), np.sin(ang)])
    nn = BioNN(Xtrain, Ytrain, lambda x: kern.K(x, u))
    return [[f(m, Xtest, Ytest) for f in (RMSE, NLPD)]
            for m in (full, vfe, nn_vfe, nn)]


perf = np.array([performance(run) for run in range(10)])

print(' ' * 12 + 'GP            VFE           BioNN       BioNN opt. z')
for i, metric in enumerate(('RMSE', 'NLPD')):
    print(metric, end='  ')
    for k in range(4):
        print('%6.3f+-%2.3f' % (perf.mean(0)[k, i],
                                perf.std(0)[k, i] / np.sqrt(len(perf) - 1)), end='  ')
    print()
