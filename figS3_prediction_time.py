import matplotlib.pyplot as plt
import numpy as np
import os

_UCI_DIRECTORY_PATH = "DropoutUncertaintyExps-master/UCI_Datasets/"
subfolders = [f.name for f in os.scandir(_UCI_DIRECTORY_PATH) if f.is_dir()]
subfolders.sort()

plt.figure(figsize=(18, 6))
for j, f in enumerate(subfolders):
    n_splits = int(np.loadtxt(_UCI_DIRECTORY_PATH + '/' + f + '/data/n_splits.txt'))
    plt.subplot(2, 5, j + 1)
    perf = np.load('results/ANN/%s.npy' % f)
    gp = np.load('results/VFE/%s.npy' % f)
    plt.errorbar(perf[:, 5:].min(1).mean(), -perf.mean(0)[1],
                 perf.std(0)[1] / np.sqrt(n_splits - 1),
                 ls='None', marker='x', ms=8, markeredgewidth=3, label='ANN')
    plt.errorbar(perf[:, 5:].min(1).mean(), -gp.mean(0)[1],
                 gp.std(0)[1] / np.sqrt(n_splits - 1),
                 ls='None', marker='+', ms=10, markeredgewidth=3, label='VFE')
    for n_layers in (1, 2):
        perf = np.load('results/PBP/%s.npy' % f)[:, n_layers - 1]
        plt.errorbar(perf[:, -1].min(0), perf.mean(0)[1],
                     perf.std(0)[1] / np.sqrt(n_splits - 1),
                     ls='None', marker='o', label='PBP %g' % n_layers)
    for n_layers in (1, 2):
        perf = np.load('results/Dropout/%s_%glayers.npy' % (f, n_layers))
        plt.errorbar(perf[:, 1:, -1].min(0), perf.mean(0)[1:, 2],
                     perf.std(0)[1:, 2] / np.sqrt(n_splits - 1),
                     marker='o', label='Dropout %g' % n_layers)
    plt.xscale('log')
    if not j:
        plt.legend()
    if j > 4:
        plt.xlabel('Prediction Time [s]')
    if not j % 5:
        plt.ylabel('Loglikelihood')
    plt.title(('boston', 'concrete', 'energy', 'kin8nm', 'naval',
                   'power', 'protein', 'wine', 'yacht', 'year')[j])
plt.tight_layout(0)
plt.savefig('fig/Loglikelihood_vs_time.pdf')
