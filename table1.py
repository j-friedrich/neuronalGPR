import numpy as np
import os


_UCI_DIRECTORY_PATH = "DropoutUncertaintyExps-master/UCI_Datasets"
subfolders = [f.name for f in os.scandir(_UCI_DIRECTORY_PATH) if f.is_dir()]
subfolders.sort()


print('%-36s' % 'loglikelihood' + 'VFE            ANN           BioNN')
for j, f in enumerate(subfolders):
    print('%29s' % f, end='  ')
    try:
        n_splits = int(np.loadtxt(_UCI_DIRECTORY_PATH + '/' + f + '/data/n_splits.txt'))
        perfs = (-np.load('results/VFE/%s.npy' % f)[:, 1],
                 -np.load('results/ANN/%s.npy' % f)[:, 1],
                 -np.load('results/BioNN/%s.npy' % f)[:, 1])
        means = np.mean(perfs, 1)
        SEMs = np.std(perfs, 1) / np.sqrt(n_splits - 1)
        best = np.isclose(means, means.max())
        for i, (mean, SEM) in enumerate(zip(means, SEMs)):
            print((('\x1b[1;32m' if best[i] else '') + '% .3f+-%.3f' % (mean, SEM) +
                   ('\x1b[0m' if best[i] else '')).replace('nan', ' NA  '), end='  ')
        print()
    except:
        print('ERROR')
