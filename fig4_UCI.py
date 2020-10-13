import matplotlib.pyplot as plt
import numpy as np
import os


plt.rc('font', size=18)
plt.rc('legend', **{'fontsize': 12})
plt.rc('pdf', fonttype=42)

_UCI_DIRECTORY_PATH = "DropoutUncertaintyExps-master/UCI_Datasets"
subfolders = [f.name for f in os.scandir(_UCI_DIRECTORY_PATH) if f.is_dir()]
subfolders.sort()

plt.figure(figsize=(18, 4.5))
for q in (0, 1):
    for i, dataset in enumerate(subfolders[:-1]):
        ax = plt.subplot(2, 9, 9 * q + 1 + i)
        ax.ticklabel_format(scilimits=(-2, 2), useMathText=True)
        try:
            bp = plt.boxplot(np.transpose([np.load('results/VFE/%s.npy' % dataset)[:, q],
                                           np.load('results/BioNN/%s.npy' % dataset)[:, q],
                                           np.load('results/FITC/%s.npy' % dataset)[:, q]]),
                             showfliers=False, showmeans=True, widths=.8,
                             meanprops=dict(marker='o', markersize=10))
            for part in ('boxes', 'whiskers', 'caps', 'means', 'medians'):
                num = len(bp[part])
                for j in range(num):
                    c = 'C{}'.format(j * 3 // num + 1)
                    bp[part][j].set(color=c, markerfacecolor=c,
                                    markeredgecolor=c, linewidth=2)
        except:
            print('Exception: make sure to execute runUCI.py first to generate the results' +
                  'before trying to plot them')
        pos = list(ax.yaxis.offsetText.get_position())
        pos[0] = -.45
        ax.yaxis.offsetText.set_position(pos)
        ax.yaxis.offsetText.set_fontsize(14)
        if not q:
            plt.title(('boston', 'concrete', 'energy', 'kin8nm', 'naval',
                       'power', 'protein', 'wine', 'yacht', 'year')[i])
            plt.xticks((1, 2, 3), (None,) * 3)
        else:
            plt.xticks((1, 2, 3), ('VFE', 'BioNN', 'FITC'), rotation=90)
        if i == 0:
            plt.ylabel(('RMSE', 'NLPD')[q])
plt.tight_layout(0, .5, 0)
plt.savefig('fig/real.pdf')
