"""Scalability figure: (a) projection runtime vs. cells, (b) violation vs. sweeps. Writes paper/figs/scale.pdf."""
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from paths import RES
from paths import FIGS as FIG
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
S1, S2, S3 = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.6, 'legend.frameon': False})
d = pd.read_csv(os.path.join(RES, 'bench_scaling.csv'))
c = pd.read_csv(os.path.join(RES, 'bench_convergence.csv'))
fig, axs = plt.subplots(1, 2, figsize=(3.4, 1.75), gridspec_kw=dict(wspace=0.45))
ax = axs[0]
style = {'GPU (ours)': (S1, '-', 'o'), 'CPU numpy': (S2, '-', 's'), 'SCS (exp. cone)': (S3, (0, (3, 1.5)), '^'), 'OSQP': (S3, (0, (3, 1.5)), '^')}
for m in ['GPU (ours)', 'CPU numpy', 'SCS (exp. cone)']:
    g = d[(d.geometry == 'kl') & (d.method == m) & d.sec.notna()]
    col, ls, mk = style[m]
    ax.plot(g.cells, g.sec, color=col, ls=ls, marker=mk, ms=3, lw=1.0, label=m.replace(' (exp. cone)', ''))
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('cells'); ax.set_ylabel('seconds (50 sweeps)')
ax.set_title('(a) KL projection', fontsize=8, loc='left')
ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.legend(fontsize=6, loc='lower right', handlelength=1.4)
ax = axs[1]
for nv, col, mk in [(2, S1, 'o'), (3, S2, 's')]:
    g = c[c.views == nv]
    ax.plot(g.iters, g.viol_kl, color=col, marker=mk, ms=3, lw=1.0, label=f'KL, {nv} views')
    ax.plot(g.iters, g.viol_l2, color=col, marker=mk, ms=3, lw=1.0, ls=(0, (3, 1.5)), label=f'$\\ell_2$, {nv} views')
ax.set_xscale('log', base=2); ax.set_yscale('log')
ax.set_xticks([1, 4, 16, 64]); ax.set_xticklabels(['1', '4', '16', '64'])
ax.set_xlabel('sweeps'); ax.set_ylabel('max. rel. violation')
ax.set_title('(b) Convergence', fontsize=8, loc='left')
ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.legend(fontsize=5.5, loc='upper right', handlelength=1.6)
os.makedirs(FIG, exist_ok=True)
fig.savefig(os.path.join(FIG, 'scale.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(FIG, 'scale.png'), bbox_inches='tight', dpi=220)
print('saved')
