"""Case-study figure: (a) Guangdong load around the 2024 Spring Festival, (b) Shandong RT price
around the 2024 National Day holiday. Writes paper/figs/case.pdf."""
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train import get_ds, eval_specs, RES

from paths import FIGS as FIG
os.makedirs(FIG, exist_ok=True)
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
S1, S2, S3 = '#2a78d6', '#eb6834', '#1baf7a'   # categorical slots 1-3 (fixed order)
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.6, 'legend.frameon': False})


def window(dsn, month, ent, d0, nd, methods):
    ds = get_ds(dsn)
    specs = eval_specs(ds, 'test')
    k = [i for i, s in enumerate(specs) if ds.ts[s[0]].strftime('%Y-%m') == month][0]
    s = specs[k]
    b = ds.block(s)
    sl = slice(d0 * 24, (d0 + nd) * 24)
    out = {'truth': b['y'][ent, sl]}
    for m in methods:
        out[m] = np.load(os.path.join(RES, f"{dsn}_{m}_s0_pred.npy"), allow_pickle=True)[k][ent, sl]
    return ds.ts[s[0]:s[1]][sl], out, ds


def style(ax):
    ax.grid(axis='y', color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.2), gridspec_kw=dict(wspace=0.2))
# (a) CN31: Guangdong, 2024-02-04 .. 2024-02-14 (Spring Festival 2024-02-10)
ds = get_ds('cn31')
gd = list(ds.names).index('Guangdong')
ts, o, _ = window('cn31', '2024-02', gd, 6, 8, ['me', 'prof', 'granite'])
ax = axs[0]
ax.plot(ts, o['truth'], color=INK, lw=1.4, label='Truth')
ax.plot(ts, o['granite'], color=S1, lw=1.0, label='\\textsc{Granite}' if False else 'GRANITE')
ax.plot(ts, o['prof'], color=S2, lw=1.0, label='Profile + IPF')
ax.plot(ts, o['me'], color=S3, lw=1.0, ls=(0, (3, 1.5)), label='MaxEnt (IPF)')
ax.axvline(pd.Timestamp('2024-02-10'), color=INK2, lw=0.6, ls=':')
ax.text(pd.Timestamp('2024-02-10 03:00'), ax.get_ylim()[1] * 0.995, 'Spring Festival', color=INK2, fontsize=7, va='top')
ax.set_ylabel('Load (GWh/h)')
ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=2))
ax.set_title('(a) CN31-Load: Guangdong, Feb 2024', fontsize=8, loc='left', color=INK, pad=24)
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d'))
style(ax)
ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), fontsize=6.5, ncol=2, handlelength=1.6, columnspacing=1.0, borderaxespad=0.2)
# (b) SD: real-time price, 2024-10-01 .. 2024-10-08
ts, o, dsd = window('sd', '2024-10', 1, 0, 8, ['gbdt', 'granite'])
ax = axs[1]
dm = np.nanmean(o['truth'].reshape(-1, 24), 1).repeat(24)
ax.plot(ts, dm, color=INK2, lw=0.8, ls=(0, (1, 1)), label='Published day avg.')
ax.plot(ts, o['truth'], color=INK, lw=1.4, label='Truth')
ax.plot(ts, o['granite'], color=S1, lw=1.0, label='GRANITE')
ax.plot(ts, o['gbdt'], color=S2, lw=1.0, label='GBDT + proj.')
ax.axhline(0, color=INK2, lw=0.5)
ax.set_ylabel('RT price (CNY/MWh)')
ax.set_title('(b) SD-Price: real-time market, Oct 2024', fontsize=8, loc='left', color=INK, pad=24)
ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=2))
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d'))
style(ax)
ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), fontsize=6.5, ncol=2, handlelength=1.6, columnspacing=1.0, borderaxespad=0.2)
fig.savefig(os.path.join(FIG, 'case.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(FIG, 'case.png'), bbox_inches='tight', dpi=200)
for k in ['me', 'prof', 'granite']:
    pass
print('saved')
