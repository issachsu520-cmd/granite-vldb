"""Figure 3: reconstructions on held-out weeks, drawn at print size (7.0 x 2.25 in).

(a) CN31-Load: Guangdong around the 2024 Spring Festival (truth, GRANITE, profile + IPF, MaxEnt) and errors
(b) SD-Price: Shandong real-time prices in the first week of October 2024 (truth, GRANITE, GBDT + proj.)

Writes paper/figs/case.pdf and submission_package/figures_highres/fig3_case.png.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from train import get_ds, eval_specs, RES
from plot_style import (SLATE, CORAL, SAGE, AMBER, IRIS, INK, INK2, MUTED, CARD_AMBER, CARD_CORAL, METHOD,
                        TEXTWIDTH, apply_style, clean_ax, save)

from paths import FIGS as FIG
HIGH = FIG
apply_style(6.5)


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


def lim(*arrs, pad=0.06):
    lo = min(np.nanmin(a) for a in arrs)
    hi = max(np.nanmax(a) for a in arrs)
    return lo - pad * (hi - lo), hi + pad * (hi - lo)


def draw():
    fig = plt.figure(figsize=(TEXTWIDTH, 2.25))
    outer = fig.add_gridspec(1, 2, wspace=0.2)
    gs_a = outer[0].subgridspec(2, 1, height_ratios=[2.4, 1.0], hspace=0.1)
    gs_b = outer[1].subgridspec(2, 1, height_ratios=[2.4, 1.0], hspace=0.1)
    gbdt = METHOD['gbdt']['color']
    prof = METHOD['prof']['color']
    me = METHOD['me']['color']

    # (a) CN31, Guangdong, Feb 2024
    ds = get_ds('cn31')
    gd = list(ds.names).index('Guangdong')
    ts, o, _ = window('cn31', '2024-02', gd, 6, 8, ['me', 'prof', 'granite'])
    ax = fig.add_subplot(gs_a[0])
    axr = fig.add_subplot(gs_a[1], sharex=ax)
    col_a = (ax, axr)
    hol = (pd.Timestamp('2024-02-09'), pd.Timestamp('2024-02-13'))
    for a in (ax, axr):
        a.axvspan(*hol, color=CARD_AMBER, zorder=0, lw=0)
    ax.plot(ts, o['me'], color=me, lw=0.8, ls=(0, (3, 1.3)), label='MaxEnt', zorder=2)
    ax.plot(ts, o['prof'], color=prof, lw=0.9, label='Profile + IPF', zorder=3)
    ax.plot(ts, o['truth'], color=INK, lw=1.2, label='Truth', zorder=4)
    ax.plot(ts, o['granite'], color=SLATE, lw=1.0, label='GRANITE', zorder=5)
    lo, hi = lim(o['truth'], o['granite'], o['prof'], o['me'])
    ax.set_ylim(lo, hi + 0.12 * (hi - lo))
    ax.text(hol[0] + (hol[1] - hol[0]) / 2, hi + 0.06 * (hi - lo), 'Spring Festival holiday', fontsize=5.8,
            color='#8A5A00', ha='center', va='center')
    ax.set_ylabel('Load (GWh/h)')
    ax.set_title('(a) CN31-Load: Guangdong, Feb. 2024', loc='left', pad=12)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=4, fontsize=5.8, borderaxespad=0.15,
              handlelength=1.6)
    plt.setp(ax.get_xticklabels(), visible=False)
    clean_ax(ax)
    axr.axhline(0, color=INK2, lw=0.5)
    axr.plot(ts, o['prof'] - o['truth'], color=prof, lw=0.8)
    axr.plot(ts, o['granite'] - o['truth'], color=SLATE, lw=0.9)
    axr.set_ylabel('Error')
    axr.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    axr.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    clean_ax(axr)

    # (b) SD real-time price, Oct 2024
    ts, o, _ = window('sd', '2024-10', 1, 0, 8, ['gbdt', 'granite'])
    ax = fig.add_subplot(gs_b[0])
    axr = fig.add_subplot(gs_b[1], sharex=ax)
    col_b = (ax, axr)
    dm = np.nanmean(o['truth'].reshape(-1, 24), 1).repeat(24)
    for d in range(8):
        t0 = pd.Timestamp('2024-10-01') + pd.Timedelta(days=d, hours=10)
        for a in (ax, axr):
            a.axvspan(t0, t0 + pd.Timedelta(hours=4), color='#EEF2F7', zorder=0, lw=0)
    ax.axhline(0, color=INK2, lw=0.5)
    ax.plot(ts, dm, color=MUTED, lw=0.8, ls=(0, (1.2, 1.2)), label='Daily avg. (released)', zorder=2)
    ax.plot(ts, o['gbdt'], color=gbdt, lw=0.85, label='GBDT + proj.', zorder=3)
    ax.plot(ts, o['truth'], color=INK, lw=1.2, label='Truth', zorder=4)
    ax.plot(ts, o['granite'], color=SLATE, lw=1.0, label='GRANITE', zorder=5)
    lo, hi = lim(o['truth'], o['granite'], o['gbdt'])
    ax.set_ylim(lo, hi)
    ax.set_ylabel('RT price (CNY/MWh)')
    ax.set_title('(b) SD-Price: real-time market, Oct. 2024 (shaded: 10:00–14:00)', loc='left', pad=12)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=4, fontsize=5.8, borderaxespad=0.15,
              handlelength=1.6)
    plt.setp(ax.get_xticklabels(), visible=False)
    clean_ax(ax)
    axr.axhline(0, color=INK2, lw=0.5)
    axr.plot(ts, o['gbdt'] - o['truth'], color=gbdt, lw=0.8)
    axr.plot(ts, o['granite'] - o['truth'], color=SLATE, lw=0.9)
    axr.set_ylabel('Error')
    axr.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    axr.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    clean_ax(axr)

    fig.align_ylabels(col_a)
    fig.align_ylabels(col_b)
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(HIGH, exist_ok=True)
    save(fig, os.path.join(FIG, 'case.pdf'), os.path.join(HIGH, 'fig3_case.png'))
    plt.close(fig)
    print("Saved case.pdf and fig3_case.png")


if __name__ == '__main__':
    draw()
