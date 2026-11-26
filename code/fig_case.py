"""Figure 3: reconstructions on held-out periods, drawn at print size (7.0 x 2.4 in).

(a) CN31-Load: Guangdong around the 2024 Spring Festival (truth, GRANITE, profile + IPF, MaxEnt) and errors
(b) SD-Price: Shandong real-time prices, 1-8 October 2024 (truth, GRANITE, GBDT + proj.) and errors
(c) SD-Price: all 31 daily real-time price profiles of October 2024 as a 3D waterfall: GRANITE (filled ribbon),
    truth (black line) and the released daily average (flat gray line) for each day.

Writes paper/figs/case.pdf and submission_package/figures_highres/fig3_case.png.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import PolyCollection

from train import get_ds, eval_specs, RES
from plot_style import (SLATE, INK, INK2, MUTED, CARD_AMBER, METHOD, TEXTWIDTH, apply_style, clean_ax,
                        style_3d, save)

from paths import FIGS as FIG
HIGH = FIG
apply_style(6.8)


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
    fig = plt.figure(figsize=(TEXTWIDTH, 2.4))
    outer = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.08, 1.0], wspace=0.24, left=0.06, right=0.995,
                             top=0.8, bottom=0.1)
    gs_a = outer[0].subgridspec(2, 1, height_ratios=[2.4, 1.0], hspace=0.1)
    gs_b = outer[1].subgridspec(2, 1, height_ratios=[2.4, 1.0], hspace=0.1)
    gbdt = METHOD['gbdt']['color']
    prof = METHOD['prof']['color']
    me = METHOD['me']['color']
    handles = {}

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
    handles['me'], = ax.plot(ts, o['me'], color=me, lw=0.8, ls=(0, (3, 1.3)), label='MaxEnt', zorder=2)
    handles['prof'], = ax.plot(ts, o['prof'], color=prof, lw=0.9, label='Profile + IPF', zorder=3)
    handles['truth'], = ax.plot(ts, o['truth'], color=INK, lw=1.2, label='Truth', zorder=4)
    handles['granite'], = ax.plot(ts, o['granite'], color=SLATE, lw=1.0, label='GRANITE', zorder=5)
    lo, hi = lim(o['truth'], o['granite'], o['prof'], o['me'])
    ax.set_ylim(lo, hi + 0.12 * (hi - lo))
    ax.text(hol[0] + (hol[1] - hol[0]) / 2, hi + 0.06 * (hi - lo), 'Spring Festival holiday', fontsize=6.0,
            color='#7A5200', ha='center', va='center')
    ax.set_ylabel('Load (GWh/h)')
    ax.set_title('(a) CN31-Load: Guangdong, Feb. 2024', loc='left', pad=3)
    plt.setp(ax.get_xticklabels(), visible=False)
    clean_ax(ax)
    axr.axhline(0, color=INK2, lw=0.5)
    axr.plot(ts, o['prof'] - o['truth'], color=prof, lw=0.8)
    axr.plot(ts, o['granite'] - o['truth'], color=SLATE, lw=0.9)
    axr.set_ylabel('Error')
    axr.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    axr.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    clean_ax(axr)

    # (b) SD real-time price, 1-8 Oct 2024
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
    handles['avg'], = ax.plot(ts, dm, color=MUTED, lw=0.8, ls=(0, (1.2, 1.2)), label='Released daily avg.',
                              zorder=2)
    handles['gbdt'], = ax.plot(ts, o['gbdt'], color=gbdt, lw=0.85, label='GBDT + proj.', zorder=3)
    ax.plot(ts, o['truth'], color=INK, lw=1.2, zorder=4)
    ax.plot(ts, o['granite'], color=SLATE, lw=1.0, zorder=5)
    lo, hi = lim(o['truth'], o['granite'], o['gbdt'])
    ax.set_ylim(lo, hi)
    ax.set_ylabel('RT price (CNY/MWh)')
    ax.set_title('(b) SD-Price: real-time, Oct. 1–8, 2024', loc='left', pad=3)
    plt.setp(ax.get_xticklabels(), visible=False)
    clean_ax(ax)
    axr.axhline(0, color=INK2, lw=0.5)
    axr.plot(ts, o['gbdt'] - o['truth'], color=gbdt, lw=0.8)
    axr.plot(ts, o['granite'] - o['truth'], color=SLATE, lw=0.9)
    axr.set_ylabel('Error')
    axr.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    axr.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    clean_ax(axr)
    fig.align_ylabels(col_a)
    fig.align_ylabels(col_b)

    # shared legend across (a) and (b)
    order = ['truth', 'granite', 'prof', 'me', 'gbdt', 'avg']
    fig.legend([handles[k] for k in order], [handles[k].get_label() for k in order], loc='upper left',
               bbox_to_anchor=(0.055, 1.0), ncol=6, fontsize=6.2, handlelength=1.7, columnspacing=1.1,
               borderaxespad=0.1)

    # (c) 3D waterfall of the 31 daily RT price profiles, Oct 2024
    ts, o, _ = window('sd', '2024-10', 1, 0, 31, ['granite'])
    T = o['truth'].reshape(31, 24)
    Gp = o['granite'].reshape(31, 24)
    with np.errstate(all='ignore'):
        davg = np.array([np.nanmean(r) if np.isfinite(r).any() else np.nan for r in T])
    pos = outer[2].get_position(fig)
    a3 = fig.add_axes([pos.x0 - 0.03, 0.0, 0.995 - pos.x0 + 0.03, 0.93], projection='3d', computed_zorder=False)
    style_3d(a3, fs=6.2)
    zmin = -300.0
    hrs = np.arange(24)
    for d in range(30, -1, -1):                     # back (late days) to front (early days)
        if not np.isfinite(T[d]).any():
            continue
        zo = 2 + 4 * (30 - d)                       # nearer ribbons are drawn later
        verts = [[(0, zmin)] + list(zip(hrs, np.clip(Gp[d], zmin, None))) + [(23, zmin)]]
        pc = PolyCollection(verts, facecolors=['#E6EEF7'], edgecolors=['#A9C4E2'], linewidths=[0.25])
        pc.set_zorder(zo)
        a3.add_collection3d(pc, zs=[d + 1], zdir='y')
        a3.plot([0, 23], [d + 1] * 2, [davg[d]] * 2, color=MUTED, lw=0.4, ls=(0, (1.2, 1.0)), zorder=zo + 1)
        a3.plot(hrs, [d + 1] * 24, T[d], color=INK, lw=0.45, zorder=zo + 2)
        a3.plot(hrs, [d + 1] * 24, Gp[d], color=SLATE, lw=0.6, zorder=zo + 3)
    a3.set_xlim(0, 23)
    a3.set_ylim(1, 31)
    a3.set_zlim(zmin, 650)
    a3.set_xticks([0, 6, 12, 18])
    a3.set_yticks([1, 8, 15, 22, 29])
    a3.set_yticklabels(['10-01', '10-08', '10-15', '10-22', '10-29'])
    a3.set_zticks([-200, 0, 200, 400, 600])
    a3.set_xlabel('hour of day')
    a3.xaxis.labelpad = -6
    a3.set_ylabel('day')
    a3.set_zlabel('CNY/MWh')
    a3.view_init(elev=30, azim=-78)
    a3.set_box_aspect((1.0, 1.5, 0.7), zoom=1.12)
    wape = 100 * np.nansum(np.abs(Gp - T)) / np.nansum(np.abs(T))
    fig.text(pos.x0 - 0.01, 0.965, '(c) SD-Price: 31 daily profiles, Oct. 2024', fontsize=7.3, fontweight='bold',
             color=INK, va='top')
    fig.text(pos.x0 - 0.01, 0.895, f'blue: GRANITE; black: truth; dotted: released\ndaily average; WAPE over the month {wape:.1f}%',
             fontsize=5.8, color=INK2, va='top', linespacing=1.15)

    os.makedirs(FIG, exist_ok=True)
    os.makedirs(HIGH, exist_ok=True)
    save(fig, os.path.join(FIG, 'case.pdf'), os.path.join(HIGH, 'fig3_case.png'))
    plt.close(fig)
    print("Saved case.pdf and fig3_case.png; Oct 2024 RT WAPE", round(wape, 2))


if __name__ == '__main__':
    draw()
