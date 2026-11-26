"""Figure 2: spatial and stratified results, drawn at print size (7.0 x 2.2 in).

(a) TS-GHI: township WAPE reduction of GRANITE (lambda chosen on validation) over MaxEnt, 300 random counties,
    with a latitude profile (median and IQR). Township centroids only; no administrative boundaries are drawn.
(b) CN31-Load: test WAPE per grid region and (c) per day type for four estimators.

Writes paper/figs/spatial_strata.pdf and submission_package/figures_highres/fig2_spatial_strata.png.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from train import get_ds, eval_specs, RES
from plot_style import SLATE, INK, INK2, MUTED, GRID, METHOD, TEXTWIDTH, apply_style, clean_ax, save

from paths import FIGS as FIG
HIGH = FIG
apply_style(6.5)

METHODS = ('prof', 'gbdt', 'posthoc', 'granite')


def township_gain(ds):
    csv_p = os.path.join(RES, 'ghi_township_errors.csv')
    if os.path.exists(csv_p):
        df = pd.read_csv(csv_p)
        return df['idx'].values.astype(int), (1.0 - df['wape_granite'].values / df['wape_me'].values) * 100.0
    specs = eval_specs(ds, 'test', 300)
    P = {'me': np.load(os.path.join(RES, 'ghi_me_c300_s0_pred.npy'), allow_pickle=True),
         'granite': np.load(os.path.join(RES, 'ghi_e2e_s0_pred.npy'), allow_pickle=True)}
    num = {m: {} for m in P}
    den = {}
    for i, sp in enumerate(specs):
        b = ds.block(sp)
        for j, t in enumerate(b['meta']['towns']):
            y = b['y'][j]
            den[t] = den.get(t, 0.0) + np.nansum(np.abs(y))
            for m in P:
                num[m][t] = num[m].get(t, 0.0) + np.nansum(np.abs(P[m][i][j] - y))
    towns = np.array(sorted(den))
    return towns, (1.0 - np.array([num['granite'][t] / num['me'][t] for t in towns])) * 100.0


def dots(ax, labels, table):
    """Cleveland dot plot: one row per stratum, one marker per method."""
    y = np.arange(len(labels))[::-1]
    for yi, lbl in zip(y, labels):
        vals = [table[m][lbl] for m in METHODS]
        ax.hlines(yi, min(vals), max(vals), color='#DDE2E8', lw=1.5, zorder=1)
        ax.hlines(yi, table['granite'][lbl], table['posthoc'][lbl], color=SLATE, lw=1.5, alpha=0.35, zorder=2)
        for m in METHODS:
            st = METHOD[m]
            ax.scatter(table[m][lbl], yi, marker=st['marker'], color=st['color'], s=12, edgecolor='white',
                       linewidth=0.35, zorder=3)
    return y


def draw():
    ds = get_ds('ghi')
    towns, red = township_gain(ds)
    lons, lats = ds.lon[towns], ds.lat[towns]

    W, H = TEXTWIDTH, 2.15
    fig = plt.figure(figsize=(W, H))
    y0, hh = 0.30, 1.52                     # plot band (inches): all panels share it
    map_w = hh * (62.0 / (37.0 * 1.18))     # keeps the map's geographic aspect

    def axes_at(x, w):
        return fig.add_axes([x / W, y0 / H, w / W, hh / H])

    def title_at(x, text):
        fig.text(x / W, (H - 0.1) / H, text, fontsize=7.0, fontweight='bold', color=INK, va='top')

    # (a) map of township gains
    ax = axes_at(0.34, map_w)
    cmap = LinearSegmentedColormap.from_list('gain', ['#D8E6F2', '#7FB0D5', '#2B5C8A', '#0E2A47'])
    sc = ax.scatter(lons, lats, c=np.clip(red, 0, 12), s=0.9, cmap=cmap, vmin=0, vmax=12, linewidths=0,
                    zorder=3, rasterized=True)
    ax.set_xlim(73, 135)
    ax.set_ylim(17, 54)
    ax.set_aspect(1.18, adjustable='box')
    ax.set_xticks([80, 100, 120])
    ax.set_xticklabels(['80°E', '100°E', '120°E'])
    ax.set_yticks([20, 30, 40, 50])
    ax.set_yticklabels(['20°N', '30°N', '40°N', '50°N'])
    ax.grid(color=GRID, lw=0.5, zorder=0)
    for sp in ax.spines.values():
        sp.set_color('#C9CED6')
        sp.set_linewidth(0.5)
    ax.tick_params(length=0)
    cax = ax.inset_axes([0.05, 0.07, 0.3, 0.035])
    cb = fig.colorbar(sc, cax=cax, orientation='horizontal', ticks=[0, 4, 8, 12])
    cb.ax.tick_params(labelsize=5.4, length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)
    cax.set_title('gain (%)', fontsize=5.5, pad=2, loc='left', fontweight='normal')
    title_at(0.05, f'(a) TS-GHI: gain over MaxEnt per township ({(red > 0).mean() * 100:.1f}% improve)')

    # latitude profile
    axl = axes_at(0.34 + map_w + 0.06, 0.55)
    axl.set_ylim(17, 54)
    bins = np.arange(20, 52.5, 2.5)
    mids, q25, q50, q75 = [], [], [], []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        sel = (lats >= b0) & (lats < b1)
        if sel.sum() >= 15:
            mids.append((b0 + b1) / 2)
            q25.append(np.percentile(red[sel], 25))
            q50.append(np.percentile(red[sel], 50))
            q75.append(np.percentile(red[sel], 75))
    axl.fill_betweenx(mids, q25, q75, color=SLATE, alpha=0.18, lw=0, label='IQR')
    axl.plot(q50, mids, color=SLATE, lw=1.1, marker='o', ms=2.2, mec='white', mew=0.3, label='median')
    axl.axvline(0, color=MUTED, lw=0.5, ls=(0, (3, 1.5)))
    axl.set_xlim(-0.5, 13)
    axl.set_xticks([0, 5, 10])
    axl.set_yticks([20, 30, 40, 50])
    axl.set_xlabel('Gain (%)')
    plt.setp(axl.get_yticklabels(), visible=False)
    axl.legend(loc='lower right', fontsize=5.4, handlelength=1.0, borderaxespad=0.1, title='by latitude',
               title_fontsize=5.6)
    clean_ax(axl, grid_axis='x')
    x_b = 0.34 + map_w + 0.06 + 0.55 + 0.66

    # (b) CN31 regions
    ax = axes_at(x_b, 1.22)
    reg = pd.read_csv(os.path.join(RES, 'strata_cn31_region.csv'))
    order = reg[reg.method == 'granite'].sort_values('wape')['region'].tolist()
    table = {m: reg[reg.method == m].set_index('region')['wape'].to_dict() for m in METHODS}
    y = dots(ax, order, table)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=5.9)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(0.5, 3.3)
    ax.set_xlabel('CN31 test WAPE (%)')
    title_at(x_b - 0.62, '(b) CN31: grid regions')
    handles = [plt.Line2D([], [], ls='', marker=METHOD[m]['marker'], color=METHOD[m]['color'], ms=3.0,
                          label=METHOD[m]['label'].replace(' + proj.', '').replace(' proj.', ''))
               for m in ('granite', 'posthoc', 'gbdt', 'prof')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=((x_b - 0.66) / W, (H - 0.25) / H), ncol=4,
               fontsize=5.6, handletextpad=0.05, columnspacing=0.8, borderaxespad=0.0)
    clean_ax(ax, grid_axis='x')
    x_c = x_b + 1.22 + 0.74

    # (c) CN31 day types
    ax = axes_at(x_c, W - x_c - 0.06)
    dt = pd.read_csv(os.path.join(RES, 'strata_cn31_daytype.csv'))
    dt = dt[dt.group_type == 'daytype']
    names = {'Weekday': 'Weekday', 'Weekend': 'Weekend', 'OtherHoliday': 'Public holiday',
             'SpringFestival': 'Spring Festival'}
    table = {m: {names[g]: dt[(dt.method == m) & (dt.group == g)]['wape'].values[0] for g in names}
             for m in METHODS}
    labels = list(names.values())
    y = dots(ax, labels, table)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.9)
    ax.set_ylim(-0.6, len(labels) - 0.3)
    ax.set_xlim(0.8, 3.2)
    ax.set_xlabel('CN31 test WAPE (%)')
    title_at(x_c - 0.72, '(c) CN31: day types')
    clean_ax(ax, grid_axis='x')

    os.makedirs(FIG, exist_ok=True)
    os.makedirs(HIGH, exist_ok=True)
    save(fig, os.path.join(FIG, 'spatial_strata.pdf'), os.path.join(HIGH, 'fig2_spatial_strata.png'))
    plt.close(fig)
    print("Saved spatial_strata.pdf and fig2_spatial_strata.png")


if __name__ == '__main__':
    draw()
