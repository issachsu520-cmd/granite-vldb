# Declaration of Originality and Non-Use of Artificial Intelligence
#
# The authors declare that no Artificial Intelligence (AI) tools, large language models (LLMs), or generative AI
# technologies were used in the conception, drafting, writing, editing, or polishing of this manuscript. All contents,
# data, analyses, and conclusions are entirely the original work of the authors, who take full responsibility for the
# integrity of the work.
"""Figure 2: spatial and stratified results, drawn at print size (7.0 x 2.3 in).

(a) TS-GHI: township WAPE reduction of GRANITE (lambda chosen on validation) over MaxEnt, 300 random counties,
    with a latitude profile (median and IQR). Township centroids only; no administrative boundaries are drawn.
(b) CN31-Load: test WAPE per grid region and per day type for four estimators (Cleveland dot plot).
(c) CN31-Load: WAPE reduction of GRANITE over post-hoc projection by hour of day and calendar month (3D surface,
    3-seed average, 2023-2024 test period) with its contour map on the floor.

Writes paper/figs/spatial_strata.pdf and submission_package/figures_highres/fig2_spatial_strata.png.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

from train import get_ds, eval_specs, RES
from plot_style import (SLATE, INK, INK2, MUTED, GRID, SEQ_BLUE, METHOD, TEXTWIDTH, apply_style, clean_ax,
                        style_3d, save)

from paths import FIGS as FIG
HIGH = FIG
apply_style(6.8)

METHODS = ('prof', 'gbdt', 'posthoc', 'granite')
SEQ = LinearSegmentedColormap.from_list('seq', SEQ_BLUE)
MON = dict(enumerate(['', 'Jan.', 'Feb.', 'Mar.', 'Apr.', 'May', 'Jun.', 'Jul.', 'Aug.', 'Sep.', 'Oct.',
                       'Nov.', 'Dec.']))


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


def hour_month_gain():
    """Gain (%) of GRANITE over post-hoc projection per (month, hour), pooled over provinces and test years."""
    csv_p = os.path.join(RES, 'strata_cn31_hour_month.csv')
    if not os.path.exists(csv_p):
        ds = get_ds('cn31')
        specs = eval_specs(ds, 'test')
        P = {m: [np.load(os.path.join(RES, f'cn31_{m}_s{s}_pred.npy'), allow_pickle=True) for s in range(3)]
             for m in ('granite', 'posthoc')}
        acc = {}
        for k, sp in enumerate(specs):
            y = ds.block(sp)['y']
            ts = ds.ts[sp[0]:sp[1]]
            mon, hod = ts.month.values, ts.hour.values
            for m in P:
                ae = np.mean([np.abs(P[m][s][k] - y) for s in range(3)], 0)
                for h in range(24):
                    for mo in np.unique(mon):
                        sel = (hod == h) & (mon == mo)
                        d = acc.setdefault((mo, h), {'y': 0.0})
                        d[m] = d.get(m, 0.0) + float(np.nansum(ae[:, sel]))
                        if m == 'granite':
                            d['y'] += float(np.nansum(np.abs(y[:, sel])))
        df = pd.DataFrame([dict(month=mo, hour=h, ae_granite=d['granite'], ae_posthoc=d['posthoc'], y_sum=d['y'])
                           for (mo, h), d in sorted(acc.items())])
        df['gain'] = 100 * (1 - df.ae_granite / df.ae_posthoc)
        df.to_csv(csv_p, index=False)
    return pd.read_csv(csv_p)


def dots(ax, labels, table, y):
    """Cleveland dot plot: one row per stratum, one marker per method."""
    for yi, lbl in zip(y, labels):
        vals = [table[m][lbl] for m in METHODS]
        ax.hlines(yi, min(vals), max(vals), color='#DDE2E8', lw=1.5, zorder=1)
        ax.hlines(yi, table['granite'][lbl], table['posthoc'][lbl], color=SLATE, lw=1.5, alpha=0.35, zorder=2)
        for m in METHODS:
            st = METHOD[m]
            ax.scatter(table[m][lbl], yi, marker=st['marker'], color=st['color'], s=11, edgecolor='white',
                       linewidth=0.35, zorder=3)


def draw():
    ds = get_ds('ghi')
    towns, red = township_gain(ds)
    lons, lats = ds.lon[towns], ds.lat[towns]

    W, H = TEXTWIDTH, 2.3
    fig = plt.figure(figsize=(W, H))
    y0, hh = 0.30, 1.58                     # plot band (inches) shared by the 2D panels
    map_w = hh * (62.0 / (37.0 * 1.18))     # keeps the map's geographic aspect

    def axes_at(x, w, yb=y0, hb=hh, **kw):
        return fig.add_axes([x / W, yb / H, w / W, hb / H], **kw)

    def title_at(x, text):
        fig.text(x / W, (H - 0.08) / H, text, fontsize=7.4, fontweight='bold', color=INK, va='top')

    # (a) map of township gains
    ax = axes_at(0.30, map_w)
    cmap = LinearSegmentedColormap.from_list('gain', SEQ_BLUE[:1] + SEQ_BLUE[1:])
    sc = ax.scatter(lons, lats, c=np.clip(red, 0, 12), s=0.8, cmap=cmap, vmin=0, vmax=12, linewidths=0,
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
    cax = ax.inset_axes([0.05, 0.08, 0.3, 0.035])
    cb = fig.colorbar(sc, cax=cax, orientation='horizontal', ticks=[0, 4, 8, 12])
    cb.ax.tick_params(labelsize=5.6, length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)
    cax.set_title('gain (%)', fontsize=5.8, pad=2, loc='left', fontweight='normal')
    title_at(0.02, f'(a) TS-GHI: gain over MaxEnt per township ({(red > 0).mean() * 100:.1f}% improve)')

    # latitude profile
    xl = 0.30 + map_w + 0.05
    axl = axes_at(xl, 0.46)
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
    axl.legend(loc='lower right', fontsize=5.6, handlelength=1.0, borderaxespad=0.1, title='by latitude',
               title_fontsize=5.8)
    clean_ax(axl, grid_axis='x')

    # (b) CN31 regions and day types in one dot plot
    x_b = xl + 0.46 + 0.80
    w_b = 1.02
    ax = axes_at(x_b, w_b)
    reg = pd.read_csv(os.path.join(RES, 'strata_cn31_region.csv'))
    order = reg[reg.method == 'granite'].sort_values('wape')['region'].tolist()
    table = {m: reg[reg.method == m].set_index('region')['wape'].to_dict() for m in METHODS}
    dt = pd.read_csv(os.path.join(RES, 'strata_cn31_daytype.csv'))
    dt = dt[dt.group_type == 'daytype']
    names = {'Weekday': 'Weekday', 'Weekend': 'Weekend', 'OtherHoliday': 'Public holiday',
             'SpringFestival': 'Spring Festival'}
    for m in METHODS:
        for g, lbl in names.items():
            table[m][lbl] = dt[(dt.method == m) & (dt.group == g)]['wape'].values[0]
    labels = order + list(names.values())
    ypos = np.r_[np.arange(len(order))[::-1] + len(names) + 1.2, np.arange(len(names))[::-1]]
    dots(ax, labels, table, ypos)
    ax.axhline(len(names) + 0.1, color='#C9CED6', lw=0.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=6.0)
    ax.set_ylim(-0.6, ypos.max() + 0.6)
    ax.set_xlim(0.5, 3.3)
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel('CN31 test WAPE (%)')
    ax.text(3.3, len(names) + 0.45, 'day type', fontsize=5.6, color=INK2, ha='right', va='center', style='italic')
    ax.text(3.3, ypos.max() + 0.35, 'grid region', fontsize=5.6, color=INK2, ha='right', va='bottom',
            style='italic')
    title_at(x_b - 0.78, '(b) CN31: strata')
    handles = [plt.Line2D([], [], ls='', marker=METHOD[m]['marker'], color=METHOD[m]['color'], ms=3.0,
                          label=METHOD[m]['label'].replace(' + proj.', '').replace(' proj.', ''))
               for m in ('granite', 'posthoc', 'gbdt', 'prof')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=((x_b - 0.80) / W, (H - 0.24) / H), ncol=2,
               fontsize=5.8, handletextpad=0.05, columnspacing=0.6, borderaxespad=0.0, labelspacing=0.15)
    clean_ax(ax, grid_axis='x')

    # (c) 3D surface: gain over post-hoc by hour of day x month
    hm = hour_month_gain()
    G = hm.pivot(index='month', columns='hour', values='gain').values        # 12 x 24
    hrs, mons = np.arange(24), np.arange(1, 13)
    HH, MM = np.meshgrid(hrs, mons)
    x_c = x_b + w_b + 0.06
    a3 = axes_at(x_c + 0.04, W - x_c - 0.22, yb=-0.03, hb=H - 0.3, projection='3d', computed_zorder=False)
    style_3d(a3, fs=6.0)
    norm = Normalize(0, 22)
    a3.contourf(HH, MM, G, levels=np.arange(0, 24, 3), zdir='z', offset=0, cmap=SEQ, norm=norm, alpha=0.75,
                zorder=0)
    a3.plot_surface(HH, MM, G, facecolors=SEQ(norm(G)), rstride=1, cstride=1, linewidth=0.2,
                    edgecolor=(1, 1, 1, 0.55), shade=False, antialiased=True, zorder=2)
    a3.set_xlim(0, 23)
    a3.set_ylim(1, 12)
    a3.set_zlim(0, 22)
    a3.set_xticks([0, 6, 12, 18, 23])
    a3.set_yticks([1, 4, 7, 10])
    a3.set_yticklabels(['Jan', 'Apr', 'Jul', 'Oct'])
    a3.set_zticks([0, 10, 20])
    a3.set_zticklabels([])
    a3.zaxis.set_tick_params(length=0)
    a3.set_xlabel('hour of day')
    a3.set_ylabel('month')
    a3.view_init(elev=27, azim=-57)
    a3.set_box_aspect((1.45, 1.0, 0.62), zoom=1.02)
    cbx = fig.add_axes([(W - 0.16) / W, 0.95 / H, 0.05 / W, 0.75 / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SEQ), cax=cbx, ticks=[0, 10, 20])
    cb.ax.tick_params(labelsize=5.6, length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)
    cbx.set_title('gain\n(%)', fontsize=5.8, pad=2, fontweight='normal', linespacing=1.0)
    top = hm.loc[hm.gain.idxmax()]
    low = hm.loc[hm.gain.idxmin()]
    title_at(x_c - 0.02, '(c) CN31: gain over post-hoc, hour × month')
    fig.text((x_c + 0.02) / W, (H - 0.27) / H,
             f"overall {100 * (1 - hm.ae_granite.sum() / hm.ae_posthoc.sum()):.1f}%; "
             f"{top.gain:.0f}% at {int(top.hour):02d}:00 in {MON[int(top.month)]}, "
             f"{low.gain:.1f}% at {int(low.hour):02d}:00 in {MON[int(low.month)]}",
             fontsize=5.6, color=INK2, va='top')

    os.makedirs(FIG, exist_ok=True)
    os.makedirs(HIGH, exist_ok=True)
    save(fig, os.path.join(FIG, 'spatial_strata.pdf'), os.path.join(HIGH, 'fig2_spatial_strata.png'))
    plt.close(fig)
    print("Saved spatial_strata.pdf and fig2_spatial_strata.png")


if __name__ == '__main__':
    draw()
