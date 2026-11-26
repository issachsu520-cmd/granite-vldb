"""Figure 5: release design, backbones, and scaling, drawn at print size (7.0 x 2.05 in).

(a) CN31 release design: test WAPE of four estimators under six releases (Table 5)
(b) backbones: WAPE of end-to-end coherent training relative to post-hoc projection, CN31 and SD
(c) runtime of 50 projection sweeps vs. lattice size: GPU layer, vectorized CPU, conic/QP solvers
(d) GPU runtime of 50 KL sweeps over lattice size N and view count K (3D surface; median of three
    timed runs after warm-up; results/bench_views.csv from bench_views.py)

Writes paper/figs/views_scale.pdf and submission_package/figures_highres/fig5_views_scale.png.
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

from train import RES
from plot_style import (SLATE, CORAL, SAGE, AMBER, IRIS, INK, INK2, MUTED, METHOD, SEQ_BLUE, TEXTWIDTH,
                        apply_style, clean_ax, style_3d, save)

SEQ = LinearSegmentedColormap.from_list('seq', SEQ_BLUE)

from paths import FIGS as FIG
HIGH = FIG
from paths import TABLES
apply_style(6.8)

RELEASES = [  # label, MaxEnt, Profile, GBDT, GRANITE (CN31 rows of tables/views_rows.tex)
    ("P-month, R-hour (default)", 3.70, 2.01, 2.25, 1.30),
    ("P-month, N-hour", 5.26, 3.02, 3.11, 2.30),
    ("P-week, R-hour", 3.50, 1.66, 1.96, 1.23),
    ("P-day, R-hour", 3.17, 1.14, 1.54, 0.85),
    ("P-day, R-day", 8.53, 1.94, 2.22, 1.23),
    ("default + N-hour", 3.70, 2.01, 2.21, 1.27),
]


def check_releases():
    """Guard against drift between this figure and Table 5."""
    path = os.path.join(TABLES, 'views_rows.tex')
    if not os.path.exists(path):
        return
    rows = [l for l in open(path, encoding='utf-8') if l.startswith('\\dataset{CN31}')]
    for (lbl, *vals), line in zip(RELEASES, rows):
        nums = [float(t.split('\\')[0].strip()) for t in line.split('&')[2:6]]
        assert np.allclose(nums, vals), (lbl, nums, vals)


def draw():
    check_releases()
    rm = json.load(open(os.path.join(RES, 'robustness_master.json')))
    scale = pd.read_csv(os.path.join(RES, 'bench_scaling.csv'))

    fig, axs = plt.subplots(1, 4, figsize=(TEXTWIDTH, 2.05),
                            gridspec_kw=dict(wspace=0.55, width_ratios=[1.3, 0.92, 1.18, 1.05], left=0.17,
                                             right=0.985, top=0.88, bottom=0.19))

    # (a) release design
    ax = axs[0]
    y = np.arange(len(RELEASES))[::-1]
    xmax = 6.0
    for yi, (lbl, me, pr, gb, gr) in zip(y, RELEASES):
        ax.hlines(yi, min(gr, pr, gb), min(me, xmax), color='#DDE2E8', lw=1.6, zorder=1)
        for key, v in [('me', me), ('prof', pr), ('gbdt', gb), ('granite', gr)]:
            st = METHOD[key]
            vv = min(v, xmax - 0.1)
            ax.scatter(vv, yi, marker='>' if v > xmax else st['marker'], s=13, color=st['color'],
                       edgecolor='white', linewidth=0.35, zorder=3)
            if v > xmax:
                ax.text(vv - 0.25, yi + 0.33, f"{v:.1f}", fontsize=5.4, color=st['color'], ha='center')
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in RELEASES], fontsize=5.8)
    ax.set_xlim(0.5, xmax)
    ax.set_ylim(-0.6, len(RELEASES) - 0.2)
    ax.set_xlabel('CN31 test WAPE (%)')
    ax.set_title('(a) Release design', loc='left', x=-0.62, pad=11)
    handles = [plt.Line2D([], [], ls='', marker=METHOD[k]['marker'], color=METHOD[k]['color'], ms=3.0,
                          label=METHOD[k]['label'].replace(' + proj.', '')) for k in ('granite', 'gbdt', 'prof', 'me')]
    ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(-0.05, 0.98), ncol=4, fontsize=5.5,
              handletextpad=0.05, columnspacing=0.6, borderaxespad=0.0)
    clean_ax(ax, grid_axis='x')

    # (b) backbones: relative change vs post-hoc
    ax = axs[1]
    bb = {(r['ds'], r['arch'], r['regime']): r for r in rm['backbones']}
    tr = {('cn31', 'posthoc'): (1.4392, 0.0216), ('cn31', 'granite'): (1.2980, 0.0211),
          ('sd', 'posthoc'): (19.232, 0.42), ('sd', 'granite'): (19.38, 0.47)}   # Table 2, 3 seeds
    archs = [('transformer', 'Axial Transf.'), ('resnet1d', 'ResNet-1D'), ('bigru', 'BiGRU')]
    yb = np.arange(len(archs))[::-1]
    for k, (ds, col, dy) in enumerate([('cn31', SLATE, 0.18), ('sd', AMBER, -0.18)]):
        rel, err = [], []
        for a, _ in archs:
            if a == 'transformer':
                (p, ps), (g, gs) = tr[(ds, 'posthoc')], tr[(ds, 'granite')]
            else:
                p, ps = bb[(ds, a, 'posthoc')]['wape_mean'], bb[(ds, a, 'posthoc')]['wape_std']
                g, gs = bb[(ds, a, 'granite')]['wape_mean'], bb[(ds, a, 'granite')]['wape_std']
            rel.append((g / p - 1) * 100)
            err.append(np.hypot(gs, ps) / p * 100)
        ax.barh(yb + dy, rel, height=0.32, color=col, zorder=3, label='CN31' if ds == 'cn31' else 'SD')
        ax.errorbar(rel, yb + dy, xerr=err, fmt='none', ecolor=INK2, elinewidth=0.6, capsize=1.3, zorder=4)
        for yy, v, e in zip(yb + dy, rel, err):
            ax.text(v - e - 0.6 if v < 0 else v + e + 0.6, yy, f"{v:+.1f}".replace('-', '−'), fontsize=5.4, color=col,
                    ha='right' if v < 0 else 'left', va='center')
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_yticks(yb)
    ax.set_yticklabels([a[1] for a in archs], fontsize=5.8)
    ax.set_xlim(-17, 8)
    ax.set_xlabel('WAPE vs. post-hoc (%)')
    ax.set_title('(b) Backbones', loc='left', x=-0.5, pad=11)
    ax.legend(loc='lower left', bbox_to_anchor=(-0.05, 0.98), ncol=2, fontsize=5.5, handlelength=0.9,
              borderaxespad=0.0)
    clean_ax(ax, grid_axis='x')

    # (c) scaling
    ax = axs[2]
    lines = [('kl', 'GPU (ours)', SLATE, '-', 'o', 'GPU KL'),
             ('l2', 'GPU (ours)', SLATE, (0, (3, 1.5)), 's', 'GPU $\\ell_2$'),
             ('kl', 'CPU numpy', AMBER, '-', 'D', 'CPU KL'),
             ('kl', 'SCS (exp. cone)', CORAL, '-', '^', 'SCS'),
             ('l2', 'OSQP', IRIS, '-', 'v', 'OSQP')]
    for geom, meth, col, ls, mk, lbl in lines:
        g = scale[(scale.geometry == geom) & (scale.method == meth) & scale.sec.notna()].sort_values('cells')
        ax.plot(g.cells, g.sec, color=col, ls=ls, marker=mk, ms=2.6, mec='white', mew=0.3, lw=0.95, label=lbl)
    ax.legend(loc='lower right', fontsize=5.4, handlelength=1.6, borderaxespad=0.1, labelspacing=0.15)
    gk = scale[(scale.geometry == 'kl') & (scale.method == 'GPU (ours)')].set_index('cells').sec
    ck = scale[(scale.geometry == 'kl') & (scale.method == 'CPU numpy')].set_index('cells').sec
    sk = scale[(scale.geometry == 'kl') & (scale.method == 'SCS (exp. cone)')].set_index('cells').sec.dropna()
    n1, n2 = ck.index.max(), sk.index.max()
    ax.annotate('', (n1, gk[n1]), (n1, ck[n1]), arrowprops=dict(arrowstyle='<->', color=AMBER, lw=0.7,
                                                                  shrinkA=1, shrinkB=1))
    ax.text(n1 * 1.25, np.sqrt(gk[n1] * ck[n1]), f"{ck[n1] / gk[n1]:.0f}×", fontsize=5.6, color=AMBER, va='center')
    ax.annotate('', (n2, gk[n2]), (n2, sk[n2]), arrowprops=dict(arrowstyle='<->', color=CORAL, lw=0.7,
                                                                  shrinkA=1, shrinkB=1))
    ax.text(n2 / 1.3, np.sqrt(gk[n2] * sk[n2]), f"{sk[n2] / gk[n2]:.0f}×", fontsize=5.6, color=CORAL,
            va='center', ha='right')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(8e3, 2e8)
    ax.set_ylim(1e-3, 80)
    ax.set_xticks([1e4, 1e6, 1e8])
    ax.set_xlabel('Lattice cells $N$')
    ax.set_ylabel('Seconds (50 sweeps)')
    ax.set_title('(c) Projection runtime', loc='left', pad=11)
    clean_ax(ax, grid_axis='y')

    # (d) GPU runtime surface over lattice size N and number of views K (3D)
    bv = pd.read_csv(os.path.join(RES, 'bench_views.csv'))
    Ns = np.sort(bv.cells.unique())
    Ks = np.sort(bv.views.unique())
    T = bv.pivot(index='views', columns='cells', values='sec').loc[Ks, Ns].values      # K x N
    LN, KK = np.meshgrid(np.log10(Ns), Ks)
    LT = np.log10(T)
    pos = axs[3].get_position()
    axs[3].remove()
    a3 = fig.add_axes([pos.x0 - 0.06, pos.y0 - 0.17, pos.width + 0.075, pos.height + 0.27], projection='3d',
                      computed_zorder=False)
    style_3d(a3, fs=6.2)
    norm = Normalize(-2.0, 0.0)
    a3.contourf(LN, KK, LT, levels=np.linspace(-2.0, 0.0, 9), zdir='z', offset=-2.0, cmap=SEQ, norm=norm,
                alpha=0.7, zorder=0)
    a3.plot_surface(LN, KK, LT, facecolors=SEQ(norm(LT)), rstride=1, cstride=1, linewidth=0.3,
                    edgecolor=(1, 1, 1, 0.7), shade=False, zorder=2)
    a3.plot(LN[1], KK[1], LT[1], color=CORAL, lw=1.1, zorder=4)               # K = 2, the setting of (c)
    a3.set_xlim(np.log10(Ns[0]), np.log10(Ns[-1]))
    a3.set_ylim(Ks[0], Ks[-1])
    a3.set_zlim(-2.0, 0.0)
    a3.set_xticks([4, 5, 6, 7])
    a3.set_xticklabels(['$10^4$', '$10^5$', '$10^6$', '$10^7$'])
    a3.set_yticks([1, 2, 4, 6])
    a3.set_zticks([-2, -1, 0])
    a3.set_zticklabels(['$10^{-2}$', '$10^{-1}$', '$10^{0}$'])
    a3.set_xlabel('cells $N$')
    a3.set_ylabel('views $K$')
    a3.view_init(elev=24, azim=-128)
    a3.set_box_aspect((1.25, 1.0, 0.8), zoom=1.0)
    fig.text(pos.x0 - 0.02, 0.902, '(d) GPU time (s), $N\\times K$', fontsize=7.3, fontweight='bold', color=INK,
             va='bottom')
    slope = np.polyfit(Ks, T[:, -1], 1)[0]
    fig.text(pos.x0 - 0.02, 0.86, f'{slope:.2f} s per view at $1.2\\times10^7$ cells', fontsize=5.6, color=INK2,
             va='top')
    fig.text(pos.x0 - 0.02, 0.80, '— $K=2$ (as in (c))', fontsize=5.6, color=CORAL, va='top')

    os.makedirs(FIG, exist_ok=True)
    os.makedirs(HIGH, exist_ok=True)
    save(fig, os.path.join(FIG, 'views_scale.pdf'), os.path.join(HIGH, 'fig5_views_scale.png'))
    plt.close(fig)
    print("Saved views_scale.pdf and fig5_views_scale.png")


if __name__ == '__main__':
    draw()
