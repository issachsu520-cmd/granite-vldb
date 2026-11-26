# Declaration of Originality and Non-Use of Artificial Intelligence
#
# The authors declare that no Artificial Intelligence (AI) tools, large language models (LLMs), or generative AI
# technologies were used in the conception, drafting, writing, editing, or polishing of this manuscript. All contents,
# data, analyses, and conclusions are entirely the original work of the authors, who take full responsibility for the
# integrity of the work.
"""Figure 4: robustness, drawn at print size (7.0 x 2.0 in), one row of four panels.

(a) hard coherence vs. soft aggregate penalty (CN31)
(b) perturbed releases: rounding and Gaussian noise (CN31), 3D fences for five estimators
(c) inconsistent release (sigma = 2%): cyclic vs. damped projection, max view residual per sweep
(d) gauge weight lambda: WAPE change relative to post-hoc projection (CN31 and SD, 3 seeds)

Writes paper/figs/robustness.pdf and submission_package/figures_highres/fig4_robustness.png.
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from train import RES
from plot_style import (SLATE, CORAL, SAGE, AMBER, INK, INK2, MUTED, CARD_TEAL, METHOD, TEXTWIDTH,
                        apply_style, clean_ax, style_3d, save)

from paths import FIGS as FIG
HIGH = FIG
apply_style(6.8)

POSTHOC = {'cn31': 1.4392, 'sd': 19.232}   # 3-seed means of post-hoc projection (Table 2)


def draw():
    rm = json.load(open(os.path.join(RES, 'robustness_master.json')))
    viol = json.load(open(os.path.join(RES, 'corrected_violations.json')))
    osc = pd.read_csv(os.path.join(RES, 'stress_oscillation.csv'))

    fig, axs = plt.subplots(1, 4, figsize=(TEXTWIDTH, 2.0),
                            gridspec_kw=dict(wspace=0.5, width_ratios=[1.0, 1.32, 0.95, 0.88], left=0.06,
                                             right=0.995, top=0.88, bottom=0.2))

    # (a) soft penalty vs hard coherence
    ax = axs[0]
    ax.axvspan(5e-7, 1e-5, color=CARD_TEAL, zorder=0)
    ax.text(2.6e-6, 1.76, "coherent", fontsize=5.4, color=SAGE, ha='center', va='center')
    sp = sorted([r for r in rm['softpen'] if r['ds'] == 'cn31'], key=lambda r: r['gamma'])
    ax.plot([r['viol'] for r in sp], [r['wape'] for r in sp], color=MUTED, marker='o', ms=3.2, lw=0.9,
            mec='white', mew=0.4, zorder=3, label='soft penalty')
    for r in sp:
        dx, dy = {0.1: (3, -7), 1.0: (4, 1), 10.0: (-3, 4), 100.0: (-4, 2)}[r['gamma']]
        ax.annotate(f"$\\gamma$={r['gamma']:g}", (r['viol'], r['wape']), xytext=(dx, dy), textcoords='offset points',
                    fontsize=5.5, color=INK2, ha='left' if dx > 0 else 'right', va='center')
    v_ph = viol['cn31_posthoc']['max_viol']
    v_gr = viol['cn31_granite']['max_viol']
    lam = {r['lam']: r for r in rm['lambda_sweep'] if r['ds'] == 'cn31'}
    ax.scatter([v_ph], [POSTHOC['cn31']], **{k: METHOD['posthoc'][k] for k in ('color', 'marker')}, s=16,
               edgecolor='white', linewidth=0.4, zorder=5)
    ax.scatter([v_gr], [lam[1.0]['wape_mean']], color=SLATE, marker='o', s=18, edgecolor='white', linewidth=0.4, zorder=5)
    ax.scatter([v_gr], [lam[3.0]['wape_mean']], color=SLATE, marker='o', s=18, facecolor='white', linewidth=0.9, zorder=5)
    ax.annotate("post-hoc", (v_ph, POSTHOC['cn31']), xytext=(4, 0), textcoords='offset points', fontsize=5.6,
                color=CORAL, va='center')
    ax.annotate("GRANITE $\\lambda$=1", (v_gr, lam[1.0]['wape_mean']), xytext=(4, 0), textcoords='offset points',
                fontsize=5.6, color=SLATE, va='center')
    ax.annotate("$\\lambda$=3", (v_gr, lam[3.0]['wape_mean']), xytext=(4, 0), textcoords='offset points',
                fontsize=5.6, color=SLATE, va='center')
    ratio = min(r['viol'] for r in sp) / v_gr
    ax.text(1.0e-3, 1.37, f"{ratio / 1e3:.1f}$\\times10^3\\times$ lower\nviolation and error",
            fontsize=5.6, color=SLATE, ha='center', va='center', linespacing=1.1)
    ax.set_xscale('log')
    ax.set_xlim(5e-7, 8e-2)
    ax.set_ylim(1.15, 1.82)
    ax.set_xticks([1e-6, 1e-4, 1e-2])
    ax.set_xlabel('Max. relative violation')
    ax.set_ylabel('CN31 test WAPE (%)')
    ax.set_title('(a) Hard vs. soft coherence', loc='left')
    clean_ax(ax, grid_axis='y')

    # (b) perturbed releases: 3D fences, one per method, ordered front (lowest error) to back
    pos = axs[1].get_position()
    axs[1].remove()
    a3 = fig.add_axes([pos.x0 - 0.045, pos.y0 - 0.16, pos.width + 0.075, pos.height + 0.26], projection='3d',
                      computed_zorder=False)
    style_3d(a3, fs=6.2)
    cats = [('noise', 0.0, 'exact'), ('round', 3, '3sf'), ('round', 2, '2sf'), ('noise', 0.001, '0.1'),
            ('noise', 0.005, '0.5'), ('noise', 0.01, '1'), ('noise', 0.02, '2'), ('noise', 0.05, '5')]
    rnd = {(r['fam'], r['sig_figs']): r for r in rm['stress_rounding'] if r['ds'] == 'cn31'}
    noi = {(r['fam'], round(r['sigma'], 4)): r for r in rm['stress_noise'] if r['ds'] == 'cn31'}
    fams = [('GRANITE_cyclic', 'granite', 'GRANITE'), ('Post-hoc', 'posthoc', 'Post-hoc'),
            ('Profile+proj', 'prof', 'Profile'), ('GBDT+proj', 'gbdt', 'GBDT'), ('MaxEnt', 'me', 'MaxEnt')]
    xs = np.arange(len(cats), dtype=float)
    for depth, (fam, key, nm) in list(enumerate(fams))[::-1]:          # back to front
        col = METHOD[key]['color']
        vals = np.array([(rnd if kind == 'round' else noi)[(fam, v if kind == 'round' else round(v, 4))]['wape_mean']
                         for kind, v, _ in cats])
        verts = [[(xs[0], 0)] + list(zip(xs, vals)) + [(xs[-1], 0)]]
        pc = PolyCollection(verts, facecolors=[col], edgecolors=['none'], alpha=0.16)
        pc.set_zorder(2 + 3 * (len(fams) - depth))
        a3.add_collection3d(pc, zs=[depth], zdir='y')
        a3.plot(xs, [depth] * len(xs), vals, color=col, lw=1.0, zorder=3 + 3 * (len(fams) - depth))
        a3.scatter(xs, [depth] * len(xs), vals, color=col, marker=METHOD[key]['marker'], s=6, edgecolor='white',
                   linewidth=0.3, depthshade=False, zorder=4 + 3 * (len(fams) - depth))
    for xv in (0.5, 2.5):
        a3.plot([xv, xv], [-0.5, len(fams) - 0.5], [0, 0], color='#C9CED6', lw=0.6, zorder=1)
    a3.set_xlim(0, len(cats) - 1)
    a3.set_ylim(-0.3, len(fams) - 0.7)
    a3.set_zlim(0, 7)
    a3.set_xticks(xs)
    a3.set_xticklabels([c[2] for c in cats], fontsize=5.4)
    a3.set_yticks([])
    a3.set_zticks([0, 2, 4, 6])
    a3.set_xlabel('rounding | noise $\\sigma$ (%)')
    a3.view_init(elev=22, azim=-58)
    a3.set_box_aspect((1.75, 1.0, 0.85), zoom=1.08)
    fig.text(pos.x0 - 0.02, 0.902, '(b) Perturbed releases: CN31 WAPE (%)', fontsize=7.3, fontweight='bold', color=INK,
             va='bottom')
    hs = [plt.Line2D([], [], color=METHOD[k]['color'], marker=METHOD[k]['marker'], ms=2.8, lw=1.0, mec='white',
                     mew=0.3, label=nm) for _, k, nm in fams]
    fig.legend(handles=hs, loc='upper left', bbox_to_anchor=(pos.x0 - 0.025, 0.875), ncol=5, fontsize=5.5,
               handlelength=1.0, handletextpad=0.25, columnspacing=0.45, borderaxespad=0.0)

    # (c) inconsistent release: cyclic vs damped residual
    ax = axs[2]
    for ds, col, nm in [('cn31', SLATE, 'CN31'), ('sd', CORAL, 'SD')]:
        for proj, ls in [('cyclic', (0, (3, 1.5))), ('damped', '-')]:
            g = osc[(osc.ds == ds) & (osc.proj == proj)].sort_values('iters')
            ax.plot(g.iters, g.viol * 100, color=col, ls=ls, lw=1.0, marker='o' if proj == 'damped' else None,
                    ms=2.4, mec='white', mew=0.3)
            end = g.viol.iloc[-1] * 100
            dy = {('sd', 'cyclic'): 0.27, ('cn31', 'cyclic'): 0.24, ('sd', 'damped'): -0.25,
                  ('cn31', 'damped'): -0.25}[(ds, proj)]
            ax.text(62, end + dy, f"{nm} {proj}: {end:.2f}%", fontsize=5.5, color=col, va='center', ha='right')
    ax.set_xscale('log', base=2)
    ax.set_xticks([1, 4, 16, 64])
    ax.set_xticklabels(['1', '4', '16', '64'])
    ax.set_xlim(0.8, 64)
    ax.set_ylim(1.2, 6.0)
    ax.set_xlabel('Projection sweeps')
    ax.set_ylabel('Max. view residual (%)')
    ax.set_title('(c) Inconsistent release ($\\sigma$=2%)', loc='left')
    clean_ax(ax, grid_axis='y')

    # (d) gauge weight lambda, relative to post-hoc projection
    ax = axs[3]
    ax.axhline(0, color=CORAL, lw=0.8, ls=(0, (3, 1.5)))
    ax.text(-0.35, -1.2, 'post-hoc', fontsize=5.6, color=CORAL, ha='left', va='top')
    for k, (ds, col, mk, nm, dx) in enumerate([('cn31', SLATE, 'o', 'CN31', -0.08), ('sd', AMBER, 's', 'SD', 0.08)]):
        rows = sorted([r for r in rm['lambda_sweep'] if r['ds'] == ds], key=lambda r: r['lam'])
        x = np.arange(len(rows)) + dx
        rel = [(r['wape_mean'] / POSTHOC[ds] - 1) * 100 for r in rows]
        err = [r['wape_std'] / POSTHOC[ds] * 100 for r in rows]
        ax.errorbar(x, rel, yerr=err, color=col, marker=mk, ms=3.0, lw=1.0, capsize=1.6, elinewidth=0.6,
                    mec='white', mew=0.35, label=nm)
    ax.set_xticks(range(4))
    ax.set_xticklabels(['0', '0.3', '1', '3'])
    ax.set_xlabel('Gauge weight $\\lambda$')
    ax.set_ylabel('WAPE vs. post-hoc (%)')
    ax.set_ylim(-18, 14)
    ax.set_title('(d) Gauge weight', loc='left')
    ax.legend(loc='upper right', fontsize=5.8, handlelength=1.2)
    clean_ax(ax, grid_axis='y')

    os.makedirs(FIG, exist_ok=True)
    os.makedirs(HIGH, exist_ok=True)
    save(fig, os.path.join(FIG, 'robustness.pdf'), os.path.join(HIGH, 'fig4_robustness.png'))
    plt.close(fig)
    print("Saved robustness.pdf and fig4_robustness.png")


GRID_C = '#C9CED6'

if __name__ == '__main__':
    draw()
