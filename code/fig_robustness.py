"""Figure 4: robustness, drawn at print size (7.0 x 1.85 in), one row of four panels.

(a) hard coherence vs. soft aggregate penalty (CN31)
(b) perturbed releases: rounding and Gaussian noise (CN31)
(c) inconsistent release (sigma = 2%): cyclic vs. damped projection, max view residual per sweep
(d) gauge weight lambda: WAPE change relative to post-hoc projection (CN31 and SD, 3 seeds)

Writes paper/figs/robustness.pdf and submission_package/figures_highres/fig4_robustness.png.
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from train import RES
from plot_style import (SLATE, CORAL, SAGE, AMBER, INK, INK2, MUTED, CARD_TEAL, METHOD, TEXTWIDTH,
                        apply_style, clean_ax, save)

from paths import FIGS as FIG
HIGH = FIG
apply_style(6.5)

POSTHOC = {'cn31': 1.4392, 'sd': 19.232}   # 3-seed means of post-hoc projection (Table 2)


def draw():
    rm = json.load(open(os.path.join(RES, 'robustness_master.json')))
    viol = json.load(open(os.path.join(RES, 'corrected_violations.json')))
    osc = pd.read_csv(os.path.join(RES, 'stress_oscillation.csv'))

    fig, axs = plt.subplots(1, 4, figsize=(TEXTWIDTH, 1.85),
                            gridspec_kw=dict(wspace=0.45, width_ratios=[1.05, 1.1, 1.0, 0.95]))

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

    # (b) perturbed releases
    ax = axs[1]
    cats = [('round', 3, '3sf'), ('round', 2, '2sf'), ('noise', 0.001, '0.1'), ('noise', 0.005, '0.5'),
            ('noise', 0.01, '1'), ('noise', 0.02, '2'), ('noise', 0.05, '5')]
    rnd = {(r['fam'], r['sig_figs']): r for r in rm['stress_rounding'] if r['ds'] == 'cn31'}
    noi = {(r['fam'], round(r['sigma'], 4)): r for r in rm['stress_noise'] if r['ds'] == 'cn31'}
    xpos = np.array([0, 1, 2.4, 3.4, 4.4, 5.4, 6.4])
    for fam, key in [('Profile+proj', 'prof'), ('Post-hoc', 'posthoc'), ('GRANITE_cyclic', 'granite')]:
        st = METHOD[key]
        vals = [(rnd if kind == 'round' else noi)[(fam, v if kind == 'round' else round(v, 4))]['wape_mean']
                for kind, v, _ in cats]
        ax.plot(xpos[:2], vals[:2], color=st['color'], marker=st['marker'], ms=3.0, lw=0, mec='white', mew=0.35)
        ax.plot(xpos[2:], vals[2:], color=st['color'], marker=st['marker'], ms=3.0, lw=1.0, mec='white', mew=0.35,
                label=st['label'].replace(' + proj.', '').replace(' proj.', ''))
    ax.axvline(1.7, color=GRID_C, lw=0.6)
    ax.text(0.5, 6.15, 'rounding', fontsize=5.6, color=INK2, ha='center')
    ax.text(4.4, 6.15, 'noise $\\sigma$ (%)', fontsize=5.6, color=INK2, ha='center')
    ax.set_xlim(-0.6, 6.9)
    ax.set_xticks(xpos)
    ax.set_xticklabels([c[2] for c in cats], fontsize=5.6)
    ax.set_ylim(0.9, 6.5)
    ax.set_ylabel('CN31 test WAPE (%)')
    ax.set_title('(b) Perturbed releases', loc='left')
    ax.legend(loc='center left', bbox_to_anchor=(0.0, 0.62), fontsize=5.6, handlelength=1.2)
    clean_ax(ax, grid_axis='y')

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
