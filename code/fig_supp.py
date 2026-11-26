# Declaration of Originality and Non-Use of Artificial Intelligence
#
# The authors declare that no Artificial Intelligence (AI) tools, large language models (LLMs), or generative AI
# technologies were used in the conception, drafting, writing, editing, or polishing of this manuscript. All contents,
# data, analyses, and conclusions are entirely the original work of the authors, who take full responsibility for the
# integrity of the work.
"""Supplementary figures for the Technical Appendix:
  1. supp_cn31_heatmap.pdf: 31 provinces x 24 test months WAPE reduction (GRANITE vs. Post-hoc)
  2. supp_sd_breakdown.pdf: SD-Price stratified by TOU Period and Price Regime
  3. supp_noisy_oscillation.pdf: Cyclic vs. damped/tolerance-band sweep trajectory under noise=2%
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from train import RES
from plot_style import SLATE, CORAL, SAGE, AMBER, IRIS, INK, INK2, GRID, apply_style, clean_ax

from paths import FIGS as FIG
HIGH = FIG
apply_style(7.0)


def draw_cn31_heatmap():
    df = pd.read_csv(os.path.join(RES, 'strata_cn31_prov_month.csv'))
    p_gr = df[df.method == 'granite'].pivot(index='province', columns='month', values='wape')
    p_ph = df[df.method == 'posthoc'].pivot(index='province', columns='month', values='wape')
    diff = p_ph - p_gr  # positive = GRANITE better (lower WAPE)
    order = diff.mean(1).sort_values(ascending=False).index
    diff = diff.loc[order]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    im = ax.imshow(diff.values, aspect='auto', cmap='YlGnBu', vmin=0.0, vmax=0.45)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order, fontsize=6.2)
    cols = list(diff.columns)
    ax.set_xticks(np.arange(0, len(cols), 2))
    ax.set_xticklabels([cols[i][2:] for i in range(0, len(cols), 2)], fontsize=6.5)
    ax.set_xlabel('Test month (YY-MM)')
    ax.set_title('CN31-Load: Per-Province × Per-Month WAPE Reduction (Post-hoc − GRANITE, pp; 31/31 prov., 24/24 mo.)', fontsize=8.2, loc='left')
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label('WAPE gain (pp)', fontsize=7.5)
    cb.ax.tick_params(labelsize=6.8)
    os.makedirs(FIG, exist_ok=True); os.makedirs(HIGH, exist_ok=True)
    fig.savefig(os.path.join(FIG, 'supp_cn31_heatmap.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(HIGH, 'supp_cn31_heatmap.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)


def draw_sd_breakdown():
    df = pd.read_csv(os.path.join(RES, 'strata_sd_breakdown.csv'))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.55), gridspec_kw=dict(wspace=0.28))

    # (a) By TOU period
    ax = axs[0]
    tous = ['DeepValley(10-14h)', 'Valley(22-07h)', 'Flat(07-10,14-17h)', 'Peak(17-22h)']
    tlbl = ['Deep Valley\n(10–14h)', 'Valley\n(22–07h)', 'Flat\n(07–10,14–17h)', 'Peak\n(17–22h)']
    x = np.arange(len(tous)); w = 0.20
    sub = df[df.dim == 'tou']
    for i, (m, col, lbl) in enumerate([('prof', AMBER, 'Profile+proj.'), ('gbdt', IRIS, 'GBDT+proj.'),
                                       ('posthoc', CORAL, 'Post-hoc'), ('granite', SLATE, 'GRANITE')]):
        vals = [sub[(sub.method == m) & (sub.group == g)]['wape'].values[0] for g in tous]
        ax.bar(x + (i - 1.5) * w, vals, width=w * 0.9, color=col, label=lbl)
    ax.set_xticks(x); ax.set_xticklabels(tlbl, fontsize=6.5)
    ax.set_ylabel('Test WAPE (%)')
    ax.set_title('(a) SD-Price by TOU Period', loc='left', fontsize=8.2)
    clean_ax(ax)
    ax.legend(fontsize=6.1, ncol=2, loc='upper right', frameon=True, facecolor='white', edgecolor='#E2E8F0', framealpha=0.92)

    # (b) By Price Regime
    ax = axs[1]
    regs = ['NonPos(<=0)', 'Normal(0-600]', 'Spike(>600)']
    rlbl = ['Non-positive\n(≤ 0)', 'Normal\n(0, 600]', 'Spike\n(> 600)']
    x = np.arange(len(regs))
    sub = df[df.dim == 'regime']
    for i, (m, col, lbl) in enumerate([('prof', AMBER, 'Profile+proj.'), ('gbdt', IRIS, 'GBDT+proj.'),
                                       ('posthoc', CORAL, 'Post-hoc'), ('granite', SLATE, 'GRANITE')]):
        vals = [sub[(sub.method == m) & (sub.group == g)]['wape'].values[0] for g in regs]
        ax.bar(x + (i - 1.5) * w, vals, width=w * 0.9, color=col, label=lbl)
    ax.set_xticks(x); ax.set_xticklabels(rlbl, fontsize=6.5)
    ax.set_ylabel('Test WAPE (%)')
    ax.set_title('(b) SD-Price by Price Regime', loc='left', fontsize=8.2)
    clean_ax(ax)

    fig.savefig(os.path.join(FIG, 'supp_sd_breakdown.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(HIGH, 'supp_sd_breakdown.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)


def draw_oscillation():
    df = pd.read_csv(os.path.join(RES, 'stress_oscillation.csv'))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.35), gridspec_kw=dict(wspace=0.28))
    specs = [
        (axs[0], 'cn31', '(a) CN31-Load (σ = 2% noisy release)', (0.98, 0.41), 'center right'),
        (axs[1], 'sd',   '(b) SD-Price (σ = 2% noisy release)',  (0.98, 0.96), 'upper right'),
    ]
    for ax, dsn, title, anchor, loc in specs:
        sub = df[df.ds == dsn]
        for proj, col, mk, lbl in [('cyclic', CORAL, 'o', 'Standard cyclic'),
                                   ('damped', SLATE, 's', 'Damped (α=0.5)'),
                                   ('tolband', SAGE, '^', 'Tol.-band (3%)')]:
            g = sub[sub.proj == proj]
            ax.plot(g.iters, g.viol * 100, color=col, marker=mk, ms=3.8, mec='white', mew=0.45, lw=1.3, label=lbl)
        ax.set_xscale('log', base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64]); ax.set_xticklabels(['1', '2', '4', '8', '16', '32', '64'])
        ax.set_xlabel('Projection sweeps')
        ax.set_ylabel('Max view residual (%)')
        ax.set_title(title, loc='left', fontsize=8.0)
        clean_ax(ax, grid_axis='both')
        ax.legend(fontsize=6.2, loc=loc, bbox_to_anchor=anchor, frameon=True, facecolor='white', edgecolor='#E2E8F0', framealpha=0.94)
    fig.savefig(os.path.join(FIG, 'supp_noisy_oscillation.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(HIGH, 'supp_noisy_oscillation.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)


def draw_convergence():
    """Max KL and l2 violation per sweep for the synthetic two- and three-view releases of bench.py."""
    conv = pd.read_csv(os.path.join(RES, 'bench_convergence.csv'))
    fig, ax = plt.subplots(figsize=(3.33, 1.75))
    for nv, col, mk in [(2, SLATE, 'o'), (3, SAGE, 's')]:
        g = conv[conv.views == nv].sort_values('iters')
        ax.plot(g.iters, g.viol_kl, color=col, marker=mk, ms=3.0, mec='white', mew=0.35, lw=1.0,
                label=f'KL, {nv} views')
        ax.plot(g.iters, g.viol_l2, color=col, marker=mk, ms=2.6, mfc='white', mew=0.6, lw=0.8,
                ls=(0, (3, 1.5)), label=f'$\\ell_2$, {nv} views')
    ax.axhline(1e-5, color='#9AA3AF', lw=0.6, ls=(0, (3, 1.5)))
    ax.text(128, 1.3e-5, '$10^{-5}$', fontsize=6.0, color=INK2, ha='right', va='bottom')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks([1, 4, 16, 64])
    ax.set_xticklabels(['1', '4', '16', '64'])
    ax.set_ylim(1e-7, 0.3)
    ax.set_xlabel('Projection sweeps')
    ax.set_ylabel('Max. relative violation')
    clean_ax(ax, grid_axis='y')
    ax.legend(fontsize=6.0, ncol=2, loc='upper right', handlelength=1.6, columnspacing=0.8)
    fig.savefig(os.path.join(FIG, 'supp_convergence.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(HIGH, 'supp_convergence.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    draw_cn31_heatmap()
    draw_sd_breakdown()
    draw_oscillation()
    draw_convergence()
    print("Saved supplementary figures")
