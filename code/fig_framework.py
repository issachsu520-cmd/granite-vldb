"""Figure 1: overview of GRANITE, drawn at print size (7.0 x 2.8 in).

(a) pipeline: release -> coherent anchor -> dual-stream encoder -> coherence layer -> coherent output
(b) the coherence layer: batching by group-id offsets, the KL / l2 view updates, and their SQL form
(c) projection geometry: gauge fiber and the Pythagorean property

Writes paper/figs/framework.pdf and submission_package/figures_highres/fig1_framework.png.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

from plot_style import (SLATE, CORAL, SAGE, AMBER, IRIS, INK, INK2, MUTED, CARD_BLUE, CARD_TEAL, CARD_CORAL,
                        CARD_IRIS, CARD_GRAY, TEXTWIDTH, apply_style, rounded_box, shadow_card, save)

from paths import FIGS as FIG
HIGH = FIG
apply_style(6.5)

STEEL = '#46566A'
FS = 6.3      # body text
FS_H = 6.8    # card headers
FS_T = 7.6    # panel titles


def arrow(ax, p1, p2, col=INK2, lw=0.9, ls='-', rad=0.0, mut=7, style='-|>', zorder=6):
    ax.add_patch(FancyArrowPatch(p1, p2, connectionstyle=f"arc3,rad={rad}", arrowstyle=style,
                                 mutation_scale=mut, color=col, linewidth=lw, linestyle=ls, zorder=zorder,
                                 shrinkA=0, shrinkB=0))


def card(ax, x, y, w, h, num, title, col, fc):
    shadow_card(ax, x, y, w, h, fc=fc, ec=col, lw=0.7, r=0.6, header_h=2.5, header_fc=col, zorder=2)
    ax.add_patch(Circle((x + 1.45, y + h - 1.25), 0.85, fc='white', ec='none', zorder=4))
    ax.text(x + 1.45, y + h - 1.27, str(num), fontsize=6.0, fontweight='bold', color=col,
            ha='center', va='center', zorder=5)
    ax.text(x + 2.75, y + h - 1.25, title, fontsize=FS_H, fontweight='bold', color='white',
            ha='left', va='center', zorder=5)


def inner(ax, x, y, w, h, ec, fc='white', ls='-', lw=0.6):
    rounded_box(ax, x, y, w, h, fc=fc, ec=ec, lw=lw, r=0.45, ls=ls, zorder=3)


def draw():
    fig = plt.figure(figsize=(TEXTWIDTH, 2.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis('off')

    # ------------------------------------------------------------------ (a) pipeline
    ax.text(0.6, 38.7, "(a) Pipeline: training through the exact coherence layer (Sections 2–4)",
            fontsize=FS_T, fontweight='bold', color=INK, va='center')
    w, g, y0, h = 17.88, 2.4, 19.6, 17.2
    xs = [0.5 + i * (w + g) for i in range(5)]

    # 1 Release
    x = xs[0]
    card(ax, x, y0, w, h, 1, "Aggregate release", STEEL, CARD_GRAY)
    gx, gy, cw, ch = x + 1.2, 28.1, 1.32, 1.02
    rng = np.random.default_rng(3)
    shades = ['#E9EEF4', '#D3DDE8', '#BCCADA']
    for r in range(4):
        for c in range(6):
            ax.add_patch(Rectangle((gx + c * cw, gy + r * ch), cw * 0.92, ch * 0.88,
                                   fc=shades[rng.integers(0, 3)], ec='none', zorder=3))
    ax.text(gx + 3 * cw, gy + 2 * ch - 0.05, "?", fontsize=7.5, color=STEEL, ha='center', va='center',
            fontweight='bold', zorder=4)
    # row sums (view 1) and column sums (view 2) as bars along the grid edges
    ax.add_patch(Rectangle((gx + 6 * cw + 0.25, gy), 0.42, 4 * ch - 0.12, fc=SLATE, ec='none', zorder=3))
    ax.add_patch(Rectangle((gx, gy - 0.65), 6 * cw - 0.1, 0.42, fc=SAGE, ec='none', zorder=3))
    ax.text(gx + 6 * cw + 1.2, gy + 3.05, "hidden fine", fontsize=FS, color=INK, va='center')
    ax.text(gx + 6 * cw + 1.2, gy + 1.65, "table $\\mathbf{x}$", fontsize=FS, color=INK, va='center')
    for i, (col, txt) in enumerate([(SLATE, "V1  province × month, sum"),
                                    (SAGE, "V2  region × hour, sum"),
                                    (IRIS, "V$K$  day / TOU, average")]):
        yy = 24.55 - i * 2.05
        inner(ax, x + 0.8, yy, w - 1.6, 1.75, ec=col)
        ax.add_patch(Rectangle((x + 1.25, yy + 0.45), 0.45, 0.85, fc=col, ec='none', zorder=4))
        ax.text(x + 2.15, yy + 0.86, txt, fontsize=FS - 0.2, color=INK, va='center', zorder=4)

    # 2 Anchor
    x = xs[1]
    card(ax, x, y0, w, h, 2, "Coherent anchor", SAGE, CARD_TEAL)
    inner(ax, x + 0.8, 27.6, w - 1.6, 6.0, ec=SAGE)
    ax.text(x + w / 2, 32.0, "$\\mathbf{x}^{\\mathrm{ME}}=\\Pi_{\\mathcal{C}}(\\mathbf{1})$ (KL)",
            fontsize=FS + 0.3, color=INK, ha='center', va='center', zorder=4)
    ax.text(x + w / 2, 30.3, "or $\\Pi_{\\mathcal{C}}(\\mathbf{0})$ ($\\ell_2$)",
            fontsize=FS + 0.3, color=INK, ha='center', va='center', zorder=4)
    ax.text(x + w / 2, 28.5, "max. entropy, no gradient", fontsize=FS - 0.3, color=INK2,
            ha='center', va='center', zorder=4)
    arrow(ax, (x + w / 2, 27.6), (x + w / 2, 26.3), col=SAGE)
    inner(ax, x + 0.8, 20.2, w - 1.6, 6.1, ec=SAGE)
    ax.text(x + w / 2, 25.0, "Scale-free inputs", fontsize=FS, fontweight='bold', color=INK, ha='center',
            va='center', zorder=4)
    for i, t in enumerate(["normalized anchor,", "view means, coverage,", "weather, calendar $\\mathbf{W}$"]):
        ax.text(x + w / 2, 23.45 - i * 1.25, t, fontsize=FS - 0.2, color=INK2, ha='center', va='center', zorder=4)

    # 3 Encoder
    x = xs[2]
    card(ax, x, y0, w, h, 3, "Dual-stream encoder ×3", SLATE, CARD_BLUE)
    inner(ax, x + 0.8, 27.6, w - 1.6, 6.0, ec=SLATE)
    tx, ty, s = x + 1.5, 28.25, 0.92
    for r in range(4):
        for c in range(4):
            hi_r, hi_c = (r == 1), (c == 2)
            fc = '#F5D38A' if hi_r else ('#AFC8E2' if hi_c else '#EEF1F5')
            ax.add_patch(Rectangle((tx + c * s, ty + r * s), s * 0.86, s * 0.86, fc=fc, ec='none', zorder=4))
    ax.text(x + 6.0, 32.25, "Day tokens $\\mathbf{H}$", fontsize=FS, fontweight='bold', color=INK, va='center', zorder=4)
    ax.text(x + 6.0, 30.6, "axial attention over", fontsize=FS - 0.2, color=INK2, va='center', zorder=4)
    ax.text(x + 6.0, 29.15, "days and entities", fontsize=FS - 0.2, color=INK2, va='center', zorder=4)
    arrow(ax, (x + w / 2, 27.55), (x + w / 2, 26.35), col=SLATE, style='<|-|>')
    ax.text(x + w / 2 + 0.7, 26.95, "patch / unpatch", fontsize=FS - 0.6, color=INK2, va='center')
    inner(ax, x + 0.8, 20.2, w - 1.6, 6.1, ec=SLATE)
    ax.text(x + w / 2, 25.0, "Hourly cells $\\mathbf{h}_c$", fontsize=FS, fontweight='bold', color=INK,
            ha='center', va='center', zorder=4)
    xx = np.linspace(x + 2.0, x + w - 2.0, 40)
    ax.plot(xx, 23.4 + 0.55 * np.sin((xx - xx[0]) / (xx[-1] - xx[0]) * 4 * np.pi), color=SLATE, lw=0.8, zorder=4)
    ax.text(x + w / 2, 21.25, "messages along view groups", fontsize=FS - 0.2, color=INK2, ha='center',
            va='center', zorder=4)

    # 4 Coherence layer
    x = xs[3]
    card(ax, x, y0, w, h, 4, "Coherence layer", IRIS, CARD_IRIS)
    inner(ax, x + 0.8, 29.9, w - 1.6, 3.7, ec=IRIS)
    ax.text(x + w / 2, 32.55, "Seed", fontsize=FS, fontweight='bold', color=INK, ha='center', va='center', zorder=4)
    ax.text(x + w / 2, 30.95, "$\\log\\mathbf{q}=\\log\\mathbf{x}^{\\mathrm{ME}}+\\mathbf{f}_\\theta$",
            fontsize=FS + 0.2, color=INK, ha='center', va='center', zorder=4)
    arrow(ax, (x + w / 2, 29.9), (x + w / 2, 28.75), col=IRIS)
    inner(ax, x + 0.8, 24.6, w - 1.6, 4.15, ec=IRIS, lw=1.1)
    ax.text(x + w / 2, 27.45, "$\\hat{\\mathbf{x}}=\\Pi_{\\mathcal{C}}(\\mathbf{q})$, $I$ sweeps",
            fontsize=FS + 0.3, color=INK, ha='center', va='center', zorder=4)
    ax.text(x + w / 2, 25.65, "exact, $O(KN)$ per sweep", fontsize=FS - 0.2, color=INK2, ha='center',
            va='center', zorder=4)
    inner(ax, x + 0.8, 20.2, w - 1.6, 3.6, ec=CORAL, fc=CARD_CORAL, ls=(0, (3, 1.5)))
    ax.text(x + w / 2, 22.85, "training loss", fontsize=FS - 0.3, color=CORAL, ha='center',
            va='center', zorder=4)
    ax.text(x + w / 2, 21.2, "$\\ell(\\hat{\\mathbf{x}},\\mathbf{x}^\\star)+\\lambda\\,\\ell(\\mathbf{q},\\mathbf{x}^\\star)$",
            fontsize=FS + 0.2, color=INK, ha='center', va='center', zorder=4)

    # 5 Output
    x = xs[4]
    card(ax, x, y0, w, h, 5, "Coherent output", STEEL, CARD_GRAY)
    ax.text(x + w / 2, 32.15, "$\\mathbf{A}_k\\hat{\\mathbf{x}}=\\mathbf{s}_k$ for every view",
            fontsize=FS + 0.2, color=INK, ha='center', va='center', zorder=4)
    ax.text(x + w / 2, 30.55, "max. violation $<10^{-5}$", fontsize=FS - 0.2, color=INK2, ha='center',
            va='center', zorder=4)
    for i, t in enumerate(["certain answers (row space)", "peak, ramp, load factor",
                           "capture price, arbitrage", "inverter clipping"]):
        yy = 27.55 - i * 2.05
        inner(ax, x + 0.8, yy, w - 1.6, 1.75, ec=STEEL)
        ax.text(x + w / 2, yy + 0.86, t, fontsize=FS - 0.2, color=INK, ha='center', va='center', zorder=4)

    # pipeline arrows
    for i, yy in [(0, 30.6), (1, 23.25), (2, 31.6), (3, 26.6)]:
        arrow(ax, (xs[i] + w + 0.15, yy), (xs[i + 1] - 0.15, yy), col=INK2, lw=1.0)
    # gradient back into the encoder
    arrow(ax, (xs[3] - 0.15, 21.9), (xs[2] + w + 0.15, 21.9), col=CORAL, lw=0.9, ls=(0, (2.5, 1.2)))
    ax.text(xs[3] - g / 2, 20.55, "$\\nabla_\\theta$", fontsize=FS, color=CORAL, ha='center', va='center')

    # ------------------------------------------------------------------ (b) operator
    shadow_card(ax, 0.5, 0.4, 55.0, 17.6, fc='white', ec='#D5DBE3', lw=0.6, r=0.6, zorder=1)
    ax.text(1.4, 16.4, "(b) Coherence layer: one sweep is $K$ grouped updates (Section 3.1)",
            fontsize=FS_T, fontweight='bold', color=INK, va='center')
    # batching by offset group ids
    ax.text(1.6, 13.85, "Batch:", fontsize=FS, fontweight='bold', color=INK, va='center')
    bx = 6.3
    for wd, col, lbl in [(8.4, SLATE, "block 1: $g$"), (10.6, SAGE, "block 2: $g{+}G_1$"),
                         (12.6, AMBER, "block 3: $g{+}G_1{+}G_2$")]:
        rounded_box(ax, bx, 12.95, wd - 0.3, 1.8, fc=col, ec='none', r=0.35, alpha=0.9, zorder=3)
        ax.text(bx + (wd - 0.3) / 2, 13.85, lbl, fontsize=FS - 0.3, color='white', fontweight='bold',
                ha='center', va='center', zorder=4)
        bx += wd
    ax.text(bx + 0.4, 13.85, "one flat cell vector;\ngroup ids offset, no padding",
            fontsize=FS - 0.3, color=INK2, va='center', linespacing=1.15)
    # updates
    inner(ax, 1.4, 6.0, 26.2, 5.9, ec=SAGE, fc=CARD_TEAL)
    ax.text(14.5, 10.85, "KL view update (log domain)", fontsize=FS, fontweight='bold', color=SAGE,
            ha='center', va='center', zorder=4)
    ax.text(14.5, 8.85, "$z_c \\leftarrow z_c+\\log s_{k,j}-\\mathrm{LSE}_{c'\\in j}\\,z_{c'}$",
            fontsize=FS + 0.4, color=INK, ha='center', va='center', zorder=4)
    ax.text(14.5, 6.95, "scatter-max + scatter-add", fontsize=FS - 0.3, color=INK2, ha='center', va='center', zorder=4)
    inner(ax, 28.6, 6.0, 26.0, 5.9, ec=IRIS, fc=CARD_IRIS)
    ax.text(41.6, 10.85, "$\\ell_2$ view update (signed prices)", fontsize=FS, fontweight='bold', color=IRIS,
            ha='center', va='center', zorder=4)
    ax.text(41.6, 8.85, "$q_c \\leftarrow q_c+(s_{k,j}-\\sum_{c'\\in j} q_{c'})/n_{k,j}$",
            fontsize=FS + 0.4, color=INK, ha='center', va='center', zorder=4)
    ax.text(41.6, 6.95, "one scatter-add", fontsize=FS - 0.3, color=INK2, ha='center', va='center', zorder=4)
    # SQL
    rounded_box(ax, 1.4, 1.0, 53.2, 4.2, fc='#1F2937', ec='none', r=0.45, zorder=3)
    ax.text(2.4, 4.05, "-- one view update as a window aggregate", fontsize=FS - 0.6, color='#9CA3AF',
            family='monospace', va='center', zorder=4)
    ax.text(2.4, 2.85, "KL:  x * s / SUM(x) OVER (PARTITION BY g)", fontsize=FS - 0.4, color='#F9FAFB',
            family='monospace', va='center', zorder=4)
    ax.text(2.4, 1.7, "L2:  x + (s - SUM(x) OVER w) / COUNT(*) OVER w", fontsize=FS - 0.4, color='#F9FAFB',
            family='monospace', va='center', zorder=4)

    # ------------------------------------------------------------------ (c) geometry
    shadow_card(ax, 56.5, 0.4, 43.0, 17.6, fc='white', ec='#D5DBE3', lw=0.6, r=0.6, zorder=1)
    ax.text(57.4, 16.4, "(c) Projection geometry (Section 3.2)", fontsize=FS_T, fontweight='bold',
            color=INK, va='center')
    ax.add_patch(Polygon([[58.5, 3.3], [80.5, 2.6], [89.5, 8.9], [67.5, 9.6]], closed=True,
                         fc=CARD_BLUE, ec=SLATE, lw=0.8, zorder=2))
    ax.text(61.4, 4.15, "$\\mathcal{C}$ (coherent set)", fontsize=FS - 0.3, color=SLATE,
            va='center', zorder=3)
    xs_, ys_ = 70.5, 6.5      # truth
    xh, yh = 79.0, 6.1        # projection
    xq, yq = 79.0, 12.9       # seed
    xg, yg = 86.6, 11.8       # gauge-shifted seed
    gxx = np.linspace(73.0, 90.0, 40)
    ax.plot(gxx, 13.05 - 0.012 * (gxx - 79.0) ** 2, color=IRIS, lw=0.9, ls=(0, (3, 1.5)), zorder=3)
    ax.text(90.4, 11.3, "gauge fiber\n$\\mathbf{q}\\odot e^{\\mathbf{A}^{\\top}\\phi}$", fontsize=FS - 0.3,
            color=IRIS, va='center', linespacing=1.1, zorder=3)
    ax.plot([xs_, xq], [ys_, yq], color=MUTED, lw=0.7, ls=':', zorder=3)
    ax.plot([xs_, xh], [ys_, yh], color=SAGE, lw=1.3, zorder=4)
    arrow(ax, (xq, yq - 0.35), (xh, yh + 0.45), col=CORAL, lw=1.0, mut=7)
    arrow(ax, (xg - 0.3, yg - 0.3), (xh + 0.45, yh + 0.4), col=IRIS, lw=0.8, ls=(0, (3, 1.5)), mut=6)
    ax.plot([xh - 0.75, xh - 0.75, xh], [yh + 0.03, yh + 0.75, yh + 0.75], color=INK2, lw=0.5, zorder=4)
    for px, py, col in [(xs_, ys_, INK), (xh, yh, SLATE), (xq, yq, CORAL), (xg, yg, IRIS)]:
        ax.plot(px, py, 'o', ms=3.4, color=col, mec='white', mew=0.5, zorder=6)
    ax.text(xs_ - 0.4, ys_ + 0.95, "truth $\\mathbf{x}^\\star$", fontsize=FS - 0.2, color=INK, ha='center', zorder=6)
    ax.text(xh - 0.3, yh - 1.25, "$\\hat{\\mathbf{x}}=\\Pi_{\\mathcal{C}}(\\mathbf{q})$", fontsize=FS - 0.2,
            color=SLATE, ha='right', va='center', zorder=6)
    ax.text(xq, yq + 1.25, "seed $\\mathbf{q}$", fontsize=FS - 0.2, color=CORAL, ha='center', va='center', zorder=6)
    ax.text(57.4, 1.75, "$D(\\mathbf{x}^\\star\\|\\mathbf{q})=D(\\mathbf{x}^\\star\\|\\hat{\\mathbf{x}})+"
            "D(\\hat{\\mathbf{x}}\\|\\mathbf{q})$: projecting never moves away from $\\mathbf{x}^\\star$",
            fontsize=FS - 0.3, color=INK, va='center', zorder=4)

    os.makedirs(FIG, exist_ok=True)
    os.makedirs(HIGH, exist_ok=True)
    save(fig, os.path.join(FIG, 'framework.pdf'), os.path.join(HIGH, 'fig1_framework.png'))
    plt.close(fig)
    print("Saved framework.pdf and fig1_framework.png")


if __name__ == '__main__':
    draw()
