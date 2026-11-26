"""Figure 1: overview of GRANITE, drawn at print size (7.0 x 3.1 in).

(a) pipeline: release -> coherent anchor -> dual-stream encoder -> coherence layer -> coherent output.
    The hidden table is drawn as a data cube R(entity, day, hour) whose released views are marginal faces.
(b) the coherence layer: one view partitions the cube into groups (exploded slabs); batching by group-id
    offsets; the KL / l2 view updates; their SQL window-aggregate form.
(c) exact 3D geometry of the projection for a 2x2 table with a row-sum view and a column-sum view.
    The tetrahedron is the simplex of all 2x2 tables (barycentric coordinates). In the coordinates
    X = 2r - 1 (row share), Y = 2c - 1 (column share), Z = x11 - x12 - x21 + x22 (association), the coherent
    set C is a vertical segment and the gauge fiber of a seed q (all q * exp(row effect + column effect),
    i.e. all tables with the odds ratio of q) is a curved sheet. Cyclic projection (IPF) never leaves the sheet
    and stops where the sheet meets C. The Pythagorean identity is evaluated numerically.

Writes paper/figs/framework.pdf and submission_package/figures_highres/fig1_framework.png.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

from plot_style import (SLATE, CORAL, SAGE, AMBER, IRIS, INK, INK2, MUTED, GRID, CARD_BLUE, CARD_TEAL,
                        CARD_CORAL, CARD_IRIS, CARD_GRAY, SEQ_BLUE, TEXTWIDTH, apply_style, rounded_box,
                        shadow_card, iso, iso_box, save)

from paths import FIGS as FIG
HIGH = FIG
apply_style(6.8)

STEEL = '#46566A'
FS = 6.6      # body text
FS_H = 7.0    # card headers
FS_T = 7.9    # panel titles
H_IN = 3.1
YMAX = 100.0 * H_IN / TEXTWIDTH      # equal data units per inch on both axes


def arrow(ax, p1, p2, col=INK2, lw=0.9, ls='-', rad=0.0, mut=7, style='-|>', zorder=6):
    ax.add_patch(FancyArrowPatch(p1, p2, connectionstyle=f"arc3,rad={rad}", arrowstyle=style,
                                 mutation_scale=mut, color=col, linewidth=lw, linestyle=ls, zorder=zorder,
                                 shrinkA=0, shrinkB=0))


def card(ax, x, y, w, h, num, title, col, fc):
    shadow_card(ax, x, y, w, h, fc=fc, ec=col, lw=0.7, r=0.6, header_h=2.5, header_fc=col, zorder=2)
    ax.add_patch(Circle((x + 1.45, y + h - 1.25), 0.85, fc='white', ec='none', zorder=4))
    ax.text(x + 1.45, y + h - 1.27, str(num), fontsize=6.3, fontweight='bold', color=col,
            ha='center', va='center', zorder=5)
    ax.text(x + 2.75, y + h - 1.25, title, fontsize=FS_H, fontweight='bold', color='white',
            ha='left', va='center', zorder=5)


def inner(ax, x, y, w, h, ec, fc='white', ls='-', lw=0.6):
    rounded_box(ax, x, y, w, h, fc=fc, ec=ec, lw=lw, r=0.45, ls=ls, zorder=3)


def face_grid(ax, corners, n_u, n_v, col='white', lw=0.35, zorder=5):
    """Cell lines on a parallelogram face given as [p00, p10, p11, p01]."""
    p00, p10, _, p01 = [np.asarray(c) for c in corners]
    u, v = p10 - p00, p01 - p00
    for i in range(1, n_u):
        a = p00 + u * i / n_u
        ax.plot([a[0], (a + v)[0]], [a[1], (a + v)[1]], color=col, lw=lw, zorder=zorder)
    for j in range(1, n_v):
        a = p00 + v * j / n_v
        ax.plot([a[0], (a + u)[0]], [a[1], (a + u)[1]], color=col, lw=lw, zorder=zorder)


def data_cube(ax, ox, oy, dx, dy, dz, top, front, side, grid=(6, 5, 4), qmark=True):
    """Oblique data cube R(entity, day, hour) with cell grids on the three visible faces."""
    P = lambda x, y, z: iso(x, y, z, ox=ox, oy=oy)
    iso_box(ax, 0, 0, 0, dx, dy, dz, top, front, side, ec=INK2, lw=0.5, zorder=4, ox=ox, oy=oy)
    ne, nd, nh = grid
    face_grid(ax, [P(0, 0, 0), P(dx, 0, 0), P(dx, 0, dz), P(0, 0, dz)], ne, nh)
    face_grid(ax, [P(0, 0, dz), P(dx, 0, dz), P(dx, dy, dz), P(0, dy, dz)], ne, nd)
    face_grid(ax, [P(dx, 0, 0), P(dx, dy, 0), P(dx, dy, dz), P(dx, 0, dz)], nd, nh)
    if qmark:
        cx, cy = P(dx / 2, 0, dz / 2)
        ax.text(cx, cy, "?", fontsize=8.5, fontweight='bold', color=STEEL, ha='center', va='center', zorder=6)
    return P


# ---------------------------------------------------------------- geometry of panel (c)
def table_from_margins(r, c, theta):
    """2x2 table (x11, x12, x21, x22) with row share r, column share c and odds ratio theta."""
    lo, hi = max(0.0, r + c - 1.0), min(r, c)
    if abs(theta - 1.0) < 1e-12:
        t = r * c
    else:
        a, b, cc = 1.0 - theta, 1.0 - r - c + theta * (r + c), -theta * r * c
        disc = np.sqrt(max(b * b - 4 * a * cc, 0.0))
        roots = [(-b + disc) / (2 * a), (-b - disc) / (2 * a)]
        t = [x for x in roots if lo - 1e-12 <= x <= hi + 1e-12][0]
    return np.array([t, r - t, c - t, 1.0 - r - c + t])


def xyz(p):
    """Coordinates of a 2x2 table inside the regular tetrahedron with vertices (+-1, +-1, +-1)."""
    p11, p12, p21, p22 = p
    return np.array([p11 + p12 - p21 - p22, p11 - p12 + p21 - p22, p11 - p12 - p21 + p22])


def odds(p):
    return p[0] * p[3] / (p[1] * p[2])


def ipf(p, r, c, sweeps=6):
    """Cyclic KL projection onto {row share r} and {column share c}; returns all half-step iterates."""
    pts = [p.copy()]
    x = p.copy()
    for _ in range(sweeps):
        rr = x[0] + x[1]
        x = x * np.array([r / rr, r / rr, (1 - r) / (1 - rr), (1 - r) / (1 - rr)])
        pts.append(x.copy())
        cc = x[0] + x[2]
        x = x * np.array([c / cc, (1 - c) / (1 - cc), c / cc, (1 - c) / (1 - cc)])
        pts.append(x.copy())
    return np.array(pts)


def kl(a, b):
    return float(np.sum(a * np.log(a / b)))


def draw_geometry(fig, ax):
    """Panel (c): exact geometry of the KL projection for a 2x2 table, row and column views."""
    R_T, C_T = 0.75, 0.65                                   # released row share and column share
    q = np.array([0.08, 0.22, 0.30, 0.40])                  # seed
    theta = odds(q)
    g = table_from_margins(0.30, 0.92, theta)               # a gauge-shifted seed: same odds ratio as q
    xhat = table_from_margins(R_T, C_T, theta)
    xstar = table_from_margins(R_T, C_T, 4.0)               # a truth on C
    xme = table_from_margins(R_T, C_T, 1.0)                 # anchor Pi_C(1)
    path_q, path_g = ipf(q, R_T, C_T), ipf(g, R_T, C_T)
    assert np.allclose(path_q[-1], xhat, atol=1e-6) and np.allclose(path_g[-1], xhat, atol=1e-6)
    assert np.allclose([odds(p) for p in path_q], theta) and np.allclose([odds(p) for p in path_g], theta)
    d_total, d_a, d_b = kl(xstar, q), kl(xstar, xhat), kl(xhat, q)
    assert abs(d_total - d_a - d_b) < 1e-12

    x0, y0, w, h = 51.7 / 100, 0.7 / YMAX, 28.0 / 100, 17.2 / YMAX
    a3 = fig.add_axes([x0, y0, w, h], projection='3d', computed_zorder=False)
    a3.set_axis_off()
    a3.view_init(elev=19, azim=-60)
    a3.set_box_aspect((1, 1, 0.78), zoom=1.3)
    a3.set_xlim(-1, 1)
    a3.set_ylim(-1, 1)
    a3.set_zlim(-1.3, 1)
    FL = -1.3                                               # floor height

    # floor: the (row share, column share) square with a hairline grid
    for v in np.linspace(-1, 1, 5):
        a3.plot([v, v], [-1, 1], [FL, FL], color=GRID, lw=0.45, zorder=0)
        a3.plot([-1, 1], [v, v], [FL, FL], color=GRID, lw=0.45, zorder=0)
    a3.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], [FL] * 5, color='#C9CED6', lw=0.5, zorder=0)

    # fiber of q: all tables with the odds ratio of q (a curved sheet over the square)
    rs = np.linspace(0.05, 0.97, 37)
    S = np.array([[xyz(table_from_margins(r, c, theta)) for c in rs] for r in rs])
    a3.plot_surface(S[..., 0], S[..., 1], S[..., 2], color='#A9C9E8', alpha=0.32, linewidth=0,
                    antialiased=False, shade=False, zorder=1)
    for i in range(0, 37, 4):
        a3.plot(S[i, :, 0], S[i, :, 1], S[i, :, 2], color=SLATE, lw=0.28, alpha=0.5, zorder=1.5)
        a3.plot(S[:, i, 0], S[:, i, 1], S[:, i, 2], color=SLATE, lw=0.28, alpha=0.5, zorder=1.5)

    # tetrahedron (simplex of 2x2 tables)
    V = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    for i in range(4):
        for j in range(i + 1, 4):
            a3.plot(*zip(V[i], V[j]), color=INK2, lw=0.55, alpha=0.85, zorder=2)

    # coherent set C: vertical segment (row and column shares fixed)
    lo, hi = xyz(table_from_margins(R_T, C_T, 1e-6)), xyz(table_from_margins(R_T, C_T, 1e6))
    a3.plot([lo[0], hi[0]], [lo[1], hi[1]], [lo[2], hi[2]], color=SAGE, lw=2.0, zorder=3, solid_capstyle='round')
    a3.plot([lo[0]], [lo[1]], [FL], marker='+', color=SAGE, ms=4, mew=0.8, zorder=3)
    a3.plot([lo[0], lo[0]], [lo[1], lo[1]], [FL, lo[2]], color=SAGE, lw=0.5, ls=(0, (1, 1.2)), zorder=2)

    # cyclic projections (IPF) on the sheet, their floor shadows and drop lines
    for path, col, ls in ((path_q, CORAL, '-'), (path_g, IRIS, (0, (2.2, 1.0)))):
        P = np.array([xyz(p) for p in path])
        a3.plot(P[:, 0], P[:, 1], P[:, 2], color=col, lw=0.95, ls=ls, zorder=4)
        a3.scatter(P[1:-1, 0], P[1:-1, 1], P[1:-1, 2], s=3.5, color=col, edgecolor='none', zorder=4,
                   depthshade=False)
        a3.plot(P[:, 0], P[:, 1], [FL] * len(P), color=col, lw=0.6, ls=ls, alpha=0.55, zorder=0.5)
        a3.plot([P[0, 0]] * 2, [P[0, 1]] * 2, [FL, P[0, 2]], color=col, lw=0.45, ls=(0, (1, 1.2)), zorder=2)

    pts = {'q': (q, CORAL, CORAL, 16), 'g': (g, IRIS, IRIS, 16), 'xhat': (xhat, SLATE, SLATE, 22),
           'xstar': (xstar, INK, INK, 15), 'xme': (xme, SAGE, 'white', 15)}
    for p, col, fc, s in pts.values():
        c = xyz(p)
        a3.scatter([c[0]], [c[1]], [c[2]], s=s, color=fc, edgecolor=col if fc == 'white' else 'white',
                   linewidth=0.6, zorder=8, depthshade=False)

    # labels are placed in 2D (overlay axis) at the projected positions, with short leaders
    fig.canvas.draw()
    from mpl_toolkits.mplot3d import proj3d
    ov = fig.add_axes([0, 0, 1, 1], zorder=10)
    ov.set_xlim(0, 100)
    ov.set_ylim(0, YMAX)
    ov.axis('off')

    def to2d(c):
        u, v, _ = proj3d.proj_transform(c[0], c[1], c[2], a3.get_proj())
        return ov.transData.inverted().transform(a3.transData.transform((u, v)))

    def tag(c3, text, col, dx, dy, ha='left', fs=6.4, leader=True):
        px, py = to2d(c3)
        ov.text(px + dx, py + dy, text, fontsize=fs, color=col, ha=ha, va='center')
        if leader:
            ov.plot([px, px + dx - (0.25 if ha == 'left' else -0.25)], [py, py + dy], color=col, lw=0.4)

    tag(xyz(q), '$\\mathbf{q}$', CORAL, -1.6, 0.9, ha='right')
    tag(xyz(g), "$\\mathbf{q}'$", IRIS, -1.6, 0.9, ha='right')
    tag(xyz(xstar), '$\\mathbf{x}^\\star$', INK, 1.9, 0.5)
    tag(xyz(xme), '$\\mathbf{x}^{\\mathrm{ME}}$', SAGE, 1.9, 0.2)
    tag(xyz(xhat), '$\\hat{\\mathbf{x}}$', SLATE, 1.9, -0.5)
    for v, lbl, (dx, dy) in zip(V, ['$e_{11}$', '$e_{12}$', '$e_{21}$', '$e_{22}$'],
                                [(0.9, 0.3), (0.9, -0.2), (-0.8, -0.5), (-0.9, 0.3)]):
        tag(v, lbl, INK2, dx, dy, ha='left' if dx > 0 else 'right', fs=5.6, leader=False)
    px, py = to2d((-0.1, -1.0, FL))
    ov.text(px, py - 0.7, 'row share $r$', fontsize=5.5, color=INK2, ha='center', va='top')
    px, py = to2d((1.0, 0.35, FL))
    ov.text(px + 0.8, py - 0.3, 'column\nshare $c$', fontsize=5.5, color=INK2, ha='left', va='center',
            linespacing=1.0)

    # legend column
    lx, ly, dy = 80.4, 15.3, 1.55
    items = [
        ('seg', SAGE, r"$\mathcal{C}$: tables matching both views"),
        ('sheet', '#9CC3E6', "gauge fiber of $\\mathbf{q}$ (same odds ratio)"),
        ('dot', CORAL, "seed $\\mathbf{q}$; cyclic sweeps"),
        ('dot', IRIS, "$\\mathbf{q}'=\\mathbf{q}\\odot e^{\\mathbf{A}^{\\top}\\phi}$; sweeps"),
        ('dot', SLATE, "$\\hat{\\mathbf{x}}=\\Pi_{\\mathcal{C}}(\\mathbf{q})=\\Pi_{\\mathcal{C}}(\\mathbf{q}')$"),
        ('dot', INK, "truth $\\mathbf{x}^\\star\\in\\mathcal{C}$"),
        ('ring', SAGE, "anchor $\\mathbf{x}^{\\mathrm{ME}}=\\Pi_{\\mathcal{C}}(\\mathbf{1})$"),
    ]
    for i, (kind, col, txt) in enumerate(items):
        yy = ly - i * dy
        if kind == 'seg':
            ax.plot([lx - 0.2, lx + 1.4], [yy, yy], color=col, lw=2.0, solid_capstyle='round')
        elif kind == 'sheet':
            ax.add_patch(Rectangle((lx - 0.2, yy - 0.45), 1.6, 0.9, fc=col, alpha=0.45, ec=SLATE, lw=0.3))
        elif kind == 'ring':
            ax.plot(lx + 0.6, yy, 'o', ms=3.3, mfc='white', mec=col, mew=0.7)
        else:
            ax.plot(lx + 0.6, yy, 'o', ms=3.3, color=col, mec='white', mew=0.4)
        ax.text(lx + 2.1, yy, txt, fontsize=FS - 0.7, color=INK, va='center')
    ax.text(lx - 0.3, 4.15, "Pythagorean identity (exact):", fontsize=FS - 0.6, color=INK2, va='center')
    ax.text(lx - 0.3, 2.7, "$D(\\mathbf{x}^\\star\\|\\mathbf{q})=D(\\mathbf{x}^\\star\\|\\hat{\\mathbf{x}})"
            "+D(\\hat{\\mathbf{x}}\\|\\mathbf{q})$", fontsize=FS - 0.3, color=INK, va='center')
    ax.text(lx - 0.3, 1.35, f"${d_total:.3f}={d_a:.3f}+{d_b:.3f}$ nats", fontsize=FS - 0.3, color=INK,
            va='center')
    return dict(theta=theta, xhat=xhat, xstar=xstar, d=(d_total, d_a, d_b))


def draw():
    fig = plt.figure(figsize=(TEXTWIDTH, H_IN))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, YMAX)
    ax.axis('off')

    # ------------------------------------------------------------------ (a) pipeline
    ax.text(0.6, YMAX - 1.05, "(a) Pipeline: training through the exact coherence layer (Sections 2–4)",
            fontsize=FS_T, fontweight='bold', color=INK, va='center')
    w, gap, y0, h = 17.88, 2.4, 21.0, 20.4
    top = y0 + h - 2.5
    xs = [0.5 + i * (w + gap) for i in range(5)]

    # 1 Release: the hidden data cube; views are marginal faces
    x = xs[0]
    card(ax, x, y0, w, h, 1, "Aggregate release", STEEL, CARD_GRAY)
    ox, oy = x + 2.0, 30.3
    P = data_cube(ax, ox, oy, 6.0, 5.4, 4.3, top='#BFD5EC', front='#E4E9EF', side='#BFE4D6')
    ax.text(*iso(3.0, 0, -0.75, ox=ox, oy=oy), "entity", fontsize=5.5, color=INK2, ha='center', va='center')
    ax.text(*iso(-0.7, 0, 2.15, ox=ox, oy=oy), "hour", fontsize=5.5, color=INK2, ha='center', va='center',
            rotation=90)
    ax.text(*iso(6.75, 2.7, -0.55, ox=ox, oy=oy), "day", fontsize=5.5, color=INK2, ha='center', va='center',
            rotation=34)
    ax.text(x + 12.3, 36.0, "hidden", fontsize=FS - 0.2, color=INK, va='center')
    ax.text(x + 12.3, 34.7, "fine table", fontsize=FS - 0.2, color=INK, va='center')
    ax.text(x + 12.3, 33.3, "$\\mathbf{x}$", fontsize=FS + 0.6, color=INK, va='center')
    for i, (col, txt) in enumerate([(SLATE, "V1  province × month, sum"),
                                    (SAGE, "V2  region × hour, sum"),
                                    (IRIS, "V$K$  day / TOU, average")]):
        yy = 26.6 - i * 2.0
        inner(ax, x + 0.8, yy, w - 1.6, 1.7, ec=col)
        ax.add_patch(Rectangle((x + 1.25, yy + 0.42), 0.45, 0.86, fc=col, ec='none', zorder=4))
        ax.text(x + 2.15, yy + 0.84, txt, fontsize=FS - 0.4, color=INK, va='center', zorder=4)

    # 2 Anchor
    x = xs[1]
    card(ax, x, y0, w, h, 2, "Coherent anchor", SAGE, CARD_TEAL)
    inner(ax, x + 0.8, 31.6, w - 1.6, 6.6, ec=SAGE)
    ax.text(x + w / 2, 36.5, "$\\mathbf{x}^{\\mathrm{ME}}=\\Pi_{\\mathcal{C}}(\\mathbf{1})$ (KL)",
            fontsize=FS + 0.3, color=INK, ha='center', va='center', zorder=4)
    ax.text(x + w / 2, 34.7, "or $\\Pi_{\\mathcal{C}}(\\mathbf{0})$ ($\\ell_2$)",
            fontsize=FS + 0.3, color=INK, ha='center', va='center', zorder=4)
    ax.text(x + w / 2, 32.75, "max. entropy, no gradient", fontsize=FS - 0.3, color=INK2,
            ha='center', va='center', zorder=4)
    arrow(ax, (x + w / 2, 31.6), (x + w / 2, 30.3), col=SAGE)
    inner(ax, x + 0.8, 21.8, w - 1.6, 8.5, ec=SAGE)
    ax.text(x + w / 2, 28.6, "Scale-free inputs", fontsize=FS, fontweight='bold', color=INK, ha='center',
            va='center', zorder=4)
    for i, t in enumerate(["normalized anchor,", "view means, coverage,", "weather, calendar $\\mathbf{W}$"]):
        ax.text(x + w / 2, 26.6 - i * 1.6, t, fontsize=FS - 0.2, color=INK2, ha='center', va='center', zorder=4)

    # 3 Encoder: stacked token planes (x3 blocks), axial attention along days and entities
    x = xs[2]
    card(ax, x, y0, w, h, 3, "Dual-stream encoder ×3", SLATE, CARD_BLUE)
    inner(ax, x + 0.8, 30.9, w - 1.6, 7.3, ec=SLATE)
    lox, loy = x + 1.6, 31.9
    for k, z in enumerate((0.0, 1.35, 2.7)):
        fcs = ('#DCE6F2', '#C3D5EA', '#A9C4E2')[k]
        iso_box(ax, 0, 0, z, 4.6, 4.0, 0.32, fc_top=fcs, fc_front='#8FB0D6', fc_side='#7FA3CC', ec=INK2, lw=0.35,
                zorder=4 + k, ox=lox, oy=loy)
    zt = 2.7 + 0.32
    for i in range(4):          # day tokens on the top plane: highlight one day row and one entity column
        for j in range(4):
            fc = '#F2C46D' if j == 1 else ('#5B92CC' if i == 2 else None)
            if fc is None:
                continue
            cell = [iso(i * 1.15, j * 1.0, zt, ox=lox, oy=loy), iso((i + 1) * 1.15, j * 1.0, zt, ox=lox, oy=loy),
                    iso((i + 1) * 1.15, (j + 1) * 1.0, zt, ox=lox, oy=loy), iso(i * 1.15, (j + 1) * 1.0, zt, ox=lox, oy=loy)]
            ax.add_patch(Polygon(cell, closed=True, fc=fc, ec='white', lw=0.3, zorder=8, alpha=0.95))
    ax.text(x + 9.2, 36.7, "Day tokens $\\mathbf{H}$", fontsize=FS, fontweight='bold', color=INK, va='center', zorder=9)
    ax.text(x + 9.2, 35.0, "axial attention", fontsize=FS - 0.3, color=INK2, va='center', zorder=9)
    ax.text(x + 9.2, 33.6, "over days and", fontsize=FS - 0.3, color=INK2, va='center', zorder=9)
    ax.text(x + 9.2, 32.2, "entities", fontsize=FS - 0.3, color=INK2, va='center', zorder=9)
    arrow(ax, (x + w / 2, 30.85), (x + w / 2, 29.65), col=SLATE, style='<|-|>')
    ax.text(x + w / 2 + 0.7, 30.25, "patch / unpatch", fontsize=FS - 0.7, color=INK2, va='center')
    inner(ax, x + 0.8, 21.8, w - 1.6, 7.8, ec=SLATE)
    ax.text(x + w / 2, 28.1, "Hourly cells $\\mathbf{h}_c$", fontsize=FS, fontweight='bold', color=INK,
            ha='center', va='center', zorder=4)
    xx = np.linspace(x + 2.0, x + w - 2.0, 60)
    ph = (xx - xx[0]) / (xx[-1] - xx[0])
    ax.plot(xx, 25.6 + 0.65 * np.sin(ph * 4 * np.pi) + 0.25 * np.sin(ph * 12 * np.pi), color=SLATE, lw=0.8,
            zorder=4)
    ax.text(x + w / 2, 23.75, "messages along", fontsize=FS - 0.2, color=INK2, ha='center', va='center', zorder=4)
    ax.text(x + w / 2, 22.45, "view groups", fontsize=FS - 0.2, color=INK2, ha='center', va='center', zorder=4)

    # 4 Coherence layer
    x = xs[3]
    card(ax, x, y0, w, h, 4, "Coherence layer", IRIS, CARD_IRIS)
    inner(ax, x + 0.8, 34.4, w - 1.6, 3.8, ec=IRIS)
    ax.text(x + w / 2, 37.15, "Seed", fontsize=FS, fontweight='bold', color=INK, ha='center', va='center', zorder=4)
    ax.text(x + w / 2, 35.45, "$\\log\\mathbf{q}=\\log\\mathbf{x}^{\\mathrm{ME}}+\\mathbf{f}_\\theta$",
            fontsize=FS + 0.2, color=INK, ha='center', va='center', zorder=4)
    arrow(ax, (x + w / 2, 34.4), (x + w / 2, 33.2), col=IRIS)
    inner(ax, x + 0.8, 28.4, w - 1.6, 4.8, ec=IRIS, lw=1.1)
    ax.text(x + w / 2, 31.55, "$\\hat{\\mathbf{x}}=\\Pi_{\\mathcal{C}}(\\mathbf{q})$, $I$ sweeps",
            fontsize=FS + 0.3, color=INK, ha='center', va='center', zorder=4)
    ax.text(x + w / 2, 29.55, "exact, $O(KN)$ per sweep", fontsize=FS - 0.2, color=INK2, ha='center',
            va='center', zorder=4)
    inner(ax, x + 0.8, 21.8, w - 1.6, 5.6, ec=CORAL, fc=CARD_CORAL, ls=(0, (3, 1.5)))
    ax.text(x + w / 2, 25.85, "training loss", fontsize=FS - 0.3, color=CORAL, ha='center',
            va='center', zorder=4)
    ax.text(x + w / 2, 23.6, "$\\ell(\\hat{\\mathbf{x}},\\mathbf{x}^\\star)+\\lambda\\,\\ell(\\mathbf{q},\\mathbf{x}^\\star)$",
            fontsize=FS + 0.2, color=INK, ha='center', va='center', zorder=4)

    # 5 Output: a filled, coherent cube
    x = xs[4]
    card(ax, x, y0, w, h, 5, "Coherent output", STEEL, CARD_GRAY)
    vox, voy = x + 1.6, 31.3
    rng = np.random.default_rng(7)
    n, s = 3, 1.45
    for k in range(n):              # back-to-front, bottom-to-top so that near voxels overdraw far ones
        for j in range(n - 1, -1, -1):
            for i in range(n):
                v = rng.integers(1, 5)
                iso_box(ax, i * s, j * s, k * s, s, s, s, fc_top=SEQ_BLUE[v - 1], fc_front=SEQ_BLUE[v],
                        fc_side=SEQ_BLUE[min(v + 1, 4)], ec='white', lw=0.3, zorder=4 + k * 0.01, ox=vox, oy=voy)
    ax.text(x + 9.0, 36.7, "$\\mathbf{A}_k\\hat{\\mathbf{x}}=\\mathbf{s}_k$", fontsize=FS + 0.5, color=INK,
            va='center', zorder=4)
    ax.text(x + 9.0, 35.0, "for every view", fontsize=FS - 0.2, color=INK2, va='center', zorder=4)
    ax.text(x + 9.0, 33.4, "max. violation", fontsize=FS - 0.2, color=INK2, va='center', zorder=4)
    ax.text(x + 9.0, 31.9, "$<10^{-5}$", fontsize=FS - 0.2, color=INK2, va='center', zorder=4)
    for i, t in enumerate(["certain answers (row space)", "peak, ramp, load factor",
                           "capture price, arbitrage", "inverter clipping"]):
        yy = 28.6 - i * 1.95
        inner(ax, x + 0.8, yy, w - 1.6, 1.65, ec=STEEL)
        ax.text(x + w / 2, yy + 0.82, t, fontsize=FS - 0.3, color=INK, ha='center', va='center', zorder=4)

    # pipeline arrows
    for i, yy in [(0, 34.0), (1, 26.0), (2, 35.2), (3, 30.8)]:
        arrow(ax, (xs[i] + w + 0.15, yy), (xs[i + 1] - 0.15, yy), col=INK2, lw=1.0)
    arrow(ax, (xs[3] - 0.15, 23.9), (xs[2] + w + 0.15, 23.9), col=CORAL, lw=0.9, ls=(0, (2.5, 1.2)))
    ax.text(xs[3] - gap / 2, 22.5, "$\\nabla_\\theta$", fontsize=FS, color=CORAL, ha='center', va='center')

    # ------------------------------------------------------------------ (b) operator
    shadow_card(ax, 0.5, 0.4, 50.0, 19.6, fc='white', ec='#D5DBE3', lw=0.6, r=0.6, zorder=1)
    ax.text(1.4, 18.65, "(b) Coherence layer: one sweep is $K$ grouped updates (Section 3.1)",
            fontsize=FS_T, fontweight='bold', color=INK, va='center')
    # one view partitions the cube into groups (exploded slabs along the entity axis)
    ox, oy = 2.3, 4.6
    gaps = 0.55
    for gi in range(4):
        hi = gi == 1
        iso_box(ax, gi * (1.75 + gaps), 0, 0, 1.75, 5.6, 6.2,
                fc_top='#8FB4DD' if hi else '#E3E8EF', fc_front='#5B92CC' if hi else '#EEF1F5',
                fc_side='#3F77B5' if hi else '#D6DCE4', ec=SLATE if hi else INK2, lw=0.55 if hi else 0.4,
                zorder=4, ox=ox, oy=oy)
    ax.text(*iso(4.5, 0, -0.9, ox=ox, oy=oy), "groups $j$ of view $k$", fontsize=FS - 0.6, color=INK2,
            ha='center', va='center')
    hx, hy = iso(2.3 + 0.9, 2.8, 6.2, ox=ox, oy=oy)
    ax.text(hx, hy + 3.6, "$\\Sigma_{c\\in j}\\,x_c\\;\\to\\;s_{k,j}$", fontsize=FS + 0.1, color=SLATE, ha='center',
            va='center')
    arrow(ax, (hx, hy + 2.6), (hx, hy + 0.35), col=SLATE, lw=0.7, mut=5)

    rx = 15.2
    bx = rx
    for wd, col, lbl in [(5.4, SLATE, "1: $g$"), (7.4, SAGE, "2: $g{+}G_1$"), (9.8, AMBER, "3: $g{+}G_1{+}G_2$")]:
        rounded_box(ax, bx, 15.1, wd - 0.3, 1.75, fc=col, ec='none', r=0.35, alpha=0.92, zorder=3)
        ax.text(bx + (wd - 0.3) / 2, 15.98, lbl, fontsize=FS - 0.4, color='white', fontweight='bold',
                ha='center', va='center', zorder=4)
        bx += wd
    ax.text(bx + 0.3, 15.98, "one flat batch vector,\nids offset, no padding", fontsize=FS - 0.9,
            color=INK2, va='center', linespacing=1.1)
    inner(ax, rx, 10.0, 34.6, 4.5, ec=SAGE, fc=CARD_TEAL)
    ax.text(rx + 0.7, 13.5, "KL view update (log domain)", fontsize=FS - 0.1, fontweight='bold', color=SAGE,
            va='center', zorder=4)
    ax.text(rx + 33.9, 13.5, "scatter-max + scatter-add", fontsize=FS - 0.6, color=INK2, ha='right',
            va='center', zorder=4)
    ax.text(rx + 17.3, 11.35, "$z_c \\leftarrow z_c+\\log s_{k,j}-\\mathrm{LSE}_{c'\\in j}\\,z_{c'}$",
            fontsize=FS + 0.5, color=INK, ha='center', va='center', zorder=4)
    inner(ax, rx, 5.1, 34.6, 4.5, ec=IRIS, fc=CARD_IRIS)
    ax.text(rx + 0.7, 8.6, "$\\ell_2$ view update (signed prices)", fontsize=FS - 0.1, fontweight='bold',
            color=IRIS, va='center', zorder=4)
    ax.text(rx + 33.9, 8.6, "one scatter-add", fontsize=FS - 0.6, color=INK2, ha='right', va='center', zorder=4)
    ax.text(rx + 17.3, 6.4, "$q_c \\leftarrow q_c+(s_{k,j}-\\Sigma_{c'\\in j}\\,q_{c'})/n_{k,j}$",
            fontsize=FS + 0.5, color=INK, ha='center', va='center', zorder=4)
    rounded_box(ax, rx, 0.9, 34.6, 4.0, fc='#1F2937', ec='none', r=0.45, zorder=3)
    ax.text(rx + 0.8, 3.95, "-- one view update = one window aggregate", fontsize=FS - 1.1,
            color='#9CA3AF', family='monospace', va='center', zorder=4)
    ax.text(rx + 0.8, 2.85, "KL:  x * s / SUM(x) OVER (PARTITION BY g)", fontsize=FS - 1.0, color='#F9FAFB',
            family='monospace', va='center', zorder=4)
    ax.text(rx + 0.8, 1.75, "L2:  x + (s - SUM(x) OVER w) / COUNT(*) OVER w", fontsize=FS - 1.0, color='#F9FAFB',
            family='monospace', va='center', zorder=4)
    # ------------------------------------------------------------------ (c) geometry (3D)
    shadow_card(ax, 51.5, 0.4, 48.0, 19.6, fc='white', ec='#D5DBE3', lw=0.6, r=0.6, zorder=1)
    ax.text(52.4, 18.65, "(c) Exact projection geometry, 2×2 table (Section 3.2)", fontsize=FS_T,
            fontweight='bold', color=INK, va='center')
    geo = draw_geometry(fig, ax)

    os.makedirs(FIG, exist_ok=True)
    os.makedirs(HIGH, exist_ok=True)
    save(fig, os.path.join(FIG, 'framework.pdf'), os.path.join(HIGH, 'fig1_framework.png'))
    plt.close(fig)
    print("Saved framework.pdf and fig1_framework.png;",
          f"odds ratio {geo['theta']:.4f}, xhat {np.round(geo['xhat'], 4)}, D = {np.round(geo['d'], 4)}")


if __name__ == '__main__':
    draw()
