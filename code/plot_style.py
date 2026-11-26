"""Shared figure style for the GRANITE paper.

Figures are drawn at their printed size (PVLDB/acmart: text width 7.0 in, column width 3.33 in), so font sizes
below are the sizes that appear in print. Text uses Linux Biolinum, the sans-serif face of the acmart template,
embedded as TrueType (Type 42). Every method has one fixed color and marker across all figures (METHOD).
"""
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

TEXTWIDTH = 7.0     # inches, acmart sigconf \textwidth
COLWIDTH = 3.33     # inches, acmart sigconf \columnwidth

# Colorblind-safe palette (Okabe-Ito hues, slightly deepened for print)
SLATE = '#2B5C8A'   # GRANITE
CORAL = '#D55E00'   # post-hoc projection
SAGE = '#009E73'    # MaxEnt / exact coherence
AMBER = '#C98A00'   # profile seed
IRIS = '#7B5EA7'    # GBDT / l2 geometry
SKY = '#56B4E9'     # secondary accent
INK = '#1B1F24'     # text, ground truth
INK2 = '#4A5260'    # secondary text, spines
MUTED = '#9AA3AF'   # reference lines
GRID = '#E7EAEE'    # hairline grid

CARD_BLUE = '#EDF3F9'
CARD_TEAL = '#E8F5F0'
CARD_AMBER = '#FBF4E3'
CARD_CORAL = '#FCEEE6'
CARD_IRIS = '#F1EDF7'
CARD_GRAY = '#F4F5F7'

# One style per method, used by every figure.
METHOD = {
    'granite': dict(color=SLATE, marker='o', label='GRANITE'),
    'posthoc': dict(color=CORAL, marker='s', label='Post-hoc proj.'),
    'prof': dict(color=AMBER, marker='D', label='Profile + proj.'),
    'gbdt': dict(color=IRIS, marker='^', label='GBDT + proj.'),
    'me': dict(color=SAGE, marker='v', label='MaxEnt'),
    'n2proj': dict(color=MUTED, marker='P', label='$N^2$-Norm + proj.'),
    'truth': dict(color=INK, marker=None, label='Truth'),
}

_FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')
_FAMILY = 'DejaVu Sans'
for _f in glob.glob(os.path.join(_FONT_DIR, '*.ttf')):
    font_manager.fontManager.addfont(_f)
    _FAMILY = 'Linux Biolinum O'


def apply_style(base_size=7.0):
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': [_FAMILY, 'DejaVu Sans'],
        'mathtext.fontset': 'custom',
        'mathtext.rm': _FAMILY,
        'mathtext.it': f'{_FAMILY}:italic',
        'mathtext.bf': f'{_FAMILY}:bold',
        'mathtext.sf': _FAMILY,
        'mathtext.cal': 'cmsy10',        # TeX calligraphic (the default 'cursive' resolves to Comic Sans)
        'mathtext.fallback': 'stixsans',
        'font.size': base_size,
        'axes.titlesize': base_size + 0.5,
        'axes.titleweight': 'bold',
        'axes.titlepad': 4,
        'axes.labelsize': base_size,
        'axes.labelpad': 2,
        'axes.edgecolor': INK2,
        'axes.labelcolor': INK,
        'axes.linewidth': 0.6,
        'xtick.color': INK2,
        'ytick.color': INK2,
        'xtick.labelcolor': INK,
        'ytick.labelcolor': INK,
        'xtick.labelsize': base_size - 0.5,
        'ytick.labelsize': base_size - 0.5,
        'xtick.major.pad': 1.5,
        'ytick.major.pad': 1.5,
        'lines.linewidth': 1.1,
        'lines.markersize': 3.6,
        'legend.frameon': False,
        'legend.fontsize': base_size - 0.5,
        'legend.handlelength': 1.4,
        'legend.handletextpad': 0.4,
        'legend.borderaxespad': 0.3,
        'legend.labelspacing': 0.3,
        'legend.columnspacing': 0.9,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.dpi': 300,
        'savefig.pad_inches': 0.02,
    })


def clean_ax(ax, grid_axis='y'):
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=2.4, width=0.55)


def panel_title(ax, text, x=0.0, y=1.0, **kw):
    """Left-aligned bold panel title, e.g. '(a) ...'."""
    ax.set_title(text, loc='left', x=x, y=y, **kw)


def rounded_box(ax, x, y, w, h, fc=CARD_GRAY, ec=INK2, lw=0.6, r=0.02, ls='-', alpha=1.0, zorder=2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
                       alpha=alpha, zorder=zorder)
    ax.add_patch(p)
    return p


def shadow_card(ax, x, y, w, h, fc='white', ec='#CBD5E1', lw=0.6, r=0.5,
                header_h=0.0, header_fc=SLATE, zorder=2, shadow_dx=0.0, shadow_dy=0.0):
    """Card with an optional colored header band (no drop shadow by default)."""
    if shadow_dx or shadow_dy:
        ax.add_patch(FancyBboxPatch((x + shadow_dx, y + shadow_dy), w, h,
                                    boxstyle=f"round,pad=0,rounding_size={r}",
                                    facecolor='#E2E8F0', edgecolor='none', alpha=0.55, zorder=max(1, zorder - 1)))
    card = rounded_box(ax, x, y, w, h, fc=fc, ec=ec, lw=lw, r=r, zorder=zorder)
    if header_h > 0:
        ax.add_patch(FancyBboxPatch((x, y + h - header_h), w, header_h,
                                    boxstyle=f"round,pad=0,rounding_size={r}",
                                    facecolor=header_fc, edgecolor=header_fc, lw=lw, zorder=zorder + 0.2))
    return card


def pill_badge(ax, x, y, text, fc='white', ec=SLATE, tc=INK, fs=6.3, fw='bold',
               ha='center', va='center', pad=0.22, alpha=1.0, zorder=8, **kw):
    return ax.text(x, y, text, fontsize=fs, fontweight=fw, color=tc, ha=ha, va=va, zorder=zorder,
                   bbox=dict(boxstyle=f'round,pad={pad}', facecolor=fc, edgecolor=ec, linewidth=0.6, alpha=alpha),
                   **kw)


def save(fig, pdf_path, png_path=None):
    fig.savefig(pdf_path, bbox_inches='tight')
    if png_path:
        fig.savefig(png_path, bbox_inches='tight', dpi=300)
