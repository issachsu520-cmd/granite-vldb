# Declaration of Originality and Non-Use of Artificial Intelligence
#
# The authors declare that no Artificial Intelligence (AI) tools, large language models (LLMs), or generative AI
# technologies were used in the conception, drafting, writing, editing, or polishing of this manuscript. All contents,
# data, analyses, and conclusions are entirely the original work of the authors, who take full responsibility for the
# integrity of the work.
"""Shared figure style for the GRANITE paper.

Figures are drawn at their printed size (PVLDB/acmart: text width 7.0 in, column width 3.33 in), so font sizes
below are the sizes that appear in print. Text is set in Times New Roman with STIX (Times-compatible) math and is
embedded as TrueType (Type 42). Times New Roman is a system font and is not redistributed here; where it is
missing, a metric-compatible free face (Liberation Serif, TeX Gyre Termes, Nimbus Roman) is used instead.
Every method has one fixed color and marker across all figures (METHOD). The five categorical hues pass the
lightness, chroma, color-vision-deficiency and contrast checks of a palette validator, and every series also
carries a distinct marker or a direct label.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

TEXTWIDTH = 7.0     # inches, acmart sigconf \textwidth
COLWIDTH = 3.33     # inches, acmart sigconf \columnwidth

# Colorblind-safe categorical palette (Okabe-Ito hues, deepened for print)
SLATE = '#1F5FA6'   # GRANITE
CORAL = '#D55E00'   # post-hoc projection
SAGE = '#009E73'    # MaxEnt / exact coherence
AMBER = '#B98000'   # profile seed
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

# Sequential (one hue, light to dark) and diverging (two hues, gray midpoint) ramps
SEQ_BLUE = ['#E3EDF7', '#A9C7E6', '#5B92CC', '#1F5FA6', '#0D3463']
DIV_GAIN = ['#C4501A', '#EDB48F', '#F2F2F0', '#93B9E0', '#1F5FA6', '#0D3463']

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

SERIF = ['Times New Roman', 'Liberation Serif', 'TeX Gyre Termes', 'Nimbus Roman', 'DejaVu Serif']


def apply_style(base_size=7.0):
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': SERIF,
        'mathtext.fontset': 'stix',     # Times-compatible math
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


def style_3d(ax, fs=6.0, pad=-3.5):
    """Light panes, hairline grid and tight tick labels for a print-size mplot3d axis."""
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.975, 0.98, 0.985, 1.0))
        axis.pane.set_edgecolor('#C9CED6')
        axis.pane.set_linewidth(0.4)
        axis._axinfo['grid'].update(color=GRID, linewidth=0.4)
        axis._axinfo['tick'].update(inward_factor=0.0, outward_factor=0.25)
        axis.line.set_linewidth(0.5)
        axis.line.set_color(INK2)
        axis.set_tick_params(labelsize=fs - 0.6, pad=pad, colors=INK2, labelcolor=INK)
        axis.label.set_size(fs)
        axis.labelpad = pad + 1.0
    ax.set_facecolor('none')


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


# ---------------------------------------------------------------- isometric (oblique) 3D drawing in 2D axes
def iso(x, y, z, ox=0.0, oy=0.0, sx=1.0, sy=1.0, depth=(0.55, 0.42)):
    """Oblique projection of a 3D point onto 2D data coordinates: x right, z up, y receding up-right."""
    return ox + sx * (x + depth[0] * y), oy + sy * (z + depth[1] * y)


def iso_box(ax, x0, y0, z0, dx, dy, dz, fc_top, fc_front, fc_side, ec=INK2, lw=0.4, zorder=4, alpha=1.0,
            **proj):
    """Draw a solid box (three visible faces) with its front-bottom-left corner at (x0, y0, z0)."""
    P = lambda x, y, z: iso(x, y, z, **proj)
    front = [P(x0, y0, z0), P(x0 + dx, y0, z0), P(x0 + dx, y0, z0 + dz), P(x0, y0, z0 + dz)]
    side = [P(x0 + dx, y0, z0), P(x0 + dx, y0 + dy, z0), P(x0 + dx, y0 + dy, z0 + dz), P(x0 + dx, y0, z0 + dz)]
    top = [P(x0, y0, z0 + dz), P(x0 + dx, y0, z0 + dz), P(x0 + dx, y0 + dy, z0 + dz), P(x0, y0 + dy, z0 + dz)]
    for poly, fc in ((front, fc_front), (side, fc_side), (top, fc_top)):
        ax.add_patch(Polygon(poly, closed=True, fc=fc, ec=ec, lw=lw, zorder=zorder, alpha=alpha,
                             joinstyle='round'))


def save(fig, pdf_path, png_path=None):
    fig.savefig(pdf_path, bbox_inches='tight')
    if png_path:
        fig.savefig(png_path, bbox_inches='tight', dpi=300)
