"""Matplotlib style for the figures of this paper (written 2026-11-10).

The figure scripts beside this file use it as follows:

    import figstyle as fs
    fs.setup()
    fig, ax = fs.figure(height_in=1.9)            # one IEEE column wide (3.5 in); fs.figure("text", ...) for figure*
    ax.plot(x, y, color=fs.CAT[0], **fs.LINE)
    fs.save(fig, "figures/figX.pdf")              # vector PDF, fonts embedded, byte-reproducible

Colors follow the reference palette of the data-visualization method (categorical slots in a fixed order,
validated on a white surface: adjacent pairs pass; only the first three slots pass all pairs, so a scatter
or small multiple with more than three colored series needs a second encoding such as marker shape).
Aqua, yellow and magenta are below 3:1 contrast on white: use them only with direct labels or a legend
and never for thin lines alone. Text never wears a series color.

Sizes: IEEEtran conference columns are 252 pt (3.5 in) wide and the text block is 516 pt (7.17 in);
one body line is 12 pt (1/6 in), so a figure 1 in shorter frees about six lines.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLUMN_WIDTH_IN = 252 / 72.0
TEXT_WIDTH_IN = 516 / 72.0

# Categorical slots, fixed order: blue, orange, aqua, yellow, magenta, green, violet, red.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# Sequential (magnitude) ramp, light to dark; steps 100..700 of the blue hue.
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
# Ordinal ramp for ordered classes (dark = best/finest); starts no lighter than step 250 on white.
ORD = ["#0d366b", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef"]
DIVERGING_MID = "#f0efec"          # neutral gray midpoint; poles are SEQ (blue) and RED below
RED = ["#f6c9c8", "#ee9e9d", "#e34948", "#b83634", "#8c2826"]  # red arm for diverging use only
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

INK = "#0b0b0b"          # primary text
INK2 = "#52514e"         # secondary text, tick labels
MUTED = "#898781"        # muted labels, de-emphasized series
GRAY = "#b4b2aa"         # de-emphasis fill for context series
GRID = "#e1e0d9"         # hairline gridlines (solid, never dashed)
AXIS = "#c3c2b7"         # baselines and spines
SURFACE = "#ffffff"      # the page

LINE = {"linewidth": 1.1, "solid_capstyle": "round", "solid_joinstyle": "round"}
MARK = {"markersize": 4.2, "markeredgecolor": SURFACE, "markeredgewidth": 0.6}  # dots carry a surface ring
SCATTER = {"s": 16, "edgecolors": SURFACE, "linewidths": 0.5}
REFLINE = {"color": INK2, "linewidth": 0.7, "linestyle": (0, (3, 2))}  # thresholds such as delta (dashed is
# reserved for real thresholds; gridlines stay solid)
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]  # second encoding, same order as CAT


def setup(base_pt=7.0):
    """Apply the shared rcParams (sans serif, embedded TrueType fonts, hairline chrome)."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": base_pt,
        "axes.titlesize": base_pt + 0.5,
        "axes.titleweight": "regular",
        "axes.labelsize": base_pt,
        "axes.labelcolor": INK,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.prop_cycle": matplotlib.cycler(color=CAT),
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.labelsize": base_pt - 0.5,
        "ytick.labelsize": base_pt - 0.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "legend.fontsize": base_pt - 0.5,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.2,
        "legend.columnspacing": 1.0,
        "legend.handletextpad": 0.4,
        "text.color": INK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.hashsalt": "icde27",
        "hatch.linewidth": 0.5,
    })


def figure(width="column", height_in=2.0, **subplots_kw):
    """A figure at its final printed size, laid out to fit it exactly (include it at width=\\columnwidth or
    \\textwidth so LaTeX never scales the fonts)."""
    w = COLUMN_WIDTH_IN if width == "column" else TEXT_WIDTH_IN if width == "text" else float(width)
    subplots_kw.setdefault("layout", "constrained")
    return plt.subplots(figsize=(w, height_in), **subplots_kw)


def grid(ax, axis="y"):
    """Solid hairline gridlines behind the data, on one axis by default."""
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5, linestyle="-")
    ax.set_axisbelow(True)


def panel_label(ax, letter, x=-0.02, y=1.02):
    """'(a)' style panel label in primary ink, left-aligned above the axes."""
    ax.text(x, y, "(%s)" % letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=plt.rcParams["font.size"] + 0.5,
            color=INK)


def save(fig, path):
    """Vector PDF with fonts embedded and no timestamps, so reruns give identical bytes."""
    fig.savefig(path, format="pdf", metadata={"CreationDate": None, "ModDate": None, "Creator": None, "Producer": None})
    plt.close(fig)
