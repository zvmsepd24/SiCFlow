"""Shared figure style: journal sizes, recessive axes, a validated categorical palette."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MM = 1 / 25.4
SINGLE, DOUBLE = 88 * MM, 180 * MM
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
# first three slots of the reference categorical palette (pass all-pairs CVD checks)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
BLUES = ["#c6dbf2", "#94bce8", "#5e9ad9", "#2a78d6", "#1b4f90"]  # light -> dark, one hue

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
    "legend.frameon": False,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})


def panel_label(ax, text, x=-0.18, y=1.04):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left", color=INK)


def image_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def plain_log_axis(ax, axis="y", ticks=None):
    """Log axis labelled with plain numbers (0.3, 1, 3) instead of powers of ten."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    a = ax.yaxis if axis == "y" else ax.xaxis
    if ticks is not None:
        a.set_major_locator(FixedLocator(ticks))
    a.set_minor_locator(NullLocator())
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

# Two-scale model of the paper (state layout model stopped after 1000 steps, fiber model trained with noise on the layout
# and stopped after 2000 steps; results/sic/bandmodels): one name per training seed.
TWO_SCALE = ["twoscale_early_2k_s1", "twoscale_early_2k_s2", "twoscale_early_2k_s3"]
