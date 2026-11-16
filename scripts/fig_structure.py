"""Figure 6: two-point and chord-length statistics along and across the tow bands, real and generated.

a: a region of the held-out band and its segmentation (fibers white, matrix gray, pores black), with
   the two directions. b, c: two-point correlation of the fiber phase along (rows) and across (columns)
the tow bands. d, e: fraction of the length of the non-fiber phase (matrix and pores) in chords longer
than L, for L from 16 um (shorter chords coincide for all models). Real: training band and
held-out band; generated: 16 fields of 1024 x 1024 px per model.
Data: results/sic/bandmodels/*.json (scripts/evaluate_band_models.py).
"""
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genmicro.descriptors import CHORD_BINS  # noqa: E402
from genmicro.phases import phases  # noqa: E402
from scripts.figstyle import BLUES, DOUBLE, INK, INK2, SERIES, TWO_SCALE, image_axis, panel_label, plain_log_axis, plt  # noqa: E402

BM = ROOT / "results/sic/bandmodels"
UM = 1.04
LMIN = 16  # shortest chord length shown in d and e (µm)
MODELS = [("single scale, 64 px", ["single64"], BLUES[1]), ("single scale, 128 px", ["single128"], BLUES[2]),
          ("single scale, 256 px", ["single256"], BLUES[3]), ("single scale, 512 px", ["single512"], BLUES[4]),
          ("two-scale", TWO_SCALE, SERIES[2])]
PHASE_CMAP = ListedColormap(["#9a9a9a", "white", "black"])


def load(n):
    return json.loads((BM / f"{n}.json").read_text())["curves"]


def chord_centres():
    b = CHORD_BINS.numpy().astype(float)
    lo, hi = b[:-1], b[1:].copy()
    hi[-1] = 2 * lo[-1]
    return np.sqrt(lo * np.maximum(hi - 1, lo))


def corr(d, key):
    return np.array(d[key])[5:401]


def longer_than(d, key):
    w = np.array(d[key]) * chord_centres()
    w = w / w.sum()
    return np.cumsum(w[::-1])[::-1][1:]


def curves(ax, key, transform, x):
    ax.plot(x, transform(load("training_band"), key), color=INK, lw=1.4, label="real, training band")
    ax.plot(x, transform(load("held_out_band"), key), ":", color=INK, lw=1.4, label="real, held-out band")
    for lab, names, col in MODELS:
        names = [n for n in names if (BM / f"{n}.json").exists()]
        if not names:
            continue
        v = np.array([transform(load(n), key) for n in names])
        if len(names) > 1:
            ax.fill_between(x, v.min(0), v.max(0), color=col, alpha=0.25, lw=0)
        ax.plot(x, v.mean(0), color=col, lw=1.2, label=lab)


def main():
    plane = np.load(ROOT / "data/sic/section_plane.npy")
    region = plane[1200:1584, 2800:3184]
    fig = plt.figure(figsize=(DOUBLE, 0.56 * DOUBLE))
    side = 0.27
    a1 = fig.add_axes([0.0, 0.53, side * 0.56 / 0.56 * 0.56, side * 1.0])
    a2 = fig.add_axes([0.0, 0.08, side * 0.56, side * 1.0])
    a1.imshow(region, cmap="gray", vmin=0, vmax=255, interpolation="lanczos")
    a2.imshow(phases(region.astype(np.float32) / 127.5 - 1), cmap=PHASE_CMAP, vmin=0, vmax=2, interpolation="nearest")
    for ax in (a1, a2):
        image_axis(ax)
    for p, q, lab, off in (((30, 350), (200, 350), "along", (0, -22)), ((350, 340), (350, 170), "across", (-8, 0))):
        a1.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=1.6, color=SERIES[1]))
        a1.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], lab, color="white", fontsize=6.3,
                ha="right" if lab == "across" else "center", va="center",
                bbox=dict(boxstyle="square,pad=0.15", fc="black", ec="none", alpha=0.6))
    a1.set_title("held-out section, 0.4 mm", fontsize=6.8, pad=2)
    a2.set_title("fibers / matrix / pores", fontsize=6.8, pad=2)
    panel_label(a1, "a", x=-0.02, y=1.12)
    boxes = [[0.25, 0.6, 0.33, 0.3], [0.66, 0.6, 0.33, 0.3], [0.25, 0.1, 0.33, 0.3], [0.66, 0.1, 0.33, 0.3]]
    axs = [fig.add_axes(bx) for bx in boxes]
    lags = np.arange(5, 401) * UM
    for ax, key, title in ((axs[0], "corr_x", "two-point correlation of the fiber phase,\nalong the tow bands"),
                           (axs[1], "corr_y", "two-point correlation of the fiber phase,\nacross the tow bands")):
        curves(ax, key, corr, lags)
        ax.axhline(0, color=INK2, lw=0.5)
        ax.set_xscale("log")
        ax.set_xlim(5 * UM, 400 * UM)
        plain_log_axis(ax, "x", [10, 30, 100, 300])
        ax.set_ylim(-0.06, 0.32)
        ax.set_xlabel("separation (µm)")
        ax.set_title(title, fontsize=7, pad=3)
    edges = CHORD_BINS.numpy().astype(float)[1:-1] * UM
    keep = edges >= LMIN  # shorter chords are reproduced by every model; the curves differ in the tail
    for ax, key, title, ticks in ((axs[2], "chord_x", "chords of matrix and pores,\nalong the tow bands", [0.1, 0.2, 0.5]),
                                  (axs[3], "chord_y", "chords of matrix and pores,\nacross the tow bands", [0.03, 0.1, 0.3])):
        curves(ax, key, lambda d, k: longer_than(d, k)[keep], edges[keep])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(edges[keep][0], edges[keep][-1])
        ys = np.concatenate([ln.get_ydata() for ln in ax.get_lines()])
        ax.set_ylim(ys.min() / 1.15, ys.max() * 1.15)
        plain_log_axis(ax, "x", [20, 50, 100])
        plain_log_axis(ax, "y", ticks)
        ax.set_xlabel("chord length L (µm)")
        ax.set_title(title, fontsize=7, pad=3)
    axs[0].set_ylabel("correlation")
    axs[2].set_ylabel("fraction of length in\nchords longer than L")
    axs[1].legend(fontsize=5.5, loc="upper right")
    for ax, letter in zip(axs, "bcde"):
        panel_label(ax, letter, x=-0.2)
    fig.savefig(ROOT / "figures/fig6_structure.pdf", dpi=300)
    fig.savefig(ROOT / "figures/fig6_structure.png", dpi=220)


if __name__ == "__main__":
    main()
