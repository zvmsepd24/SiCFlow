"""Figure 2: from memorization to generalization as the training area grows (checkerboard set A, replicate 0).

One row per training area. Four generated 64 px crops, each above the most similar 64 px window of the
model's training tiles, and the distribution of that similarity for 256 generated crops (blue) and for 256
held-out crops compared with the same tiles (gray). All rows use the same noise.
Data: results/sic/convergence/rep0_samples.npz (scripts/evaluate_checkerboard.py).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.figstyle import DOUBLE, INK, INK2, SERIES, image_axis, plt  # noqa: E402
from genmicro.checkerboard import LEVELS, MM2_PER_TILE  # noqa: E402

PAIRS = 4
FIBERS_PER_MM2 = 16704 / 9.24  # segmented fibers in the section per mm^2


def to8(x):
    return np.clip((x.astype(np.float32) + 1) * 127.5, 0, 255)


def main():
    d = np.load(ROOT / "results/sic/convergence/rep0_samples.npz")
    levels = [k for k in range(len(LEVELS)) if f"genA{k}" in d.files]
    side, gap, pair_gap, lab_w, hist_w = 0.56, 0.03, 0.12, 1.0, 1.35
    W = DOUBLE
    row_h = 2 * side + gap + 0.2
    H = len(levels) * row_h + 0.45
    fig = plt.figure(figsize=(W, H))
    idx = np.arange(PAIRS) * 3 + 1
    for r, lev in enumerate(levels):
        y_top = 0.35 + r * row_h
        area = LEVELS[lev] * MM2_PER_TILE
        n_fib = area * FIBERS_PER_MM2
        fig.text(0.02 / W, 1 - (y_top + side + gap / 2) / H,
                 f"{area:.2g} mm²\n≈{round(n_fib, -1 if n_fib < 1000 else -2):,.0f} fibers", fontsize=7, color=INK, va="center")
        gen, near, sim = d[f"genA{lev}"], d[f"nearA{lev}"], d[f"simA{lev}"]
        for c, i in enumerate(idx):
            x = lab_w + c * (side + pair_gap)
            for k, img in enumerate((gen[i], near[i])):
                ax = fig.add_axes([x / W, 1 - (y_top + k * (side + gap) + side) / H, side / W, side / H])
                ax.imshow(to8(img), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                image_axis(ax)
                if k == 0:
                    ax.text(0.5, 1.03, f"{sim[i]:.2f}", transform=ax.transAxes, fontsize=5.8, color=INK2, ha="center", va="bottom")
        if r == 0:
            for k, lab in enumerate(("generated", "closest\ntraining window")):
                fig.text((lab_w - 0.06) / W, 1 - (y_top + k * (side + gap) + side / 2) / H, lab, fontsize=6, color=INK2,
                         ha="right", va="center", rotation=90)
        hx = lab_w + PAIRS * (side + pair_gap) + 0.2
        ax = fig.add_axes([hx / W, 1 - (y_top + 2 * side + gap) / H, hist_w / W, (2 * side + gap) / H])
        bins = np.linspace(0, 1, 41)
        ax.hist(d[f"simheld{lev}"], bins=bins, color="#cfcdc8", lw=0, density=True, label="held-out windows")
        ax.hist(d[f"simA{lev}"], bins=bins, color=SERIES[0], lw=0, alpha=0.85, density=True, label="generated windows")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlim(0, 1.02)
        ax.set_xticks([0, 0.5, 1])
        if r < len(levels) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("similarity to closest\ntraining window")
        if r == 0:
            ax.legend(fontsize=5.5, loc="lower left", frameon=False, bbox_to_anchor=(0, 1.0))
    fig.savefig(ROOT / "figures/fig2_memorization.pdf", dpi=300)
    fig.savefig(ROOT / "figures/fig2_memorization.png", dpi=220)


if __name__ == "__main__":
    main()
