"""Figure 5: generated microstructures of models trained on the whole training band, next to the held-out section.

Top row: 512 x 512 px fields (0.53 mm), where the arrangement of tows and pores is visible.
Bottom row: the 128 x 128 px centre of each field (0.13 mm), where single fibers are visible.
Data: results/sic/bandmodels/*_fields.npz (scripts/evaluate_band_models.py) and the section plane.
"""
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.figstyle import DOUBLE, INK, TWO_SCALE, image_axis, panel_label, plt  # noqa: E402

COLUMNS = [("held-out section", None), ("single scale,\n64 px crops", "single64"), ("single scale,\n128 px crops", "single128"),
           ("single scale,\n256 px crops", "single256"), ("single scale,\n512 px crops", "single512"),
           ("two-scale", TWO_SCALE[0])]
S, Z = 512, 128


def scale_bar(ax, size, um, pad):
    """White bar of `um` micrometres on a dark box in the lower right corner of an image of `size` px."""
    L = um / 1.04
    ax.add_patch(Rectangle((size - 3 * pad - L, size - 0.17 * size), L + 2 * pad, 0.17 * size - pad, fc="black", ec="none",
                           alpha=0.6))
    ax.plot([size - 2 * pad - L, size - 2 * pad], [size - 2.2 * pad] * 2, color="white", lw=2)
    ax.text(size - 2 * pad - L / 2, size - 3.2 * pad, f"{um} µm", color="white", fontsize=6, ha="center", va="bottom")


def to8(x):
    return np.clip((x.astype(np.float32) + 1) * 127.5, 0, 255)


def main():
    plane = np.load(ROOT / "data/sic/section_plane.npy")
    images = []
    for _, key in COLUMNS:
        if key is None:
            images.append(plane[1200:1200 + S, 2800:2800 + S].astype(np.float32))
        else:
            images.append(to8(np.load(ROOT / f"results/sic/bandmodels/{key}_fields.npz")["fields"][0][:S, :S]))
    n = len(COLUMNS)
    W = DOUBLE
    side = (W - 0.1 - (n - 1) * 0.06) / n
    H = 2 * side + 0.75
    fig = plt.figure(figsize=(W, H))
    for k, ((label, _), img) in enumerate(zip(COLUMNS, images)):
        x = 0.05 + k * (side + 0.06)
        a = fig.add_axes([x / W, 1 - (0.42 + side) / H, side / W, side / H])
        a.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="lanczos")
        c0 = (S - Z) // 2
        a.add_patch(Rectangle((c0, c0), Z, Z, fill=False, ec="#eb6834", lw=1.0))
        image_axis(a)
        a.set_title(label, fontsize=7, pad=3, color=INK)
        b = fig.add_axes([x / W, 1 - (0.42 + side + 0.06 + side) / H, side / W, side / H])
        b.imshow(img[c0:c0 + Z, c0:c0 + Z], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        image_axis(b)
        if k == 0:
            panel_label(a, "a", x=-0.08, y=1.15)
            panel_label(b, "b", x=-0.08, y=1.02)
            scale_bar(a, S, 250, 10)
            scale_bar(b, Z, 50, 3)
    fig.savefig(ROOT / "figures/fig5_examples.pdf", dpi=300)
    fig.savefig(ROOT / "figures/fig5_examples.png", dpi=220)


if __name__ == "__main__":
    main()
