"""Figure 1: experimental setup.

a: the polished section; training band (left) and held-out band (right).
b: zoom into the training band: a checkerboard of 128 px tiles; tiles of one colour form training set A,
   the other set B.
c: the training tiles of sets A and B at the five training areas (replicate 0): neighbouring A and B
   tiles are added in pairs, in random order, so the two sets always cover the same parts of the section.
   Marked: a 64 px window (the crop of the fiber-scale test) and a crop of 0.53 mm (the smallest crop that holds the
   tow bands).
d: a 64 px window, the pixels brighter than the fiber threshold, and the segmentation into fibers,
   matrix and pores (genmicro.phases): the fiber scale.
e: the tow scale: local fiber fraction of the section (genmicro.twoscale.state_fields: Gaussian of 12 um, 8.3 um per
   pixel) with a 0.53 mm crop, and the correlation of the fiber fraction of the training band along and across the
   bands.
"""
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genmicro.phases import FIBER_T, phases  # noqa: E402
from scripts.figstyle import DOUBLE, INK, INK2, SERIES, image_axis, plain_log_axis, plt  # noqa: E402
from genmicro.checkerboard import COLS, LEVELS, MM2_PER_TILE, ROWS, T, pair_order, pairs  # noqa: E402

UM = 1.04
W, H = DOUBLE, 0.86 * DOUBLE
PHASE_CMAP = ListedColormap(["#9a9a9a", "white", "black"])
K = 64


def tile_pairs(rep=0):
    """Pairs of (row, column) of the A and B tiles, in the order in which they are added."""
    p = pairs()
    return [tuple(divmod(t, COLS) for t in p[i]) for i in pair_order(rep)]


def label(fig, x, y, letter, text=""):
    fig.text(x, y, letter, fontsize=8, fontweight="bold", va="bottom", color=INK)
    if text:
        fig.text(x + 0.022, y, text, fontsize=7, va="bottom", color=INK)


def ax_at(fig, x, y_top, w_in, h_in):
    """Axes from inches: x and y_top measured from the top-left corner of the figure."""
    return fig.add_axes([x / W, 1 - (y_top + h_in) / H, w_in / W, h_in / H])


def correlation_along(phase_window, lags):
    m = phase_window == 1
    p = m.mean()
    s = np.array([(m[:, :-r] & m[:, r:]).mean() for r in lags])
    return (s - p * p) / max(p - p * p, 1e-6)


def main():
    plane = np.load(ROOT / "data/sic/section_plane.npy")
    fig = plt.figure(figsize=(W, H))
    # a: the section
    aw = 4.15
    ah = aw * plane.shape[0] / plane.shape[1]
    a = ax_at(fig, 0.05, 0.22, aw, ah)
    a.imshow(plane[::2, ::2], cmap="gray", vmin=0, vmax=255, interpolation="lanczos")
    s2 = lambda v: v / 2
    a.add_patch(Rectangle((0, 0), s2(COLS * T) - 1, s2(ROWS * T) - 1, fill=False, ec=SERIES[0], lw=1.4))
    a.add_patch(Rectangle((s2(2768), 0), s2(3597 - 2768) - 1, s2(plane.shape[0]) - 1, fill=False, ec=INK, lw=1.4))
    box = dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.9)
    a.text(12, 12, "training band", fontsize=6.5, va="top", bbox=box)
    a.text(s2(2768) + 12, 12, "held-out band", fontsize=6.5, va="top", bbox=box)
    zy, zx, zn = 8, 6, 4
    a.add_patch(Rectangle((s2(zx * T), s2(zy * T)), s2(zn * T), s2(zn * T), fill=False, ec="white", lw=1.4))
    a.text(s2((zx + zn) * T) + 8, s2(zy * T), "b", color="white", fontsize=7, fontweight="bold", va="top",
           bbox=dict(boxstyle="square,pad=0.15", fc="black", ec="none", alpha=0.6))
    L = s2(500 / UM)
    x0, y0 = s2(2768) + 20, s2(plane.shape[0]) - 16
    a.add_patch(Rectangle((x0 - 10, y0 - 50), L + 20, 60, fc="black", ec="none", alpha=0.6))
    a.plot([x0, x0 + L], [y0 - 6] * 2, color="white", lw=2)
    a.text(x0 + L / 2, y0 - 14, "0.5 mm", color="white", fontsize=6, ha="center", va="bottom")
    lab_box = dict(boxstyle="square,pad=0.15", fc="black", ec="none", alpha=0.6)
    a.add_patch(Rectangle((s2(300), s2(1700)), s2(512), s2(512), fill=False, ec=SERIES[2], lw=1.3))
    a.text(s2(300), s2(1700) - 8, "0.53 mm crop (tow scale)", color="white", fontsize=5.8, va="bottom", bbox=lab_box)
    a.add_patch(Rectangle((s2(1500), s2(560)), s2(64), s2(64), fill=False, ec="white", lw=1.1))
    a.text(s2(1564) + 8, s2(592), "64 px window (fiber scale)", color="white", fontsize=5.8, va="center", bbox=lab_box)
    image_axis(a)
    label(fig, 0.0, 1 - 0.2 / H, "a", "one polished section of a SiC/SiC composite (9.2 mm²)")
    # b: zoom with checkerboard
    bw = 2.35
    b = ax_at(fig, W - bw - 0.05, 0.22 + (ah - bw) / 2, bw, bw)
    b.imshow(plane[zy * T:(zy + zn) * T, zx * T:(zx + zn) * T], cmap="gray", vmin=0, vmax=255, interpolation="lanczos")
    for di in range(zn):
        for dj in range(zn):
            is_a = (zy + di + zx + dj) % 2 == 0
            col = SERIES[0] if is_a else SERIES[1]
            b.add_patch(Rectangle((dj * T + 2, di * T + 2), T - 4, T - 4, fill=False, ec=col, lw=1.4))
            b.text(dj * T + 8, di * T + 8, "A" if is_a else "B", fontsize=6.5, color="white", va="top", fontweight="bold",
                   bbox=dict(boxstyle="square,pad=0.1", fc=col, ec="none"))
    image_axis(b)
    label(fig, (W - bw - 0.05) / W, 1 - (0.2 + (ah - bw) / 2) / H, "b", "checkerboard of 0.13 mm tiles")
    # c: nested training areas
    order = tile_pairs()
    band = plane[:ROWS * T, :COLS * T]
    cy = 0.22 + ah + 0.42
    tw = (W - 0.1 - 4 * 0.12) / 5
    th = tw * band.shape[0] / band.shape[1]
    for lev, n in enumerate(LEVELS):
        c = ax_at(fig, 0.05 + lev * (tw + 0.12), cy, tw, th)
        c.imshow(band[::4, ::4], cmap="gray", vmin=0, vmax=255, interpolation="lanczos", alpha=0.55)
        for (ia, ja), (ib, jb) in order[:n]:
            c.add_patch(Rectangle((ja * T / 4, ia * T / 4), T / 4, T / 4, fc=SERIES[0], ec="none", alpha=0.9))
            c.add_patch(Rectangle((jb * T / 4, ib * T / 4), T / 4, T / 4, fc=SERIES[1], ec="none", alpha=0.9))
        image_axis(c)
        for sp in c.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(INK2)
            sp.set_linewidth(0.5)
        c.set_title(f"{n * MM2_PER_TILE:.2g} mm² per set", fontsize=6.5, pad=2)
    label(fig, 0.0, 1 - (cy - 0.2) / H, "c", "training tiles of set A (blue) and set B (orange) at five training areas")
    # d: segmentation of one window
    dy = cy + th + 0.5
    win = plane[960:960 + K, 1216:1216 + K].astype(np.float32) / 127.5 - 1
    steps = [(np.clip((win + 1) * 127.5, 0, 255), "window", "gray", 0, 255),
             ((win > FIBER_T).astype(float), "bright pixels", "gray", 0, 1),
             (phases(win), "fibers, matrix, pores", PHASE_CMAP, 0, 2)]
    dw = 1.05
    for k, (img, lab, cmap, lo, hi) in enumerate(steps):
        d = ax_at(fig, 0.05 + k * (dw + 0.1), dy, dw, dw)
        d.imshow(img, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        image_axis(d)
        for sp in d.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(INK2)
            sp.set_linewidth(0.5)
        d.set_title(lab, fontsize=6.5, pad=2)
    label(fig, 0.0, 1 - (dy - 0.2) / H, "d", "segmentation of a 64 px window")
    # e: the tow scale: fiber-fraction layout of the section and its correlation along and across the bands
    import json
    from genmicro.twoscale import state_fields
    lay = (state_fields(plane.astype(np.float32) / 127.5 - 1)[0, 0].cpu().numpy() + 1) / 2
    ex = 0.05 + 3 * (dw + 0.1) + 0.3
    ew = dw * lay.shape[1] / lay.shape[0]
    e = ax_at(fig, ex, dy, ew, dw)
    e.imshow(lay, cmap="Greys_r", vmin=0, vmax=0.65, interpolation="nearest")
    e.add_patch(Rectangle((40, 120), 64, 64, fill=False, ec=SERIES[2], lw=1.2))
    e.text(40, 114, "0.53 mm", color="white", fontsize=5.6, ha="left", va="bottom",
           bbox=dict(boxstyle="square,pad=0.12", fc="black", ec="none", alpha=0.6))
    image_axis(e)
    for sp in e.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(INK2)
        sp.set_linewidth(0.5)
    e.set_title("fiber fraction, 8.3 µm per pixel", fontsize=6.5, pad=2)
    real = json.loads((ROOT / "results/sic/towscale/real.json").read_text())
    px = 8 * UM
    gx = ex + ew + 0.5
    g = ax_at(fig, gx, dy + 0.02, W - gx - 0.08, dw - 0.3)
    lags = np.arange(65) * px
    g.plot(lags, real["training band"]["corr_along"], color=INK, lw=1.2, label="along")
    g.plot(lags, real["training band"]["corr_across"], color=INK2, lw=1.0, ls="--", label="across")
    g.axhline(0, color=INK2, lw=0.5)
    g.set_xlim(0, lags[-1])
    g.set_xlabel("separation (µm)")
    g.set_ylabel("correlation")
    g.legend(fontsize=5.6, loc="upper right", handlelength=1.6)
    label(fig, (ex - 0.25) / W, 1 - (dy - 0.2) / H, "e", "the tow scale: local fiber fraction of the section")
    fig.savefig(ROOT / "figures/fig1_setup.pdf", dpi=300)
    fig.savefig(ROOT / "figures/fig1_setup.png", dpi=220)


if __name__ == "__main__":
    main()
