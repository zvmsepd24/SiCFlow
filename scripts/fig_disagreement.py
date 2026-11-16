"""Figure: per-sample agreement of the paired models from the same noise, and the samples on which they disagree.

Reads results/sic/windows/disagreement.json and disagreement_windows.npz (scripts/disagreement.py).
Writes figures/fig_disagreement.pdf/.png and, if figures/wide/ exists, figures/wide/disagreement_samples.png.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.figstyle import BLUES, DOUBLE, GRID, INK, INK2, image_axis, panel_label, plt  # noqa: E402

V = ROOT / "results/sic/windows"
AREAS = [0.018, 0.071, 0.28, 1.1, 3.2]
SAME = "single64_seed1,single64"  # the 64-pixel model of the whole training band, two seeds
N_WORST, N_MID = 6, 4


def load():
    res = json.loads((V / "disagreement.json").read_text())
    win = dict(np.load(V / "disagreement_windows.npz"))
    return res, win


def pick(res, win, pair):
    """Worst and median pairs of windows of one model pair, with their correlations."""
    r = np.array(res[pair]["agreement"])
    keep = np.array(res[pair]["kept"])
    a, b = win[f"{pair}|1"].astype(np.float32), win[f"{pair}|2"].astype(np.float32)
    worst = list(range(N_WORST))
    k = len(keep) // 3
    mid = list(range(k + k // 2 - N_MID // 2, k + k // 2 + N_MID // 2))
    return [(a[i], b[i], r[keep[i]]) for i in worst], [(a[i], b[i], r[keep[i]]) for i in mid]


def ridge(ax, res, fontsize=6.5, same=True):
    bins = np.linspace(-0.3, 1.0, 53)
    rows = [(f"A{l},B{l}", f"{AREAS[l]:g} mm$^2$", BLUES[l]) for l in range(5)]
    if same and SAME in res:
        rows.append((SAME, "64-px model, 2 seeds", "#8a8986"))
    for j, (p, label, c) in enumerate(rows):
        r = np.array(res[p]["agreement"])
        h, _ = np.histogram(r, bins, density=True)
        y0 = -j * 1.0
        h = h / h.max() * 0.9
        ax.fill_between(bins[:-1], y0, y0 + h, step="post", color=c, lw=0)
        ax.plot([r.mean()] * 2, [y0, y0 + 0.9], color=INK, lw=0.8)
        ax.text(-0.33, y0 + 0.2, label, ha="right", va="bottom", fontsize=fontsize, color=INK)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.3, 1.0)
    ax.set_ylim(-len(rows) + 0.9, 1.0)
    ax.set_xlabel("correlation of the windows of A and B")


def show(ax, img, cmap="gray", vmin=-1, vmax=1):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    image_axis(ax)


def pore_panel(ax, res, pair="A4,B4", ms=3):
    """Per-sample agreement against the difference of the pore fractions of the two windows."""
    d = res[pair]
    r = np.array(d["agreement"])
    dp = np.abs(np.array(d["pore1"]) - np.array(d["pore2"]))
    ax.plot(r, dp, "o", ms=ms, mew=0, color=BLUES[3], alpha=0.35)
    ax.set_xlim(-0.05, 1.0)
    ax.set_xlabel("correlation of the windows of A and B")
    ax.set_ylabel("pore-fraction difference")
    low = r < 0.5
    return dict(share_below_half=float(low.mean()), pore_diff_low=float(np.median(dp[low])),
                pore_diff_rest=float(np.median(dp[~low])), share_low_with_pore_diff_over_0p2=float((dp[low] > 0.2).mean()))


def pairs_grid(fig, gs, rows, worst, mid, fontsize, label_r_below=False, row_labels=("A", "B")):
    for j, (a, b, r) in enumerate(worst + mid):
        col = j if j < len(worst) else j + 1
        for i, img in enumerate((a, b)):
            axi = fig.add_subplot(gs[rows[i], col])
            show(axi, img)
            if i == 0 and not label_r_below:
                axi.set_title(f"{r:.2f}", fontsize=fontsize, color=INK, pad=2)
            if i == 1 and label_r_below:
                axi.text(0.5, -0.05, f"r = {r:.2f}", transform=axi.transAxes, ha="center", va="top", fontsize=fontsize,
                         color=INK)
            if col == 0:
                axi.text(-0.1, 0.5, row_labels[i], transform=axi.transAxes, ha="right", va="center", fontsize=fontsize,
                         color=INK, rotation=0 if len(row_labels[i]) < 3 else 90)


def paper_figure(res, win):
    worst, mid = pick(res, win, "A4,B4")
    fig = plt.figure(figsize=(DOUBLE, 0.42 * DOUBLE))
    top = fig.add_gridspec(1, 2, left=0.085, right=0.99, top=0.96, bottom=0.66, wspace=0.38, width_ratios=[1.15, 1])
    ax = fig.add_subplot(top[0])
    ridge(ax, res, same=SAME in res)
    panel_label(ax, "a", x=-0.2, y=1.06)
    stats = None
    if "pore1" in res["A4,B4"]:
        axc = fig.add_subplot(top[1])
        stats = pore_panel(axc, res)
        panel_label(axc, "c", x=-0.2, y=1.06)
    cols = N_WORST + N_MID + 1
    gs = fig.add_gridspec(2, cols, left=0.085, right=0.99, top=0.42, bottom=0.01, wspace=0.05, hspace=0.05,
                          width_ratios=[1] * N_WORST + [0.3] + [1] * N_MID)
    pairs_grid(fig, gs, (0, 1), worst, mid, 6.5)
    fig.text(0.03, 0.475, "b", fontsize=8, fontweight="bold", color=INK)
    fig.text(0.085, 0.475, "lowest correlation at 3.2 mm$^2$", fontsize=7, color=INK)
    fig.text(0.085 + 0.905 * (N_WORST + 0.3) / (cols - 0.7), 0.475, "median", fontsize=7, color=INK)
    fig.savefig(ROOT / "figures/fig_disagreement.pdf")
    fig.savefig(ROOT / "figures/fig_disagreement.png", dpi=220)
    plt.close(fig)
    return stats


def slide(res, win):
    """Wide version: at 3.2 mm^2, the six most-disagreeing noise samples next to four typical ones."""
    worst, mid = pick(res, win, "A4,B4")
    plt.rcParams.update({"font.size": 14})
    cols = N_WORST + N_MID + 1
    fig = plt.figure(figsize=(12.3, 3.55))
    gs = fig.add_gridspec(2, cols, left=0.05, right=0.995, top=0.87, bottom=0.1, wspace=0.05, hspace=0.05,
                          width_ratios=[1] * N_WORST + [0.35] + [1] * N_MID)
    pairs_grid(fig, gs, (0, 1), worst, mid, 13, label_r_below=True, row_labels=("model A", "model B"))
    fig.text(0.05, 0.92, "Most-disagreeing noise samples", fontsize=15, color=INK, fontweight="bold")
    fig.text(0.05 + 0.945 * (N_WORST + 0.35) / (cols - 0.65), 0.92, "Typical samples", fontsize=15, color=INK,
             fontweight="bold")
    out = ROOT / "figures/wide/disagreement_samples.png"
    if out.parent.exists():
        fig.savefig(out, dpi=200)
    plt.close(fig)
    plt.rcParams.update({"font.size": 7})


if __name__ == "__main__":
    res, win = load()
    stats = paper_figure(res, win)
    slide(res, win)
    if stats:
        print(json.dumps(stats))
