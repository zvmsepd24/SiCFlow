"""Supplementary figure: does the area at which the paired models stop copying depend on the width of the network?

Paired flow models of replicate 0 with half (3.5 M parameters), the usual (13.8 M) and 1.5 times (29 M) the channels,
trained with the same curriculum and evaluated as in Fig. 2 and 3.
a: copying: fraction of generated windows more similar to the training tiles than any held-out window.
b: same-noise agreement of models A and B.
c: Levina-Bickel dimension of clouds of two-point curves; dotted: held-out windows.
d: FID to held-out windows (mean of A and B); e: FID between A and B; dashed and dotted: real tiles.
Data: results/sic/convergence/rep0.json, width_w05_rep0.json, width_w15_rep0.json (scripts/evaluate_checkerboard.py).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.figstyle import BLUES, DOUBLE, INK, INK2, panel_label, plain_log_axis, plt  # noqa: E402

CV = ROOT / "results/sic/convergence"
RUNS = [("half width, 3.5 M", "width_w05_rep0.json", BLUES[1]), ("U-Net of the paper, 13.8 M", "rep0.json", BLUES[3]),
        ("1.5 times the width, 29 M", "width_w15_rep0.json", BLUES[4])]


def above(c):
    return float(np.mean(np.array(c["generated_all"]) > max(c["held_out_all"])))


def area_axis(ax):
    ax.set_xscale("log")
    plain_log_axis(ax, "x", [0.02, 0.1, 0.3, 1, 3])
    ax.set_xlim(0.013, 4.5)
    ax.set_xlabel("area per model (mm$^2$)")


def main():
    runs = [(lab, json.loads((CV / f).read_text()), col) for lab, f, col in RUNS if (CV / f).exists()]
    fig, axs = plt.subplots(1, 5, figsize=(DOUBLE, 0.27 * DOUBLE))
    plt.subplots_adjust(wspace=0.55, left=0.06, right=0.99, bottom=0.36, top=0.92)
    for lab, rows, col in runs:
        x = [r["area_mm2"] for r in rows]
        kw = dict(color=col, marker="o", ms=3, lw=1.2, label=lab)
        axs[0].plot(x, [np.mean([above(r["copy"][s]) for s in "AB"]) for r in rows], **kw)
        axs[1].plot(x, [r["agreement"] for r in rows], **kw)
        axs[2].plot(x, [np.mean([r["dimension"]["generated"][s]["two_point"] for s in "AB"]) for r in rows], **kw)
        axs[3].plot(x, [np.mean([r["fidelity"][s]["fid"] for s in "AB"]) for r in rows], **kw)
        axs[4].plot(x, [r["convergence"]["fid"] for r in rows], **kw)
    ref = runs[0][1] if runs else []
    if ref:
        x = [r["area_mm2"] for r in ref]
        axs[2].axhline(np.mean([r["dimension"]["held_out"]["two_point"] for r in ref]), color=INK, ls=":", lw=1.0)
        axs[3].plot(x, [r["real_fidelity"]["fid"] for r in ref], ls="--", color=INK2, lw=1.0)
        axs[4].plot(x, [r["real_convergence"]["fid"] for r in ref], ls=":", color=INK2, lw=1.0)
    for ax, ylab, letter in zip(axs, ("copying fraction", "same-noise agreement", "intrinsic dimension",
                                      "FID to held-out", "FID, A vs B"), "abcde"):
        area_axis(ax)
        ax.set_ylabel(ylab)
        panel_label(ax, letter, x=-0.42)
    for ax in axs[3:]:
        ax.set_yscale("log")
        plain_log_axis(ax, "y", [10, 20, 50, 100, 200])
    axs[0].set_ylim(-0.03, 1.03)
    axs[1].set_ylim(-0.03, 1.03)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=6)
    fig.savefig(ROOT / "figures/figS_width.pdf", dpi=300)
    fig.savefig(ROOT / "figures/figS_width.png", dpi=220)


if __name__ == "__main__":
    main()
