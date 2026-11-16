"""Figures 3 and 4: when is there enough data? Paired models on the checkerboard split.

Figure 3 (descriptor convergence), against training area per model:
a: copying (median similarity of generated windows to the training tiles, beyond held-out windows) and
   agreement (correlation between the fields A and B generate from the same noise);
b-f: distance to the held-out band (fidelity, models A and B) and between models A and B (convergence),
   for FID, the two-point function of the fiber phase, the chord lengths of the non-fiber phase, the pore
   fraction and the fiber diameter; dashed and dotted gray: the same for real windows of the training
   tiles; gray band: two independent sets of held-out windows.
Figure 4 (Levina-Bickel dimension of clouds of windows) for flattened pixels, two-point vectors and
   chord vectors: generated (A and B), real windows of the training tiles, held-out windows.
Lines are means over three replicates (different random orders of the tile pairs); shading spans them.
Data: results/sic/convergence/rep*.json (scripts/evaluate_checkerboard.py).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.figstyle import DOUBLE, INK, INK2, SERIES, panel_label, plain_log_axis, plt  # noqa: E402

AREA_TICKS = [0.02, 0.1, 0.3, 1, 3]
TICKS = {"fid": [10, 20, 50, 100, 200], "two_point": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
         "chords": [0.001, 0.003, 0.01, 0.03, 0.1], "diameter": [0.03, 0.1, 0.3, 1, 3]}


def load():
    reps = [json.loads(p.read_text()) for p in sorted((ROOT / "results/sic/convergence").glob("rep*.json"))]
    n = min(len(r) for r in reps)
    return [r[:n] for r in reps]


def series(reps, f):
    v = np.array([[f(row) for row in rep] for rep in reps])
    return v.mean(0), v.min(0), v.max(0)


def line(ax, x, reps, f, color, ls="-", label=None, marker="o"):
    m, lo, hi = series(reps, f)
    if len(reps) > 1:
        ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0)
    ax.plot(x, m, ls, color=color, marker=marker, ms=2.8, lw=1.3, label=label)


def area_axis(ax):
    ax.set_xscale("log")
    plain_log_axis(ax, "x", AREA_TICKS)
    ax.set_xlabel("training area per model (mm$^2$)")


def convergence_figure(reps):
    x = np.array([row["area_mm2"] for row in reps[0]])
    fig, axs = plt.subplots(2, 3, figsize=(DOUBLE, 0.64 * DOUBLE))
    plt.subplots_adjust(wspace=0.42, hspace=0.62, left=0.08, right=0.99, bottom=0.17, top=0.94)
    axs = axs.ravel()
    ax = axs[0]
    line(ax, x, reps, lambda r: np.mean([r["copy"][s]["generated"] - r["copy"][s]["held_out"] for s in "AB"]), INK,
         label="copying")
    line(ax, x, reps, lambda r: r["agreement"], SERIES[1], label="agreement of A and B")
    ax.axhline(0, color=INK2, lw=0.5)
    ax.set_ylim(-0.05, 1.0)
    ax.set_ylabel("correlation")
    ax.set_title("copying and agreement", fontsize=7, pad=3)
    ax.legend(fontsize=5.6, loc="center left", bbox_to_anchor=(0.0, 0.42))
    area_axis(ax)
    panel_label(ax, "a", x=-0.3)
    panels = [("fid", "FID", "Fréchet Inception distance"), ("two_point", "Fréchet distance", "two-point function, fiber phase"),
              ("chords", "Fréchet distance", "chord lengths, matrix and pores"), ("pore", "Wasserstein-1 distance", "pore fraction"),
              ("diameter", "Wasserstein-1 distance (µm)", "fiber diameter")]
    for ax, (key, ylab, title), letter in zip(axs[1:], panels, "bcdef"):
        fid_f = lambda r, k=key: np.mean([r["fidelity"][s][k] for s in "AB"])
        conv_f = lambda r, k=key: r["convergence"][k]
        real_f = lambda r, k=key: r["real_fidelity"][k]
        real_c = lambda r, k=key: r["real_convergence"][k]
        floor = lambda r, k=key: r["floor"][k]
        line(ax, x, reps, fid_f, SERIES[0], label="generated vs held-out")
        line(ax, x, reps, conv_f, SERIES[1], label="model A vs model B")
        line(ax, x, reps, real_f, INK2, ls="--", marker=None, label="training tiles vs held-out")
        line(ax, x, reps, real_c, INK2, ls=":", marker=None, label="tiles A vs tiles B")
        fl = series(reps, floor)[0].mean()
        if key == "pore":  # no pores in the smallest training areas: distances of zero
            ax.set_ylim(bottom=0)
            ax.axhspan(0, fl, color="#e3e2de", lw=0, zorder=0, label="two held-out sets")
        else:
            vals = np.concatenate([np.ravel(series(reps, f)[1:]) for f in (fid_f, conv_f, real_f, real_c)] + [[fl]])
            lo, hi = vals.min() / 1.6, vals.max() * 1.6
            ax.set_yscale("log")
            ax.set_ylim(lo, hi)
            ax.axhspan(lo, fl, color="#e3e2de", lw=0, zorder=0, label="two held-out sets")
            plain_log_axis(ax, "y", [t for t in TICKS[key] if lo <= t <= hi])
        area_axis(ax)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=7, pad=3)
        panel_label(ax, letter, x=-0.3)
    h, lab = axs[1].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, fontsize=6, frameon=False)
    fig.savefig(ROOT / "figures/fig3_convergence.pdf")
    fig.savefig(ROOT / "figures/fig3_convergence.png", dpi=220)


def dimension_figure(reps):
    x = np.array([row["area_mm2"] for row in reps[0]])
    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 0.3 * DOUBLE))
    plt.subplots_adjust(wspace=0.4, left=0.08, right=0.99, bottom=0.22, top=0.86)
    for ax, (key, title), letter in zip(axs, (("pixels", "flattened images"), ("two_point", "two-point curves"),
                                              ("chords", "chord-length distributions")), "abc"):
        line(ax, x, reps, lambda r, k=key: np.mean([r["dimension"]["generated"][s][k] for s in "AB"]), SERIES[0],
             label="generated windows")
        line(ax, x, reps, lambda r, k=key: np.mean([r["dimension"]["training"][s][k] for s in "AB"]), INK2, ls="--",
             marker=None, label="windows of the training tiles")
        m, _, _ = series(reps, lambda r, k=key: r["dimension"]["held_out"][k])
        ax.axhline(m.mean(), color=INK, lw=1.0, ls=":", label="held-out windows")
        area_axis(ax)
        ax.set_ylabel("intrinsic dimension")
        ax.set_title(title, fontsize=7, pad=3)
        panel_label(ax, letter, x=-0.28)
    axs[0].legend(fontsize=5.6, loc="lower right")
    fig.savefig(ROOT / "figures/fig4_dimension.pdf")
    fig.savefig(ROOT / "figures/fig4_dimension.png", dpi=220)


def main():
    reps = load()
    convergence_figure(reps)
    dimension_figure(reps)


if __name__ == "__main__":
    main()
