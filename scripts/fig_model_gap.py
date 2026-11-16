"""Figure 3: at the fiber scale, the difference that remains from held-out material belongs to the model, not the data.

a-c: distance to the held-out band (generated vs held-out, mean of models A and B) and between the paired models
     (A vs B) against training area, for FID, the two-point function of the fiber phase and the pore fraction; the
     same for real windows (training tiles vs held-out, tiles A vs tiles B); gray band: two sets of held-out
     windows; dash-dot: windows of another section of the specimen, about 0.1 mm deeper (section 99, its in-focus
     right half), against the held-out windows (scripts/evaluate_independent.py --real). Means over three
     replicates; bands span them.
d:   FID to held-out against training area: paired models, the paired models of the largest area trained from
     scratch for 8000 steps, the 64-pixel model trained for 8000 steps on the whole training band (twice the area),
     and real windows.
e:   FID of the 64-pixel model against the number of sampler steps, uniform and bunched near the noise.
f:   FID of the 64-pixel model, of the network with 1.5 times the channels and of the network trained 1.5 times
     longer (bars: means; points: single evaluations).
Data: results/sic/convergence/rep*.json (scripts/evaluate_checkerboard.py); results/sic/windows/*.json
      (scripts/evaluate_windows.py).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.figstyle import DOUBLE, INK, INK2, SERIES, panel_label, plain_log_axis, plt  # noqa: E402
from scripts.fig_convergence import TICKS, area_axis, line, load, series  # noqa: E402
from scripts.fig_data_model import area_panel, checks_panel, paired, steps_panel  # noqa: E402

FLOOR = "#e3e2de"


def other_section():
    """Distances between held-out windows and windows of section 99 (None if not computed)."""
    p = ROOT / "results/sic/sections/independent_real.json"
    return json.loads(p.read_text())["held_vs_s99"] if p.exists() else None


def convergence_panel(ax, x, reps, key, ylab, title, other=None):
    fid_f = lambda r: np.mean([r["fidelity"][s][key] for s in "AB"])
    conv_f = lambda r: r["convergence"][key]
    real_f = lambda r: r["real_fidelity"][key]
    real_c = lambda r: r["real_convergence"][key]
    line(ax, x, reps, fid_f, SERIES[0], label="generated vs held-out")
    line(ax, x, reps, conv_f, SERIES[1], label="model A vs model B")
    line(ax, x, reps, real_f, INK2, ls="--", marker=None, label="training tiles vs held-out")
    line(ax, x, reps, real_c, INK2, ls=":", marker=None, label="tiles A vs tiles B")
    fl = series(reps, lambda r: r["floor"][key])[0].mean()
    if other is not None:
        ax.axhline(other[key], color=INK2, ls="-.", lw=1, label="another section vs held-out")
    if key == "pore":  # no pores in the smallest training areas: distances of zero
        ax.set_ylim(bottom=0)
        ax.axhspan(0, fl, color=FLOOR, lw=0, zorder=0, label="two held-out sets")
    else:
        vals = np.concatenate([np.ravel(series(reps, f)[1:]) for f in (fid_f, conv_f, real_f, real_c)] + [[fl]]
                              + ([[other[key]]] if other is not None else []))
        lo, hi = vals.min() / 1.6, vals.max() * 1.6
        ax.set_yscale("log")
        ax.set_ylim(lo, hi)
        ax.axhspan(lo, fl, color=FLOOR, lw=0, zorder=0, label="two held-out sets")
        plain_log_axis(ax, "y", [t for t in TICKS[key] if lo <= t <= hi])
    area_axis(ax)
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=7, pad=3)


def main():
    reps = load()
    x = np.array([row["area_mm2"] for row in reps[0]])
    ref = json.loads((ROOT / "results/sic/windows/references.json").read_text())
    fig = plt.figure(figsize=(DOUBLE, 0.74 * DOUBLE))
    w, h = 0.24, 0.27
    xs = (0.08, 0.41, 0.745)
    top = [fig.add_axes([x_, 0.655, w, h]) for x_ in xs]
    bot = [fig.add_axes([x_, 0.165, w, h]) for x_ in xs]
    for ax, (key, ylab, title), letter in zip(top, (("fid", "FID", "Fréchet Inception distance"),
                                                    ("two_point", "Fréchet distance", "two-point function, fiber phase"),
                                                    ("pore", "Wasserstein-1 distance", "pore fraction")), "abc"):
        convergence_panel(ax, x, reps, key, ylab, title, other_section())
        panel_label(ax, letter, x=-0.32, y=1.04)
    h1, l1 = top[0].get_legend_handles_labels()
    fig.legend(h1, l1, loc="center", ncol=3, bbox_to_anchor=(0.5, 0.525), fontsize=6, handlelength=2.2)

    p = paired()
    area_panel(bot[0], p, ref, "fid", "FID to held-out", 3.5)
    plain_log_axis(bot[0], "y", [20, 50, 100])
    bot[0].set_title("more training data", fontsize=7, pad=3)
    steps_panel(bot[1], ref, 3.5, 1.3)
    bot[1].set_title("sampler", fontsize=7, pad=3)
    bot[1].legend(loc="upper right", fontsize=5.5)
    checks_panel(bot[2], ref, "fid", "FID to held-out", 4.5)
    bot[2].set_title("network and training", fontsize=7, pad=3)
    h2, l2 = bot[0].get_legend_handles_labels()
    fig.legend(h2, l2, loc="center", ncol=3, bbox_to_anchor=(0.5, 0.025), fontsize=6, handlelength=2.2)
    for ax, letter in zip(bot, "def"):
        panel_label(ax, letter, x=-0.32, y=1.04)
    fig.savefig(ROOT / "figures/fig3_model_gap.pdf", dpi=300)
    fig.savefig(ROOT / "figures/fig3_model_gap.png", dpi=220)


if __name__ == "__main__":
    main()
