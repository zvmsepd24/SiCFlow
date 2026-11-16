"""Figure: more training data against changes of the model, for the distance between generated and held-out windows.

Top row (more data): the paired models against training area per model (results/sic/convergence/rep{0,1,2}.json,
scripts/evaluate_checkerboard.py); the paired models of the largest area trained from scratch for 8000 steps
(A4scratch, B4scratch), and the 64-pixel model of the whole training band (6.8 mm^2, two training seeds, 8000 steps),
both evaluated on the same 64 x 64 px windows (results/sic/windows/eval_*.json, scripts/evaluate_windows.py).
Bottom row (changes of the model, all on the 64-pixel model): FID against the number of Heun steps with uniform steps
(the sampler of the paper, 32 steps) and with steps bunched near the noise; FID with 1.5 times the channels or 1.5
times the training steps. Every model of the
whole band is evaluated with three noise sets (the 64-pixel model: two training seeds, so six evaluations); points
show single evaluations, markers and bars their mean.
Writes figures/fig_data_model.pdf/.png and figures/wide/data_model.png.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.figstyle import DOUBLE, GRID, INK, INK2, SERIES, panel_label, plain_log_axis, plt  # noqa: E402

W = ROOT / "results/sic/windows"
BAND_MM2 = 2376 * 2640 * 1.04 ** 2 / 1e6  # 6.8 mm^2
SEEDS = ["single64", "single64_seed1"]  # the 64-pixel model, two training seeds
CHECKS = [("64-pixel\nmodel", SEEDS), ("1.5×\nchannels", ["single64_wide"]),
          ("1.5× training\nsteps", ["single64_steps12000"])]
STEPS = [8, 16, 32, 64, 128]
PAPER, BLUE, ORANGE = INK, SERIES[0], SERIES[1]


def ev(tag):
    f = W / f"eval_{tag}.json"
    return json.loads(f.read_text())["fidelity"] if f.exists() else None


def paired():
    reps = [json.loads((ROOT / f"results/sic/convergence/rep{r}.json").read_text()) for r in range(3)]
    out = dict(area=np.array([row["area_mm2"] for row in reps[0]]))
    for k in ("fid", "two_point", "chords"):
        out[k] = np.array([[(row["fidelity"]["A"][k] + row["fidelity"]["B"][k]) / 2 for row in rep] for rep in reps])
        out["tiles_" + k] = np.array([[row["real_fidelity"][k] for row in rep] for rep in reps]).mean(0)
    return out


def area_panel(ax, p, ref, k, ylabel, ms):
    x = p["area"]
    ax.axhspan(0, ref["floor"][k], color=GRID, lw=0)
    ax.plot(x, p["tiles_" + k], ls="--", color=INK2, lw=1.0, label="real windows of the training tiles")
    ax.plot([BAND_MM2], [ref["training_band"][k]], "s", color=INK2, ms=ms, mfc="white", label="real windows of the training band")
    ax.fill_between(x, p[k].min(0), p[k].max(0), color=BLUE, alpha=0.18, lw=0)
    ax.plot(x, p[k].mean(0), "o-", color=BLUE, ms=ms, label="paired models A and B")
    sv = check_values(["A4scratch", "B4scratch"], k)
    ax.errorbar([x[-1] * 1.25], [np.mean(sv)], yerr=[[np.mean(sv) - min(sv)], [max(sv) - np.mean(sv)]], fmt="D",
                color=BLUE, mfc="white", ms=ms, capsize=2, label="paired models, trained from scratch (8000 steps)")
    vals = check_values(SEEDS, k)
    ax.errorbar([BAND_MM2 * 1.22], [np.mean(vals)], yerr=[[np.mean(vals) - min(vals)], [max(vals) - np.mean(vals)]], fmt="D",
                color=ORANGE, ms=ms, capsize=2, label="64-pixel model, whole training band (8000 steps)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.012, 11)
    ax.set_xticks([0.02, 0.1, 1, 6.8])
    ax.set_xticklabels(["0.02", "0.1", "1", "6.8"])
    ax.minorticks_off()
    ax.set_xlabel("training area (mm$^2$)")
    ax.set_ylabel(ylabel)


def steps_panel(ax, ref, ms, lw):
    ax.axhspan(0, ref["floor"]["fid"], color=GRID, lw=0)
    ax.axhline(ref["training_band"]["fid"], color=INK2, ls="--", lw=1.0)
    uni = {s: ev("single64" if s == 32 else f"single64@{s}") for s in STEPS}
    near = {s: ev(f"single64@{s}r0.7") for s in STEPS}
    xu = [s for s in STEPS if uni[s]]
    ax.plot(xu, [uni[s]["fid"] for s in xu], "o-", color=ORANGE, ms=ms, lw=lw, label="uniform steps")
    xn = [s for s in STEPS if near[s]]
    ax.plot(xn, [near[s]["fid"] for s in xn], "^:", color=ORANGE, mfc="white", ms=ms + 1, lw=lw,
            label="steps bunched near the noise")
    ax.plot([32], [uni[32]["fid"]], "o", ms=ms * 2.1, mfc="none", mec=INK, mew=1.0, label="sampler of the paper")
    ax.set_xscale("log")
    ax.set_xticks(STEPS)
    ax.set_xticklabels([str(s) for s in STEPS])
    ax.minorticks_off()
    ax.set_ylim(0, 95)
    ax.set_xlabel("sampler steps")
    ax.set_ylabel("FID to held-out")


def check_values(names, k):
    v = []
    for n in names:
        for suffix in ("", "@32n1", "@32n2"):
            e = ev(n + suffix)
            if e:
                v.append(e[k])
    return v


def checks_panel(ax, ref, k, ylabel, ms, fontsize=None):
    ax.axhspan(0, ref["floor"][k], color=GRID, lw=0, zorder=1)
    ax.axhline(ref["training_band"][k], color=INK2, ls="--", lw=1.0, zorder=4)
    for i, (label, names) in enumerate(CHECKS):
        v = check_values(names, k)
        ax.bar(i, np.mean(v), width=0.62, color=ORANGE if i == 0 else "#f2ad8c", lw=0, zorder=2)
        ax.plot(i + np.linspace(-0.12, 0.12, len(v)), v, "o", color="white", mec=INK, ms=ms * 0.75, mew=0.7, zorder=3)
    ax.set_xticks(range(len(CHECKS)))
    ax.set_xticklabels([c[0] for c in CHECKS], fontsize=fontsize)
    ax.set_xlim(-0.6, len(CHECKS) - 0.4)
    ax.set_ylabel(ylabel)


def main():
    p = paired()
    ref = json.loads((W / "references.json").read_text())
    fig, axs = plt.subplots(2, 2, figsize=(DOUBLE * 0.82, 0.66 * DOUBLE))
    plt.subplots_adjust(wspace=0.38, hspace=0.62, left=0.1, right=0.99, bottom=0.08, top=0.83)
    area_panel(axs[0, 0], p, ref, "fid", "FID to held-out", 4)
    area_panel(axs[0, 1], p, ref, "two_point", "two-point distance to held-out", 4)
    plain_log_axis(axs[0, 0], "y", [20, 50, 100])
    plain_log_axis(axs[0, 1], "y", [0.001, 0.01, 0.1])
    steps_panel(axs[1, 0], ref, 4, 1.4)
    checks_panel(axs[1, 1], ref, "fid", "FID to held-out", 5)
    axs[1, 0].legend(loc="upper right", fontsize=6)
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.0), fontsize=6.5)
    fig.text(0.1, 0.875, "More training data", fontsize=7.5, fontweight="bold", color=INK)
    fig.text(0.1, 0.43, "Changes of the model (64-pixel model)", fontsize=7.5, fontweight="bold", color=INK)
    for ax, l in zip(axs.flat, "abcd"):
        panel_label(ax, l, x=-0.3, y=1.03)
    fig.savefig(ROOT / "figures/fig_data_model.pdf")
    fig.savefig(ROOT / "figures/fig_data_model.png", dpi=220)
    plt.close(fig)

    # wide version: FID against area, FID against sampler steps, two-point distance of the checks
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "legend.fontsize": 12, "lines.linewidth": 2})
    fig, axs = plt.subplots(1, 3, figsize=(12.3, 4.4))
    plt.subplots_adjust(wspace=0.36, left=0.06, right=0.99, bottom=0.15, top=0.8)
    area_panel(axs[0], p, ref, "fid", "FID to held-out", 7)
    plain_log_axis(axs[0], "y", [20, 50, 100])
    steps_panel(axs[1], ref, 7, 2.2)
    checks_panel(axs[2], ref, "fid", "FID to held-out", 8, fontsize=12)
    axs[1].legend(loc="upper right", fontsize=11, bbox_to_anchor=(1.0, 1.04))
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), fontsize=12)
    out = ROOT / "figures/wide/data_model.png"
    if out.parent.exists():
        fig.savefig(out, dpi=200)
    plt.close("all")


if __name__ == "__main__":
    main()
