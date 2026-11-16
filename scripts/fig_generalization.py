"""Figure 2: at the fiber scale, memorization gives way to generalization near 1 mm^2, by three independent signals.

a: generated 64 px windows (top) above the most similar window of the model's training tiles (bottom), with
   their normalized cross-correlation (NCC), for models of set A at three training areas (replicate 0, same noise).
b: copying: fraction of generated windows more similar to the training tiles than any held-out window is
   (models A and B, three replicates).
c: agreement: correlation of the windows models A and B generate from the same noise; mean over 1024 samples per
   replicate, and the quartiles of 2048 samples of replicate 0; dashed: two seeds of the 64-pixel model, trained on
   identical data.
d: Levina-Bickel dimension of clouds of 512 two-point curves: generated windows (A and B), windows of the training
   tiles, held-out windows.
Shaded: the areas between which copying stops (0.28 to 1.1 mm^2). Lines: means over three replicates; bands span them.
Data: results/sic/convergence/rep*.json, rep0_samples.npz (scripts/evaluate_checkerboard.py);
      results/sic/windows/disagreement.json (scripts/disagreement.py).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.figstyle import DOUBLE, GRID, INK, INK2, SERIES, image_axis, panel_label, plain_log_axis, plt  # noqa: E402
from scripts.fig_convergence import load, series  # noqa: E402
from genmicro.checkerboard import LEVELS, MM2_PER_TILE  # noqa: E402

FIBERS_PER_MM2 = 16704 / 9.24  # segmented fibers in the section per mm^2
SHOW = [0, 2, 4]  # training areas shown in a
N_EX = 3
TRANSITION = (0.28, 1.13)
GRAY_FILL = "#cfcdc8"


def to8(x):
    return np.clip((x.astype(np.float32) + 1) * 127.5, 0, 255)


def area_axis(ax, fibers=False):
    ax.set_xscale("log")
    plain_log_axis(ax, "x", [0.02, 0.1, 0.3, 1, 3])
    ax.set_xlim(0.013, 4.5)
    ax.set_xlabel("training area per model (mm$^2$)")
    ax.axvspan(*TRANSITION, color=GRID, alpha=0.45, lw=0, zorder=0)
    if fibers:
        top = ax.secondary_xaxis("top", functions=(lambda a: a * FIBERS_PER_MM2, lambda f: f / FIBERS_PER_MM2))
        top.set_xscale("log")
        plain_log_axis(top, "x", [30, 100, 300, 1000, 3000])
        top.tick_params(labelsize=5.8, length=2)
        top.set_xlabel("fibers per model", fontsize=6, labelpad=2)


def heading(ax, letter, title, x):
    panel_label(ax, letter, x=x, y=1.2)
    ax.text(x + 0.09, 1.2, title, transform=ax.transAxes, fontsize=7, color=INK, va="bottom", ha="left")


def band(ax, x, reps, f, color, label, ls="-", marker="o"):
    m, lo, hi = series(reps, f)
    ax.fill_between(x, lo, hi, color=color, alpha=0.2, lw=0)
    ax.plot(x, m, ls, color=color, marker=marker, ms=3, lw=1.3, label=label)
    return m


def examples(fig, d, W, H, top=0.26, side=0.52, gap=0.025, block_gap=0.3, x0=0.52):
    for b, lev in enumerate(SHOW):
        area = LEVELS[lev] * MM2_PER_TILE
        n_fib = area * FIBERS_PER_MM2
        bx = x0 + b * (N_EX * (side + 0.06) + block_gap)
        fig.text((bx + (N_EX * (side + 0.06) - 0.06) / 2) / W, 1 - (top - 0.06) / H,
                 f"{area:.2g} mm$^2$ per model (≈{round(n_fib, -1 if n_fib < 1000 else -2):,.0f} fibers)",
                 fontsize=6.5, color=INK, ha="center", va="bottom")
        gen, near, sim = d[f"genA{lev}"], d[f"nearA{lev}"], d[f"simA{lev}"]
        for c, i in enumerate(np.arange(N_EX) * 3 + 1):
            x = bx + c * (side + 0.06)
            for k, img in enumerate((gen[i], near[i])):
                ax = fig.add_axes([x / W, 1 - (top + 0.1 + k * (side + gap) + side) / H, side / W, side / H])
                ax.imshow(to8(img), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                image_axis(ax)
                if k == 0:
                    ax.text(0.5, 1.02, f"NCC {sim[i]:.2f}", transform=ax.transAxes, fontsize=5.5, color=INK2, ha="center",
                            va="bottom")
                if b == 0 and c == 0:
                    ax.text(-0.08, 0.5, ("generated", "closest\ntraining\nwindow")[k], transform=ax.transAxes, fontsize=5.8,
                            color=INK2, ha="right", va="center")
                    if k == 0:
                        panel_label(ax, "a", x=-0.85, y=1.18)


def main():
    reps = load()
    x = np.array([row["area_mm2"] for row in reps[0]])
    d = np.load(ROOT / "results/sic/convergence/rep0_samples.npz")
    dis = json.loads((ROOT / "results/sic/windows/disagreement.json").read_text())
    W, H = DOUBLE, 0.56 * DOUBLE
    fig = plt.figure(figsize=(W, H))
    examples(fig, d, W, H)
    y0, h, w = 0.11, 0.34, 0.235
    axs = [fig.add_axes([x_, y0, w, h]) for x_ in (0.075, 0.415, 0.755)]

    ax = axs[0]
    copy = lambda r: np.mean([np.mean(np.array(r["copy"][s]["generated_all"]) > max(r["copy"][s]["held_out_all"]))
                              for s in "AB"])
    band(ax, x, reps, copy, INK, "generated windows")
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("fraction of generated windows closer\nto the training tiles than any held-out one")
    area_axis(ax, fibers=True)
    heading(ax, "b", "copying", -0.36)

    ax = axs[1]
    m = band(ax, x, reps, lambda r: r["agreement"], SERIES[1], "models A and B, mean")
    q = np.array([np.percentile(dis[f"A{l},B{l}"]["agreement"], [25, 75]) for l in range(len(x))])
    ax.vlines(x, q[:, 0], q[:, 1], color=SERIES[1], lw=0.8, alpha=0.6, label="A and B, quartiles of samples")
    seeds = np.mean(dis["single64_seed1,single64"]["agreement"])
    ax.axhline(seeds, color=INK2, ls="--", lw=1.0, label="two seeds, identical data")
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("correlation of the windows\nA and B generate from the same noise")
    area_axis(ax, fibers=True)
    ax.legend(fontsize=5.3, loc="upper left", bbox_to_anchor=(0.0, 0.86), handlelength=1.6)
    heading(ax, "c", "agreement from the same noise", -0.36)

    ax = axs[2]
    band(ax, x, reps, lambda r: np.mean([r["dimension"]["generated"][s]["two_point"] for s in "AB"]), SERIES[0],
         "generated windows")
    band(ax, x, reps, lambda r: np.mean([r["dimension"]["training"][s]["two_point"] for s in "AB"]), INK2,
         "windows of the training tiles", ls="--", marker=None)
    held = series(reps, lambda r: r["dimension"]["held_out"]["two_point"])[0].mean()
    ax.axhline(held, color=INK, lw=1.0, ls=":", label="held-out windows")
    ax.set_ylabel("intrinsic dimension\nof the two-point curves")
    area_axis(ax, fibers=True)
    ax.legend(fontsize=5.3, loc="lower right", handlelength=1.6)
    heading(ax, "d", "intrinsic dimension", -0.3)
    fig.savefig(ROOT / "figures/fig2_generalization.pdf", dpi=300)
    fig.savefig(ROOT / "figures/fig2_generalization.png", dpi=220)
    print("copying", np.round(series(reps, copy)[0], 3), "agreement", np.round(m, 3), "seeds", round(seeds, 3))


if __name__ == "__main__":
    main()
