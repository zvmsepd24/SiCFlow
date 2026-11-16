"""Supplementary figure: section 99 of the specimen as yardstick and as independent test set.

a-c: scores of 128 px windows against the held-out band of the training section (as in Fig. 4 and 5): FID, KID
     (x1000) and the Fréchet distance of DINOv2 features. Real material: the training band (same section, another
     place); section 99 at the place of the held-out band and over its in-focus right half (other fibers, 89
     sections further into the specimen);
     the held-out band blurred to the sharpness of section 99 (the cost of the blur alone). Models: the 64-pixel
     model of the whole band and the two-scale model (three seeds).
d-f: paired models (Fig. 2 and 3; mean of A and B and of three replicates) scored on 64 px windows against the
     held-out band (solid) and against section 99 (dashed): FID, two-point distance, and the copying fraction
     (generated windows more similar to the training tiles than every held-out window, or than every window of
     section 99). Horizontal lines: real windows, held-out band against section 99 (dash-dot) and against itself
     (dotted).
Data: results/sic/sections/references.json (scripts/evaluate_sections.py), results/sic/bandmodels (Fig. 4, 5),
results/sic/sections/independent_rep{0,1,2}.json (scripts/evaluate_independent.py).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.figstyle import DOUBLE, GRID, INK, INK2, SERIES, TWO_SCALE, panel_label, plain_log_axis, plt  # noqa: E402

R = ROOT / "results/sic/sections"
BM = ROOT / "results/sic/bandmodels"


def area_axis(ax):
    ax.set_xscale("log")
    plain_log_axis(ax, "x", [0.02, 0.1, 0.3, 1, 3])
    ax.set_xlim(0.013, 4.5)
    ax.set_xlabel("area per model (mm$^2$)")


def main():
    ref = json.loads((R / "references.json").read_text())
    blur = next(k for k in ref if k.startswith("held_blur"))
    rows = [("training band", ref["training_band"], INK2, "o"),
            ("section 99, same place", ref["section99_held"], SERIES[1], "D"),
            ("section 99, right half", ref["section99_right"], SERIES[1], "D"),
            ("held-out band, blurred", ref[blur], INK2, "s"),
            ("64-pixel model", json.loads((BM / "single64.json").read_text()), SERIES[2], "^")]
    rows += [(f"two-scale model, seed {i + 1}", json.loads((BM / f"{m}.json").read_text()), SERIES[2], "v")
             for i, m in enumerate(TWO_SCALE)]
    fig = plt.figure(figsize=(DOUBLE, 0.66 * DOUBLE))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1], hspace=0.5, wspace=0.12, left=0.19, right=0.985, top=0.97, bottom=0.09)
    y = np.arange(len(rows))[::-1]
    for j, (key, lab) in enumerate([("inception_128", "FID to held-out band"), ("kid_128", "KID ×1000 to held-out band"),
                                    ("dinov2_128", "DINOv2 distance to held-out band")]):
        ax = fig.add_subplot(gs[0, j])
        for yy, (name, r, col, mk) in zip(y, rows):
            ax.plot([r["scores"][key]], [yy], mk, color=col, ms=4.5)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows] if j == 0 else [])
        ax.set_ylim(-0.7, len(rows) - 0.3)
        top = max(r[1]["scores"][key] for r in rows)
        ax.set_xlim(-0.03 * top, 1.08 * top)
        ax.set_xlabel(lab)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        panel_label(ax, "abc"[j], x=-0.08 if j else -0.75)
    ind = [json.loads(p.read_text()) for p in sorted(R.glob("independent_rep[0-9].json"))]
    if ind:
        n = min(len(d["rows"]) for d in ind)
        x = [r["area_mm2"] for r in ind[0]["rows"][:n]]
        mean = lambda f: np.mean([[f(r) for r in d["rows"][:n]] for d in ind], 0)
        real = ind[0]["real"]  # the real windows are the same for every replicate
        gs2 = gs[1, :].subgridspec(1, 3, wspace=0.42)
        for j, (key, lab, ticks) in enumerate([("fid", "FID (64 px windows)", [10, 20, 50, 100]),
                                               ("two_point", "two-point distance", [0.001, 0.01, 0.1])]):
            ax = fig.add_subplot(gs2[0, j])
            ax.plot(x, mean(lambda r: np.mean([r["to_held"][s][key] for s in "AB"])), "o-", color=SERIES[0], ms=3, label="to held-out band")
            ax.plot(x, mean(lambda r: np.mean([r["to_s99"][s][key] for s in "AB"])), "o--", color=SERIES[1], ms=3, mfc="white",
                    label="to section 99")
            ax.axhline(real["held_vs_s99"][key], color=INK2, ls="-.", lw=1)
            ax.axhline(real["floor_held"][key], color=INK2, ls=":", lw=1)
            ax.text(0.015, real["held_vs_s99"][key] * 0.9, "section 99 vs held-out", va="top", fontsize=5.5, color=INK2)
            ax.text(0.015, real["floor_held"][key] * 1.1, "held-out vs held-out", va="bottom", fontsize=5.5, color=INK2)
            area_axis(ax)
            ax.set_yscale("log")
            plain_log_axis(ax, "y", ticks)
            ax.set_ylabel(lab)
            if j == 0:
                ax.legend(loc="upper right")
            panel_label(ax, "de"[j])
        ax = fig.add_subplot(gs2[0, 2])
        ax.plot(x, mean(lambda r: np.mean([r["copy"][s]["vs_held"] for s in "AB"])), "o-", color=SERIES[0], ms=3,
                label="held-out band")
        ax.plot(x, mean(lambda r: np.mean([r["copy"][s]["vs_s99"] for s in "AB"])), "o--", color=SERIES[1], ms=3, mfc="white",
                label="section 99")
        area_axis(ax)
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel("copying fraction")
        ax.legend(loc="lower left", title="reference", title_fontsize=6)
        panel_label(ax, "f")
    fig.savefig(ROOT / "figures/figS_independent.pdf")
    fig.savefig(ROOT / "figures/figS_independent.png", dpi=220)


if __name__ == "__main__":
    main()
