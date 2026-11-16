"""Supplementary figure: the fiber stage of the two-scale model. Given the exact layout of a training region, a fiber model returns that region; with noise
on the layout it draws new fibers, and its fibers are as good as those of the single-scale models when training stops
early, and degrade with longer training.

a: 256 px regions (0.27 mm) of the training band (top) and the held-out band (bottom); images generated from their
   layouts by the fiber model trained on exact layouts (8000 steps) and by the fiber model trained with noise on the
   layout (2000 steps, sampled at noise 0.5).
b: copying: correlation of the image generated from the layout of a training-band region (1024 px) with that region,
   after removing the layout scale (Gaussian of 12 px).
c: fiber models (noise on the layout) trained on the left (A) and right (B) halves of the band and on the left half with
   another seed, given the same layout and the same noise: correlation of their images after removing the layout scale,
   against training steps.
d: FID of 128 px windows (to the held-out band) against the correlation of the fiber-fraction layout along the bands at
   0.27 mm: training band, single-scale models, two-scale models (layout model after 1000 steps) with the fiber model
   after 2000 steps (filled; three seeds) and after 1000, 3000 and 4000 (three seeds), 6000, 8000 (three seeds) and 16000 steps
   (open, labelled); the line follows seed 1 through its training lengths.
Data: results/sic/towscale/given_layout*.json/npz, fiber_pairs*.json, *.json; results/sic/bandmodels/*.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.figstyle import BLUES, DOUBLE, INK, INK2, SERIES, image_axis, panel_label, plt  # noqa: E402

TS = ROOT / "results/sic/towscale"
BM = ROOT / "results/sic/bandmodels"
EXACT, NOISY = "fiber_state_s1", "fiber_stateaug2k_s1@0.5"
BARS = [("exact\nlayouts", "fiber_state_s1", SERIES[1]), ("noisy,\n2000 st.", "fiber_stateaug2k_s1@0.5", SERIES[2]),
        ("noisy,\n8000 st.", "fiber_stateaug_s1@0.5", SERIES[2])]
TR, HO = (slice(256, 512), slice(256, 512)), (slice(384, 640), slice(300, 556))
PAIRS = {2000: "fiber_pairs_stateaug2k.json", 4000: "fiber_pairs_stateaugshort.json", 8000: "fiber_pairs.json", 16000: "fiber_pairs_stateauglong.json"}
FIBER_STEPS = [("2000 steps", [f"twoscale_early_2k_s{k}" for k in (1, 2, 3)]),
               ("1000 steps", ["twoscale_early_1k_s1"]), ("3000 steps", [f"twoscale_early_3k_s{k}" for k in (1, 2, 3)]),
               ("4000 steps", [f"twoscale_earlyshort_s{k}" for k in (1, 2, 3)]), ("6000 steps", ["twoscale_early_6k_s1"]),
               ("8000 steps", [f"twoscale_early_s{k}" for k in (1, 2, 3)]), ("16000 steps", ["twoscale_early_long_s1"])]
PATH = ["twoscale_early_1k_s1", "twoscale_early_2k_s1", "twoscale_early_3k_s1", "twoscale_earlyshort_s1", "twoscale_early_6k_s1", "twoscale_early_s1", "twoscale_early_long_s1"]


def band(n):
    p = BM / f"{n}.json"
    return json.loads(p.read_text()) if p.exists() else None


def tow_along(n):
    p = TS / f"{n}.json"  # always the 24-field tow evaluation, so that every point shares its noise
    return json.loads(p.read_text())["tow"]["corr_along"][32] if p.exists() else None


def main():
    res = json.loads((TS / "given_layout.json").read_text())
    plane = np.load(ROOT / "data/sic/section_plane.npy").astype(np.float32) / 127.5 - 1
    train = plane[600:1624, 800:1824]
    held = plane[:1024, 2768:2768 + 824]
    ex, no = (np.load(TS / f"given_layout_{n}.npz") for n in (EXACT, NOISY))
    rows = [("training band", [train[TR], ex["on_train"].astype(np.float32)[TR], no["on_train"].astype(np.float32)[TR]]),
            ("held-out band", [held[HO], ex["image"].astype(np.float32)[HO], no["image"].astype(np.float32)[HO]])]
    heads = ["real", "fiber model,\nexact layouts", "fiber model, noisy\nlayouts, 2000 steps"]
    W, H = DOUBLE, 0.5 * DOUBLE
    fig = plt.figure(figsize=(W, H))
    side = 1.08
    for r, (lab, ims) in enumerate(rows):
        for c, im in enumerate(ims):
            a = fig.add_axes([(0.2 + c * (side + 0.05)) / W, 1 - (0.42 + (r + 1) * side + r * 0.08) / H, side / W, side / H])
            a.imshow(im, cmap="gray", vmin=-1, vmax=1, interpolation="lanczos")
            image_axis(a)
            if r == 0:
                a.set_title(heads[c], fontsize=6.3, pad=2)
            if c == 0:
                a.text(-0.06, 0.5, lab, rotation=90, transform=a.transAxes, ha="right", va="center", fontsize=6.5)
            if r == 0 and c == 0:
                panel_label(a, "a", x=-0.2, y=1.2)
    b = fig.add_axes([0.64, 0.62, 0.12, 0.28])
    for i, (lab, name, col) in enumerate(BARS):
        if name in res:
            b.bar(i, res[name]["copy_training_high"], color=col, width=0.6)
    b.set_xticks(range(len(BARS)))
    b.set_xticklabels([x[0] for x in BARS], fontsize=5.3)
    b.set_ylim(0, 1)
    b.set_ylabel("correlation with the\ntraining region", fontsize=6.5)
    b.set_title("copying", fontsize=7, pad=3)
    panel_label(b, "b", x=-0.75)
    c = fig.add_axes([0.86, 0.62, 0.12, 0.28])
    st, same, cross = [], [], []
    for steps, f in PAIRS.items():
        p = TS / f
        if p.exists():
            dd = json.loads(p.read_text())["agreement_high"]
            st.append(steps)
            same.append(dd["A | A2"])
            cross.append(np.mean([dd["A | B"], dd["A2 | B"]]))
    c.plot(st, same, "o-", color=SERIES[2], ms=3, label="A vs A,\nother seed")
    c.plot(st, cross, "s-", color=SERIES[1], ms=3, label="A vs B")
    c.set_xscale("log")
    c.set_xticks(st, [f"{s // 1000}k" for s in st])
    c.minorticks_off()
    c.set_ylim(0, 1)
    c.set_xlabel("training steps", fontsize=6.5)
    c.set_ylabel("same-noise agreement", fontsize=6.5)
    c.set_title("fiber models of\nthe two halves", fontsize=7, pad=3)
    c.legend(fontsize=5, loc="lower left")
    panel_label(c, "c", x=-0.75)
    d = fig.add_axes([0.66, 0.1, 0.33, 0.36])
    real = json.loads((TS / "real.json").read_text())
    d.scatter([real["training band"]["corr_along"][32]], [band("training_band")["scores"]["inception_128"]], marker="*",
              s=40, color=INK, label="training band", zorder=4)
    xs = [tow_along(n) for n in ("single64", "single128", "single256", "single512")]
    ys = [band(n)["scores"]["inception_128"] for n in ("single64", "single128", "single256", "single512")]
    d.scatter(xs, ys, color=BLUES[3], s=12, label="single scale", zorder=3)
    other = False
    for lab, names in FIBER_STEPS:
        pts = [(tow_along(n), band(n)["scores"]["inception_128"]) for n in names if band(n) and tow_along(n) is not None]
        if not pts:
            continue
        full = lab.startswith("2000")
        d.scatter(*zip(*pts), s=14, zorder=3, facecolors=SERIES[2] if full else "white", edgecolors=SERIES[2],
                  linewidths=0.9, label=("two-scale, fiber model 2000 steps" if full else
                                         None if other else "two-scale, other fiber training lengths"))
        other = other or not full
        if not full:  # label the training length next to the group
            left = False
            x, y = (min if left else max)(p[0] for p in pts), np.mean([p[1] for p in pts])
            below = lab.startswith(("1000", "3000"))
            d.annotate(f"{int(lab.split()[0]) // 1000}k", (x, y), xytext=(0, -5) if below else (-4 if left else 4, 0),
                       textcoords="offset points", fontsize=5.5, color=INK2, va="top" if below else "center",
                       ha="center" if below else "right" if left else "left")
    path = [(tow_along(n), band(n)["scores"]["inception_128"]) for n in PATH if band(n) and tow_along(n) is not None]
    d.plot(*zip(*path), color=SERIES[2], lw=0.6, alpha=0.6, zorder=2)  # seed 1 through its training lengths
    d.set_xlabel("correlation along the bands at 0.27 mm")
    d.set_ylabel("FID, 128 px windows")
    d.set_ylim(0, 165)
    d.legend(fontsize=5.2, loc="upper left")
    panel_label(d, "d", x=-0.27)
    fig.savefig(ROOT / "figures/fig11_fiber_given_layout.pdf", dpi=300)
    fig.savefig(ROOT / "figures/fig11_fiber_given_layout.png", dpi=220)


if __name__ == "__main__":
    main()
