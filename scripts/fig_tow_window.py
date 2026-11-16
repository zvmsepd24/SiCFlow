"""Figure 5: at the tow scale one section supports the band structure up to about 0.3 mm, within a window of training
time that the paired test finds; a two-scale model built from it has the fibers and these bands.

a: the fiber-fraction layout of the training band, split into a left half (model A) and a right half (model B).
b: layouts of 1.06 mm generated from the same noise by model A, by model A with another seed and by model B, after 500
   (top) and 6000 training steps (bottom).
c: same-noise agreement (correlation of the layouts, 24 noise samples) against training steps: A vs A with another
   seed, and A vs B; dashed and dotted: the fiber-scale values (two seeds of the 64-pixel model; paired models at
   3.2 mm^2).
d: copying: fraction of generated 0.53 mm windows more similar to the model's training region than any held-out
   window, for models of the halves (mean of three) and of the whole band.
e: Levina-Bickel dimension of clouds of 512 generated 0.53 mm layout windows, described by their autocovariance along
   and across (lags 0-0.27 mm), against training steps; lines: the same for windows cut from a half and from the
   whole band. Bars: range over models; five repeats each.
f: correlation of the fiber-fraction layout along the bands: training band; layout models of the whole band after
   1000 and 6000 steps; the two-scale model (layout model 1000 steps, fiber model 2000 steps; three seeds).
g: a field of the two-scale model (0.86 mm), its fiber fraction, and the fiber fraction of the held-out band.
h: FID of 128 px windows against the correlation along the bands at 0.27 mm: training band, single-scale models
   (64-512 px crops) and the two-scale model (three seeds).
Data: results/sic/towscale/*.json, agreement.json, dimension.json (scripts/evaluate_tow_scale.py,
      scripts/evaluate_tow_dimension.py); results/sic/bandmodels/*.json (scripts/evaluate_band_models.py).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.figstyle import BLUES, DOUBLE, INK, INK2, SERIES, TWO_SCALE, image_axis, panel_label, plt  # noqa: E402
from scripts.fig_tow_paired import FULL, NAMES, STEPS, above, frac, key, lname, load  # noqa: E402
from genmicro.twoscale import state_fields  # noqa: E402

TS = ROOT / "results/sic/towscale"
BM = ROOT / "results/sic/bandmodels"
GRAY = "#9a9a9a"
GREEN, ORANGE = SERIES[2], SERIES[1]
PX = 8 * 1.04
SINGLE = ["single64", "single128", "single256", "single512"]


def bm(n):
    p = BM / f"{n}.json"
    return json.loads(p.read_text()) if p.exists() else None


def steps_axis(ax, ticks=(500, 1000, 2000, 6000)):
    ax.set_xscale("log")
    ax.set_xticks(list(ticks), [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(400, 7500)
    ax.set_xlabel("training steps")


def main():
    agree = json.loads((TS / "agreement.json").read_text())
    real = json.loads((TS / "real.json").read_text())
    dim = json.loads((TS / "dimension.json").read_text())
    plane = np.load(ROOT / "data/sic/section_plane.npy").astype(np.float32) / 127.5 - 1
    band = frac(state_fields(plane[:, :2640])[0, 0].cpu().numpy())
    W, H = DOUBLE, 0.95 * DOUBLE
    fig = plt.figure(figsize=(W, H))
    fy = lambda y_in, h_in: 1 - (y_in + h_in) / H  # inches from the top -> figure fraction of the bottom edge

    # a: the two halves
    a = fig.add_axes([0.0, fy(0.32, 1.95), 1.8 / W, 1.95 / H])
    a.imshow(band, cmap="Greys_r", vmin=0, vmax=0.65)
    a.axvline(165, color=INK, lw=1.4)
    for x, t, col in ((82, "A", INK), (247, "B", INK)):
        a.text(x, -8, t, ha="center", va="bottom", fontsize=8, fontweight="bold", color=col)
    image_axis(a)
    a.set_title("training band, fiber fraction\n(2.5 × 2.7 mm)", fontsize=6.3, pad=12)
    panel_label(a, "a", x=-0.02, y=1.15)
    # b: layouts from the same noise
    side, x0 = 0.86, 2.05
    for r, st in enumerate((500, 6000)):
        for c, n in enumerate(NAMES[st]):
            lays = np.load(TS / f"layouts_{n}_fields.npz")["layouts"]
            ax = fig.add_axes([(x0 + c * (side + 0.05)) / W, fy(0.42 + r * (side + 0.06), side), side / W, side / H])
            ax.imshow(frac(lays[0, 0]), cmap="Greys_r", vmin=0, vmax=0.65)
            image_axis(ax)
            if r == 0:
                ax.set_title(["model A", "model A,\nother seed", "model B"][c], fontsize=6, pad=2)
            if c == 0:
                ax.text(-0.06, 0.5, f"{st} steps", rotation=90, transform=ax.transAxes, ha="right", va="center", fontsize=6)
            if r == 0 and c == 0:
                panel_label(ax, "b", x=-0.25, y=1.32)
    # c: agreement
    c = fig.add_axes([5.32 / W, fy(0.38, 1.62), 1.7 / W, 1.62 / H])
    same = [agree[f"{key(NAMES[s][0])} | {key(NAMES[s][1])}"] for s in STEPS]
    cross = [np.mean([agree[f"{key(NAMES[s][0])} | {key(NAMES[s][2])}"], agree[f"{key(NAMES[s][1])} | {key(NAMES[s][2])}"]])
             for s in STEPS]
    c.plot(STEPS, same, "o-", color=INK2, ms=3.5, label="A vs A, other seed")
    c.plot(STEPS, cross, "s-", color=ORANGE, ms=3.5, label="A vs B")
    c.axhline(0.93, color=GRAY, ls="--", lw=0.8, label="fiber scale, two seeds")
    c.axhline(0.78, color=GRAY, ls=":", lw=0.8, label="fiber scale, A vs B")
    steps_axis(c, STEPS)
    c.set_ylim(0, 1)
    c.set_ylabel("same-noise agreement")
    c.legend(fontsize=5.2, loc="lower left", handlelength=1.6)
    panel_label(c, "c", x=-0.36, y=1.03)

    row2, h2 = 2.55, 1.45
    # d: copying against training steps
    d = fig.add_axes([0.45 / W, fy(row2, h2), 1.75 / W, h2 / H])
    halves = [np.mean([above(n, real) for n in NAMES[s]]) for s in STEPS]
    fsteps = [s for s in FULL if all(load(lname(n)) for n in FULL[s])]
    full = [np.mean([above(n, real) for n in FULL[s]]) for s in fsteps]
    d.plot(STEPS, halves, "s-", color=ORANGE, ms=3.5, label="halves (12 crops)")
    d.plot(fsteps, full, "o-", color=INK, ms=3.5, label="whole band (24 crops)")
    steps_axis(d)
    d.set_ylim(0, 0.7)
    d.set_ylabel("generated windows closer to the\ntraining data than any held-out one")
    d.legend(fontsize=5.2, loc="upper left", handlelength=1.6)
    d.set_title("copying", fontsize=7, pad=3)
    panel_label(d, "d", x=-0.36, y=1.03)
    # e: intrinsic dimension against training steps
    e = fig.add_axes([2.85 / W, fy(row2, h2), 1.75 / W, h2 / H])
    g = dim["generated"]
    for names, steps, col, mk, lab in ((NAMES, STEPS, ORANGE, "s", "halves"), (FULL, fsteps, INK, "o", "whole band")):
        xs, ms, lo, hi = [], [], [], []
        for s in steps:
            v = [g[key(n)]["two_point"]["mean"] for n in names[s] if key(n) in g]
            if v:
                xs.append(s)
                ms.append(np.mean(v))
                lo.append(min(v))
                hi.append(max(v))
        e.plot(xs, ms, mk + "-", color=col, ms=3.5, label=f"generated, {lab}")
        e.vlines(xs, lo, hi, color=col, lw=0.9)
    rr = dim["real"]
    e.axhline(np.mean([rr["left half"]["two_point"]["mean"], rr["right half"]["two_point"]["mean"]]), color=ORANGE, ls="--",
              lw=0.9, label="windows of a half")
    e.axhline(rr["training band"]["two_point"]["mean"], color=INK, ls="--", lw=0.9, label="windows of the band")
    steps_axis(e)
    e.set_ylim(0, 8)
    e.set_ylabel("intrinsic dimension of\nthe layout windows")
    e.legend(fontsize=5.2, loc="lower left", handlelength=1.6)
    e.set_title("dimension", fontsize=7, pad=3)
    panel_label(e, "e", x=-0.3, y=1.03)
    # f: correlation along the bands
    f = fig.add_axes([5.32 / W, fy(row2, h2), 1.7 / W, h2 / H])
    lags = np.arange(65) * PX
    f.plot(lags, real["training band"]["corr_along"], color=INK, lw=1.3, label="real, training band")
    for st, col, lab in ((1000, GRAY, "layout models, 1000 steps"), (6000, ORANGE, "layout models, 6000 steps")):
        v = np.mean([load(lname(n))["tow"]["corr_along"] for n in FULL[st]], 0)
        f.plot(lags, v, color=col, lw=1.0, label=lab)
    two = np.array([load(n)["tow"]["corr_along"] for n in TWO_SCALE if load(n)])
    f.fill_between(lags, two.min(0), two.max(0), color=GREEN, alpha=0.25, lw=0)
    f.plot(lags, two.mean(0), color=GREEN, lw=1.3, label="two-scale model")
    f.axhline(0, color=INK2, lw=0.5)
    f.axvline(32 * PX, color=GRAY, lw=0.5, ls=":")
    f.set_xlim(0, lags[-1])
    f.set_ylim(-0.05, 1.0)
    f.set_xlabel("separation along the bands (µm)")
    f.set_ylabel("correlation of the fiber fraction")
    f.legend(fontsize=5.2, loc="upper right", handlelength=1.6)
    f.set_title("bands", fontsize=7, pad=3)
    panel_label(f, "f", x=-0.36, y=1.03)

    row3, h3 = 4.62, 1.55
    # g: a field of the two-scale model and its layout
    S = 824  # 0.86 mm, the width of the held-out band
    im = np.load(TS / f"{TWO_SCALE[0]}_fields.npz")["images"][0].astype(np.float32)[:S, :S]
    held = plane[400:400 + S, 2768:2768 + S]
    s3 = 1.2
    for k, (img, cmap, lim, lab) in enumerate(((im, "gray", (-1, 1), "two-scale model"),
                                               (frac(state_fields(im)[0, 0].cpu().numpy()), "Greys_r", (0, 0.65),
                                                "its fiber fraction"),
                                               (frac(state_fields(held)[0, 0].cpu().numpy()), "Greys_r", (0, 0.65),
                                                "held-out band"))):
        ax = fig.add_axes([(0.1 + k * (s3 + 0.06)) / W, fy(row3 + 0.12, s3), s3 / W, s3 / H])
        ax.imshow(img, cmap=cmap, vmin=lim[0], vmax=lim[1], interpolation="lanczos" if k == 0 else "nearest")
        ax.set_title(lab, fontsize=6.3, pad=2)
        image_axis(ax)
        if k == 0:
            panel_label(ax, "g", x=-0.05, y=1.1)
    # h: fibers against bands
    hx = fig.add_axes([4.55 / W, fy(row3, h3), 2.45 / W, h3 / H])
    tb = bm("training_band")["scores"]["inception_128"]
    hx.scatter([real["training band"]["corr_along"][32]], [tb], marker="*", s=45, color=INK, zorder=4, label="training band")
    xs = [load(n)["tow"]["corr_along"][32] for n in SINGLE]
    ys = [bm(n)["scores"]["inception_128"] for n in SINGLE]
    hx.scatter(xs, ys, color=BLUES[3], s=14, zorder=3, label="single scale, 64-512 px crops")
    pts = [(load(n)["tow"]["corr_along"][32], bm(n)["scores"]["inception_128"]) for n in TWO_SCALE]
    hx.scatter(*zip(*pts), color=GREEN, s=16, zorder=3, label="two-scale model, three seeds")
    hx.set_xlim(-0.01, 0.3)
    hx.set_ylim(0, 50)
    hx.set_xlabel("correlation along the bands at 0.27 mm")
    hx.set_ylabel("FID, 128 px windows")
    hx.legend(fontsize=5.4, loc="lower left", handlelength=1.4)
    hx.set_title("fibers and bands", fontsize=7, pad=3)
    panel_label(hx, "h", x=-0.2, y=1.03)
    fig.savefig(ROOT / "figures/fig5_tow_window.pdf", dpi=300)
    fig.savefig(ROOT / "figures/fig5_tow_window.png", dpi=220)
    print("two-scale FID", [round(p[1], 1) for p in pts], "bands", [round(p[0], 3) for p in pts], "training band FID", round(tb, 1))


if __name__ == "__main__":
    main()
