"""Figure 9: the paired test at the tow scale depends on training time. Layout models of the two halves of the band agree
from the same noise early in training, and copy their own half and disagree later.

a: the fiber-fraction layout of the training band, split into a left half (model A) and a right half (model B).
b: fiber-fraction layouts (1.06 mm square) generated from the same noise by model A, by model A trained with another
   seed, and by model B, after 500 (top) and 6000 training steps (bottom).
c: same-noise agreement (correlation of the fiber-fraction layouts, 24 noise samples) against training steps: A vs A
   with another seed, and A vs B (mean over the two seeds of A); dashed and dotted: the fiber-scale values (two seeds of
   the 64-pixel model, 0.93; paired models at 3.2 mm^2, 0.78).
d: copying against training steps: fraction of generated 0.53 mm windows more similar to the model's training region
   than any held-out window is; models of the halves (mean of A, A with another seed, and B) and of the whole band.
e: correlation of the fiber-fraction layout along the bands: training band, and models of the whole band after 500 and
   6000 steps.
Data: results/sic/towscale/*.json, agreement.json (scripts/evaluate_tow_scale.py).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.figstyle import DOUBLE, INK, INK2, SERIES, image_axis, panel_label, plt  # noqa: E402
from genmicro.twoscale import state_fields  # noqa: E402

TS = ROOT / "results/sic/towscale"
GRAY = "#9a9a9a"
STEPS = [500, 1000, 6000]
NAMES = {500: ("A500_s1", "A500_s2", "B500_s1"), 1000: ("A1000_s1", "A1000_s2", "B1000_s1"), 6000: ("A_s1", "A_s2", "B_s1")}
FULL = {500: ("steps500_s1", "steps500_s2"), 1000: ("steps1000_s1", "steps1000_s2"), 2000: ("steps2000_s1",),
        6000: ("layout_state_s1", "full_s2")}


def load(name):
    p = TS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def lname(n):
    return f"twoscale_{n}" if n.startswith("layout_state") else f"layouts_{n}"


def key(n):
    return f"twoscale/{n}" if n.startswith("layout_state") else f"layouts/{n}"


def frac(lay):
    return (lay.astype(np.float32) + 1) / 2


def above(name, real):
    d = load(lname(name))
    g = np.array(d["copy"]["64"])
    h = np.array(real["copy_held"][f"{d['cols'][0]}:{d['cols'][1]} 64"])
    return float(np.mean(g > h.max()))


def main():
    agree = json.loads((TS / "agreement.json").read_text())
    real = json.loads((TS / "real.json").read_text())
    plane = np.load(ROOT / "data/sic/section_plane.npy").astype(np.float32) / 127.5 - 1
    band = frac(state_fields(plane[:, :2640])[0, 0].cpu().numpy())
    W, H = DOUBLE, 0.62 * DOUBLE
    fig = plt.figure(figsize=(W, H))
    a = fig.add_axes([0.0, 0.5, 0.27, 0.42])
    a.imshow(band, cmap="Greys_r", vmin=0, vmax=0.65)
    a.axvline(165, color=INK, lw=1.5)
    for x, t, col in ((82, "A", SERIES[2]), (247, "B", SERIES[1])):
        a.text(x, -8, t, ha="center", va="bottom", fontsize=8, fontweight="bold", color=col)
    image_axis(a)
    a.set_title("training band, fiber fraction (2.5 × 2.7 mm)", fontsize=6.8, pad=14)
    panel_label(a, "a", x=-0.08, y=1.12)
    heads = ["model A", "model A,\nother seed", "model B"]
    side = 0.11
    hh = side * W / H
    for r, st in enumerate((500, 6000)):
        for c, n in enumerate(NAMES[st]):
            lays = np.load(TS / f"layouts_{n}_fields.npz")["layouts"]
            ax = fig.add_axes([0.33 + c * (side + 0.01), 0.5 + (1 - r) * (hh + 0.015), side, hh])
            ax.imshow(frac(lays[0, 0]), cmap="Greys_r", vmin=0, vmax=0.65)
            image_axis(ax)
            if r == 0:
                ax.set_title(heads[c], fontsize=6.3, pad=2)
            if c == 0:
                ax.text(-0.08, 0.5, f"{st} steps", rotation=90, transform=ax.transAxes, ha="right", va="center", fontsize=6.3)
            if r == 0 and c == 0:
                panel_label(ax, "b", x=-0.3, y=1.3)
    c = fig.add_axes([0.75, 0.55, 0.23, 0.37])
    same = [agree[f"{key(NAMES[s][0])} | {key(NAMES[s][1])}"] for s in STEPS]
    cross = [np.mean([agree[f"{key(NAMES[s][0])} | {key(NAMES[s][2])}"], agree[f"{key(NAMES[s][1])} | {key(NAMES[s][2])}"]])
             for s in STEPS]
    c.plot(STEPS, same, "o-", color=SERIES[2], label="A vs A, other seed")
    c.plot(STEPS, cross, "s-", color=SERIES[1], label="A vs B")
    c.axhline(0.93, color=GRAY, ls="--", lw=0.8, label="fiber scale, two seeds")
    c.axhline(0.78, color=GRAY, ls=":", lw=0.8, label="fiber scale, A vs B")
    c.set_xscale("log")
    c.set_xticks(STEPS, [str(s) for s in STEPS])
    c.minorticks_off()
    c.set_ylim(0, 1)
    c.set_xlabel("training steps")
    c.set_ylabel("same-noise agreement")
    c.legend(fontsize=5.5, loc="lower left")
    panel_label(c, "c", x=-0.42)
    d = fig.add_axes([0.1, 0.08, 0.36, 0.3])
    halves = [np.mean([above(n, real) for n in NAMES[s]]) for s in STEPS]
    fsteps = [s for s in FULL if all(load(lname(n)) for n in FULL[s])]
    full = [np.mean([above(n, real) for n in FULL[s]]) for s in fsteps]
    d.plot(STEPS, halves, "s-", color=SERIES[1], label="halves (12 crops)")
    d.plot(fsteps, full, "o-", color=INK, label="whole band (24 crops)")
    d.set_xscale("log")
    d.set_xticks([500, 1000, 2000, 6000], ["500", "1000", "2000", "6000"])
    d.minorticks_off()
    d.set_ylim(0, 0.7)
    d.set_xlabel("training steps")
    d.set_ylabel("generated windows closer to the\ntraining data than any held-out window")
    d.legend(fontsize=5.5, loc="upper left")
    panel_label(d, "d", x=-0.22)
    e = fig.add_axes([0.6, 0.08, 0.36, 0.3])
    lags = np.arange(65) * 8 * 1.04
    e.plot(lags, real["training band"]["corr_along"], color=INK, lw=1.2, label="real, training band")
    for st, col in ((500, SERIES[2]), (6000, SERIES[1])):
        v = np.mean([load(lname(n))["tow"]["corr_along"] for n in FULL[st]], 0)
        e.plot(lags, v, color=col, lw=1.1, label=f"whole band, {st} steps")
    e.axhline(0, color=INK2, lw=0.5)
    e.set_xlim(0, lags[-1])
    e.set_xlabel("separation along the bands (µm)")
    e.set_ylabel("correlation of the\nfiber-fraction layout")
    e.legend(fontsize=5.5, loc="upper right")
    panel_label(e, "e", x=-0.22)
    fig.savefig(ROOT / "figures/fig10_tow_paired.pdf", dpi=300)
    fig.savefig(ROOT / "figures/fig10_tow_paired.png", dpi=220)


if __name__ == "__main__":
    main()
