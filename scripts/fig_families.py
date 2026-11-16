"""Supplementary figure: the paired test for other families of generative models, trained on the same checkerboard.

Families (genmicro.families; scripts/train_family.py): flow matching (the models of the paper), denoising diffusion
(DDPM, the same U-Net, sampled with deterministic DDIM), a variational autoencoder (VAE) and a normalizing flow (Glow).
All trained with the curriculum of the paired models (lines: means over the replicates evaluated, bands: their range;
flow matching three replicates), and a second model of set A with another seed (A2) for the other families.
a: generated windows of model A at the largest area (3.2 mm^2).
b: copying: fraction of generated windows more similar to the training tiles than any held-out window.
c: same-noise agreement: A vs B (solid) and A vs A2 (dashed, other families); flow matching: the two seeds of the
   64-pixel model (cross).
d: FID to held-out windows (mean of A and B); dashed gray: real windows of the training tiles.
e: FID between models A and B; dotted gray: tiles A vs tiles B.
f: two-point distance between models A and B; dotted gray: tiles A vs tiles B.
g: likelihood: bits per dimension of held-out windows minus that of windows of the own training tiles (medians over
   windows where stored; positive: the model prefers its training windows), flow matching (as a continuous normalizing
   flow) and Glow.
Data: results/sic/families/*_rep0.json (scripts/evaluate_family.py); results/sic/convergence/rep*.json,
      likelihood_rep*.json (scripts/evaluate_checkerboard.py, scripts/evaluate_likelihood.py).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.figstyle import DOUBLE, INK, INK2, image_axis, panel_label, plain_log_axis, plt  # noqa: E402
from scripts.fig_convergence import load as load_fm  # noqa: E402

FAM = ROOT / "results/sic/families"
CV = ROOT / "results/sic/convergence"
# categorical slots 1-4 of the reference palette, in fixed order (validated; markers as second encoding)
STYLE = {"flow matching": ("#2a78d6", "o"), "DDPM": ("#eb6834", "s"), "VAE": ("#1baf7a", "^"), "Glow": ("#eda100", "D")}
KEYS = {"DDPM": "ddpm", "VAE": "vae", "Glow": "glow"}


def above(c):
    g, h = np.array(c["generated_all"]), c.get("held_out_all")
    return c["above_held"] if h is None else float(np.mean(g > max(h)))


def fam_rows(key):
    """Rows of every replicate evaluated (scripts/evaluate_family.py), each a list over the training areas."""
    out = [json.loads(p.read_text()) for p in sorted(FAM.glob(f"{key}_rep[0-9].json"))]
    n = min((len(r) for r in out), default=0)
    return [r[:n] for r in out] if n else None


def fplot(ax, reps, f, lab, ls="-", mfc=None, label=True):
    """Mean over replicates of f(row) against area, with the range shaded when there are several."""
    col, mk = STYLE[lab]
    x = [r["area_mm2"] for r in reps[0]]
    v = np.array([[f(r) for r in rows] for rows in reps])
    if len(reps) > 1:
        ax.fill_between(x, v.min(0), v.max(0), color=col, alpha=0.15, lw=0)
    ax.plot(x, v.mean(0), ls, marker=mk, color=col, mfc=mfc or col, ms=3, label=lab if label else None)


def area_axis(ax):
    ax.set_xscale("log")
    plain_log_axis(ax, "x", [0.02, 0.1, 0.3, 1, 3])
    ax.set_xlim(0.013, 4.5)
    ax.set_xlabel("training area per model (mm$^2$)")


def main():
    fm = load_fm()
    x_fm = np.array([r["area_mm2"] for r in fm[0]])
    fams = {lab: fam_rows(k) for lab, k in KEYS.items()}
    fams = {k: v for k, v in fams.items() if v}
    W, H = DOUBLE, 1.08 * DOUBLE
    fig = plt.figure(figsize=(W, H))
    # a: samples at the largest area
    labs = ["flow matching"] + list(fams)
    n_show, side = 8, 0.62
    fm_samples = np.load(CV / "rep0_samples.npz")
    for r, lab in enumerate(labs):
        if lab == "flow matching":
            wins = fm_samples["genA4"].astype(np.float32)
        else:
            sp = FAM / f"{KEYS[lab]}_rep0_samples.npz"  # samples of replicate 0
            if not sp.exists():
                continue
            d = np.load(sp)
            last = max(int(k[1:]) for k in d.files if k.startswith("A") and k[1:].isdigit())
            wins = d[f"A{last}"].astype(np.float32)
        for c in range(n_show):
            ax = fig.add_axes([(0.95 + c * (side + 0.04)) / W, 1 - (0.25 + (r + 1) * (side + 0.05)) / H, side / W, side / H])
            ax.imshow(wins[c], cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
            image_axis(ax)
            if c == 0:
                ax.text(-0.1, 0.5, lab, transform=ax.transAxes, ha="right", va="center", fontsize=6.5, color=STYLE[lab][0])
            if r == 0 and c == 0:
                panel_label(ax, "a", x=-1.4, y=1.08)
    fig.text(0.95 / W, 1 - 0.2 / H, "generated windows, model A, 3.2 mm$^2$ (flow matching: same noise as Fig. 2)", fontsize=6.5,
             va="bottom")
    y0, h, w = 0.08, 0.2, 0.25
    pos = [(0.08, 0.37), (0.41, 0.37), (0.74, 0.37), (0.08, y0), (0.41, y0), (0.74, y0)]
    axs = [fig.add_axes([px, py, w, h]) for px, py in pos]
    # b: copying
    ax = axs[0]
    v = np.array([[np.mean([above(r["copy"][s]) for s in "AB"]) for r in rep] for rep in fm])
    ax.plot(x_fm, v.mean(0), marker="o", color=STYLE["flow matching"][0], ms=3, label="flow matching")
    for lab, rr in fams.items():
        fplot(ax, rr, lambda r: np.mean([r["copy"][s]["above_held"] for s in "AB"]), lab)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("copying fraction")
    area_axis(ax)
    ax.legend(fontsize=5.3, loc="upper right")
    panel_label(ax, "b", x=-0.3)
    # c: agreement
    ax = axs[1]
    ax.plot(x_fm, np.mean([[r["agreement"] for r in rep] for rep in fm], 0), marker="o", color=STYLE["flow matching"][0], ms=3)
    dis = json.loads((ROOT / "results/sic/windows/disagreement.json").read_text())
    ax.plot([6.8], [np.mean(dis["single64_seed1,single64"]["agreement"])], "x", color=STYLE["flow matching"][0], ms=5)
    for lab, rr in fams.items():
        fplot(ax, rr, lambda r: r["agreement"], lab)
        fplot(ax, rr, lambda r: r["agreement_seeds"], lab, ls="--", mfc="white", label=False)
    ax.set_ylim(-0.15, 1.03)
    ax.axhline(0, color=INK2, lw=0.5)
    ax.set_ylabel("same-noise agreement")
    area_axis(ax)
    ax.set_xlim(0.013, 9)
    panel_label(ax, "c", x=-0.3)
    # d, e: FID to held-out and between A and B
    for ax, key, ylab, letter, real in ((axs[2], "fidelity", "FID to held-out", "d", "real_fidelity"),
                                        (axs[3], "convergence", "FID, model A vs model B", "e", "real_convergence")):
        f = (lambda r: np.mean([r["fidelity"][s]["fid"] for s in "AB"])) if key == "fidelity" else (lambda r: r["convergence"]["fid"])
        ax.plot(x_fm, np.mean([[f(r) for r in rep] for rep in fm], 0), marker="o", color=STYLE["flow matching"][0], ms=3)
        ax.plot(x_fm, np.mean([[r[real]["fid"] for r in rep] for rep in fm], 0), ls="--" if key == "fidelity" else ":",
                color=INK2, lw=1.0)
        for lab, rr in fams.items():
            fplot(ax, rr, f, lab)
        ax.set_yscale("log")
        plain_log_axis(ax, "y", [10, 20, 50, 100, 200, 500])
        ax.set_ylabel(ylab)
        area_axis(ax)
        panel_label(ax, letter, x=-0.3)
    # f: two-point distance between A and B
    ax = axs[4]
    ax.plot(x_fm, np.mean([[r["convergence"]["two_point"] for r in rep] for rep in fm], 0), marker="o",
            color=STYLE["flow matching"][0], ms=3)
    ax.plot(x_fm, np.mean([[r["real_convergence"]["two_point"] for r in rep] for rep in fm], 0), ls=":", color=INK2, lw=1.0)
    for lab, rr in fams.items():
        fplot(ax, rr, lambda r: r["convergence"]["two_point"], lab)
    ax.set_yscale("log")
    ax.set_ylabel("two-point distance, A vs B")
    area_axis(ax)
    panel_label(ax, "f", x=-0.3)
    # g: likelihood gap (held-out minus training; medians over windows where stored)
    ax = axs[5]
    med = lambda v: float(np.median(v))
    lk = sorted(CV.glob("likelihood_rep*.json"))
    if lk:
        reps = [json.loads(p.read_text()) for p in lk]
        gap = [[np.mean([med(r[s]["held_out"]) - med(r[s]["training"]) for s in "AB" if s in r]) for r in rep] for rep in reps]
        n = min(len(g) for g in gap)
        ax.plot(x_fm[:n], np.mean([g[:n] for g in gap], 0), marker="o", color=STYLE["flow matching"][0], ms=3,
                label="flow matching (as a CNF)")
    if "Glow" in fams:
        b = lambda d, k: d.get(k + "_median", d[k])
        fplot(ax, fams["Glow"], lambda r: np.mean([b(r["bits_per_dim"][s], "held_out") - b(r["bits_per_dim"][s], "training")
                                                   for s in "AB"]), "Glow")
    ax.set_yscale("log")
    ax.set_ylabel("bits per dimension,\nheld-out minus training")
    area_axis(ax)
    ax.legend(fontsize=5.3, loc="upper right")
    panel_label(ax, "g", x=-0.3)
    for a_ in axs[4:5]:
        plain_log_axis(a_, "y", [0.001, 0.003, 0.01, 0.03, 0.1])
    fig.savefig(ROOT / "figures/figS_families.pdf", dpi=300)
    fig.savefig(ROOT / "figures/figS_families.png", dpi=220)


if __name__ == "__main__":
    main()
