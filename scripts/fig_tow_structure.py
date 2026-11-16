"""Figure 4: at the tow scale, only models whose crops contain the bands at low resolution generate them, and those
models copy them from the training band; at both scales copying stops at roughly a hundred training crops.

a: fields of 0.8 mm of the held-out band, of the single-scale models with 64 and 512 px crops, and of the two-scale
   model with its layout model trained for 6000 steps (top), and their fiber-fraction layouts (bottom; local fiber
   fraction, Gaussian of 12 um, 8.3 um per pixel).
b: correlation of the fiber-fraction layout along the bands; real: training band (solid), held-out band (dotted);
   models: 24 fields of 1.06 mm on the same noise.
c: windows of 0.53 mm of fields of that two-scale model (top; the most similar window to the training band, and the
   windows at the top 10% and 25%) and the most similar region of the training band (bottom, flipped as matched),
   matched by the normalized cross-correlation (NCC) of their fiber-fraction layouts.
d: fraction of generated windows more similar to the training data than any held-out window, against the number of
   crops that fit into the training data: paired fiber-scale models (64 px crops, five areas, three replicates),
   single-scale models of the whole band, and layout models (0.13-0.53 mm crops; a quarter, a half and all of the band).
Data: results/sic/towscale/*.json, *_fields.npz (scripts/evaluate_tow_scale.py); results/sic/convergence/rep*.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.figstyle import BLUES, DOUBLE, INK, INK2, SERIES, image_axis, panel_label, plt  # noqa: E402
from scripts.fig_tow_copying import scaling  # noqa: E402
from genmicro.twoscale import state_fields  # noqa: E402

TS = ROOT / "results/sic/towscale"
BAND, HELD = 2640, 2768
PX = 8 * 1.04
SHOW = 768
COPYING = "twoscale_stateaug_s1"  # two-scale model whose layout model is trained for 6000 steps
FIELDS = [("held-out section", None), ("single scale,\n64 px crops", "single64"), ("single scale,\n512 px crops", "single512"),
          ("two-scale, layout model\n6000 steps", COPYING)]
CURVES = [("single scale, 64 px", "single64", BLUES[1]), ("single scale, 128 px", "single128", BLUES[2]),
          ("single scale, 256 px", "single256", BLUES[3]), ("single scale, 512 px", "single512", BLUES[4]),
          ("two-scale (layout 6000 steps)", COPYING, SERIES[1])]


def load(name):
    p = TS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def lay_of(im):
    return (state_fields(im)[0, 0].cpu().numpy() + 1) / 2


def box_sum(a, K):
    c = torch.nn.functional.pad(a.double().cumsum(0).cumsum(1), (1, 0, 1, 0))
    return (c[K:, K:] - c[:-K, K:] - c[K:, :-K] + c[:-K, :-K]).float()


def best_match(q, fr, mu, sd, H, W):
    """Position and flip of the window of a region (given by its FFT and window statistics) most similar to q."""
    K = q.shape[-1]
    best = (-2, None)
    for flip in ((), (-1,), (-2,), (-1, -2)):
        t = torch.as_tensor(q.copy()).float()
        t = t.flip(flip) if flip else t
        t = (t - t.mean()) / (t.std(unbiased=False) + 1e-8)
        c = torch.fft.irfft2(fr * torch.fft.rfft2(t, s=(H, W)).conj(), s=(H, W))[:H - K + 1, :W - K + 1] / (K * K * sd)
        v, j = c.flatten().max(0)
        if v > best[0]:
            best = (float(v), (int(j) // (W - K + 1), int(j) % (W - K + 1), flip))
    return best


def copy_examples(plane):
    ims = np.load(TS / f"{COPYING}_fields.npz")["images"].astype(np.float32)
    band = plane[:, :BAND]
    r = torch.as_tensor(lay_of(band)).float()
    H, W = r.shape
    fr = torch.fft.rfft2(r)
    mu = box_sum(r, 64) / 64 ** 2
    sd = (box_sum(r * r, 64) / 64 ** 2 - mu * mu).clamp_min(1e-8).sqrt()
    cands = []
    for im in ims:
        lay = lay_of(im)
        for y in range(0, lay.shape[0] - 63, 32):
            for x in range(0, lay.shape[1] - 63, 32):
                v, pos = best_match(lay[y:y + 64, x:x + 64], fr, mu, sd, H, W)
                cands.append((v, im, y, x, pos))
    cands.sort(key=lambda c: -c[0])
    out = []
    for v, im, y, x, (py, px, flip) in [cands[0], cands[len(cands) // 10], cands[len(cands) // 4]]:
        t = band[8 * py:8 * py + 512, 8 * px:8 * px + 512]
        out.append((v, im[8 * y:8 * y + 512, 8 * x:8 * x + 512], np.flip(t, flip) if flip else t))
    return out, len(cands)


def main():
    real = json.loads((TS / "real.json").read_text())
    plane = np.load(ROOT / "data/sic/section_plane.npy").astype(np.float32) / 127.5 - 1
    W, H = DOUBLE, 0.79 * DOUBLE
    fig = plt.figure(figsize=(W, H))
    side, gap, x0, top = 1.04, 0.06, 0.24, 0.3
    for i, (lab, name) in enumerate(FIELDS):
        im = plane[800:800 + SHOW, HELD:HELD + SHOW] if name is None else \
            np.load(TS / f"{name}_fields.npz")["images"][0].astype(np.float32)[:SHOW, :SHOW]
        xa = x0 + i * (side + gap)
        a = fig.add_axes([xa / W, 1 - (top + side) / H, side / W, side / H])
        a.imshow(im, cmap="gray", vmin=-1, vmax=1, interpolation="lanczos")
        a.set_title(lab, fontsize=6.3, pad=2)
        image_axis(a)
        b = fig.add_axes([xa / W, 1 - (top + 2 * side + 0.05) / H, side / W, side / H])
        b.imshow(lay_of(im), cmap="Greys_r", vmin=0, vmax=0.65, interpolation="nearest")
        image_axis(b)
        if i == 0:
            panel_label(a, "a", x=-0.2, y=1.12)
            b.text(-0.05, 0.5, "fiber fraction", rotation=90, transform=b.transAxes, ha="right", va="center", fontsize=6.3)
            a.add_patch(plt.Rectangle((SHOW - 330, SHOW - 110), 310, 95, color="black", alpha=0.6, lw=0))
            a.plot([SHOW - 280, SHOW - 40], [SHOW - 35] * 2, color="white", lw=1.6)
            a.text(SHOW - 160, SHOW - 55, "250 µm", color="white", ha="center", va="bottom", fontsize=5.8)
    # b: correlation along the bands
    bx = x0 + 4 * (side + gap) + 0.5
    ax = fig.add_axes([bx / W, 1 - (top + 2 * side + 0.05) / H, (W - bx - 0.03) / W, (2 * side) / H])
    lags = np.arange(65) * PX
    ax.plot(lags, real["training band"]["corr_along"], color=INK, lw=1.4, label="real, training band")
    ax.plot(lags, real["held-out band"]["corr_along"], ":", color=INK, lw=1.4, label="real, held-out band")
    for lab, name, col in CURVES:
        d = load(name)
        if d:
            ax.plot(lags, d["tow"]["corr_along"], color=col, lw=1.1, label=lab)
    ax.axhline(0, color=INK2, lw=0.5)
    ax.set_xlim(0, 64 * PX)
    ax.set_xlabel("separation along the bands (µm)")
    ax.set_ylabel("correlation of the fiber fraction")
    ax.legend(fontsize=5.2, loc="upper right", handlelength=1.5)
    panel_label(ax, "b", x=-0.24, y=1.02)
    # c: copies
    ex, n = copy_examples(plane)
    s2, top2 = 0.95, top + 2 * side + 0.55
    for i, (v, g, t) in enumerate(ex):
        for j, (img, lab) in enumerate(((g, f"generated, NCC {v:.2f}"), (t, "training band"))):
            a = fig.add_axes([(0.05 + i * (s2 + 0.07)) / W, 1 - (top2 + (j + 1) * s2 + j * 0.2) / H, s2 / W, s2 / H])
            a.imshow(img, cmap="gray", vmin=-1, vmax=1, interpolation="lanczos")
            a.set_title(lab, fontsize=6, pad=1.5)
            image_axis(a)
            if i == 0 and j == 0:
                panel_label(a, "c", x=-0.05, y=1.15)
    # d: copying against the number of crops
    ax = fig.add_axes([0.6, 0.075, 0.38, 0.37])
    scaling(ax, real, layout_color=SERIES[1])
    panel_label(ax, "d", x=-0.32, y=1.02)
    fig.savefig(ROOT / "figures/fig4_tow_copying.pdf", dpi=300)
    fig.savefig(ROOT / "figures/fig4_tow_copying.png", dpi=220)
    print("windows searched", n, "NCC of examples", [round(e[0], 2) for e in ex])


if __name__ == "__main__":
    main()
