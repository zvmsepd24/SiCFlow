"""Figure 8: only the models that learn the fiber-fraction layout itself generate the tow bands.

a: a field of the held-out band and of each model of the whole training band (single scale, crops of 64 to 512 px;
two-scale), with the fiber-fraction layout of each field below it (genmicro.twoscale.state_fields: local fiber
fraction, Gaussian of 12 px, 8.3 um per pixel). b, c: correlation of the fiber-fraction layout along and across the
bands against separation; real: training and held-out bands; models: 24 fields of 1024 px each.
Data: results/sic/towscale/*.json and *_fields.npz (scripts/evaluate_tow_scale.py).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.figstyle import BLUES, DOUBLE, INK, INK2, SERIES, image_axis, panel_label, plt  # noqa: E402
from genmicro.twoscale import state_fields  # noqa: E402

TS = ROOT / "results/sic/towscale"
PX = 8 * 1.04  # um per layout pixel
SHOW = 768  # px of each field shown
MODELS = [("single scale,\n64 px crops", "single64", BLUES[1]), ("single scale,\n256 px crops", "single256", BLUES[3]),
          ("single scale,\n512 px crops", "single512", BLUES[4]), ("two-scale", "twoscale_early_2k_s1", SERIES[2])]
CURVES = [("single scale, 64 px", "single64", BLUES[1]), ("single scale, 128 px", "single128", BLUES[2]),
          ("single scale, 256 px", "single256", BLUES[3]), ("single scale, 512 px", "single512", BLUES[4]),
          ("two-scale", "twoscale_early_2k_s1", SERIES[2])]


def load(name):
    p = TS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def main():
    real = json.loads((TS / "real.json").read_text())
    plane = np.load(ROOT / "data/sic/section_plane.npy").astype(np.float32) / 127.5 - 1
    held = plane[800:800 + SHOW, 2768:2768 + SHOW]
    cols = [("held-out section", held, None)]
    for lab, name, _ in MODELS:
        f = TS / f"{name}_fields.npz"
        if f.exists():
            cols.append((lab, np.load(f)["images"][0].astype(np.float32)[:SHOW, :SHOW], name))
    n = len(cols)
    W = DOUBLE
    side = (W - 0.1 * (n - 1)) / n
    fig = plt.figure(figsize=(W, 2 * side + 2.8))
    H = fig.get_figheight()
    top = 0.32
    for i, (lab, im, name) in enumerate(cols):
        x0 = i * (side + 0.1) / W
        a = fig.add_axes([x0, 1 - (top + side) / H, side / W, side / H])
        a.imshow(im, cmap="gray", vmin=-1, vmax=1, interpolation="lanczos")
        a.set_title(lab, fontsize=6.8, pad=2)
        image_axis(a)
        b = fig.add_axes([x0, 1 - (top + 2 * side + 0.06) / H, side / W, side / H])
        lay = (state_fields(im)[0, 0].cpu().numpy() + 1) / 2
        b.imshow(lay, cmap="Greys_r", vmin=0, vmax=0.65, interpolation="nearest")
        image_axis(b)
        if i == 0:
            panel_label(a, "a", x=-0.04, y=1.13)
            b.text(-0.04, 0.5, "fiber fraction", rotation=90, transform=b.transAxes, ha="right", va="center", fontsize=6.5)
            a.add_patch(plt.Rectangle((SHOW - 330, SHOW - 110), 310, 95, color="black", alpha=0.6, lw=0))
            a.plot([SHOW - 40 - 240, SHOW - 40], [SHOW - 35] * 2, color="white", lw=2)
            a.text(SHOW - 40 - 120, SHOW - 55, "250 µm", color="white", ha="center", va="bottom", fontsize=6.3)
    lags = np.arange(65) * PX
    axs = [fig.add_axes([0.08, 0.5 / H, 0.38, 1.5 / H]), fig.add_axes([0.58, 0.5 / H, 0.38, 1.5 / H])]
    for ax, key, title in ((axs[0], "corr_along", "fiber fraction, along the tow bands"),
                           (axs[1], "corr_across", "fiber fraction, across the tow bands")):
        ax.plot(lags, real["training band"][key], color=INK, lw=1.4, label="real, training band")
        ax.plot(lags, real["held-out band"][key], ":", color=INK, lw=1.4, label="real, held-out band")
        for lab, name, col in CURVES:
            d = load(name)
            if d:
                ax.plot(lags, d["tow"][key], color=col, lw=1.2, label=lab)
        ax.axhline(0, color=INK2, lw=0.5)
        ax.set_xlim(0, 64 * PX)
        ax.set_xlabel("separation (µm)")
        ax.set_title(title, fontsize=7, pad=3)
    axs[0].set_ylabel("correlation")
    axs[1].legend(fontsize=5.5, loc="upper right")
    panel_label(axs[0], "b", x=-0.16)
    panel_label(axs[1], "c", x=-0.16)
    fig.savefig(ROOT / "figures/fig8_tow_fields.pdf", dpi=300)
    fig.savefig(ROOT / "figures/fig8_tow_fields.png", dpi=220)


if __name__ == "__main__":
    main()
