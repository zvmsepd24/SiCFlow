"""Figure 9: models that generate the tow bands copy them from the training band.

a: windows of 0.53 mm of fields of the two-scale model with the layout model trained for 6000 steps (top; the most similar
   of 98 windows to the training band, and those at the top 10% and 25%) and the most similar region of the training
   band (bottom, flipped
   as matched), found by the normalized cross-correlation (NCC) of their fiber-fraction layouts.
b: NCC of 128 windows of 0.53 mm (fiber-fraction layout) with the most similar window of the training band, for
   held-out windows and for the fields of each model of the whole training band. Dots: windows; bar: median.
c: copying against the number of crops that fit in the training data (training area / crop area): fraction of
   generated windows (of the crop size) more similar to the training data than any held-out window is, for the
   paired fiber-scale models (64 px crops, five areas, three replicates), the single-scale models of the whole band
   (image windows) and layout models (crops of 0.13, 0.27 and 0.53 mm; a quarter, half and all of the band).
Data: results/sic/towscale/*.json, *_fields.npz (scripts/evaluate_tow_scale.py); results/sic/convergence/rep*.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.figstyle import BLUES, DOUBLE, INK, INK2, SERIES, image_axis, panel_label, plain_log_axis, plt  # noqa: E402
from scripts.evaluate_tow_scale import box_sum  # noqa: E402
from genmicro.twoscale import state_fields  # noqa: E402

TS = ROOT / "results/sic/towscale"
CV = ROOT / "results/sic/convergence"
BAND = 2640
GRAY = "#9a9a9a"
GROUPS = [("held-out band", None, GRAY), ("single scale, 64 px", "single64", BLUES[1]),
          ("single scale, 128 px", "single128", BLUES[2]), ("single scale, 256 px", "single256", BLUES[3]),
          ("single scale, 512 px", "single512", BLUES[4]),
          ("two-scale, layout 6000 steps", "twoscale_stateaug_s1", SERIES[1]),
          ("two-scale, both stages\nstopped early", "twoscale_early_2k_s1", SERIES[2])]
LAYOUTS = [("layouts_quarter_s1", 64, 660), ("layouts_A_s1", 64, 1320), ("layouts_A_s2", 64, 1320),
           ("layouts_B_s1", 64, 1320), ("twoscale_layout_state_s1", 64, BAND), ("layouts_full_s2", 64, BAND),
           ("layouts_crop32_A_s1", 32, 1320), ("layouts_crop32_A_s2", 32, 1320), ("layouts_crop32_B_s1", 32, 1320),
           ("layouts_crop32_s1", 32, BAND), ("layouts_crop16_s1", 16, BAND)]


def load(name):
    p = TS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


@torch.no_grad()
def best_match(q, region):
    """Position (y, x) and flip of the K x K window of region most similar (NCC) to q, and the NCC."""
    K = q.shape[-1]
    H, W = region.shape
    r = torch.as_tensor(region).float().cuda()
    fr = torch.fft.rfft2(r)
    mu = box_sum(r, K) / K ** 2
    sd = (box_sum(r * r, K) / K ** 2 - mu * mu).clamp_min(1e-8).sqrt()
    best = (-2, None)
    for f, flip in enumerate(((), (-1,), (-2,), (-1, -2))):
        t = torch.as_tensor(q).float().cuda()
        t = t.flip(flip) if flip else t
        t = (t - t.mean()) / (t.std(unbiased=False) + 1e-8)
        c = torch.fft.irfft2(fr * torch.fft.rfft2(t, s=(H, W)).conj(), s=(H, W))[:H - K + 1, :W - K + 1] / (K * K * sd)
        v, j = c.flatten().max(0)
        if v > best[0]:
            best = (float(v), (int(j) // (W - K + 1), int(j) % (W - K + 1), flip))
    return best


def examples(fig, plane, name="twoscale_stateaug_s1", n=3):
    """Windows of 0.53 mm of two-scale fields (top) and the most similar region of the training band (bottom)."""
    d = np.load(TS / f"{name}_fields.npz")
    ims = d["images"].astype(np.float32)
    band = plane[:, :BAND]
    lay_band = (state_fields(band)[0, 0].cpu().numpy() + 1) / 2
    cands = []
    for im in ims:
        lay = (state_fields(im)[0, 0].cpu().numpy() + 1) / 2
        for y in range(0, lay.shape[0] - 63, 32):
            for x in range(0, lay.shape[1] - 63, 32):
                v, pos = best_match(lay[y:y + 64, x:x + 64], lay_band)
                cands.append((v, im, y, x, pos))
    cands.sort(key=lambda c: -c[0])
    picks = [cands[0], cands[len(cands) // 10], cands[len(cands) // 4]][:n]  # best, top 10%, top 25%
    W, H = fig.get_figwidth(), fig.get_figheight()
    side, gap, x0, top = 1.0, 0.06, 0.05, 0.32
    for i, (v, im, y, x, (py, px, flip)) in enumerate(picks):
        g = im[8 * y:8 * y + 512, 8 * x:8 * x + 512]
        t = band[8 * py:8 * py + 512, 8 * px:8 * px + 512]
        t = np.flip(t, flip) if flip else t
        for j, (img, lab) in enumerate(((g, f"generated, NCC {v:.2f}"), (t, "training band"))):
            a = fig.add_axes([(x0 + i * (side + gap)) / W, 1 - (top + (j + 1) * side + j * 0.32) / H, side / W, side / H])
            a.imshow(img, cmap="gray", vmin=-1, vmax=1, interpolation="lanczos")
            a.set_title(lab, fontsize=6, pad=1.5)
            image_axis(a)
            if i == 0 and j == 0:
                panel_label(a, "a", x=-0.08, y=1.25)
    return cands


def strips(ax, real):
    for i, (lab, name, col) in enumerate(GROUPS):
        if name is None:
            v = np.array(real["copy_held"][f"0:{BAND} 64"])
        else:
            d = load(name)
            if d is None:
                continue
            v = np.array(d["copy"]["64"])
        y = len(GROUPS) - 1 - i + np.random.default_rng(i).uniform(-0.25, 0.25, len(v))
        ax.scatter(v, y, s=2.5, color=col, lw=0, alpha=0.7)
        ax.plot([np.median(v)] * 2, [len(GROUPS) - 1 - i - 0.38, len(GROUPS) - 1 - i + 0.38], color=INK, lw=1.2)
    ax.set_yticks(range(len(GROUPS)))
    ax.set_yticklabels([g[0] for g in GROUPS][::-1])
    ax.set_xlim(0, 1)
    ax.set_xlabel("NCC with the most similar region\nof the training band (0.53 mm windows)")


def scaling(ax, real, layout_color=SERIES[2]):
    """Fraction of generated windows more similar to the training data than any held-out window is."""
    px = 1.04e-3
    pts = []
    for r in (0, 1, 2):
        for e in json.loads((CV / f"rep{r}.json").read_text()):
            g, h = np.array(e["copy"]["A"]["generated_all"]), np.array(e["copy"]["A"]["held_out_all"])
            pts.append(("paired", e["area_mm2"] / (64 * px) ** 2, np.mean(g > h.max())))
    for name, crop in (("single64", 64), ("single128", 128), ("single256", 256), ("single512", 512)):
        d = load(name)
        if d is not None and str(crop) in d.get("copy_image", {}):
            g, h = np.array(d["copy_image"][str(crop)]), np.array(real["copy_image_held"][str(crop)])
            pts.append(("single512" if crop == 512 else "single", 2376 * 2640 / crop ** 2, np.mean(g > h.max())))
    for name, crop, cols in LAYOUTS:
        d = load(name)
        if d is None:
            continue
        held = np.array(real["copy_held"][f"{d['cols'][0]}:{d['cols'][1]} {crop}"])
        gen = np.array(d["copy"][str(crop)])
        pts.append(("layout", 297 * (cols // 8) / crop ** 2, np.mean(gen > held.max())))
    styles = {"paired": ("paired models", BLUES[3], "o"),
              "single": ("single-scale models", BLUES[4], "D"),
              "layout": ("layout models", layout_color, "s"),
              "single512": ("512 px crops (no bands)", BLUES[4], "d")}
    for key, (lab, col, m) in styles.items():
        p = [(n, v) for k, n, v in pts if k == key]
        if p:
            n, v = np.array(p).T
            if key == "single512":
                ax.scatter(n, v, facecolor="white", edgecolor=col, marker=m, s=16, label=lab, zorder=3, lw=0.8)
            else:
                ax.scatter(n, v, color=col, marker=m, s=16, label=lab, zorder=3, edgecolor="white", lw=0.4)
    ax.set_xscale("log")
    plain_log_axis(ax, "x", [3, 10, 30, 100, 300, 1000])
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("training area / crop area")
    ax.set_ylabel("fraction of generated windows closer to\nthe training data than any held-out window")
    ax.legend(fontsize=5.5, loc="upper right")


def main():
    real = json.loads((TS / "real.json").read_text())
    plane = np.load(ROOT / "data/sic/section_plane.npy").astype(np.float32) / 127.5 - 1
    fig = plt.figure(figsize=(DOUBLE, 0.5 * DOUBLE))
    examples(fig, plane)
    a2 = fig.add_axes([0.7, 0.6, 0.28, 0.36])
    strips(a2, real)
    panel_label(a2, "b", x=-0.78)
    a3 = fig.add_axes([0.7, 0.1, 0.28, 0.33])
    scaling(a3, real)
    panel_label(a3, "c", x=-0.42)
    fig.savefig(ROOT / "figures/fig9_tow_copying.pdf", dpi=300)
    fig.savefig(ROOT / "figures/fig9_tow_copying.png", dpi=220)


if __name__ == "__main__":
    main()
