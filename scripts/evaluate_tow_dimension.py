"""Intrinsic dimension at the tow scale: do layout models that copy sit at the dimension of their training windows?

The fiber-scale test (scripts/evaluate_checkerboard.py) found that models which copy their training tiles
generate windows whose Levina-Bickel dimension equals that of windows cut from the training tiles, and that
models which generalize reach the dimension of held-out material. Here the same estimate is made for windows
of 64 layout px (0.53 mm, the crop of the layout models) of the fiber-fraction layout (genmicro.twoscale).
Clouds of N windows, each window described by
  * pixels: the fiber-fraction layout (4096 values);
  * two-point: its autocovariance along and across, lags 0-32 layout px (66 values).
Real clouds: windows at random positions (with random flips, as in training) inside regions of increasing area
cut from the left edge of the training band, full height; the two halves; the band; the held-out band.
Generated clouds: N layouts of 64 px from noise, for every layout model given on the command line.
Each estimate is repeated R times with new windows or new noise; the mean and spread are stored.
Usage: python scripts/evaluate_tow_dimension.py [MODEL ...]   MODEL = layouts/NAME
       -> results/sic/towscale/dimension.json (results are merged into an existing file)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genmicro.lid import levina_bickel  # noqa: E402
from genmicro.scores import load_model  # noqa: E402
from genmicro.train import sample  # noqa: E402
from genmicro.twoscale import state_fields  # noqa: E402

OUT = ROOT / "results/sic/towscale/dimension.json"
BAND, HELD = 2640, 2768
K, N, R, KNN, LAG = 64, 512, 5, 10, 32
WIDTHS = (66, 82, 124, 165, 248, 330)  # layout px from the left edge of the training band (layout height 297)
MODELS = ["layouts/A500_s1", "layouts/A500_s2", "layouts/B500_s1", "layouts/A1000_s1", "layouts/A1000_s2",
          "layouts/B1000_s1", "layouts/A_s1", "layouts/A_s2", "layouts/B_s1", "layouts/steps500_s1",
          "layouts/steps500_s2", "layouts/steps1000_s1", "layouts/steps1000_s2", "layouts/steps1000_s3",
          "layouts/steps2000_s1", "layouts/full_s2", "layouts/quarter_s1"]


def describe(w):
    """w: (n, K, K) fiber-fraction windows in [0, 1] -> dict of descriptor clouds."""
    a = w - w.mean((1, 2), keepdims=True)
    cx = [(a[:, :, :K - k] * a[:, :, k:]).mean((1, 2)) for k in range(LAG + 1)]
    cy = [(a[:, :K - k] * a[:, k:]).mean((1, 2)) for k in range(LAG + 1)]
    return dict(pixels=w.reshape(len(w), -1).astype(np.float64), two_point=np.stack(cx + cy, 1).astype(np.float64))


def dims(w):
    return {k: float(np.nanmean(levina_bickel(v, k=KNN))) for k, v in describe(w).items()}


def real_windows(region, n, rng):
    h, wd = region.shape
    ys, xs = rng.integers(0, h - K + 1, n), rng.integers(0, wd - K + 1, n)
    out = np.stack([region[y:y + K, x:x + K] for y, x in zip(ys, xs)])
    fx, fy = rng.random(n) < 0.5, rng.random(n) < 0.5
    out[fx] = out[fx, :, ::-1]
    out[fy] = out[fy, ::-1]
    return out


@torch.no_grad()
def generated_windows(model, n, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = []
    for i in range(0, n, 128):
        z = torch.randn(min(128, n - i), 2, K, K, device="cuda", generator=g)
        with torch.autocast("cuda", torch.bfloat16):
            out.append(sample(model, z).float()[:, 0].cpu().numpy())
    return (np.concatenate(out) + 1) / 2


def summary(vals):
    return {k: dict(mean=float(np.mean([v[k] for v in vals])), sd=float(np.std([v[k] for v in vals])),
                    all=[round(v[k], 3) for v in vals]) for k in vals[0]}


def main(names):
    torch.backends.cudnn.benchmark = True
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    plane = np.load(ROOT / "data/sic/section_plane.npy").astype(np.float32) / 127.5 - 1
    frac = lambda c0, c1: ((state_fields(plane[:, c0:c1])[0][0].cpu().numpy()) + 1) / 2
    band, held = frac(0, BAND), frac(HELD, plane.shape[1])
    regions = {f"left {w}": band[:, :w] for w in WIDTHS}
    regions.update({"left half": band[:, :165], "right half": band[:, 165:], "training band": band, "held-out band": held})
    real = res.setdefault("real", {})
    for name, reg in regions.items():
        if name in real:
            continue
        vals = [dims(real_windows(reg, N, np.random.default_rng(100 + r))) for r in range(R)]
        real[name] = dict(shape=list(reg.shape), crops=round(reg.size / K ** 2, 1), **summary(vals))
        print(name, real[name]["crops"], {k: round(v["mean"], 2) for k, v in real[name].items() if isinstance(v, dict)},
              flush=True)
    gen = res.setdefault("generated", {})
    for name in names:
        model = load_model(ROOT / "results/sic" / f"{name}.pt")
        vals = [dims(generated_windows(model, N, 7000 + r)) for r in range(R)]
        gen[name] = summary(vals)
        print(name, {k: f"{v['mean']:.2f} ± {v['sd']:.2f}" for k, v in gen[name].items()}, flush=True)
        OUT.write_text(json.dumps(res, indent=1))
        torch.cuda.empty_cache()
    res.update(meta=dict(window=K, n=N, repeats=R, k=KNN, lags=LAG))
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.45)
    main(sys.argv[1:] or MODELS)
