"""Paired test of the fiber model: fiber models trained (with noise on the layout) on the left and right halves of the
training band, and a second seed on the left half, given the same layouts and the same noise.

Same split as the paired test of the layout models (scripts/train_layout.py A/B), so the two tests differ only in scale.
Computed, for 4 noise samples:
* same-noise agreement: correlation of two models' images generated from the same layout (the held-out band, rows
  0-1023, columns 2768-3591) and the same noise, after removing the layout scale (Gaussian of 12 px), and without
  removing it;
* copying: correlation (layout scale removed) of the image generated from the layout of a region of each half (rows
  600-1623, columns 200-1223 in half A and 1400-2423 in half B) with that region.
Usage: python scripts/evaluate_fiber_pairs.py [VARIANT] -> results/sic/towscale/fiber_pairs[_VARIANT].json
       (VARIANT: stateauglong for the fiber models trained twice as long)
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genmicro.scores import load_model  # noqa: E402
from genmicro.train import sample  # noqa: E402
from genmicro.twoscale import COND_NOISE, STATE_SIGMA, ImageConditioned, condition, gaussian_blur, state_fields  # noqa: E402

OUT = ROOT / "results/sic/towscale"
TW = ROOT / "results/sic/twoscale"
MODELS = {"A": "fiber_stateaug_A_s1", "A2": "fiber_stateaug_A_s2", "B": "fiber_stateaug_B_s1", "band": "fiber_stateaug_s1"}
REGIONS = {"A": (slice(600, 1624), slice(200, 1224)), "B": (slice(600, 1624), slice(1400, 2424))}
N = 4


def high(a):
    t = torch.as_tensor(a)[None, None].float()
    return (t - gaussian_blur(t, STATE_SIGMA))[0, 0].numpy()


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


@torch.no_grad()
def generate(fine, img, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    lay = state_fields(img).cuda()
    with torch.autocast("cuda", torch.bfloat16):
        c = condition(lay, COND_NOISE, g)
        x = sample(fine, torch.randn(1, 1, *img.shape, device="cuda", generator=g), cond=c)
    return x.float()[0, 0].cpu().numpy()


def main(variant="stateaug"):
    global MODELS
    MODELS = {k: v.replace("stateaug", variant) for k, v in MODELS.items()}
    plane = np.load(ROOT / "data/sic/section_plane.npy").astype(np.float32) / 127.5 - 1
    held = plane[:1024, 2768:2768 + 824]
    models = {k: ImageConditioned(load_model(TW / f"{v}.pt")).eval() for k, v in MODELS.items() if (TW / f"{v}.pt").exists()}
    ims = {k: [generate(m, held, s) for s in range(N)] for k, m in models.items()}
    res = dict(models=MODELS, agreement_high={}, agreement={}, copying={})
    for a, b in combinations(models, 2):
        res["agreement_high"][f"{a} | {b}"] = float(np.mean([corr(high(x), high(y)) for x, y in zip(ims[a], ims[b])]))
        res["agreement"][f"{a} | {b}"] = float(np.mean([corr(x, y) for x, y in zip(ims[a], ims[b])]))
    for k, m in models.items():
        res["copying"][k] = {}
        for r, (ys, xs) in REGIONS.items():
            reg = plane[ys, xs]
            res["copying"][k][r] = corr(high(generate(m, reg, 100)), high(reg))
    (OUT / ("fiber_pairs.json" if variant == "stateaug" else f"fiber_pairs_{variant}.json")).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.4)
    main(sys.argv[1] if len(sys.argv) > 1 else "stateaug")
