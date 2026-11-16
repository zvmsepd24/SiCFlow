"""Layout models alone, for the tow-scale tests: copying against training area and crop size, and paired models.

A layout model is the first stage of the two-scale model (genmicro.twoscale): an unconditional FlowUNet on
crops of the state layout (local fiber and pore fractions at 8.3 um per pixel), with random horizontal and
vertical flips. Here the training region and the crop size vary:
  --cols a:b    columns of the section (full-resolution pixels) used for training; default the training band
  --crop c      crop size in layout pixels (64 = 0.53 mm, as in the two-scale model)
The number of crops per step is 64 * (64 / c)^2, so every model sees the same number of pixels per step.
Usage: python scripts/train_layout.py NAME [--cols 0:2640] [--crop 64] [--seed 1] [--steps 6000]
       -> results/sic/layouts/NAME.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genmicro import sic  # noqa: E402
from genmicro.net import UNET, FlowUNet  # noqa: E402
from genmicro.train import train  # noqa: E402
from genmicro.twoscale import DOWN, flips, state_fields, strip  # noqa: E402

OUT = ROOT / "results/sic/layouts"


def main(a):
    torch.backends.cudnn.benchmark = True
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / f"{a.name}.pt").exists():
        print(f"{a.name} exists, skipped")
        return
    c0, c1 = map(int, a.cols.split(":"))
    lay = state_fields(sic.load_plane()[:, c0:c1])[0]
    h, w = lay.shape[-2:]
    c = a.crop
    n = 64 * (64 // c) ** 2
    g = torch.Generator(device="cuda").manual_seed(500 + a.seed)
    cfg = dict(UNET, channels=lay.shape[0])

    def batch(m, step):
        ys = torch.randint(0, h - c + 1, (m,), device="cuda", generator=g)
        xs = torch.randint(0, w - c + 1, (m,), device="cuda", generator=g)
        crops = torch.stack([lay[:, y:y + c, x:x + c] for y, x in zip(ys.tolist(), xs.tolist())])
        return flips(crops, g), None

    t0 = time.perf_counter()
    ema, _ = train(FlowUNet(**cfg).cuda().to(memory_format=torch.channels_last), batch, a.steps, batch=n, seed=a.seed)
    meta = dict(cols=[c0, c1], crop=c, steps=a.steps, seed=a.seed, layout="state", layout_shape=[h, w],
                crop_areas=round(h * w / c ** 2, 1), seconds=round(time.perf_counter() - t0))
    torch.save(dict(model_config=cfg, ema=strip(ema.state_dict()), meta=meta), OUT / f"{a.name}.pt")
    print(json.dumps(dict(name=a.name, **meta)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("--cols", default="0:2640")
    p.add_argument("--crop", type=int, default=64)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps", type=int, default=6000)
    torch.cuda.set_per_process_memory_fraction(0.45)
    main(p.parse_args())
