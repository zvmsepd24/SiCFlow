"""Two-scale models of the whole training band (genmicro.twoscale).

Layout model: unconditional FlowUNet on 64 x 64 px crops of the layout of the training band (0.53 mm),
with random horizontal and vertical flips; 6000 steps of 64 crops.
Fiber model: FlowUNet on 128 x 128 px crops of the full-resolution image, with the layout of the same
region, upsampled, as extra input channels; 32 crops per step. Crop corners are multiples of 8 px, so the
crops are aligned with the layout pixels.
Variants of the fiber model: base (8000 steps); long (the same, 16000 steps); wide (channels 96, 192, 384 and
384; 12000 steps); guided (the layout is dropped for 10% of the crops, so the model can be sampled with
classifier-free guidance; 12000 steps). With the variant gray, the layout is instead the gray-level image
average-pooled 8x (one channel), and both models are trained for that layout. With the variant state, the
layout is the local fiber and pore fraction (genmicro.twoscale.state_fields), and both models are trained for it.
The variant stateaug uses the state layout model and trains the fiber model with conditioning augmentation: noise of a
level drawn from [0, 1] is added to each layout crop at layout resolution, and the level is a third channel
(genmicro.twoscale.noisy_condition).
For the paired test of the fiber model, a fiber model of the variant stateaug can be trained on a column range
of the training band alone (python scripts/train_two_scale.py SEED stateaug --cols 0:1320 --tag A -> fiber_stateaug_A_s{SEED}.pt;
no layout model is trained).
Checks of the noise: stateaug05 (noise levels from [0, 0.5]). Training length of the fiber model with noise:
stateaug1k, stateaug2k, stateaug3k, stateaugshort, stateaug6k, stateaug, stateauglong (1000 to 16000 steps).
Usage: python scripts/train_two_scale.py SEED [base|long|wide|guided|gray|state|stateaug|stateaug05|stateauglong] [--cols a:b --tag NAME]
       -> results/sic/twoscale/layout_s{SEED}.pt and fiber_s{SEED}.pt (base), fiber_s{SEED}_{long,wide,guided}.pt,
          or layout_{gray,state}_s{SEED}.pt and fiber_{gray,state}_s{SEED}.pt
"""
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genmicro import sic  # noqa: E402
from genmicro.net import UNET, FlowUNet  # noqa: E402
from genmicro.train import train  # noqa: E402
from genmicro.twoscale import (DOWN, LAYOUT_DROPOUT, ImageConditioned, flips, fraction_fields, gray_layout,  # noqa: E402
                               COND_NOISE_MAX, noisy_condition, state_fields, strip, unwrap, upsample)

OUT = ROOT / "results/sic/twoscale"
BAND = 2640  # columns of the training band
FIBER = {"base": (dict(UNET, image_cond=2), 8000), "wide": (dict(UNET, widths=(96, 192, 384, 384), image_cond=2), 12000),
         "guided": (dict(UNET, image_cond=3), 12000), "long": (dict(UNET, image_cond=2), 16000),
         "gray": (dict(UNET, image_cond=1), 8000), "state": (dict(UNET, image_cond=2), 8000),
         "stateaug": (dict(UNET, image_cond=3), 8000), "stateaug05": (dict(UNET, image_cond=3), 8000),
         "stateauglong": (dict(UNET, image_cond=3), 16000), "stateblur": (dict(UNET, image_cond=3), 8000),
         "stateaug256": (dict(UNET, image_cond=3), 8000), "stateaugshort": (dict(UNET, image_cond=3), 4000),
         "stateaug2k": (dict(UNET, image_cond=3), 2000), "stateaug6k": (dict(UNET, image_cond=3), 6000),
         "stateaug1k": (dict(UNET, image_cond=3), 1000), "stateaug3k": (dict(UNET, image_cond=3), 3000)}
CROP = {"stateaug256": 256}  # fiber-model crop size (default 128 px); the batch keeps the pixels per step
NOISE_MAX = {"stateaug": COND_NOISE_MAX, "stateaug05": 0.5, "stateauglong": COND_NOISE_MAX, "stateblur": 0.5,
             "stateaug256": COND_NOISE_MAX, "stateaugshort": COND_NOISE_MAX, "stateaug2k": COND_NOISE_MAX,
             "stateaug6k": COND_NOISE_MAX, "stateaug1k": COND_NOISE_MAX, "stateaug3k": COND_NOISE_MAX}  # ceiling of the layout noise
COND_BLUR = {"stateblur": 3.0}  # Gaussian blur (layout pixels) of the noisy layout


def train_layout(lay, path, g, seed, steps=6000):
    """lay: (C, h, w) layout of the training band."""
    cfg = dict(UNET, channels=lay.shape[0])
    h, w = lay.shape[-2:]

    def batch(n, step):
        ys = torch.randint(0, h - 63, (n,), device="cuda", generator=g)
        xs = torch.randint(0, w - 63, (n,), device="cuda", generator=g)
        crops = torch.stack([lay[:, y:y + 64, x:x + 64] for y, x in zip(ys.tolist(), xs.tolist())])
        return flips(crops, g), None

    ema, _ = train(FlowUNet(**cfg).cuda().to(memory_format=torch.channels_last), batch, steps, batch=64,
                   seed=seed)
    torch.save(dict(model_config=cfg, ema=strip(ema.state_dict())), path)


def train_fiber(band, variant, path, g, seed):
    cfg, steps = FIBER[variant]
    H, W = band.shape
    low = state_fields(band)[0] if variant.startswith("state") else fraction_fields(band)[0]
    lay = None if variant == "gray" else upsample(low[None])[0]

    C = CROP.get(variant, 128)

    def batch(n, step):
        ys = torch.randint(0, (H - C) // DOWN, (n,), device="cuda", generator=g) * DOWN
        xs = torch.randint(0, (W - C) // DOWN, (n,), device="cuda", generator=g) * DOWN
        x = torch.stack([band[y:y + C, x_:x_ + C] for y, x_ in zip(ys.tolist(), xs.tolist())])[:, None]
        if variant == "gray":
            return x, gray_layout(x)
        if variant.startswith("stateaug") or variant == "stateblur":  # includes stateaug256 and stateauglong
            lo = torch.stack([low[:, y // DOWN:(y + C) // DOWN, x_ // DOWN:(x_ + C) // DOWN]
                              for y, x_ in zip(ys.tolist(), xs.tolist())])
            level = torch.rand(n, device="cuda", generator=g) * NOISE_MAX[variant]
            return x, noisy_condition(lo, level, g, COND_BLUR.get(variant, 0.0))
        c = torch.stack([lay[:, y:y + 128, x_:x_ + 128] for y, x_ in zip(ys.tolist(), xs.tolist())])
        if variant == "guided":
            keep = (torch.rand(n, 1, 1, 1, device="cuda", generator=g) > LAYOUT_DROPOUT).float()
            c = torch.cat([c * keep, keep.expand(-1, 1, 128, 128)], 1)
        return x, c

    ema, _ = train(ImageConditioned(FlowUNet(**cfg)).cuda().to(memory_format=torch.channels_last), batch, steps,
                   batch=32 * 128 * 128 // (C * C), seed=seed)
    torch.save(dict(model_config=cfg, ema=unwrap(ema.state_dict()), steps=steps), path)


def main(seed, variant="base", cols=None, tag=None):
    torch.backends.cudnn.benchmark = True
    OUT.mkdir(parents=True, exist_ok=True)
    if cols is not None:  # fiber model alone, on a column range of the training band
        c0, c1 = map(int, cols.split(":"))
        band = sic.load_plane()[:, c0:c1]
        g = torch.Generator(device="cuda").manual_seed(300 + seed)
        fpath = OUT / f"fiber_{variant}_{tag}_s{seed}.pt"
        t0 = time.perf_counter()
        if not fpath.exists():
            train_fiber(band, variant, fpath, g, 10 * seed + 5)
        print(json.dumps(dict(seed=seed, variant=variant, cols=cols, tag=tag, fiber_seconds=round(time.perf_counter() - t0))))
        return
    band = sic.load_plane()[:, :BAND]
    g = torch.Generator(device="cuda").manual_seed(200 + seed)
    gray = variant == "gray"
    lname = "state_" if variant.startswith("state") else {"gray": "gray_"}.get(variant, "")  # stateaug*: the state layout model
    lpath = OUT / f"layout_{lname}s{seed}.pt"
    fpath = OUT / {"base": f"fiber_s{seed}.pt", "gray": f"fiber_gray_s{seed}.pt", "state": f"fiber_state_s{seed}.pt",
                   }.get(variant, f"fiber_{variant}_s{seed}.pt" if variant.startswith("state") else f"fiber_s{seed}_{variant}.pt")
    t0 = time.perf_counter()
    if not lpath.exists():
        lay = (F.avg_pool2d(band[None, None], DOWN)[0] if gray else state_fields(band)[0] if variant.startswith("state")
               else fraction_fields(band)[0])
        train_layout(lay, lpath, g, 10 * seed + 3)
    t1 = time.perf_counter()
    if not fpath.exists():
        train_fiber(band, variant, fpath, g, 10 * seed + 4)
    print(json.dumps(dict(seed=seed, variant=variant, layout_seconds=round(t1 - t0), fiber_seconds=round(time.perf_counter() - t1))))


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.45)
    args = sys.argv[1:]
    opt = {k: args[args.index(k) + 1] for k in ("--cols", "--tag") if k in args}
    pos = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or not args[i - 1].startswith("--"))]
    main(int(pos[0]), pos[1] if len(pos) > 1 else "base", opt.get("--cols"), opt.get("--tag"))
