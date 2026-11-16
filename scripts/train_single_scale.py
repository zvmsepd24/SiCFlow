"""Single-scale models of the whole training band, trained on crops of 64, 128 or 256 px.

All use the same network and 8000 steps; the batch is scaled so that every step contains the same number of
pixels (64 px: 128 crops, 128 px: 32 crops, 256 px: 8 crops).
A model of 512 px crops, large enough to contain the tow bands, uses a deeper network so that its receptive field
spans the crop: six levels (64, 128, 256, 256, 256, 256 channels; 1, 2, 2, 2, 2, 4 blocks), 4 crops per step.
Checks of the 64 px model change one thing at a time: the training seed, 1.5 times the channels of every level
(96, 192, 384, 384), or 1.5 times the training steps (12000). For these runs the flow-matching loss of the EMA
weights is also recorded every 5% of training on fixed held-out crops (1024 crops of the held-out band, fixed
noise, eight fixed flow times), with the training time.
Usage: python scripts/train_single_scale.py CROP                 -> results/sic/single/single{CROP}.pt (paper seeds)
       python scripts/train_single_scale.py 64 --seed 1           -> single64_seed1.pt and .json
       python scripts/train_single_scale.py 64 --wide             -> single64_wide.pt and .json
       python scripts/train_single_scale.py 64 --steps 12000      -> single64_steps12000.pt and .json
       python scripts/train_single_scale.py 512                   -> single512.pt (deeper network)
"""
import argparse
import json
import time
import sys
from pathlib import Path

import torch
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genmicro import sic  # noqa: E402
from genmicro.net import UNET, FlowUNet  # noqa: E402
from genmicro.train import train  # noqa: E402
from genmicro.twoscale import strip  # noqa: E402

OUT = ROOT / "results/sic/single"
BAND, HELD = 2640, 2768  # training band: columns 0-2639; held-out band: columns 2768 to the end
SEEDS = {64: 64, 128: 0, 256: 256, 512: 512}  # seeds of the models in the paper
WIDE = dict(UNET, widths=(96, 192, 384, 384))
DEEP = dict(UNET, widths=(64, 128, 256, 256, 256, 256), depths=(1, 2, 2, 2, 2, 4))  # for 512 px crops


def held_out_loss_fn(plane, crop, seed=123, n=1024):
    """Flow-matching loss on fixed held-out crops, at eight flow times spread like the training times."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x1 = sic.CropSampler(plane[:, HELD:], crop, generator=g)(n).contiguous(memory_format=torch.channels_last)
    x0 = torch.randn(x1.shape, device="cuda", generator=g)
    ts = torch.sigmoid(torch.as_tensor(norm.ppf((torch.arange(8) + 0.5) / 8), dtype=torch.float32)).cuda()

    @torch.no_grad()
    def loss(model):
        out = []
        for t in ts:
            tot = 0.0
            for i in range(0, n, 256):
                a, b = x0[i:i + 256], x1[i:i + 256]
                with torch.autocast("cuda", torch.bfloat16):
                    v = model((1 - t) * a + t * b, t.expand(len(a)))
                tot += float(((v.float() - (b - a)) ** 2).sum())
            out.append(tot / x1.numel())
        return out
    return loss, ts.tolist()


def main(crop, seed=None, steps=8000, wide=False):
    torch.backends.cudnn.benchmark = True
    OUT.mkdir(parents=True, exist_ok=True)
    check = (seed is not None or steps != 8000 or wide) and crop < 512  # held-out loss only for the checks of small crops
    seed = SEEDS[crop] if seed is None else seed
    name = f"single{crop}" + (f"_seed{seed}" if seed != SEEDS[crop] else "") + ("_wide" if wide else "") + \
        (f"_steps{steps}" if steps != 8000 else "")
    plane = sic.load_plane()
    batch = max(4, 32 * 128 * 128 // crop ** 2)
    crops = sic.CropSampler(plane[:, :BAND], crop, generator=torch.Generator(device="cuda").manual_seed(seed))
    cfg = WIDE if wide else DEEP if crop >= 512 else UNET
    model = FlowUNet(**cfg).cuda().to(memory_format=torch.channels_last)
    log, clock, cb = [], dict(train=0.0, last=None), None
    if check:
        loss_fn, ts = held_out_loss_fn(plane, crop)

        def cb(step, ema):
            torch.cuda.synchronize()
            clock["train"] += time.perf_counter() - clock["last"]
            log.append(dict(step=step, seconds=round(clock["train"], 1), held_out=loss_fn(ema)))
            print(json.dumps(dict(name=name, step=step, seconds=log[-1]["seconds"],
                                  held_out=round(sum(log[-1]["held_out"]) / 8, 5))), flush=True)
            torch.cuda.synchronize()
            clock["last"] = time.perf_counter()
    t0 = clock["last"] = time.perf_counter()
    ema, info = train(model, lambda n, s: (crops(n).contiguous(memory_format=torch.channels_last), None), steps,
                      batch=batch, seed=seed, callback=cb, every=max(100, steps // 20))
    seconds = time.perf_counter() - t0
    torch.save(dict(model_config=cfg, ema=strip(ema.state_dict()), crop=crop, batch=batch, steps=steps, seed=seed,
                    seconds=seconds), OUT / f"{name}.pt")
    if check:
        (OUT / f"{name}.json").write_text(json.dumps(dict(
            name=name, seed=seed, steps=steps, params=sum(p.numel() for p in ema.parameters()), flow_times=ts, log=log,
            config={k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()},
            train_loss_every_100=info["loss_every_100_steps"])))
    print(json.dumps(dict(name=name, crop=crop, batch=batch, steps=steps, seconds=round(seconds))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("crop", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--wide", action="store_true")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.85)  # one training at a time on a 12 GB GPU
    main(a.crop, a.seed, a.steps, a.wide)
