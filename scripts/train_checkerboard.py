"""Paired models trained on the checkerboard split of the training band (genmicro.checkerboard).

For one replicate, a model of set A and a model of set B are trained on 64 x 64 px crops lying inside
their tiles: 2000 steps on the smallest area, then 1500 more steps on each larger area, continuing from
the previous model. The checkpoint after each area is kept.
Control (--scratch): models A and B trained from scratch on the largest area only, for 8000 steps, like the
64-pixel model of the whole training band, so the two can be compared at equal training.
Width check (--widths w1,w2,w3,w4 --tag TAG): the same curriculum with another number of channels per level.
Usage: python scripts/train_checkerboard.py REPLICATE -> results/sic/checkerboard/rep{R}_level{L}_{A,B}.pt
       python scripts/train_checkerboard.py REPLICATE --scratch -> results/sic/checkerboard/rep{R}_level4_{A,B}_scratch.pt
       python scripts/train_checkerboard.py REPLICATE --widths 32,64,128,128 --tag w05
                                                      -> results/sic/checkerboard/rep{R}_w05_level{L}_{A,B}.pt
"""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genmicro import sic  # noqa: E402
from genmicro.checkerboard import LEVELS, STEPS, chosen_tiles, crops_from, tile_grid  # noqa: E402
from genmicro.net import UNET, FlowUNet  # noqa: E402
from genmicro.train import train  # noqa: E402
from genmicro.twoscale import strip  # noqa: E402

OUT = ROOT / "results/sic/checkerboard"


def main(rep, widths=None, tag=None):
    torch.backends.cudnn.benchmark = True
    OUT.mkdir(parents=True, exist_ok=True)
    tiles = tile_grid(sic.load_plane())
    g = torch.Generator(device="cuda").manual_seed(7000 + rep)
    cfg = UNET if widths is None else dict(UNET, widths=tuple(widths))
    prefix = f"rep{rep}" if tag is None else f"rep{rep}_{tag}"
    models = {s: FlowUNet(**cfg).cuda().to(memory_format=torch.channels_last) for s in "AB"}
    for lev, steps in enumerate(STEPS):
        idx = chosen_tiles(rep, lev)
        for s in "AB":
            set_tiles = tiles[idx[s]]
            t0 = time.perf_counter()
            ema, _ = train(models[s], lambda n, step: (crops_from(set_tiles, n, g).contiguous(memory_format=torch.channels_last), None),
                           steps, seed=100 * rep + 10 * lev + (s == "B"))
            torch.save(dict(model_config=cfg, ema=strip(ema.state_dict()), seconds=time.perf_counter() - t0),
                       OUT / f"{prefix}_level{lev}_{s}.pt")
        print(json.dumps(dict(rep=rep, level=lev, pairs=LEVELS[lev])), flush=True)


def scratch(rep, steps=8000):
    torch.backends.cudnn.benchmark = True
    tiles = tile_grid(sic.load_plane())
    lev = len(STEPS) - 1
    idx = chosen_tiles(rep, lev)
    for s in "AB":
        g = torch.Generator(device="cuda").manual_seed(7500 + rep + (s == "B"))
        set_tiles = tiles[idx[s]]
        model = FlowUNet(**UNET).cuda().to(memory_format=torch.channels_last)
        t0 = time.perf_counter()
        ema, _ = train(model, lambda n, step: (crops_from(set_tiles, n, g).contiguous(memory_format=torch.channels_last), None),
                       steps, seed=500 + 10 * rep + (s == "B"))
        torch.save(dict(model_config=UNET, ema=strip(ema.state_dict()), steps=steps, seconds=time.perf_counter() - t0),
                   OUT / f"rep{rep}_level{lev}_{s}_scratch.pt")
        print(json.dumps(dict(rep=rep, level=lev, set=s, steps=steps, scratch=True)), flush=True)


if __name__ == "__main__":
    if "--scratch" in sys.argv:
        torch.cuda.set_per_process_memory_fraction(0.85)
        scratch(int(sys.argv[1]))
    else:
        a = sys.argv[1:]
        torch.cuda.set_per_process_memory_fraction(0.7 if "--widths" in a else 0.3)  # wider networks need more
        w = [int(v) for v in a[a.index("--widths") + 1].split(",")] if "--widths" in a else None
        main(int(a[0]), w, a[a.index("--tag") + 1] if "--tag" in a else None)
