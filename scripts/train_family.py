"""Paired models of other families (genmicro.families: ddpm, vae, glow) on the checkerboard split, as the flow models.

For one replicate: models of set A, of set B and of set A with another seed (A2, the reference for agreement),
trained on 64 x 64 px crops inside their tiles with the curriculum of scripts/train_checkerboard.py (2000 steps on
the smallest area, 1500 more on each larger area, continuing; 128 crops per step; the learning rate warms up and
decays at each area; moving average restarted at each area). DDPM uses the U-Net and optimizers of the flow
models; the VAE and Glow use AdamW (learning rate 1e-3 and 5e-4) and Glow clips the gradient norm at 50.
Usage: python scripts/train_family.py FAMILY REPLICATE [SETS]   SETS default A,B,A2
       -> results/sic/families/{FAMILY}_rep{R}_level{L}_{A,B,A2}.pt
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
from genmicro.families import build  # noqa: E402
from genmicro.train import train  # noqa: E402

OUT = ROOT / "results/sic/families"
LR = dict(ddpm=2e-3, vae=1e-3, glow=5e-4)
CLIP = dict(glow=50.0)


def adamw(model, lr):
    return [torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.0)]


def strip(state):
    return {k.replace("_orig_mod.", ""): v for k, v in state.items()}


def main(family, rep, sets):
    torch.backends.cudnn.benchmark = True
    OUT.mkdir(parents=True, exist_ok=True)
    tiles = tile_grid(sic.load_plane())
    for s in sets:
        seed = 100 * rep + {"A": 0, "B": 1, "A2": 2}[s]
        g = torch.Generator(device="cuda").manual_seed(7000 + seed)
        torch.manual_seed(seed)
        model = build(family).cuda()
        tile_set = "A" if s == "A2" else s
        if family == "glow":  # data-dependent initialization of the actnorm layers
            with torch.no_grad():
                model.loss(crops_from(tiles[chosen_tiles(rep, 0)[tile_set]], 512, g))
        n_par = sum(p.numel() for p in model.parameters())
        # DDPM: the U-Net compiled as for the flow models (the compiled module shares the weights)
        net = torch.compile(model.net) if family == "ddpm" else None
        loss_fn = (lambda m, x, c, gen: m.loss(x, gen, net)) if net is not None else (lambda m, x, c, gen: m.loss(x, gen))
        for lev, steps in enumerate(STEPS):
            out = OUT / f"{family}_rep{rep}_level{lev}_{s}.pt"
            set_tiles = tiles[chosen_tiles(rep, lev)[tile_set]]
            t0 = time.perf_counter()
            ema, info = train(model, lambda n, step: (crops_from(set_tiles, n, g), None), steps,
                              lr=LR[family], seed=seed + 10 * lev, compile=False,
                              loss_fn=loss_fn,
                              optimizers=None if family == "ddpm" else adamw, clip=CLIP.get(family))
            torch.save(dict(family=family, ema=strip(ema.state_dict()), seconds=time.perf_counter() - t0, parameters=n_par,
                            losses=info["loss_every_100_steps"]), out)
            print(json.dumps(dict(family=family, rep=rep, level=lev, set=s, pairs=LEVELS[lev], parameters=n_par,
                                  seconds=round(time.perf_counter() - t0), loss=info["loss_every_100_steps"][-1])), flush=True)


def load(path):
    ck = torch.load(path, map_location="cuda")
    model = build(ck["family"]).cuda()
    model.load_state_dict(ck["ema"])
    return model.eval()


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.8)
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3].split(",") if len(sys.argv) > 3 else ["A", "B", "A2"])
