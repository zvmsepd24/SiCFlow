"""Evaluation of any model on 64 x 64 px windows, with the protocol of the checkerboard evaluation
(scripts/evaluate_checkerboard.py), so that models of the whole training band and the paired checkerboard
models can be put on the same axes.

Each model generates 1024 windows from the noise that replicate 0 of the checkerboard evaluation uses
(seed 9000, or 9000 + N with the suffix nN), so windows of any two models can also be compared one to one. Distances to 1024
held-out windows (seed 1): FID, Fréchet distances of two-point and chord vectors, Wasserstein-1 distances of
pore fraction and fiber diameter. References: a second set of held-out windows (the floor) and real
windows of the whole training band. Also the Levina-Bickel dimension of the pixels (512 windows, k = 10),
copying (largest normalized cross-correlation of 256 windows with any window inside the 360 tiles of the
checkerboard, which cover 94% of the training band) and the sampling time.
Usage: python scripts/evaluate_windows.py NAME[@STEPS[rRHO][nN]] ...   (STEPS: Heun steps, default 32; RHO: exponent
       of the time grid t_i = 1 - (1 - i/STEPS)^RHO, default 1, uniform steps, the sampler of the paper; RHO = 0.7
       puts short steps near the noise; N: another noise set)
       NAME is a model of results/sic/single (single64, single64_seed1, single64_wide, single64_steps12000) or
       A0-A4 / B0-B4 (checkerboard replicate 0, training areas 0-4), A4scratch / B4scratch (largest area, from scratch)
       -> results/sic/windows/eval_{NAME}[@...].json, windows_{NAME}[@...].npz
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genmicro import sic  # noqa: E402
from genmicro.checkerboard import K, match_tiles, tile_grid  # noqa: E402
from genmicro.lid import levina_bickel  # noqa: E402
from genmicro.scores import load_model  # noqa: E402
from genmicro.train import RHO, STEPS, sample  # noqa: E402
from scripts.evaluate_checkerboard import N_WIN, describe, distances, real_windows  # noqa: E402

OUT = ROOT / "results/sic/windows"
BAND, HELD = 2640, 2768
NAMED = {}
for lev in range(5):
    for s in "AB":
        NAMED[f"{s}{lev}"] = f"results/sic/checkerboard/rep0_level{lev}_{s}.pt"
        NAMED[f"{s}4scratch"] = f"results/sic/checkerboard/rep0_level4_{s}_scratch.pt"  # trained from scratch, 8000 steps


def tag_of(name, steps=STEPS, rho=RHO, noise=0):
    """File tag of an evaluation: the model name, plus @STEPS[rRHO][nN] where the sampler or noise set differ."""
    if (steps, rho, noise) == (STEPS, RHO, 0):
        return name
    return f"{name}@{steps}" + (f"r{rho:g}" if rho != RHO else "") + (f"n{noise}" if noise else "")


def checkpoint(name):
    return NAMED.get(name, f"results/sic/single/{name}.pt")


@torch.no_grad()
def generate(model, z, steps=STEPS, rho=RHO):
    out = []
    for i in range(0, len(z), 256):
        with torch.autocast("cuda", torch.bfloat16):
            out.append(sample(model, z[i:i + 256], steps, rho=rho).float())
    return torch.cat(out)[:, 0].cpu().numpy()


def references(plane):
    path = OUT / "references.json"
    held_box = [(0, HELD, plane.shape[0], plane.shape[1] - HELD)]
    held1 = describe(real_windows(plane, held_box, N_WIN, 1))
    if not path.exists():
        held2 = describe(real_windows(plane, held_box, N_WIN, 2))
        train = describe(real_windows(plane, [(0, 0, plane.shape[0], BAND)], N_WIN, 3))
        idx = np.random.default_rng(5).choice(N_WIN, 512, replace=False)
        path.write_text(json.dumps(dict(floor=distances(held1, held2), training_band=distances(train, held1),
                                        dim_held=float(np.nanmean(levina_bickel(held1["pixels"][idx], k=10))))))
    return held1


def main(names):
    OUT.mkdir(parents=True, exist_ok=True)
    plane_t = sic.load_plane()
    plane = plane_t.cpu().numpy()
    held1 = references(plane)
    tiles = tile_grid(plane_t)
    z = torch.randn(N_WIN, 1, K, K, device="cuda", generator=torch.Generator(device="cuda").manual_seed(9000))
    for spec in names:
        name, grid = (spec.split("@") + [str(STEPS)])[:2]
        m = re.fullmatch(r"(\d+)(?:r([\d.]+))?(?:n(\d+))?", grid)
        steps, rho, noise = int(m[1]), float(m[2] or RHO), int(m[3] or 0)
        zz = z if noise == 0 else torch.randn(N_WIN, 1, K, K, device="cuda",
                                              generator=torch.Generator(device="cuda").manual_seed(9000 + noise))
        model = load_model(checkpoint(name))
        generate(model, zz[:256], steps, rho)  # warm-up
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gray = generate(model, zz, steps, rho)
        sec = time.perf_counter() - t0
        gen = describe(gray)
        idx = np.random.default_rng(3).choice(N_WIN, 512, replace=False)
        sim = match_tiles(torch.as_tensor(gray[:256, None]).cuda(), tiles)[0].cpu().numpy()
        row = dict(name=name, heun_steps=steps, rho=rho, noise=noise, fidelity=distances(gen, held1),
                   dim_pixels=float(np.nanmean(levina_bickel(gen["pixels"][idx], k=10))),
                   copy_median=float(np.median(sim)), copies=float((sim > 0.9).mean()), sample_seconds_1024=sec,
                   fiber_median=float(np.median(gen["diameter"])), pore_mean=float(gen["pore"].mean()),
                   pore_mean_held=float(held1["pore"].mean()), fiber_median_held=float(np.median(held1["diameter"])))
        tag = tag_of(name, steps, rho, noise)
        (OUT / f"eval_{tag}.json").write_text(json.dumps(row))
        np.savez_compressed(OUT / f"windows_{tag}.npz", gray=gray.astype(np.float16))
        print(json.dumps(dict(name=tag, **{k: round(v, 4) for k, v in row["fidelity"].items()},
                              dim=round(row["dim_pixels"], 1), copy=round(row["copy_median"], 3), sec=round(sec, 1))), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.3)
    main(sys.argv[1:])
