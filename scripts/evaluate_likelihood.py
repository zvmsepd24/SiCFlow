"""Likelihood of training and held-out windows under the flow models, which are continuous normalizing flows.

A flow-matching model defines the ODE dx/dt = v(x, t) from noise (t = 0) to data (t = 1); the log-density of a window x
follows from the instantaneous change of variables (Chen et al. 2018; Grathwohl et al. 2019):
    log p_1(x) = log N(x_0; 0, I) - integral_0^1 div v(x_t, t) dt,
integrated backwards from x_1 = x with Heun steps (grid concentrated near the data, rho = 2), the divergence from one
Rademacher probe per evaluation (Hutchinson), with the same probes for every model. Reported in bits per dimension of
8-bit gray levels (a continuous density on [-1, 1] at 1/127.5 per gray level).
For each paired model (checkerboard, every area and replicate), 256 windows of its own training tiles and the same
256 held-out windows; also the 64-pixel model of the whole band. A model that copies gives its training windows a much
higher likelihood than held-out windows; the gap closes as it generalizes.
Usage: python scripts/evaluate_likelihood.py REPLICATE [...] -> results/sic/convergence/likelihood_rep{R}.json
       python scripts/evaluate_likelihood.py single64      -> results/sic/convergence/likelihood_single64.json
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genmicro import sic  # noqa: E402
from genmicro.checkerboard import COLS, LEVELS, MM2_PER_TILE, T, chosen_tiles  # noqa: E402
from genmicro.scores import load_model  # noqa: E402
from genmicro.train import time_grid  # noqa: E402
from scripts.evaluate_checkerboard import real_windows  # noqa: E402

CK = ROOT / "results/sic/checkerboard"
OUT = ROOT / "results/sic/convergence"
N, STEPS, RHO, BATCH = 256, 64, 2.0, 128
D = 64 * 64


def bits_per_dim(model, x, seed=0):
    """x (B, 1, 64, 64) in [-1, 1] -> bits per dimension of each window."""
    model.requires_grad_(False)
    g = torch.Generator(device="cuda").manual_seed(seed)
    ts = time_grid(STEPS, RHO, x.device).flip(0)  # 1 -> 0

    def f(z, t):
        with torch.autocast("cuda", torch.bfloat16):
            return model(z, t.expand(len(z))).float()

    def vel_div(z, t, e):
        v, jv = torch.func.jvp(lambda q: f(q, t), (z,), (e,))
        return v, (e * jv).flatten(1).sum(1)

    z = x.float()
    div_int = torch.zeros(len(x), device=x.device)
    for i in range(STEPS):
        t, dt = ts[i], ts[i + 1] - ts[i]  # dt < 0
        e = torch.randint(0, 2, z.shape, device=z.device, generator=g).float() * 2 - 1
        v1, d1 = vel_div(z, t, e)
        ze = z + dt * v1
        v2, d2 = vel_div(ze, ts[i + 1], e)
        z = z + dt / 2 * (v1 + v2)
        div_int = div_int + dt / 2 * (d1 + d2)  # integral from 1 to 0 of div = -(integral from 0 to 1)
    logp0 = -0.5 * (z.flatten(1).square().sum(1) + D * math.log(2 * math.pi))
    logp1 = logp0 + div_int  # log p_1(x) = log p_0(x_0) - int_0^1 div dt = log p_0 + int_1^0 div dt
    nats = -logp1 / D + math.log(127.5)  # density per gray level of 8-bit data, 1/127.5 wide in [-1, 1]
    return (nats / math.log(2)).cpu().numpy()


def evaluate(model, x, seed=0):
    return np.concatenate([bits_per_dim(model, x[i:i + BATCH], seed + i) for i in range(0, len(x), BATCH)])


def main(args):
    torch.backends.cudnn.benchmark = True
    plane_t = sic.load_plane()
    plane = plane_t.cpu().numpy()
    held = torch.as_tensor(real_windows(plane, [(0, 2768, plane.shape[0], plane.shape[1] - 2768)], N, 1)[:, None]).cuda()
    for a in args:
        if a == "single64":
            m = load_model(ROOT / "results/sic/single/single64.pt")
            tr = torch.as_tensor(real_windows(plane, [(0, 0, plane.shape[0], 2640)], N, 31)[:, None]).cuda()
            res = dict(training=evaluate(m, tr).tolist(), held_out=evaluate(m, held).tolist())
            (OUT / "likelihood_single64.json").write_text(json.dumps(res))
            print("single64", np.mean(res["training"]), np.mean(res["held_out"]), flush=True)
            continue
        rep = int(a)
        rows = []
        for lev, n_pairs in enumerate(LEVELS):
            tile_idx = chosen_tiles(rep, lev)
            row = dict(replicate=rep, level=lev, area_mm2=n_pairs * MM2_PER_TILE)
            for s in "AB":
                p = CK / f"rep{rep}_level{lev}_{s}.pt"
                if not p.exists():
                    continue
                boxes = [((t // COLS) * T, (t % COLS) * T, T, T) for t in tile_idx[s]]
                tr = torch.as_tensor(real_windows(plane, boxes, N, 20 + lev + (s == "B"))[:, None]).cuda()
                m = load_model(p)
                row[s] = dict(training=evaluate(m, tr).tolist(), held_out=evaluate(m, held).tolist())
                print(rep, lev, s, f"train {np.mean(row[s]['training']):.3f} held {np.mean(row[s]['held_out']):.3f}", flush=True)
            rows.append(row)
            (OUT / f"likelihood_rep{rep}.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.45)
    main(sys.argv[1:])
