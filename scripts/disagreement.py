"""Per-sample agreement of two models that start from the same noise, and the noise samples on which they
disagree most.

For each pair of models, 2048 windows of 64 x 64 px are generated from the same noise (seed 9100, not the
noise of the evaluation). Agreement of one pair of windows: correlation of their pixels after removing each
window's mean (genmicro.checkerboard.agreement, one window at a time). The pore fraction of every window
(genmicro.phases) is kept too, to see which features the models disagree on. Pairs: models A and B of replicate 0
of the checkerboard at each training area, and pairs of models of the whole training band.
Usage: python scripts/disagreement.py [PAIR ...]  (default: the checkerboard pairs, levels 0-4)
       PAIR is NAME1,NAME2 with names as in scripts/evaluate_windows.py (e.g. single64_seed1,single64)
       -> results/sic/windows/disagreement.json (per-sample agreement) and disagreement_windows.npz
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genmicro.checkerboard import K  # noqa: E402
from genmicro.phases import phases  # noqa: E402
from genmicro.scores import load_model  # noqa: E402
from scripts.evaluate_windows import checkpoint, generate  # noqa: E402

OUT = ROOT / "results/sic/windows"
N, KEEP = 2048, 12


def per_sample(a, b):
    a = a.reshape(len(a), -1).astype(np.float64)
    b = b.reshape(len(b), -1).astype(np.float64)
    a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)


def main(pairs):
    OUT.mkdir(parents=True, exist_ok=True)
    z = torch.randn(N, 1, K, K, device="cuda", generator=torch.Generator(device="cuda").manual_seed(9100))
    jpath, npath = OUT / "disagreement.json", OUT / "disagreement_windows.npz"
    res = json.loads(jpath.read_text()) if jpath.exists() else {}
    win = dict(np.load(npath)) if npath.exists() else {}
    for p in pairs:
        n1, n2 = p.split(",")
        g1 = generate(load_model(checkpoint(n1)), z)
        g2 = generate(load_model(checkpoint(n2)), z)
        r = per_sample(g1, g2)
        order = np.argsort(r)
        keep = np.concatenate([order[:KEEP], order[len(order) // 2 - KEEP // 2:len(order) // 2 + KEEP // 2], order[-KEEP:]])
        pores = [[float((phases(g) == 2).mean()) for g in gg] for gg in (g1, g2)]
        res[p] = dict(agreement=r.tolist(), mean=float(r.mean()), kept=keep.tolist(), pore1=pores[0], pore2=pores[1])
        win[f"{p}|1"], win[f"{p}|2"] = g1[keep].astype(np.float16), g2[keep].astype(np.float16)
        win[f"{p}|z"] = z[keep, 0].cpu().numpy().astype(np.float16)
        print(json.dumps(dict(pair=p, mean=round(float(r.mean()), 3), worst=[round(float(v), 3) for v in r[order[:5]]],
                              frac_below_half=round(float((r < 0.5).mean()), 3))), flush=True)
        jpath.write_text(json.dumps(res))
        np.savez_compressed(npath, **win)


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.25)
    main(sys.argv[1:] or [f"A{lev},B{lev}" for lev in range(5)])
