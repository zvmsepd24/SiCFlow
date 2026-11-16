"""Paired models scored against another section of the specimen as well as against the held-out band.

The held-out band of the paper comes from the training section itself, so it shares the tows of the training
tiles. Section 99 of the series lies 89 sections deeper: no fiber of the training section is found again, and its
tow layout correlates only in part with that of the training section (scripts/compare_sections.py). Its in-focus
right half (columns 1700-3469 of the frame of the training section) gives an independent test set.

For every replicate and training area (scripts/train_checkerboard.py), models A and B generate 1024 windows of
64 px from the same noise as in scripts/evaluate_checkerboard.py. Real windows: two sets from the held-out band
and two sets from section 99. Distances (FID, two-point, chords, pore fraction, fiber diameter, as in
scripts/evaluate_checkerboard.py) of each model to both test sets, and between the real sets.
Copying: the largest normalized cross-correlation of 256 generated windows, 256 held-out windows and 256 windows
of section 99 with any window inside the model's training tiles. A generated window copies when it is more
similar to the training tiles than every held-out window (the criterion of the paper) or than every window of
section 99.
Real windows alone (--real): the distances between the real sets, and between the held-out windows and the same
windows blurred to the sharpness of section 99 (Gaussian of 0.4 px, scripts/evaluate_sections.py): the cost of the
blur alone; and the Levina-Bickel dimension of each real set (as in scripts/evaluate_checkerboard.py).
Usage: python scripts/evaluate_independent.py REPLICATE [REPLICATE ...]
       python scripts/evaluate_independent.py --real -> results/sic/sections/independent_real.json
       options: --s99cols a:b (columns of section 99 used as test data; default 1700:3470), --name FILE (output name)
   -> results/sic/sections/independent_rep{R}.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genmicro import sic  # noqa: E402
from genmicro.checkerboard import COLS, LEVELS, MM2_PER_TILE, K, T, agreement, chosen_tiles, match_tiles, tile_grid  # noqa: E402
from genmicro.scores import load_model  # noqa: E402
from scripts.evaluate_checkerboard import N_WIN, describe, dimension, distances, generate, real_windows  # noqa: E402

CK = ROOT / "results/sic/checkerboard"
OUT = ROOT / "results/sic/sections"
HELD, RIGHT, END99 = 2768, 1700, 3470


def real_only(sigma=0.4):
    import cv2
    plane = sic.load_plane().cpu().numpy()
    s99 = sic.load_section(99).cpu().numpy()
    hbox = [(0, HELD, plane.shape[0], plane.shape[1] - HELD)]
    sbox = [(0, RIGHT, s99.shape[0], END99 - RIGHT)]
    h1, h2 = real_windows(plane, hbox, N_WIN, 1), real_windows(plane, hbox, N_WIN, 2)
    blurred = np.clip(cv2.GaussianBlur(((plane + 1) * 127.5).astype(np.float32), (0, 0), sigma), 0, 255) / 127.5 - 1
    b1 = real_windows(blurred, hbox, N_WIN, 1)  # the same positions as h1
    sa, sb = real_windows(s99, sbox, N_WIN, 21), real_windows(s99, sbox, N_WIN, 22)
    d = {k: describe(v) for k, v in dict(h1=h1, h2=h2, b1=b1, sa=sa, sb=sb).items()}
    out = dict(floor_held=distances(d["h1"], d["h2"]), floor_s99=distances(d["sa"], d["sb"]),
               held_vs_s99=distances(d["h1"], d["sa"]), held2_vs_s99b=distances(d["h2"], d["sb"]),
               held_vs_blurred=distances(d["h2"], d["b1"]), blur_sigma_px=sigma,
               dimension={k: dimension(d[k], 5) for k in ("h1", "h2", "b1", "sa", "sb")})
    (OUT / "independent_real.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


def copying(sims, ref):
    return float(np.mean(np.asarray(sims) > np.max(ref)))


def main(rep, cols=None, name=None):
    right, end99 = cols or (RIGHT, END99)
    OUT.mkdir(parents=True, exist_ok=True)
    prefix = f"rep{rep}"
    plane_t = sic.load_plane()
    plane = plane_t.cpu().numpy()
    s99 = sic.load_section(99).cpu().numpy()
    hbox = [(0, HELD, plane.shape[0], plane.shape[1] - HELD)]
    sbox = [(0, right, s99.shape[0], end99 - right)]
    gray = {"held1": real_windows(plane, hbox, N_WIN, 1), "held2": real_windows(plane, hbox, N_WIN, 2),
            "s99a": real_windows(s99, sbox, N_WIN, 21), "s99b": real_windows(s99, sbox, N_WIN, 22)}
    real = {k: describe(v) for k, v in gray.items()}
    reals = dict(floor_held=distances(real["held1"], real["held2"]), floor_s99=distances(real["s99a"], real["s99b"]),
                 held_vs_s99=distances(real["held1"], real["s99a"]), held2_vs_s99b=distances(real["held2"], real["s99b"]))
    print(json.dumps({k: {m: round(v, 3) for m, v in d.items()} for k, d in reals.items()}), flush=True)
    tiles = tile_grid(plane_t)
    z = torch.randn(N_WIN, 1, K, K, device="cuda", generator=torch.Generator(device="cuda").manual_seed(9000 + rep))
    hw = torch.as_tensor(gray["held1"][:256, None]).cuda()
    sw = torch.as_tensor(gray["s99a"][:256, None]).cuda()
    rows = []
    for lev, n_pairs in enumerate(LEVELS):
        paths = {s: CK / f"{prefix}_level{lev}_{s}.pt" for s in "AB"}
        if not all(p.exists() for p in paths.values()):
            break
        idx = chosen_tiles(rep, lev)
        g = {s: generate(load_model(paths[s]), z) for s in "AB"}
        gen = {s: describe(g[s]) for s in "AB"}
        row = dict(replicate=rep, level=lev, pairs=n_pairs, area_mm2=n_pairs * MM2_PER_TILE, agreement=agreement(g["A"], g["B"]),
                   to_held={s: distances(gen[s], real["held1"]) for s in "AB"},
                   to_s99={s: distances(gen[s], real["s99a"]) for s in "AB"},
                   convergence=distances(gen["A"], gen["B"]), copy={})
        for s in "AB":
            sel = torch.stack([tiles[t] for t in idx[s]])
            vg = match_tiles(torch.as_tensor(g[s][:256, None]).cuda(), sel)[0].cpu().numpy()
            vh = match_tiles(hw, sel)[0].cpu().numpy()
            vs = match_tiles(sw, sel)[0].cpu().numpy()
            row["copy"][s] = dict(vs_held=copying(vg, vh), vs_s99=copying(vg, vs), held_max=float(vh.max()), s99_max=float(vs.max()),
                                  held_median=float(np.median(vh)), s99_median=float(np.median(vs)), generated_median=float(np.median(vg)),
                                  held_vs_s99=copying(vh, vs))
        rows.append(row)
        print(json.dumps(dict(rep=rep, level=lev, area=round(row["area_mm2"], 3),
                              fid_held=round(row["to_held"]["A"]["fid"], 1), fid_s99=round(row["to_s99"]["A"]["fid"], 1),
                              tp_held=round(row["to_held"]["A"]["two_point"], 4), tp_s99=round(row["to_s99"]["A"]["two_point"], 4),
                              copy_held=round(row["copy"]["A"]["vs_held"], 3), copy_s99=round(row["copy"]["A"]["vs_s99"], 3),
                              max_held=round(row["copy"]["A"]["held_max"], 3), max_s99=round(row["copy"]["A"]["s99_max"], 3))), flush=True)
        out = OUT / (name or f"independent_rep{rep}.json")
        out.write_text(json.dumps(dict(real=reals, rows=rows)))


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.3)
    a = sys.argv[1:]
    if "--real" in a:
        real_only()
        sys.exit()
    cols = tuple(int(v) for v in a[a.index("--s99cols") + 1].split(":")) if "--s99cols" in a else None
    name = a[a.index("--name") + 1] if "--name" in a else None
    opts = ("--s99cols", "--name")
    for r in [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] not in opts)]:
        main(int(r), cols, name)
