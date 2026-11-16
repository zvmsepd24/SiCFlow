"""Fidelity and convergence of paired models trained on the checkerboard split (scripts/train_checkerboard.py).

For every replicate and training area, models A and B generate 1024 windows of 64 x 64 px, the size of
their training crops, from the same noise. Real windows of the same size are cut at random positions
from the held-out band and from inside the training tiles of set A and of set B. Every window, real or
generated, is segmented on its own into fibers, pores and matrix (genmicro.phases).
Per window: the flattened gray image, the two-point function of the fiber phase along and across the
tow bands (separations 1-32 px), the chord-length distribution of the non-fiber phase along and across
(14 bins each), the pore fraction, the diameters of whole fibers, and Inception-v3 features.
Distances between ensembles: Fréchet distance between Gaussian fits (Inception features = FID; two-point
vectors; chord vectors) and Wasserstein-1 distances (pore fraction per window; fiber diameter). For each:
fidelity (generated vs held-out), convergence (A vs B), the same for real windows (real windows of the
training tiles vs held-out; real A vs real B), and two independent sets of held-out windows.
Levina-Bickel dimension (k = 10) of clouds of 512 windows: pixels, two-point vectors, chord vectors.
Copying: largest normalized cross-correlation of 256 generated windows with any window inside the model's
training tiles, and the same for held-out windows. Agreement: correlation between the windows A and B
generate from the same noise.
Usage: python scripts/evaluate_checkerboard.py REPLICATE [REPLICATE ...] -> results/sic/convergence/rep{R}.json
       (replicate 0 also writes rep0_samples.npz, the generated windows and similarities shown in the figures)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genmicro import fibers, sic  # noqa: E402
from genmicro.checkerboard import COLS, LEVELS, MM2_PER_TILE, K, T, agreement, chosen_tiles, match_tiles, tile_grid  # noqa: E402
from genmicro.descriptors import chords, two_point  # noqa: E402
from genmicro.lid import levina_bickel  # noqa: E402
from genmicro.phases import phases  # noqa: E402
from genmicro.scores import features, frechet, load_model  # noqa: E402
from genmicro.train import RHO, STEPS, sample  # noqa: E402

CK = ROOT / "results/sic/checkerboard"
OUT = ROOT / "results/sic/convergence"
N_WIN, N_LB = 1024, 512
UM = 1.04


@torch.no_grad()
def generate(model, z):
    out = []
    for i in range(0, len(z), 256):
        with torch.autocast("cuda", torch.bfloat16):
            out.append(sample(model, z[i:i + 256], STEPS, rho=RHO).float())
    return torch.cat(out)[:, 0].cpu().numpy()


def real_windows(plane, boxes, n, seed):
    """n random K x K gray windows lying inside the given (y, x, h, w) boxes, chosen in proportion to box area."""
    rng = np.random.default_rng(seed)
    w = np.array([(h - K + 1) * (wd - K + 1) for _, _, h, wd in boxes], float)
    which = rng.choice(len(boxes), n, p=w / w.sum())
    out = []
    for b in which:
        y0, x0, h, wd = boxes[b]
        y, x = y0 + rng.integers(0, h - K + 1), x0 + rng.integers(0, wd - K + 1)
        out.append(plane[y:y + K, x:x + K])
    return np.stack(out)


def describe(gray):
    ph = np.stack([phases(g) for g in gray])
    fib = torch.as_tensor(ph == 1).cuda()
    sx, sy, _ = two_point(fib, K // 2, periodic=False)
    hx, _ = chords(~fib, "x")
    hy, _ = chords(~fib, "y")
    return dict(pixels=gray.reshape(len(gray), -1).astype(np.float64),
                two_point=torch.cat([sx[:, 1:], sy[:, 1:]], 1).double().cpu().numpy(),
                chords=torch.cat([hx, hy], 1).double().cpu().numpy(),
                pore=(ph == 2).mean((1, 2)), diameter=fibers.collect(gray)["diameter"] * UM,
                inception=features(torch.as_tensor(gray[:, None]), "inception"))


def distances(a, b):
    return dict(fid=frechet(a["inception"], b["inception"]), two_point=frechet(a["two_point"], b["two_point"]),
                chords=frechet(a["chords"], b["chords"]), pore=float(wasserstein_distance(a["pore"], b["pore"])),
                diameter=float(wasserstein_distance(a["diameter"], b["diameter"])))


def dimension(d, seed):
    idx = np.random.default_rng(seed).choice(len(d["pixels"]), N_LB, replace=False)
    return {k: float(np.nanmean(levina_bickel(d[k][idx], k=10))) for k in ("pixels", "two_point", "chords")}


def main(rep, tag=None):
    OUT.mkdir(parents=True, exist_ok=True)
    prefix = f"rep{rep}" if tag is None else f"rep{rep}_{tag}"
    out = OUT / (f"rep{rep}.json" if tag is None else f"width_{tag}_rep{rep}.json")  # not matched by rep*.json
    plane_t = sic.load_plane()
    plane = plane_t.cpu().numpy()
    held_box = [(0, 2768, plane.shape[0], plane.shape[1] - 2768)]
    held1 = describe(real_windows(plane, held_box, N_WIN, 1))
    held2 = describe(real_windows(plane, held_box, N_WIN, 2))
    tiles = tile_grid(plane_t)
    z = torch.randn(N_WIN, 1, K, K, device="cuda", generator=torch.Generator(device="cuda").manual_seed(9000 + rep))
    rows, samples = [], {}
    for lev, n_pairs in enumerate(LEVELS):
        paths = {s: CK / f"{prefix}_level{lev}_{s}.pt" for s in "AB"}
        if not all(p.exists() for p in paths.values()):
            break
        tile_idx = chosen_tiles(rep, lev)
        boxes = {s: [((t // COLS) * T, (t % COLS) * T, T, T) for t in tile_idx[s]] for s in "AB"}
        real = {s: describe(real_windows(plane, boxes[s], N_WIN, 10 + lev + (s == "B"))) for s in "AB"}
        real_ab = {k: np.concatenate([real["A"][k][:N_WIN // 2], real["B"][k][:N_WIN // 2]]) for k in ("pixels", "two_point", "chords", "pore", "inception")}
        real_ab["diameter"] = np.concatenate([real["A"]["diameter"], real["B"]["diameter"]])
        gray = {s: generate(load_model(paths[s]), z) for s in "AB"}
        gen = {s: describe(gray[s]) for s in "AB"}
        sel = {s: torch.stack([tiles[t] for t in tile_idx[s]]) for s in "AB"}
        hw = torch.as_tensor(held1["pixels"][:256].reshape(-1, 1, K, K).astype(np.float32)).cuda()
        copy, sims = {}, {}
        for s in "AB":
            v, ti, yy, xx = match_tiles(torch.as_tensor(gray[s][:256, None]).cuda(), sel[s])
            sims[s] = (v, ti, yy, xx)
            vh = match_tiles(hw, sel[s])[0]
            copy[s] = dict(generated=float(v.median()), held_out=float(vh.median()), copies=float((v > 0.9).float().mean()),
                           generated_all=v.cpu().numpy().tolist(), held_out_all=vh.cpu().numpy().tolist())
        row = dict(replicate=rep, level=lev, pairs=n_pairs, area_mm2=n_pairs * MM2_PER_TILE,
                   fidelity={s: distances(gen[s], held1) for s in "AB"}, convergence=distances(gen["A"], gen["B"]),
                   real_fidelity=distances(real_ab, held1), real_convergence=distances(real["A"], real["B"]),
                   floor=distances(held1, held2),
                   dimension=dict(generated={s: dimension(gen[s], 3) for s in "AB"}, training={s: dimension(real[s], 4) for s in "AB"},
                                  held_out=dimension(held1, 5)),
                   copy=copy, agreement=agreement(gray["A"], gray["B"]),
                   fiber_median={s: float(np.median(gen[s]["diameter"])) for s in "AB"},
                   fiber_median_held=float(np.median(held1["diameter"])))
        rows.append(row)
        print(json.dumps(dict(rep=rep, level=lev, area=round(row["area_mm2"], 3), agree=round(row["agreement"], 3),
                              copy=round(copy["A"]["generated"], 3), fid=round(row["fidelity"]["A"]["fid"], 1),
                              fid_AB=round(row["convergence"]["fid"], 1), tp=round(row["fidelity"]["A"]["two_point"], 4),
                              tp_AB=round(row["convergence"]["two_point"], 4), dim_pix=round(row["dimension"]["generated"]["A"]["pixels"], 1),
                              dim_held=round(row["dimension"]["held_out"]["pixels"], 1))), flush=True)
        if rep == 0:
            v, ti, yy, xx = sims["A"]
            idx = np.arange(16)
            near = np.stack([sel["A"][ti[i]].cpu().numpy()[yy[i]:yy[i] + K, xx[i]:xx[i] + K] for i in idx])
            samples.update({f"genA{lev}": gray["A"][:64], f"genB{lev}": gray["B"][:64], f"nearA{lev}": near,
                            f"simA{lev}": v.cpu().numpy(), f"simheld{lev}": np.array(copy["A"]["held_out_all"])})
        out.write_text(json.dumps(rows))
    if rep == 0 and tag is None and samples:
        np.savez_compressed(OUT / "rep0_samples.npz", **{k: v.astype(np.float16) if v.dtype != np.float32 or "sim" not in k else v
                                                         for k, v in samples.items()})


if __name__ == "__main__":
    a = sys.argv[1:]
    torch.cuda.set_per_process_memory_fraction(0.5 if "--tag" in a else 0.25)
    tag = a[a.index("--tag") + 1] if "--tag" in a else None
    for r in [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] != "--tag")]:
        main(int(r), tag)
