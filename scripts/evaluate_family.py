"""The paired test for other model families (scripts/train_family.py), as for the flow models.

For every training area: models A, B and A2 (set A, another seed) generate 1024 windows of 64 px from the same
latent. Computed as in scripts/evaluate_checkerboard.py (same real windows, same descriptors and distances):
fidelity (generated vs held-out), convergence (A vs B), seeds (A vs A2), copying of the own tiles, same-latent
agreement of A and B and of A and A2, and the Levina-Bickel dimension of the clouds. For Glow also the
likelihood: bits per dimension of 512 windows of the own training tiles and of 512 held-out windows.
Usage: python scripts/evaluate_family.py FAMILY REPLICATE -> results/sic/families/{FAMILY}_rep{R}.json
       python scripts/evaluate_family.py glow 0 --temperature 0.7 -> results/sic/families/glow_T0.7_rep0.json
       (and {FAMILY}_rep{R}_samples.npz: 64 windows of A, B and A2 at every area)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genmicro import sic  # noqa: E402
from genmicro.checkerboard import COLS, LEVELS, MM2_PER_TILE, T, agreement, chosen_tiles, match_tiles, tile_grid  # noqa: E402
from scripts.evaluate_checkerboard import N_WIN, describe, dimension, real_windows  # noqa: E402
from scripts.evaluate_checkerboard import distances as _distances  # noqa: E402


def distances(a, b):
    """As for the flow models; windows without any segmented fiber (blurry samples) give no diameter distance."""
    if len(a["diameter"]) == 0 or len(b["diameter"]) == 0:
        a, b = dict(a, diameter=np.array([np.nan])), dict(b, diameter=np.array([np.nan]))
    return _distances(a, b)
from scripts.train_family import OUT, load  # noqa: E402

KEYS = ("pixels", "two_point", "chords", "pore", "inception")


@torch.no_grad()
def generate(model, z):
    out = []
    for i in range(0, len(z[0]) if isinstance(z, list) else len(z), 256):
        zi = [x[i:i + 256] for x in z] if isinstance(z, list) else z[i:i + 256]
        with torch.autocast("cuda", torch.bfloat16):
            out.append(model.generate(zi).float())
    return torch.cat(out)[:, 0].cpu().numpy()


def main(family, rep, temperature=None):
    tag = family if temperature is None else f"{family}_T{temperature:g}"
    plane_t = sic.load_plane()
    plane = plane_t.cpu().numpy()
    held_box = [(0, 2768, plane.shape[0], plane.shape[1] - 2768)]
    held_gray = real_windows(plane, held_box, N_WIN, 1)
    held1, held2 = describe(held_gray), describe(real_windows(plane, held_box, N_WIN, 2))
    tiles = tile_grid(plane_t)
    rows, samples = [], {}
    for lev, n_pairs in enumerate(LEVELS):
        paths = {s: OUT / f"{family}_rep{rep}_level{lev}_{s}.pt" for s in ("A", "B", "A2")}
        if not all(p.exists() for p in paths.values()):
            break
        models = {s: load(p) for s, p in paths.items()}
        if temperature is not None:  # Glow sampled at a lower temperature (Kingma and Dhariwal use 0.7)
            for m in models.values():
                m.temperature = temperature
        z = models["A"].noise(N_WIN, torch.Generator(device="cuda").manual_seed(9000 + rep))
        tile_idx = chosen_tiles(rep, lev)
        boxes = {s: [((t // COLS) * T, (t % COLS) * T, T, T) for t in tile_idx[s]] for s in "AB"}
        real = {s: describe(real_windows(plane, boxes[s], N_WIN, 10 + lev + (s == "B"))) for s in "AB"}
        gray = {s: generate(m, z) for s, m in models.items()}
        gen = {s: describe(g) for s, g in gray.items()}
        sel = {s: torch.stack([tiles[t] for t in tile_idx[s]]) for s in "AB"}
        hw = torch.as_tensor(held_gray[:256, None]).cuda()
        copy = {}
        for s in ("A", "B", "A2"):
            own = sel["A" if s == "A2" else s]
            v = match_tiles(torch.as_tensor(gray[s][:256, None]).cuda(), own)[0]
            vh = match_tiles(hw, own)[0]
            copy[s] = dict(generated=float(v.median()), held_out=float(vh.median()),
                           above_held=float((v > vh.max()).float().mean()), generated_all=v.cpu().numpy().tolist())
        row = dict(family=family, replicate=rep, level=lev, pairs=n_pairs, area_mm2=n_pairs * MM2_PER_TILE,
                   fidelity={s: distances(gen[s], held1) for s in gen}, convergence=distances(gen["A"], gen["B"]),
                   seeds=distances(gen["A"], gen["A2"]), real_convergence=distances(real["A"], real["B"]),
                   floor=distances(held1, held2),
                   dimension=dict(generated={s: dimension(gen[s], 3) for s in gen}, training={s: dimension(real[s], 4) for s in "AB"},
                                  held_out=dimension(held1, 5)),
                   copy=copy, agreement=agreement(gray["A"], gray["B"]), agreement_seeds=agreement(gray["A"], gray["A2"]),
                   pore_fraction={s: float(np.mean(gen[s]["pore"])) for s in gen},
                   fibers_per_window={s: len(gen[s]["diameter"]) / N_WIN for s in gen})
        if family == "glow":
            gbit = torch.Generator(device="cuda").manual_seed(5)
            tr = {s: torch.as_tensor(real_windows(plane, boxes[s], 512, 20 + lev)[:, None]).cuda() for s in "AB"}
            ho = torch.as_tensor(held_gray[:512, None]).cuda()
            row["bits_per_dim"] = {}
            for s in "AB":  # per window; held-out windows far outside the learned density dominate the mean
                bt, bh = models[s].bits_per_dim(tr[s], 4, gbit).cpu().numpy(), models[s].bits_per_dim(ho, 4, gbit).cpu().numpy()
                row["bits_per_dim"][s] = dict(training=float(bt.mean()), held_out=float(bh.mean()),
                                              training_median=float(np.median(bt)), held_out_median=float(np.median(bh)),
                                              training_all=bt.tolist(), held_out_all=bh.tolist())
        rows.append(row)
        print(json.dumps(dict(family=family, level=lev, area=round(row["area_mm2"], 3), agree=round(row["agreement"], 3),
                              agree_seeds=round(row["agreement_seeds"], 3), copy=round(copy["A"]["generated"], 3),
                              copy_held=round(copy["A"]["held_out"], 3), above=round(copy["A"]["above_held"], 3),
                              fid=round(row["fidelity"]["A"]["fid"], 1), fid_AB=round(row["convergence"]["fid"], 1),
                              fid_seeds=round(row["seeds"]["fid"], 1), tp=round(row["fidelity"]["A"]["two_point"], 4),
                              tp_AB=round(row["convergence"]["two_point"], 4),
                              bpd={s: {k: round(v, 3) for k, v in d.items() if not k.endswith("_all")}
                                   for s, d in row.get("bits_per_dim", {}).items()})), flush=True)
        samples.update({f"{s}{lev}": gray[s][:64].astype(np.float16) for s in gray})
        (OUT / f"{tag}_rep{rep}.json").write_text(json.dumps(rows))
        torch.cuda.empty_cache()
    np.savez_compressed(OUT / f"{tag}_rep{rep}_samples.npz", **samples)


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.8)
    a = sys.argv[1:]
    main(a[0], int(a[1]), float(a[a.index("--temperature") + 1]) if "--temperature" in a else None)
