"""Tow-scale evaluation: does a model generate the tow structure, and does it copy it from the training band?

Every model is compared through the state layout of its output (genmicro.twoscale.state_fields: local fiber and
pore fractions at 8.3 um per pixel). Image models (single-scale and two-scale) generate 24 fields of 1024 px, whose
state layout is computed as for the section; layout models generate 24 layouts of 128 px directly.
Computed:
* copying: largest normalized cross-correlation of windows of the fiber-fraction layout with any window of the
  training region (flips allowed), for 128 generated windows and 128 held-out windows; windows of 64 layout px
  (0.53 mm) and of the model's own crop size; for image models also 256 px windows of the image against the
  training band (64 windows each);
* tow statistics of the fiber-fraction layout: correlation along and across the bands (lags 0-64 layout px),
  standard deviation of the layout, and standard deviation of row means over 64 layout px (the strength of the
  bands at the scale of a model crop);
* same-noise agreement of layout models: correlation of the fiber-fraction layouts of two models from the same
  noise (computed for every pair of layout models evaluated together).
Usage: python scripts/evaluate_tow_scale.py MODEL ...   MODEL = single64 | twoscale_s1 | twoscale_state_s1 |
       layouts/NAME (results/sic/layouts/NAME.pt) -> results/sic/towscale/{MODEL}.json, fields in {MODEL}_fields.npz
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genmicro.scores import load_model  # noqa: E402
from genmicro.train import sample  # noqa: E402
from genmicro.twoscale import COND_NOISE, ImageConditioned, condition, state_fields  # noqa: E402

OUT = ROOT / "results/sic/towscale"
TW = ROOT / "results/sic/twoscale"
BAND, HELD = 2640, 2768
N_FIELDS, S = 24, 1024
LAGS = 64
IMAGE_CROP = {"single64": 64, "single128": 128, "single256": 256, "single512": 512,
              "single512_steps24000": 512}  # two-scale fiber models: 128


def box_sum(a, K):
    c = torch.nn.functional.pad(a.double().cumsum(0).cumsum(1), (1, 0, 1, 0))
    return (c[K:, K:] - c[:-K, K:] - c[K:, :-K] + c[:-K, :-K]).float()


@torch.no_grad()
def max_ncc(queries, region, chunk=8):
    """Largest NCC of each query (B, K, K) with any K x K window of region (H, W), over the four flips."""
    B, K, _ = queries.shape
    H, W = region.shape
    r = region.float().cuda()
    fr = torch.fft.rfft2(r)
    n = K * K
    mu = box_sum(r, K) / n
    sd = (box_sum(r * r, K) / n - mu * mu).clamp_min(1e-8).sqrt()
    best = torch.full((B,), -2.0, device="cuda")
    for flip in ((), (-1,), (-2,), (-1, -2)):
        for i in range(0, B, chunk):
            q = queries[i:i + chunk].float().cuda()
            q = q.flip(flip) if flip else q
            q = (q - q.mean((1, 2), keepdim=True)) / (q.std((1, 2), keepdim=True, unbiased=False) + 1e-8)
            c = torch.fft.irfft2(fr[None] * torch.fft.rfft2(q, s=(H, W)).conj(), s=(H, W))
            v = (c[:, :H - K + 1, :W - K + 1] / (n * sd[None])).flatten(1).amax(1)
            best[i:i + chunk] = torch.maximum(best[i:i + chunk], v)
    return best.cpu().numpy()


def windows(fields, K, n, rng):
    out = []
    for _ in range(n):
        f = fields[rng.integers(len(fields))]
        y, x = rng.integers(0, f.shape[-2] - K + 1), rng.integers(0, f.shape[-1] - K + 1)
        out.append(f[..., y:y + K, x:x + K])
    return torch.stack([torch.as_tensor(o) for o in out])


def correlation(fields):
    """Normalized autocorrelation of the fiber-fraction layout along x (rows) and across (columns), pooled."""
    num_x, num_y, den_x, den_y = 0, 0, 0, 0
    allv = np.concatenate([f.ravel() for f in fields])
    mu, var = allv.mean(), allv.var()
    for f in fields:
        a = f - mu
        h, w = a.shape
        num_x = num_x + np.array([(a[:, :w - k] * a[:, k:]).sum() for k in range(LAGS + 1)])
        den_x = den_x + np.array([h * (w - k) for k in range(LAGS + 1)])
        num_y = num_y + np.array([(a[:h - k] * a[k:]).sum() for k in range(LAGS + 1)])
        den_y = den_y + np.array([(h - k) * w for k in range(LAGS + 1)])
    return (num_x / den_x / var).tolist(), (num_y / den_y / var).tolist()


def tow_stats(fields):
    """fields: list of (h, w) fiber-fraction layouts in [0, 1]."""
    cx, cy = correlation(fields)
    rows = np.concatenate([f[:, x:x + 64].mean(1) for f in fields for x in range(0, f.shape[1] - 63, 32)])
    return dict(corr_along=cx, corr_across=cy, sd=float(np.concatenate([f.ravel() for f in fields]).std()),
                mean=float(np.mean([f.mean() for f in fields])), sd_row_means_64=float(rows.std()),
                pore_mean=None)


def load_layout_model(name):
    p = ROOT / "results/sic" / f"{name}.pt"
    ck = torch.load(p, map_location="cpu")
    return load_model(p), ck.get("meta", {})


@torch.no_grad()
def generate(name, seed=91):
    """-> state layouts (list of (2, h, w) in [-1, 1]), images (list or None), training columns."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    if name.startswith("layouts/") or name.startswith("twoscale/layout"):
        m, meta = load_layout_model(name)
        with torch.autocast("cuda", torch.bfloat16):
            lays = [sample(m, torch.randn(1, 2, S // 8, S // 8, device="cuda", generator=g)).float()[0].cpu().numpy()
                    for _ in range(N_FIELDS)]
        return lays, None, tuple(meta.get("cols", (0, BAND))), meta.get("crop", 64)
    if name.startswith("single"):
        m = load_model(ROOT / f"results/sic/single/{name}.pt")
        ims = []
        for _ in range(N_FIELDS):
            with torch.autocast("cuda", torch.bfloat16):
                ims.append(sample(m, torch.randn(1, 1, S, S, device="cuda", generator=g)).float()[0, 0].cpu().numpy())
    else:  # twoscale_s{k}, twoscale_state_s{k}, twoscale_stateaug_s{k} or twoscale_early_s{k}
        k = name.split("_s")[-1]
        variant = name.split("_")[1] if name.count("_") == 2 else ""
        length = name.split("_")[2] if name.startswith("twoscale_early_") and name.count("_") == 3 else None
        if length is not None:  # twoscale_early_{2k,6k,long}_s{k}: the fiber model after 2000, 6000 or 16000 steps
            coarse = load_model(ROOT / f"results/sic/layouts/steps1000_s{k}.pt")
            fine = ImageConditioned(load_model(TW / f"fiber_stateaug{length}_s{k}.pt")).eval()
            level = COND_NOISE
        elif variant in ("early", "earlyshort"):  # layout model stopped after 1000 steps; fiber model after 8000 or 4000
            coarse = load_model(ROOT / f"results/sic/layouts/steps1000_s{k}.pt")
            fine = ImageConditioned(load_model(TW / f"fiber_stateaug{'short' if variant == 'earlyshort' else ''}_s{k}.pt")).eval()
            level = COND_NOISE
        else:
            coarse = load_model(TW / f"layout_{'state_' if variant else ''}s{k}.pt")
            fine = ImageConditioned(load_model(TW / f"fiber_{variant + '_' if variant else ''}s{k}.pt")).eval()
            level = COND_NOISE if variant == "stateaug" else None
        ims = []
        for _ in range(N_FIELDS):
            with torch.autocast("cuda", torch.bfloat16):
                lay = sample(coarse, torch.randn(1, 2, S // 8, S // 8, device="cuda", generator=g)).float()
                ims.append(sample(fine, torch.randn(1, 1, S, S, device="cuda", generator=g), cond=condition(lay, level, g))
                           .float()[0, 0].cpu().numpy())
    lays = [state_fields(im)[0].cpu().numpy() for im in ims]
    return lays, ims, (0, BAND), 64


def frac(lay):
    return (lay + 1) / 2


def main(names):
    torch.backends.cudnn.benchmark = True
    OUT.mkdir(parents=True, exist_ok=True)
    plane = np.load(ROOT / "data/sic/section_plane.npy").astype(np.float32) / 127.5 - 1
    state = {}

    def region_state(c0, c1):
        if (c0, c1) not in state:
            state[(c0, c1)] = frac(state_fields(plane[:, c0:c1])[0].cpu().numpy())
        return state[(c0, c1)]

    held = region_state(HELD, plane.shape[1])
    train_band = region_state(0, BAND)
    ref_path = OUT / "real.json"
    if not ref_path.exists():
        halves = {"training band": (0, BAND), "left half": (0, 1320), "right half": (1320, BAND), "held-out band": (HELD, plane.shape[1])}
        ref = {k: tow_stats([region_state(*v)[0]]) for k, v in halves.items()}
        rng = np.random.default_rng(1)
        ref["copy_held"] = {}
        for K in (16, 32, 64):
            hw = windows([held[0]], K, 128, rng)
            for reg in ((0, BAND), (0, 1320), (1320, BAND), (0, 660)):
                ref["copy_held"][f"{reg[0]}:{reg[1]} {K}"] = max_ncc(hw, torch.as_tensor(region_state(*reg)[0])).tolist()
        ref["copy_image_held"] = {}
        for K in (64, 128, 256):
            hi = windows([plane[:, HELD:]], K, 128 if K < 256 else 64, rng)
            ref["copy_image_held"][K] = max_ncc(hi, torch.as_tensor(plane[:, :BAND])).tolist()
        ref_path.write_text(json.dumps(ref))
    ref = json.loads(ref_path.read_text())
    if "512" not in ref["copy_image_held"]:  # added later, without changing the other references
        ref["copy_image_held"]["512"] = max_ncc(windows([plane[:, HELD:]], 512, 32, np.random.default_rng(5)),
                                                torch.as_tensor(plane[:, :BAND])).tolist()
        ref_path.write_text(json.dumps(ref))
    layouts = {}
    for name in names:
        lays, ims, (c0, c1), crop = generate(name)
        layouts[name] = lays
        fib = [frac(l[0]) for l in lays]
        res = dict(model=name, cols=[c0, c1], crop=crop, tow=tow_stats(fib), copy={}, copy_other={})
        res["tow"]["pore_mean"] = float(np.mean([frac(l[1]).mean() for l in lays]))
        rng = np.random.default_rng(2)
        for K in sorted({64, crop}):
            gw = windows(fib, K, 128, rng)
            res["copy"][K] = max_ncc(gw, torch.as_tensor(region_state(c0, c1)[0])).tolist()
            if (c0, c1) in ((0, 1320), (1320, BAND)):
                other = (1320, BAND) if (c0, c1) == (0, 1320) else (0, 1320)
                res["copy_other"][K] = max_ncc(gw, torch.as_tensor(region_state(*other)[0])).tolist()
        if ims is not None:  # image level, at the crop size of the (fiber) model and at 256 px
            res["copy_image"] = {}
            for K in sorted({IMAGE_CROP.get(name, 128), 256}):
                iw = windows(ims, K, 128 if K < 256 else 64 if K < 512 else 32, rng)
                res["copy_image"][K] = max_ncc(iw, torch.as_tensor(plane[:, :BAND])).tolist()
        safe = name.replace("/", "_")
        (OUT / f"{safe}.json").write_text(json.dumps(res))
        np.savez_compressed(OUT / f"{safe}_fields.npz", layouts=np.stack(lays[:8]).astype(np.float16),
                            **({} if ims is None else dict(images=np.stack(ims[:2]).astype(np.float16))))
        q = np.array(res["copy"][64])
        print(name, f"copy median {np.median(q):.3f} q90 {np.quantile(q, 0.9):.3f}; sd {res['tow']['sd']:.3f}; "
              f"row-mean sd {res['tow']['sd_row_means_64']:.3f}; corr along 32 {res['tow']['corr_along'][32]:.3f}", flush=True)
        torch.cuda.empty_cache()
    pairs = {}
    lnames = [n for n in names if n.startswith(("layouts/", "twoscale/layout"))]
    for i, a in enumerate(lnames):
        for b in lnames[i + 1:]:
            x = torch.as_tensor(np.stack([l[0] for l in layouts[a]])).flatten(1)
            y = torch.as_tensor(np.stack([l[0] for l in layouts[b]])).flatten(1)
            x, y = x - x.mean(1, keepdim=True), y - y.mean(1, keepdim=True)
            pairs[f"{a} | {b}"] = float(((x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1))).mean())
    if pairs:
        p = OUT / "agreement.json"
        old = json.loads(p.read_text()) if p.exists() else {}
        p.write_text(json.dumps({**old, **pairs}, indent=1))
        print(json.dumps(pairs, indent=1))


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.4)
    main(sys.argv[1:])
