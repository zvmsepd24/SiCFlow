"""Evaluation of models trained on the whole training band: structure within and beyond the training crop.

Each model generates 16 fields of 1024 x 1024 px. Real references: the training band and the held-out
band. Every image is segmented as a whole into fibers, pores and matrix (genmicro.phases). Computed:
* two-point correlation C(r) = (S2 - p^2) / (p - p^2) of the fiber phase along the tow bands (rows) and
  across them (columns), r = 0-400 px, pooled over the images;
* chord-length distributions of the non-fiber phase (matrix and pores) along and across;
* pore fraction of 256 px windows, and the standard deviation of the pore fraction between
  non-overlapping W x W windows for W = 8-512 px;
* fiber statistics (diameter, distance to neighboring fibers) from 256 px windows;
* FID (Inception-v3) to the held-out band on 1000 windows of 128 and 512 px; KID and the Fréchet distance
  with DINOv2 features for the supplement. Only the 128 px scores are reported: the held-out band holds 108
  non-overlapping 128 px windows but only four of 512 px;
* Levina-Bickel dimension (k = 10, 20) of clouds of 100 non-overlapping 128 px windows: flattened pixels,
  two-point vectors (fiber phase, separations 1-64 px along and across) and chord vectors; mean and s.d.
  over 20 random subsets.
Usage: python scripts/evaluate_band_models.py [MODEL ...] -> results/sic/bandmodels/{MODEL}.json, {MODEL}_fields.npz
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genmicro import fibers  # noqa: E402
from genmicro.descriptors import autocorrelation, chords, two_point  # noqa: E402
from genmicro.lid import levina_bickel  # noqa: E402
from genmicro.phases import phases  # noqa: E402
from genmicro.scores import features, frechet, kid, load_model  # noqa: E402
from genmicro.train import RHO, STEPS, sample  # noqa: E402
from genmicro.twoscale import (COND_NOISE, Guided, ImageConditioned, Mixed, condition, generate as generate_two_scale,  # noqa: E402
                               state_fields)

OUT = ROOT / "results/sic/bandmodels"
TW = ROOT / "results/sic/twoscale"
LAGS = 400
WINDOWS = (8, 16, 32, 64, 128, 256, 512)
N_FIELDS, S = 16, 1024
BAND, HELD = 2640, 2768  # training band: columns 0-2639; held-out band: columns 2768 to the end
MODELS = {
    **{f"single{c}": dict(single=f"results/sic/single/single{c}.pt") for c in (64, 128, 256, 512)},
    # two-scale model: layout model (state layout: local fiber and pore fractions) stopped after 1000 steps, before it
    # copies, and a fiber model trained with noise on the layout
    **{f"twoscale_early_s{k}": dict(coarse=f"../layouts/steps1000_s{k}.pt", fine=f"fiber_stateaug_s{k}.pt", ch=2,
                                    cond_noise=COND_NOISE) for k in (1, 2, 3)},
    # fiber models stopped after 4000 steps (with the layout model after 1000 steps)
    **{f"twoscale_earlyshort_s{k}": dict(coarse=f"../layouts/steps1000_s{k}.pt", fine=f"fiber_stateaugshort_s{k}.pt", ch=2,
                                         cond_noise=COND_NOISE) for k in (1, 2, 3)},
    # the same fiber models with the layout model trained for 6000 steps
    **{f"twoscale_stateaug_s{k}": dict(coarse=f"layout_state_s{k}.pt", fine=f"fiber_stateaug_s{k}.pt", ch=2,
                                       cond_noise=COND_NOISE) for k in (1, 2, 3)},
}
# check of the two-scale model: the fiber model trained on exact layouts (no noise); evaluated by name
# earlier two-scale models with the original layout (gray-level thresholds) and its fiber-model variants: not in the
# paper, evaluated only by name
EXTRA = {
    "twoscale_state_s1": dict(coarse="layout_state_s1.pt", fine="fiber_state_s1.pt", ch=2),
    # the fiber stage alone: the fiber model of seed 1 given the real layouts of 1024 px tiles of the training band
    "fibers_on_real_layouts": dict(fine="fiber_stateaug_s1.pt", real_layout=True, cond_noise=COND_NOISE),
    "fibers_on_real_layouts_n0": dict(fine="fiber_stateaug_s1.pt", real_layout=True, cond_noise=0.0),
    # fiber model trained on blurred noisy layouts (noise up to 0.5, Gaussian of 3 layout px), sampled at noise 0.25
    "twoscale_stateblur_s1": dict(coarse="../layouts/steps1000_s1.pt", fine="fiber_stateblur_s1.pt", ch=2,
                                  cond_noise=0.25, cond_blur=3.0),
    "fibers_blur_on_real_layouts": dict(fine="fiber_stateblur_s1.pt", real_layout=True, cond_noise=0.25, cond_blur=3.0),
    # checks of the fiber model: twice the training steps; 256 px crops
    "twoscale_early_1k_s1": dict(coarse="../layouts/steps1000_s1.pt", fine="fiber_stateaug1k_s1.pt", ch=2, cond_noise=COND_NOISE),
    **{f"twoscale_early_2k_s{k}": dict(coarse=f"../layouts/steps1000_s{k}.pt", fine=f"fiber_stateaug2k_s{k}.pt", ch=2,
                                       cond_noise=COND_NOISE) for k in (1, 2, 3)},
    **{f"twoscale_early_6k_s{k}": dict(coarse=f"../layouts/steps1000_s{k}.pt", fine=f"fiber_stateaug6k_s{k}.pt", ch=2,
                                       cond_noise=COND_NOISE) for k in (1, 2, 3)},
    **{f"twoscale_early_3k_s{k}": dict(coarse=f"../layouts/steps1000_s{k}.pt", fine=f"fiber_stateaug3k_s{k}.pt", ch=2,
                                       cond_noise=COND_NOISE) for k in (1, 2, 3)},
    "twoscale_early_long_s1": dict(coarse="../layouts/steps1000_s1.pt", fine="fiber_stateauglong_s1.pt", ch=2,
                                   cond_noise=COND_NOISE),
    "twoscale_early_short_s1": dict(coarse="../layouts/steps1000_s1.pt", fine="fiber_stateaugshort_s1.pt", ch=2,
                                    cond_noise=COND_NOISE),
    "twoscale_early_crop256_s1": dict(coarse="../layouts/steps1000_s1.pt", fine="fiber_stateaug256_s1.pt", ch=2,
                                      cond_noise=COND_NOISE),
    # velocity mixed from the fiber model (noisy layout) and the 128 px single-scale model
    **{f"twoscale_mix_l{int(10 * lam)}": dict(coarse="../layouts/steps1000_s1.pt", fine="fiber_stateaug_s1.pt", ch=2,
                                             cond_noise=COND_NOISE, mix="results/sic/single/single128.pt", lam=lam)
       for lam in (0.3, 0.5, 0.7)},
    # noise level of the layout when sampling the two-scale model (seed 1)
    **{f"twoscale_stateaug_s1_n{str(v).replace('.', '')}": dict(coarse="layout_state_s1.pt", fine="fiber_stateaug_s1.pt",
                                                               ch=2, cond_noise=v) for v in (0.0, 0.25, 1.0)},
    **{f"twoscale_s{k}": dict(coarse=f"layout_s{k}.pt", fine=f"fiber_s{k}.pt", ch=2) for k in (1, 2, 3)},
    **{f"twoscale_wide_s{k}": dict(coarse=f"layout_s{k}.pt", fine=f"fiber_s{k}_wide.pt", ch=2) for k in (1, 2, 3)},
    **{f"twoscale_guided_s{k}": dict(coarse=f"layout_s{k}.pt", fine=f"fiber_s{k}_guided.pt", ch=2, guidance=1.5) for k in (1, 2, 3)},
    "twoscale_long_s1": dict(coarse="layout_s1.pt", fine="fiber_s1_long.pt", ch=2),
    "twoscale_gray": dict(coarse="layout_gray_s1.pt", fine="fiber_gray_s1.pt", ch=1),
    # the fiber models of the two-scale model (2000 steps) with layout models trained for 6000 steps on the whole
    # training band
    **{f"twoscale_full_s{k}": dict(coarse=f"../layouts/full_s{k}.pt", fine=f"fiber_stateaug2k_s{k}.pt", ch=2,
                                   cond_noise=COND_NOISE) for k in (1, 2, 3)},
}


@torch.no_grad()
def generate(spec, seed=77):
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = []
    if "single" in spec:
        m = load_model(spec["single"])
        for _ in range(N_FIELDS):
            with torch.autocast("cuda", torch.bfloat16):
                out.append(sample(m, torch.randn(1, 1, S, S, device="cuda", generator=g), STEPS, rho=RHO).float()[0, 0].cpu().numpy())
        return out
    if spec.get("real_layout"):
        fine = ImageConditioned(load_model(TW / spec["fine"])).eval()
        plane = np.load(ROOT / "data/sic/section_plane.npy")
        plane = plane.astype(np.float32) / 127.5 - 1
        corners = spec.get("corners", [(y, x) for y in (0, 1024, 1352) for x in (0, 1024, 1616)])  # 1024 px tiles covering the band
        for k in range(N_FIELDS):
            y, x = corners[k % len(corners)]
            lay = state_fields(plane[y:y + S, x:x + S]).cuda()
            with torch.autocast("cuda", torch.bfloat16):
                c = condition(lay, spec["cond_noise"], g, spec.get("cond_blur", 0.0))
                out.append(sample(fine, torch.randn(1, 1, S, S, device="cuda", generator=g), STEPS, cond=c,
                                  rho=RHO).float()[0, 0].cpu().numpy())
        return out
    coarse = load_model(TW / spec["coarse"])
    net = load_model(TW / spec["fine"])
    fine = Guided(net, spec["guidance"]).eval() if "guidance" in spec else ImageConditioned(net).eval()
    if "mix" in spec:
        fine = Mixed(fine, load_model(spec["mix"]), spec["lam"]).eval()
    for _ in range(N_FIELDS):
        out.append(generate_two_scale(coarse, fine, S, g, channels=spec["ch"], cond_noise=spec.get("cond_noise"),
                                      cond_blur=spec.get("cond_blur", 0.0)).cpu().numpy())
    return out


def tiles(a, w):
    h, v = a.shape[0] // w, a.shape[1] // w
    return a[:h * w, :v * w].reshape(h, w, v, w).transpose(0, 2, 1, 3).reshape(-1, w, w)


def random_windows(images, w, n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        im = images[rng.integers(len(images))]
        y, x = rng.integers(0, im.shape[0] - w + 1), rng.integers(0, im.shape[1] - w + 1)
        out.append(im[y:y + w, x:x + w])
    return np.stack(out)


def curves(phase_maps):
    """Two-point correlation of the fiber phase and chord histograms of the non-fiber phase, pooled."""
    fib = [torch.as_tensor(p == 1).float().cuda()[None] for p in phase_maps]
    p = float(np.mean([f.mean().item() * f.numel() for f in fib]) / np.mean([f.numel() for f in fib]))
    c = torch.stack([autocorrelation(f, periodic=False)[0, :LAGS + 1, :LAGS + 1] for f in fib]).mean(0)
    norm = lambda s: ((s - p * p) / (p - p * p)).cpu().numpy().tolist()
    non = [torch.as_tensor(q != 1).cuda()[None] for q in phase_maps]
    hx = torch.stack([chords(m, "x")[0][0] for m in non]).mean(0).cpu().numpy().tolist()
    hy = torch.stack([chords(m, "y")[0][0] for m in non]).mean(0).cpu().numpy().tolist()
    return dict(fiber_fraction=p, corr_x=norm(c[0]), corr_y=norm(c[:, 0]), chord_x=hx, chord_y=hy)


def pore_stats(phase_maps):
    pore = [(q == 2).astype(np.float64) for q in phase_maps]
    frac = float(np.mean([p.mean() for p in pore]))
    sd = {}
    for w in WINDOWS:
        vals = np.concatenate([tiles(p, w).mean((1, 2)) for p in pore if min(p.shape) >= w])
        if len(vals) >= 16:
            sd[w] = float(vals.std())
    win256 = np.concatenate([tiles(p, 256).mean((1, 2)) for p in pore])
    return dict(fraction=frac, sd_between_windows=sd, window256=win256.tolist())


def cloud(gray_windows, phase_windows):
    fib = torch.as_tensor(phase_windows == 1).cuda()
    w = fib.shape[-1]
    sx, sy, _ = two_point(fib, w // 2, periodic=False)
    hx, _ = chords(~fib, "x")
    hy, _ = chords(~fib, "y")
    return dict(pixels=gray_windows.reshape(len(gray_windows), -1).astype(np.float64),
                two_point=torch.cat([sx[:, 1:], sy[:, 1:]], 1).double().cpu().numpy(),
                chords=torch.cat([hx, hy], 1).double().cpu().numpy())


def dimension(c, n=100, repeats=20, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for key, pts in c.items():
        for k in (10, 20):
            vals = [np.nanmean(levina_bickel(pts[rng.choice(len(pts), n, replace=False)], k=k)) for _ in range(repeats)]
            out[f"{key}_k{k}"] = dict(mean=float(np.mean(vals)), sd=float(np.std(vals)))
    return out


def evaluate(name, images, ref=None):
    ph = [phases(im) for im in images]
    res = dict(model=name, curves=curves(ph), pores=pore_stats(ph))
    from scripts.evaluate_tow_scale import tow_stats  # tow statistics of the fiber-fraction layout
    res["tow"] = tow_stats([(state_fields(im)[0, 0].cpu().numpy() + 1) / 2 for im in images])
    g256 = np.concatenate([tiles(im, 256) for im in images])
    res["fibers"] = {k: v.tolist() for k, v in fibers.collect(g256).items() if k in ("diameter", "edge_length")}
    g128 = np.concatenate([tiles(im, 128) for im in images])
    p128 = np.concatenate([tiles(q, 128) for q in ph])
    res["dimension"] = dimension(cloud(g128, p128))
    feats = {}
    for w in (128, 512):
        x = torch.as_tensor(random_windows(images, w, 1000, w)[:, None])
        feats[w] = {kind: features(x, kind) for kind in ("inception", "dinov2")}
    if ref is not None:
        rf = ref["_features"]
        res["scores"] = {f"{kind}_{w}": frechet(feats[w][kind], rf[w][kind]) for w in (128, 512) for kind in ("inception", "dinov2")}
        res["scores"].update({f"kid_{w}": kid(feats[w]["inception"], rf[w]["inception"]) for w in (128, 512)})
        rfib = {k: np.array(v) for k, v in ref["fibers"].items()}
        res["fiber_w1_um"] = {k: float(fibers.wasserstein_distance(np.array(res["fibers"][k]), rfib[k]) * 1.04) for k in rfib}
    res["_features"] = feats
    return res


def strip_private(res):
    return rounded({k: v for k, v in res.items() if not k.startswith("_")})


def rounded(o, digits=5):
    """Floats rounded to `digits` significant digits, to keep the result files small."""
    if isinstance(o, float):
        return float(f"{o:.{digits}g}")
    if isinstance(o, dict):
        return {k: rounded(v, digits) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [rounded(v, digits) for v in o]
    return o


def main(names):
    torch.backends.cudnn.benchmark = True
    OUT.mkdir(parents=True, exist_ok=True)
    plane = np.load(ROOT / "data/sic/section_plane.npy").astype(np.float32) / 127.5 - 1
    held = evaluate("held_out_band", [plane[:, HELD:]])
    train = evaluate("training_band", [plane[:, :BAND]], ref=held)
    for r in (held, train):
        (OUT / f"{r['model']}.json").write_text(json.dumps(strip_private(r)))
    print("real", json.dumps(train["scores"]), json.dumps(train["fiber_w1_um"]), flush=True)
    for name in names:
        if name not in MODELS and name not in EXTRA:
            continue
        spec = MODELS.get(name) or EXTRA[name]
        paths = ([ROOT / spec["single"]] if "single" in spec else [TW / spec["fine"]] if spec.get("real_layout")
                 else [TW / spec["coarse"], TW / spec["fine"]])
        if not all(p.exists() for p in paths):
            print(name, "missing checkpoint", flush=True)
            continue
        fields = generate(spec)
        res = evaluate(name, fields, ref=held)
        (OUT / f"{name}.json").write_text(json.dumps(strip_private(res)))
        np.savez_compressed(OUT / f"{name}_fields.npz", fields=np.stack(fields[:1]).astype(np.float16))
        print(name, json.dumps({k: round(v, 1) for k, v in res["scores"].items()}), json.dumps(res["fiber_w1_um"]),
              "pore", round(res["pores"]["fraction"], 4), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.3)
    main(sys.argv[1:] or list(MODELS))
