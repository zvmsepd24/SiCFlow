"""Fiber models of the two-scale model on real layouts: what does the layout fix, and does the model copy?

For each fiber model, the layout of the held-out band (rows 0-2375, columns 2768-3591) is computed as in training
(the original layout for fiber_s{k}, the state layout for fiber_state_s{k} and fiber_stateaug_s{k}) and three images
are generated from it with different noise. Fiber models trained with conditioning augmentation are evaluated at
several noise levels of the layout (name@level). Computed:
* pixel correlation of each generated image with the real held-out image (after removing the means): close to 1
  if the layout fixes the positions of single fibers;
* correlation between two generated images from the same layout (different noise);
* the same correlation after blurring both images with the state-layout Gaussian (12 px): agreement at the scale
  of the layout;
* rim gray level: median, over segmented fibers, of the mean gray level of the 2 px ring outside each fiber (real
  fibers have dark rims; fibers without rims are the half-formed fibers);
* fraction of rim-dark pixels (gray levels 45-89: the dark rims around fibers; pores are darker);
* fiber fraction, pore fraction and median fiber diameter of the generated images;
* copying of the training band: correlation of an image generated from the layout of a training-band region
  (rows 600-1623, columns 800-1823) with that region, after removing the layout scale (Gaussian of 12 px).
Usage: python scripts/evaluate_fiber_given_layout.py [fiber_s1 fiber_state_s1 ...]
       -> results/sic/towscale/given_layout.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genmicro import fibers  # noqa: E402
from genmicro.phases import fiber_labels, phases  # noqa: E402
from genmicro.scores import load_model  # noqa: E402
from genmicro.train import sample  # noqa: E402
from genmicro.twoscale import (STATE_SIGMA, ImageConditioned, condition, fraction_fields, gaussian_blur,  # noqa: E402
                               state_fields)

OUT = ROOT / "results/sic/towscale"
TW = ROOT / "results/sic/twoscale"
HELD = 2768


def rim_gray(img):
    lab, n = fiber_labels(img)
    if n == 0:
        return float("nan")
    fib = lab > 0
    ring = ndimage.binary_dilation(fib, iterations=2) & ~fib
    grown = ndimage.grey_dilation(lab, size=5)
    return float(np.nanmedian(ndimage.mean(img, grown * ring, np.arange(1, n + 1))))


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def blur(a):
    return gaussian_blur(torch.as_tensor(a)[None, None].float(), STATE_SIGMA)[0, 0].numpy()


def high_pass(a):
    return a - blur(a)


def dark_fraction(img):
    g = (img + 1) * 127.5
    return float(((g > 45) & (g < 89)).mean())


def describe(img):
    ph = phases(img)
    d = fibers.collect(np.stack([img[y:y + 256, x:x + 256] for y in range(0, img.shape[0] - 255, 256)
                                 for x in range(0, img.shape[1] - 255, 256)]))
    return dict(rim_gray=rim_gray(img), dark_fraction=dark_fraction(img), fiber_fraction=float((ph == 1).mean()),
                pore_fraction=float((ph == 2).mean()),
                diameter_um=float(np.median(d["diameter"]) * 1.04))


@torch.no_grad()
def main(names):
    OUT.mkdir(parents=True, exist_ok=True)
    plane = np.load(ROOT / "data/sic/section_plane.npy").astype(np.float32) / 127.5 - 1
    real = plane[:, HELD:HELD + 824]
    train = plane[600:1624, 800:1824]
    res = {"real": describe(real)}
    for spec in names:
        name, level = (spec.split("@") + [None])[:2]
        level = None if level is None else float(level)
        fine = ImageConditioned(load_model(TW / f"{name}.pt")).eval()
        lay_of = lambda im: (state_fields(im) if "state" in name else fraction_fields(torch.as_tensor(im).cuda())).cuda()
        ims = []
        for seed in range(4):
            g = torch.Generator(device="cuda").manual_seed(seed)
            src = train if seed == 3 else real
            with torch.autocast("cuda", torch.bfloat16):
                c = condition(lay_of(src), level, g, 3.0 if "stateblur" in name else 0.0)
                x = sample(fine, torch.randn(1, 1, *src.shape, device="cuda", generator=g), cond=c)
            ims.append(x.float()[0, 0].cpu().numpy())
        copy = corr(high_pass(ims[3]), high_pass(train))
        on_train = ims[3]
        ims = ims[:3]
        r = dict(corr_real=[corr(im, real) for im in ims], corr_pair=corr(ims[0], ims[1]),
                 corr_real_high=[corr(high_pass(im), high_pass(real)) for im in ims],
                 corr_pair_high=corr(high_pass(ims[0]), high_pass(ims[1])),
                 corr_real_blur=[corr(blur(im), blur(real)) for im in ims], corr_pair_blur=corr(blur(ims[0]), blur(ims[1])),
                 copy_training_high=copy, **describe(ims[0]))
        name = spec
        res[name] = r
        np.savez_compressed(OUT / f"given_layout_{name}.npz", image=ims[0][:1024].astype(np.float16),
                            on_train=on_train.astype(np.float16))
        print(name, json.dumps({k: (np.round(v, 3).tolist() if isinstance(v, list) else round(v, 3)) for k, v in r.items()}), flush=True)
    print("real", json.dumps({k: round(v, 3) for k, v in res["real"].items()}))
    p = OUT / "given_layout.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    p.write_text(json.dumps({**old, **res}, indent=1))


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.4)
    main(sys.argv[1:] or ["fiber_s1", "fiber_state_s1"])
