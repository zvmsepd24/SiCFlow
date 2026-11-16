"""Scores of real sets and band models against three references: is a model nearer its own section than section 99?

References (128 px windows, 1000 per set, as in scripts/evaluate_band_models.py):
* the held-out band of the training section (section 10, columns 2768 to the end);
* section 99 (89 sections deeper; other fibers, plies in part the same), its in-focus right half (columns 1700-3469);
* section 99 at the columns of the held-out band (2768-3469).
Compared sets: the training band and the held-out band of section 10, the references themselves, the 64-pixel
model of the whole band and the two-scale model (three seeds), whose fields are generated with the same seed as
in scripts/evaluate_band_models.py. Scores: FID (Inception-v3), KID (x1000) and the Fréchet distance of DINOv2
features.
Usage: python scripts/evaluate_against_sections.py [MODEL ...] -> results/sic/sections/cross_scores.json
       (MODEL: names of scripts/evaluate_band_models.py; default: the 64-pixel and two-scale models; the other entries
       of the file are kept)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genmicro.scores import features, frechet, kid  # noqa: E402
from scripts.evaluate_band_models import BAND, EXTRA, HELD, MODELS, generate, random_windows  # noqa: E402
from scripts.figstyle import TWO_SCALE  # noqa: E402

SEC = ROOT / "data/sic/sections"
OUT = ROOT / "results/sic/sections/cross_scores.json"
RIGHT, END99 = 1700, 3470
W, N = 128, 1000


def feats(images, seed):
    x = torch.as_tensor(random_windows(images, W, N, seed)[:, None])
    return {k: features(x, k) for k in ("inception", "dinov2")}


def scores(a, b):
    return dict(fid=frechet(a["inception"], b["inception"]), dinov2=frechet(a["dinov2"], b["dinov2"]),
                kid=kid(a["inception"], b["inception"]))


def main(names=None):
    unit = lambda a: a.astype(np.float32) / 127.5 - 1
    plane = np.load(ROOT / "data/sic/section_plane.npy")
    real = {"held_out_band": [unit(plane[:, HELD:])], "training_band": [unit(plane[:, :BAND])],
            "section99_right": [unit(np.load(SEC / "section99.npy")[:, RIGHT:END99])],
            "section99_held": [unit(np.load(SEC / "section99.npy")[:, HELD:END99])]}
    refs = ("held_out_band", "section99_right", "section99_held")
    # the reference windows (seed W) and a second draw of each real set (seed W + 1) to compare with them
    ref_f = {r: feats(real[r], W) for r in refs}
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, imgs in real.items():  # the real sets: with the default models, or when a reference is new to them
        if names and all(r in res.get(name, {}) for r in refs):
            continue
        f = feats(imgs, W + 1)
        res[name] = {r: scores(f, ref_f[r]) for r in refs}
        print(name, json.dumps({r: {k: round(v, 1) for k, v in s.items()} for r, s in res[name].items()}), flush=True)
    for name in names or ["single64", *TWO_SCALE]:
        spec = MODELS.get(name) or EXTRA[name]
        f = feats(generate(spec), W)
        res[name] = {r: scores(f, ref_f[r]) for r in refs}
        print(name, json.dumps({r: {k: round(v, 1) for k, v in s.items()} for r, s in res[name].items()}), flush=True)
        OUT.write_text(json.dumps(res, indent=1))
        torch.cuda.empty_cache()
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.4)
    torch.set_grad_enabled(False)
    main(sys.argv[1:])
