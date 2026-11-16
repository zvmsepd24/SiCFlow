"""Section 99 as a real reference for the band models: how far is another section of the same specimen?

The real sections are scored exactly like the generated fields of the models of the whole training band
(scripts/evaluate_band_models.py: FID, KID and DINOv2 distance of 128 px windows to the held-out band of the
training section, fiber statistics, pores, tow statistics, dimension). Sections registered to the training
section and cut in its frame (scripts/register_sections.py):
* section 99 (89 sections away: other fibers, the same tow bands in part): the columns 2700-3469 (the held-out band as far as
  its montage covers (2768-3475), its right half (1700-3469, in focus) and all columns it covers. Its left third
  is out of focus (a tilt of the specimen during imaging);
* imaging control: the held-out band of the training section blurred with the Gaussian whose sharpness (fine over
  medium detail, genmicro-free measure below) matches that of section 99's right half, to separate the cost of
  blur from that of other material.
Usage: python scripts/evaluate_sections.py [--all] -> results/sic/sections/references.json (without --all, references
       already in the file are kept)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_band_models import BAND, HELD, evaluate, strip_private  # noqa: E402

SEC = ROOT / "data/sic/sections"
RES = ROOT / "results/sic/sections"
RIGHT, END99, HELD99 = 1700, 3470, 2700  # HELD99: six 128 px windows wide, as the held-out band


def sharpness(a, k=64, n=3000, seed=0):
    """Median over random k-px windows of the fine-detail energy (a - G1) over the medium-detail energy (G1 - G3)."""
    a = a.astype(np.float32)
    g1, g3 = cv2.GaussianBlur(a, (0, 0), 1.0), cv2.GaussianBlur(a, (0, 0), 3.0)
    s = cv2.boxFilter((a - g1) ** 2, -1, (k, k)) / (cv2.boxFilter((g1 - g3) ** 2, -1, (k, k)) + 1e-3)
    rng = np.random.default_rng(seed)
    ys, xs = rng.integers(k, a.shape[0] - k, n), rng.integers(k, a.shape[1] - k, n)
    return float(np.median(s[ys, xs]))


def to_unit(a):
    return a.astype(np.float32) / 127.5 - 1


def main():
    RES.mkdir(parents=True, exist_ok=True)
    plane8 = np.load(ROOT / "data/sic/section_plane.npy")
    held8 = plane8[:, HELD:]
    held = evaluate("held_out_band", [to_unit(held8)])
    refs = {"training_band": [to_unit(plane8[:, :BAND])]}
    s99 = np.load(SEC / "section99.npy")
    m99 = np.load(SEC / "section99_mask.npy")
    assert m99[:, :END99].all(), "section 99 should cover columns 0-3469"
    refs["section99_held"] = [to_unit(s99[:, HELD99:END99])]
    refs["section99_right"] = [to_unit(s99[:, RIGHT:END99])]
    refs["section99_all"] = [to_unit(s99[:, :END99])]
    target = sharpness(s99[:, RIGHT:END99])
    sig = {f"{v:.2f}": sharpness(cv2.GaussianBlur(held8.astype(np.float32), (0, 0), v)) for v in np.arange(0.2, 0.81, 0.05)}
    best = min(sig, key=lambda k: abs(sig[k] - target))
    refs[f"held_blur{best}"] = [to_unit(np.clip(cv2.GaussianBlur(held8.astype(np.float32), (0, 0), float(best)), 0, 255))]
    out = {"held_out_band": strip_private(held),
           "sharpness": {"held_out_band": sharpness(held8), "training_band": sharpness(plane8[:, :BAND]),
                         "section99_right": target, "section99_held": sharpness(s99[:, HELD99:END99]),
                         "section99_all": sharpness(s99[:, :END99]), "blur_sigma_px": float(best), "blurred_held": sig[best]}}
    print(json.dumps(out["sharpness"]), flush=True)
    p = RES / "references.json"
    done = json.loads(p.read_text()) if p.exists() else {}
    for name, images in refs.items():
        if name in done and "--all" not in sys.argv:  # scored before (the held-out reference is deterministic)
            out[name] = done[name]
            continue
        r = evaluate(name, images, ref=held)
        out[name] = strip_private(r)
        print(name, json.dumps({k: round(v, 2) for k, v in r["scores"].items()}), json.dumps(r["fiber_w1_um"]),
              "pore", round(r["pores"]["fraction"], 4), "fiber", round(r["curves"]["fiber_fraction"], 4), flush=True)
        (RES / "references.json").write_text(json.dumps(out))
        torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.5)
    main()
