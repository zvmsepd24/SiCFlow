"""Register a stitched section (section 99) to the training section and cut it in its frame.

The training section (data/sic/section_plane.npy) is section 10 of the S200 series (Sample1_10), cut from the
stitch at half resolution. A stitched section (scripts/stitch_section.py) is registered to section 10
at the tow scale: both are reduced 8 times (4.2 um/px), blurred (sigma 2 px) and aligned with a Euclidean
transform (rotation and shift) maximizing the enhanced correlation coefficient, starting from phase
correlation. Each section is then resampled at half resolution (1.04 um/px) on the pixel grid of the training
section, with a mask of the pixels its montage covers.
Usage: python scripts/register_sections.py 10 99
   -> data/sic/sections/section{NN}.npy (uint8, the frame of section_plane.npy), section{NN}_mask.npy,
      results/sic/sections/registration.json
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parents[1]
ST = ROOT / "data/sic/stitched"
OUT = ROOT / "data/sic/sections"
RES = ROOT / "results/sic/sections"
F = 8  # reduction for the tow-scale registration


def half(img):
    return np.asarray(Image.fromarray(img).resize((img.shape[1] // 2, img.shape[0] // 2), Image.Resampling.LANCZOS))


def small(img):
    s = cv2.resize(img, (img.shape[1] // F, img.shape[0] // F), interpolation=cv2.INTER_AREA).astype(np.float32)
    return cv2.GaussianBlur(s, (0, 0), 2)


def frame_origin(plane, ref_half):
    """Top-left of the training section in the half-resolution stitch of section 10 (y, x)."""
    t = plane[800:1100, 1200:1500].astype(np.float32)
    r = cv2.matchTemplate(ref_half.astype(np.float32), t, cv2.TM_CCOEFF_NORMED)
    _, pk, _, (x, y) = cv2.minMaxLoc(r)
    return y - 800, x - 1200, float(pk)


def register(ref_s, mov_s):
    """Euclidean warp (2 x 3) taking reference coordinates to moving coordinates, at the reduced scale."""
    h, w = min(ref_s.shape[0], mov_s.shape[0]), min(ref_s.shape[1], mov_s.shape[1])
    ref_s, mov_s = ref_s[:h, :w].copy(), mov_s[:h, :w].copy()
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (sx, sy), _ = cv2.phaseCorrelate(ref_s, mov_s, win)
    warp = np.array([[1, 0, sx], [0, 1, sy]], np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-7)
    cc, warp = cv2.findTransformECC(ref_s, mov_s, warp, cv2.MOTION_EUCLIDEAN, crit, None, 5)
    return warp, float(cc)


def main(sections):
    OUT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    plane = np.load(ROOT / "data/sic/section_plane.npy")
    ref = np.asarray(Image.open(ST / "slice10.png"))
    ref_half = half(ref)
    oy, ox, pk = frame_origin(plane, ref_half)
    ref_s = small(ref)
    info = {"frame_origin_half_res_yx": [int(oy), int(ox)], "frame_match": round(pk, 4), "sections": {}}
    H, W = plane.shape
    for s in sections:
        mov = np.asarray(Image.open(ST / f"slice{s:02d}.png"))
        warp, cc = register(ref_s, small(mov))
        # warp at half resolution: reference half-res coords (frame + origin) -> moving half-res coords
        a = warp[:, :2].astype(np.float64)
        t = warp[:, 2].astype(np.float64) * F / 2
        mov_half = half(mov)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        rx, ry = xx + ox, yy + oy
        mx = a[0, 0] * rx + a[0, 1] * ry + t[0]
        my = a[1, 0] * rx + a[1, 1] * ry + t[1]
        cut = cv2.remap(mov_half, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_CUBIC, borderValue=0)
        mask = ((mx >= 0) & (mx <= mov_half.shape[1] - 1) & (my >= 0) & (my <= mov_half.shape[0] - 1))
        np.save(OUT / f"section{s:02d}.npy", cut)
        np.save(OUT / f"section{s:02d}_mask.npy", mask)
        angle = float(np.degrees(np.arctan2(a[1, 0], a[0, 0])))
        info["sections"][str(s)] = dict(ecc=round(cc, 4), angle_deg=round(angle, 4),
                                        shift_um=[round(float(t[0]) * 1.04, 1), round(float(t[1]) * 1.04, 1)],
                                        covered=round(float(mask.mean()), 4))
        print(s, info["sections"][str(s)], flush=True)
    p = RES / "registration.json"
    old = json.loads(p.read_text()) if p.exists() else {"sections": {}}
    old["sections"].update(info["sections"])
    info["sections"] = old["sections"]
    p.write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main([int(v) for v in sys.argv[1:]])
