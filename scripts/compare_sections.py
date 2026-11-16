"""How alike are section 99 and the training section? Persistence of the structure between two sections.

A section, registered to the training section (section 10) and cut in its frame (scripts/register_sections.py),
is compared with the training section at two scales:
- fiber scale: blocks of 128 x 128 px on a grid of the frame are matched in the other section within +-24 px
  (normalized cross-correlation); the peak tells whether the same fibers are still there, and the shift of the
  peak how far they moved (the drift of the fibers between the sections);
- tow scale: the state layout (local fiber and pore fractions, genmicro.twoscale.state_fields; Gaussian of 12 px,
  pooled 8x) of the two sections is correlated pixel by pixel, and the fiber-fraction profile across the tow bands
  (the layout averaged along the bands, i.e. the plies of the laminate) is correlated row by row; the layout minus
  that profile (the arrangement of the tows within the plies) is correlated separately.
- drift by height: the median shift of the blocks in each 48 px row band (the plies alternate in drift direction).
Only pixels covered by both montages are used.
Usage: python scripts/compare_sections.py 99 (and 10: section 10 stitched again) -> results/sic/sections/persistence.json
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genmicro.twoscale import state_fields  # noqa: E402

SEC = ROOT / "data/sic/sections"
RES = ROOT / "results/sic/sections"
B, R = 128, 24


def blocks(ref, mov, mask):
    out = []
    H, W = ref.shape
    for y in range(R + 32, H - B - R - 32, 160):
        for x in range(R + 32, W - B - R - 32, 160):
            if not mask[y - R:y + B + R, x - R:x + B + R].all():
                continue
            t = ref[y:y + B, x:x + B].astype(np.float32)
            if t.std() < 5:
                continue
            area = mov[y - R:y + B + R, x - R:x + B + R].astype(np.float32)
            r = cv2.matchTemplate(area, t, cv2.TM_CCOEFF_NORMED)
            _, pk, _, (u, v) = cv2.minMaxLoc(r)
            out.append((pk, u - R, v - R))
    a = np.array(out)
    return dict(n=len(a), peak_median=float(np.median(a[:, 0])), peak_q=[float(q) for q in np.percentile(a[:, 0], [10, 25, 75, 90])],
                shift_median_px=float(np.median(np.hypot(a[:, 1], a[:, 2]))),
                shift_mean_xy_px=[float(a[:, 1].mean()), float(a[:, 2].mean())],
                shift_std_xy_px=[float(a[:, 1].std()), float(a[:, 2].std())])


def drift_by_height(ref, mov, mask, b=96, r=24, band=48):
    """Median x-shift of fiber-scale blocks (peak > 0.6) in each band of rows: the drift of the fibers by ply."""
    H, W = ref.shape
    pts = []
    for y in range(r + 16, H - b - r - 16, band):
        for x in range(r + 16, W - b - r - 16, 96):
            if not mask[y - r:y + b + r, x - r:x + b + r].all():
                continue
            t = ref[y:y + b, x:x + b].astype(np.float32)
            if t.std() < 5:
                continue
            res = cv2.matchTemplate(mov[y - r:y + b + r, x - r:x + b + r].astype(np.float32), t, cv2.TM_CCOEFF_NORMED)
            _, pk, _, (u, v) = cv2.minMaxLoc(res)
            if pk > 0.6:
                pts.append((y + b / 2, u - r))
    a = np.array(pts) if pts else np.zeros((0, 2))
    out = []
    for y0 in range(0, H, band):
        s = a[(a[:, 0] >= y0) & (a[:, 0] < y0 + band)]
        out.append([y0 + band / 2, int(len(s)), float(np.median(s[:, 1])) if len(s) >= 5 else None])
    return out


def layout(img):
    return state_fields(img.astype(np.float32) / 127.5 - 1)[0].cpu().numpy()  # (2, H/8, W/8)


def tow(ref_lay, mov_lay, mask):
    m = cv2.resize(mask.astype(np.uint8), (ref_lay.shape[2], ref_lay.shape[1]), interpolation=cv2.INTER_NEAREST) > 0
    m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    out = {}
    for c, name in enumerate(("fiber", "pore")):
        a, b = ref_lay[c][m], mov_lay[c][m]
        out[f"{name}_layout_corr"] = float(np.corrcoef(a, b)[0, 1])
        out[f"{name}_fraction"] = [float((ref_lay[c][m] + 1).mean() / 2), float((mov_lay[c][m] + 1).mean() / 2)]
    cols = m.all(0)  # columns covered over the whole height
    pa, pb = ref_lay[0][:, cols].mean(1), mov_lay[0][:, cols].mean(1)
    out["band_profile_corr"] = float(np.corrcoef(pa, pb)[0, 1])
    # within the plies: the fiber-fraction layout minus its row means (the band profile)
    ra, rb = (ref_lay[0] - pa[:, None])[:, cols], (mov_lay[0] - pb[:, None])[:, cols]
    mm = m[:, cols]
    out["within_ply_corr"] = float(np.corrcoef(ra[mm], rb[mm])[0, 1])
    lag = np.arange(-40, 41)
    xc = [np.corrcoef(pa[max(0, -l):len(pa) - max(0, l)], pb[max(0, l):len(pb) - max(0, -l)])[0, 1] for l in lag]
    out["band_profile_best_lag_um"] = float(lag[int(np.argmax(xc))] * 8 * 1.04)
    out["band_profile_best_corr"] = float(np.max(xc))
    return out


def main(sections):
    RES.mkdir(parents=True, exist_ok=True)
    ref = np.load(ROOT / "data/sic/section_plane.npy")
    ref_lay = layout(ref)
    p = RES / "persistence.json"
    res = json.loads(p.read_text()) if p.exists() else {}
    for s in sections:
        mov = np.load(SEC / f"section{s:02d}.npy")
        mask = np.load(SEC / f"section{s:02d}_mask.npy")
        row = dict(section=s, offset=s - 10, fiber_scale=blocks(ref, mov, mask), tow_scale=tow(ref_lay, layout(mov), mask))
        if s != 99:  # the fibers of section 99 are no longer found
            row["drift_by_height"] = drift_by_height(ref, mov, mask)
        res[str(s)] = row
        print(s, round(row["fiber_scale"]["peak_median"], 3), round(row["fiber_scale"]["shift_median_px"], 1),
              {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row["tow_scale"].items()}, flush=True)
        p.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main([int(v) for v in sys.argv[1:]])
