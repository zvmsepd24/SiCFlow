"""Stitch the 6 x 6 montage of optical micrographs of one S200 serial section into a single image.

The tiles (1292 x 968 px, 0.52 um/px, 8-bit gray) are numbered in serpentine order: row 0 left to right,
row 1 right to left, and so on. Neighbouring tiles overlap by about 57 px horizontally and 44 px
vertically. Each pair of neighbours is registered by normalized cross-correlation of a strip of one tile
inside the other; the positions of all tiles are then the least-squares solution of the pairwise
offsets, weighted by the correlation peak. Overlaps are blended with linear ramps. No flat-field
correction is applied, so the stitch is processed like the stitched section the models were trained on.

Usage: python scripts/stitch_section.py TILE_DIR OUT.png [--pattern "{i:02d}.png"]
Writes OUT.png (the stitch) and OUT.json (tile positions, pairwise offsets and correlation peaks).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

N = 6
STRIP = 40  # width of the strip of one tile searched for in its neighbour (px); the overlaps are 44-57 px
SEARCH = 220  # depth of the neighbour searched (px)
SLACK = 80  # allowed misalignment across the seam (px)


def grid(i):
    r = i // N
    return r, (i % N if r % 2 == 0 else N - 1 - i % N)


def pair_offset(a, b, horizontal):
    """Offset (dx, dy) of tile b relative to tile a, and the correlation peak.

    horizontal: b is right of a, so the left strip of b is searched for in the right part of a;
    otherwise b is below a and the top strip of b is searched for in the bottom part of a."""
    h, w = a.shape
    if horizontal:
        tmpl = b[SLACK:h - SLACK, :STRIP]
        area = a[:, w - SEARCH:]
        r = cv2.matchTemplate(area, tmpl, cv2.TM_CCOEFF_NORMED)
        _, peak, _, (x, y) = cv2.minMaxLoc(r)
        return (w - SEARCH + x, y - SLACK), peak
    tmpl = b[:STRIP, SLACK:w - SLACK]
    area = a[h - SEARCH:, :]
    r = cv2.matchTemplate(area, tmpl, cv2.TM_CCOEFF_NORMED)
    _, peak, _, (x, y) = cv2.minMaxLoc(r)
    return (x - SLACK, h - SEARCH + y), peak


def solve(pairs, n):
    """Least-squares tile positions from weighted pairwise offsets; tile 0 at the origin."""
    rows, rhs, wts = [], [], []
    for i, j, (dx, dy), peak in pairs:
        wgt = max(peak, 1e-3) ** 2
        for k, d in ((0, dx), (1, dy)):
            row = np.zeros(2 * n)
            row[2 * j + k], row[2 * i + k] = 1, -1
            rows.append(row), rhs.append(d), wts.append(wgt)
    for k in (0, 1):  # anchor tile 0
        row = np.zeros(2 * n)
        row[k] = 1
        rows.append(row), rhs.append(0.0), wts.append(1e3)
    A, y, sw = np.array(rows), np.array(rhs), np.sqrt(np.array(wts))
    p = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0].reshape(n, 2)
    res = A @ p.ravel() - y
    return p, res[:-2]


def blend(tiles, pos):
    h, w = tiles[0].shape
    pos = np.round(pos - pos.min(0)).astype(int)
    H, W = pos[:, 1].max() + h, pos[:, 0].max() + w
    acc = np.zeros((H, W), np.float64)
    wsum = np.zeros((H, W), np.float64)
    ry = np.minimum(np.arange(h) + 1, h - np.arange(h)).astype(np.float64)
    rx = np.minimum(np.arange(w) + 1, w - np.arange(w)).astype(np.float64)
    ramp = np.minimum(ry[:, None], 60) * np.minimum(rx[None, :], 60)
    for t, (x, y) in zip(tiles, pos):
        acc[y:y + h, x:x + w] += ramp * t
        wsum[y:y + h, x:x + w] += ramp
    out = np.where(wsum > 0, acc / np.maximum(wsum, 1e-9), 0)
    return np.clip(np.round(out), 0, 255).astype(np.uint8), pos


def main(tile_dir, out, pattern="{i:02d}.png"):
    tiles = [np.asarray(Image.open(Path(tile_dir) / pattern.format(i=i)).convert("L")) for i in range(N * N)]
    at = {grid(i): i for i in range(N * N)}
    pairs = []
    for (r, c), i in at.items():
        for (dr, dc), horizontal in (((0, 1), True), ((1, 0), False)):
            j = at.get((r + dr, c + dc))
            if j is not None:
                d, peak = pair_offset(tiles[i], tiles[j], horizontal)
                pairs.append((i, j, d, peak))
    pos, res = solve(pairs, N * N)
    img, ipos = blend(tiles, pos)
    out = Path(out)
    Image.fromarray(img).save(out)
    info = {"tiles": [{"index": i, "row": grid(i)[0], "col": grid(i)[1], "x": int(ipos[i, 0]), "y": int(ipos[i, 1])}
                      for i in range(N * N)],
            "pairs": [{"a": i, "b": j, "dx": int(d[0]), "dy": int(d[1]), "peak": round(float(p), 4),
                       "residual_px": round(float(np.hypot(res[2 * k], res[2 * k + 1])), 2)}
                      for k, (i, j, d, p) in enumerate(pairs)],
            "size": [int(img.shape[1]), int(img.shape[0])]}
    out.with_suffix(".json").write_text(json.dumps(info, indent=1))
    peaks = np.array([p for *_, p in pairs])
    print(out.name, img.shape, "peak min/median", peaks.min().round(3), np.median(peaks).round(3),
          "max residual", np.abs(res).max().round(2))


if __name__ == "__main__":
    a = sys.argv[1:]
    pat = a[a.index("--pattern") + 1] if "--pattern" in a else "{i:02d}.png"
    main(a[0], a[1], pat)
