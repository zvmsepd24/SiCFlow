"""Segmentation of SiC/SiC images into fibers, pores and matrix.

Images are in [-1, 1] (gray level g in [0, 255] maps to g / 127.5 - 1).
Fibers: pixels brighter than gray level 183, opened with a 3 x 3 structuring element; connected regions
of 50-600 px (15-600 px if the region touches the image border, where fibers are cut) whose shape is
close to an ellipse are kept. The test compares the region's area with the area of the ellipse that has
the same second moments (fill ratio > 0.85); it removes bright, irregular patches of matrix and keeps
fibers cut by the image border, whose fill ratio stays above 0.9.
Pores: pixels darker than gray level 45, opened with a 3 x 3 structuring element; connected regions of
at least 40 px are kept. The dark rims around fibers are thinner than the opening and are removed.
Matrix: everything else.
Small windows can be segmented on their own, as in the checkerboard evaluation, where real and generated
64 px windows are treated alike; large fields are segmented whole and cut into windows afterwards.
"""
import numpy as np
from scipy import ndimage

FIBER_GRAY, PORE_GRAY = 183, 45
FIBER_T, PORE_T = FIBER_GRAY / 127.5 - 1, PORE_GRAY / 127.5 - 1
FIBER_AREA = (50, 600)
BORDER_MIN_AREA = 15
FILL_RATIO = 0.85
PORE_MIN_AREA = 40


def _labels(mask):
    lab, n = ndimage.label(mask)
    return lab, n


def fiber_labels(img):
    """Labelled fibers (0 = background) and the number of fibers."""
    m = ndimage.binary_opening(np.asarray(img) > FIBER_T, iterations=1)
    lab, n = _labels(m)
    if n == 0:
        return lab, 0
    idx = np.arange(1, n + 1)
    yy, xx = np.indices(m.shape, dtype=np.float64)
    area = ndimage.sum(m, lab, idx)
    sy, sx = ndimage.sum(yy, lab, idx), ndimage.sum(xx, lab, idx)
    syy, sxx, sxy = ndimage.sum(yy * yy, lab, idx), ndimage.sum(xx * xx, lab, idx), ndimage.sum(xx * yy, lab, idx)
    a = np.maximum(area, 1)
    vy, vx = syy / a - (sy / a) ** 2, sxx / a - (sx / a) ** 2
    cxy = sxy / a - (sx / a) * (sy / a)
    ellipse = 4 * np.pi * np.sqrt(np.maximum(vx * vy - cxy ** 2, 1e-9))
    border = np.zeros(n + 1, bool)
    border[np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))] = True
    min_area = np.where(border[1:], BORDER_MIN_AREA, FIBER_AREA[0])
    ok = (area >= min_area) & (area <= FIBER_AREA[1]) & (area / ellipse > FILL_RATIO)
    keep = np.zeros(n + 1, bool)
    keep[1:] = ok
    new = np.zeros(n + 1, np.int64)
    new[1:][ok] = np.arange(1, ok.sum() + 1)
    return new[lab] * keep[lab], int(ok.sum())


def fiber_mask(img):
    return fiber_labels(img)[0] > 0


def pore_mask(img):
    m = ndimage.binary_opening(np.asarray(img) < PORE_T, iterations=1)
    lab, n = _labels(m)
    if n == 0:
        return m
    area = ndimage.sum(m, lab, np.arange(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = area >= PORE_MIN_AREA
    return keep[lab]


def phases(img):
    """img (H, W) in [-1, 1] -> uint8 map: 0 matrix, 1 fiber, 2 pore."""
    out = np.zeros(np.asarray(img).shape, np.uint8)
    out[fiber_mask(img)] = 1
    out[pore_mask(img)] = 2
    return out
