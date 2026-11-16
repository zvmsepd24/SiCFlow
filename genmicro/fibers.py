"""Fiber statistics of SiC/SiC images: segmented fibers, their sizes and the Delaunay graph of centroids.

Fibers are segmented with genmicro.phases (threshold, opening, size and ellipse-shape test). Per fiber:
equivalent diameter, aspect ratio and orientation from second moments; fibers touching the image border
are cut and excluded from these. Per image: the Delaunay triangulation of all fiber centroids gives the
distance to neighboring fibers (edges between fibers at least EDGE px from the border) and the number of
neighbors per interior fiber. Two sets of images are compared by the Wasserstein-1 distance per statistic.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import Delaunay
from scipy.stats import wasserstein_distance

from genmicro.phases import fiber_labels

EDGE = 20


def fiber_table(img):
    """img: (H, W) array in [-1, 1] -> dict of per-fiber arrays (diameter etc. for whole fibers only)."""
    lab, n = fiber_labels(img)
    empty = dict(centroid=np.zeros((0, 2)), diameter=np.zeros(0), aspect=np.zeros(0), angle=np.zeros(0))
    if n == 0:
        return empty
    idx = np.arange(1, n + 1)
    yy, xx = np.indices(lab.shape, dtype=np.float64)
    m = lab > 0
    cnt = ndimage.sum(m, lab, idx)
    my, mx = ndimage.sum(yy, lab, idx) / cnt, ndimage.sum(xx, lab, idx) / cnt
    syy = ndimage.sum(yy * yy, lab, idx) / cnt - my ** 2
    sxx = ndimage.sum(xx * xx, lab, idx) / cnt - mx ** 2
    sxy = ndimage.sum(xx * yy, lab, idx) / cnt - mx * my
    tr, det = sxx + syy, sxx * syy - sxy ** 2
    disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
    l1, l2 = tr / 2 + disc, np.maximum(tr / 2 - disc, 1e-9)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    whole = ~np.isin(idx, border)
    return dict(centroid=np.stack([my, mx], 1), diameter=np.sqrt(4 * cnt / np.pi)[whole], aspect=np.sqrt(l1 / l2)[whole],
                angle=np.degrees(0.5 * np.arctan2(2 * sxy, sxx - syy))[whole])


def graph_stats(centroid, shape):
    """Delaunay edge lengths and neighbor counts (interior fibers only)."""
    if len(centroid) < 4:
        return np.zeros(0), np.zeros(0)
    tri = Delaunay(centroid)
    e = np.concatenate([tri.simplices[:, [0, 1]], tri.simplices[:, [1, 2]], tri.simplices[:, [0, 2]]])
    e = np.unique(np.sort(e, 1), axis=0)
    length = np.linalg.norm(centroid[e[:, 0]] - centroid[e[:, 1]], axis=1)
    deg = np.bincount(e.ravel(), minlength=len(centroid))
    inner = ((centroid[:, 0] > EDGE) & (centroid[:, 0] < shape[0] - EDGE)
             & (centroid[:, 1] > EDGE) & (centroid[:, 1] < shape[1] - EDGE))
    both_inner = inner[e[:, 0]] & inner[e[:, 1]]
    return length[both_inner], deg[inner]


def collect(images):
    """Pool fiber and graph statistics over a set of images (N, H, W)."""
    out = {k: [] for k in ("diameter", "aspect", "angle", "edge_length", "neighbors", "count")}
    for img in images:
        f = fiber_table(img)
        el, deg = graph_stats(f["centroid"], img.shape)
        for k in ("diameter", "aspect", "angle"):
            out[k].append(f[k])
        out["edge_length"].append(el)
        out["neighbors"].append(deg)
        out["count"].append(np.array([len(f["centroid"]) / img.size * 1e4]))  # fibers per 100 x 100 px
    return {k: np.concatenate(v) for k, v in out.items()}


def compare(a, b):
    """Wasserstein-1 distance per statistic between two pooled sets."""
    return {k: float(wasserstein_distance(a[k], b[k])) if len(a[k]) and len(b[k]) else float("nan") for k in a}
