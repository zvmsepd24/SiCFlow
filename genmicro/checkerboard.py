"""Checkerboard split of the training band into two statistically equivalent training sets.

Rows 0-2303 and columns 0-2559 of the section are tiled into 18 x 20 tiles of 128 x 128 px. Tiles of one
colour of the checkerboard form set A, the others set B. Each A tile and its neighbour in the same row form
a pair (180 pairs). For replicate r the pairs are put in a random order (seed 1000 + r), and the first
1, 4, 16, 64 and 180 pairs give five nested training areas per set (0.018-3.2 mm^2). Models are trained on
64 x 64 px crops lying inside their tiles.
"""
import numpy as np
import torch
import torch.nn.functional as F

T, K = 128, 64            # tile size, crop size (px)
ROWS, COLS = 18, 20       # tiles in the training band
LEVELS = [1, 4, 16, 64, 180]
STEPS = [2000, 1500, 1500, 1500, 1500]  # training steps added at each area
MM2_PER_TILE = (T * 1.04) ** 2 / 1e6


def pairs():
    """(A tile, B tile) as flat indices i * COLS + j, row by row."""
    out = []
    for i in range(ROWS):
        for k in range(COLS // 2):
            j0, j1 = 2 * k, 2 * k + 1
            a, b = (j0, j1) if (i + j0) % 2 == 0 else (j1, j0)
            out.append((i * COLS + a, i * COLS + b))
    return out


def pair_order(rep):
    return np.random.default_rng(1000 + rep).permutation(ROWS * COLS // 2)


def chosen_tiles(rep, level):
    """Flat tile indices of sets A and B at a training area."""
    p = pairs()
    chosen = [p[i] for i in pair_order(rep)[:LEVELS[level]]]
    return {"A": [a for a, _ in chosen], "B": [b for _, b in chosen]}


def tile_grid(band):
    """band (H, W) tensor -> (ROWS * COLS, T, T) tiles."""
    return band[:ROWS * T, :COLS * T].reshape(ROWS, T, COLS, T).permute(0, 2, 1, 3).reshape(ROWS * COLS, T, T)


def crops_from(tiles, n, g):
    """n random K x K crops lying inside the given tiles (n_tiles, T, T) -> (n, 1, K, K)."""
    idx = torch.randint(0, len(tiles), (n,), device=tiles.device, generator=g)
    oy = torch.randint(0, T - K + 1, (n,), device=tiles.device, generator=g)
    ox = torch.randint(0, T - K + 1, (n,), device=tiles.device, generator=g)
    ar = torch.arange(K, device=tiles.device)
    ys, xs = oy[:, None] + ar[None], ox[:, None] + ar[None]
    return tiles[idx[:, None, None], ys[:, :, None], xs[:, None, :]][:, None]


# windows of the tiles flatter than this (inside large pores, where the gray levels are clipped) are not matched:
# their correlation is undefined
MIN_STD = 0.02


@torch.no_grad()
def match_tiles(x, tiles, chunk=8, min_std=MIN_STD):
    """Largest normalized cross-correlation of each K x K window in x (B, 1, K, K) with any K x K window
    lying inside one of `tiles` (n, T, T), windows of standard deviation below min_std left out;
    returns (similarity, tile index, y, x)."""
    shape = (T + K - 1, T + K - 1)
    chunk = max(1, chunk * 400 // max(len(tiles), 400))  # large sets of tiles in smaller chunks
    r = tiles.float()
    fr = torch.fft.rfft2(r, s=shape)
    ones = torch.ones(1, 1, K, K, device=r.device)
    ws = F.conv2d(r[:, None], ones)[:, 0]
    wq = F.conv2d(r[:, None].square(), ones)[:, 0]
    std = (wq / K ** 2 - (ws / K ** 2).square()).clamp_min(1e-8).sqrt()  # (n, T-K+1, T-K+1)
    flat = std < min_std
    best, arg = [], []
    for i in range(0, len(x), chunk):
        t = x[i:i + chunk, 0].float()
        t = (t - t.mean((1, 2), keepdim=True)) / (t.std((1, 2), keepdim=True, unbiased=False) + 1e-8)
        ft = torch.fft.rfft2(t.flip(-2, -1), s=shape)
        corr = torch.fft.irfft2(fr[None] * ft[:, None], s=shape)[:, :, K - 1:T, K - 1:T] / (K ** 2 * std[None])
        corr = corr.masked_fill(flat[None], -1.0)
        v, j = corr.flatten(1).max(1)
        best.append(v)
        arg.append(j)
    best, arg = torch.cat(best), torch.cat(arg)
    m = T - K + 1
    return best, arg // (m * m), (arg % (m * m)) // m, arg % m


def agreement(a, b):
    """Mean correlation between paired windows of a and b (B, ...) after removing each window's mean."""
    a = torch.as_tensor(a).flatten(1).float()
    b = torch.as_tensor(b).flatten(1).float()
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    return float(((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)).mean())
