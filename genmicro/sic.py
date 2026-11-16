"""SiC section data: one plane at 1.04 um/px, split into disjoint column bands, cropped on the GPU."""
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PLANE = ROOT / "data/sic/section_plane.npy"


def load_plane(device="cuda"):
    """Plane scaled to [-1, 1], shape (H, W)."""
    a = np.load(PLANE).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(a).to(device)


def load_section(s, device="cuda"):
    """Section s of the S200 series in the frame of the training section (section 10), scaled to [-1, 1].

    Section 10 is the training section itself; the others are stitched and registered to it
    (scripts/stitch_section.py, scripts/register_sections.py)."""
    if s == 10:
        return load_plane(device)
    a = np.load(ROOT / f"data/sic/sections/section{s:02d}.npy").astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(a).to(device)


def column_bands(plane, edges):
    """Split the plane into column bands [edges[i], edges[i+1]); a gap is any band left unused."""
    return [plane[:, a:b] for a, b in zip(edges[:-1], edges[1:])]


class CropSampler:
    """Uniform random crops from a region, drawn on the GPU.

    With `positions` given, crops come only from that fixed set of top-left corners
    (a finite training set); otherwise every valid position is allowed.
    """

    def __init__(self, region, size, positions=None, generator=None):
        self.region, self.size, self.gen = region, size, generator
        h, w = region.shape
        self.max_y, self.max_x = h - size + 1, w - size + 1
        self.positions = None if positions is None else torch.as_tensor(positions, device=region.device)
        offs = torch.arange(size, device=region.device)
        self.dy, self.dx = offs[:, None], offs[None, :]

    def random_positions(self, n):
        y = torch.randint(0, self.max_y, (n,), device=self.region.device, generator=self.gen)
        x = torch.randint(0, self.max_x, (n,), device=self.region.device, generator=self.gen)
        return torch.stack([y, x], 1)

    def crops_at(self, pos):
        ys = pos[:, 0, None, None] + self.dy
        xs = pos[:, 1, None, None] + self.dx
        return self.region[ys, xs].unsqueeze(1)

    def __call__(self, n):
        if self.positions is None:
            pos = self.random_positions(n)
        else:
            idx = torch.randint(0, len(self.positions), (n,), device=self.region.device, generator=self.gen)
            pos = self.positions[idx]
        return self.crops_at(pos)
