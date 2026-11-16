"""Image-level comparison of ensembles of windows, and loading of trained models.

Windows are (B, 1, s, s) tensors in [-1, 1]. Features are 2048-dimensional Inception-v3 pool features
(torch-fidelity weights, as used for the FID) or DINOv2 ViT-B/14 class tokens (images resized to 224 px).
Ensembles are compared by the Fréchet distance between Gaussian fits to their feature vectors and by the
kernel inception distance (KID, cubic polynomial kernel, reported x 1000).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg

from .net import FlowUNet

ROOT = Path(__file__).resolve().parents[1]
_EXTRACTORS = {}


def _extractor(kind):
    if kind not in _EXTRACTORS:
        if kind == "inception":
            from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
            _EXTRACTORS[kind] = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"]).cuda().eval()
        elif kind == "dinov2":
            _EXTRACTORS[kind] = torch.hub.load("facebookresearch/dinov2:main", "dinov2_vitb14", trust_repo=True).cuda().eval()
        else:
            raise ValueError(kind)
    return _EXTRACTORS[kind]


@torch.no_grad()
def features(x, kind="inception", batch=64):
    """x: (B, 1, s, s) in [-1, 1] -> (B, D) float64 features."""
    fe, out = _extractor(kind), []
    for i in range(0, len(x), batch):
        b = torch.as_tensor(x[i:i + batch]).float().cuda()
        if kind == "inception":
            u8 = ((b.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).repeat(1, 3, 1, 1)
            out.append(fe(u8)[0].double().cpu())
        else:
            im = F.interpolate((b.clamp(-1, 1) + 1) / 2, size=224, mode="bicubic", align_corners=False).repeat(1, 3, 1, 1)
            mean = torch.tensor([0.485, 0.456, 0.406], device=im.device)[:, None, None]
            std = torch.tensor([0.229, 0.224, 0.225], device=im.device)[:, None, None]
            out.append(fe((im - mean) / std).double().cpu())
    return torch.cat(out).numpy()


def frechet(a, b):
    """Fréchet distance between Gaussian fits to the rows of a and b."""
    m1, m2 = a.mean(0), b.mean(0)
    c1, c2 = np.atleast_2d(np.cov(a, rowvar=False)), np.atleast_2d(np.cov(b, rowvar=False))
    s = linalg.sqrtm(c1 @ c2)
    return float(((m1 - m2) ** 2).sum() + np.trace(c1 + c2 - 2 * s.real))


def kid(a, b, subsets=50, size=200, seed=0):
    """Kernel inception distance x 1000, averaged over random subsets."""
    rng = np.random.default_rng(seed)
    d = a.shape[1]
    k = lambda x, y: (x @ y.T / d + 1) ** 3
    vals = []
    for _ in range(subsets):
        x = a[rng.choice(len(a), min(size, len(a)), replace=False)]
        y = b[rng.choice(len(b), min(size, len(b)), replace=False)]
        m = len(x)
        kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean())
    return float(np.mean(vals)) * 1000


def load_model(path):
    """FlowUNet with the EMA weights of a checkpoint (path relative to the repository root)."""
    ck = torch.load(ROOT / path, map_location="cuda")
    net = FlowUNet(**ck["model_config"]).cuda()
    net.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["ema"].items()})
    return net.eval()
