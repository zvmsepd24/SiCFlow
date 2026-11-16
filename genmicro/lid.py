"""Local intrinsic dimension (LID) estimates.

levina_bickel: nearest-neighbour maximum-likelihood estimate (Levina and Bickel 2004) on a set of
points, here descriptor vectors or distances between material states.

flipd: estimate from a trained flow model (the FLIPD idea of Kamkari et al. 2024, written for the
straight path x_t = (1 - t) x0 + t x1 used here). For data blurred by Gaussian noise of standard
deviation sigma, d log p_sigma(x) / d log sigma -> -(D - d) on a d-dimensional set as sigma -> 0, so
    LID(x) = D + sigma^2 (tr grad s + |s|^2),
with s the score of the blurred density at x. At flow time t, x_t / t is the data blurred with
sigma = (1 - t) / t, and the score follows from the velocity v. Substituting gives
    LID(x) = D + (1 - t) (t tr J - D) + t^2 |v(t x, t) - x|^2,   J = dv/dx_t at (t x, t),
with tr J from Hutchinson probes. The estimate depends on sigma: it counts the directions in which
the data spread by more than about sigma near x.
"""
import numpy as np
import torch
from scipy.spatial.distance import cdist


def levina_bickel(x=None, k=10, distances=None):
    """Per-point MLE of intrinsic dimension from the k nearest neighbours ((k - 2) correction).
    Pass points x (n, p) or a precomputed distance matrix (n, n). Points with an exact duplicate in the
    cloud get NaN."""
    d = cdist(x, x) if distances is None else np.array(distances, dtype=float)
    np.fill_diagonal(d, np.inf)
    near = np.sort(d, axis=1)[:, :k]
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.log(near[:, -1:] / near[:, :-1]).sum(1)
    out = np.full(len(d), np.nan)
    ok = (near[:, 0] > 0) & (logs > 0)
    out[ok] = (k - 2) / logs[ok]
    return out


def flipd(model, x, sigma, probes=4, cond=None, generator=None):
    """LID of each image in x (B, C, H, W) under the model's learned distribution, at noise level sigma."""
    model.requires_grad_(False)  # forward-mode only; no backward graph through the weights
    t = 1.0 / (1.0 + sigma)
    D = x[0].numel()
    xt = (t * x).float()
    tt = torch.full((len(x),), t, device=x.device)

    def f(z):
        return model(z, tt, cond).float()

    tr = torch.zeros(len(x), device=x.device)
    with torch.no_grad():
        v = f(xt)
    for _ in range(probes):
        e = torch.randint(0, 2, x.shape, device=x.device, generator=generator).float() * 2 - 1
        _, jv = torch.func.jvp(f, (xt,), (e,))
        tr += (e * jv).flatten(1).sum(1) / probes
    resid = (v - x.float()).flatten(1).square().sum(1)
    return (D + (1 - t) * (t * tr - D) + t * t * resid).cpu()
