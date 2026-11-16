"""Microstructure descriptors of phase images m (B, H, W) with values 0/1.

two_point:      S2 along x and y and the radial average, from FFT autocorrelation. Periodic images
                use the circular correlation; others are zero padded and divided by the number of
                overlapping pixel pairs at each lag.
chords:         lengths of runs of the phase along rows (x) or columns (y), as a histogram on fixed
                bins, plus the average chord length (phase pixels / number of chords). Runs cut by the
                image edge are kept, so long chords are slightly undercounted in small images.
"""
import torch

CHORD_BINS = torch.tensor([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 1 << 30])


def autocorrelation(m, periodic):
    m = m.float()
    b, h, w = m.shape
    if periodic:
        f = torch.fft.rfft2(m)
        return torch.fft.irfft2(f * f.conj(), s=(h, w)) / (h * w)
    s = (2 * h, 2 * w)
    f = torch.fft.rfft2(m, s=s)
    c = torch.fft.irfft2(f * f.conj(), s=s)
    one = torch.fft.rfft2(torch.ones(1, h, w, device=m.device), s=s)
    n = torch.fft.irfft2(one * one.conj(), s=s).clamp_min(1)
    return c / n


def two_point(m, rmax, periodic=False):
    """Returns S2 along x (B, rmax+1), along y (B, rmax+1) and radially averaged (B, rmax+1), lags 0..rmax."""
    c = autocorrelation(m, periodic)
    sx, sy = c[:, 0, :rmax + 1], c[:, :rmax + 1, 0]
    h, w = c.shape[-2:]
    yy = torch.arange(h, device=m.device)
    xx = torch.arange(w, device=m.device)
    yy = torch.minimum(yy, h - yy)
    xx = torch.minimum(xx, w - xx)
    r = torch.sqrt(yy[:, None].float() ** 2 + xx[None, :].float() ** 2).round().long()
    keep = r <= rmax
    idx = r[keep]
    vals = c[:, keep]
    radial = torch.zeros(len(m), rmax + 1, device=m.device).index_add_(1, idx, vals)
    radial = radial / torch.bincount(idx, minlength=rmax + 1).clamp_min(1)
    return sx, sy, radial


def chords(m, axis):
    """Chord-length histogram (B, len(CHORD_BINS) - 1, fraction of chords) and average chord length (B,)."""
    x = m if axis == "x" else m.transpose(-1, -2)
    x = x.bool()
    b, h, w = x.shape
    pad = torch.zeros(b, h, 1, dtype=torch.bool, device=x.device)
    z = torch.cat([pad, x, pad], 2).flatten()
    d = z[1:].int() - z[:-1].int()
    starts = torch.nonzero(d == 1).flatten()
    ends = torch.nonzero(d == -1).flatten()
    length = ends - starts
    image = starts // (h * (w + 2))
    bins = CHORD_BINS.to(x.device)
    k = torch.bucketize(length, bins[1:-1], right=True)
    nb = len(bins) - 1
    hist = torch.zeros(b * nb, device=x.device).index_add_(0, image * nb + k, torch.ones_like(k, dtype=torch.float))
    hist = hist.view(b, nb)
    count = hist.sum(1)
    acl = x.flatten(1).float().sum(1) / count.clamp_min(1)
    return hist / count.clamp_min(1)[:, None], acl

