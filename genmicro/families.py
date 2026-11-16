"""Other families of generative models for the paired test, trained on the same windows as the flow models.

Each family is a module with
  loss(x, generator)      training loss of a batch x (B, 1, 64, 64) in [-1, 1]
  noise(n, generator)     the latent of n samples (the same latent is given to models A and B)
  generate(z)             windows (B, 1, 64, 64) in [-1, 1] from a latent, deterministically
so that copying, convergence, fidelity and same-noise agreement are measured as for the flow models.

ddpm   Denoising diffusion (Ho et al. 2020): the U-Net of the flow models predicts the noise, linear beta
       schedule 1e-4 to 0.02 over 1000 steps, unweighted noise loss; sampled with deterministic DDIM (Song et
       al. 2021), 64 steps, so that the noise fixes the window as for the flow models.
vae    Variational autoencoder (Kingma and Welling 2014) with a Gaussian decoder of one learned variance, set
       to its optimum on every batch (sigma-VAE, Rybkin et al. 2021); 256-dimensional latent; convolutional
       encoder and decoder with residual blocks. Samples: decoder mean of z ~ N(0, I).
glow   Normalizing flow (Glow, Kingma and Dhariwal 2018; a descendant of RealNVP, Dinh et al. 2017): three
       levels of squeeze, K steps of actnorm, invertible 1x1 convolution and affine coupling, and a split with
       a learned conditional prior; affine couplings with a scale exp(tanh(.)), as in RealNVP, so that the inverse
       stays finite. Trained by maximum likelihood on dequantized 8-bit gray levels. Samples at
       temperature 1 (the model's own distribution) unless given.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from genmicro.net import UNET, FlowUNet

# ----------------------------------------------------------------------------------------------- DDPM


class DDPM(nn.Module):
    T = 1000

    def __init__(self, sample_steps=64, **unet):
        super().__init__()
        self.net = FlowUNet(**(unet or UNET))
        beta = torch.linspace(1e-4, 0.02, self.T, dtype=torch.float64)
        self.register_buffer("abar", torch.cumprod(1 - beta, 0).float())
        self.sample_steps = sample_steps

    def eps(self, x, t_index, net=None):
        # network time input: 1 at the data, 0 at the noise, as for the flow models
        return (self.net if net is None else net)(x, 1 - (t_index.float() + 1) / self.T).float()

    def loss(self, x, generator=None, net=None):
        n = len(x)
        t = torch.randint(0, self.T, (n,), device=x.device, generator=generator)
        e = torch.randn(x.shape, device=x.device, generator=generator)
        a = self.abar[t][:, None, None, None]
        return F.mse_loss(self.eps(a.sqrt() * x + (1 - a).sqrt() * e, t, net), e)

    def noise(self, n, generator):
        return torch.randn(n, 1, 64, 64, device="cuda", generator=generator)

    @torch.no_grad()
    def generate(self, z):
        ts = torch.linspace(self.T - 1, 0, self.sample_steps, device=z.device).round().long()
        x = z
        for i, t in enumerate(ts):
            a = self.abar[t]
            a_prev = self.abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=z.device)
            e = self.eps(x, t.expand(len(x)))
            x0 = ((x - (1 - a).sqrt() * e) / a.sqrt()).clamp(-1, 1)
            e = (x - a.sqrt() * x0) / (1 - a).sqrt()
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * e
        return x


# ------------------------------------------------------------------------------------------------ VAE


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.f = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(8, c), nn.SiLU(),
                               nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x):
        return x + self.f(x)


class VAE(nn.Module):
    def __init__(self, latent=256, widths=(32, 64, 128, 256, 256), blocks=1):
        super().__init__()
        self.latent = latent
        enc = [nn.Conv2d(1, widths[0], 3, padding=1)]
        for i, w in enumerate(widths):
            enc += [Res(w) for _ in range(blocks)]
            if i + 1 < len(widths):
                enc.append(nn.Conv2d(w, widths[i + 1], 4, stride=2, padding=1))
        self.enc = nn.Sequential(*enc, nn.GroupNorm(8, widths[-1]), nn.SiLU())
        side = 64 // 2 ** (len(widths) - 1)
        self.side, self.top = side, widths[-1]
        self.to_z = nn.Linear(widths[-1] * side * side, 2 * latent)
        self.from_z = nn.Linear(latent, widths[-1] * side * side)
        dec = []
        for i, w in enumerate(reversed(widths)):
            dec += [Res(w) for _ in range(blocks)]
            if i + 1 < len(widths):
                dec.append(nn.ConvTranspose2d(w, list(reversed(widths))[i + 1], 4, stride=2, padding=1))
        self.dec = nn.Sequential(*dec, nn.GroupNorm(8, widths[0]), nn.SiLU(), nn.Conv2d(widths[0], 1, 3, padding=1))

    def decode(self, z):
        return self.dec(self.from_z(z).view(len(z), self.top, self.side, self.side))

    def loss(self, x, generator=None):
        mu, logvar = self.to_z(self.enc(x).flatten(1)).float().chunk(2, 1)
        logvar = logvar.clamp(-12, 8)
        z = mu + (0.5 * logvar).exp() * torch.randn(mu.shape, device=x.device, generator=generator)
        xh = self.decode(z).float()
        mse = (x.float() - xh).square().mean()
        log_sigma = 0.5 * mse.detach().clamp_min(1e-6).log()  # optimal shared variance (sigma-VAE)
        D = x[0].numel()
        rec = 0.5 * ((x.float() - xh).square().flatten(1).sum(1) / (2 * log_sigma).exp()) + D * log_sigma
        kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(1)
        return (rec + kl).mean() / D

    def noise(self, n, generator):
        return torch.randn(n, self.latent, device="cuda", generator=generator)

    @torch.no_grad()
    def generate(self, z):
        return self.decode(z).float().clamp(-1, 1)


# ----------------------------------------------------------------------------------------------- Glow


class ActNorm(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.logs = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.register_buffer("ready", torch.tensor(False))

    def forward(self, x):
        if not self.ready:  # data-dependent initialization on the first batch
            with torch.no_grad():
                self.loc.copy_(-x.mean((0, 2, 3), keepdim=True))
                self.logs.copy_(-x.std((0, 2, 3), keepdim=True).clamp_min(1e-6).log())
                self.ready.fill_(True)
        return (x + self.loc) * self.logs.exp(), self.logs.sum() * x.shape[2] * x.shape[3]

    def reverse(self, y):
        return y * (-self.logs).exp() - self.loc


class InvConv(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w = nn.Parameter(torch.linalg.qr(torch.randn(c, c))[0])

    def forward(self, x):
        return F.conv2d(x, self.w[:, :, None, None]), torch.slogdet(self.w)[1] * x.shape[2] * x.shape[3]

    def reverse(self, y):
        return F.conv2d(y, torch.inverse(self.w)[:, :, None, None])


class ZeroConv(nn.Conv2d):
    def __init__(self, cin, cout):
        super().__init__(cin, cout, 3, padding=1)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)
        self.scale = nn.Parameter(torch.zeros(1, cout, 1, 1))

    def forward(self, x):
        return super().forward(x) * (3 * self.scale).exp()


class Coupling(nn.Module):
    def __init__(self, c, width):
        super().__init__()
        self.nn = nn.Sequential(nn.Conv2d(c // 2, width, 3, padding=1), nn.ReLU(), nn.Conv2d(width, width, 1), nn.ReLU(),
                                ZeroConv(width, c))

    def _st(self, xa):
        with torch.autocast("cuda", torch.bfloat16):  # the coupling network only; the flow itself is float32
            h = self.nn(xa).float()
        shift, raw = h[:, 0::2], h[:, 1::2]
        return shift, torch.tanh(raw).exp()  # scale bounded to [1/e, e], so that the inverse stays finite

    def forward(self, x):
        xa, xb = x.chunk(2, 1)
        shift, s = self._st(xa)
        return torch.cat([xa, (xb + shift) * s], 1), s.log().flatten(1).sum(1)

    def reverse(self, y):
        ya, yb = y.chunk(2, 1)
        shift, s = self._st(ya)
        return torch.cat([ya, yb / s - shift], 1)


class FlowStep(nn.Module):
    def __init__(self, c, width):
        super().__init__()
        self.layers = nn.ModuleList([ActNorm(c), InvConv(c), Coupling(c, width)])

    def forward(self, x):
        total = 0
        for layer in self.layers:
            x, ld = layer(x)
            total = total + ld
        return x, total

    def reverse(self, y):
        for layer in reversed(self.layers):
            y = layer.reverse(y)
        return y


def squeeze(x):
    b, c, h, w = x.shape
    return x.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, 4 * c, h // 2, w // 2)


def unsqueeze(x):
    b, c, h, w = x.shape
    return x.view(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)


LOG2PI = math.log(2 * math.pi)


def gauss_logp(x, mean, logs):
    return (-0.5 * (LOG2PI + 2 * logs + (x - mean).square() * (-2 * logs).exp())).flatten(1).sum(1)


class Glow(nn.Module):
    def __init__(self, levels=3, steps=12, width=384, temperature=1.0, checkpoint=False):
        super().__init__()
        self.levels, self.temperature = levels, temperature
        self.checkpoint = checkpoint  # recompute flow steps in the backward pass (for wide networks)
        self.flows, self.priors = nn.ModuleList(), nn.ModuleList()
        c = 1
        for i in range(levels):
            c *= 4
            self.flows.append(nn.ModuleList([FlowStep(c, width) for _ in range(steps)]))
            last = i == levels - 1
            # prior of the factored-out half (from the half that continues), or of everything at the top
            self.priors.append(ZeroConv(c // 2, c) if not last else None)
            if not last:
                c //= 2
        self.top = nn.Parameter(torch.zeros(1, 2 * c, 1, 1))  # learned mean and log-scale of the top latent
        self.c_top = c

    def encode(self, x):
        """x in data space -> (latents, log p(x))."""
        logp = torch.zeros(len(x), device=x.device)
        zs, h = [], x
        for i in range(self.levels):
            h = squeeze(h)
            for step in self.flows[i]:
                h, ld = checkpoint(step, h, use_reentrant=False) if self.checkpoint and torch.is_grad_enabled() else step(h)
                logp = logp + ld
            if i < self.levels - 1:
                h, z = h.chunk(2, 1)
                mean, logs = self.priors[i](h).chunk(2, 1)
                logs = logs.clamp(-6, 3)
                logp = logp + gauss_logp(z, mean, logs)
                zs.append((z - mean) * (-logs).exp())
            else:
                mean, logs = self.top.expand(len(h), -1, *h.shape[2:]).chunk(2, 1)
                logp = logp + gauss_logp(h, mean, logs)
                zs.append((h - mean) * (-logs).exp())
        return zs, logp

    def loss(self, x, generator=None):
        """Bits per dimension of 8-bit gray levels, uniformly dequantized."""
        with torch.autocast("cuda", enabled=False):
            y = self.dequantize(x, generator)
            _, logp = self.encode(y)
            D = x[0].numel()
            return (-(logp / D) + math.log(256)).mean() / math.log(2)

    def dequantize(self, x, generator=None):
        levels = ((x.float() + 1) * 127.5).round()
        u = torch.rand(x.shape, device=x.device, generator=generator)
        return (levels + u) / 256 - 0.5

    @torch.no_grad()
    @torch.autocast("cuda", enabled=False)
    def bits_per_dim(self, x, repeats=1, generator=None):
        """Per window, averaged over dequantization noise."""
        D = x[0].numel()
        out = 0
        for _ in range(repeats):
            _, logp = self.encode(self.dequantize(x, generator))
            out = out + (-(logp / D) + math.log(256)) / math.log(2)
        return out / repeats

    def noise(self, n, generator):
        shapes, c, s = [], 1, 64
        for i in range(self.levels):
            c, s = c * 4, s // 2
            if i < self.levels - 1:
                shapes.append((c // 2, s, s))
                c //= 2
            else:
                shapes.append((c, s, s))
        return [torch.randn(n, *sh, device="cuda", generator=generator) for sh in shapes]

    @torch.no_grad()
    @torch.autocast("cuda", enabled=False)
    def generate(self, z, temperature=None):
        T = self.temperature if temperature is None else temperature
        mean, logs = self.top.expand(len(z[-1]), -1, *z[-1].shape[2:]).chunk(2, 1)
        h = mean + T * logs.exp() * z[-1]
        for i in reversed(range(self.levels)):
            if i < self.levels - 1:
                mean, logs = self.priors[i](h).chunk(2, 1)
                h = torch.cat([h, mean + T * logs.clamp(-6, 3).exp() * z[i]], 1)
            for step in reversed(self.flows[i]):
                h = step.reverse(h)
            h = unsqueeze(h)
        return (2 * h).clamp(-1, 1)


FAMILIES = {"ddpm": DDPM, "vae": VAE, "glow": Glow}


def build(family, **kw):
    return FAMILIES[family](**kw)
