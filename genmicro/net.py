"""Velocity network for flow matching on micrographs: a fully convolutional U-Net.

* No attention and no positional input, so the network is translation-equivariant and generates images
  of any size (a multiple of 8 px with four levels).
* Residual blocks with channel RMS normalisation and adaLN-zero modulation: the flow time sets a
  per-channel scale, shift and gate; gates start at zero, so every block starts as the identity.
* Lossless resampling (pixel-unshuffle down, pixel-shuffle up) with 1x1 projections; an optional
  patchify stem (patch > 1) runs all convolutions at reduced resolution.
* Extra image channels (`image_cond`), such as an upsampled layout, are concatenated to the input.
* Options explored after the paper (defaults reproduce the paper's network): `block="next"` uses
  ConvNeXt-style blocks (depthwise 7 x 7 convolution, then a pointwise MLP) in place of the two 3 x 3
  convolutions; `attn` > 0 adds that many self-attention blocks (no positional input) at the deepest level.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# The network used throughout the paper: four levels of 64, 128, 256 and 256 channels with 1, 2, 2 and 4
# residual blocks, no patchify stem (13.8 million parameters with one input channel).
UNET = dict(patch=1, widths=(64, 128, 256, 256), depths=(1, 2, 2, 4))


def rms_norm(x, eps=1e-6):
    return x * torch.rsqrt(x.square().mean(1, keepdim=True) + eps)


class Fourier(nn.Module):
    """Fixed sinusoidal features of a scalar in [0, 1]."""

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("freq", math.pi * 2.0 ** torch.linspace(0, 10, dim // 2), persistent=False)

    def forward(self, t):
        a = t[:, None].float() * self.freq
        return torch.cat([a.sin(), a.cos()], 1)


class Block(nn.Module):
    def __init__(self, width, emb_dim):
        super().__init__()
        self.mod = nn.Linear(emb_dim, 3 * width)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)

    def forward(self, x, emb):
        scale, shift, gate = self.mod(emb)[:, :, None, None].chunk(3, 1)
        h = rms_norm(x) * (1 + scale) + shift
        h = self.conv2(F.silu(self.conv1(F.silu(h))))
        return x + gate * h


class NextBlock(nn.Module):
    """ConvNeXt-style residual block: depthwise k x k convolution, modulated RMS norm, pointwise MLP."""

    def __init__(self, width, emb_dim, kernel=7, mult=4):
        super().__init__()
        self.mod = nn.Linear(emb_dim, 3 * width)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)
        self.dw = nn.Conv2d(width, width, kernel, padding=kernel // 2, groups=width)
        self.pw1 = nn.Conv2d(width, mult * width, 1)
        self.pw2 = nn.Conv2d(mult * width, width, 1)

    def forward(self, x, emb):
        scale, shift, gate = self.mod(emb)[:, :, None, None].chunk(3, 1)
        h = rms_norm(self.dw(x)) * (1 + scale) + shift
        return x + gate * self.pw2(F.silu(self.pw1(h)))


class AttnBlock(nn.Module):
    """Multi-head self-attention over all positions, with query/key RMS norm and adaLN-zero modulation."""

    def __init__(self, width, emb_dim, head_dim=64):
        super().__init__()
        self.heads = width // head_dim
        self.mod = nn.Linear(emb_dim, 3 * width)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)
        self.qkv = nn.Conv2d(width, 3 * width, 1)
        self.out = nn.Conv2d(width, width, 1)

    def forward(self, x, emb):
        b, c, hh, ww = x.shape
        scale, shift, gate = self.mod(emb)[:, :, None, None].chunk(3, 1)
        h = rms_norm(x) * (1 + scale) + shift
        q, k, v = self.qkv(h).reshape(b, 3, self.heads, c // self.heads, hh * ww).transpose(-1, -2).unbind(1)
        q, k = rms_norm(q.transpose(1, -1)).transpose(1, -1), rms_norm(k.transpose(1, -1)).transpose(1, -1)
        o = F.scaled_dot_product_attention(q, k, v)
        return x + gate * self.out(o.transpose(-1, -2).reshape(b, c, hh, ww))


class Down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.proj = nn.Conv2d(4 * cin, cout, 1)

    def forward(self, x):
        return self.proj(F.pixel_unshuffle(x, 2))


class Up(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.proj = nn.Conv2d(cin, 4 * cout, 1)

    def forward(self, x):
        return F.pixel_shuffle(self.proj(x), 2)


class FlowUNet(nn.Module):
    def __init__(self, channels=1, widths=(128, 256, 256), depths=(2, 2, 4), patch=2, emb_dim=256, image_cond=0,
                 block="res", attn=0):
        super().__init__()
        Blk = {"res": Block, "next": NextBlock}[block]
        self.patch = patch
        self.time = nn.Sequential(Fourier(128), nn.Linear(128, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.stem = nn.Conv2d((channels + image_cond) * patch**2, widths[0], 3, padding=1)
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        for i, (w, d) in enumerate(zip(widths, depths)):
            last = i == len(widths) - 1
            # The deepest level is the bottleneck: its blocks run once, in the encoder.
            blocks = [Blk(w, emb_dim) for _ in range(d)]
            if last and attn:  # attention blocks spread evenly among the bottleneck blocks
                every = max(1, d // attn)
                blocks = sum(([b, AttnBlock(w, emb_dim)] if (j + 1) % every == 0 and (j + 1) // every <= attn else [b]
                              for j, b in enumerate(blocks)), [])
            self.enc.append(nn.ModuleList(blocks))
            self.dec.append(nn.ModuleList([] if last else [Blk(w, emb_dim) for _ in range(d)]))
            if not last:
                self.down.append(Down(w, widths[i + 1]))
                self.up.append(Up(widths[i + 1], w))
        self.head = nn.Conv2d(widths[0], channels * patch**2, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, t, image=None):
        """x (B, channels, H, W) at flow time t (B,); image (B, image_cond, H, W) or None."""
        if image is not None:
            x = torch.cat([x, image], 1)
        emb = F.silu(self.time(t))
        h = self.stem(F.pixel_unshuffle(x, self.patch))
        skips = []
        for i, blocks in enumerate(self.enc):
            for b in blocks:
                h = b(h, emb)
            if i < len(self.down):
                skips.append(h)
                h = self.down[i](h)
        for i in reversed(range(len(self.up))):
            h = self.up[i](h) + skips.pop()
            for b in self.dec[i]:
                h = b(h, emb)
        return F.pixel_shuffle(self.head(F.silu(rms_norm(h))), self.patch)
