"""Two-scale model: a layout model at one-eighth resolution and a fiber model conditioned on the layout.

Layout of a region (fraction_fields): the indicators of pixels brighter than gray level 183 and darker than
gray level 89, blurred with a Gaussian of 6 px standard deviation (about half a fiber diameter, so single
fibers disappear), average-pooled 8x and scaled to [-1, 1]; two channels at 8.3 um/px. The dark indicator
includes the dark rims around fibers as well as pores, so the two channels together describe how densely
fibers are packed and where the large pores are.
The layout model is an unconditional FlowUNet on 64 px layout crops (0.53 mm), trained with random
horizontal and vertical flips. The fiber model is a FlowUNet on full-resolution crops that receives the
layout, upsampled bilinearly, as extra input channels (ImageConditioned). A variant trained with the
layout dropped for 10% of the crops can be sampled with classifier-free guidance (Guided).
"""
import numpy as np
import torch
import torch.nn.functional as F

from .train import RHO, STEPS, sample

DOWN = 8
SIGMA = 6.0
BRIGHT, DARK = 183 / 127.5 - 1, 89 / 127.5 - 1  # gray levels 183 and 89 in [-1, 1] units
LAYOUT_DROPOUT = 0.1


class ImageConditioned(torch.nn.Module):
    """Passes `cond` to the network as extra image channels."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, t, cond=None):
        return self.net(x, t, image=cond)


class Mixed(torch.nn.Module):
    """Velocity mixed from a fiber model given the layout and an unconditional single-scale model:
    v = lam * v_fiber(x | layout) + (1 - lam) * v_single(x)."""

    def __init__(self, fine, single, lam):
        super().__init__()
        self.fine, self.single, self.lam = fine, single, lam

    def forward(self, x, t, cond=None):
        return self.lam * self.fine(x, t, cond) + (1 - self.lam) * self.single(x, t)


class Guided(torch.nn.Module):
    """Classifier-free guidance for a fiber model trained with layout dropout.

    The network receives the layout and a channel of ones (layout present) or zeros everywhere (dropped).
    """

    def __init__(self, net, w):
        super().__init__()
        self.net, self.w = net, w

    def forward(self, x, t, cond=None):
        on = torch.cat([cond, torch.ones_like(cond[:, :1])], 1)
        v1 = self.net(x, t, image=on)
        if self.w == 1:
            return v1
        v0 = self.net(x, t, image=torch.zeros_like(on))
        return v0 + self.w * (v1 - v0)


def gaussian_blur(x, sigma):
    r = int(3 * sigma)
    k = torch.exp(-torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    c = x.shape[1]
    x = F.pad(x, (r, r, r, r), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(x, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def fraction_fields(img):
    """img (H, W) or (B, 1, H, W) in [-1, 1] -> (B, 2, H/8, W/8) layout in [-1, 1]."""
    x = img if img.dim() == 4 else img[None, None]
    ind = torch.cat([(x > BRIGHT).float(), (x < DARK).float()], 1)
    return F.avg_pool2d(gaussian_blur(ind, SIGMA), DOWN) * 2 - 1


STATE_SIGMA = 12.0


def state_fields(img):
    """State layout: local fiber and pore fractions. img (H, W) in [-1, 1] (array or tensor) -> (1, 2, H/8, W/8).

    The image is segmented (genmicro.phases); the fiber and pore indicators are blurred with a Gaussian of 12 px
    standard deviation, which removes the positions of single fibers (spacing about 17 px) but keeps the tow
    bands and the matrix-rich layers between them, average-pooled 8x and scaled to [-1, 1].
    """
    from .phases import phases
    a = img.detach().float().cpu().numpy() if torch.is_tensor(img) else img
    ph = phases(a.squeeze())
    ind = torch.as_tensor(np.stack([ph == 1, ph == 2]).astype(np.float32))[None]
    if torch.cuda.is_available():
        ind = ind.cuda()
    return F.avg_pool2d(gaussian_blur(ind, STATE_SIGMA), DOWN) * 2 - 1


def gray_layout(x):
    """Alternative layout: the gray-level image average-pooled 8x and upsampled again (one channel)."""
    return F.interpolate(F.avg_pool2d(x, DOWN), scale_factor=DOWN, mode="bilinear", align_corners=False)


def upsample(lay):
    return F.interpolate(lay, scale_factor=DOWN, mode="bilinear", align_corners=False)


COND_NOISE_MAX = 1.0  # conditioning augmentation: noise levels drawn uniformly from [0, 1] in training
COND_NOISE = 0.5  # noise level of the layout when sampling a fiber model trained with conditioning augmentation


def noisy_condition(lay, level, generator=None, blur=0.0):
    """Conditioning augmentation (Ho et al. 2022): Gaussian noise of standard deviation `level` added to the layout at
    its own resolution, then upsampled, with the level as an extra constant channel.

    A layout computed from the training band is unique to its location, so a fiber model conditioned on the exact
    layout can learn to return the training image for each layout. Noise makes many locations share a condition.
    blur > 0: the noisy layout is blurred with a Gaussian of `blur` layout pixels, so that the condition carries no
    structure at the scale of single fibers (noise at layout resolution, upsampled, makes fiber-sized blobs).
    lay (B, C, h, w) in [-1, 1]; level float or (B,) -> (B, C + 1, 8h, 8w).
    """
    B = lay.shape[0]
    lev = torch.as_tensor(level, device=lay.device, dtype=lay.dtype).reshape(-1, 1, 1, 1).expand(B, 1, 1, 1)
    noisy = lay + lev * torch.randn(lay.shape, device=lay.device, generator=generator, dtype=lay.dtype)
    if blur > 0:
        noisy = gaussian_blur(noisy, blur)
    up = upsample(noisy)
    return torch.cat([up, lev.expand(B, 1, *up.shape[-2:])], 1)


def flips(x, g):
    """Random horizontal and vertical flips (the tow bands stay horizontal)."""
    f = torch.rand(2, len(x), 1, 1, 1, device=x.device, generator=g) < 0.5
    x = torch.where(f[0], x.flip(-1), x)
    return torch.where(f[1], x.flip(-2), x)


def strip(state):
    """State dict without the prefixes added by torch.compile."""
    return {k.replace("_orig_mod.", ""): v for k, v in state.items()}


def unwrap(state):
    """State dict of the FlowUNet inside an ImageConditioned or Guided wrapper."""
    return {k[4:] if k.startswith("net.") else k: v for k, v in strip(state).items()}


LAYOUT_STEPS, LAYOUT_RHO = STEPS, RHO  # the sampler of every model: 32 uniform steps
FIBER_STEPS, FIBER_RHO = STEPS, RHO


def condition(lay, cond_noise=None, generator=None, cond_blur=0.0):
    """Condition of a fiber model from a layout at layout resolution: upsampled, or with conditioning augmentation."""
    return upsample(lay) if cond_noise is None else noisy_condition(lay, cond_noise, generator, cond_blur)


@torch.no_grad()
def generate(coarse, fine, size, generator, channels=2, cond_noise=None, cond_blur=0.0):
    """One field of size x size px: a layout from `coarse`, upsampled, then the image from `fine`.

    Both models are sampled with 32 uniform steps. With fewer steps, or steps concentrated near the noise,
    generated layouts are smoother than real ones, and the fiber model then draws smaller and fewer fibers.
    cond_noise: noise level of the layout for fiber models trained with conditioning augmentation.
    """
    with torch.autocast("cuda", torch.bfloat16):
        z = torch.randn(1, channels, size // DOWN, size // DOWN, device="cuda", generator=generator)
        lay = sample(coarse, z, LAYOUT_STEPS, rho=LAYOUT_RHO).float()
        cond = condition(lay, cond_noise, generator, cond_blur)
        x = sample(fine, torch.randn(1, 1, size, size, device="cuda", generator=generator), FIBER_STEPS, cond=cond, rho=FIBER_RHO)
    return x.float()[0, 0]
