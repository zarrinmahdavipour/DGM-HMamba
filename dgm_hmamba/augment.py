"""Training-time augmentation for hyperspectral patches (B, bands, H, W)."""

from __future__ import annotations

import torch


def augment_patches(x: torch.Tensor, spectral_scale: float = 0.1, noise_std: float = 0.01) -> torch.Tensor:
    """Random flips / 90-degree rotations, per-sample brightness scaling and
    additive Gaussian noise. Applied to raw (min-max normalised) patches."""
    b = x.shape[0]
    dev = x.device
    # spatial: horizontal / vertical flips and transposition give the 8
    # symmetries of the square together
    flip_h = torch.rand(b, device=dev) < 0.5
    flip_v = torch.rand(b, device=dev) < 0.5
    transp = torch.rand(b, device=dev) < 0.5
    x = torch.where(flip_h.view(-1, 1, 1, 1), x.flip(-1), x)
    x = torch.where(flip_v.view(-1, 1, 1, 1), x.flip(-2), x)
    if x.shape[-1] == x.shape[-2]:
        x = torch.where(transp.view(-1, 1, 1, 1), x.transpose(-1, -2), x)
    # spectral: illumination-like scaling and sensor noise
    if spectral_scale > 0:
        scale = 1.0 + spectral_scale * (2 * torch.rand(b, 1, 1, 1, device=dev) - 1)
        x = x * scale
    if noise_std > 0:
        x = x + noise_std * torch.randn_like(x)
    return x
