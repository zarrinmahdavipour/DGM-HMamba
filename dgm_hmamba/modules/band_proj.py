"""Per-sensor 1x1 band projection to shared C-dimensional space (Sec. 3.1)."""

import torch
import torch.nn as nn


class BandProjection(nn.Module):
    """Learned linear map from raw sensor bands -> shared C channels."""

    def __init__(self, in_bands: int, out_channels: int = 200):
        super().__init__()
        self.proj = nn.Conv2d(in_bands, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, bands, H, W)
        return self.proj(x)
