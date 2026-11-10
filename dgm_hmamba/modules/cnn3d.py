"""3-D CNN encoder for the "w/ CNN" ablation (HDPM replaced by a 3-D CNN).

A residual 3-D CNN in the style of the feature extractors used by DFSL / DCFSL
for few-shot hyperspectral classification: two residual blocks of three 3x3x3
convolutions, spectral-spatial max pooling after each, a final 3x3x3
convolution, average pooling to four spectral cells and a linear map to the
HDPM output width. The diffusion noise map is accepted and ignored, so the
diffusion modulation is absent by construction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Residual3d(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.c1 = nn.Conv3d(c_in, c_out, 3, padding=1)
        self.c2 = nn.Conv3d(c_out, c_out, 3, padding=1)
        self.c3 = nn.Conv3d(c_out, c_out, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = F.relu(self.c1(x))
        return F.relu(x1 + self.c3(F.relu(self.c2(x1))))


class CNN3DEncoder(nn.Module):
    def __init__(self, d_out: int, channels=(16, 32, 64)):
        super().__init__()
        c1, c2, c3 = channels
        self.block1 = _Residual3d(1, c1)
        self.pool1 = nn.MaxPool3d((4, 2, 2), stride=(4, 2, 2), padding=(0, 1, 1))
        self.block2 = _Residual3d(c1, c2)
        self.pool2 = nn.MaxPool3d((4, 2, 2), stride=(4, 2, 2), padding=(2, 1, 1))
        self.conv = nn.Conv3d(c2, c3, 3, padding=1, bias=False)
        self.pool = nn.AdaptiveAvgPool3d((4, 1, 1))
        self.out = nn.Linear(4 * c3, d_out)

    def forward(self, x_hat: torch.Tensor, eps_hat: torch.Tensor = None) -> torch.Tensor:
        """x_hat (B, C, H, W) band-projected patch; returns (B, d_out)."""
        z = x_hat.unsqueeze(1)
        z = self.pool1(self.block1(z))
        z = self.pool2(self.block2(z))
        z = F.relu(self.conv(z))
        return self.out(self.pool(z).flatten(1))
