"""Lightweight selective-state-space (Mamba-style) block.

This is a clean, pure-PyTorch implementation that captures the essential
selective SSM idea (input-dependent parameters + causal convolution +
state recurrence) without requiring the official mamba-ssm CUDA kernels.
It is sufficient for research reproduction and runs on CPU/GPU.
"""

from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        conv_kernel: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=conv_kernel,
            groups=self.d_inner,
            padding=conv_kernel - 1,
            bias=True,
        )
        self.act = nn.SiLU()

        # Selective parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # dt, B, C
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # A is diagonal and constrained to be negative
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize dt bias for reasonable timescales
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        returns: (B, L, D)
        """
        B, L, _ = x.shape
        residual = x

        xz = self.in_proj(x)                       # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)

        # Causal depthwise conv
        x = x.transpose(1, 2)                      # (B, d_inner, L)
        x = self.conv1d(x)[..., :L]
        x = x.transpose(1, 2)
        x = self.act(x)

        # Selective SSM parameters
        x_dbl = self.x_proj(x)                     # (B, L, 2*d_state + 1)
        dt, B_param, C_param = torch.split(
            x_dbl, [1, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))          # (B, L, d_inner)

        A = -torch.exp(self.A_log.float())         # (d_inner, d_state)

        # Discretize and scan (sequential for clarity / portability)
        y = self._selective_scan(x, dt, A, B_param, C_param)

        y = y * self.act(z)
        out = self.out_proj(y)
        out = self.dropout(out)
        return out + residual

    def _selective_scan(
        self,
        u: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        u, dt : (B, L, d_inner)
        A     : (d_inner, d_state)
        B, C  : (B, L, d_state)
        """
        Bsz, L, D = u.shape
        N = A.shape[1]

        # Discretization
        # deltaA: (B, L, D, N)
        deltaA = torch.exp(dt.unsqueeze(-1) * A)          # broadcast
        deltaB_u = dt.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)

        # Sequential scan
        x = torch.zeros(Bsz, D, N, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(L):
            x = deltaA[:, t] * x + deltaB_u[:, t]
            y_t = (x * C[:, t].unsqueeze(1)).sum(-1)      # (B, D)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)                        # (B, L, D)
        y = y + u * self.D
        return y
