"""Hierarchical Dual-Path Mamba Encoder (HDPM), Sec. 3.3."""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

from .mamba_block import MambaBlock
from ..config import HDPMConfig


class ModulationHead(nn.Module):
    """Projects pooled noise map to (gamma, beta) for each block."""

    def __init__(self, c_in: int, d_level: int):
        super().__init__()
        self.proj = nn.Linear(c_in, 2 * d_level)
        # Eq. (modulation): u' = gamma * u + beta. Initialise the bias so that
        # gamma starts at 1 and beta at 0; with PyTorch's default bias init
        # gamma would start near 0 and every block would scale its input
        # down to almost nothing.
        with torch.no_grad():
            self.proj.bias.zero_()
            self.proj.bias[:d_level].fill_(1.0)

    def forward(self, eps_bar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma, beta = self.proj(eps_bar).chunk(2, dim=-1)
        return gamma, beta


class CrossPathInteraction(nn.Module):
    """Cross-attention fusion between spectral summary and spatial path."""

    def __init__(self, d_level: int):
        super().__init__()
        self.wq = nn.Linear(d_level, d_level, bias=False)
        self.wk = nn.Linear(d_level, d_level, bias=False)
        self.wv = nn.Linear(d_level, d_level, bias=False)
        self.scale = d_level ** -0.5

    def forward(self, spec_summary: torch.Tensor, spat: torch.Tensor) -> torch.Tensor:
        q = self.wq(spec_summary)
        k = self.wk(spat)
        v = self.wv(spat)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        return attn @ v


class LevelPaths(nn.Module):
    def __init__(
        self,
        c_bands: int,
        d_level: int,
        blocks_per_level: int,
        state_dim: int,
        expand_factor: int,
        conv_kernel: int,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        self.spectral_blocks = nn.ModuleList(
            [MambaBlock(d_level, state_dim, expand_factor, conv_kernel) for _ in range(blocks_per_level)]
        )
        self.spatial_blocks = nn.ModuleList(
            [MambaBlock(d_level, state_dim, expand_factor, conv_kernel) for _ in range(blocks_per_level)]
        )
        self.spec_mod = nn.ModuleList(
            [ModulationHead(c_bands, d_level) for _ in range(blocks_per_level)]
        )
        self.spat_mod = nn.ModuleList(
            [ModulationHead(c_bands, d_level) for _ in range(blocks_per_level)]
        )
        self.cpi = CrossPathInteraction(d_level)

    def forward(self, spec_seq: torch.Tensor, eps_bar: torch.Tensor):
        """
        spec_seq : (B, N_l, C, d)
        eps_bar  : (B, C)
        Returns fused (B, N_l, d), updated spec_seq (B, N_l, C, d)
        """
        B, N_l, C, d = spec_seq.shape

        # Spectral path: each spatial location is a length-C sequence
        spec = spec_seq.reshape(B * N_l, C, d)
        for blk, mod in zip(self.spectral_blocks, self.spec_mod):
            gamma, beta = mod(eps_bar)  # (B, d)
            gamma = gamma.unsqueeze(1).expand(B, N_l, d).reshape(B * N_l, 1, d)
            beta = beta.unsqueeze(1).expand(B, N_l, d).reshape(B * N_l, 1, d)
            spec = gamma * spec + beta
            spec = self._run(blk, spec)
        spec = spec.reshape(B, N_l, C, d)
        spec_summary = spec.mean(dim=2)  # (B, N_l, d)

        # Spatial path: treat the N_l locations as a sequence
        spat = spec_summary
        for blk, mod in zip(self.spatial_blocks, self.spat_mod):
            gamma, beta = mod(eps_bar)
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
            spat = gamma * spat + beta
            spat = self._run(blk, spat)

        fused = self.cpi(spec_summary, spat) + spat
        return fused, spec

    def _run(self, blk, x):
        if self.grad_checkpoint and torch.is_grad_enabled() and x.requires_grad:
            return checkpoint(blk, x, use_reentrant=False)
        return blk(x)


class PatchMerge(nn.Module):
    """Simple spatial 2x2 average-pool + linear channel projection."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)

    def forward(self, spec_seq: torch.Tensor) -> torch.Tensor:
        # spec_seq: (B, N, C, d_in) with N = H*W assumed square
        B, N, C, d = spec_seq.shape
        side = int(round(N ** 0.5))
        x = spec_seq.reshape(B, side, side, C, d)
        # Adaptive average pool handles odd spatial sizes
        x = x.permute(0, 3, 4, 1, 2)          # (B, C, d, H, W)
        x = x.reshape(B * C, d, side, side)
        out_side = max(1, side // 2)
        x = F.adaptive_avg_pool2d(x, (out_side, out_side))
        x = x.reshape(B, C, d, out_side, out_side)
        x = x.permute(0, 3, 4, 1, 2).reshape(B, out_side * out_side, C, d)
        return self.proj(x)


class HDPM(nn.Module):
    def __init__(self, cfg: HDPMConfig, c_bands: int = 200):
        super().__init__()
        self.cfg = cfg
        self.c_bands = c_bands

        # 3D-style patch embedding: treat bands as depth
        self.patch_embed = nn.Conv3d(
            1, cfg.widths[0], kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)
        )

        self.levels = nn.ModuleList()
        for i, d in enumerate(cfg.widths):
            self.levels.append(
                LevelPaths(
                    c_bands,
                    d,
                    cfg.blocks_per_level,
                    cfg.state_dim,
                    cfg.expand_factor,
                    cfg.conv_kernel,
                    cfg.grad_checkpoint,
                )
            )

        self.merges = nn.ModuleList(
            [PatchMerge(cfg.widths[i], cfg.widths[i + 1]) for i in range(len(cfg.widths) - 1)]
        )
        self.fused_proj = nn.ModuleList(
            [nn.Linear(cfg.widths[i], cfg.widths[i + 1]) for i in range(len(cfg.widths) - 1)]
        )

    def forward(self, x_hat: torch.Tensor, eps_hat: torch.Tensor) -> torch.Tensor:
        """
        x_hat, eps_hat : (B, C, H, W)
        Returns f : (B, d3)
        """
        B, C, H, W = x_hat.shape
        eps_bar = eps_hat.mean(dim=(2, 3))  # (B, C)

        # Embed: (B, 1, C, H, W) -> (B, d1, C, H, W)
        tok = self.patch_embed(x_hat.unsqueeze(1))
        # -> (B, N, C, d1)
        spec_seq = tok.permute(0, 3, 4, 2, 1).reshape(B, H * W, C, -1)

        fused = None
        for i, level in enumerate(self.levels):
            fused, spec_seq_out = level(spec_seq, eps_bar)
            if i < len(self.merges):
                spec_seq = self.merges[i](spec_seq_out)
                # residual from previous fused summary
                fused_summary = self.fused_proj[i](fused.mean(dim=1))  # (B, d_{i+1})
                spec_seq = spec_seq + fused_summary.unsqueeze(1).unsqueeze(2)

        f = fused.mean(dim=1)  # (B, d3)
        return f
