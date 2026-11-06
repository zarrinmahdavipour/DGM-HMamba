"""Domain-Adaptive Diffusion-Guided Feature Enhancement (DADGE).

Implements:
  - Forward diffusion process (Eq. forward_diff)
  - Domain discriminator + Gradient Reversal Layer
  - Conditional denoiser eps_theta(X_t, t, c)
  - Reverse sampling (DDPM / DDIM) for enhancement
"""

from typing import Optional, Tuple
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DADGEConfig


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_grl * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, lambda_grl: float = 1.0):
        return GradientReversalFunction.apply(x, lambda_grl)


class DomainDiscriminator(nn.Module):
    """Two-layer MLP: GAP -> C -> 128 -> 1 (PAPER)."""

    def __init__(self, C: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(C, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.grl = GradientReversal()

    def forward(self, x0: torch.Tensor, lambda_grl: float = 1.0):
        """
        x0: (B, C, H, W)
        Returns:
          domain_logit: (B, 1)
          cond: (B, 128)  -- penultimate activation after GRL (used as c)
        """
        h = x0.mean(dim=(2, 3))                    # GAP -> (B, C)
        h = self.grl(h, lambda_grl)
        hidden = F.relu(self.fc1(h))               # (B, 128)
        logit = self.fc2(hidden)
        return logit, hidden


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


class ConditionalDenoiser(nn.Module):
    """Shared denoising network eps_theta(X_t, t, c).

    A lightweight residual U-Net style stack conditioned on timestep and c.
    """

    def __init__(self, C: int, cond_dim: int = 128, base_ch: int = 64, n_blocks: int = 2):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(base_ch, base_ch * 4),
            nn.SiLU(),
            nn.Linear(base_ch * 4, base_ch * 4),
        )
        self.cond_proj = nn.Linear(cond_dim, base_ch * 4)

        self.in_conv = nn.Conv2d(C, base_ch, 3, padding=1)
        self.blocks = nn.ModuleList([ResidualBlock(base_ch) for _ in range(n_blocks)])
        self.mid = ResidualBlock(base_ch)
        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, C, 3, padding=1)
        # Full-rank per-pixel path. The main stack squeezes the C bands
        # through base_ch << C feature maps, so it cannot predict noise that is
        # independent across bands. skip(X_t) / sigma_t can (it learns to remove
        # the low-dimensional spectral signal). It is used only where
        # sigma_t >= skip_min_sigma, which bounds the 1/sigma amplification and
        # keeps the reverse chain stable. Zero-initialised.
        self.skip = nn.Conv2d(C, C, kernel_size=1, bias=False)
        nn.init.zeros_(self.skip.weight)
        self.skip_min_sigma = 0.1

        # FiLM-style modulation from time + condition, initialised to the
        # identity (gamma = 1, beta = 0) so the residual stack starts intact.
        self.film = nn.Linear(base_ch * 4, base_ch * 2)
        with torch.no_grad():
            self.film.weight.zero_()
            self.film.bias.zero_()
            self.film.bias[:base_ch].fill_(1.0)

    def _timestep_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        noise_std: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x_t       : (B, C, H, W)
        t         : (B,) integer timesteps
        c         : (B, cond_dim)
        noise_std : (B,) sqrt(1 - alpha_bar_t), a function of t
        """
        temb = self._timestep_embedding(t, self.time_embed[0].in_features)
        temb = self.time_embed(temb) + self.cond_proj(c)          # (B, 4*base)

        h = self.in_conv(x_t)
        gamma, beta = self.film(temb).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        for blk in self.blocks:
            h = blk(h)
            h = gamma * h + beta
        h = self.mid(h)
        h = F.silu(self.out_norm(h))
        out = self.out_conv(h)
        if noise_std is not None:
            scale = torch.where(noise_std >= self.skip_min_sigma,
                                1.0 / noise_std.clamp(min=self.skip_min_sigma),
                                torch.zeros_like(noise_std))
            out = out + self.skip(x_t) * scale.view(-1, 1, 1, 1)
        return out


class DADGE(nn.Module):
    def __init__(self, cfg: DADGEConfig):
        super().__init__()
        self.cfg = cfg
        self.discriminator = DomainDiscriminator(cfg.C, cfg.disc_hidden)
        self.denoiser = ConditionalDenoiser(
            cfg.C, cfg.cond_dim, cfg.denoiser_base_channels, cfg.denoiser_num_res_blocks
        )
        self._register_schedule()

    def _register_schedule(self):
        T = self.cfg.T
        betas = torch.linspace(self.cfg.beta1, self.cfg.beta_T, T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_ac * x0 + sqrt_om * noise, noise

    def diffusion_loss(self, x0: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        B = x0.shape[0]
        t = torch.randint(0, self.cfg.T, (B,), device=x0.device, dtype=torch.long)
        x_t, noise = self.q_sample(x0, t)
        pred = self.denoiser(x_t, t, c, self.sqrt_one_minus_alphas_cumprod[t])
        return F.mse_loss(pred, noise)

    def adversarial_loss(
        self, x0_source: torch.Tensor, x0_target: torch.Tensor, lambda_grl: float
    ) -> torch.Tensor:
        logit_s, _ = self.discriminator(x0_source, lambda_grl)
        logit_t, _ = self.discriminator(x0_target, lambda_grl)
        # domain label: 0 = source, 1 = target
        loss_s = F.binary_cross_entropy_with_logits(logit_s, torch.zeros_like(logit_s))
        loss_t = F.binary_cross_entropy_with_logits(logit_t, torch.ones_like(logit_t))
        return loss_s + loss_t

    @torch.no_grad()
    def enhance(
        self,
        x0: torch.Tensor,
        num_steps: Optional[int] = None,
        ddim: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run reverse process starting from a partially-noised version of x0
        (as described in the paper) and return (x_hat, eps_hat, c).
        """
        B = x0.shape[0]
        device = x0.device
        steps = num_steps or self.cfg.default_inference_steps
        if steps < self.cfg.T and not ddim:
            warnings.warn(
                f"DDPM sampling with {steps} < T={self.cfg.T} steps skips timesteps with "
                "single-step updates; use ddim=True for accelerated sampling.",
                stacklevel=2,
            )

        # Conditioning vector from discriminator (no GRL at inference)
        _, c = self.discriminator(x0, lambda_grl=0.0)

        # Initialize from partially noised x0 at t = T-1
        t_start = self.cfg.T - 1
        noise = torch.randn_like(x0)
        x = (
            self.sqrt_alphas_cumprod[t_start] * x0
            + self.sqrt_one_minus_alphas_cumprod[t_start] * noise
        )

        times = torch.linspace(t_start, 0, steps, device=device).long()
        eps_hat = None

        for i, t_val in enumerate(times):
            t = torch.full((B,), t_val.item(), device=device, dtype=torch.long)
            eps = self.denoiser(x, t, c, self.sqrt_one_minus_alphas_cumprod[t])
            eps_hat = eps  # keep last prediction map

            alpha = self.alphas[t]
            alpha_bar = self.alphas_cumprod[t]
            # Noise level the step moves to: the next timestep of the (possibly
            # strided) schedule. With steps == T this is t-1, as before.
            t_next = int(times[i + 1].item()) if i + 1 < len(times) else -1
            alpha_bar_prev = (
                self.alphas_cumprod[t_next].expand_as(alpha_bar)
                if t_next >= 0 else torch.ones_like(alpha_bar)
            )

            if ddim:
                # DDIM deterministic step
                x0_pred = (x - torch.sqrt(1 - alpha_bar).view(-1, 1, 1, 1) * eps) / torch.sqrt(
                    alpha_bar
                ).view(-1, 1, 1, 1)
                x = (
                    torch.sqrt(alpha_bar_prev).view(-1, 1, 1, 1) * x0_pred
                    + torch.sqrt(1 - alpha_bar_prev).view(-1, 1, 1, 1) * eps
                )
            else:
                # DDPM step
                coef = (1 - alpha) / torch.sqrt(1 - alpha_bar)
                x = (1 / torch.sqrt(alpha)).view(-1, 1, 1, 1) * (
                    x - coef.view(-1, 1, 1, 1) * eps
                )
                if t_val > 0:
                    noise = torch.randn_like(x)
                    sigma = torch.sqrt(self.posterior_variance[t]).view(-1, 1, 1, 1)
                    x = x + sigma * noise

        return x, eps_hat, c
