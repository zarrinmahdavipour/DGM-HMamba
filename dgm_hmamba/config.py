"""Central configuration for DGM-HMamba.

Fields tagged PAPER are values taken directly from the manuscript.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DADGEConfig:
    C: int = 200                  # PAPER: shared band-projected channels
    T: int = 50                   # PAPER: diffusion timesteps
    beta1: float = 1e-4           # PAPER
    beta_T: float = 0.02          # PAPER
    cond_dim: int = 128           # PAPER: conditioning vector dim
    disc_hidden: int = 128        # PAPER: discriminator C -> 128 -> 1
    grl_warmup_epochs: int = 30   # PAPER: lambda_grl 0 -> 1

    denoiser_base_channels: int = 64
    denoiser_num_res_blocks: int = 2
    default_inference_steps: int = 50
    # Ablation "w/o Diff": skip the reverse diffusion and feed X_0 (with a
    # zero noise map) to HDPM. The denoiser is still trained with L_diff.
    bypass_enhancement: bool = False
    # Use the deterministic DDIM update (no fresh noise per step) for the
    # enhancement reverse chain instead of ancestral DDPM sampling.
    deterministic_sampling: bool = False
    # Pass the gradient of the downstream losses (L_cls, L_epi) through the
    # enhancement to the band projections with a straight-through estimator
    # (identity Jacobian), so the per-sensor projections are trained by
    # classification and not only by L_adv. DADGE parameters are unaffected.
    straight_through: bool = True


@dataclass
class HDPMConfig:
    state_dim: int = 16                       # PAPER: Mamba state n=16
    levels: Tuple[int, int, int] = (81, 25, 9) # PAPER: approx token counts
    widths: Tuple[int, int, int] = (256, 512, 1024)  # PAPER: d1,d2,d3
    blocks_per_level: int = 2                 # PAPER: L_ell = 2
    patch_size: int = 9                       # PAPER: 9x9 patches

    expand_factor: int = 2
    conv_kernel: int = 4
    # Ablation "w/ CNN": "cnn" replaces HDPM by a residual 3-D CNN
    # (modules/cnn3d.py). The full model uses "hdpm".
    encoder: str = "hdpm"
    # Recompute each Mamba block in the backward pass instead of storing the
    # scan states (same results, much less memory, ~30 % more compute).
    grad_checkpoint: bool = False


@dataclass
class DSSPEConfig:
    T_p: int = 5                              # PAPER: evolution steps
    alpha_init: Tuple[float, float, float] = (0.6, 0.2, 0.2)  # PAPER
    anchor_glove_dim: int = 300               # PAPER: GloVe dim
    contrastive_temp: float = 0.07            # PAPER: tau

    momentum_mu: float = 0.9
    eta_r: float = 1.0
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 2
    edge_dropout_p: float = 0.2
    feature_noise_std: float = 0.05
    # Scale s applied to the cosine logits in L_epi (softmax over s * cos).
    # With s = 1 the logits lie in [-1, 1], the softmax is almost uniform and
    # L_epi stays near ln N with very small gradients. Prediction (arg max)
    # does not depend on s.
    logit_scale: float = 10.0


@dataclass
class LossWeights:
    # PAPER (Sec. 3.5)
    lambda_epi: float = 0.1
    lambda_gcl: float = 0.1
    lambda_diff: float = 0.5
    lambda_adv: float = 0.1
    # NOT IN PAPER: focal modulation of the episodic loss, (1 - p_t)^gamma;
    # 0 = plain cross-entropy (the paper's L_epi)
    focal_gamma: float = 0.0


@dataclass
class EpisodeConfig:
    n_way: int = 5
    k_shot: int = 5      # PAPER headline setting
    q_query: int = 15    # PAPER


@dataclass
class DGMHMambaConfig:
    dadge: DADGEConfig = field(default_factory=DADGEConfig)
    hdpm: HDPMConfig = field(default_factory=HDPMConfig)
    dsspe: DSSPEConfig = field(default_factory=DSSPEConfig)
    loss: LossWeights = field(default_factory=LossWeights)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)

    # PAPER values for stages
    stage1_lr: float = 1e-3
    stage1_epochs: int = 200
    stage2_lr: float = 1e-4
    stage2_steps: int = 100
    weight_decay: float = 1e-2


def get_default_config() -> DGMHMambaConfig:
    return DGMHMambaConfig()
