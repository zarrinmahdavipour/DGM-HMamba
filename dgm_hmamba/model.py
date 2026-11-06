"""DGM-HMamba end-to-end model (Fig. 1 of the paper)."""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from .config import DGMHMambaConfig
from .modules import BandProjection, DADGE, HDPM, DSSPE, graph_contrastive_loss
from .losses import (
    SourceClassifierHead,
    source_classification_loss,
    episodic_loss,
)


class DGMHMamba(nn.Module):
    def __init__(
        self,
        cfg: DGMHMambaConfig,
        source_bands: int,
        target_bands: int,
        num_source_classes: int,
    ):
        super().__init__()
        self.cfg = cfg
        C = cfg.dadge.C

        self.source_band_proj = BandProjection(source_bands, C)
        self.target_band_proj = BandProjection(target_bands, C)

        self.dadge = DADGE(cfg.dadge)
        self.hdpm = HDPM(cfg.hdpm, c_bands=C)
        self.dsspe = DSSPE(cfg.dsspe, d_model=cfg.hdpm.widths[-1])
        self.source_head = SourceClassifierHead(cfg.hdpm.widths[-1], num_source_classes)

    def project_bands(self, raw_patch: torch.Tensor, is_source: bool) -> torch.Tensor:
        proj = self.source_band_proj if is_source else self.target_band_proj
        return proj(raw_patch)

    def encode(
        self,
        x0: torch.Tensor,
        num_inference_steps: Optional[int] = None,
        ddim: bool = False,
    ) -> torch.Tensor:
        """Full inference path: DADGE enhance -> HDPM -> embedding (B, d3)."""
        x_hat, eps_hat = self._enhance(x0, num_inference_steps, ddim)
        return self.hdpm(x_hat, eps_hat)

    def _enhance(self, x0, num_inference_steps=None, ddim=False):
        if self.cfg.dadge.bypass_enhancement:
            return x0, torch.zeros_like(x0)
        x_hat, eps_hat, _ = self.dadge.enhance(x0, num_steps=num_inference_steps, ddim=ddim)
        if self.cfg.dadge.straight_through and torch.is_grad_enabled() and x0.requires_grad:
            x_hat = x_hat + (x0 - x0.detach())
        return x_hat, eps_hat

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------
    def stage1_losses(
        self,
        x0_source: torch.Tensor,
        y_source: torch.Tensor,
        x0_target_unlabeled: torch.Tensor,
        lambda_grl: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Domain discriminator (GRL is active for L_adv)
        _, c_source = self.dadge.discriminator(x0_source, lambda_grl)

        # Stop-grad on the conditioning vector for L_diff (standard DANN practice).
        # The gradient of L_diff must not flow back through the GRL; otherwise it
        # would be reversed and push the features towards a larger diffusion loss.
        c_for_diff = c_source.detach()

        x_hat_s, eps_hat_s = self._enhance(x0_source)
        f_source = self.hdpm(x_hat_s, eps_hat_s)

        l_cls = source_classification_loss(self.source_head, f_source, y_source)
        # X_0 is detached as well: otherwise the band projection could minimise
        # L_diff trivially by shrinking X_0 towards zero (then X_t is pure noise
        # and the noise is perfectly predictable), destroying the input signal.
        l_diff = self.dadge.diffusion_loss(x0_source.detach(), c_for_diff)
        l_adv = self.dadge.adversarial_loss(x0_source, x0_target_unlabeled, lambda_grl)
        return l_cls, l_diff, l_adv

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------
    def stage2_episode(
        self,
        support_x0: torch.Tensor,
        support_labels: torch.Tensor,
        query_x0: torch.Tensor,
        query_labels: torch.Tensor,
        n_way: int,
        semantic_anchors: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f_support = self.encode(support_x0)
        f_query = self.encode(query_x0)

        prototypes = self.dsspe.evolve(
            f_support, support_labels, n_way, semantic_anchors
        )
        cos_logits = self.dsspe.classify(f_query, prototypes)
        l_epi = episodic_loss(
            cos_logits, query_labels, temperature=1.0 / self.cfg.dsspe.logit_scale
        )
        l_gcl = graph_contrastive_loss(
            self.dsspe.graph_encoder,
            f_support,
            self.cfg.dsspe.edge_dropout_p,
            self.cfg.dsspe.feature_noise_std,
            self.cfg.dsspe.contrastive_temp,
        )
        return l_epi, l_gcl, cos_logits

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_episode(
        self,
        support_x0: torch.Tensor,
        support_labels: torch.Tensor,
        query_x0: torch.Tensor,
        n_way: int,
        semantic_anchors: torch.Tensor,
    ) -> torch.Tensor:
        f_support = self.encode(support_x0)
        f_query = self.encode(query_x0)
        prototypes = self.dsspe.evolve(
            f_support, support_labels, n_way, semantic_anchors
        )
        return self.dsspe.classify(f_query, prototypes).argmax(dim=-1)

    @torch.no_grad()
    def predict_zero_shot(
        self, query_x0: torch.Tensor, semantic_anchors: torch.Tensor
    ) -> torch.Tensor:
        f_query = self.encode(query_x0)
        q = torch.nn.functional.normalize(f_query, dim=-1)
        if semantic_anchors.shape[-1] != q.shape[-1]:
            s = self.dsspe.anchor_proj(semantic_anchors)
        else:
            s = semantic_anchors
        s = torch.nn.functional.normalize(s, dim=-1)
        return (q @ s.t()).argmax(dim=-1)
