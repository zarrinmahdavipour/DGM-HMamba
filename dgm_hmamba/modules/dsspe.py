"""Dynamic Spectral-Semantic Prototype Evolution (DSSPE)."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DSSPEConfig


class SimpleGNN(nn.Module):
    """Lightweight graph encoder for contrastive regularization."""

    def __init__(self, d_model: int, hidden: int = 128, n_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = d_model
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU(inplace=True)]
            in_dim = hidden
        layers.append(nn.Linear(hidden, d_model))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DSSPE(nn.Module):
    def __init__(self, cfg: DSSPEConfig, d_model: int):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model

        # Learnable fusion weights (initialized to paper's alpha_init)
        init = torch.tensor(cfg.alpha_init, dtype=torch.float32)
        # convert to logits so that softmax(init_logits) ≈ alpha_init
        self.fusion_logits = nn.Parameter(torch.log(init + 1e-8))

        # Project GloVe anchors (300-d) -> d_model
        self.anchor_proj = nn.Linear(cfg.anchor_glove_dim, d_model)

        self.graph_encoder = SimpleGNN(d_model, cfg.gnn_hidden_dim, cfg.gnn_num_layers)

    def _init_prototypes(
        self, support_feats: torch.Tensor, support_labels: torch.Tensor, n_way: int
    ) -> torch.Tensor:
        """Mean prototype per class. Returns (N, d)."""
        protos = []
        for k in range(n_way):
            mask = support_labels == k
            if mask.any():
                protos.append(support_feats[mask].mean(0))
            else:
                protos.append(torch.zeros(self.d_model, device=support_feats.device))
        return torch.stack(protos, dim=0)

    def evolve(
        self,
        support_feats: torch.Tensor,
        support_labels: torch.Tensor,
        n_way: int,
        semantic_anchors: torch.Tensor,
    ) -> torch.Tensor:
        """
        support_feats   : (N*K, d)
        support_labels  : (N*K,)
        semantic_anchors: (N, glove_dim) or already projected (N, d)
        Returns evolved prototypes (N, d)
        """
        P0 = self._init_prototypes(support_feats, support_labels, n_way)  # (N, d), Eq. proto_init
        P = P0
        bar_P = P0.clone()

        # Semantic anchors S (Eq. semantic). Proj(.) is not updated during
        # episodic training (the manuscript freezes it in Stage 2).
        if semantic_anchors.shape[-1] != self.d_model:
            with torch.no_grad():
                S = self.anchor_proj(semantic_anchors)  # (N, d)
        else:
            S = semantic_anchors
        S = F.normalize(S, dim=-1)

        # Graph-Laplacian diffusion operator exp(-eta L) (Eqs. graph_diffusion,
        # adjacency): non-negative cosine adjacency of the initial prototypes
        # P^(0), normalized Laplacian, exact matrix exponential.
        A = torch.clamp(F.cosine_similarity(P0.unsqueeze(1), P0.unsqueeze(0), dim=-1), min=0.0)
        deg = A.sum(-1).clamp(min=1e-6)
        D_inv_sqrt = torch.diag(deg.pow(-0.5))
        L = torch.eye(n_way, device=P0.device, dtype=P0.dtype) - D_inv_sqrt @ A @ D_inv_sqrt
        diffusion = torch.matrix_exp(-self.cfg.eta_r * L)

        alphas = F.softmax(self.fusion_logits, dim=0)  # (3,)

        for r in range(self.cfg.T_p):
            # 1) Momentum (Eq. momentum)
            bar_P = self.cfg.momentum_mu * bar_P + (1 - self.cfg.momentum_mu) * P
            P_mom = bar_P

            # 2) Graph-Laplacian diffusion (Eq. graph_diffusion)
            P_diff = diffusion @ P

            # 3) Semantic anchoring (Eq. semantic)
            P_sem = P @ (S.t() @ S)

            # Convex fusion
            P = alphas[0] * P_mom + alphas[1] * P_diff + alphas[2] * P_sem

        return P

    def classify(self, query_feats: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """Cosine similarity logits (B_q, N)."""
        q = F.normalize(query_feats, dim=-1)
        p = F.normalize(prototypes, dim=-1)
        return q @ p.t()


def graph_contrastive_loss(
    encoder: nn.Module,
    feats: torch.Tensor,
    edge_dropout_p: float = 0.2,
    feature_noise_std: float = 0.05,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Simple two-view graph contrastive loss on support features."""
    B, D = feats.shape
    if B < 2:
        return feats.new_zeros(())

    # Two augmented views
    noise1 = torch.randn_like(feats) * feature_noise_std
    noise2 = torch.randn_like(feats) * feature_noise_std
    z1 = F.normalize(encoder(feats + noise1), dim=-1)
    z2 = F.normalize(encoder(feats + noise2), dim=-1)

    # InfoNCE
    logits = z1 @ z2.t() / temperature
    labels = torch.arange(B, device=feats.device)
    return F.cross_entropy(logits, labels)
