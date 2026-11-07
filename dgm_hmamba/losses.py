"""Training objectives (Sec. 3.5)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LossWeights


class SourceClassifierHead(nn.Module):
    def __init__(self, d_model: int, num_source_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_model, num_source_classes)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.fc(f)


def source_classification_loss(
    head: SourceClassifierHead, f_source: torch.Tensor, y_source: torch.Tensor
) -> torch.Tensor:
    return F.cross_entropy(head(f_source), y_source)


def episodic_loss(
    cos_logits: torch.Tensor, query_labels: torch.Tensor, temperature: float = 1.0,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy over cosine logits; with focal_gamma > 0 each query's
    term is weighted by (1 - p_true)^gamma (focal loss), so confidently
    classified queries count less and the hard ones (confusable classes)
    more. focal_gamma = 0 is exactly the plain cross-entropy."""
    logits = cos_logits / temperature
    if focal_gamma <= 0:
        return F.cross_entropy(logits, query_labels)
    ce = F.cross_entropy(logits, query_labels, reduction="none")
    p_true = torch.exp(-ce)
    return ((1.0 - p_true) ** focal_gamma * ce).mean()


def total_objective_stage1(
    l_cls: torch.Tensor, l_diff: torch.Tensor, l_adv: torch.Tensor, weights: LossWeights
) -> torch.Tensor:
    return l_cls + weights.lambda_diff * l_diff + weights.lambda_adv * l_adv


def total_objective_stage2(
    l_epi: torch.Tensor, l_gcl: torch.Tensor, weights: LossWeights
) -> torch.Tensor:
    return weights.lambda_epi * l_epi + weights.lambda_gcl * l_gcl
