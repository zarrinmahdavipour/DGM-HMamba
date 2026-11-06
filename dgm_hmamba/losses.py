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
    cos_logits: torch.Tensor, query_labels: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    return F.cross_entropy(cos_logits / temperature, query_labels)


def total_objective_stage1(
    l_cls: torch.Tensor, l_diff: torch.Tensor, l_adv: torch.Tensor, weights: LossWeights
) -> torch.Tensor:
    return l_cls + weights.lambda_diff * l_diff + weights.lambda_adv * l_adv


def total_objective_stage2(
    l_epi: torch.Tensor, l_gcl: torch.Tensor, weights: LossWeights
) -> torch.Tensor:
    return weights.lambda_epi * l_epi + weights.lambda_gcl * l_gcl
