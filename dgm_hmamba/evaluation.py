"""Episodic evaluation on a labelled pool (Sec. 4.2 / 4.4)."""

from __future__ import annotations

import random
from typing import Dict

import numpy as np
import torch

from .episodes import ByClassIndex, gather_episode
from .metrics import classification_report


@torch.no_grad()
def evaluate_episodes(
    model,
    patches: torch.Tensor,
    labels: torch.Tensor,
    n_way: int,
    k_shot: int,
    q_query: int,
    anchors: torch.Tensor,
    device: torch.device,
    n_classes: int,
    n_episodes: int = 50,
    seed: int = 0,
) -> Dict[str, object]:
    """Run several N-way K-shot episodes on a fixed pool.

    Returns OA as mean / sample SD over episodes, plus AA and Cohen's kappa
    computed from the confusion matrix pooled over all episodes.
    """
    was_training = model.training
    model.eval()
    index = ByClassIndex(labels.tolist())
    rng = random.Random(seed)
    accs = []
    all_true, all_pred = [], []

    for _ in range(n_episodes):
        try:
            s_idx, q_idx, ep_classes = index.sample_episode(
                n_way, k_shot, q_query, rng
            )
        except ValueError:
            continue
        support_x, support_y, query_x, query_y = gather_episode(
            patches, labels, s_idx, q_idx, ep_classes
        )
        support_x0 = model.project_bands(support_x.to(device), is_source=False)
        query_x0 = model.project_bands(query_x.to(device), is_source=False)
        preds = model.predict_episode(
            support_x0,
            support_y.to(device),
            query_x0,
            n_way,
            anchors[:n_way].to(device),
        ).cpu()
        accs.append((preds == query_y).float().mean().item())
        # map episode labels 0..N-1 back to pool class ids
        ep = torch.as_tensor(ep_classes, dtype=torch.long)
        all_true.append(ep[query_y])
        all_pred.append(ep[preds])

    model.train(was_training)

    if not accs:
        return {"oa_mean": 0.0, "oa_std": 0.0, "aa": 0.0, "kappa": 0.0, "n_episodes": 0}

    report = classification_report(
        torch.cat(all_true).numpy(), torch.cat(all_pred).numpy(), n_classes
    )
    return {
        "oa_mean": float(np.mean(accs)),
        "oa_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "aa": report["aa"],
        "kappa": report["kappa"],
        "per_class_acc": report["per_class"],
        "n_episodes": len(accs),
    }


def format_metrics(m: Dict[str, object]) -> str:
    return "OA = %.2f ± %.2f %%  AA = %.2f %%  κ = %.4f  (%d episodes)" % (
        m["oa_mean"] * 100, m["oa_std"] * 100, m["aa"] * 100, m["kappa"], m["n_episodes"],
    )
