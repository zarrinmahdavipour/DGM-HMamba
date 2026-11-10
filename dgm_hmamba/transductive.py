"""
Transductive prototype refinement (NOT IN PAPER; an optional inference mode).

Few-shot prototypes come from K labelled support pixels per class. At
inference time the unlabelled pixels that are being classified (the queries
of an episode, or every pixel of the scene region being mapped) are also
available; soft k-means (Ren et al., ICLR 2018) uses them to move each
prototype towards the class as it actually looks in that region:

    w_uc = softmax_c(scale * cos(f_u, p_c))
    p_c  = normalise( (sum_{support of c} f + sum_u w_uc f_u) / (K_c + sum_u w_uc) )

repeated ``iters`` times. With ``topk`` > 0 only the ``topk`` unlabelled
pixels with the largest weight for a class contribute to that class.

No label of an unlabelled pixel is used. Results obtained with this mode are
transductive and must be reported as such, next to the inductive results.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F


def refine_prototypes(
    prototypes: torch.Tensor,
    support_feats: torch.Tensor,
    support_labels: torch.Tensor,
    unlabelled_feats: torch.Tensor,
    iters: int = 3,
    scale: float = 10.0,
    topk: int = 0,
) -> torch.Tensor:
    """prototypes (N, d) from the inductive model; support_feats (Ns, d) with
    labels in 0..N-1; unlabelled_feats (Nu, d). Returns refined (N, d)."""
    n_way = prototypes.shape[0]
    P = F.normalize(prototypes, dim=-1)
    if iters <= 0 or unlabelled_feats.numel() == 0:
        return P
    u = F.normalize(unlabelled_feats, dim=-1)
    s = F.normalize(support_feats, dim=-1)
    onehot = F.one_hot(support_labels.long(), n_way).to(s.dtype)      # (Ns, N)
    sup_sum = onehot.t() @ s                                            # (N, d)
    sup_cnt = onehot.sum(0)                                             # (N,)
    for _ in range(iters):
        w = torch.softmax(scale * (u @ P.t()), dim=-1)                  # (Nu, N)
        if topk > 0 and topk < w.shape[0]:
            idx = w.topk(topk, dim=0).indices                           # (topk, N)
            w = w * torch.zeros_like(w).scatter_(0, idx, 1.0)
        P = (sup_sum + w.t() @ u) / (sup_cnt + w.sum(0)).clamp(min=1e-6).unsqueeze(1)
        P = F.normalize(P, dim=-1)
    return P


def parse_grid(spec: str) -> List[Dict[str, float]]:
    """'iters=1,3,10;scale=10,20;topk=0,200' -> list of configs (cartesian)."""
    keys, values = [], []
    for part in spec.split(";"):
        k, v = part.split("=")
        keys.append(k.strip())
        values.append([float(x) for x in v.split(",")])
    grid: List[Dict[str, float]] = [{}]
    for k, vs in zip(keys, values):
        grid = [dict(g, **{k: (int(v) if k in ("iters", "topk") else v)}) for g in grid for v in vs]
    return grid


def config_name(cfg: Optional[Dict[str, float]]) -> str:
    if not cfg:
        return "inductive"
    return ",".join(f"{k}={cfg[k]}" for k in sorted(cfg))
