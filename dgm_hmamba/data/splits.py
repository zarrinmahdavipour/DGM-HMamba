"""
Cross-domain episodic split helpers.

Following the protocol described in the paper (Sec. 4.2):

  - Source domain: all labelled pixels are available for Stage-1.
  - Target domain: labelled pixels are split into
        P_t^tr  – meta-training pool (used for episodic adaptation)
        P_t^te  – held-out meta-test pool (used only for final evaluation)
  - A 50 % cap is applied on very small classes so that both pools remain non-empty.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .datasets import HSIDataset
from .patches import extract_patches, remap_labels


def _split_indices_per_class(
    labels: np.ndarray,
    train_ratio: float = 0.5,
    seed: int = 0,
    min_test: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) index arrays."""
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []

    for c in np.unique(labels):
        if c < 0:
            continue
        cls_idx = np.where(labels == c)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_test = max(min_test, int(round(n * (1 - train_ratio))))
        # 50 % cap for tiny classes
        n_test = min(n_test, n // 2) if n >= 2 else 0
        test_idx.append(cls_idx[:n_test])
        train_idx.append(cls_idx[n_test:])

    return (
        np.concatenate(train_idx) if train_idx else np.array([], dtype=int),
        np.concatenate(test_idx) if test_idx else np.array([], dtype=int),
    )


def split_validation(
    labels: np.ndarray, val_fraction: float, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Carve a per-class validation subset out of a labelled pool.

    Used to select checkpoints inside the target meta-training pool, so that
    no label of the meta-testing pool influences training or model selection.
    Returns (keep_idx, val_idx) into ``labels``.
    """
    if val_fraction <= 0:
        return np.arange(len(labels)), np.array([], dtype=int)
    return _split_indices_per_class(labels, train_ratio=1.0 - val_fraction, seed=seed)


def make_cross_domain_split(
    source: HSIDataset,
    target: HSIDataset,
    patch_size: int = 9,
    target_train_ratio: float = 0.5,
    seed: int = 0,
    shared_labels: Optional[Sequence[int]] = None,
) -> Dict:
    """
    Build the full cross-domain data package used by the training scripts.

    Returns a dict with keys:
        source_patches, source_labels,
        target_tr_patches, target_tr_labels,
        target_te_patches, target_te_labels,
        target_tr_coords, target_te_coords   ((row, col) of each centre pixel),
        source_classes, target_classes
    """
    # ----- Source (all labelled pixels) -----
    src_patches, src_labs, _ = extract_patches(source, patch_size)
    src_keep = sorted(c for c in np.unique(src_labs) if c not in source.ignored_labels)
    src_labs = remap_labels(src_labs, src_keep)

    # ----- Target -----
    tgt_patches, tgt_labs, tgt_coords = extract_patches(target, patch_size)
    if shared_labels is not None:
        # keep only classes that exist in both domains (common in CDFSL)
        tgt_keep = [c for c in shared_labels if c not in target.ignored_labels]
    else:
        tgt_keep = sorted(c for c in np.unique(tgt_labs) if c not in target.ignored_labels)

    mask = np.isin(tgt_labs, tgt_keep)
    if not mask.all():  # avoid copying the (large) patch array when nothing is dropped
        tgt_patches = tgt_patches[mask]
        tgt_coords = tgt_coords[mask]
        tgt_labs = tgt_labs[mask]
    tgt_labs = remap_labels(tgt_labs, tgt_keep)

    tr_idx, te_idx = _split_indices_per_class(
        tgt_labs, train_ratio=target_train_ratio, seed=seed
    )

    return {
        "source_patches": src_patches,
        "source_labels": src_labs,
        "target_tr_patches": tgt_patches[tr_idx],
        "target_tr_labels": tgt_labs[tr_idx],
        "target_te_patches": tgt_patches[te_idx],
        "target_te_labels": tgt_labs[te_idx],
        "target_tr_coords": tgt_coords[tr_idx],
        "target_te_coords": tgt_coords[te_idx],
        "source_classes": src_keep,
        "target_classes": tgt_keep,
        "source_name": source.name,
        "target_name": target.name,
    }
