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


def _block_split(
    coords: np.ndarray,
    labels: np.ndarray,
    image_shape: Tuple[int, int],
    block_size: int,
    buffer: int,
    train_ratio: float = 0.5,
    seed: int = 0,
    min_per_class: int = 40,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Spatially disjoint split: square blocks of the scene are assigned to
    the training or the testing pool, and training pixels closer than
    ``buffer`` pixels (Chebyshev distance) to any testing pixel are dropped,
    so that no training and testing patch share input pixels when
    buffer >= patch_size - 1."""
    classes = np.unique(labels)
    best = None
    for attempt in range(50):
        tr, te, info = _block_split_once(coords, labels, image_shape, block_size, buffer,
                                         train_ratio, seed + 1000 * attempt)
        n_tr = np.array([(labels[tr] == c).sum() for c in classes])
        n_te = np.array([(labels[te] == c).sum() for c in classes])
        score = min(n_tr.min(), n_te.min())
        if best is None or score > best[0]:
            best = (score, tr, te, dict(info, attempt=attempt))
        if score >= min_per_class:
            break
    if best[0] < min_per_class:
        print(f"[warn] block split: smallest class has only {best[0]} pixels in one pool")
    return best[1], best[2], best[3]


def _block_split_once(coords, labels, image_shape, block_size, buffer, train_ratio, seed):
    from scipy.ndimage import maximum_filter

    rng = np.random.RandomState(seed)
    H, W = image_shape
    nby, nbx = -(-H // block_size), -(-W // block_size)
    assign = rng.rand(nby, nbx) < train_ratio            # True -> training block
    by, bx = coords[:, 0] // block_size, coords[:, 1] // block_size

    # make sure every class appears in both pools
    for c in np.unique(labels):
        m = labels == c
        for want_train in (True, False):
            if not (assign[by[m], bx[m]] == want_train).any():
                ids = by[m] * nbx + bx[m]
                blk = np.bincount(ids).argmax()
                assign[blk // nbx, blk % nbx] = want_train
    is_train = assign[by, bx]

    test_img = np.zeros((H, W), dtype=np.uint8)
    test_img[coords[~is_train, 0], coords[~is_train, 1]] = 1
    near_test = maximum_filter(test_img, size=2 * buffer + 1) > 0
    dropped = is_train & near_test[coords[:, 0], coords[:, 1]]
    train_idx = np.where(is_train & ~dropped)[0]
    test_idx = np.where(~is_train)[0]
    info = {"block_size": block_size, "buffer": buffer, "dropped_train_pixels": int(dropped.sum())}
    return train_idx, test_idx, info


def leakage_stats(
    tr_coords: np.ndarray,
    tr_labels: np.ndarray,
    te_coords: np.ndarray,
    te_labels: np.ndarray,
    image_shape: Tuple[int, int],
    patch_size: int,
) -> Dict[str, float]:
    """How much the testing patches overlap labelled training pixels.

    any_train_in_patch: fraction of testing pixels whose patch contains at
        least one training pixel;
    same_class_train_in_patch: ... at least one training pixel of the same class;
    patches_share_pixels: fraction of testing pixels whose patch overlaps the
        patch of some training pixel (input pixels shared).
    """
    from scipy.ndimage import maximum_filter

    r = patch_size // 2
    H, W = image_shape
    tr_img = np.zeros((H, W), dtype=np.uint8)
    tr_img[tr_coords[:, 0], tr_coords[:, 1]] = 1
    in_patch = maximum_filter(tr_img, size=2 * r + 1) > 0
    share = maximum_filter(tr_img, size=4 * r + 1) > 0
    same = np.zeros(len(te_labels), dtype=bool)
    for c in np.unique(te_labels):
        img = np.zeros((H, W), dtype=np.uint8)
        sel = tr_labels == c
        img[tr_coords[sel, 0], tr_coords[sel, 1]] = 1
        near = maximum_filter(img, size=2 * r + 1) > 0
        m = te_labels == c
        same[m] = near[te_coords[m, 0], te_coords[m, 1]]
    return {
        "any_train_in_patch": float(in_patch[te_coords[:, 0], te_coords[:, 1]].mean()),
        "same_class_train_in_patch": float(same.mean()),
        "patches_share_pixels": float(share[te_coords[:, 0], te_coords[:, 1]].mean()),
    }


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
    split_mode: str = "random",
    block_size: int = 32,
    buffer: Optional[int] = None,
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

    split_info: Dict = {"mode": split_mode}
    if split_mode == "random":
        # manuscript protocol: per-class random pixel split (not spatially disjoint)
        tr_idx, te_idx = _split_indices_per_class(
            tgt_labs, train_ratio=target_train_ratio, seed=seed
        )
    elif split_mode == "block":
        tr_idx, te_idx, info = _block_split(
            tgt_coords, tgt_labs, target.gt.shape, block_size,
            patch_size - 1 if buffer is None else buffer, target_train_ratio, seed,
        )
        split_info.update(info)
    else:
        raise ValueError(f"unknown split_mode {split_mode!r}")
    split_info["leakage"] = leakage_stats(
        tgt_coords[tr_idx], tgt_labs[tr_idx], tgt_coords[te_idx], tgt_labs[te_idx],
        target.gt.shape, patch_size,
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
        "split_info": split_info,
    }
