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

import dataclasses
import hashlib

from .datasets import HSIDataset, minmax_normalize
from .patches import extract_patches, remap_labels

PARTITIONS = ("train", "val", "test")     # block codes 0, 1, 2


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


# --------------------------------------------------------------------------
# Three-way geographic split (train / validation / test regions)
# --------------------------------------------------------------------------

def _near(mask, coords, image_shape, radius):
    """For every pixel in ``coords``: is a pixel of ``mask`` within Chebyshev
    distance <= radius?"""
    from scipy.ndimage import maximum_filter

    img = np.zeros(image_shape, dtype=np.uint8)
    img[coords[mask, 0], coords[mask, 1]] = 1
    return maximum_filter(img, size=2 * radius + 1)[coords[:, 0], coords[:, 1]] > 0


def assign_blocks(coords, image_shape, grid, block_size, buffer):
    """Pixel partitions from a block grid (0 train, 1 validation, 2 test).

    Test pixels are kept as they are; validation pixels within ``buffer``
    (Chebyshev distance) of a test pixel are dropped; training pixels within
    ``buffer`` of a test or of a kept validation pixel are dropped. With
    buffer >= patch_size - 1, no two patches from different partitions share
    an input pixel. Returns (train_idx, val_idx, test_idx, dropped).
    """
    part = np.asarray(grid)[coords[:, 0] // block_size, coords[:, 1] // block_size]
    te = part == 2
    va = (part == 1) & ~_near(te, coords, image_shape, buffer)
    tr = (part == 0) & ~_near(te | va, coords, image_shape, buffer)
    dropped = {"val": int(((part == 1) & ~va).sum()), "train": int(((part == 0) & ~tr).sum())}
    return np.where(tr)[0], np.where(va)[0], np.where(te)[0], dropped


def geographic_split(
    coords: np.ndarray,
    labels: np.ndarray,
    image_shape: Tuple[int, int],
    block_size: int = 64,
    buffer: int = 6,
    ratios: Tuple[float, float, float] = (0.35, 0.15, 0.5),
    seed: int = 0,
    min_counts: Tuple[int, int, int] = (50, 30, 100),
    max_attempts: int = 20000,
):
    """Random block assignment to train / validation / test regions.

    Attempt i draws every block independently with probabilities ``ratios``
    from a generator seeded by (seed, i). The first attempt in which every
    class has at least ``min_counts`` labelled pixels in the training,
    validation and testing partitions (after the buffers) is accepted. The
    rule depends only on the ground-truth map, the seed and these constants,
    never on any model output, so the split is fixed before training and is
    shared by every configuration that uses the same seed.
    """
    H, W = image_shape
    nby, nbx = -(-H // block_size), -(-W // block_size)
    classes = np.unique(labels)
    for attempt in range(max_attempts):
        ss = np.random.SeedSequence([int(seed), attempt])
        rng = np.random.RandomState(ss.generate_state(1)[0])
        grid = rng.choice(3, size=(nby, nbx), p=np.asarray(ratios) / np.sum(ratios))
        tr, va, te, dropped = assign_blocks(coords, image_shape, grid, block_size, buffer)
        counts = np.array([[(labels[ix] == c).sum() for c in classes] for ix in (tr, va, te)])
        if all((counts[i] >= min_counts[i]).all() for i in range(3)):
            return tr, va, te, grid, {"attempt": attempt, "dropped_pixels": dropped}
    raise RuntimeError(
        f"no block assignment met the minimum class counts {min_counts} in {max_attempts} attempts")


def min_center_distance(a: np.ndarray, b: np.ndarray, image_shape) -> Optional[int]:
    """Smallest Chebyshev distance between a pixel in ``a`` and one in ``b``
    ((N, 2) row/col arrays). Square patches of side P centred on the two
    pixels share no input pixel iff this distance is >= P."""
    from scipy.ndimage import distance_transform_cdt

    if len(a) == 0 or len(b) == 0:
        return None
    img = np.ones(image_shape, dtype=bool)
    img[a[:, 0], a[:, 1]] = False
    d = distance_transform_cdt(img, metric="chessboard")
    return int(d[b[:, 0], b[:, 1]].min())


def grid_strings(grid, coords=None, block_size=None):
    """Block grid as strings of '0' (train), '1' (validation), '2' (test);
    blocks without labelled pixels are shown as '.' when coords are given."""
    g = np.asarray(grid).astype(str)
    if coords is not None:
        has = np.zeros(g.shape, dtype=bool)
        has[coords[:, 0] // block_size, coords[:, 1] // block_size] = True
        g = np.where(has, g, ".")
    return ["".join(r) for r in g]


def _sha(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
        h.update(b"|")
    return h.hexdigest()[:16]


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
    val_ratio: float = 0.0,
    min_counts: Tuple[int, int, int] = (50, 30, 100),
    block_grid: Optional[np.ndarray] = None,
    target_norm: str = "scene",
) -> Dict:
    """
    Build the full cross-domain data package used by the training scripts.

    split_mode
        "random": per-class random pixel split (manuscript protocol, spatially
                  dependent: neighbouring pixels fall in both pools);
        "block":  square blocks assigned to pools, with a buffer.
                  val_ratio = 0: two-way train / test blocks (training share
                  ``target_train_ratio``);
                  val_ratio > 0: three-way geographic split; blocks go to
                  train / validation / test with probabilities
                  (1 - test - val_ratio, val_ratio, test = 1 - target_train_ratio),
                  see ``geographic_split``. ``block_grid`` reuses a saved
                  block assignment instead of drawing one.
    target_norm
        "scene": the target cube is used as loaded (whole-scene scaling);
        "train": the target must be loaded with normalize=False; per-band
                 min-max scaling is fitted on the training-pool pixels only.

    Returns a dict with keys:
        source_patches, source_labels,
        target_tr_patches, target_tr_labels, target_tr_coords,
        target_val_patches, target_val_labels, target_val_coords  (None
            unless a geographic validation region is used),
        target_te_patches, target_te_labels, target_te_coords,
        source_classes, target_classes, split_info
    """
    # ----- Source (all labelled pixels) -----
    src_patches, src_labs, _ = extract_patches(source, patch_size)
    src_keep = sorted(c for c in np.unique(src_labs) if c not in source.ignored_labels)
    src_labs = remap_labels(src_labs, src_keep)

    # ----- Target: labelled pixels (same order as extract_patches) -----
    gt = target.gt
    rows, cols = np.where(~np.isin(gt, target.ignored_labels))
    all_coords = np.stack([rows, cols], 1).astype(np.int64)
    all_labs = gt[rows, cols].astype(np.int64)
    if shared_labels is not None:
        # keep only classes that exist in both domains (common in CDFSL)
        tgt_keep = [c for c in shared_labels if c not in target.ignored_labels]
    else:
        tgt_keep = sorted(c for c in np.unique(all_labs) if c not in target.ignored_labels)
    mask = np.isin(all_labs, tgt_keep)
    tgt_coords = all_coords[mask]
    tgt_labs = remap_labels(all_labs[mask], tgt_keep)

    split_info: Dict = {"mode": split_mode, "split_seed": int(seed), "patch_size": int(patch_size)}
    va_idx = None
    if split_mode == "random":
        # manuscript protocol: per-class random pixel split (not spatially disjoint)
        tr_idx, te_idx = _split_indices_per_class(
            tgt_labs, train_ratio=target_train_ratio, seed=seed
        )
    elif split_mode == "block":
        buf = patch_size - 1 if buffer is None else buffer
        if buf < patch_size - 1:
            raise ValueError(f"buffer {buf} < patch_size - 1 = {patch_size - 1}: "
                             "training and testing patches could share input pixels")
        if val_ratio > 0 or block_grid is not None:
            ratios = (1.0 - (1.0 - target_train_ratio) - val_ratio, val_ratio, 1.0 - target_train_ratio)
            if block_grid is not None:
                grid = np.asarray(block_grid, dtype=np.int64)
                tr_idx, va_idx, te_idx, dropped = assign_blocks(tgt_coords, gt.shape, grid, block_size, buf)
                info = {"attempt": None, "dropped_pixels": dropped, "grid_source": "given"}
            else:
                tr_idx, va_idx, te_idx, grid, info = geographic_split(
                    tgt_coords, tgt_labs, gt.shape, block_size, buf, ratios, seed, min_counts)
            split_info.update(info)
            split_info.update({
                "block_size": block_size, "buffer": buf,
                "block_ratios": None if block_grid is not None else [round(r, 4) for r in ratios],
                "min_counts": None if block_grid is not None else list(min_counts),
                "block_grid": grid_strings(grid),
                "block_grid_labelled": grid_strings(grid, tgt_coords, block_size),
                "grid_fingerprint": _sha(grid.astype(np.int8), np.array([block_size, buf, *gt.shape])),
            })
        else:
            tr_idx, te_idx, info = _block_split(
                tgt_coords, tgt_labs, gt.shape, block_size, buf, target_train_ratio, seed,
            )
            split_info.update(info)
    else:
        raise ValueError(f"unknown split_mode {split_mode!r}")

    parts = {"train": tr_idx, "test": te_idx}
    if va_idx is not None:
        parts["val"] = va_idx
    n_cls = len(tgt_keep)
    split_info["class_counts"] = {k: np.bincount(tgt_labs[v], minlength=n_cls).tolist() for k, v in parts.items()}
    split_info["index_fingerprint"] = _sha(*(np.sort(parts[k]) for k in ("train", "val", "test") if k in parts))

    # receptive field: the model sees one patch_size x patch_size window per
    # pixel (no operation mixes patches), so windows of different
    # partitions are disjoint iff the centre distance is >= patch_size.
    dist = {}
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        if a in parts and b in parts:
            dist[f"{a}-{b}"] = min_center_distance(tgt_coords[parts[a]], tgt_coords[parts[b]], gt.shape)
    split_info["min_center_distance"] = dist
    split_info["receptive_field"] = int(patch_size)
    split_info["windows_disjoint"] = all(d is None or d >= patch_size for d in dist.values())
    if split_mode == "block" and not split_info["windows_disjoint"]:
        raise RuntimeError(f"partitions too close for {patch_size}x{patch_size} patches: {dist}")

    # ----- Target normalisation -----
    if target_norm == "train":
        if target.normalized:
            raise ValueError("target_norm='train' needs the target loaded with normalize=False")
        data, (lo, hi) = minmax_normalize(target.data, tgt_coords[tr_idx])
        target = dataclasses.replace(target, data=data, normalized=True)
        split_info["normalization"] = {"fit_on": "training pool", "pixels": int(len(tr_idx)),
                                       "stats_fingerprint": _sha(lo, hi)}
    elif target_norm == "scene":
        if not target.normalized:
            data, _ = minmax_normalize(target.data)
            target = dataclasses.replace(target, data=data, normalized=True)
        split_info["normalization"] = {"fit_on": "whole scene"}
    else:
        raise ValueError(f"unknown target_norm {target_norm!r}")

    tgt_patches, _, p_coords = extract_patches(target, patch_size)
    assert np.array_equal(p_coords, all_coords)
    if not mask.all():  # avoid copying the (large) patch array when nothing is dropped
        tgt_patches = tgt_patches[mask]

    split_info["leakage"] = leakage_stats(
        tgt_coords[tr_idx], tgt_labs[tr_idx], tgt_coords[te_idx], tgt_labs[te_idx],
        gt.shape, patch_size,
    )
    if va_idx is not None:
        split_info["leakage_train_val"] = leakage_stats(
            tgt_coords[tr_idx], tgt_labs[tr_idx], tgt_coords[va_idx], tgt_labs[va_idx],
            gt.shape, patch_size,
        )

    take = lambda ix: (None, None, None) if ix is None else (tgt_patches[ix], tgt_labs[ix], tgt_coords[ix])
    val_p, val_l, val_c = take(va_idx)
    return {
        "source_patches": src_patches,
        "source_labels": src_labs,
        "target_tr_patches": tgt_patches[tr_idx],
        "target_tr_labels": tgt_labs[tr_idx],
        "target_tr_coords": tgt_coords[tr_idx],
        "target_val_patches": val_p,
        "target_val_labels": val_l,
        "target_val_coords": val_c,
        "target_te_patches": tgt_patches[te_idx],
        "target_te_labels": tgt_labs[te_idx],
        "target_te_coords": tgt_coords[te_idx],
        "source_classes": src_keep,
        "target_classes": tgt_keep,
        "source_name": source.name,
        "target_name": target.name,
        "split_info": split_info,
    }
