"""Patch extraction utilities."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .datasets import HSIDataset


def extract_patches(
    dataset: HSIDataset,
    patch_size: int = 9,
    labels: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract centred patches for every labelled pixel.

    Returns
    -------
    patches : (N, C, patch_size, patch_size) float32
    labels  : (N,) int64
    coords  : (N, 2) int64  – (row, col) of the centre pixel
    """
    data, gt = dataset.data, dataset.gt
    H, W, C = data.shape
    pad = patch_size // 2

    # Reflect padding so border pixels also get full patches
    padded = np.pad(data, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")

    if labels is None:
        valid = ~np.isin(gt, dataset.ignored_labels)
    else:
        valid = np.isin(gt, list(labels))

    rows, cols = np.where(valid)
    n = len(rows)
    patches = np.empty((n, C, patch_size, patch_size), dtype=np.float32)
    labs = np.empty(n, dtype=np.int64)
    coords = np.empty((n, 2), dtype=np.int64)

    for i, (r, c) in enumerate(zip(rows, cols)):
        patches[i] = padded[r : r + patch_size, c : c + patch_size, :].transpose(2, 0, 1)
        labs[i] = gt[r, c]
        coords[i] = (r, c)

    return patches, labs, coords


class PatchDataset(Dataset):
    """Torch Dataset of pre-extracted patches."""

    def __init__(
        self,
        patches: np.ndarray,
        labels: np.ndarray,
        transform=None,
    ):
        assert len(patches) == len(labels)
        self.patches = torch.from_numpy(patches)          # (N, C, H, W)
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = self.patches[idx]
        y = self.labels[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


def remap_labels(labels: np.ndarray, keep: Sequence[int]) -> np.ndarray:
    """Map a subset of original labels onto a contiguous 0..K-1 range."""
    mapping = {int(l): i for i, l in enumerate(keep)}
    out = np.full_like(labels, fill_value=-1)
    for old, new in mapping.items():
        out[labels == old] = new
    return out
