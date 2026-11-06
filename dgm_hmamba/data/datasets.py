"""
Hyperspectral dataset loaders for the standard cross-domain benchmarks
used in the DGM-HMamba paper:

  - Indian Pines (AVIRIS)
  - Pavia University (ROSIS)
  - Pavia Centre   (ROSIS)
  - Salinas        (AVIRIS)
  - Houston 2013   (ITRES CASI)

Expected file layout (after download or manual placement):

  data_root/
  ├── IndianPines/
  │   ├── Indian_pines_corrected.mat
  │   └── Indian_pines_gt.mat
  ├── PaviaU/
  │   ├── PaviaU.mat
  │   └── PaviaU_gt.mat
  ├── PaviaC/
  │   ├── Pavia.mat
  │   └── Pavia_gt.mat
  ├── Salinas/
  │   ├── Salinas_corrected.mat
  │   └── Salinas_gt.mat
  └── Houston2013/
      ├── Houston13.mat          # or GRSS2013.mat
      └── Houston13_gt.mat       # one merged (train + test) label map

MATLAB v7.3 files (HDF5 based, common for Houston 2013) are read with
``h5py`` when it is installed (``pip install h5py``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from scipy.io import loadmat
except ImportError:
    loadmat = None


# Canonical short names used throughout the code
DATASET_INFO = {
    "indian_pines": {
        "folder": "IndianPines",
        "data_files": ["Indian_pines_corrected.mat", "Indian_pines.mat"],
        "gt_files": ["Indian_pines_gt.mat"],
        "data_keys": ["indian_pines_corrected", "indian_pines"],
        "gt_keys": ["indian_pines_gt"],
        "n_bands": 200,          # after removing water-absorption bands
        "ignored_labels": [0],
    },
    "pavia_u": {
        "folder": "PaviaU",
        "data_files": ["PaviaU.mat"],
        "gt_files": ["PaviaU_gt.mat"],
        "data_keys": ["paviaU", "paviaU_im"],
        "gt_keys": ["paviaU_gt"],
        "n_bands": 103,
        "ignored_labels": [0],
    },
    "pavia_c": {
        "folder": "PaviaC",
        "data_files": ["Pavia.mat", "PaviaC.mat"],
        "gt_files": ["Pavia_gt.mat", "PaviaC_gt.mat"],
        "data_keys": ["pavia", "paviaC"],
        "gt_keys": ["pavia_gt", "paviaC_gt"],
        "n_bands": 102,
        "ignored_labels": [0],
    },
    "salinas": {
        "folder": "Salinas",
        "data_files": ["Salinas_corrected.mat", "Salinas.mat"],
        "gt_files": ["Salinas_gt.mat"],
        "data_keys": ["salinas_corrected", "salinas"],
        "gt_keys": ["salinas_gt"],
        "n_bands": 204,          # after removing water-absorption bands
        "ignored_labels": [0],
    },
    "houston2013": {
        "folder": "Houston2013",
        "data_files": ["Houston13.mat", "GRSS2013.mat", "houston.mat"],
        "gt_files": ["Houston13_gt.mat", "GRSS2013_gt.mat", "houston_gt.mat"],
        "data_keys": ["Houston13", "houston", "data"],
        "gt_keys": ["Houston13_gt", "houston_gt", "gt"],
        "n_bands": 144,
        "ignored_labels": [0],
    },
}


# Class names in ground-truth order (label 1 = first entry). Used for plots and
# as the vocabulary for class-name (GloVe) semantic anchors.
CLASS_NAMES = {
    "indian_pines": [
        "Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture",
        "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats",
        "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat", "Woods",
        "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers",
    ],
    "pavia_u": [
        "Asphalt", "Meadows", "Gravel", "Trees", "Painted metal sheets",
        "Bare soil", "Bitumen", "Self-blocking bricks", "Shadows",
    ],
    "pavia_c": [
        "Water", "Trees", "Asphalt", "Self-blocking bricks", "Bitumen", "Tiles",
        "Shadows", "Meadows", "Bare soil",
    ],
    "salinas": [
        "Brocoli green weeds 1", "Brocoli green weeds 2", "Fallow",
        "Fallow rough plow", "Fallow smooth", "Stubble", "Celery",
        "Grapes untrained", "Soil vinyard develop", "Corn senesced green weeds",
        "Lettuce romaine 4wk", "Lettuce romaine 5wk", "Lettuce romaine 6wk",
        "Lettuce romaine 7wk", "Vinyard untrained", "Vinyard vertical trellis",
    ],
    "houston2013": [
        "Healthy grass", "Stressed grass", "Synthetic grass", "Trees", "Soil",
        "Water", "Residential", "Commercial", "Road", "Highway", "Railway",
        "Parking lot 1", "Parking lot 2", "Tennis court", "Running track",
    ],
}


@dataclass
class HSIDataset:
    """In-memory hyperspectral cube + ground-truth map."""

    name: str
    data: np.ndarray          # (H, W, C) float32
    gt: np.ndarray            # (H, W) int64
    ignored_labels: Tuple[int, ...] = (0,)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape

    @property
    def n_bands(self) -> int:
        return self.data.shape[-1]

    @property
    def labels(self) -> np.ndarray:
        """Unique labels excluding ignored ones."""
        labs = np.unique(self.gt)
        return labs[~np.isin(labs, self.ignored_labels)]

    def __repr__(self) -> str:
        h, w, c = self.data.shape
        n_cls = len(self.labels)
        return f"HSIDataset(name={self.name!r}, shape=({h},{w},{c}), classes={n_cls})"


def _find_file(folder: Path, candidates: list) -> Optional[Path]:
    for name in candidates:
        p = folder / name
        if p.is_file():
            return p
    return None


def _load_mat_v73(path: Path) -> Dict[str, np.ndarray]:
    """Read a MATLAB v7.3 (HDF5) file into a dict like ``scipy.io.loadmat``."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            f"{path.name} is a MATLAB v7.3 file. Install h5py to read it "
            "(pip install h5py) or re-save it in MATLAB with save(..., '-v7')."
        ) from exc
    out: Dict[str, np.ndarray] = {}
    with h5py.File(str(path), "r") as f:
        for key, item in f.items():
            if isinstance(item, h5py.Dataset):
                # HDF5 stores MATLAB arrays column-major: reverse the axes.
                out[key] = np.asarray(item).T
    return out


def _load_mat_array(path: Path, candidate_keys: list) -> np.ndarray:
    if loadmat is None:
        raise ImportError("scipy is required to load .mat files: pip install scipy")
    try:
        mat = loadmat(str(path))
    except NotImplementedError:
        # scipy raises this for MATLAB v7.3 files
        mat = _load_mat_v73(path)
    for key in candidate_keys:
        if key in mat:
            arr = mat[key]
            # some files wrap the array inside an extra dimension
            while arr.ndim > 3 and arr.shape[0] == 1:
                arr = arr[0]
            return np.asarray(arr)
    # fallback: largest numeric array
    candidates = [
        (k, v) for k, v in mat.items()
        if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim >= 2
    ]
    if not candidates:
        raise KeyError(f"No suitable array found in {path}. Keys: {list(mat.keys())}")
    key, arr = max(candidates, key=lambda kv: kv[1].size)
    print(f"[warn] using fallback key '{key}' from {path.name}")
    return np.asarray(arr)


def load_hsi(name: str, data_root: str | Path) -> HSIDataset:
    """Load a named hyperspectral dataset from disk."""
    name = name.lower().replace("-", "_").replace(" ", "_")
    if name not in DATASET_INFO:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose from: {list(DATASET_INFO.keys())}"
        )

    info = DATASET_INFO[name]
    folder = Path(data_root) / info["folder"]
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {folder}\n"
            f"Please place the .mat files there (see README)."
        )

    data_path = _find_file(folder, info["data_files"])
    gt_path = _find_file(folder, info["gt_files"])
    if data_path is None or gt_path is None:
        raise FileNotFoundError(
            f"Could not find data/gt files in {folder}.\n"
            f"Expected data: {info['data_files']}\n"
            f"Expected gt  : {info['gt_files']}"
        )

    data = _load_mat_array(data_path, info["data_keys"]).astype(np.float32)
    gt = _load_mat_array(gt_path, info["gt_keys"]).astype(np.int64)

    # Ensure (H, W, C)
    if data.ndim == 3 and data.shape[0] < data.shape[-1] and data.shape[0] <= 300:
        # sometimes stored as (C, H, W)
        if data.shape[0] != gt.shape[0]:
            data = np.transpose(data, (1, 2, 0))

    if data.shape[:2] != gt.shape[:2]:
        raise ValueError(
            f"Spatial shape mismatch: data {data.shape[:2]} vs gt {gt.shape[:2]}"
        )

    # Simple per-band min-max normalisation to [0, 1]
    data = data.copy()
    for b in range(data.shape[-1]):
        band = data[:, :, b]
        lo, hi = band.min(), band.max()
        if hi > lo:
            data[:, :, b] = (band - lo) / (hi - lo)

    return HSIDataset(
        name=name,
        data=data,
        gt=gt,
        ignored_labels=tuple(info["ignored_labels"]),
    )


def get_dataset(name: str, data_root: str | Path = "./data") -> HSIDataset:
    """Public entry point."""
    return load_hsi(name, data_root)
