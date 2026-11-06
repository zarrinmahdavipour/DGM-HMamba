#!/usr/bin/env python3
"""Write small random Indian Pines / Pavia University look-alikes (same file
names, keys and band counts as the real scenes) for tests and CI."""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import savemat

SCENES = {
    # folder, data file, data key, gt file, gt key, H, W, bands, classes
    "indian_pines": ("IndianPines", "Indian_pines_corrected.mat", "indian_pines_corrected",
                     "Indian_pines_gt.mat", "indian_pines_gt", 30, 30, 200, 16),
    "pavia_u": ("PaviaU", "PaviaU.mat", "paviaU", "PaviaU_gt.mat", "paviaU_gt", 40, 25, 103, 9),
}


def make(root: Path, seed: int = 0) -> Path:
    rng = np.random.RandomState(seed)
    for folder, dfile, dkey, gfile, gkey, h, w, c, k in SCENES.values():
        d = root / folder
        d.mkdir(parents=True, exist_ok=True)
        savemat(d / dfile, {dkey: (rng.rand(h, w, c) * 1000).astype(np.float32)})
        savemat(d / gfile, {gkey: rng.randint(0, k + 1, size=(h, w)).astype(np.uint8)})
    return root


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="./data_synthetic")
    print("synthetic data written to", make(Path(ap.parse_args().root)).resolve())
