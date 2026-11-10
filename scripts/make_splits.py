#!/usr/bin/env python3
"""
Write the three-way geographic splits (train / validation / test regions) of
the target scene to JSON, so that every experiment can be reproduced from
the saved block assignment (``demo_real.py --split-file``).

Only the ground-truth map is used: the split never depends on spectra or on
any model output. Usage:
    python scripts/make_splits.py --data-root data --seeds 42 1 2 3 4
    python scripts/make_splits.py --gt-file PaviaU_gt.mat --seeds 42 1 2 3 4
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dgm_hmamba.data.datasets import CLASS_NAMES, HSIDataset, _load_mat_array, DATASET_INFO, load_hsi  # noqa: E402
from dgm_hmamba.data.splits import make_cross_domain_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--gt-file", default=None, help="ground-truth .mat (no data cube needed)")
    ap.add_argument("--target", default="pavia_u")
    ap.add_argument("--patch-size", type=int, default=7)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--val-blocks", type=float, default=0.15)
    ap.add_argument("--test-share", type=float, default=0.5)
    ap.add_argument("--min-counts", default="50,30,100")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    ap.add_argument("--out-dir", default=str(ROOT / "splits"))
    args = ap.parse_args()

    if args.gt_file:
        gt = _load_mat_array(Path(args.gt_file), DATASET_INFO[args.target]["gt_keys"]).astype(np.int64)
    else:
        gt = load_hsi(args.target, args.data_root, normalize=False).gt
    ignored = tuple(DATASET_INFO[args.target]["ignored_labels"])
    # the split depends on the ground truth only: a one-band placeholder cube suffices
    tgt = HSIDataset(args.target, np.zeros(gt.shape + (1,), np.float32), gt, ignored)
    src = HSIDataset("placeholder", np.zeros((8, 8, 1), np.float32), np.ones((8, 8), np.int64), (0,))
    names = CLASS_NAMES.get(args.target)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        sp = make_cross_domain_split(src, tgt, patch_size=args.patch_size, seed=seed, split_mode="block",
                                     block_size=args.block_size, target_train_ratio=1 - args.test_share,
                                     val_ratio=args.val_blocks,
                                     min_counts=tuple(int(v) for v in args.min_counts.split(",")))
        si = dict(sp["split_info"])
        si.pop("normalization", None)       # depends on the data cube, recorded by each run
        si["class_names"] = names
        si["image_shape"] = list(gt.shape)
        path = out / f"{args.target}_block{args.block_size}_p{args.patch_size}_seed{seed}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"split_info": si}, f, indent=1)
        cc = si["class_counts"]
        print(f"{path.name}: attempt {si['attempt']}, grid {si['grid_fingerprint']}, "
              f"indices {si['index_fingerprint']}, min centre distance {si['min_center_distance']}")
        print("   " + " ".join(si["block_grid_labelled"]))
        for k in ("train", "val", "test"):
            print(f"   {k:5s} {sum(cc[k]):6d}: " + " ".join(f"{v:5d}" for v in cc[k]))


if __name__ == "__main__":
    main()
