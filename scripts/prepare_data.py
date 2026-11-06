#!/usr/bin/env python3
"""
Check that the benchmark .mat files are present and load correctly, and
optionally download the scenes that have direct public links.

Usage
-----
    python scripts/prepare_data.py --data-root ./data
    python scripts/prepare_data.py --data-root ./data --download indian_pines pavia_u

Expected layout:

    data/
    ├── IndianPines/   Indian_pines_corrected.mat, Indian_pines_gt.mat
    ├── PaviaU/        PaviaU.mat, PaviaU_gt.mat
    ├── PaviaC/        Pavia.mat, Pavia_gt.mat
    ├── Salinas/       Salinas_corrected.mat, Salinas_gt.mat
    └── Houston2013/   Houston13.mat, Houston13_gt.mat

Public source: http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
(Houston 2013: IEEE GRSS Data Fusion Contest 2013.)
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dgm_hmamba.console import use_utf8_stdout
from dgm_hmamba.data import get_dataset, download_dataset, DOWNLOAD_URLS


DATASETS = ["indian_pines", "pavia_u", "pavia_c", "salinas", "houston2013"]


def main():
    use_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument(
        "--download", nargs="*", default=None, metavar="NAME",
        help=f"download these scenes first (available: {', '.join(sorted(DOWNLOAD_URLS))}); "
             "no names = all available",
    )
    args = parser.parse_args()

    root = Path(args.data_root)
    if args.download is not None:
        names = args.download or sorted(DOWNLOAD_URLS)
        print(f"Downloading {', '.join(names)} into {root.resolve()} ...")
        for name in names:
            download_dataset(name, root)
        print()

    print(f"Looking for datasets under: {root.resolve()}\n")

    ok = 0
    failed_requested = []
    requested = set(args.download or (sorted(DOWNLOAD_URLS) if args.download is not None else []))
    for name in DATASETS:
        try:
            ds = get_dataset(name, root)
            print(f"  [ok]      {name:13s} {ds}")
            ok += 1
        except FileNotFoundError:
            print(f"  [missing] {name:13s}")
            if name in requested:
                failed_requested.append(name)
        except Exception as e:
            print(f"  [error]   {name:13s} {type(e).__name__}: {e}")
            if name in requested:
                failed_requested.append(name)

    print(f"\n{ok}/{len(DATASETS)} datasets ready.")
    if ok == 0:
        print(
            "\nNo datasets found. Run with --download, or place the .mat files in the "
            "folders listed in the docstring of this script."
        )
    if failed_requested:
        sys.exit(f"Requested datasets could not be loaded: {', '.join(failed_requested)}")


if __name__ == "__main__":
    main()
