"""Download public benchmark scenes from the EHU Hyperspectral Remote Sensing
Scenes page (http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes).

Only scenes with stable direct links are listed. Salinas and Houston 2013 must
be placed manually (see README).
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path
from typing import Iterable

from .datasets import DATASET_INFO

EHU = "http://www.ehu.eus/ccwintco/uploads"

URLS = {
    "indian_pines": [
        f"{EHU}/6/67/Indian_pines_corrected.mat",
        f"{EHU}/c/c4/Indian_pines_gt.mat",
    ],
    "pavia_u": [
        f"{EHU}/e/ee/PaviaU.mat",
        f"{EHU}/5/50/PaviaU_gt.mat",
    ],
    "pavia_c": [
        f"{EHU}/e/e3/Pavia.mat",
        f"{EHU}/5/53/Pavia_gt.mat",
    ],
}


def download_dataset(name: str, data_root: str | Path = "./data", verbose: bool = True) -> Path:
    """Download one scene into ``data_root/<folder>/`` if it is not there yet."""
    name = name.lower().replace("-", "_")
    if name not in URLS:
        raise ValueError(
            f"No automatic download for '{name}'. Available: {sorted(URLS)}. "
            "Place the .mat files manually (see README)."
        )
    folder = Path(data_root) / DATASET_INFO[name]["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    for url in URLS[name]:
        dest = folder / url.rsplit("/", 1)[-1]
        if dest.is_file() and dest.stat().st_size > 0:
            if verbose:
                print(f"  [skip] {dest} already exists")
            continue
        if verbose:
            print(f"  [get ] {url}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        req = urllib.request.Request(url, headers={"User-Agent": "DGM-HMamba-downloader"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        tmp.replace(dest)
    return folder


def download_datasets(names: Iterable[str], data_root: str | Path = "./data") -> None:
    for n in names:
        download_dataset(n, data_root)
