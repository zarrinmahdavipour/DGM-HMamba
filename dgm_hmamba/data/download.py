"""Download public benchmark scenes from the EHU Hyperspectral Remote Sensing
Scenes page (http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes).

Only scenes with stable direct links are listed. Salinas and Houston 2013 must
be placed manually (see README).
"""

from __future__ import annotations

import shutil
import time
import urllib.request
from pathlib import Path
from typing import Iterable

from .datasets import DATASET_INFO

EHU = "www.ehu.eus/ccwintco/uploads"

# Paths below EHU; each is tried over https first, then http.
FILES = {
    "indian_pines": ["6/67/Indian_pines_corrected.mat", "c/c4/Indian_pines_gt.mat"],
    "pavia_u": ["e/ee/PaviaU.mat", "5/50/PaviaU_gt.mat"],
    "pavia_c": ["e/e3/Pavia.mat", "5/53/Pavia_gt.mat"],
}
URLS = {name: [f"http://{EHU}/{p}" for p in paths] for name, paths in FILES.items()}

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DGM-HMamba-downloader)"}


def _is_mat_file(path: Path) -> bool:
    """MATLAB v5 and v7.3 files both start with a 'MATLAB' text header."""
    with open(path, "rb") as fh:
        return fh.read(6) == b"MATLAB"


def _fetch(url: str, dest: Path, timeout: int = 180) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    if not _is_mat_file(tmp):
        head = tmp.read_bytes()[:200].decode("utf-8", "replace").replace("\n", " ")
        tmp.unlink()
        raise IOError(f"server did not return a .mat file (starts with: {head[:120]!r})")
    tmp.replace(dest)


def download_dataset(name: str, data_root: str | Path = "./data", verbose: bool = True) -> Path:
    """Download one scene into ``data_root/<folder>/`` if it is not there yet."""
    name = name.lower().replace("-", "_")
    if name not in FILES:
        raise ValueError(
            f"No automatic download for '{name}'. Available: {sorted(FILES)}. "
            "Place the .mat files manually (see README)."
        )
    folder = Path(data_root) / DATASET_INFO[name]["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    for rel in FILES[name]:
        dest = folder / rel.rsplit("/", 1)[-1]
        if dest.is_file() and dest.stat().st_size > 0 and _is_mat_file(dest):
            if verbose:
                print(f"  [skip] {dest} already exists")
            continue
        errors = []
        for attempt, scheme in enumerate(("https", "http", "https")):
            url = f"{scheme}://{EHU}/{rel}"
            try:
                if verbose:
                    print(f"  [get ] {url}", flush=True)
                _fetch(url, dest)
                break
            except Exception as exc:  # network errors, HTTP errors, HTML instead of .mat
                errors.append(f"{url}: {type(exc).__name__}: {exc}")
                time.sleep(2 * attempt)
        else:
            raise RuntimeError(
                f"Could not download {dest.name}.\n  " + "\n  ".join(errors) +
                f"\nDownload it manually from "
                "http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes "
                f"and put it in {folder}/"
            )
        if verbose:
            print(f"         {dest.stat().st_size / 1e6:.1f} MB")
    return folder


def download_datasets(names: Iterable[str], data_root: str | Path = "./data") -> None:
    for n in names:
        download_dataset(n, data_root)
