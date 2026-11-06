"""Download public benchmark scenes.

Sources, tried in order for every file:
  1. the Hugging Face mirror of the EHU files (https://huggingface.co/danaroth),
     which is fast and reachable from cloud machines such as Google Colab;
  2. the original EHU page
     (http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes),
     over https and then http.

Every file is checked after download: it must be a MATLAB file and contain an
array of the expected shape, otherwise it is discarded and the next source is
tried. Houston 2013 must be placed manually (see README).
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from .datasets import DATASET_INFO

EHU = "www.ehu.eus/ccwintco/uploads"
HF = "https://huggingface.co/datasets/danaroth"

# name -> [(file name, expected array shape, [urls ...]), ...]
FILES = {
    "indian_pines": [
        ("Indian_pines_corrected.mat", (145, 145, 200), [
            f"{HF}/indian_pines/resolve/main/Indian_pines_corrected.mat",
            f"https://{EHU}/6/67/Indian_pines_corrected.mat",
            f"http://{EHU}/6/67/Indian_pines_corrected.mat"]),
        ("Indian_pines_gt.mat", (145, 145), [
            f"{HF}/indian_pines/resolve/main/Indian_pines_gt.mat",
            f"https://{EHU}/c/c4/Indian_pines_gt.mat",
            f"http://{EHU}/c/c4/Indian_pines_gt.mat"]),
    ],
    "pavia_u": [
        ("PaviaU.mat", (610, 340, 103), [
            f"{HF}/pavia/resolve/main/PaviaU.mat",
            f"https://{EHU}/e/ee/PaviaU.mat",
            f"http://{EHU}/e/ee/PaviaU.mat"]),
        ("PaviaU_gt.mat", (610, 340), [
            f"{HF}/pavia/resolve/main/PaviaU_gt.mat",
            f"https://{EHU}/5/50/PaviaU_gt.mat",
            f"http://{EHU}/5/50/PaviaU_gt.mat"]),
    ],
    "pavia_c": [
        ("Pavia.mat", (1096, 715, 102), [
            f"{HF}/pavia/resolve/main/Pavia.mat",
            f"https://{EHU}/e/e3/Pavia.mat",
            f"http://{EHU}/e/e3/Pavia.mat"]),
        ("Pavia_gt.mat", (1096, 715), [
            f"{HF}/pavia/resolve/main/Pavia_gt.mat",
            f"https://{EHU}/5/53/Pavia_gt.mat",
            f"http://{EHU}/5/53/Pavia_gt.mat"]),
    ],
    "salinas": [
        ("Salinas_corrected.mat", (512, 217, 204), [
            f"{HF}/salinas/resolve/main/Salinas_corrected.mat"]),
        ("Salinas_gt.mat", (512, 217), [
            f"{HF}/salinas/resolve/main/Salinas_gt.mat"]),
    ],
}
URLS = {name: [urls[0] for _, _, urls in files] for name, files in FILES.items()}

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DGM-HMamba-downloader)"}


def _is_mat_file(path: Path) -> bool:
    """MATLAB v5 and v7.3 files both start with a 'MATLAB' text header."""
    with open(path, "rb") as fh:
        return fh.read(6) == b"MATLAB"


def _largest_array_shape(path: Path) -> Tuple[int, ...]:
    from scipy.io import loadmat
    arrays = [v for k, v in loadmat(str(path)).items()
              if not k.startswith("__") and isinstance(v, np.ndarray)]
    return max(arrays, key=lambda a: a.size).shape if arrays else ()


def _check(path: Path, expected: Tuple[int, ...]) -> None:
    if not _is_mat_file(path):
        head = path.read_bytes()[:120].decode("utf-8", "replace").replace("\n", " ")
        raise IOError(f"not a .mat file (starts with: {head!r})")
    shape = _largest_array_shape(path)
    if tuple(shape) != tuple(expected):
        raise IOError(f"unexpected array shape {shape}, expected {expected}")


def _fetch(url: str, dest: Path, expected: Tuple[int, ...], timeout: int = 60) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        _check(tmp, expected)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
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
    for fname, expected, urls in FILES[name]:
        dest = folder / fname
        if dest.is_file():
            try:
                _check(dest, expected)
                if verbose:
                    print(f"  [skip] {dest} already exists")
                continue
            except Exception:
                dest.unlink()
        errors = []
        for url in urls:
            try:
                if verbose:
                    print(f"  [get ] {url}", flush=True)
                _fetch(url, dest, expected)
                if verbose:
                    print(f"         ok, {dest.stat().st_size / 1e6:.1f} MB", flush=True)
                break
            except Exception as exc:  # network/HTTP errors, wrong content
                errors.append(f"{url}\n      {type(exc).__name__}: {exc}")
                if verbose:
                    print(f"         failed: {type(exc).__name__}: {exc}", flush=True)
        else:
            raise RuntimeError(
                f"Could not download {fname} from any source:\n  " + "\n  ".join(errors) +
                "\nDownload it manually from "
                "http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes "
                f"and put it in {folder}/"
            )
    return folder


def download_datasets(names: Iterable[str], data_root: str | Path = "./data") -> None:
    for n in names:
        download_dataset(n, data_root)
