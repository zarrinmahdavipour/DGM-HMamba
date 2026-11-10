#!/usr/bin/env python3
"""
Dataset diagnostics for the target scene (default: Pavia University).

Prints
  * labelled pixels per class and imbalance;
  * spectral angle (degrees) between class-mean spectra, most similar pairs;
  * intrinsic spectral separability: 9-way nearest-centroid classification of
    raw centre-pixel spectra (cosine), with 5 random support pixels per class
    (20 repeats) and with all meta-training pixels as centroids;
  * spatial structure: connected components and boundary pixels per class;
  * train/test leakage of the random pixel split (manuscript protocol) and of
    spatially disjoint block splits.

No model is trained. Usage:
    python scripts/analyze_dataset.py --data-root data --target pavia_u
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dgm_hmamba.console import use_utf8_stdout
from dgm_hmamba.data import get_dataset
from dgm_hmamba.data.datasets import CLASS_NAMES
from dgm_hmamba.data.splits import _block_split, _split_indices_per_class, leakage_stats
from dgm_hmamba.metrics import classification_report


def table(rows, header):
    widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    line = "  ".join(str(h).rjust(w) for h, w in zip(header, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(v).rjust(w) for v, w in zip(r, widths)))


def main():
    use_utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--target", default="pavia_u")
    ap.add_argument("--patch-size", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = get_dataset(args.target, args.data_root)
    gt, cube = ds.gt, ds.data
    classes = [int(c) for c in ds.labels]
    names = CLASS_NAMES.get(ds.name, [str(c) for c in classes])
    name = lambda c: names[c - 1][:20]
    rows, cols = np.where(np.isin(gt, classes))
    coords = np.stack([rows, cols], 1)
    lab = np.array([classes.index(int(v)) for v in gt[rows, cols]])
    spec = cube[rows, cols]                       # (N, bands)
    K = len(classes)
    print(f"Scene {ds.name}: {cube.shape[0]}x{cube.shape[1]} px, {cube.shape[2]} bands, "
          f"{len(lab)} labelled pixels, {K} classes\n")

    # ---------------------------------------------------------------- counts
    counts = np.bincount(lab, minlength=K)
    print("== Labelled pixels per class ==")
    table([(name(c), counts[i], f"{100 * counts[i] / counts.sum():.1f}%") for i, c in enumerate(classes)],
          ("class", "pixels", "share"))
    print(f"imbalance (largest / smallest): {counts.max() / counts.min():.1f}\n")

    # ------------------------------------------------------ spectral angles
    means = np.stack([spec[lab == i].mean(0) for i in range(K)])
    un = means / np.linalg.norm(means, axis=1, keepdims=True)
    sam = np.degrees(np.arccos(np.clip(un @ un.T, -1, 1)))
    print("== Spectral angle between class means (degrees) ==")
    table([(name(c),) + tuple(f"{v:.1f}" for v in sam[i]) for i, c in enumerate(classes)],
          ("class",) + tuple(str(i + 1) for i in range(K)))
    pairs = sorted(((sam[i, j], i, j) for i in range(K) for j in range(i + 1, K)))[:6]
    print("most similar pairs: " + "; ".join(f"{name(classes[i])} / {name(classes[j])} {a:.1f}°" for a, i, j in pairs))
    within = [np.degrees(np.arccos(np.clip((spec[lab == i] / np.linalg.norm(spec[lab == i], axis=1, keepdims=True)) @ un[i], -1, 1))).mean() for i in range(K)]
    print("mean within-class angle to own mean: " + ", ".join(f"{name(c)} {w:.1f}°" for c, w in zip(classes, within)) + "\n")

    # ------------------------------------- intrinsic spectral separability
    tr, te = _split_indices_per_class(lab, train_ratio=0.5, seed=args.seed)
    unit = spec / np.linalg.norm(spec, axis=1, keepdims=True)
    rng = np.random.RandomState(args.seed)
    cms = np.zeros((K, K), dtype=np.int64)
    oas, aas = [], []
    for _ in range(20):
        sup = np.concatenate([rng.choice(te[lab[te] == i], 5, replace=False) for i in range(K)])
        qry = np.setdiff1d(te, sup)
        protos = np.stack([unit[sup[lab[sup] == i]].mean(0) for i in range(K)])
        pred = np.argmax(unit[qry] @ protos.T, 1)
        rep = classification_report(lab[qry], pred, K)
        oas.append(rep["oa"]); aas.append(rep["aa"]); cms += np.array(rep["confusion_matrix"])
    print("== Raw-spectrum nearest centroid, 9-way, 5 supports per class (20 repeats) ==")
    print(f"OA {100 * np.mean(oas):.1f} ± {100 * np.std(oas, ddof=1):.1f}   AA {100 * np.mean(aas):.1f} ± {100 * np.std(aas, ddof=1):.1f}")
    rep = classification_report(np.repeat(np.arange(K), cms.sum(1)), np.concatenate([np.repeat(np.arange(K), cms[i]) for i in range(K)]), K)
    rown = cms / np.maximum(cms.sum(1, keepdims=True), 1)
    table([(name(c), f"{100 * rep['per_class'][i]:.1f}", f"{100 * rep['precision'][i]:.1f}", f"{100 * rep['f1'][i]:.1f}")
           + tuple(f"{100 * v:.0f}" for v in rown[i]) for i, c in enumerate(classes)],
          ("class", "recall", "prec", "F1") + tuple(f"->{j + 1}" for j in range(K)))
    protos = np.stack([unit[tr[lab[tr] == i]].mean(0) for i in range(K)])
    pred = np.argmax(unit[te] @ protos.T, 1)
    full = classification_report(lab[te], pred, K)
    print(f"with all meta-training pixels as centroids: OA {100 * full['oa']:.1f}  AA {100 * full['aa']:.1f}\n")

    # ----------------------------------------------------- spatial structure
    from scipy.ndimage import label as cc_label, maximum_filter, minimum_filter
    r = args.patch_size // 2
    print(f"== Spatial structure ({args.patch_size}x{args.patch_size} window) ==")
    srows = []
    for i, c in enumerate(classes):
        m = gt == c
        n_cc = cc_label(m)[1]
        other = np.isin(gt, classes) & ~m
        near_other = maximum_filter(other.astype(np.uint8), size=2 * r + 1) > 0
        srows.append((name(c), n_cc, f"{100 * near_other[m].mean():.1f}%"))
    table(srows, ("class", "regions", "boundary px"))
    print()

    # --------------------------------------------------------------- leakage
    print(f"== Train/test leakage for {args.patch_size}x{args.patch_size} patches ==")
    lrows = []
    ls = leakage_stats(coords[tr], lab[tr], coords[te], lab[te], gt.shape, args.patch_size)
    lrows.append(("random 50/50 (manuscript)", len(tr), len(te), "-",
                  f"{100 * ls['any_train_in_patch']:.1f}%", f"{100 * ls['same_class_train_in_patch']:.1f}%",
                  f"{100 * ls['patches_share_pixels']:.1f}%"))
    for bs in (32, 64):
        btr, bte, info = _block_split(coords, lab, gt.shape, bs, args.patch_size - 1, 0.5, args.seed)
        ls = leakage_stats(coords[btr], lab[btr], coords[bte], lab[bte], gt.shape, args.patch_size)
        mn = min(min((lab[btr] == i).sum(), (lab[bte] == i).sum()) for i in range(K))
        lrows.append((f"blocks {bs}px, buffer {args.patch_size - 1}", len(btr), len(bte), mn,
                      f"{100 * ls['any_train_in_patch']:.1f}%", f"{100 * ls['same_class_train_in_patch']:.1f}%",
                      f"{100 * ls['patches_share_pixels']:.1f}%"))
    table(lrows, ("split", "train", "test", "min class", "train px in test patch",
                  "same-class train px", "patches share px"))


if __name__ == "__main__":
    main()
