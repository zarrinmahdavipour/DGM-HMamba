#!/usr/bin/env python3
"""
Real-data demo of DGM-HMamba: Indian Pines (source) -> Pavia University (target).

The model is deliberately small and the schedule short, so the whole pipeline
runs in minutes on Google Colab, a laptop GPU, or a CPU:

  1. (optional) download the two public scenes            --download
  2. Stage 1  source pre-training      L_cls + λ3·L_diff + λ_adv·L_adv
  3. Stage 2  episodic meta-training   λ1·L_epi + λ2·L_gcl  on the target
              meta-training pool P_t^tr
  4. N-way K-shot evaluation on the held-out meta-testing pool P_t^te
     (OA mean ± SD over episodes, AA, Cohen's κ) for K = 1 and K = 5
  5. a classification map of P_t^te built from K labelled pixels per class

The numbers it prints are NOT the manuscript's results: the demo model has
about 0.3 M parameters and trains for a few hundred iterations. Use
scripts/train.py for full runs.

Usage
-----
    python scripts/demo_real.py --download                 # default IP -> PU
    python scripts/demo_real.py --stage1-iters 50 --stage2-steps 20 --no-map   # quicker
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import maximum_filter
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dgm_hmamba import get_default_config, DGMHMamba
from dgm_hmamba.console import use_utf8_stdout
from dgm_hmamba.data import (
    DOWNLOAD_URLS,
    download_dataset,
    get_dataset,
    make_cross_domain_split,
    split_validation,
)
from dgm_hmamba.data.datasets import CLASS_NAMES
from dgm_hmamba.episodes import ByClassIndex, gather_episode
from dgm_hmamba.evaluation import evaluate_episodes, format_metrics
from dgm_hmamba.losses import total_objective_stage1, total_objective_stage2
from dgm_hmamba.metrics import classification_report, metrics_from_confusion
from dgm_hmamba.transductive import config_name, parse_grid, refine_prototypes
from dgm_hmamba.augment import augment_patches
import copy
import math


def demo_config(args):
    """Small configuration for the demo (paper configuration: see config.py)."""
    cfg = get_default_config()
    cfg.dadge.T = args.T
    cfg.dadge.default_inference_steps = args.T  # full reverse chain
    cfg.dadge.denoiser_base_channels = args.denoiser_ch
    cfg.dadge.denoiser_num_res_blocks = 1
    cfg.hdpm.widths = tuple(int(w) for w in args.widths.split(","))
    cfg.hdpm.blocks_per_level = args.blocks_per_level
    cfg.hdpm.state_dim = args.state_dim
    cfg.hdpm.expand_factor = args.expand
    cfg.hdpm.patch_size = args.patch_size
    cfg.hdpm.grad_checkpoint = args.grad_checkpoint
    if args.stage1_lr is not None:
        cfg.stage1_lr = args.stage1_lr
    if args.stage2_lr is not None:
        cfg.stage2_lr = args.stage2_lr
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    cfg.dadge.bypass_enhancement = args.no_diffusion
    cfg.dadge.deterministic_sampling = args.ddim
    if args.lambda_gcl is not None:
        cfg.loss.lambda_gcl = args.lambda_gcl
    if args.lambda_diff is not None:
        cfg.loss.lambda_diff = args.lambda_diff
    if args.lambda_adv is not None:
        cfg.loss.lambda_adv = args.lambda_adv
    if args.static_prototypes:          # ablation "w/o Dyn": plain class means, no L_gcl
        cfg.dsspe.T_p = 0
        cfg.loss.lambda_gcl = 0.0
    cfg.hdpm.encoder = args.encoder
    cfg.loss.focal_gamma = args.focal_gamma
    return cfg


def lr_factor(step: int, total: int, warmup: int, schedule: str) -> float:
    """Linear warm-up, then constant or cosine decay (to 5 % of the base rate)."""
    if warmup > 0 and step <= warmup:
        return step / warmup
    if schedule == "cosine":
        prog = (step - warmup) / max(1, total - warmup)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return 1.0


def parse_args():
    p = argparse.ArgumentParser(description="DGM-HMamba real-data demo")
    p.add_argument("--source", default="indian_pines")
    p.add_argument("--target", default="pavia_u")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--download", action="store_true",
                   help="download the source/target scenes if missing")
    p.add_argument("--output-dir", default="runs/demo")
    p.add_argument("--patch-size", type=int, default=7)
    p.add_argument("--stage1-iters", type=int, default=200)
    p.add_argument("--stage2-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-way", type=int, default=5)
    p.add_argument("--k-shot", type=int, default=5)
    p.add_argument("--q-query", type=int, default=15)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--val-fraction", type=float, default=0.0,
                   help="hold out this fraction of each class of the meta-training pool and "
                        "report validation OA after Stage 1 and after Stage 2 (diagnostics; "
                        "the meta-testing pool is not used for this)")
    p.add_argument("--val-episodes", type=int, default=30)
    p.add_argument("--no-diffusion", action="store_true",
                   help='ablation "w/o Diff": feed X_0 to HDPM without reverse diffusion')
    # model size
    p.add_argument("--widths", default="32,32,32", help="HDPM widths d1,d2,d3")
    p.add_argument("--blocks-per-level", type=int, default=1)
    p.add_argument("--state-dim", type=int, default=4)
    p.add_argument("--expand", type=int, default=1)
    p.add_argument("--denoiser-ch", type=int, default=32)
    p.add_argument("--T", type=int, default=10, help="diffusion timesteps")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="recompute Mamba blocks in the backward pass (saves memory)")
    # optimisation
    p.add_argument("--stage1-lr", type=float, default=None, help="default: config (1e-3)")
    p.add_argument("--stage2-lr", type=float, default=None, help="default: config (1e-4)")
    p.add_argument("--weight-decay", type=float, default=None, help="default: config (1e-2)")
    p.add_argument("--schedule", choices=["constant", "cosine"], default="constant")
    p.add_argument("--warmup-frac", type=float, default=0.0)
    p.add_argument("--train-way", type=int, default=None, help="classes per training episode")
    p.add_argument("--train-query", type=int, default=None, help="queries per class in training")
    p.add_argument("--augment", action="store_true", help="flips, rotations, scaling, noise")
    p.add_argument("--val-every", type=int, default=0,
                   help="validate every N Stage-2 episodes (needs --val-fraction)")
    p.add_argument("--lambda-gcl", type=float, default=None, help="default: config (0.1)")
    p.add_argument("--lambda-diff", type=float, default=None, help="default: config (0.5)")
    p.add_argument("--lambda-adv", type=float, default=None,
                   help="default: config (0.1); 0 removes the domain-adversarial alignment")
    p.add_argument("--static-prototypes", action="store_true",
                   help="ablation w/o Dyn: class-mean prototypes (T_p = 0) and no L_gcl")
    p.add_argument("--encoder", choices=["hdpm", "cnn"], default="hdpm",
                   help="ablation w/ CNN: replace HDPM by a residual 3-D CNN")
    p.add_argument("--focal-gamma", type=float, default=0.0,
                   help="focal modulation of the episodic loss (0 = plain cross-entropy)")
    p.add_argument("--ddim", action="store_true",
                   help="deterministic DDIM reverse chain for the enhancement")
    p.add_argument("--ema", type=float, default=0.0,
                   help="EMA decay for Stage-2 weights (0 = off); the EMA weights are evaluated")
    p.add_argument("--clip-grad", type=float, default=1.0,
                   help="clip the gradient norm (0 = off)")
    p.add_argument("--select-on-val", action="store_true",
                   help="evaluate the Stage-2 checkpoint with the best validation OA")
    p.add_argument("--no-map", action="store_true", help="skip the classification map")
    p.add_argument("--transductive", action="store_true",
                   help="also report transductive results: prototypes refined by soft k-means on the "
                        "unlabelled pixels being classified; settings chosen on the validation pool only")
    p.add_argument("--tr-grid", default="iters=1,3,10;scale=10,20;topk=0,200",
                   help="transductive settings searched on the validation pool")
    p.add_argument("--tr-val-draws", type=int, default=5,
                   help="support draws of the validation 9-class map used to choose the map setting")
    p.add_argument("--map-repeats", type=int, default=10,
                   help="number of support draws for the 9-class map (mean ± SD reported)")
    p.add_argument("--split", choices=["random", "block"], default="random",
                   help="random: per-class random pixel split (manuscript protocol); "
                        "block: spatially disjoint blocks with a buffer (no shared patch pixels)")
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--val-blocks", type=float, default=0.0,
                   help="block split only: share of blocks forming a geographically separate "
                        "validation region (three-way split; replaces --val-fraction)")
    p.add_argument("--min-counts", default="50,30,100",
                   help="three-way block split: minimum labelled pixels per class in the "
                        "training, validation and testing partitions")
    p.add_argument("--split-file", default=None,
                   help="reuse the block assignment saved in a split JSON (block_grid)")
    p.add_argument("--norm", choices=["scene", "train"], default="scene",
                   help="target scaling: whole scene (legacy) or fitted on training pixels only")
    p.add_argument("--stop-after", type=int, default=0,
                   help="stop Stage 2 after this episode and write <output-dir>/resume.pt "
                        "(lets a run longer than one CI job continue in a second job)")
    p.add_argument("--resume", default=None,
                   help="continue Stage 2 from a resume.pt written with --stop-after; all other "
                        "arguments must be identical")
    p.add_argument("--split-seed", type=int, default=None,
                   help="seed of the target split (default: --seed)")
    p.add_argument("--eval-seed", type=int, default=None,
                   help="seed of validation/test episodes and map support draws "
                        "(default: derived from --seed, legacy behaviour)")
    p.add_argument("--map-batch", type=int, default=None,
                   help="patches per forward pass for the map (default 128 on GPU, 64 on CPU)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--preset", choices=sorted(PRESETS), default=None,
                   help="'verified': the configuration of the verified results in the README "
                        "(Indian Pines -> Pavia University, spatially independent block split, "
                        "split seed 5, model seed 42); any other option given overrides it")
    p.add_argument("--weights", default=None,
                   help="evaluate a saved model_final.pt instead of training: the settings, the "
                        "split and the anchors are taken from that run (only --data-root, "
                        "--output-dir, --download, --device, --map-batch, --no-map apply)")
    pre, _ = p.parse_known_args()
    if pre.preset:
        p.set_defaults(**PRESETS[pre.preset])
    return p.parse_args()


# Configuration of the verified results (Round 9, splits/CONFIRMATION_PROTOCOL.md).
# Diffusion enhancement is bypassed (the manuscript's "w/o Diff" variant).
PRESETS = {
    "verified": dict(
        val_fraction=0.1, val_episodes=40, val_every=100, select_on_val=True,
        schedule="cosine", warmup_frac=0.05, stage1_iters=800, stage2_lr=2e-3, stage2_steps=2000,
        train_way=9, train_query=5, augment=True, expand=2, ema=0.99, no_diffusion=True,
        map_repeats=10, split="block", block_size=64, val_blocks=0.15, min_counts="50,30,100",
        norm="train", eval_seed=0, split_seed=5, seed=42,
    ),
}
# Options that still apply when --weights evaluates a saved run.
WEIGHTS_KEEP = ("data_root", "output_dir", "download", "device", "map_batch", "no_map", "weights", "preset")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def false_colour(cube: np.ndarray) -> np.ndarray:
    """Simple RGB composite with 2-98 % stretch."""
    c = cube.shape[-1]
    bands = [int(0.53 * c), int(0.40 * c), int(0.12 * c)]
    rgb = cube[:, :, bands].astype(np.float32)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)


def save_figures(out_dir, tgt_ds, target_classes, pred_map, history, map_report, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    n = len(target_classes)
    base = plt.get_cmap("tab20").colors
    cmap = ListedColormap([(0, 0, 0)] + [base[i % len(base)] for i in range(n)])
    names = CLASS_NAMES.get(tgt_ds.name, [f"class {c}" for c in target_classes])

    # Ground-truth map restricted to the evaluated classes (1..n)
    gt_vis = np.zeros_like(tgt_ds.gt)
    for i, c in enumerate(target_classes):
        gt_vis[tgt_ds.gt == c] = i + 1

    paths = []
    if pred_map is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 6.2))
        axes[0].imshow(false_colour(tgt_ds.data))
        axes[0].set_title("False colour")
        axes[1].imshow(gt_vis, cmap=cmap, vmin=0, vmax=n, interpolation="nearest")
        axes[1].set_title("Ground truth")
        axes[2].imshow(pred_map, cmap=cmap, vmin=0, vmax=n, interpolation="nearest")
        axes[2].set_title(
            "DGM-HMamba demo (meta-testing pool)\n"
            f"OA {map_report['oa']*100:.1f}%  AA {map_report['aa']*100:.1f}%  "
            f"κ {map_report['kappa']:.3f}"
        )
        for ax in axes:
            ax.axis("off")
        handles = [Patch(color=cmap(i + 1), label=names[c - 1] if c - 1 < len(names) else str(c))
                   for i, c in enumerate(target_classes)]
        fig.legend(handles=handles, loc="lower center", ncol=min(n, 5), fontsize=9, frameon=False)
        fig.suptitle(title, fontsize=11)
        fig.tight_layout(rect=(0, 0.1, 1, 0.95))
        p = Path(out_dir) / "demo_maps.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    s1 = history["stage1"]
    if s1:
        it = [h["iter"] for h in s1]
        for key, lab in (("l_cls", "L_cls"), ("l_diff", "L_diff"), ("l_adv", "L_adv")):
            axes[0].plot(it, [h[key] for h in s1], label=lab, lw=1)
        axes[0].set_xlabel("Stage-1 iteration")
        axes[0].set_title("Stage 1: source pre-training")
        axes[0].legend()
    s2 = history["stage2"]
    if s2:
        st = [h["step"] for h in s2]
        axes[1].plot(st, [h["l_epi"] for h in s2], label="L_epi", lw=1)
        axes[1].plot(st, [h["l_gcl"] for h in s2], label="L_gcl", lw=1)
        ax2 = axes[1].twinx()
        acc = np.array([h["acc"] for h in s2])
        k = max(1, len(acc) // 10)
        smooth = np.convolve(acc, np.ones(k) / k, mode="valid")
        ax2.plot(st[k - 1:], smooth * 100, color="0.3", ls="--", label="episode acc (smoothed)")
        ax2.set_ylabel("query accuracy (%)")
        axes[1].set_xlabel("Stage-2 episode")
        axes[1].set_title("Stage 2: episodic meta-training")
        axes[1].legend(loc="upper left")
        ax2.legend(loc="upper right")
    fig.tight_layout()
    p = Path(out_dir) / "demo_curves.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)
    return paths


RESUME_IGNORED = ("stop_after", "resume", "output_dir", "data_root", "download", "device",
                  "preset", "weights")


def _resume_args(args) -> dict:
    return {k: v for k, v in sorted(vars(args).items()) if k not in RESUME_IGNORED}


def _rng_state(rng: random.Random) -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "episode_rng": rng.getstate()}


def _set_rng_state(state: dict, rng: random.Random) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    rng.setstate(state["episode_rng"])


def main():
    use_utf8_stdout()
    args = parse_args()
    saved = None
    if args.weights:
        saved = torch.load(args.weights, map_location="cpu", weights_only=False)
        for k, v in saved["args"].items():
            if k not in WEIGHTS_KEEP:
                setattr(args, k, v)
        args.stop_after, args.resume = 0, None
        print(f"Evaluating saved weights {args.weights} (selected episode {saved.get('selected_episode')}); "
              f"settings and split taken from that run, training skipped.")
    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timings = {}
    t_all = time.time()

    print(f"Device: {device}   torch {torch.__version__}")
    print(f"Transfer: {args.source} → {args.target}\n")

    # ------------------------------------------------------------------ data
    if args.download:
        for name in (args.source, args.target):
            if name in DOWNLOAD_URLS:
                download_dataset(name, args.data_root)
            else:
                print(f"  [warn] no automatic download for {name}; place it manually")

    t0 = time.time()
    src_ds = get_dataset(args.source, args.data_root)
    tgt_ds = get_dataset(args.target, args.data_root, normalize=(args.norm == "scene"))
    cfg = demo_config(args)
    split_seed = args.seed if args.split_seed is None else args.split_seed
    eval_seed = args.seed if args.eval_seed is None else args.eval_seed
    grid = None
    if args.split_file:
        with open(args.split_file, encoding="utf-8") as f:
            saved = json.load(f)
        saved = saved.get("split_info", saved)
        grid = np.array([[int(ch) for ch in row] for row in saved["block_grid"]])
        args.block_size = saved.get("block_size", args.block_size)
        if args.split_seed is None and saved.get("split_seed") is not None:
            split_seed = int(saved["split_seed"])      # report the seed that generated the grid
    split = make_cross_domain_split(src_ds, tgt_ds, patch_size=args.patch_size, seed=split_seed,
                                    split_mode=args.split, block_size=args.block_size,
                                    val_ratio=args.val_blocks if args.split == "block" else 0.0,
                                    min_counts=tuple(int(v) for v in args.min_counts.split(",")),
                                    block_grid=grid, target_norm=args.norm)
    if args.split_file:
        split["split_info"]["split_file"] = args.split_file
    print(f"  Seeds: model {args.seed}, split {split_seed}, evaluation {eval_seed}")

    src_x = torch.from_numpy(split["source_patches"])
    src_y = torch.from_numpy(split["source_labels"])
    tr_x = torch.from_numpy(split["target_tr_patches"])
    tr_y = torch.from_numpy(split["target_tr_labels"])
    te_x = torch.from_numpy(split["target_te_patches"])
    te_y = torch.from_numpy(split["target_te_labels"])
    target_classes = list(split["target_classes"])
    n_cls_t = len(target_classes)
    n_way = min(args.n_way, n_cls_t)
    val_x = val_y = None
    if split["target_val_labels"] is not None:
        # geographically separate validation region (three-way block split)
        val_x = torch.from_numpy(split["target_val_patches"])
        val_y = torch.from_numpy(split["target_val_labels"])
        if args.val_fraction > 0:
            print("  (--val-fraction ignored: the validation region comes from the block split)")
    elif args.val_fraction > 0:
        keep, val = split_validation(split["target_tr_labels"], args.val_fraction, seed=args.seed + 1)
        keep, val = torch.as_tensor(keep, dtype=torch.long), torch.as_tensor(val, dtype=torch.long)
        val_x, val_y = tr_x[val], tr_y[val]
        tr_x, tr_y = tr_x[keep], tr_y[keep]
    timings["data_s"] = time.time() - t0

    print(f"  Source  {src_ds}  → {len(src_y)} labelled patches")
    print(f"  Target  {tgt_ds}")
    print(f"          meta-training pool P_t^tr: {len(tr_y)}   meta-testing pool P_t^te: {len(te_y)}")
    if val_y is not None:
        where = "geographic region" if split["target_val_labels"] is not None else "held out from P_t^tr"
        print(f"          validation pool ({where}): {len(val_y)}")
    si = split["split_info"]
    lk = si["leakage"]
    hidden = ("leakage", "leakage_train_val", "block_grid", "block_grid_labelled", "class_counts")
    print(f"          split: {si['mode']} {({k: v for k, v in si.items() if k not in hidden})}")
    print(f"          test patches containing training pixels: {100 * lk['any_train_in_patch']:.1f}% "
          f"(same class {100 * lk['same_class_train_in_patch']:.1f}%), "
          f"sharing input pixels with a training patch: {100 * lk['patches_share_pixels']:.1f}%")
    if "leakage_train_val" in si:
        lv = si["leakage_train_val"]
        print(f"          validation patches sharing input pixels with a training patch: "
              f"{100 * lv['patches_share_pixels']:.1f}%")
    if "block_grid_labelled" in si:
        print(f"          block assignment ({si['block_size']} px blocks; 0 train, 1 validation, 2 test, "
              f". no labelled pixels): " + " ".join(si["block_grid_labelled"]))
    names0 = CLASS_NAMES.get(tgt_ds.name)
    pools = {"train": tr_y.numpy(), "val": None if val_y is None else val_y.numpy(), "test": te_y.numpy()}
    print(f"          {'class':<22}" + "".join(f"{k:>16}" for k, v in pools.items() if v is not None))
    for i, c in enumerate(target_classes):
        cells = "".join(f"{int((v == i).sum()):>8} {100 * (v == i).mean():5.1f}%"
                        for v in pools.values() if v is not None)
        print(f"          {(names0[c - 1] if names0 else str(c)):<22}{cells}")
    print(f"          {'total':<22}" + "".join(f"{len(v):>8} {100.0:5.1f}%" for v in pools.values() if v is not None))
    si["pool_class_counts"] = {k: np.bincount(v, minlength=n_cls_t).tolist()
                               for k, v in pools.items() if v is not None}
    with open(out_dir / "split.json", "w", encoding="utf-8") as f:
        json.dump({"split_info": si, "seeds": {"model": args.seed, "split": split_seed, "eval": eval_seed},
                   "source": args.source, "target": args.target}, f, indent=1)
    np.savez_compressed(out_dir / "split_coords.npz", train=split["target_tr_coords"],
                        test=split["target_te_coords"],
                        **({} if split["target_val_coords"] is None else {"val": split["target_val_coords"]}))

    # ----------------------------------------------------------------- model
    model = DGMHMamba(cfg, src_ds.n_bands, tgt_ds.n_bands, len(split["source_classes"])).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Demo model: {n_params/1e6:.3f} M trainable parameters "
          f"(widths {cfg.hdpm.widths}, T={cfg.dadge.T}, {args.patch_size}×{args.patch_size} patches)")
    print(f"  Settings: {vars(args)}\n")

    # Placeholder semantic anchors (GloVe class-name anchors not implemented yet)
    anchors = torch.randn(n_cls_t, cfg.dsspe.anchor_glove_dim)
    history = {"stage1": [], "stage2": [], "validation": {}}
    skipped = {"stage1": 0, "stage2": 0}   # updates skipped because the loss was not finite
    history["skipped_updates"] = skipped
    ckpt = None
    if saved is not None:                       # --weights: evaluate a saved run
        model.load_state_dict(saved["model"])
        anchors = saved["anchors"]
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if ckpt["args"] != _resume_args(args):
            diff = {k: (ckpt["args"].get(k), v) for k, v in _resume_args(args).items() if ckpt["args"].get(k) != v}
            raise SystemExit(f"--resume: arguments differ from the interrupted run: {diff}")
        if ckpt["index_fingerprint"] != split["split_info"]["index_fingerprint"]:
            raise SystemExit("--resume: the data split differs from the interrupted run")
        anchors = ckpt["anchors"]
        history, skipped = ckpt["history"], ckpt["skipped"]
        history["skipped_updates"] = skipped
        timings["stage1_s"] = ckpt["timings"]["stage1_s"]
        t_all -= ckpt["timings"].get("total_s", 0.0)     # report the time of both parts
        print(f"  resuming Stage 2 after episode {ckpt['step']} from {args.resume}")

    def validate(tag, net=None):
        if val_y is None:
            return None
        m = evaluate_episodes(net or model, val_x, val_y, n_way, args.k_shot, args.q_query, anchors,
                              device, n_cls_t, n_episodes=args.val_episodes, seed=777 + eval_seed)
        history["validation"][tag] = m["oa_mean"]
        print(f"  [validation, {tag}] {n_way}-way {args.k_shot}-shot {format_metrics(m)}", flush=True)
        return m["oa_mean"]

    def maybe_augment(x):
        return augment_patches(x) if args.augment else x

    # --------------------------------------------------------------- stage 1
    print("=" * 64)
    print(f"STAGE 1 – source pre-training + domain-adversarial diffusion ({args.stage1_iters} iters)")
    print("=" * 64)
    opt = AdamW(model.parameters(), lr=cfg.stage1_lr, weight_decay=cfg.weight_decay)
    stage1_iters = 0 if (ckpt is not None or saved is not None) else args.stage1_iters   # resumed or --weights
    warmup = max(1, int(0.15 * args.stage1_iters))   # paper: λ_grl ramps over 30 of 200 epochs
    lr_warm1 = int(args.warmup_frac * args.stage1_iters)
    gen = torch.Generator().manual_seed(args.seed)
    model.train()
    t0 = time.time()
    for it in range(1, stage1_iters + 1):
        lambda_grl = min(1.0, it / warmup)
        for g in opt.param_groups:
            g["lr"] = cfg.stage1_lr * lr_factor(it, args.stage1_iters, lr_warm1, args.schedule)
        si = torch.randint(len(src_y), (args.batch_size,), generator=gen)
        ti = torch.randint(len(tr_y), (args.batch_size,), generator=gen)   # unlabeled target
        x0_s = model.project_bands(maybe_augment(src_x[si].to(device)), is_source=True)
        x0_t = model.project_bands(maybe_augment(tr_x[ti].to(device)), is_source=False)

        opt.zero_grad(set_to_none=True)
        l_cls, l_diff, l_adv = model.stage1_losses(x0_s, src_y[si].to(device), x0_t, lambda_grl)
        loss = total_objective_stage1(l_cls, l_diff, l_adv, cfg.loss)
        if not torch.isfinite(loss):
            skipped["stage1"] += 1
            continue
        loss.backward()
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        opt.step()

        history["stage1"].append({
            "iter": it, "lambda_grl": lambda_grl, "l_cls": l_cls.item(),
            "l_diff": l_diff.item(), "l_adv": l_adv.item(), "total": loss.item(),
        })
        if it == 1 or it % 20 == 0 or it == args.stage1_iters:
            print(f"  iter {it:4d} | λ_grl={lambda_grl:.2f} | L_cls={l_cls.item():.3f} "
                  f"L_diff={l_diff.item():.3f} L_adv={l_adv.item():.3f}")
    if ckpt is None:
        timings["stage1_s"] = time.time() - t0
        print(f"  Stage 1 done in {timings['stage1_s']:.0f} s")
        validate("after Stage 1")
    print()

    # --------------------------------------------------------------- stage 2
    print("=" * 64)
    print(f"STAGE 2 – {n_way}-way {args.k_shot}-shot episodic meta-training on P_t^tr "
          f"({args.stage2_steps} episodes)")
    print("=" * 64)
    opt = AdamW(model.parameters(), lr=cfg.stage2_lr, weight_decay=cfg.weight_decay)
    lr_warm2 = int(args.warmup_frac * args.stage2_steps)
    train_way = min(args.train_way or n_way, n_cls_t)
    train_query = args.train_query or args.q_query
    print(f"  training episodes: {train_way}-way {args.k_shot}-shot {train_query}-query, "
          f"lr {cfg.stage2_lr:g} ({args.schedule}, warm-up {lr_warm2}), augment={args.augment}")
    index = ByClassIndex(tr_y.tolist())
    rng = random.Random(args.seed)
    best = (-1.0, None, 0)
    ema_model = copy.deepcopy(model) if args.ema > 0 else None
    start, stage2_before = 1, 0.0
    if saved is not None:
        start = args.stage2_steps + 1           # --weights: no training
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        if ema_model is not None:
            ema_model.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        best = (ckpt["best"][0], ckpt["best"][1], ckpt["best"][2])
        _set_rng_state(ckpt["rng"], rng)
        start, stage2_before = ckpt["step"] + 1, ckpt["timings"]["stage2_s"]
        del ckpt
    t0 = time.time()
    for step in range(start, args.stage2_steps + 1):
        if args.stop_after and step > args.stop_after:
            path = out_dir / "resume.pt"
            torch.save({
                "step": step - 1, "args": _resume_args(args),
                "index_fingerprint": split["split_info"]["index_fingerprint"],
                "model": model.state_dict(), "ema": None if ema_model is None else ema_model.state_dict(),
                "opt": opt.state_dict(), "best": best, "rng": _rng_state(rng), "anchors": anchors,
                "history": history, "skipped": skipped,
                "timings": {"stage1_s": timings["stage1_s"], "stage2_s": stage2_before + time.time() - t0,
                            "total_s": time.time() - t_all},
            }, path)
            print(f"  stopped after episode {step - 1}; resume checkpoint → {path}")
            return
        for g in opt.param_groups:
            g["lr"] = cfg.stage2_lr * lr_factor(step, args.stage2_steps, lr_warm2, args.schedule)
        try:
            s_idx, q_idx, ep_classes = index.sample_episode(train_way, args.k_shot, train_query, rng)
        except ValueError:
            continue
        sx, sy, qx, qy = gather_episode(tr_x, tr_y, s_idx, q_idx, ep_classes)
        sx0 = model.project_bands(maybe_augment(sx.to(device)), is_source=False)
        qx0 = model.project_bands(maybe_augment(qx.to(device)), is_source=False)

        opt.zero_grad(set_to_none=True)
        l_epi, l_gcl, logits = model.stage2_episode(
            sx0, sy.to(device), qx0, qy.to(device), train_way,
            anchors[torch.as_tensor(ep_classes, dtype=torch.long)].to(device)
        )
        loss = total_objective_stage2(l_epi, l_gcl, cfg.loss)
        if not torch.isfinite(loss):
            skipped["stage2"] += 1
            continue
        loss.backward()
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        opt.step()
        if ema_model is not None:
            with torch.no_grad():
                for pe, pm in zip(ema_model.parameters(), model.parameters()):
                    pe.mul_(args.ema).add_(pm.detach(), alpha=1 - args.ema)
                for be, bm in zip(ema_model.buffers(), model.buffers()):
                    be.copy_(bm)

        acc = (logits.argmax(-1).cpu() == qy).float().mean().item()
        history["stage2"].append({"step": step, "l_epi": l_epi.item(),
                                  "l_gcl": l_gcl.item(), "acc": acc})
        if step == 1 or step % 25 == 0 or step == args.stage2_steps:
            recent = [h["acc"] for h in history["stage2"][-25:]]
            print(f"  episode {step:4d} | L_epi={l_epi.item():.3f} L_gcl={l_gcl.item():.3f} "
                  f"train acc (last 25)={np.mean(recent)*100:5.1f}%  "
                  f"lr={opt.param_groups[0]['lr']:.2e}", flush=True)
        if args.val_every and step % args.val_every == 0 and step < args.stage2_steps:
            net = ema_model or model
            v = validate(f"episode {step}", net)
            if v is not None and v > best[0]:
                best = (v, copy.deepcopy(net.state_dict()), step)
    timings["stage2_s"] = stage2_before + time.time() - t0
    print(f"  Stage 2 done in {timings['stage2_s']:.0f} s")
    if ema_model is not None:
        model.load_state_dict(ema_model.state_dict())   # evaluate the EMA weights
    v = validate("after Stage 2")
    if v is not None and v > best[0]:
        best = (v, None, args.stage2_steps)
    if args.select_on_val and best[1] is not None:
        model.load_state_dict(best[1])
        print(f"  using the checkpoint from episode {best[2]} (best validation OA {best[0]*100:.2f}%)")
    history["selected_episode"] = best[2] if saved is None else saved.get("selected_episode")
    n_tail = max(1, len(history["stage2"]) // 10)
    history["train_acc_last10pct"] = (float(np.mean([h["acc"] for h in history["stage2"][-n_tail:]]))
                                      if history["stage2"] else float("nan"))
    torch.save({"model": model.state_dict(), "anchors": anchors, "selected_episode": best[2],
                "args": vars(args)}, out_dir / "model_final.pt")
    print()

    def encode_pool(x):
        model.eval()
        out = []
        bsz = args.map_batch or (128 if device.type == "cuda" else 64)
        with torch.no_grad():
            for b in range(0, len(x), bsz):
                out.append(model.encode(model.project_bands(x[b:b + bsz].to(device), is_source=False)).cpu())
        return torch.cat(out)

    # transductive settings are chosen on the validation pool only
    tr_sel: dict = {}
    if args.transductive and val_y is None:
        print("  --transductive needs a validation pool: transductive results skipped")
    elif args.transductive:
        t_tr = time.time()
        grid = parse_grid(args.tr_grid)
        by_name = {config_name(c): c for c in grid}
        tr_val = {}
        print(f"Transductive settings, chosen on the validation pool ({len(grid)} candidates + inductive):")
        for k in sorted({1, args.k_shot}):
            m = evaluate_episodes(model, val_x, val_y, n_way, k, args.q_query, anchors, device, n_cls_t,
                                  n_episodes=args.val_episodes, seed=555 + eval_seed, transductive=grid)
            sc = {"inductive": m["oa_mean"], **{n: r["oa_mean"] for n, r in m["transductive"].items()}}
            pick = max(sc, key=sc.get)
            tr_sel[f"{n_way}way_{k}shot"] = by_name.get(pick)
            tr_val[f"{n_way}way_{k}shot"] = sc
            print(f"  {n_way}-way {k}-shot (validation OA): inductive {100 * sc['inductive']:.2f}%, "
                  f"best {pick} {100 * sc[pick]:.2f}%")
        vfeats, vlab = encode_pool(val_x), val_y.numpy()
        rsv = np.random.RandomState(eval_seed + 1)
        vdraws = [np.concatenate([rsv.choice(np.where(vlab == c)[0], size=min(args.k_shot, int((vlab == c).sum())),
                                             replace=False) for c in range(n_cls_t)])
                  for _ in range(args.tr_val_draws)]
        sc = {}
        with torch.no_grad():
            for c in [None] + grid:
                oas, aas = [], []
                for sup in vdraws:
                    st = torch.as_tensor(sup, dtype=torch.long)
                    P = model.dsspe.evolve(vfeats[st].to(device), val_y[st].to(device), n_cls_t,
                                           anchors[:n_cls_t].to(device))
                    if c is not None:
                        P = refine_prototypes(P, vfeats[st].to(device), val_y[st].to(device), vfeats.to(device), **c)
                    pr = model.dsspe.classify(vfeats.to(device), P).argmax(-1).cpu().numpy()
                    q = np.ones(len(vlab), dtype=bool)
                    q[sup] = False
                    r_ = classification_report(vlab[q], pr[q], n_cls_t)
                    oas.append(r_["oa"]); aas.append(r_["aa"])
                sc[config_name(c)] = (float(np.mean(oas)) + float(np.mean(aas))) / 2
        pick = max(sc, key=sc.get)
        tr_sel["map"] = by_name.get(pick)
        tr_val["map_mean_of_oa_aa"] = sc
        print(f"  9-class map (validation, mean of OA and AA over {args.tr_val_draws} draws): "
              f"inductive {100 * sc['inductive']:.2f}%, best {pick} {100 * sc[pick]:.2f}%")
        history["transductive_validation"] = {"scores": tr_val,
                                              "selected": {k: config_name(v) for k, v in tr_sel.items()}}
        timings["transductive_selection_s"] = time.time() - t_tr
        print()

    # ------------------------------------------------------------ evaluation
    print("=" * 64)
    print(f"EVALUATION – {args.eval_episodes} episodes on the held-out meta-testing pool P_t^te")
    print("=" * 64)
    t0 = time.time()
    metrics = {}
    for k in sorted({1, args.k_shot}):
        key = f"{n_way}way_{k}shot"
        cfg_t = tr_sel.get(key)
        m = evaluate_episodes(
            model, te_x, te_y, n_way, k, args.q_query, anchors, device, n_cls_t,
            n_episodes=args.eval_episodes, seed=10_000 + eval_seed,
            transductive=[cfg_t] if cfg_t else None,
        )
        if args.transductive and key in tr_sel:
            # the setting chosen on validation; "inductive" when no refinement won there
            m["transductive_selected"] = dict(m.pop("transductive")[config_name(cfg_t)] if cfg_t else
                                              {kk: vv for kk, vv in m.items() if kk != "transductive"},
                                              config=config_name(cfg_t))
        metrics[key] = m
        print(f"  {n_way}-way {k}-shot: {format_metrics(m)}")
        print("    per-class accuracy: " + ", ".join(
            "-" if a is None else f"{100 * a:.1f}" for a in m.get("per_class_acc", [])))
        if "transductive_selected" in m:
            print(f"    transductive ({m['transductive_selected']['config']}): "
                  f"{format_metrics(m['transductive_selected'])}")
    timings["eval_s"] = time.time() - t0

    # ---------------------------------------------------- classification map
    pred_map, map_report, map_summary = None, None, None
    names = CLASS_NAMES.get(tgt_ds.name, [str(c) for c in target_classes])
    cname = lambda i: names[target_classes[i] - 1] if len(names) >= target_classes[i] else str(target_classes[i])
    if not args.no_map:
        t0 = time.time()
        k = args.k_shot
        print(f"\nClassification map: {n_cls_t}-way, {k} labelled pixels per class drawn from P_t^te "
              f"({args.map_repeats} independent draws); all other P_t^te pixels are queries ...")
        model.eval()
        te_lab = te_y.numpy()
        bs = args.map_batch or (128 if device.type == "cuda" else 64)
        feats = []
        t_enc = time.time()
        with torch.no_grad():
            for b in range(0, len(te_lab), bs):
                xb = te_x[b:b + bs].to(device)
                feats.append(model.encode(model.project_bands(xb, is_source=False)).cpu())
        feats = torch.cat(feats)
        timings["inference_ms_per_patch"] = 1000 * (time.time() - t_enc) / len(te_lab)
        rs = np.random.RandomState(eval_seed)
        te_coords = split["target_te_coords"]
        H, W = tgt_ds.gt.shape
        cm_sum = np.zeros((n_cls_t, n_cls_t), dtype=np.int64)
        reps, reps_sep = [], []
        cfg_map = tr_sel.get("map") if "map" in tr_sel else None
        do_tr = "map" in tr_sel
        cm_t = np.zeros((n_cls_t, n_cls_t), dtype=np.int64)
        reps_t, reps_t_sep = [], []
        with torch.no_grad():
            for r in range(args.map_repeats):
                sup_idx = np.concatenate([
                    rs.choice(np.where(te_lab == c)[0], size=min(k, int((te_lab == c).sum())), replace=False)
                    for c in range(n_cls_t)])
                sup_t = torch.as_tensor(sup_idx, dtype=torch.long)
                protos = model.dsspe.evolve(feats[sup_t].to(device), te_y[sup_t].to(device),
                                            n_cls_t, anchors[:n_cls_t].to(device))
                preds = model.dsspe.classify(feats.to(device), protos).argmax(-1).cpu().numpy()
                q = np.ones(len(te_lab), dtype=bool)
                q[sup_idx] = False
                rep = classification_report(te_lab[q], preds[q], n_cls_t)
                reps.append(rep)
                cm_sum += np.array(rep["confusion_matrix"])
                # stricter variant: drop queries whose patch shares input pixels with a support patch
                img = np.zeros((H, W), dtype=np.uint8)
                img[te_coords[sup_idx, 0], te_coords[sup_idx, 1]] = 1
                near = maximum_filter(img, size=2 * args.patch_size - 1)[te_coords[:, 0], te_coords[:, 1]] > 0
                qs = q & ~near
                rep_s = classification_report(te_lab[qs], preds[qs], n_cls_t)
                rep_s["excluded"] = int((q & near).sum())
                reps_sep.append(rep_s)
                if do_tr:   # same supports, prototypes refined on all unlabelled test-region pixels
                    P_t = protos if cfg_map is None else refine_prototypes(
                        protos, feats[sup_t].to(device), te_y[sup_t].to(device), feats.to(device), **cfg_map)
                    preds_t = model.dsspe.classify(feats.to(device), P_t).argmax(-1).cpu().numpy()
                    rt = classification_report(te_lab[q], preds_t[q], n_cls_t)
                    reps_t.append(rt)
                    cm_t += np.array(rt["confusion_matrix"])
                    reps_t_sep.append(classification_report(te_lab[qs], preds_t[qs], n_cls_t))
                if r == 0:
                    coords = split["target_te_coords"]
                    pred_map = np.zeros(tgt_ds.gt.shape, dtype=np.int64)
                    pred_map[coords[:, 0], coords[:, 1]] = preds + 1
        timings["map_s"] = time.time() - t0

        mean_sd = lambda key: (float(np.mean([r[key] for r in reps])),
                               float(np.std([r[key] for r in reps], ddof=1)) if len(reps) > 1 else 0.0)
        oa_m, oa_s = mean_sd("oa"); aa_m, aa_s = mean_sd("aa"); kp_m, kp_s = mean_sd("kappa")
        rec = np.array([[np.nan if v is None else v for v in r["per_class"]] for r in reps])
        pooled = metrics_from_confusion(cm_sum)
        rown = cm_sum / np.maximum(cm_sum.sum(1, keepdims=True), 1)
        map_report = {"oa": oa_m, "aa": aa_m, "kappa": kp_m}
        map_summary = {
            "repeats": len(reps), "oa_mean": oa_m, "oa_sd": oa_s, "aa_mean": aa_m, "aa_sd": aa_s,
            "kappa_mean": kp_m, "kappa_sd": kp_s,
            "recall_mean": np.nanmean(rec, 0).tolist(), "recall_sd": np.nanstd(rec, 0, ddof=1).tolist(),
            "pooled_precision": pooled["precision"], "pooled_recall": pooled["per_class"],
            "pooled_f1": pooled["f1"], "pooled_confusion": cm_sum.tolist(),
            "query_support_per_draw": (cm_sum.sum(1) // len(reps)).tolist(),
        }
        so, ss_ = (float(np.mean([r["oa"] for r in reps_sep])), float(np.std([r["oa"] for r in reps_sep], ddof=1)) if len(reps_sep) > 1 else 0.0)
        sa, sas = (float(np.mean([r["aa"] for r in reps_sep])), float(np.std([r["aa"] for r in reps_sep], ddof=1)) if len(reps_sep) > 1 else 0.0)
        rec_s = np.array([[np.nan if v is None else v for v in r["per_class"]] for r in reps_sep])
        map_summary["support_separated"] = {
            "oa_mean": so, "oa_sd": ss_, "aa_mean": sa, "aa_sd": sas,
            "recall_mean": np.nanmean(rec_s, 0).tolist(),
            "excluded_queries_mean": float(np.mean([r["excluded"] for r in reps_sep])),
        }
        print(f"  OA {100 * oa_m:.2f} ± {100 * oa_s:.2f} %   AA {100 * aa_m:.2f} ± {100 * aa_s:.2f} %   "
              f"κ {kp_m:.4f} ± {kp_s:.4f}   ({len(reps)} draws, "
              f"{timings['inference_ms_per_patch']:.2f} ms/patch inference)")
        print(f"  {'class':<22}{'recall':>14}{'prec':>7}{'F1':>7}   row-normalised confusion (%)")
        for i in range(n_cls_t):
            print(f"  {cname(i):<22}{100 * np.nanmean(rec[:, i]):7.1f} ±{100 * np.nanstd(rec[:, i], ddof=1):4.1f}"
                  f"{100 * (pooled['precision'][i] or 0):7.1f}{100 * (pooled['f1'][i] or 0):7.1f}   "
                  + " ".join(f"{100 * v:3.0f}" for v in rown[i]))
        off = sorted(((rown[i, j], i, j) for i in range(n_cls_t) for j in range(n_cls_t) if i != j), reverse=True)[:5]
        print("  most frequent confusions: " + "; ".join(
            f"{cname(i)} → {cname(j)} {100 * v:.1f}%" for v, i, j in off))
        print(f"  confusion matrix, counts per draw (mean of {len(reps)} draws; rows true, columns predicted):")
        for i in range(n_cls_t):
            print(f"  {cname(i):<22}" + "".join(f"{v / len(reps):8.1f}" for v in cm_sum[i])
                  + f"   | queries {cm_sum[i].sum() // len(reps)}")
        ssep = map_summary["support_separated"]
        print(f"  queries not sharing input pixels with any support patch "
              f"({ssep['excluded_queries_mean']:.0f} queries per draw excluded): "
              f"OA {100 * so:.2f} ± {100 * ss_:.2f} %   AA {100 * sa:.2f} ± {100 * sas:.2f} %")
        if do_tr:
            msd = lambda rs_, key: (float(np.mean([r[key] for r in rs_])),
                                    float(np.std([r[key] for r in rs_], ddof=1)) if len(rs_) > 1 else 0.0)
            pooled_t = metrics_from_confusion(cm_t)
            rec_t = np.array([[np.nan if v is None else v for v in r["per_class"]] for r in reps_t])
            rec_ts = np.array([[np.nan if v is None else v for v in r["per_class"]] for r in reps_t_sep])
            (to, tos), (ta, tas), (tk, tks) = msd(reps_t, "oa"), msd(reps_t, "aa"), msd(reps_t, "kappa")
            (tso, tsos), (tsa, tsas) = msd(reps_t_sep, "oa"), msd(reps_t_sep, "aa")
            map_summary["transductive"] = {
                "config": config_name(cfg_map), "oa_mean": to, "oa_sd": tos, "aa_mean": ta, "aa_sd": tas,
                "kappa_mean": tk, "kappa_sd": tks,
                "recall_mean": np.nanmean(rec_t, 0).tolist(), "recall_sd": np.nanstd(rec_t, 0, ddof=1).tolist(),
                "pooled_precision": pooled_t["precision"], "pooled_recall": pooled_t["per_class"],
                "pooled_f1": pooled_t["f1"], "pooled_confusion": cm_t.tolist(),
                "support_separated": {"oa_mean": tso, "oa_sd": tsos, "aa_mean": tsa, "aa_sd": tsas,
                                      "recall_mean": np.nanmean(rec_ts, 0).tolist()},
            }
            rown_t = cm_t / np.maximum(cm_t.sum(1, keepdims=True), 1)
            print(f"\n  TRANSDUCTIVE map ({config_name(cfg_map)}, chosen on validation; same supports): "
                  f"OA {100 * to:.2f} ± {100 * tos:.2f} %   AA {100 * ta:.2f} ± {100 * tas:.2f} %   κ {tk:.4f}")
            print(f"  {'class':<22}{'recall':>14}{'prec':>7}{'F1':>7}   row-normalised confusion (%)")
            for i in range(n_cls_t):
                print(f"  {cname(i):<22}{100 * np.nanmean(rec_t[:, i]):7.1f} ±{100 * np.nanstd(rec_t[:, i], ddof=1):4.1f}"
                      f"{100 * (pooled_t['precision'][i] or 0):7.1f}{100 * (pooled_t['f1'][i] or 0):7.1f}   "
                      + " ".join(f"{100 * v:3.0f}" for v in rown_t[i]))
            print(f"  transductive, support-separated queries: OA {100 * tso:.2f} ± {100 * tsos:.2f} %   "
                  f"AA {100 * tsa:.2f} ± {100 * tsas:.2f} %")

    timings["total_s"] = time.time() - t_all

    # --------------------------------------------------------------- outputs
    title = (f"DGM-HMamba demo: {args.source} → {args.target} | {n_params/1e6:.2f} M-param demo model, "
             f"{args.stage1_iters} + {args.stage2_steps} iterations (not the paper configuration)")
    try:
        figs = save_figures(out_dir, tgt_ds, target_classes, pred_map, history, map_report, title)
        for f in figs:
            print(f"  figure → {f}")
    except ImportError:
        print("  (matplotlib not installed: figures skipped)")

    summary = {
        "note": "Demo configuration with short training; these are not the manuscript's results.",
        "args": vars(args),
        "seeds": {"model": args.seed, "split": split_seed, "eval": eval_seed},
        "device": str(device),
        "torch": torch.__version__,
        "trainable_parameters": n_params,
        "pool_sizes": {"source": len(src_y), "target_meta_train": len(tr_y),
                       "target_validation": 0 if val_y is None else len(val_y),
                       "target_meta_test": len(te_y)},
        "episodic_metrics": metrics,
        "map_metrics": map_summary,
        "split_info": split["split_info"],
        "timings_s": timings,
        "history": history,
    }
    with open(out_dir / "demo_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary → {out_dir / 'demo_summary.json'}")
    print(f"\nDemo finished in {timings['total_s']/60:.1f} min.")


if __name__ == "__main__":
    main()
