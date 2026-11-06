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
from dgm_hmamba.metrics import classification_report
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
    if args.stage1_lr is not None:
        cfg.stage1_lr = args.stage1_lr
    if args.stage2_lr is not None:
        cfg.stage2_lr = args.stage2_lr
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    cfg.dadge.bypass_enhancement = args.no_diffusion
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
    p.add_argument("--select-on-val", action="store_true",
                   help="evaluate the Stage-2 checkpoint with the best validation OA")
    p.add_argument("--no-map", action="store_true", help="skip the classification map")
    p.add_argument("--map-batch", type=int, default=None,
                   help="patches per forward pass for the map (default 128 on GPU, 64 on CPU)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


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


def main():
    use_utf8_stdout()
    args = parse_args()
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
    tgt_ds = get_dataset(args.target, args.data_root)
    cfg = demo_config(args)
    split = make_cross_domain_split(src_ds, tgt_ds, patch_size=args.patch_size, seed=args.seed)

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
    if args.val_fraction > 0:
        keep, val = split_validation(split["target_tr_labels"], args.val_fraction, seed=args.seed + 1)
        keep, val = torch.as_tensor(keep, dtype=torch.long), torch.as_tensor(val, dtype=torch.long)
        val_x, val_y = tr_x[val], tr_y[val]
        tr_x, tr_y = tr_x[keep], tr_y[keep]
    timings["data_s"] = time.time() - t0

    print(f"  Source  {src_ds}  → {len(src_y)} labelled patches")
    print(f"  Target  {tgt_ds}")
    print(f"          meta-training pool P_t^tr: {len(tr_y)}   meta-testing pool P_t^te: {len(te_y)}")
    if val_y is not None:
        print(f"          validation pool (held out from P_t^tr): {len(val_y)}")

    # ----------------------------------------------------------------- model
    model = DGMHMamba(cfg, src_ds.n_bands, tgt_ds.n_bands, len(split["source_classes"])).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Demo model: {n_params/1e6:.3f} M trainable parameters "
          f"(widths {cfg.hdpm.widths}, T={cfg.dadge.T}, {args.patch_size}×{args.patch_size} patches)")
    print(f"  Settings: {vars(args)}\n")

    # Placeholder semantic anchors (GloVe class-name anchors not implemented yet)
    anchors = torch.randn(n_cls_t, cfg.dsspe.anchor_glove_dim)
    history = {"stage1": [], "stage2": [], "validation": {}}

    def validate(tag):
        if val_y is None:
            return None
        m = evaluate_episodes(model, val_x, val_y, n_way, args.k_shot, args.q_query, anchors,
                              device, n_cls_t, n_episodes=args.val_episodes, seed=777 + args.seed)
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
    warmup = max(1, int(0.15 * args.stage1_iters))   # paper: λ_grl ramps over 30 of 200 epochs
    lr_warm1 = int(args.warmup_frac * args.stage1_iters)
    gen = torch.Generator().manual_seed(args.seed)
    model.train()
    t0 = time.time()
    for it in range(1, args.stage1_iters + 1):
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
        loss.backward()
        opt.step()

        history["stage1"].append({
            "iter": it, "lambda_grl": lambda_grl, "l_cls": l_cls.item(),
            "l_diff": l_diff.item(), "l_adv": l_adv.item(), "total": loss.item(),
        })
        if it == 1 or it % 20 == 0 or it == args.stage1_iters:
            print(f"  iter {it:4d} | λ_grl={lambda_grl:.2f} | L_cls={l_cls.item():.3f} "
                  f"L_diff={l_diff.item():.3f} L_adv={l_adv.item():.3f}")
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
    t0 = time.time()
    for step in range(1, args.stage2_steps + 1):
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
        loss.backward()
        opt.step()

        acc = (logits.argmax(-1).cpu() == qy).float().mean().item()
        history["stage2"].append({"step": step, "l_epi": l_epi.item(),
                                  "l_gcl": l_gcl.item(), "acc": acc})
        if step == 1 or step % 25 == 0 or step == args.stage2_steps:
            recent = [h["acc"] for h in history["stage2"][-25:]]
            print(f"  episode {step:4d} | L_epi={l_epi.item():.3f} L_gcl={l_gcl.item():.3f} "
                  f"train acc (last 25)={np.mean(recent)*100:5.1f}%  "
                  f"lr={opt.param_groups[0]['lr']:.2e}", flush=True)
        if args.val_every and step % args.val_every == 0 and step < args.stage2_steps:
            v = validate(f"episode {step}")
            if v is not None and v > best[0]:
                best = (v, copy.deepcopy(model.state_dict()), step)
    timings["stage2_s"] = time.time() - t0
    print(f"  Stage 2 done in {timings['stage2_s']:.0f} s")
    v = validate("after Stage 2")
    if v is not None and v > best[0]:
        best = (v, None, args.stage2_steps)
    if args.select_on_val and best[1] is not None:
        model.load_state_dict(best[1])
        print(f"  using the checkpoint from episode {best[2]} (best validation OA {best[0]*100:.2f}%)")
    history["selected_episode"] = best[2]
    n_tail = max(1, len(history["stage2"]) // 10)
    history["train_acc_last10pct"] = float(np.mean([h["acc"] for h in history["stage2"][-n_tail:]]))
    print()

    # ------------------------------------------------------------ evaluation
    print("=" * 64)
    print(f"EVALUATION – {args.eval_episodes} episodes on the held-out meta-testing pool P_t^te")
    print("=" * 64)
    t0 = time.time()
    metrics = {}
    for k in sorted({1, args.k_shot}):
        m = evaluate_episodes(
            model, te_x, te_y, n_way, k, args.q_query, anchors, device, n_cls_t,
            n_episodes=args.eval_episodes, seed=10_000 + args.seed,
        )
        metrics[f"{n_way}way_{k}shot"] = m
        print(f"  {n_way}-way {k}-shot: {format_metrics(m)}")
    timings["eval_s"] = time.time() - t0

    # ---------------------------------------------------- classification map
    pred_map, map_report = None, None
    if not args.no_map:
        t0 = time.time()
        k = args.k_shot
        print(f"\nClassification map: {n_cls_t}-way, {k} labelled pixels per class from P_t^te, "
              f"all other P_t^te pixels are queries ...")
        model.eval()
        te_lab = te_y.numpy()
        rs = np.random.RandomState(args.seed)
        sup_idx = []
        for c in range(n_cls_t):
            idx = np.where(te_lab == c)[0]
            sup_idx.extend(rs.choice(idx, size=min(k, len(idx)), replace=False).tolist())
        sup_t = torch.as_tensor(sup_idx, dtype=torch.long)
        is_sup = np.zeros(len(te_lab), dtype=bool)
        is_sup[sup_idx] = True

        bs = args.map_batch or (128 if device.type == "cuda" else 64)
        preds = np.empty(len(te_lab), dtype=np.int64)
        with torch.no_grad():
            f_s = model.encode(model.project_bands(te_x[sup_t].to(device), is_source=False))
            protos = model.dsspe.evolve(
                f_s, te_y[sup_t].to(device), n_cls_t, anchors[:n_cls_t].to(device)
            )
            for b in range(0, len(te_lab), bs):
                xb = te_x[b:b + bs].to(device)
                f_q = model.encode(model.project_bands(xb, is_source=False))
                preds[b:b + len(xb)] = model.dsspe.classify(f_q, protos).argmax(-1).cpu().numpy()

        q = ~is_sup
        map_report = classification_report(te_lab[q], preds[q], n_cls_t)
        coords = split["target_te_coords"]
        pred_map = np.zeros(tgt_ds.gt.shape, dtype=np.int64)
        pred_map[coords[:, 0], coords[:, 1]] = preds + 1
        timings["map_s"] = time.time() - t0
        print(f"  {int(q.sum())} query pixels: OA {map_report['oa']*100:.2f}%  "
              f"AA {map_report['aa']*100:.2f}%  κ {map_report['kappa']:.4f}  "
              f"({timings['map_s']:.0f} s)")

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
        "device": str(device),
        "torch": torch.__version__,
        "trainable_parameters": n_params,
        "pool_sizes": {"source": len(src_y), "target_meta_train": len(tr_y),
                       "target_meta_test": len(te_y)},
        "episodic_metrics": metrics,
        "map_metrics": None if map_report is None else
            {k: map_report[k] for k in ("oa", "aa", "kappa", "per_class")},
        "timings_s": timings,
        "history": history,
    }
    with open(out_dir / "demo_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary → {out_dir / 'demo_summary.json'}")
    print(f"\nDemo finished in {timings['total_s']/60:.1f} min.")


if __name__ == "__main__":
    main()
