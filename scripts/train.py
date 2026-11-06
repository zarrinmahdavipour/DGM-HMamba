#!/usr/bin/env python3
"""
Full training entry-point for DGM-HMamba with logging and checkpoints.

Features
--------
- Stage 1 (source pre-training + domain-adversarial diffusion)
- Stage 2 (episodic meta-adaptation on target)
- Console + file logging
- Checkpoint save / resume
- Optional validation pool carved from the target meta-training pool
  (--val-fraction) for monitoring and checkpoint selection
- One final evaluation on the held-out target meta-testing pool
  (its labels never influence training or model selection)
- OA (mean +/- sample SD over episodes), AA and Cohen's kappa
- Reproducible seeds

Example
-------
    python scripts/train.py \\
        --source indian_pines --target pavia_u \\
        --data-root ./data \\
        --k-shot 5 --n-way 5 \\
        --stage1-epochs 200 --stage2-steps 100 \\
        --seed 42 --output-dir runs/ip2pu_5shot_s42 \\
        --device cuda

Resume:
    python scripts/train.py --resume runs/ip2pu_5shot/checkpoint_last.pt ...
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dgm_hmamba import get_default_config, DGMHMamba
from dgm_hmamba.console import use_utf8_stdout
from dgm_hmamba.data import get_dataset, make_cross_domain_split, split_validation
from dgm_hmamba.episodes import ByClassIndex, gather_episode
from dgm_hmamba.losses import total_objective_stage1, total_objective_stage2
from dgm_hmamba.evaluation import evaluate_episodes as evaluate, format_metrics


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(output_dir: Path, name: str = "train") -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s – %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    fh = logging.FileHandler(output_dir / "train.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[object],
    meta: Dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "meta": meta,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("meta", {})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train DGM-HMamba")
    p.add_argument("--source", type=str, default="indian_pines")
    p.add_argument("--target", type=str, default="pavia_u")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--n-way", type=int, default=5)
    p.add_argument("--k-shot", type=int, default=5)
    p.add_argument("--q-query", type=int, default=15)
    p.add_argument("--stage1-epochs", type=int, default=200, help="paper: 200")
    p.add_argument("--stage2-steps", type=int, default=100, help="paper: 100")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--val-fraction", type=float, default=0.0,
                   help="fraction of each class of the target meta-training pool held "
                        "out for validation / checkpoint selection (0 = use the final "
                        "Stage-2 model, as in the paper)")
    p.add_argument("--eval-every", type=int, default=50,
                   help="validate every N stage-2 steps (only if --val-fraction > 0)")
    p.add_argument("--eval-episodes", type=int, default=50,
                   help="episodes for validation and for the final meta-test evaluation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--preset", choices=["auto", "paper", "reduced", "tiny"], default="auto",
                   help="model size. paper: manuscript configuration (config.py); "
                        "reduced: widths 64/128/256, 1 block per level, T=20; "
                        "tiny: demo-size model for smoke tests; "
                        "auto: paper on GPU, reduced on CPU")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def main():
    use_utf8_stdout()
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    # Output directory
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"runs/{args.source}_to_{args.target}_{args.k_shot}shot_{stamp}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir)
    logger.info("Command: %s", " ".join(sys.argv))
    logger.info("Output directory: %s", output_dir.resolve())

    # Save args
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    cfg = get_default_config()
    cfg.episode.n_way = args.n_way
    cfg.episode.k_shot = args.k_shot
    cfg.episode.q_query = args.q_query
    cfg.stage1_epochs = args.stage1_epochs
    cfg.stage2_steps = args.stage2_steps

    # Model size preset
    preset = args.preset
    if preset == "auto":
        preset = "reduced" if device.type == "cpu" else "paper"
    if preset == "reduced":
        cfg.hdpm.widths = (64, 128, 256)
        cfg.hdpm.blocks_per_level = 1
        cfg.dadge.denoiser_base_channels = 32
        cfg.dadge.T = 20
        cfg.dadge.default_inference_steps = 20  # = T (strided DDPM is only valid with DDIM)
    elif preset == "tiny":
        cfg.hdpm.widths = (32, 32, 32)
        cfg.hdpm.blocks_per_level = 1
        cfg.hdpm.state_dim = 4
        cfg.hdpm.expand_factor = 1
        cfg.dadge.denoiser_base_channels = 32
        cfg.dadge.denoiser_num_res_blocks = 1
        cfg.dadge.T = 10
        cfg.dadge.default_inference_steps = 10
    logger.info("Model preset: %s%s", preset,
                "" if preset == "paper" else "  (NOT the manuscript configuration)")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    logger.info("Loading datasets ...")
    source_ds = get_dataset(args.source, args.data_root)
    target_ds = get_dataset(args.target, args.data_root)
    logger.info("  Source: %s", source_ds)
    logger.info("  Target: %s", target_ds)

    split = make_cross_domain_split(
        source_ds, target_ds,
        patch_size=cfg.hdpm.patch_size,
        seed=args.seed,
    )
    logger.info(
        "  Source patches : %5d  (%d classes)",
        len(split["source_labels"]), len(split["source_classes"]),
    )
    logger.info(
        "  Target meta-tr : %5d   meta-te : %5d",
        len(split["target_tr_labels"]), len(split["target_te_labels"]),
    )

    src_x = torch.from_numpy(split["source_patches"])
    src_y = torch.from_numpy(split["source_labels"])
    tgt_tr_x = torch.from_numpy(split["target_tr_patches"])
    tgt_tr_y = torch.from_numpy(split["target_tr_labels"])
    tgt_te_x = torch.from_numpy(split["target_te_patches"])
    tgt_te_y = torch.from_numpy(split["target_te_labels"])
    n_target_classes = len(split["target_classes"])

    # Optional validation pool inside P_t^tr (never touches P_t^te)
    val_x = val_y = None
    if args.val_fraction > 0:
        keep_idx, val_idx = split_validation(
            split["target_tr_labels"], args.val_fraction, seed=args.seed + 1
        )
        keep_idx = torch.as_tensor(keep_idx, dtype=torch.long)
        val_idx = torch.as_tensor(val_idx, dtype=torch.long)
        val_x, val_y = tgt_tr_x[val_idx], tgt_tr_y[val_idx]
        tgt_tr_x, tgt_tr_y = tgt_tr_x[keep_idx], tgt_tr_y[keep_idx]
        logger.info(
            "  Validation pool carved from meta-tr: %d (meta-tr now %d)",
            len(val_y), len(tgt_tr_y),
        )

    n_way = min(args.n_way, n_target_classes)
    # NOTE: placeholder anchors. GloVe class-name anchors are not implemented
    # yet (see README, "Known differences from the manuscript").
    anchors = torch.randn(n_way, cfg.dsspe.anchor_glove_dim)

    # ------------------------------------------------------------------
    # Model & optimiser
    # ------------------------------------------------------------------
    model = DGMHMamba(
        cfg,
        source_bands=source_ds.n_bands,
        target_bands=target_ds.n_bands,
        num_source_classes=len(split["source_classes"]),
    ).to(device)

    opt = AdamW(model.parameters(), lr=cfg.stage1_lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(opt, T_max=max(1, cfg.stage1_epochs))

    start_epoch = 1
    start_step = 1
    best_oa = 0.0

    if args.resume is not None:
        logger.info("Resuming from %s", args.resume)
        meta = load_checkpoint(Path(args.resume), model, opt, scheduler, device)
        start_epoch = meta.get("epoch", 1)
        start_step = meta.get("step", 1)
        best_oa = meta.get("best_oa", 0.0)
        logger.info("  Resumed at epoch=%d step=%d best_oa=%.2f%%",
                    start_epoch, start_step, best_oa * 100)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %.2f M", n_params / 1e6)

    # ------------------------------------------------------------------
    # STAGE 1
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 1 – Source pre-training + domain-adversarial diffusion")
    logger.info("=" * 60)

    src_loader = DataLoader(
        TensorDataset(src_x, src_y),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    tgt_unlab_loader = DataLoader(
        TensorDataset(tgt_tr_x),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )

    model.train()
    t0 = time.time()

    for epoch in range(start_epoch, cfg.stage1_epochs + 1):
        lambda_grl = min(1.0, epoch / max(1, cfg.dadge.grl_warmup_epochs))
        tgt_iter = iter(tgt_unlab_loader)
        running = {"cls": 0.0, "diff": 0.0, "adv": 0.0, "n": 0}

        pbar = tqdm(src_loader, desc=f"Stage1 Ep {epoch:03d}", leave=False)
        for x_src, y_src in pbar:
            try:
                x_tgt = next(tgt_iter)[0]
            except StopIteration:
                tgt_iter = iter(tgt_unlab_loader)
                x_tgt = next(tgt_iter)[0]

            x_src = x_src.to(device)
            y_src = y_src.to(device)
            x_tgt = x_tgt.to(device)

            x0_src = model.project_bands(x_src, is_source=True)
            x0_tgt = model.project_bands(x_tgt, is_source=False)

            opt.zero_grad()
            l_cls, l_diff, l_adv = model.stage1_losses(
                x0_src, y_src, x0_tgt, lambda_grl
            )
            loss = total_objective_stage1(l_cls, l_diff, l_adv, cfg.loss)
            loss.backward()
            opt.step()

            running["cls"] += l_cls.item()
            running["diff"] += l_diff.item()
            running["adv"] += l_adv.item()
            running["n"] += 1
            pbar.set_postfix(
                cls=f"{l_cls.item():.3f}",
                diff=f"{l_diff.item():.3f}",
                adv=f"{l_adv.item():.3f}",
            )

        scheduler.step()
        n = max(1, running["n"])
        logger.info(
            "  Epoch %03d/%d  λ_grl=%.2f  L_cls=%.4f  L_diff=%.4f  L_adv=%.4f  lr=%.2e",
            epoch, cfg.stage1_epochs, lambda_grl,
            running["cls"] / n, running["diff"] / n, running["adv"] / n,
            opt.param_groups[0]["lr"],
        )

        # Checkpoint after every epoch of Stage 1
        save_checkpoint(
            output_dir / "checkpoint_last.pt",
            model, opt, scheduler,
            meta={"epoch": epoch + 1, "step": 1, "best_oa": best_oa, "stage": 1},
        )

    logger.info("Stage 1 finished in %.1f min", (time.time() - t0) / 60)

    # ------------------------------------------------------------------
    # STAGE 2
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 2 – Episodic meta-adaptation")
    logger.info("=" * 60)

    # Switch learning rate
    for g in opt.param_groups:
        g["lr"] = cfg.stage2_lr

    index = ByClassIndex(tgt_tr_y.tolist())
    rng = random.Random(args.seed)
    t0 = time.time()

    model.train()
    for step in range(start_step, cfg.stage2_steps + 1):
        try:
            s_idx, q_idx, ep_classes = index.sample_episode(
                n_way, args.k_shot, args.q_query, rng
            )
        except ValueError:
            continue

        support_x, support_y, query_x, query_y = gather_episode(
            tgt_tr_x, tgt_tr_y, s_idx, q_idx, ep_classes
        )
        ep_anchors = anchors[:n_way]

        support_x0 = model.project_bands(support_x.to(device), is_source=False)
        query_x0 = model.project_bands(query_x.to(device), is_source=False)

        opt.zero_grad()
        l_epi, l_gcl, logits = model.stage2_episode(
            support_x0,
            support_y.to(device),
            query_x0,
            query_y.to(device),
            n_way,
            ep_anchors.to(device),
        )
        loss = total_objective_stage2(l_epi, l_gcl, cfg.loss)
        loss.backward()
        opt.step()

        if step % 20 == 0 or step == 1:
            acc = (logits.argmax(-1).cpu() == query_y).float().mean().item()
            logger.info(
                "  Step %04d/%d  L_epi=%.4f  L_gcl=%.4f  acc=%.1f%%",
                step, cfg.stage2_steps, l_epi.item(), l_gcl.item(), acc * 100,
            )

        # Periodic checkpoint (+ validation inside P_t^tr if requested)
        if step % args.eval_every == 0 or step == cfg.stage2_steps:
            save_checkpoint(
                output_dir / "checkpoint_last.pt",
                model, opt, None,
                meta={"epoch": cfg.stage1_epochs + 1, "step": step + 1,
                      "best_oa": best_oa, "stage": 2},
            )
            if val_x is not None:
                metrics = evaluate(
                    model, val_x, val_y,
                    n_way, args.k_shot, args.q_query,
                    anchors, device, n_target_classes,
                    n_episodes=args.eval_episodes,
                    seed=args.seed + step,
                )
                oa = metrics["oa_mean"]
                logger.info("  [Val @ step %d]  %s", step, format_metrics(metrics))
                if oa > best_oa:
                    best_oa = oa
                    save_checkpoint(
                        output_dir / "checkpoint_best.pt",
                        model, opt, None,
                        meta={"epoch": cfg.stage1_epochs + 1, "step": step,
                              "best_oa": best_oa, "stage": 2},
                    )
                    logger.info("  → New best model on validation (OA=%.2f%%)", best_oa * 100)

    logger.info("Stage 2 finished in %.1f min", (time.time() - t0) / 60)

    # ------------------------------------------------------------------
    # Final evaluation on the held-out meta-testing pool (run once)
    # ------------------------------------------------------------------
    best_path = output_dir / "checkpoint_best.pt"
    if val_x is not None and best_path.is_file():
        load_checkpoint(best_path, model, device=device)
        selected = "best validation checkpoint"
    else:
        selected = "final Stage-2 model"

    test_metrics = evaluate(
        model, tgt_te_x, tgt_te_y,
        n_way, args.k_shot, args.q_query,
        anchors, device, n_target_classes,
        n_episodes=args.eval_episodes,
        seed=10_000 + args.seed,
    )
    logger.info("Meta-test (%s): %s", selected, format_metrics(test_metrics))
    with open(output_dir / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"model": selected, **test_metrics}, f, indent=2)

    logger.info("Checkpoints and metrics saved under: %s", output_dir.resolve())
    logger.info("Done.")


if __name__ == "__main__":
    main()
