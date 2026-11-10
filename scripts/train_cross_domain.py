#!/usr/bin/env python3
"""
Full cross-domain training script for DGM-HMamba.

Example
-------
    # Indian Pines → Pavia University  (5-shot)
    python scripts/train_cross_domain.py \\
        --source indian_pines --target pavia_u \\
        --data-root ./data \\
        --k-shot 5 --n-way 5 \\
        --stage1-epochs 200 --stage2-steps 100 \\
        --device cuda

The script automatically:
  1. Loads the two scenes
  2. Extracts patches
  3. Builds the source / target-meta-train / target-meta-test splits
  4. Runs Stage 1 (source + adversarial diffusion)
  5. Runs Stage 2 (episodic meta-adaptation on target meta-train pool)
  6. Evaluates on the held-out target meta-test pool
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dgm_hmamba import get_default_config, DGMHMamba
from dgm_hmamba.console import use_utf8_stdout
from dgm_hmamba.data import get_dataset, make_cross_domain_split
from dgm_hmamba.episodes import ByClassIndex, gather_episode
from dgm_hmamba.losses import total_objective_stage1, total_objective_stage2


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_episode(model, support_x, support_y, query_x, query_y, n_way, anchors, device):
    model.eval()
    with torch.no_grad():
        preds = model.predict_episode(
            support_x.to(device),
            support_y.to(device),
            query_x.to(device),
            n_way,
            anchors.to(device),
        )
    return (preds.cpu() == query_y).float().mean().item()


def main():
    use_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="indian_pines")
    parser.add_argument("--target", type=str, default="pavia_u")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--q-query", type=int, default=15)
    parser.add_argument("--stage1-epochs", type=int, default=200)  # paper
    parser.add_argument("--stage2-steps", type=int, default=100)   # paper
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-episodes", type=int, default=20)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Transfer: {args.source} → {args.target}  ({args.k_shot}-shot)\n")

    # ------------------------------------------------------------------
    # Config (slightly reduced widths for reasonable memory)
    # ------------------------------------------------------------------
    cfg = get_default_config()
    cfg.episode.n_way = args.n_way
    cfg.episode.k_shot = args.k_shot
    cfg.episode.q_query = args.q_query
    cfg.stage1_epochs = args.stage1_epochs
    cfg.stage2_steps = args.stage2_steps
    # Keep full paper widths only if GPU memory allows; otherwise reduce
    if device.type == "cpu":
        cfg.hdpm.widths = (64, 128, 256)
        cfg.hdpm.blocks_per_level = 1
        cfg.dadge.denoiser_base_channels = 32
        cfg.dadge.T = 20
        cfg.dadge.default_inference_steps = 20  # = T (strided DDPM is only valid with DDIM)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("Loading datasets ...")
    source_ds = get_dataset(args.source, args.data_root)
    target_ds = get_dataset(args.target, args.data_root)
    print(f"  Source: {source_ds}")
    print(f"  Target: {target_ds}")

    split = make_cross_domain_split(
        source_ds,
        target_ds,
        patch_size=cfg.hdpm.patch_size,
        seed=args.seed,
    )
    print(
        f"  Source patches : {len(split['source_labels']):5d}  "
        f"({len(split['source_classes'])} classes)"
    )
    print(
        f"  Target meta-tr : {len(split['target_tr_labels']):5d}  "
        f"Target meta-te : {len(split['target_te_labels']):5d}"
    )

    # Torch tensors
    src_x = torch.from_numpy(split["source_patches"])
    src_y = torch.from_numpy(split["source_labels"])
    tgt_tr_x = torch.from_numpy(split["target_tr_patches"])
    tgt_tr_y = torch.from_numpy(split["target_tr_labels"])
    tgt_te_x = torch.from_numpy(split["target_te_patches"])
    tgt_te_y = torch.from_numpy(split["target_te_labels"])

    # Random placeholder anchors (Proj(.) is not trained here). Class-name
    # embeddings are supported by scripts/demo_real.py (--class-embeddings).
    n_way = min(args.n_way, len(split["target_classes"]))
    anchors = torch.randn(len(split["target_classes"]), cfg.dsspe.anchor_glove_dim)  # one row per class

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = DGMHMamba(
        cfg,
        source_bands=source_ds.n_bands,
        target_bands=target_ds.n_bands,
        num_source_classes=len(split["source_classes"]),
    ).to(device)

    opt = AdamW(model.parameters(), lr=cfg.stage1_lr, weight_decay=cfg.weight_decay)

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 1 – Source pre-training + domain-adversarial diffusion")
    print("=" * 60)

    src_loader = DataLoader(
        TensorDataset(src_x, src_y),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    # Unlabelled target patches for L_adv (use meta-train pool)
    tgt_unlab_loader = DataLoader(
        TensorDataset(tgt_tr_x),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model.train()
    for epoch in range(1, cfg.stage1_epochs + 1):
        lambda_grl = min(1.0, epoch / max(1, cfg.dadge.grl_warmup_epochs))
        tgt_iter = iter(tgt_unlab_loader)
        running = {"cls": 0.0, "diff": 0.0, "adv": 0.0, "n": 0}

        for x_src, y_src in src_loader:
            try:
                x_tgt = next(tgt_iter)[0]
            except StopIteration:
                tgt_iter = iter(tgt_unlab_loader)
                x_tgt = next(tgt_iter)[0]

            x_src, y_src = x_src.to(device), y_src.to(device)
            x_tgt = x_tgt.to(device)

            # Band projection
            x0_src = model.project_bands(x_src, is_source=True)
            x0_tgt = model.project_bands(x_tgt, is_source=False)

            opt.zero_grad()
            l_cls, l_diff, l_adv = model.stage1_losses(x0_src, y_src, x0_tgt, lambda_grl)
            loss = total_objective_stage1(l_cls, l_diff, l_adv, cfg.loss)
            loss.backward()
            opt.step()

            running["cls"] += l_cls.item()
            running["diff"] += l_diff.item()
            running["adv"] += l_adv.item()
            running["n"] += 1

        n = max(1, running["n"])
        print(
            f"  Epoch {epoch:03d}/{cfg.stage1_epochs}  λ_grl={lambda_grl:.2f}  "
            f"L_cls={running['cls']/n:.3f}  L_diff={running['diff']/n:.3f}  "
            f"L_adv={running['adv']/n:.3f}"
        )

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 2 – Episodic meta-adaptation")
    print("=" * 60)

    for g in opt.param_groups:
        g["lr"] = cfg.stage2_lr

    index = ByClassIndex(tgt_tr_y.tolist())
    rng = random.Random(args.seed)

    model.train()
    for step in range(1, cfg.stage2_steps + 1):
        try:
            s_idx, q_idx, ep_classes = index.sample_episode(
                n_way, args.k_shot, args.q_query, rng
            )
        except ValueError:
            # not enough samples for this n-way; skip
            continue

        support_x, support_y, query_x, query_y = gather_episode(
            tgt_tr_x, tgt_tr_y, s_idx, q_idx, ep_classes
        )
        # Use a subset of anchors matching the episode classes
        ep_anchors = anchors[torch.as_tensor(ep_classes, dtype=torch.long)]

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
            print(
                f"  Step {step:04d}/{cfg.stage2_steps}  "
                f"L_epi={l_epi.item():.3f}  L_gcl={l_gcl.item():.3f}  acc={acc:.1%}"
            )

    # ------------------------------------------------------------------
    # Evaluation on held-out target meta-test pool
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"EVALUATION – {args.eval_episodes} episodes on target meta-test pool")
    print("=" * 60)

    te_index = ByClassIndex(tgt_te_y.tolist())
    accs = []
    for _ in range(args.eval_episodes):
        try:
            s_idx, q_idx, ep_classes = te_index.sample_episode(
                n_way, args.k_shot, args.q_query, rng
            )
        except ValueError:
            continue
        support_x, support_y, query_x, query_y = gather_episode(
            tgt_te_x, tgt_te_y, s_idx, q_idx, ep_classes
        )
        support_x0 = model.project_bands(support_x.to(device), is_source=False)
        query_x0 = model.project_bands(query_x.to(device), is_source=False)
        acc = evaluate_episode(
            model, support_x0, support_y, query_x0, query_y, n_way,
            anchors[torch.as_tensor(ep_classes, dtype=torch.long)], device
        )
        accs.append(acc)

    if accs:
        mean = float(np.mean(accs))
        std = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
        print(f"  OA = {mean*100:.2f} ± {std*100:.2f} %  ({len(accs)} episodes)")
    else:
        print("  Not enough samples in the meta-test pool for the requested n-way/k-shot.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {n_params/1e6:.2f} M")
    print("Done.")


if __name__ == "__main__":
    main()
