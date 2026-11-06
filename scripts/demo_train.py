#!/usr/bin/env python3
"""
Minimal but complete training demo for DGM-HMamba.

Shows:
  - Stage 1 (source pre-training + domain-adversarial diffusion)
  - Stage 2 (episodic meta-adaptation on target)
  - Correct gradient handling of L_diff (stop-grad on c)

Uses only synthetic data so anyone can run it without downloading HSI scenes.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.optim import AdamW
from tqdm import trange

from dgm_hmamba import get_default_config, DGMHMamba
from dgm_hmamba.console import use_utf8_stdout
from dgm_hmamba.losses import total_objective_stage1, total_objective_stage2


def make_synthetic_batch(B, bands, patch, n_classes, device):
    x = torch.randn(B, bands, patch, patch, device=device)
    y = torch.randint(0, n_classes, (B,), device=device)
    return x, y


def main():
    use_utf8_stdout()
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ------------------------------------------------------------------
    # Tiny configuration for a fast, memory-friendly demo
    # ------------------------------------------------------------------
    cfg = get_default_config()
    cfg.dadge.T = 6
    cfg.dadge.default_inference_steps = 6   # = T (strided DDPM is only valid with DDIM)
    cfg.dadge.denoiser_base_channels = 16
    cfg.dadge.denoiser_num_res_blocks = 1
    cfg.hdpm.widths = (24, 24, 24)
    cfg.hdpm.blocks_per_level = 1
    cfg.hdpm.state_dim = 4
    cfg.hdpm.expand_factor = 1
    cfg.hdpm.patch_size = 5
    cfg.dsspe.T_p = 2
    cfg.stage1_epochs = 3          # demo only
    cfg.stage2_steps = 4           # demo only

    source_bands, target_bands = 40, 20
    num_source_classes = 6
    n_way, k_shot, q_query = 3, 2, 3
    patch = cfg.hdpm.patch_size

    model = DGMHMamba(cfg, source_bands, target_bands, num_source_classes).to(device)
    opt = AdamW(model.parameters(), lr=cfg.stage1_lr, weight_decay=cfg.weight_decay)

    print("=" * 60)
    print("STAGE 1  –  Source pre-training + domain-adversarial diffusion")
    print("=" * 60)

    model.train()
    for epoch in range(1, cfg.stage1_epochs + 1):
        # lambda_grl ramps 0 → 1 over the first 30 epochs (paper)
        lambda_grl = min(1.0, epoch / max(1, cfg.dadge.grl_warmup_epochs))

        x_src, y_src = make_synthetic_batch(4, source_bands, patch, num_source_classes, device)
        x_tgt, _     = make_synthetic_batch(4, target_bands, patch, num_source_classes, device)

        x0_src = model.project_bands(x_src, is_source=True)
        x0_tgt = model.project_bands(x_tgt, is_source=False)

        opt.zero_grad()
        l_cls, l_diff, l_adv = model.stage1_losses(x0_src, y_src, x0_tgt, lambda_grl)
        loss = total_objective_stage1(l_cls, l_diff, l_adv, cfg.loss)
        loss.backward()
        opt.step()

        print(f"  Epoch {epoch:02d} | λ_grl={lambda_grl:.2f} | "
              f"L_cls={l_cls.item():.3f}  L_diff={l_diff.item():.3f}  "
              f"L_adv={l_adv.item():.3f}  total={loss.item():.3f}")

    print("\n" + "=" * 60)
    print("STAGE 2  –  Episodic meta-adaptation (target domain)")
    print("=" * 60)

    # Switch to a smaller learning rate as in the paper
    for g in opt.param_groups:
        g["lr"] = cfg.stage2_lr

    model.train()
    for step in range(1, cfg.stage2_steps + 1):
        # Synthetic N-way K-shot episode
        support = torch.randn(n_way * k_shot, target_bands, patch, patch, device=device)
        query   = torch.randn(n_way * q_query, target_bands, patch, patch, device=device)
        support_y = torch.arange(n_way, device=device).repeat_interleave(k_shot)
        query_y   = torch.arange(n_way, device=device).repeat_interleave(q_query)
        anchors   = torch.randn(n_way, cfg.dsspe.anchor_glove_dim, device=device)

        support_x0 = model.project_bands(support, is_source=False)
        query_x0   = model.project_bands(query, is_source=False)

        opt.zero_grad()
        l_epi, l_gcl, logits = model.stage2_episode(
            support_x0, support_y, query_x0, query_y, n_way, anchors
        )
        loss = total_objective_stage2(l_epi, l_gcl, cfg.loss)
        loss.backward()
        opt.step()

        acc = (logits.argmax(-1) == query_y).float().mean().item()
        print(f"  Step {step:02d} | L_epi={l_epi.item():.3f}  L_gcl={l_gcl.item():.3f}  "
              f"total={loss.item():.3f}  acc={acc:.1%}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {n_params/1e6:.3f} M")
    print("Demo training finished successfully.")


if __name__ == "__main__":
    main()
