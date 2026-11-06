#!/usr/bin/env python3
"""Quick end-to-end smoke test with synthetic data.

Run from the repository root:
    python scripts/demo_forward.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from dgm_hmamba import get_default_config, DGMHMamba
from dgm_hmamba.console import use_utf8_stdout
from dgm_hmamba.losses import total_objective_stage1, total_objective_stage2


def main():
    use_utf8_stdout()
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = get_default_config()
    # Tiny settings so pure-Python sequential scan fits comfortably
    cfg.dadge.T = 4
    cfg.dadge.default_inference_steps = 4   # = T (strided DDPM is only valid with DDIM)
    cfg.dadge.denoiser_base_channels = 16
    cfg.dadge.denoiser_num_res_blocks = 1
    cfg.hdpm.widths = (32, 32, 32)
    cfg.hdpm.blocks_per_level = 1
    cfg.hdpm.state_dim = 4
    cfg.hdpm.expand_factor = 1
    cfg.hdpm.patch_size = 5
    cfg.dsspe.T_p = 1

    source_bands, target_bands = 32, 16
    num_source_classes = 5
    n_way, k_shot, q_query = 3, 1, 2
    patch = cfg.hdpm.patch_size

    model = DGMHMamba(cfg, source_bands, target_bands, num_source_classes).to(device)
    model.eval()

    # ----- Stage-1 style batch -----
    B = 2
    x_src = torch.randn(B, source_bands, patch, patch, device=device)
    y_src = torch.randint(0, num_source_classes, (B,), device=device)
    x_tgt = torch.randn(B, target_bands, patch, patch, device=device)

    x0_src = model.project_bands(x_src, is_source=True)
    x0_tgt = model.project_bands(x_tgt, is_source=False)

    with torch.no_grad():
        l_cls, l_diff, l_adv = model.stage1_losses(x0_src, y_src, x0_tgt, lambda_grl=1.0)
        loss1 = total_objective_stage1(l_cls, l_diff, l_adv, cfg.loss)
    print(f"[Stage 1] L_cls={l_cls.item():.4f}  L_diff={l_diff.item():.4f}  "
          f"L_adv={l_adv.item():.4f}  total={loss1.item():.4f}")

    # ----- Stage-2 episode -----
    support = torch.randn(n_way * k_shot, target_bands, patch, patch, device=device)
    query   = torch.randn(n_way * q_query, target_bands, patch, patch, device=device)
    support_y = torch.arange(n_way, device=device).repeat_interleave(k_shot)
    query_y   = torch.arange(n_way, device=device).repeat_interleave(q_query)
    anchors   = torch.randn(n_way, cfg.dsspe.anchor_glove_dim, device=device)

    support_x0 = model.project_bands(support, is_source=False)
    query_x0   = model.project_bands(query, is_source=False)

    with torch.no_grad():
        l_epi, l_gcl, logits = model.stage2_episode(
            support_x0, support_y, query_x0, query_y, n_way, anchors
        )
        loss2 = total_objective_stage2(l_epi, l_gcl, cfg.loss)
        preds = logits.argmax(-1)
        acc = (preds == query_y).float().mean()
    print(f"[Stage 2] L_epi={l_epi.item():.4f}  L_gcl={l_gcl.item():.4f}  "
          f"total={loss2.item():.4f}  acc={acc.item():.2%}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params/1e6:.3f} M")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
