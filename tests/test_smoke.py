"""Fast CPU tests: run with `pytest -q` from the repository root."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from dgm_hmamba import DGMHMamba, get_default_config  # noqa: E402
from dgm_hmamba.data import get_dataset, make_cross_domain_split, split_validation  # noqa: E402
from dgm_hmamba.evaluation import evaluate_episodes  # noqa: E402
from dgm_hmamba.losses import total_objective_stage1, total_objective_stage2  # noqa: E402
from dgm_hmamba.metrics import classification_report  # noqa: E402
from make_synthetic_data import make  # noqa: E402


def tiny_cfg(patch=5):
    cfg = get_default_config()
    cfg.dadge.T = 4
    cfg.dadge.default_inference_steps = 4
    cfg.dadge.denoiser_base_channels = 16
    cfg.dadge.denoiser_num_res_blocks = 1
    cfg.hdpm.widths = (16, 16, 16)
    cfg.hdpm.blocks_per_level = 1
    cfg.hdpm.state_dim = 4
    cfg.hdpm.expand_factor = 1
    cfg.hdpm.patch_size = patch
    cfg.dsspe.T_p = 2
    return cfg


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return DGMHMamba(tiny_cfg(), source_bands=12, target_bands=8, num_source_classes=4)


def test_stage1_backward(model):
    model.train()
    xs = model.project_bands(torch.randn(3, 12, 5, 5), is_source=True)
    xt = model.project_bands(torch.randn(3, 8, 5, 5), is_source=False)
    l_cls, l_diff, l_adv = model.stage1_losses(xs, torch.tensor([0, 1, 2]), xt, lambda_grl=0.5)
    loss = total_objective_stage1(l_cls, l_diff, l_adv, model.cfg.loss)
    model.zero_grad()
    loss.backward()
    assert torch.isfinite(loss)
    assert model.hdpm.patch_embed.weight.grad is not None
    assert model.dadge.denoiser.out_conv.weight.grad is not None
    assert model.dadge.discriminator.fc2.weight.grad is not None


def test_diffusion_loss_does_not_reach_discriminator(model):
    """c is detached before L_diff, so L_diff sends no gradient through the GRL."""
    model.train()
    xs = model.project_bands(torch.randn(3, 12, 5, 5), is_source=True)
    xt = model.project_bands(torch.randn(3, 8, 5, 5), is_source=False)
    _, l_diff, _ = model.stage1_losses(xs, torch.tensor([0, 1, 2]), xt, lambda_grl=1.0)
    model.zero_grad()
    l_diff.backward()
    g = model.dadge.discriminator.fc1.weight.grad
    assert g is None or torch.count_nonzero(g) == 0
    assert model.dadge.denoiser.out_conv.weight.grad is not None


def test_stage2_episode_and_prediction(model):
    n_way, k, q = 3, 2, 2
    sx = model.project_bands(torch.randn(n_way * k, 8, 5, 5), is_source=False)
    qx = model.project_bands(torch.randn(n_way * q, 8, 5, 5), is_source=False)
    sy = torch.arange(n_way).repeat_interleave(k)
    qy = torch.arange(n_way).repeat_interleave(q)
    anchors = torch.randn(n_way, model.cfg.dsspe.anchor_glove_dim)
    l_epi, l_gcl, logits = model.stage2_episode(sx, sy, qx, qy, n_way, anchors)
    loss = total_objective_stage2(l_epi, l_gcl, model.cfg.loss)
    model.zero_grad()
    loss.backward()
    assert logits.shape == (n_way * q, n_way)
    assert model.dsspe.fusion_logits.grad is not None
    assert model.dsspe.anchor_proj.weight.grad is None  # Proj is frozen during episodes
    pred = model.predict_episode(sx.detach(), sy, qx.detach(), n_way, anchors)
    assert pred.shape == (n_way * q,) and int(pred.max()) < n_way
    zs = model.predict_zero_shot(qx.detach(), anchors)
    assert zs.shape == (n_way * q,)


def test_ddim_encode(model):
    x0 = model.project_bands(torch.randn(2, 8, 5, 5), is_source=False)
    f = model.encode(x0, num_inference_steps=2, ddim=True)
    assert f.shape == (2, model.cfg.hdpm.widths[-1])


def test_metrics():
    rep = classification_report([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0], 3)
    assert rep["oa"] == pytest.approx(4 / 6)
    assert rep["aa"] == pytest.approx((0.5 + 1 + 0.5) / 3)
    cm = np.array(rep["confusion_matrix"])
    pe = (cm.sum(0) * cm.sum(1)).sum() / cm.sum() ** 2
    assert rep["kappa"] == pytest.approx((4 / 6 - pe) / (1 - pe))


def test_data_pipeline_and_evaluation(tmp_path):
    make(tmp_path)
    src = get_dataset("indian_pines", tmp_path)
    tgt = get_dataset("pavia_u", tmp_path)
    assert src.n_bands == 200 and tgt.n_bands == 103
    split = make_cross_domain_split(src, tgt, patch_size=5, seed=0)
    tr = {tuple(c) for c in split["target_tr_coords"]}
    te = {tuple(c) for c in split["target_te_coords"]}
    assert tr and te and not (tr & te)
    keep, val = split_validation(split["target_tr_labels"], 0.2, seed=1)
    assert len(set(keep) & set(val)) == 0

    torch.manual_seed(0)
    m = DGMHMamba(tiny_cfg(), src.n_bands, tgt.n_bands, len(split["source_classes"]))
    te_x = torch.from_numpy(split["target_te_patches"])
    te_y = torch.from_numpy(split["target_te_labels"])
    anchors = torch.randn(9, m.cfg.dsspe.anchor_glove_dim)
    out = evaluate_episodes(m, te_x, te_y, 3, 1, 3, anchors, torch.device("cpu"),
                            n_classes=9, n_episodes=2, seed=0)
    assert out["n_episodes"] == 2
    assert 0.0 <= out["oa_mean"] <= 1.0 and 0.0 <= out["aa"] <= 1.0
    assert m.training  # evaluate_episodes restores the training mode


def test_modulation_and_film_start_as_identity():
    from dgm_hmamba.modules.hdpm import ModulationHead
    head = ModulationHead(c_in=12, d_level=8)
    gamma, beta = head(torch.zeros(2, 12))
    assert torch.allclose(gamma, torch.ones_like(gamma))
    assert torch.allclose(beta, torch.zeros_like(beta))
    m = DGMHMamba(tiny_cfg(), source_bands=12, target_bands=8, num_source_classes=4)
    film = m.dadge.denoiser.film
    assert torch.count_nonzero(film.weight) == 0
    assert torch.allclose(film.bias[: film.out_features // 2], torch.ones(film.out_features // 2))


def test_diffusion_loss_does_not_train_band_projection():
    torch.manual_seed(0)
    m = DGMHMamba(tiny_cfg(), source_bands=12, target_bands=8, num_source_classes=4)
    xs = m.project_bands(torch.randn(3, 12, 5, 5), is_source=True)
    xt = m.project_bands(torch.randn(3, 8, 5, 5), is_source=False)
    _, l_diff, _ = m.stage1_losses(xs, torch.tensor([0, 1, 2]), xt, lambda_grl=1.0)
    m.zero_grad()
    l_diff.backward()
    assert m.source_band_proj.proj.weight.grad is None
    assert m.dadge.denoiser.out_conv.weight.grad is not None


def test_episodic_loss_uses_logit_scale():
    from dgm_hmamba.losses import episodic_loss
    cos = torch.tensor([[0.9, 0.1, 0.0]])
    y = torch.tensor([0])
    s = get_default_config().dsspe.logit_scale
    assert episodic_loss(cos, y, temperature=1.0 / s) < episodic_loss(cos, y)


def test_selective_scan_matches_reference_loop():
    from dgm_hmamba.modules.mamba_block import MambaBlock
    torch.manual_seed(0)
    blk = MambaBlock(8, d_state=4, expand=1)
    u, dt = torch.randn(2, 7, 8), torch.rand(2, 7, 8) * 0.1
    A = -torch.exp(blk.A_log.float())
    B, C = torch.randn(2, 7, 4), torch.randn(2, 7, 4)
    y = blk._selective_scan(u, dt, A, B, C)
    x = torch.zeros(2, 8, 4)
    ref = []
    for t in range(7):
        x = torch.exp(dt[:, t].unsqueeze(-1) * A) * x + dt[:, t].unsqueeze(-1) * B[:, t].unsqueeze(1) * u[:, t].unsqueeze(-1)
        ref.append((x * C[:, t].unsqueeze(1)).sum(-1))
    ref = torch.stack(ref, 1) + u * blk.D
    assert torch.allclose(y, ref, atol=1e-5)


def test_dsspe_keeps_well_separated_classes_apart():
    from dgm_hmamba.modules.dsspe import DSSPE
    torch.manual_seed(0)
    cfg = get_default_config().dsspe
    dsspe = DSSPE(cfg, d_model=16)
    centres = torch.eye(16)[:4] * 3.0
    support = centres.repeat_interleave(2, 0) + 0.05 * torch.randn(8, 16)
    labels = torch.arange(4).repeat_interleave(2)
    protos = dsspe.evolve(support, labels, 4, torch.randn(4, cfg.anchor_glove_dim))
    pred = dsspe.classify(centres + 0.05 * torch.randn(4, 16), protos).argmax(-1)
    assert pred.tolist() == [0, 1, 2, 3]


def test_denoiser_noise_path_starts_at_zero():
    m = DGMHMamba(tiny_cfg(), source_bands=12, target_bands=8, num_source_classes=4)
    den = m.dadge.denoiser
    x, t, c = torch.randn(2, 200, 5, 5), torch.tensor([1, 3]), torch.randn(2, 128)
    a = den(x, t, c)
    b = den(x, t, c, torch.tensor([0.1, 0.2]))
    assert torch.allclose(a, b)
