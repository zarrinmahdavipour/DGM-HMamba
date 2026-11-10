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


def test_classification_losses_reach_band_projections():
    torch.manual_seed(0)
    m = DGMHMamba(tiny_cfg(), source_bands=12, target_bands=8, num_source_classes=4)
    xs = m.project_bands(torch.randn(3, 12, 5, 5), is_source=True)
    xt = m.project_bands(torch.randn(3, 8, 5, 5), is_source=False)
    l_cls, _, _ = m.stage1_losses(xs, torch.tensor([0, 1, 2]), xt, lambda_grl=1.0)
    m.zero_grad()
    l_cls.backward()
    assert m.source_band_proj.proj.weight.grad is not None
    assert m.dadge.denoiser.out_conv.weight.grad is None


def test_gradient_checkpointing_gives_same_gradients():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    m1 = DGMHMamba(cfg, source_bands=12, target_bands=8, num_source_classes=4)
    cfg2 = tiny_cfg(); cfg2.hdpm.grad_checkpoint = True
    m2 = DGMHMamba(cfg2, source_bands=12, target_bands=8, num_source_classes=4)
    m2.load_state_dict(m1.state_dict())
    x = torch.randn(2, 200, 5, 5)
    eps = torch.randn(2, 200, 5, 5)
    for m in (m1, m2):
        m.zero_grad()
        m.hdpm(x, eps).sum().backward()
    g1 = m1.hdpm.levels[0].spectral_blocks[0].in_proj.weight.grad
    g2 = m2.hdpm.levels[0].spectral_blocks[0].in_proj.weight.grad
    assert torch.allclose(g1, g2, atol=1e-5)


def test_denoiser_skip_path_is_bounded_and_starts_at_zero():
    m = DGMHMamba(tiny_cfg(), source_bands=12, target_bands=8, num_source_classes=4)
    den = m.dadge.denoiser
    x, t, c = torch.randn(2, 200, 5, 5), torch.tensor([1, 3]), torch.randn(2, 128)
    assert torch.allclose(den(x, t, c), den(x, t, c, torch.tensor([0.01, 0.3])))
    torch.nn.init.normal_(den.skip.weight, std=0.01)
    small = den(x, t, c, torch.tensor([0.01, 0.01])) - den(x, t, c)
    assert torch.count_nonzero(small) == 0   # skip disabled below 0.1


def _tiled_scene(seed=0, h=120, w=90, bands=6):
    from dgm_hmamba.data.datasets import HSIDataset

    rng = np.random.RandomState(seed)
    tiles = rng.randint(0, 10, size=(-(-h // 6), -(-w // 6)))      # 0 = unlabelled
    gt = np.kron(tiles, np.ones((6, 6), dtype=np.int64))[:h, :w]
    data = (rng.rand(h, w, bands) * 900 + 100).astype(np.float32)
    return HSIDataset("pavia_u", data, gt, (0,), normalized=False)


def test_geographic_split_is_disjoint_and_normalised_on_training_pixels():
    from dgm_hmamba.data.datasets import HSIDataset
    from dgm_hmamba.data.splits import min_center_distance

    rng = np.random.RandomState(1)
    src = HSIDataset("indian_pines", rng.rand(12, 12, 4).astype(np.float32), rng.randint(1, 4, (12, 12)), (0,))
    tgt = _tiled_scene()
    P = 5
    kw = dict(patch_size=P, seed=3, split_mode="block", block_size=16, val_ratio=0.15,
              min_counts=(5, 5, 5), target_norm="train")
    sp = make_cross_domain_split(src, tgt, **kw)
    si = sp["split_info"]
    co = {k: sp[f"target_{k}_coords"] for k in ("tr", "val", "te")}
    assert all(len(v) > 0 for v in co.values())
    for a, b in (("tr", "te"), ("tr", "val"), ("val", "te")):
        assert min_center_distance(co[a], co[b], tgt.gt.shape) >= P      # no shared input pixel
    assert si["windows_disjoint"] and si["leakage"]["patches_share_pixels"] == 0.0
    assert si["leakage_train_val"]["patches_share_pixels"] == 0.0
    assert all(min(c) >= 5 for c in si["class_counts"].values())
    # scaling fitted on the training pool only: its centre pixels span exactly [0, 1]
    centre = sp["target_tr_patches"][:, :, P // 2, P // 2]
    assert np.allclose(centre.min(0), 0.0, atol=1e-6) and np.allclose(centre.max(0), 1.0, atol=1e-6)
    raw = tgt.data[co["tr"][:, 0], co["tr"][:, 1]]
    lo, hi = raw.min(0), raw.max(0)
    te_raw = tgt.data[co["te"][:, 0], co["te"][:, 1]]
    assert np.allclose(sp["target_te_patches"][:, :, P // 2, P // 2], (te_raw - lo) / (hi - lo), atol=1e-5)
    # the saved block assignment reproduces the same partitions
    grid = np.array([[int(ch) for ch in row] for row in si["block_grid"]])
    sp2 = make_cross_domain_split(src, tgt, **dict(kw, seed=99), block_grid=grid)
    assert sp2["split_info"]["index_fingerprint"] == si["index_fingerprint"]
    with pytest.raises(ValueError):          # buffer smaller than the patch would allow overlap
        make_cross_domain_split(src, tgt, **dict(kw, buffer=P - 2))


def test_scene_normalisation_unchanged():
    import dataclasses
    from dgm_hmamba.data.datasets import HSIDataset

    rng = np.random.RandomState(2)
    src = HSIDataset("indian_pines", rng.rand(12, 12, 4).astype(np.float32), rng.randint(1, 4, (12, 12)), (0,))
    raw = _tiled_scene(seed=4)
    old = raw.data.copy()
    for b in range(old.shape[-1]):          # legacy whole-scene scaling
        old[:, :, b] = (old[:, :, b] - old[:, :, b].min()) / (old[:, :, b].max() - old[:, :, b].min())
    pre = dataclasses.replace(raw, data=old, normalized=True)
    a = make_cross_domain_split(src, pre, patch_size=5, seed=0, split_mode="random")
    b = make_cross_domain_split(src, raw, patch_size=5, seed=0, split_mode="random", target_norm="scene")
    assert np.allclose(a["target_tr_patches"], b["target_tr_patches"], atol=1e-6)
    assert np.array_equal(a["target_te_labels"], b["target_te_labels"])


def test_focal_episodic_loss():
    from dgm_hmamba.losses import episodic_loss

    torch.manual_seed(0)
    logits = torch.randn(12, 5)
    y = torch.randint(0, 5, (12,))
    plain = torch.nn.functional.cross_entropy(logits / 0.1, y)
    assert torch.equal(episodic_loss(logits, y, 0.1), plain)                 # gamma 0: unchanged
    assert torch.equal(episodic_loss(logits, y, 0.1, focal_gamma=0.0), plain)
    ce = torch.nn.functional.cross_entropy(logits / 0.1, y, reduction="none")
    ref = ((1 - torch.exp(-ce)) ** 2 * ce).mean()
    assert torch.allclose(episodic_loss(logits, y, 0.1, focal_gamma=2.0), ref)
    assert episodic_loss(logits, y, 0.1, focal_gamma=2.0) <= plain           # down-weights easy queries


def test_transductive_refinement():
    from dgm_hmamba.transductive import config_name, parse_grid, refine_prototypes

    torch.manual_seed(0)
    centres = torch.nn.functional.normalize(torch.randn(3, 16), dim=-1)
    shift = 0.6 * torch.nn.functional.normalize(torch.randn(3, 16), dim=-1)   # the test region looks different
    unl = torch.cat([centres[c] + shift[c] + 0.05 * torch.randn(200, 16) for c in range(3)])
    sup = torch.cat([centres[c] + 0.05 * torch.randn(2, 16) for c in range(3)])
    sy = torch.arange(3).repeat_interleave(2)
    P0 = torch.stack([sup[sy == c].mean(0) for c in range(3)])
    assert torch.allclose(refine_prototypes(P0, sup, sy, unl, iters=0), torch.nn.functional.normalize(P0, dim=-1))
    P = refine_prototypes(P0, sup, sy, unl, iters=3, scale=10.0)
    target = torch.nn.functional.normalize(centres + shift, dim=-1)
    before = (torch.nn.functional.normalize(P0, dim=-1) * target).sum(-1)
    after = (P * target).sum(-1)
    assert (after > before).all()                       # prototypes move towards the region's class means
    assert P.shape == P0.shape and torch.allclose(P.norm(dim=-1), torch.ones(3))
    g = parse_grid("iters=1,3;scale=10;topk=0,5")
    assert len(g) == 4 and g[0] == {"iters": 1, "scale": 10.0, "topk": 0}
    assert config_name(None) == "inductive"


def test_ablation_switches():
    """w/ CNN encoder, w/o Dyn (static prototypes) and the loss-weight switches."""
    torch.manual_seed(0)
    cfg = tiny_cfg(patch=7)
    cfg.hdpm.encoder = "cnn"
    cfg.dadge.bypass_enhancement = True
    cfg.dsspe.T_p = 0
    m = DGMHMamba(cfg, source_bands=12, target_bands=8, num_source_classes=4)
    assert not any("levels" in n for n, _ in m.named_parameters())       # no HDPM inside
    x = m.project_bands(torch.randn(6, 8, 7, 7), is_source=False)
    f = m.encode(x)
    assert f.shape == (6, cfg.hdpm.widths[-1])
    sy = torch.arange(3).repeat_interleave(2)
    anchors = torch.randn(3, cfg.dsspe.anchor_glove_dim)
    P = m.dsspe.evolve(f, sy, 3, anchors)
    assert torch.allclose(P, torch.stack([f[sy == c].mean(0) for c in range(3)]))   # static class means
    l_epi, l_gcl, _ = m.stage2_episode(x, sy, x, sy, 3, anchors)
    total_objective_stage2(l_epi, l_gcl, m.cfg.loss).backward()
    assert m.hdpm.block1.c1.weight.grad is not None
    # Stage 1 with lambda_adv = 0: the adversarial loss no longer reaches the target band projection
    cfg.loss.lambda_adv = 0.0
    m.zero_grad()
    xs = m.project_bands(torch.randn(4, 12, 7, 7), is_source=True)
    xt = m.project_bands(torch.randn(4, 8, 7, 7), is_source=False)
    l_cls, l_diff, l_adv = m.stage1_losses(xs, torch.tensor([0, 1, 2, 3]), xt, 1.0)
    total_objective_stage1(l_cls, l_diff, l_adv, m.cfg.loss).backward()
    g = m.target_band_proj.proj.weight.grad
    assert g is None or torch.count_nonzero(g) == 0


def test_class_name_embeddings(tmp_path):
    from dgm_hmamba.data.class_embeddings import (build_embedding_file, load_class_embeddings,
                                                  needed_tokens, tokenize)
    from dgm_hmamba.data.datasets import CLASS_NAMES

    assert tokenize("Corn-notill") == ["corn", "no", "till"]
    assert tokenize("Vinyard untrained") == ["vineyard", "untrained"]
    assert tokenize("Lettuce romaine 4wk") == ["lettuce", "romaine", "four", "weeks"]
    rng = np.random.default_rng(0)
    vocab = {t: rng.standard_normal(8).astype(np.float32) for t in needed_tokens(CLASS_NAMES) if t != "till"}
    doc = build_embedding_file(vocab, CLASS_NAMES, "fake", tmp_path / "emb.json")
    assert doc["oov"]["indian_pines"]["Corn-notill"] == ["till"]          # out-of-vocabulary tokens are recorded
    names = ["Meadows", "Asphalt"]
    arr, info = load_class_embeddings(tmp_path / "emb.json", "pavia_u", names)
    assert arr.shape == (2, 8) and np.allclose(arr[1], vocab["asphalt"], atol=1e-5)
    assert np.allclose(load_class_embeddings(tmp_path / "emb.json", "pavia_u", ["Painted metal sheets"])[0][0],
                       np.mean([vocab["painted"], vocab["metal"], vocab["sheets"]], 0), atol=1e-5)


def test_semantic_anchor_projection_and_modes():
    torch.manual_seed(0)
    m = DGMHMamba(tiny_cfg(), source_bands=12, target_bands=8, num_source_classes=4)
    xs = m.project_bands(torch.randn(4, 12, 5, 5), is_source=True)
    xt = m.project_bands(torch.randn(4, 8, 5, 5), is_source=False)
    src_anchors = torch.randn(4, m.cfg.dsspe.anchor_glove_dim)
    l_cls, l_diff, l_adv, l_sem = m.stage1_losses_with_semantics(xs, torch.tensor([0, 1, 2, 3]), xt, 1.0, src_anchors)
    l_sem.backward()
    assert m.dsspe.anchor_proj.weight.grad is not None                    # Proj is trained in Stage 1
    f = torch.randn(6, m.cfg.hdpm.widths[-1])
    sy = torch.arange(3).repeat_interleave(2)
    a1, a2 = torch.randn(3, m.cfg.dsspe.anchor_glove_dim), torch.randn(3, m.cfg.dsspe.anchor_glove_dim)
    assert not torch.allclose(m.dsspe.evolve(f, sy, 3, a1), m.dsspe.evolve(f, sy, 3, a2))
    for mode in ("feature", "none"):                                       # anchors from features / no semantic term
        m.dsspe.cfg.anchor_init = mode
        assert torch.allclose(m.dsspe.evolve(f, sy, 3, a1), m.dsspe.evolve(f, sy, 3, a2))
    m.dsspe.cfg.anchor_init = "embedding"
