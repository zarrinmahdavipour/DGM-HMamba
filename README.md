# DGM-HMamba

[![CI](https://github.com/zarrinmahdavipour/DGM-HMamba/actions/workflows/ci.yml/badge.svg)](https://github.com/zarrinmahdavipour/DGM-HMamba/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zarrinmahdavipour/DGM-HMamba/blob/main/notebooks/DGM_HMamba_demo.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

PyTorch implementation of **DGM-HMamba: Diffusion-guided hierarchical Mamba with dynamic prototype evolution for cross-domain hyperspectral image classification under episodic meta-learning** (Z. Mahdavipour, A. Khader, L. Hu, L. Xiao; manuscript submitted to *Neurocomputing*).

DGM-HMamba has three modules:

| Module | Role | Code |
|---|---|---|
| **DADGE** – domain-adaptive diffusion-guided enhancement | A shared denoiser is conditioned on the penultimate representation of a domain discriminator trained through a gradient reversal layer; its noise predictions are reused downstream. | `dgm_hmamba/modules/dadge.py` |
| **HDPM** – hierarchical dual-path Mamba | Spectral and spatial selective state-space paths at three resolutions, modulated block by block by the diffusion noise map and fused by cross-path attention. | `dgm_hmamba/modules/hdpm.py`, `mamba_block.py` |
| **DSSPE** – dynamic spectral–semantic prototype evolution | Class prototypes are refined by momentum, graph-Laplacian diffusion and semantic anchors, with a graph contrastive regulariser. | `dgm_hmamba/modules/dsspe.py` |

Training has two stages: **Stage 1** pre-trains on the labelled source scene with `L_cls + λ3·L_diff + λ_adv·L_adv` (unlabelled target patches contribute only their domain label), and **Stage 2** meta-trains on N-way K-shot episodes drawn from the labelled target meta-training pool with `λ1·L_epi + λ2·L_gcl`. Evaluation episodes come from a held-out, label-disjoint meta-testing pool.

---

## Quick demo

**In the browser:** open the [Colab notebook](https://colab.research.google.com/github/zarrinmahdavipour/DGM-HMamba/blob/main/notebooks/DGM_HMamba_demo.ipynb), choose *Runtime → Change runtime type → T4 GPU*, and *Run all* (about 5–10 minutes).

**Locally:**

```bash
git clone https://github.com/zarrinmahdavipour/DGM-HMamba.git
cd DGM-HMamba
pip install -r requirements.txt

python scripts/demo_forward.py              # synthetic smoke test (seconds)
python scripts/demo_train.py                # a few Stage-1 / Stage-2 optimisation steps
python scripts/demo_real.py --download      # Indian Pines -> Pavia University, a few minutes
```

`demo_real.py` downloads the two scenes, trains a small model (≈0.3 M parameters) for 200 Stage-1 iterations and 100 Stage-2 episodes, reports 5-way 1-shot and 5-shot OA / AA / κ on the meta-testing pool, and writes `runs/demo/demo_maps.png` (false colour, ground truth and predicted map), `runs/demo/demo_curves.png` and `runs/demo/demo_summary.json`. **These demo numbers are not the manuscript's results**; they only show that the full pipeline runs.

On Windows the scripts switch the console to UTF-8 themselves, so no `PYTHONIOENCODING` setting is needed.

---

## Installation

Python ≥ 3.9 and PyTorch ≥ 2.0. The CI workflow runs the tests and demos on Python 3.11 with the CPU build of PyTorch.

```bash
pip install -r requirements.txt     # torch, numpy, scipy, tqdm, matplotlib
pip install h5py                     # optional: MATLAB v7.3 files (e.g. some Houston 2013 copies)
pip install pytest && pytest -q      # optional: unit tests (CPU, < 1 min)
```

## Data

Place the standard `.mat` files as follows, or let `prepare_data.py` download the scenes that have direct public links (Indian Pines, Pavia University, Pavia Centre):

```
data/
├── IndianPines/   Indian_pines_corrected.mat, Indian_pines_gt.mat
├── PaviaU/        PaviaU.mat, PaviaU_gt.mat
├── PaviaC/        Pavia.mat, Pavia_gt.mat
├── Salinas/       Salinas_corrected.mat, Salinas_gt.mat
└── Houston2013/   Houston13.mat, Houston13_gt.mat   (one merged train+test label map)
```

```bash
python scripts/prepare_data.py --data-root ./data --download indian_pines pavia_u
python scripts/prepare_data.py --data-root ./data            # check what is present
```

Sources: [EHU Hyperspectral Remote Sensing Scenes](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) (Indian Pines, Pavia, Salinas) and the IEEE GRSS 2013 Data Fusion Contest (Houston 2013). Each band is min–max normalised to [0, 1] on loading.

## Training

```bash
python scripts/train.py \
    --source indian_pines --target pavia_u --data-root ./data \
    --n-way 5 --k-shot 5 --q-query 15 \
    --stage1-epochs 200 --stage2-steps 100 \
    --seed 42 --output-dir runs/ip2pu_5shot_s42
```

* Defaults follow the manuscript: AdamW, learning rate 1e-3 (Stage 1) and 1e-4 (Stage 2), 200 epochs / 100 episodes, λ1 = λ2 = 0.1, λ3 = 0.5, λ_adv = 0.1, T = 50, 9×9 patches (`dgm_hmamba/config.py`). `train.py` uses a cosine learning-rate schedule in Stage 1.
* The meta-testing pool is evaluated **once**, after training. Its labels are never used for training or model selection. For checkpoint selection, add `--val-fraction 0.1`: a per-class validation subset is then carved out of the target meta-training pool and the best validation checkpoint is the one tested.
* Reported metrics: OA as mean ± sample SD (ddof = 1) over evaluation episodes, and AA and Cohen's κ from the confusion matrix pooled over all episodes (`final_metrics.json`).
* `--preset {auto,paper,reduced,tiny}` selects the model size. `paper` is the manuscript configuration; `auto` (default) uses it on GPU and `reduced` (widths 64/128/256, one block per level, T = 20) on CPU. The manuscript configuration needs a large GPU because the selective scan is a plain PyTorch loop (see below).
* Outputs in `--output-dir`: `train.log`, `args.json`, `checkpoint_last.pt`, `checkpoint_best.pt` (with `--val-fraction`), `final_metrics.json`. Resume with `--resume <dir>/checkpoint_last.pt` and the same arguments.
* The manuscript averages 10 runs with seeds 42, 123, 456, 789, 1024, 2048, 3141, 4096, 7777 and 9999; run the command once per seed.

`scripts/train_cross_domain.py` is a shorter variant without logging or checkpoints.

## Library use

```python
from dgm_hmamba import DGMHMamba, get_default_config

model = DGMHMamba(get_default_config(), source_bands=200, target_bands=103, num_source_classes=16)
x0 = model.project_bands(raw_target_patches, is_source=False)       # (B,103,9,9) -> (B,200,9,9)
f = model.encode(x0)                                                # DADGE (50-step DDPM) -> HDPM
f = model.encode(x0, num_inference_steps=10, ddim=True)             # accelerated DDIM sampling
pred = model.predict_episode(support_x0, support_y, query_x0, n_way, anchors)
pred = model.predict_zero_shot(query_x0, anchors)                   # support-free, anchors only
```

## Repository structure

```
dgm_hmamba/
  config.py          hyper-parameters (manuscript values tagged PAPER)
  model.py           DGMHMamba: band projection, encode, Stage-1/Stage-2 losses, prediction
  losses.py          L_cls, L_epi and the weighted stage objectives
  episodes.py        N-way K-shot episode sampling
  evaluation.py      episodic evaluation (OA / AA / κ)
  metrics.py         confusion matrix, OA, AA, κ
  modules/           band_proj, dadge, mamba_block, hdpm, dsspe
  data/              .mat loaders, patch extraction, meta-train / meta-test splits, downloader
scripts/
  demo_forward.py    synthetic smoke test
  demo_train.py      synthetic two-stage training demo
  demo_real.py       real-data demo (Indian Pines -> Pavia University)
  prepare_data.py    download / check datasets
  train.py           full two-stage training and evaluation
  train_cross_domain.py   lighter training script
notebooks/DGM_HMamba_demo.ipynb   Colab demo
tests/               pytest smoke tests and a synthetic-data generator (used by CI)
```

## Design notes

* **Diffusion loss and the GRL.** The conditioning vector `c` is the discriminator's penultimate activation. For `L_diff`, `c` is detached (`model.py`, `stage1_losses`), so the diffusion-loss gradient does not pass through the gradient reversal layer; only `L_adv` trains the discriminator path, as in standard DANN practice. A unit test checks this (`tests/test_smoke.py`).
* **Use of target data.** Stage 1 uses unlabelled patches from the target meta-training pool for the domain-adversarial loss; Stage 2 uses labelled episodes from the same pool. The support-free prediction (`predict_zero_shot`) therefore describes *label-free / support-free cross-scene transfer*, not a setting in which no target pixels were seen.

## Known differences from the manuscript

This is a clean re-implementation. Pretrained checkpoints and per-run logs are not included yet, and the following points currently differ from the manuscript text:

1. **Model size and speed.** With the manuscript configuration in `config.py` the model has about 43.5 M trainable parameters (HDPM ≈ 42.2 M), whereas the manuscript reports 8.6 M. The selective scan in `mamba_block.py` is a sequential PyTorch loop without the `mamba-ssm` CUDA kernels, so runtime and memory are far higher than the per-patch latency reported in the manuscript.
2. **Semantic anchors.** GloVe class-name anchors are not implemented yet; the scripts use fixed random placeholder vectors (class names are listed in `data/datasets.py`). The anchor projection is trained in Stage 2 rather than in Stage 1. The zero-shot experiment cannot be reproduced with this release, and the PCA band projection for unseen sensors is not implemented.
3. **Data split.** Labelled target pixels are split 50/50 per class into meta-training and meta-testing pools; the manuscript's pool sizes and small-class rules are not reproduced, and split index files are not included. Episodes that would need more samples than a class has are skipped instead of sampled with replacement.
4. **HDPM details.** Patch merging gives 81 / 16 / 4 tokens per level (manuscript ≈ 81 / 25 / 9); the spatial path takes the spectral path's output, so the paths run in sequence; the cross-path fusion adds a residual connection.
5. **DSSPE details.** Graph diffusion is approximated by three explicit Euler steps on a graph rebuilt from the current prototypes; the graph contrastive loss uses an MLP encoder with feature-noise views and no edge dropout.
6. **Experiments.** Baselines and the scripts for the ablation, sensitivity, noise, t-SNE and map experiments, the multi-seed driver and the significance tests are not included.

## Citation

```bibtex
@article{mahdavipour2026dgmhmamba,
  title   = {{DGM-HMamba}: Diffusion-guided hierarchical {Mamba} with dynamic prototype evolution
             for cross-domain hyperspectral image classification under episodic meta-learning},
  author  = {Mahdavipour, Zarrin and Khader, Abdolraheem and Hu, Ling and Xiao, Liang},
  journal = {Neurocomputing},
  note    = {Submitted},
  year    = {2026}
}
```

## License

Code: [MIT](LICENSE). The benchmark datasets are distributed by their original providers under their own terms.
