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

## Verified results (demo scale, spatially independent test regions)

Indian Pines → Pavia University with a demo-size model (0.50 M parameters) and the code in this repository. Pavia University is cut into 64 × 64-pixel blocks that are assigned to a training region (35 %), a validation region (15 %) and a test region (50 %), separated by 6-pixel buffers, so that no 7 × 7 test patch shares a pixel with a training or validation patch. Target normalisation is fitted on training pixels only, and checkpoints are chosen on the validation region only. The configuration was chosen on five development partitions and then run once on five fresh partitions (split seeds 5–9) × two model seeds (42, 7), under a protocol committed before the runs ([`splits/CONFIRMATION_PROTOCOL.md`](splits/CONFIRMATION_PROTOCOL.md)). All ten runs are reported.

| Task (test region) | OA, mean ± SD [95 % CI] | AA, mean ± SD [95 % CI] |
|---|---|---|
| 5-way 5-shot | **91.6 ± 1.5** [90.6, 92.7] | **91.8 ± 1.6** [90.6, 92.9] |
| 5-way 1-shot | 87.4 ± 1.4 [86.4, 88.4] | 87.5 ± 1.5 [86.4, 88.6] |
| 9-class map (5 labelled pixels per class from the test region, every other test pixel classified; κ 0.76) | 82.1 ± 4.6 [78.8, 85.4] | 84.7 ± 2.1 [83.2, 86.2] |

Per-class recall on the map: Asphalt 86.8, Meadows 82.4, Gravel 61.5, Trees 96.6, Painted metal 99.8, Bare soil 74.7, Bitumen 93.3, Self-blocking bricks 68.0, Shadows 99.3 %. Most errors are Gravel ↔ Self-blocking bricks and Bare soil ↔ Meadows. Per-run results: [`results/verified_ip2pu/per_run.csv`](results/verified_ip2pu/per_run.csv).

What this configuration is (`--preset verified`):

* the manuscript's **"w/o Diff" variant**: the diffusion enhancement is bypassed and HDPM receives a zero noise map (the denoiser is still trained but not used); the domain-adversarial alignment, HDPM and DSSPE are on;
* training additions that are not in the manuscript: 800 Stage-1 iterations and 2,000 Stage-2 episodes, 9-way training episodes with 5 queries per class, flips/rotations/spectral-scaling/noise augmentation, cosine learning-rate schedule with warm-up (Stage-2 learning rate 2 × 10⁻³), exponential moving average of the weights (0.99), and checkpoint selection on the validation region;
* demo-size model: widths 32/32/32, one block per level, state size 4, 7 × 7 patches;
* **semantic anchors: random placeholder vectors**, not class-name embeddings (see [Class-name embeddings](#class-name-embeddings-semantic-anchors)). The semantic term of DSSPE therefore carries no class-name information in these results.

These are demo-scale results on one transfer task. They are **not** the results reported in the manuscript, which uses the paper-size model, pixel-level meta-training/meta-testing pools and four transfer tasks.

### Reproduce

**Evaluate the saved models** (no training; a few minutes on a GPU, about 20 minutes per model on a CPU). The ten models are attached to the release [`verified-ip2pu-v1`](https://github.com/zarrinmahdavipour/DGM-HMamba/releases/tag/verified-ip2pu-v1); each file carries its settings and split seed:

```bash
curl -LO https://github.com/zarrinmahdavipour/DGM-HMamba/releases/download/verified-ip2pu-v1/verified-ip2pu-v1.zip
unzip -q verified-ip2pu-v1.zip -d verified
python scripts/demo_real.py --data-root data --download --weights verified/s5_m42/model_final.pt --output-dir runs/s5_m42
```

For `s5_m42` this prints split fingerprint `1bfe399f4c8db9ad` and 5-shot OA 91.39 / AA 91.43, 1-shot 87.25 / 87.40, map OA 83.10 / AA 84.89 (small floating-point differences between machines are possible).

**Train from scratch** (about 8 hours on a 4-core CPU; much faster on a GPU):

```bash
python scripts/demo_real.py --data-root data --download --preset verified --split-seed 5 --seed 42 --output-dir runs/verified_s5_m42
```

Any option given after `--preset` overrides it. Results differ slightly from run to run on different hardware (floating-point non-determinism); the ± values above include the variation between model seeds. The block assignments are in `splits/pavia_u_block64_p7_seed{5..9}.json`; `--split-file` reproduces a partition exactly, and every run prints the split and normalisation fingerprints. The runs above were made with `.github/workflows/experiment-long.yml` (push a branch `long/<name>` holding `exp_args.txt` and `exp_stop_after.txt`).

## Quick demo

**In the browser:** open the [Colab notebook](https://colab.research.google.com/github/zarrinmahdavipour/DGM-HMamba/blob/main/notebooks/DGM_HMamba_demo.ipynb), choose *Runtime → Change runtime type → T4 GPU*, and *Run all*. It runs the smoke tests, downloads the two scenes, evaluates a verified model on its test region and draws the classification map; training the verified configuration from scratch is an optional, longer step.

**Locally:**

```bash
git clone https://github.com/zarrinmahdavipour/DGM-HMamba.git
cd DGM-HMamba
pip install -r requirements.txt

python scripts/demo_forward.py              # synthetic smoke test (seconds)
python scripts/demo_train.py                # a few Stage-1 / Stage-2 optimisation steps
python scripts/demo_real.py --download --stage1-iters 200 --stage2-steps 100   # short real-data run
```

The short run (≈0.5 M parameters, 200 Stage-1 iterations, 100 episodes, pixel-level split) only shows that the pipeline runs; its accuracies are far below the verified results. `demo_real.py` writes `demo_maps.png` (false colour, ground truth and predicted map), `demo_curves.png`, `demo_summary.json`, `split.json` and `model_final.pt` to `--output-dir`.

Ablation switches of `demo_real.py`: `--no-diffusion` (w/o Diff), `--encoder cnn` (HDPM replaced by a residual 3-D CNN), `--static-prototypes` (class-mean prototypes, no L_gcl), `--lambda-adv 0` (no domain-adversarial alignment), `--lambda-diff`, `--lambda-gcl`.

On Windows the scripts switch the console to UTF-8 themselves, so no `PYTHONIOENCODING` setting is needed.

## Class-name embeddings (semantic anchors)

**Status.** The manuscript initialises each DSSPE anchor from the 300-d GloVe vector of the class name, s_k = Proj(e_k), trains Proj on the source classes in Stage 1, and uses the same anchors for *label-free, support-free cross-scene* classification, ŷ = argmax_k cos(f, s_k) (its cross-scene table and the support-free column of its anchor ablation). Until 10 November 2026 the code had **no class-name embeddings**: every script used random placeholder vectors and Proj was never trained. Those experiments therefore **cannot be reproduced from this repository**, and they have not been reproduced with the code added since. All verified results above use the random placeholders.

**What the code now provides.** `--class-embeddings FILE` loads one vector per class name; Stage 1 then trains Proj with an additional loss L_sem, Proj is frozen in Stage 2, and `--support-free` evaluates the support-free rule on the test region (label-free only together with `--stage2-steps 0`). `--anchor-init {embedding,feature,none}` selects the manuscript's three anchor variants (embedding-based, feature-based s_k = P_k^(0), and no semantic term with α_s = 0).

`class_embeddings/glove.6B.300d.json` is included: the 300-d [GloVe 6B](https://nlp.stanford.edu/projects/glove/) vectors (Public Domain Dedication and License) of the class names of all five scenes, built by `.github/workflows/class-embeddings.yml` (73 of 75 tokens found; "windrowed" and "senesced" are out of vocabulary, so "Hay-windrowed" and "Corn senesced green weeds" rest on their other words).

```bash
# rebuild from another GloVe file: python scripts/make_class_embeddings.py --glove glove.6B.300d.txt --out class_embeddings/glove.6B.300d.json
python scripts/demo_real.py --download --preset verified --class-embeddings class_embeddings/glove.6B.300d.json
python scripts/demo_real.py --download --preset verified --stage2-steps 0 \
    --class-embeddings class_embeddings/glove.6B.300d.json --support-free      # no target label used
```

**Choices that the manuscript does not specify** (they may differ from the authors' and should be replaced by theirs if known):

* the GloVe release (`class-embeddings.yml` builds the file from GloVe 6B, 300-d);
* how a multi-word or abbreviated class name becomes one vector: lower-cased word tokens, a few explicit replacements (`notill` → "no till", `mintill` → "minimum till", `vinyard` → "vineyard", `brocoli` → "broccoli", `4wk` → "four weeks", …; `TOKEN_ALIASES` in `dgm_hmamba/data/class_embeddings.py`), then the mean of the in-vocabulary token vectors; out-of-vocabulary tokens are recorded in the file;
* the loss that trains Proj: L_sem is the cross-entropy of the scaled cosine similarities (scale 10) between the source embeddings and the projected class-name vectors of all source classes, added to the Stage-1 objective with weight `--lambda-sem` (default 1).

---

## Installation

Python ≥ 3.9 and PyTorch ≥ 2.0. The CI workflow runs the tests and demos on Python 3.11 with the CPU build of PyTorch.

```bash
pip install -r requirements.txt     # torch, numpy, scipy, tqdm, matplotlib
pip install h5py                     # optional: MATLAB v7.3 files (e.g. some Houston 2013 copies)
pip install pytest && pytest -q      # optional: unit tests (CPU, < 1 min)
```

## Data

Place the standard `.mat` files as follows, or let `prepare_data.py` download Indian Pines, Pavia University, Pavia Centre and Salinas. It tries a [Hugging Face mirror](https://huggingface.co/danaroth) of the EHU files first, falls back to the EHU server, and checks the array shape of every file:

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
  losses.py          L_cls, L_epi (optionally focal) and the weighted stage objectives
  episodes.py        N-way K-shot episode sampling
  evaluation.py      episodic evaluation (OA / AA / κ)
  metrics.py         confusion matrix, OA, AA, κ, per-class precision / recall / F1
  augment.py         patch augmentation used in the verified configuration
  transductive.py    optional transductive prototype refinement (not used for any reported result)
  modules/           band_proj, dadge, mamba_block, hdpm, dsspe, cnn3d (ablation encoder)
  data/              .mat loaders, patch extraction, pixel-level and geographic block splits, downloader,
                     class-name embeddings (class_embeddings.py)
scripts/
  demo_forward.py    synthetic smoke test
  demo_train.py      synthetic two-stage training demo
  demo_real.py       real-data runs (Indian Pines -> Pavia University), --preset verified, --weights
  make_splits.py     writes the block assignments in splits/
  make_class_embeddings.py   builds the class-name embedding file from a GloVe text file
class_embeddings/    GloVe 6B 300-d vectors of the class names (semantic anchors)
  prepare_data.py    download / check datasets
  train.py           full two-stage training and evaluation (pixel-level pools)
  train_cross_domain.py   lighter training script
splits/              block assignments of the geographic partitions and the pre-registered protocols
results/             per-run results of the verified configuration
notebooks/DGM_HMamba_demo.ipynb   Colab demo
tests/               pytest smoke tests and a synthetic-data generator (used by CI)
```

## Design notes

* **Diffusion loss and the GRL.** The conditioning vector `c` is the discriminator's penultimate activation. For `L_diff`, `c` is detached (`model.py`, `stage1_losses`), so the diffusion-loss gradient does not pass through the gradient reversal layer; only `L_adv` trains the discriminator path, as in standard DANN practice. A unit test checks this (`tests/test_smoke.py`).
* **Use of target data.** Stage 1 uses unlabelled patches from the target meta-training pool for the domain-adversarial loss; Stage 2 uses labelled episodes from the same pool. The support-free prediction (`predict_zero_shot`) therefore describes *label-free / support-free cross-scene transfer*, not a setting in which no target pixels were seen.

## Known differences from the manuscript

This is a clean re-implementation. Saved models and per-run results are included for the verified demo configuration only, not for the manuscript's experiments. The following points differ from the manuscript text:

1. **Model size and speed.** With the manuscript configuration in `config.py` the model has about 43.5 M trainable parameters (HDPM ≈ 42.2 M), whereas the manuscript reports 8.6 M. The selective scan in `mamba_block.py` is a sequential PyTorch loop without the `mamba-ssm` CUDA kernels, so runtime and memory are far higher than the per-patch latency reported in the manuscript.
2. **Semantic anchors.** Class-name embeddings are optional (`--class-embeddings`, see [Class-name embeddings](#class-name-embeddings-semantic-anchors)); without them the anchors are random placeholder vectors, as in all verified results, and `train.py` / `train_cross_domain.py` still use placeholders only. The GloVe release, the handling of compound class names and the loss that trains Proj are not given in the manuscript, so the choices made here may differ from the authors'. The support-free cross-scene results and the anchor ablation of the manuscript have not been reproduced with this code, and the PCA band projection for unseen sensors is not implemented.
3. **Data split.** `train.py` and the default split of `demo_real.py` divide the labelled target pixels 50/50 per class into meta-training and meta-testing pools (no spatial separation); the manuscript's pool sizes and small-class rules are not reproduced. Episodes that would need more samples than a class has are skipped instead of sampled with replacement. The verified results use the geographic block split instead (`--split block`).
4. **HDPM details.** Patch merging gives 81 / 16 / 4 tokens per level (manuscript ≈ 81 / 25 / 9); the spatial path takes the spectral path's output, so the paths run in sequence; the cross-path fusion adds a residual connection.
5. **DSSPE details.** Graph diffusion uses the exact operator exp(−ηL) on the non-negative cosine graph of the initial prototypes; the graph contrastive loss uses an MLP encoder with feature-noise views and no edge dropout.
6. **Changes for stable training**, each explained in the code: LayerNorm pre-normalisation inside each Mamba block, gradient-norm clipping, identity initialisation of the modulation heads (γ = 1, β = 0), a bounded skip path in the denoiser, detached inputs to the diffusion loss, and a straight-through gradient for the band projections through the diffusion enhancement. The verified configuration also uses the training additions listed under *Verified results*.
7. **Experiments.** The ablation switches listed under *Quick demo* are included; baselines, the sensitivity, noise and t-SNE experiments, the multi-seed driver for the manuscript's tables and the significance tests are not.

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
