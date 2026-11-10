# Round 11 — semantic anchors and support-free check (fixed before any run)

Class-name vectors: `class_embeddings/glove.6B.300d.json` (GloVe 6B, 300-d; built by
`.github/workflows/class-embeddings.yml`). Nothing is selected or tuned; every result is reported.

## A. Anchor test (few-shot), paired with B8

Reference: B8 model seed 42 on partitions split seeds 5–9 (Round 9, `long/r9-b8-g{5..9}-m42`),
whose anchors are random placeholder vectors. Variants = B8 with one change, model seed 42, split seeds 5–9:

| Branch | Change | Manuscript row |
|---|---|---|
| `long/r11-glove-g{S}` | `--class-embeddings class_embeddings/glove.6B.300d.json` (anchors from GloVe class names; Proj trained in Stage 1 by L_sem, λ_sem = 1) | embedding-based initialisation (default) |
| `long/r11-feature-g{S}` | `--anchor-init feature` (s_k = P_k^(0)) | feature-based initialisation |
| `long/r11-none-g{S}` | `--anchor-init none` (α_s = 0, (α_m, α_d) re-normalised) | w/o anchors |

Analysis as in `ABLATION_PROTOCOL.md`: paired differences variant − B8 over the 5 partitions
(mean, SD, wins, two-sided paired t-test) for 1-shot and 5-shot OA/AA and map OA/AA/κ.

## B. Support-free check (exploratory, single runs)

`--preset verified --stage2-steps 0 --class-embeddings class_embeddings/glove.6B.300d.json --support-free`,
split seed 5, model seed 42: Stage 1 only, so no target label is used for training; the test region of
Pavia University is classified by argmax_k cos(f, s_k).

| Branch | Source → target |
|---|---|
| `exp/r11-sf-ip2pu` | Indian Pines → Pavia University (no class name shared) |
| `exp/r11-sf-pc2pu` | Pavia Centre → Pavia University (same sensor; 7 of 9 class names shared) |

One run each, so these show feasibility only; they are not a replication of the manuscript's
cross-scene table (whose band-projection protocol differs, see the README).
