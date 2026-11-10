# Round 10 — component ablation on the confirmation partitions (fixed before any run)

Purpose: measure, with the released code and the spatially independent protocol,
what each named component of DGM-HMamba contributes. Nothing is selected or
tuned here; every result is reported.

Reference: B8, the verified configuration of Round 9 (`CONFIRMATION_PROTOCOL.md`),
model seed 42, partitions split seeds 5–9 (results already recorded on the
`long/r9-b8-g{5..9}-m42` branches). B8 is the manuscript's "w/o Diff" variant:
the diffusion enhancement is bypassed (`--no-diffusion`), HDPM receives a zero
noise map.

Variants (B8 arguments, one change each; model seed 42; split seeds 5–9):

| Branch | Change | Manuscript row |
|---|---|---|
| `long/r10-diff-g{S}` | diffusion enhancement on (remove `--no-diffusion`; default ancestral sampling, full 10-step chain) | Full DGM-HMamba |
| `long/r10-noadv-g{S}` | `--lambda-adv 0` (no domain-adversarial alignment) | – |
| `long/r10-static-g{S}` | `--static-prototypes` (class-mean prototypes, no L_gcl) | w/o Dyn |
| `long/r10-cnn-g{S}` | `--encoder cnn` (HDPM replaced by a residual 3-D CNN) | w/ CNN |

Analysis: for each variant and each metric (5-way 1-shot and 5-shot OA/AA,
9-class map OA/AA/κ, support-separated map), the paired difference variant − B8
over the 5 partitions: mean, SD, wins, two-sided paired t-test. A component is
called helpful when removing it lowers 5-shot OA and map OA on average with
p < 0.05; harmful when removing it raises them with p < 0.05; otherwise "no
detectable effect at n = 5".
