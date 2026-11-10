# Locked confirmation round (Round 9) — fixed before any of its results exist

Written 2026-11-09, before launching the runs below.

**Final configuration (B8)**, chosen on validation evidence only (Round 8: highest
mean best-validation accuracy, 93.5 %): the r4-d configuration with 2,000 Stage-2
episodes and 800 Stage-1 iterations; no diffusion; everything else unchanged.

**Baseline (B7):** the same with 300 Stage-1 iterations.

**Partitions:** geographic split seeds 5, 6, 7, 8, 9 (`pavia_u_block64_p7_seed{5..9}.json`),
generated for this round and never used for any earlier training, selection or
evaluation. 64 px blocks, train / validation / test = 0.35 / 0.15 / 0.50, 6 px
buffers, minimum 50 / 30 / 100 pixels per class, minimum centre distance 7 px
between all partitions (verified), scaling fitted on training pixels only.

**Runs:** B8 with model seeds 42 and 7 on each partition (10 runs); B7 with model
seed 42 on each partition (5 runs). Checkpoints chosen on the validation region only.

**Evaluation (unchanged):** 50 test episodes for 5-way 1-shot and 5-way 5-shot,
9-class map with 10 support draws of 5 pixels per class, evaluation seed 0; map
also reported without queries whose patch overlaps a support patch.

**Reporting rule:** every result of these runs is reported, whatever it is. For each
task: mean ± SD over the 10 B8 runs and a 95 % t-interval. A target counts as met
only if the mean OA and the mean AA of that task both exceed 90 %. Per-class recall,
precision and F1 are reported for the map. Nothing in this protocol is changed
after results appear.
