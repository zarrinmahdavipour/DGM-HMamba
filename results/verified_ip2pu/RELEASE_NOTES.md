Saved models of the ten verified runs reported in the README ("Verified results"): Indian Pines → Pavia University, demo-size model (0.50 M parameters), spatially independent block split, partitions with split seeds 5–9 × model seeds 42 and 7.

`verified-ip2pu-v1.zip` holds one folder per run (`s<split seed>_m<model seed>/`) with
- `model_final.pt` – weights (the checkpoint chosen on the validation region), the anchors and the run's settings;
- `split.json` – the exact block assignment, class counts per region and fingerprints;
- `demo_summary.json` – the run's own test results;

and `per_run.csv`, the results of all ten runs.

Evaluate one of them (no training; settings and split are read from the file):

```bash
python scripts/demo_real.py --data-root data --download --weights verified/s5_m42/model_final.pt
```

Before publishing, the workflow re-evaluated `s5_m42` from its saved weights on a fresh runner and checked that it gives the same split fingerprint and the same test results as the original run.

These are demo-scale results on one transfer task; they are not the results reported in the manuscript.
