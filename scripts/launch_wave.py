#!/usr/bin/env python3
"""Create one experiment branch per (arm, task, partition) for the label-budget study.

Dry run by default: prints every branch and its arguments. With --push it commits
exp_args.txt (and exp_stop_after.txt for two-part runs) on a new branch made from
--base and pushes it, which starts the matching workflow (exp/** or long/**).
Branches that already exist on origin are skipped, so a wave can be pushed in
batches with --max and simply re-run until nothing is left.

    python scripts/launch_wave.py --wave w1 --arms m1,m2,c2,c6,c7,c8,c11 --tasks t1,t2 --partitions 10-19
    python scripts/launch_wave.py --wave w1 --arms m1,m2,c2,c6,c7,c8,c11 --tasks t1,t2 --partitions 10-19 --push --max 40

Branch names: <exp|long>/lb-<wave>-<arm>-<task>-g<partition>, e.g. long/lb-w1-m1-t1-g10.
"""
import argparse
import subprocess
import sys
from pathlib import Path

# Round 10 "static" configuration (the frozen pipeline), without seeds, split seed
# and checkpoint selection, which each arm sets itself.
FROZEN = ("--val-fraction 0.1 --val-episodes 40 --val-every 100 --schedule cosine --warmup-frac 0.05 "
          "--stage1-iters 800 --stage2-lr 2e-3 --train-way 9 --augment --train-query 5 --expand 2 --ema 0.99 "
          "--no-diffusion --map-repeats 10 --stage2-steps 2000 --split block --block-size 64 --val-blocks 0.15 "
          "--min-counts 50,30,100 --norm train --eval-seed 0 --static-prototypes")
SEL = "--select-on-val"                      # checkpoint chosen on the labelled validation region
CNN = "--encoder cnn"
LEAKY = "--split random --norm scene"        # the manuscript's protocol (later flags win in argparse)

TASKS = {                                     # task -> (source, target)
    "t1": ("indian_pines", "pavia_u"),
    "t2": ("pavia_c", "pavia_u"),
    "t3": ("indian_pines", "pavia_c"),
    "t4": ("indian_pines", "salinas"),
    "t5": ("indian_pines", "houston2013"),    # data must be placed by hand; not downloadable
}

# arm -> (workflow, extra arguments, model seed). "long" = two-part workflow (Mamba Stage 2).
ARMS = {
    "m1": ("long", SEL, 42),                                    # R5@pool
    "m2": ("exp", "--stage2-steps 0", 42),                      # R2@0
    "m3": ("long", SEL + " --stage1-iters 0", 42),              # R3@pool
    "m4": ("long", SEL + " --lambda-adv 0", 42),                # R4@pool
    "m5": ("long", SEL + " " + LEAKY, 42),                      # R5@pool, leaky
    "m6": ("long", SEL, 7),                                     # R5@pool, second seed
    "c1": ("exp", CNN + " --stage1-iters 0 --stage2-steps 0", 42),            # R1
    "c2": ("exp", CNN + " --stage2-steps 0", 42),                             # R2@0
    "c3": ("exp", CNN + " --stage2-steps 0 --lambda-adv 0", 42),              # R2'@0
    "c4": ("exp", CNN + " " + SEL + " --stage1-iters 0", 42),                 # R3@pool
    "c5": ("exp", CNN + " " + SEL + " --lambda-adv 0", 42),                   # R4@pool
    "c6": ("exp", CNN + " " + SEL, 42),                                       # R5@pool
    "c7": ("exp", CNN + " --target-train-per-class 10", 42),                  # R5@10
    "c8": ("exp", CNN + " --target-train-per-class 50", 42),                  # R5@50
    "c9": ("exp", CNN + " --stage1-iters 0 --target-train-per-class 10", 42), # R3@10
    "c10": ("exp", CNN + " --stage1-iters 0 --target-train-per-class 50", 42),# R3@50
    "c11": ("exp", CNN, 42),                                                  # R5@pool, no selection
    "c12": ("exp", CNN + " --stage1-iters 0", 42),                            # R3@pool, no selection
    "c13": ("exp", CNN + " --target-train-per-class 50 " + LEAKY, 42),        # R5@50, leaky
    "c14": ("exp", CNN + " " + SEL, 7),                                       # R5@pool, second seed
    "c15": ("exp", CNN + " --stage1-iters 0 --stage2-steps 0 " + LEAKY, 42),  # R1, leaky
    "c16": ("exp", CNN + " --stage2-steps 0 " + LEAKY, 42),                   # R2@0, leaky
    "c17": ("exp", CNN + " --stage1-iters 0 --target-train-per-class 50 " + LEAKY, 42),  # R3@50, leaky
}
# CNN tuning grid (development partitions 5-9 only, chosen on validation OA): Stage-1 x Stage-2 learning rate.
for s1 in ("1e-3", "2e-3"):
    for s2 in ("1e-3", "2e-3", "4e-3"):
        ARMS[f"u{s1[0]}{s2[0]}"] = ("exp", f"{CNN} {SEL} --stage1-lr {s1} --stage2-lr {s2}", 42)
STOP_AFTER = "1000"                           # long runs: part 1 stops after episode 1000


def git(*a, check=True):
    r = subprocess.run(["git", *a], capture_output=True, text=True)
    if check and r.returncode:
        sys.exit(f"git {' '.join(a)} failed:\n{r.stderr}")
    return r.stdout


def partitions(spec):
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def plan(wave, arms, tasks, parts):
    rows = []
    for arm in arms:
        wf, extra, seed = ARMS[arm]
        for task in tasks:
            src, tgt = TASKS[task]
            dl = " --download" if {src, tgt} - {"indian_pines", "pavia_u"} else ""
            for g in parts:
                args = f"{FROZEN} {extra} --source {src} --target {tgt}{dl} --split-seed {g} --seed {seed}"
                rows.append((wf != "long", g, arms.index(arm), f"{wf}/lb-{wave}-{arm}-{task}-g{g}", args,
                             STOP_AFTER if wf == "long" else None))
    rows.sort()                               # long runs first (critical path), then partition-major
    return [r[3:] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", required=True)
    ap.add_argument("--arms", required=True, help="comma list, e.g. m1,m2,c2")
    ap.add_argument("--tasks", required=True, help="comma list, e.g. t1,t2")
    ap.add_argument("--partitions", required=True, help="e.g. 10-19 or 10,11")
    ap.add_argument("--base", default="main", help="commit the branches start from")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--max", type=int, default=0, help="push at most this many new branches (0 = all)")
    a = ap.parse_args()
    arms, tasks = a.arms.split(","), a.tasks.split(",")
    for x in arms:
        if x not in ARMS:
            sys.exit(f"unknown arm {x}")
    for t in tasks:
        if t not in TASKS:
            sys.exit(f"unknown task {t}")
    rows = plan(a.wave, arms, tasks, partitions(a.partitions))
    if not a.push:
        for b, args, stop in rows:
            print(f"{b}\n    {args}" + (f"\n    stop after {stop}" if stop else ""))
        print(f"{len(rows)} branches (dry run; add --push to create them)")
        return
    if git("status", "--porcelain").strip():
        sys.exit("working tree not clean: commit or stash first")
    start = git("rev-parse", "--abbrev-ref", "HEAD").strip()
    pushed = 0
    try:
        for b, args, stop in rows:
            if git("ls-remote", "--heads", "origin", b).strip():
                continue                      # already launched
            git("switch", "-q", "-C", b, a.base)
            Path("exp_args.txt").write_text(args + "\n", encoding="utf-8")
            files = ["exp_args.txt"]
            if stop:
                Path("exp_stop_after.txt").write_text(stop + "\n", encoding="utf-8")
                files.append("exp_stop_after.txt")
            git("add", *files)
            git("commit", "-q", "-m", f"Experiment {b}")
            git("push", "-q", "-u", "origin", b)
            pushed += 1
            print(f"pushed {b}")
            if a.max and pushed >= a.max:
                break
    finally:
        git("switch", "-q", start)
    left = sum(1 for b, _, _ in rows if not git("ls-remote", "--heads", "origin", b).strip())
    print(f"pushed {pushed}; still to launch: {left}")


if __name__ == "__main__":
    main()
