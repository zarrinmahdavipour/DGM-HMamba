#!/usr/bin/env python3
"""Extract label-budget runs (branches exp/lb-* and long/lb-*) from GitHub Actions job logs.

    python analysis/lb_extract.py fetch                 # save logs of every lb branch to analysis/logs/
    python analysis/lb_extract.py blind lb_blind.csv    # integrity + validation fields only
    python analysis/lb_extract.py full  lb_full.csv     # everything, including test-region numbers

Reuses logx.py (log download and RESULT parsing) and adds the study's own columns.
"""
import ast
import csv
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import logx  # noqa: E402

BRANCH = re.compile(r"^(exp|long)/lb-([a-z]+\d+)-([a-z]\d+)-(t\d)-g(\d+)$")
EXTRA = ["wave", "arm_id", "task", "partition", "budget", "stage2_total", "stage2_min", "split_mode"]


def lb_branches():
    # Explicit page numbers: --paginate follows links of the form repositories/{id}/..., which some
    # proxies refuse.
    names, page = [], 1
    while True:
        rc, out, err = logx.gh("api", f"repos/{logx.REPO}/branches?per_page=100&page={page}", "-q", ".[].name")
        if rc:
            sys.exit(err)
        got = out.split()
        names += got
        if len(got) < 100:
            break
        page += 1
    return sorted(b for b in names if BRANCH.match(b))


def parse(branch, blind):
    row = logx.parse(branch, blind)
    row.update({k: None for k in EXTRA})
    _, wave, arm, task, g = BRANCH.match(branch).groups()
    row.update(wave=wave, arm_id=arm, task=task, partition=int(g))
    text = (logx.LOGS / (branch.replace("/", "__") + ".log")).read_text(encoding="utf-8")
    m = logx.last(r"^\S+Z RESULT split (\{.*\})\s*$", text)
    sp = logx.lit(m.group(1)) if m else None
    if sp:
        row["split_mode"] = sp.get("mode")
        row["budget"] = sp.get("stage2_label_budget", 0)          # 0 = whole training region
        counts = sp.get("stage2_class_counts")
        if counts:
            row["stage2_total"], row["stage2_min"] = sum(counts), min(counts)
    return row


def main():
    cmd = sys.argv[1]
    if cmd == "fetch":
        bs = lb_branches()
        with ThreadPoolExecutor(4) as ex:
            for i, (b, meta) in enumerate(ex.map(logx.fetch_one, bs), 1):
                print(f"[{i}/{len(bs)}] {b:34s} {(meta.get('run') or {}).get('conclusion', 'NO RUN')}", flush=True)
        return
    blind = cmd == "blind"
    rows = [parse(b, blind) for b in lb_branches()
            if (logx.LOGS / (b.replace("/", "__") + ".json")).exists()]
    fields = logx.FIELDS + EXTRA
    with open(sys.argv[2], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} runs -> {sys.argv[2]}")


if __name__ == "__main__":
    main()
