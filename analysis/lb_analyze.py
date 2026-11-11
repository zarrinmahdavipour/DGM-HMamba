#!/usr/bin/env python3
"""Confirmatory analysis F1-F10 of the label-budget study, from lb_extract.py output.

    python analysis/lb_analyze.py integrity lb_blind.csv            # completeness, budgets, fingerprints
    python analysis/lb_analyze.py confirm   lb_full.csv  > lb_confirmatory.md

Pairs are formed by (task, partition). A contrast uses only partitions present in both arms.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import lb_stats as S  # noqa: E402

MARGIN = 1.5
MAP, EPI = "map_oa", "test5_oa"          # H1, H5 on the whole-scene map; H2-H4 on 5-way 5-shot episodes
BUDGET = {"c7": 10, "c8": 50, "c9": 10, "c10": 50, "c13": 50, "c17": 50}


def load(path):
    data = defaultdict(dict)              # (arm, task) -> {partition: row}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r.get("arm_id"):
            data[(r["arm_id"], r["task"])][int(r["partition"])] = r
    return data


def val(r, key):
    try:
        return float(r[key])
    except (TypeError, ValueError, KeyError):
        return None


def diffs(data, a, b, task, key):
    A, B = data.get((a, task), {}), data.get((b, task), {})
    parts = sorted(p for p in set(A) & set(B) if val(A[p], key) is not None and val(B[p], key) is not None)
    return np.array([val(A[p], key) - val(B[p], key) for p in parts])


def contrast(data, tasks, pairs, key, kind):
    """pairs: list of (arm_a, arm_b); every pair in every task must pass (intersection-union)."""
    lines, ps, verdicts, p_sup, p_eq, signs = [], [], [], [], [], []
    for a, b in pairs:
        for t in tasks:
            d = diffs(data, a, b, t, key)
            if len(d) < 3:
                lines.append(f"| {a} − {b} | {t} | n={len(d)} | | | | too few pairs |")
                ps.append(1.0)
                verdicts.append("MISSING")
                continue
            p, e, v = S.paired(d), S.tost(d, MARGIN), S.verdict(d, MARGIN)
            p_sup.append(p["p_t"])
            p_eq.append(e["p_tost"])
            signs.append(np.sign(p["mean"]))
            pv = (p["p_t"] if p["mean"] > 0 else 1.0) if kind == "HELPS" else e["p_tost"]
            ps.append(pv)
            verdicts.append(v)
            lines.append(f"| {a} − {b} | {t} | n={p['n']} | {p['mean']:+.2f} | [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}] "
                         f"| [{e['ci90'][0]:+.2f}, {e['ci90'][1]:+.2f}] | {v} (wins {p['wins']}/{p['n']}) |")
    if kind == "SUP_EQ":
        # Fixed order: superiority (same sign in every task, IUT p < 0.05), otherwise equivalence.
        if "MISSING" not in verdicts and len(set(signs)) == 1 and S.iut(p_sup) < 0.05:
            verdicts.insert(0, "superiority:")
            return S.iut(p_sup), verdicts, lines
        verdicts.insert(0, "equivalence:")
        return S.iut(p_eq + [1.0] * verdicts.count("MISSING")), verdicts, lines
    return S.iut(ps), verdicts, lines


def trend(data, tasks, arms, key, decreasing=False, minus=None):
    ps, lines = [], []
    for t in tasks:
        cols = [data.get((a, t), {}) for a in arms]
        sub = [data.get((m, t), {}) for m in minus] if minus else None
        parts = sorted(set.intersection(*(set(c) for c in cols + (sub or []))))
        rows = []
        for p in parts:
            row = [val(c[p], key) for c in cols]
            if sub:
                row = [x - val(s[p], key) if x is not None and val(s[p], key) is not None else None
                       for x, s in zip(row, sub)]
            if None not in row:
                rows.append(row)
        if len(rows) < 3:
            ps.append(1.0)
            lines.append(f"| trend | {t} | n={len(rows)} | too few partitions |")
            continue
        pv = S.page_trend(rows, decreasing=decreasing)
        ps.append(pv)
        means = " < ".join(f"{m:.1f}" for m in np.mean(rows, 0)) if not decreasing else \
            " > ".join(f"{m:.1f}" for m in np.mean(rows, 0))
        lines.append(f"| trend | {t} | n={len(rows)} | {means} | Page p = {pv:.4g} |")
    return S.iut(ps), lines


def confirm(path, tasks):
    data = load(path)
    fam = [
        ("F1 H1 gap (map)", lambda: contrast(data, tasks, [("m1", "m2"), ("c6", "c2")], MAP, "HELPS")),
        ("F2 H1 curve (map)", lambda: trend(data, tasks, ["c2", "c7", "c8", "c11"], MAP)),
        ("F3 H2a source at 10", lambda: contrast(data, tasks, [("c7", "c9")], EPI, "HELPS")),
        ("F4 H2a training", lambda: contrast(data, tasks, [("c2", "c1")], EPI, "HELPS")),
        ("F5 H2b shrinking benefit", lambda: trend(data, tasks, ["c7", "c8", "c11"], EPI, decreasing=True,
                                                   minus=["c9", "c10", "c12"])),
        ("F6 H2c source at pool", lambda: contrast(data, tasks, [("c11", "c12")], EPI, "EQUIVALENT")),
        ("F7 H3 encoder, strict", lambda: contrast(data, tasks, [("m2", "c2")], EPI, "SUP_EQ")),
        ("F8 H3 encoder, pool", lambda: contrast(data, tasks, [("m1", "c6")], EPI, "SUP_EQ")),
        ("F9 H4 alignment, strict", lambda: contrast(data, tasks, [("c2", "c3")], EPI, "HELPS")),
        ("F10 H5a leakage at 50 (map)", lambda: contrast(data, tasks, [("c13", "c8")], MAP, "HELPS")),
    ]
    results = [(name, fn()) for name, fn in fam]
    adj = S.holm([r[0] for _, r in results])
    print(f"# Confirmatory family (tasks: {', '.join(tasks)})\n")
    print("| Claim | IUT p | Holm p | Per-task verdicts |\n|---|---|---|---|")
    for (name, r), h in zip(results, adj):
        v = ", ".join(r[1]) if len(r) == 3 else "trend"
        print(f"| {name} | {r[0]:.4g} | {h:.4g} | {v} |")
    for name, r in results:
        print(f"\n## {name}\n\n| Contrast | Task | Pairs | Mean | 95% CI | 90% CI (TOST) | Verdict |\n|---|---|---|---|---|---|---|")
        print("\n".join(r[-1]))


def integrity(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    print("| Check | Result |\n|---|---|")
    bad = [r["branch"] for r in rows if r["conclusion"] != "success"]
    print(f"| Runs not successful | {len(bad)} {bad[:10]} |")
    sk = [r["branch"] for r in rows if (val(r, "skipped_updates") or 0) > 0]
    print(f"| Runs with skipped updates | {len(sk)} {sk[:10]} |")
    wrong = [r["branch"] for r in rows if r["arm_id"] in BUDGET and val(r, "stage2_min") != BUDGET[r["arm_id"]]]
    print(f"| Budget arms whose Stage-2 pool is not N per class | {len(wrong)} {wrong[:10]} |")
    fp = defaultdict(set)
    for r in rows:
        if r.get("split_mode") == "block":
            fp[(r["task"], r["partition"])].add(r["index_fingerprint"])
    mixed = [k for k, v in fp.items() if len(v) > 1]
    print(f"| (task, partition) with more than one block-split fingerprint | {len(mixed)} {mixed[:10]} |")
    # Nested and shared label subsets (block-split arms only; the leaky arms draw from a different pool).
    by_tp = defaultdict(lambda: {"10": set(), "50": set()})
    for r in rows:
        if r.get("split_mode") != "block" or not r.get("subset_fp"):
            continue
        k = (r["task"], r["partition"])
        b = val(r, "budget")
        if b == 10:
            by_tp[k]["10"].add(r["subset_fp"])
        elif b == 50:
            by_tp[k]["50"].add(r["subset_fp"])
            if r.get("prefix10_fp"):
                by_tp[k]["10"].add(r["prefix10_fp"])
    bad_nest = [k for k, v in by_tp.items() if len(v["10"]) > 1 or len(v["50"]) > 1]
    print(f"| (task, partition) whose 10- or 50-pixel subsets differ across arms or are not nested | {len(bad_nest)} {bad_nest[:10]} |")
    print(f"| (task, partition) with nesting evidence (10-subset seen in a 50-pixel run) | "
          f"{sum(1 for v in by_tp.values() if v['10'] and v['50'])} |")
    want_sel = {a for a in ("c4", "c5", "c6", "c14", "m1", "m3", "m4", "m5", "m6")} | {f"u{i}{j}" for i in "12" for j in "124"}
    wrong_ckpt = [r["branch"] for r in rows if r.get("checkpoint") and r.get("arm_id") and
                  (r["checkpoint"] == "selected") != (r["arm_id"] in want_sel)]
    print(f"| Runs whose evaluated checkpoint (last vs selected) does not match the arm | {len(wrong_ckpt)} {wrong_ckpt[:10]} |")
    cpus = defaultdict(int)
    for r in rows:
        cpus[r.get("cpu") or "not logged"] += 1
    print(f"| CPU models | {dict(cpus)} |")
    grid = defaultdict(set)
    for r in rows:
        grid[(r["arm_id"], r["task"])].add(int(r["partition"]))
    print("\n| Arm | Task | Partitions done |\n|---|---|---|")
    for (a, t), ps in sorted(grid.items()):
        print(f"| {a} | {t} | {len(ps)}: {sorted(ps)} |")


if __name__ == "__main__":
    cmd, path = sys.argv[1], sys.argv[2]
    tasks = sys.argv[3].split(",") if len(sys.argv) > 3 else ["t1", "t2", "t3"]
    integrity(path) if cmd == "integrity" else confirm(path, tasks)
