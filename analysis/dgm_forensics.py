#!/usr/bin/env python3
"""Forensic analysis of the existing DGM-HMamba experiment runs.

Three steps, standard library only (plus the GitHub CLI `gh` for `fetch`):

  python dgm_forensics.py check   --repo zarrinmahdavipour/DGM-HMamba
  python dgm_forensics.py fetch   --repo zarrinmahdavipour/DGM-HMamba --out artifacts [--logs]
  python dgm_forensics.py collect --in artifacts --out runs.csv
  python dgm_forensics.py analyze --runs runs.csv --blind      (validation numbers only)
  python dgm_forensics.py analyze --runs runs.csv               (everything, incl. test numbers)

`check`   verifies that Python and the GitHub CLI are ready and the repository is reachable.

`fetch`   downloads the `experiment-outputs` artifact (demo_summary.json, split.json, ...)
          of the latest completed run of every exp/* and long/* branch, plus run metadata.
          Artifacts and logs expire (90 days by default), so archive the result.
`collect` turns every demo_summary.json into one row of runs.csv.
`analyze` runs the pre-specified contrasts (paired by partition) and applies the
          decision rules; nothing in the rules depends on looking at the results first.
          With --blind it shows validation-region numbers only, so that design decisions
          can be written down before any test-region number is seen.

Decision rules (OA in percentage points, delta = with component - without component):
  design decisions use the validation region (val_best), claims use the test region;
  HELPS      : mean delta >= 1.0, 95 % CI lower bound > 0, wins >= n - 1
  NO BENEFIT SHOWN : mean delta < 0.5
  otherwise  : INCONCLUSIVE (default: keep the verified B8 configuration unchanged)
"""

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
from itertools import product
from pathlib import Path

SESOI = 1.0          # smallest effect of interest, OA points
NO_BENEFIT = 0.5     # below this mean delta a component is not kept / a simplification is adopted

# --------------------------------------------------------------------------- statistics

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
        18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000}


def t975(df):
    if df in T975:
        return T975[df]
    keys = sorted(T975)
    for k in keys:
        if k >= df:
            return T975[k]
    return 1.96


def _betacf(a, b, x):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1.0 + aa / c if abs(c) > 1e-30 else 1e-30
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1.0 + aa / c if abs(c) > 1e-30 else 1e-30
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-12:
            break
    return h


def _ibeta(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1 - x) / b


def t_two_sided_p(t, df):
    return _ibeta(df / 2.0, 0.5, df / (df + t * t))


def sign_test_p(wins, n):
    if n == 0:
        return float("nan")
    k = min(wins, n - wins)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def wilcoxon_exact_p(d):
    d = [x for x in d if abs(x) > 1e-12]
    n = len(d)
    if n == 0:
        return float("nan")
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(abs(d[order[j + 1]]) - abs(d[order[i]])) < 1e-12:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    w_obs = sum(r for r, x in zip(ranks, d) if x > 0)
    mean = n * (n + 1) / 4
    extreme = 0
    for signs in product((0, 1), repeat=n):
        w = sum(r for r, s in zip(ranks, signs) if s)
        extreme += abs(w - mean) >= abs(w_obs - mean) - 1e-9
    return extreme / 2 ** n


def paired_summary(deltas):
    n = len(deltas)
    out = {"n": n, "mean": float("nan"), "sd": float("nan"), "lo": float("nan"), "hi": float("nan"),
           "wins": sum(x > 0 for x in deltas), "p_t": float("nan"), "p_sign": float("nan"), "p_wilcoxon": float("nan")}
    if n == 0:
        return out
    m = sum(deltas) / n
    out["mean"] = m
    out["p_sign"] = sign_test_p(out["wins"], n)
    out["p_wilcoxon"] = wilcoxon_exact_p(deltas)
    if n >= 2:
        sd = math.sqrt(sum((x - m) ** 2 for x in deltas) / (n - 1))
        out["sd"] = sd
        half = t975(n - 1) * sd / math.sqrt(n)
        out["lo"], out["hi"] = m - half, m + half
        out["p_t"] = t_two_sided_p(m / (sd / math.sqrt(n)), n - 1) if sd > 0 else 0.0
    return out


def verdict(s):
    if s["n"] == 0 or math.isnan(s["mean"]):
        return "NO DATA"
    if s["mean"] >= SESOI and s["lo"] > 0 and s["wins"] >= s["n"] - 1:
        return "HELPS"
    if s["mean"] < NO_BENEFIT:
        return "NO BENEFIT SHOWN"
    return "INCONCLUSIVE"


def holm(pvals):
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(idx):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def n_for_power(sd, delta, power=0.8):
    """Smallest number of pairs giving ~`power` for a two-sided paired t-test (normal approximation)."""
    if not sd or sd <= 0 or math.isnan(sd):
        return None
    z = {0.8: 0.8416, 0.9: 1.2816}[power]
    for n in range(3, 201):
        if delta * math.sqrt(n) / sd - t975(n - 1) >= z:
            return n
    return None

# --------------------------------------------------------------------------- fetch


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return subprocess.CompletedProcess(cmd, 127, "", f"program not found: {cmd[0]}")


def list_branches(repo):
    r = sh(["gh", "api", "--paginate", f"repos/{repo}/branches?per_page=100", "--jq", ".[].name"])
    names = r.stdout.split() if r.returncode == 0 else []
    if not names:                                   # fall back to git, if installed
        g = sh(["git", "ls-remote", "--heads", f"https://github.com/{repo}"])
        names = [line.split("refs/heads/", 1)[1] for line in g.stdout.splitlines() if "refs/heads/" in line]
        if not names:
            sys.exit("Could not list the repository's branches.\n" + (r.stderr or g.stderr))
    return sorted(n for n in names if re.match(r"(exp|long)/", n))


def check(args):
    print(f"Python {sys.version.split()[0]}  ({'ok' if sys.version_info >= (3, 8) else 'too old: need 3.8+'})")
    v = sh(["gh", "--version"])
    if v.returncode:
        print("GitHub CLI: NOT FOUND. Install it (winget install --id GitHub.cli), then open a NEW terminal.")
        return
    print("GitHub CLI:", v.stdout.splitlines()[0])
    a = sh(["gh", "auth", "status"])
    print("GitHub login:", "ok" if a.returncode == 0 else "NOT LOGGED IN. Run: gh auth login")
    if a.returncode:
        return
    b = list_branches(args.repo)
    print(f"Repository {args.repo}: {len(b)} experiment branches found")
    print("Ready." if b else "No experiment branches found: check the repository name.")


def fetch(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    branches = list_branches(args.repo)
    print(f"{len(branches)} experiment branches")
    for i, b in enumerate(branches, 1):
        d = out / b.replace("/", "__")
        meta_p = d / "run_meta.json"
        if meta_p.exists() and json.loads(meta_p.read_text(encoding="utf-8")).get("run"):
            print(f"  [{i}/{len(branches)}] {b:32s} already downloaded")
            continue
        if (d / "outputs").exists():                 # remove a partial earlier download
            shutil.rmtree(d / "outputs")
        r = sh(["gh", "run", "list", "-R", args.repo, "-b", b, "-L", "20",
                "--json", "databaseId,status,conclusion,createdAt,headSha,workflowName"])
        runs = json.loads(r.stdout or "[]") if r.returncode == 0 else []
        runs = [x for x in runs if x.get("status") == "completed" and x.get("workflowName", "").startswith("Experiment")]
        runs.sort(key=lambda x: (x.get("conclusion") == "success", x.get("createdAt", "")), reverse=True)
        got = None
        for run in runs:
            dl = sh(["gh", "run", "download", str(run["databaseId"]), "-R", args.repo,
                     "-n", "experiment-outputs", "-D", str(d / "outputs")])
            if dl.returncode == 0:
                got = run
                break
        meta = {"branch": b, "run": got, "candidates": len(runs)}
        if got and args.logs:
            log = sh(["gh", "run", "view", str(got["databaseId"]), "-R", args.repo, "--log"])
            (d / "run.log").write_text(log.stdout, encoding="utf-8")
        d.mkdir(parents=True, exist_ok=True)
        meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        status = (f"run {got['databaseId']} {got['conclusion']}" if got else
                  "NO ARTIFACT (normal for code branches; re-run fetch later if the run was still going)")
        print(f"  [{i}/{len(branches)}] {b:32s} {status}")

# --------------------------------------------------------------------------- collect

FIELDS = ["branch", "round", "arm", "split_seed", "model_seed", "eval_seed", "run_id", "conclusion", "head_sha",
          "split", "patch", "no_diffusion", "ddim", "T", "encoder", "static", "lambda_adv", "anchor_init",
          "class_embeddings", "stage2_steps", "params", "index_fingerprint",
          "val_after_s1", "val_best", "val_last5", "val_after_s2", "selected_episode",
          "test5_oa", "test5_aa", "test5_kappa", "test1_oa", "test1_aa", "test1_kappa",
          "map_oa", "map_aa", "map_kappa", "mapsep_oa",
          "sf_oa", "sf_aa", "sf_kappa", "sf_protocol", "test_majority",
          "leak_patches_share", "leak_any_train_in_patch", "skipped_updates",
          "time_total_s", "time_stage2_s", "inference_ms"]


def parse_branch(branch):
    name = branch.split("/", 1)[1]
    m = re.match(r"(r\d+b?|fixed\d)(?:-(.*))?$", name)
    rnd, rest = (m.group(1), m.group(2) or "") if m else (name, "")
    arm = re.sub(r"(-g\d+)?(-m\d+)?$", "", "-" + rest)       # trailing partition / model-seed tags
    if rnd == "r6":
        arm = re.sub(r"-s\d+$", "", arm)                      # r6-random-s1: seed tag
    return rnd, arm.strip("-") or "base"


def pct(x):
    return None if x is None else round(100 * x, 3)


def row_from_summary(branch, meta, s):
    a, h, si = s.get("args", {}), s.get("history", {}), s.get("split_info", {})
    seeds = s.get("seeds", {})
    val = h.get("validation", {}) or {}
    eps = sorted(((int(k.split()[1]), v) for k, v in val.items() if k.startswith("episode")), key=lambda t: t[0])
    sel = [v for _, v in eps] + ([val["after Stage 2"]] if "after Stage 2" in val else [])
    em = s.get("episodic_metrics", {}) or {}
    k5 = next((k for k in em if k.endswith("way_5shot")), None)
    k1 = next((k for k in em if k.endswith("way_1shot")), None)
    mm, sf = s.get("map_metrics") or {}, s.get("support_free") or {}
    counts = (si.get("class_counts") or {}).get("test")
    sk = h.get("skipped_updates") or {}
    rnd, arm = parse_branch(branch)
    run = (meta or {}).get("run") or {}
    return {
        "branch": branch, "round": rnd, "arm": arm,
        "split_seed": seeds.get("split", a.get("split_seed") if a.get("split_seed") is not None else a.get("seed")),
        "model_seed": seeds.get("model", a.get("seed")), "eval_seed": seeds.get("eval", a.get("eval_seed")),
        "run_id": run.get("databaseId"), "conclusion": run.get("conclusion"), "head_sha": run.get("headSha"),
        "split": a.get("split", "random"), "patch": a.get("patch_size"), "no_diffusion": a.get("no_diffusion"),
        "ddim": a.get("ddim"), "T": a.get("T"), "encoder": a.get("encoder", "hdpm"),
        "static": a.get("static_prototypes", False), "lambda_adv": a.get("lambda_adv"),
        "anchor_init": a.get("anchor_init", "embedding"), "class_embeddings": a.get("class_embeddings"),
        "stage2_steps": a.get("stage2_steps"), "params": s.get("trainable_parameters"),
        "index_fingerprint": si.get("index_fingerprint"),
        "val_after_s1": pct(val.get("after Stage 1")),
        "val_best": pct(max(sel)) if sel else None,
        "val_last5": pct(sum(v for _, v in eps[-5:]) / len(eps[-5:])) if eps else None,
        "val_after_s2": pct(val.get("after Stage 2")),
        "selected_episode": h.get("selected_episode"),
        "test5_oa": pct(em.get(k5, {}).get("oa_mean")) if k5 else None,
        "test5_aa": pct(em.get(k5, {}).get("aa")) if k5 else None,
        "test5_kappa": em.get(k5, {}).get("kappa") if k5 else None,
        "test1_oa": pct(em.get(k1, {}).get("oa_mean")) if k1 else None,
        "test1_aa": pct(em.get(k1, {}).get("aa")) if k1 else None,
        "test1_kappa": em.get(k1, {}).get("kappa") if k1 else None,
        "map_oa": pct(mm.get("oa_mean")), "map_aa": pct(mm.get("aa_mean")), "map_kappa": mm.get("kappa_mean"),
        "mapsep_oa": pct((mm.get("support_separated") or {}).get("oa_mean")),
        "sf_oa": pct(sf.get("oa")), "sf_aa": pct(sf.get("aa")), "sf_kappa": sf.get("kappa"),
        "sf_protocol": sf.get("protocol"),
        "test_majority": pct(max(counts) / sum(counts)) if counts else None,
        "leak_patches_share": pct((si.get("leakage") or {}).get("patches_share_pixels")),
        "leak_any_train_in_patch": pct((si.get("leakage") or {}).get("any_train_in_patch")),
        "skipped_updates": sum(v for v in sk.values() if isinstance(v, (int, float))) if isinstance(sk, dict) else sk,
        "time_total_s": (s.get("timings_s") or {}).get("total_s"),
        "time_stage2_s": (s.get("timings_s") or {}).get("stage2_s"),
        "inference_ms": (s.get("timings_s") or {}).get("inference_ms_per_patch"),
    }


def collect(args):
    rows = []
    for d in sorted(p for p in Path(args.inp).iterdir() if p.is_dir()):
        meta_p = d / "run_meta.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
        branch = meta.get("branch") or d.name.replace("__", "/")
        found = sorted(d.rglob("demo_summary.json"), key=lambda p: len(p.parts))
        if not found:
            print(f"  {branch:32s} no demo_summary.json")
            continue
        rows.append(row_from_summary(branch, meta, json.loads(found[0].read_text(encoding="utf-8"))))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} runs -> {args.out}")

# --------------------------------------------------------------------------- analyze

# Round 10/11 ablations: (label, family, component-present arm, component-absent arm)
ABLATIONS = [
    ("Diffusion enhancement (DADGE, T=10 demo scale)", "R10", ("r10", "diff"), ("r9", "b8")),
    ("DSSPE evolution + L_gcl (B8 vs static class means)", "R10", ("r9", "b8"), ("r10", "static")),
    ("Domain-adversarial alignment", "R10", ("r9", "b8"), ("r10", "noadv")),
    ("Encoder HDPM vs 3D-CNN (UNMATCHED pilot)", "R10", ("r9", "b8"), ("r10", "cnn")),
    ("GloVe anchors + L_sem vs random anchors", "R11", ("r11", "glove"), ("r9", "b8")),
    ("Random-anchor semantic term vs none", "R11", ("r9", "b8"), ("r11", "none")),
    ("GloVe anchors + L_sem vs no semantic term (same code)", "R11", ("r11", "glove"), ("r11", "none")),
    ("Feature anchors vs no semantic term (same code)", "R11", ("r11", "feature"), ("r11", "none")),
]
# Development rounds, pixel split, single runs: (round, no-diffusion branch, diffusion branches)
DEV_DIFFUSION = [
    ("fixed2", "exp/fixed2-nodiff", ["exp/fixed2"]),
    ("fixed4", "exp/fixed4-nodiff", ["exp/fixed4"]),
    ("r1", "exp/r1-f-nodiff", ["exp/r1-a-way9-aug"]),
    ("r3", "exp/r3-a-nodiff-ema", ["exp/r3-c-ddpm-ema", "exp/r3-b-ddim-ema"]),
    ("r4", "exp/r4-a-nodiff", ["exp/r4-b-ddpm", "exp/r4-c-ddim"]),
    ("r5", "exp/r5-a-nodiff-1400", ["exp/r5-c-ddim-1400"]),
]
METRICS = [("val_best", "validation OA (selection metric)"), ("val_last5", "validation OA, mean of last 5 checks"),
           ("test5_oa", "test 5-shot OA"), ("test1_oa", "test 1-shot OA"), ("map_oa", "test 9-class map OA")]


def fnum(x, d=2):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def load(path):
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        for k, v in r.items():
            if v in ("", "None"):
                r[k] = None
            else:
                try:
                    r[k] = float(v) if re.fullmatch(r"-?\d+(\.\d+)?(e-?\d+)?", v) else v
                except ValueError:
                    pass
        rows.append(r)
    return rows


def pick(rows, rnd, arm, model_seed=None):
    out = {}
    for r in rows:
        if r["round"] == rnd and r["arm"] == arm and (model_seed is None or r["model_seed"] == model_seed):
            out[int(r["split_seed"])] = r
    return out


def analyze(args):
    rows = load(args.runs)
    blind = args.blind
    metrics = METRICS[:2] if blind else METRICS
    L, warnings = [], []
    L.append("# Forensic analysis of existing runs" + (" – VALIDATION ONLY (blind)" if blind else "") + "\n")
    if blind:
        L.append("Test-region numbers are hidden. Write down your design decisions from this report "
                 "before running `analyze` without --blind.\n")
    L.append(f"Rules: HELPS = mean ≥ {SESOI}, 95 % CI > 0, wins ≥ n−1; NO BENEFIT SHOWN = mean < {NO_BENEFIT}; "
             "else INCONCLUSIVE. Δ = with component − without, OA points, paired by partition.\n")

    # ---- integrity of Round 9-11 pairs
    claims = {"R10": [], "R11": []}
    L.append("## 1. Round 10 / 11 ablations (block split, partitions 5–9, model seed 42)\n")
    for label, fam, (r_with, a_with), (r_wo, a_wo) in ABLATIONS:
        W, O = pick(rows, r_with, a_with, 42), pick(rows, r_wo, a_wo, 42)
        parts = sorted(set(W) & set(O))
        L.append(f"### {label}\n\nwith = `{r_with}-{a_with}`, without = `{r_wo}-{a_wo}`, partitions {parts}\n")
        for s in parts:
            fw, fo = W[s].get("index_fingerprint"), O[s].get("index_fingerprint")
            if fw and fo and fw != fo:
                warnings.append(f"{label}: partition {s} split fingerprints differ ({fw} vs {fo}) – not a valid pair")
            for r in (W[s], O[s]):
                if r.get("conclusion") not in (None, "success"):
                    warnings.append(f"{r['branch']}: run conclusion {r['conclusion']}")
                if r.get("skipped_updates"):
                    warnings.append(f"{r['branch']}: {int(r['skipped_updates'])} skipped non-finite updates")
        pilot = "UNMATCHED" in label
        L.append("| metric | mean Δ | SD | 95 % CI | wins | p (t) | p (sign) | p (Wilcoxon) | verdict |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        pilot_sd = {}
        for key, name in metrics:
            d = [W[s][key] - O[s][key] for s in parts if W[s].get(key) is not None and O[s].get(key) is not None]
            st = paired_summary(d)
            L.append(f"| {name} | {fnum(st['mean'])} | {fnum(st['sd'])} | [{fnum(st['lo'])}, {fnum(st['hi'])}] | "
                     f"{st['wins']}/{st['n']} | {fnum(st['p_t'], 3)} | {fnum(st['p_sign'], 3)} | "
                     f"{fnum(st['p_wilcoxon'], 3)} | {'PILOT (unmatched)' if pilot else verdict(st)} |")
            if key == "test5_oa" and st["n"] >= 2 and not pilot:
                claims[fam].append((label, st))
            if pilot and st["n"] >= 2:
                pilot_sd[key] = st["sd"]
        for key, sdv in pilot_sd.items():
            if key in ("val_best", "test5_oa"):
                L.append(f"\nPilot only (capacity and tuning not matched). {key}: SD of paired difference "
                         f"{fnum(sdv)}; pairs needed for 80 % power at Δ=1.0 / 1.5 / 2.0: "
                         f"{n_for_power(sdv, 1.0)} / {n_for_power(sdv, 1.5)} / {n_for_power(sdv, 2.0)}")
        tw = [W[s].get("time_total_s") for s in parts if W[s].get("time_total_s")]
        to = [O[s].get("time_total_s") for s in parts if O[s].get("time_total_s")]
        if tw and to:
            L.append(f"\nRun time (CPU): with {sum(tw)/len(tw)/3600:.1f} h, without {sum(to)/len(to)/3600:.1f} h; "
                     f"params with {W[parts[0]].get('params')}, without {O[parts[0]].get('params')}")
        L.append("")

    if not blind:
        L.append("## 2. Claims check (test 5-shot OA, Holm-adjusted within each round)\n")
        L.append("| family | contrast | mean Δ | p (t) | Holm p | claimable (mean ≥ 1, Holm p < 0.05) |")
        L.append("|---|---|---|---|---|---|")
    for fam, items in claims.items():
        if not items or blind:
            continue
        adj = holm([st["p_t"] for _, st in items])
        for (label, st), pa in zip(items, adj):
            L.append(f"| {fam} | {label} | {fnum(st['mean'])} | {fnum(st['p_t'], 3)} | {fnum(pa, 3)} | "
                     f"{'yes' if st['mean'] >= SESOI and pa < 0.05 else 'no'} |")
    L.append("")

    # ---- development diffusion pairs
    by_branch = {r["branch"]: r for r in rows}
    L.append("## 3. Diffusion in development rounds (pixel split, single runs, leaky validation)\n")
    L.append("Δ = mean(diffusion arms) − no-diffusion arm. One value per round; sign test across rounds.\n")
    L.append("| round | Δ val_best | Δ test 5-shot | Δ test 1-shot | diffusion arms found | skipped updates (diff arms) | time ratio diff/nodiff |")
    L.append("|---|---|---|---|---|---|---|")
    dv, dt = [], []
    for rnd, nb, dbs in DEV_DIFFUSION:
        base, arms = by_branch.get(nb), [by_branch[b] for b in dbs if b in by_branch]
        if not base or not arms:
            L.append(f"| {rnd} | – | – | – | {len(arms)} | – | – |")
            continue
        def dm(key):
            vals = [a[key] for a in arms if a.get(key) is not None]
            return None if not vals or base.get(key) is None else sum(vals) / len(vals) - base[key]
        a1, a2, a3 = dm("val_best"), dm("test5_oa"), dm("test1_oa")
        if a1 is not None: dv.append(a1)
        if a2 is not None: dt.append(a2)
        sk = sum(int(a.get("skipped_updates") or 0) for a in arms)
        tr = [a["time_total_s"] / base["time_total_s"] for a in arms if a.get("time_total_s") and base.get("time_total_s")]
        L.append(f"| {rnd} | {fnum(a1)} | {'hidden' if blind else fnum(a2)} | {'hidden' if blind else fnum(a3)} | "
                 f"{len(arms)} | {sk} | {fnum(sum(tr)/len(tr), 1) if tr else '–'} |")
    for name, d in ((("val_best", dv),) if blind else (("val_best", dv), ("test 5-shot", dt))):
        st = paired_summary(d)
        L.append(f"\n{name}: {st['wins']}/{st['n']} rounds favour diffusion, mean Δ {fnum(st['mean'])}, "
                 f"sign-test p {fnum(st['p_sign'], 3)}")
    L.append("")

    # ---- leakage
    if blind:
        L.append("## 4. Leakage (Round 6): test-region numbers, hidden in blind mode\n")
    else:
        L.append("## 4. Leakage: pixel split vs block split (Round 6, same settings, whole-scene scaling)\n")
    rnd6 = [] if blind else [r for r in rows if r["round"] == "r6"]
    if not blind:
        L.append("| arm | split seed | test 5-shot OA | test 1-shot OA | map OA | patches sharing pixels with training (%) |")
        L.append("|---|---|---|---|---|---|")
    for r in sorted(rnd6, key=lambda r: (r["arm"], r["split_seed"] or 0)):
        L.append(f"| {r['arm']} | {fnum(r['split_seed'], 0)} | {fnum(r['test5_oa'])} | {fnum(r['test1_oa'])} | "
                 f"{fnum(r['map_oa'])} | {fnum(r['leak_patches_share'], 1)} |")
    for key in ("test5_oa", "test1_oa", "map_oa"):
        ra = [r[key] for r in rnd6 if r["arm"] == "random" and r.get(key) is not None]
        bl = [r[key] for r in rnd6 if r["arm"] == "block" and r.get(key) is not None]
        if ra and bl:
            L.append(f"\n{key}: random {sum(ra)/len(ra):.2f} (n={len(ra)}) − block {sum(bl)/len(bl):.2f} (n={len(bl)}) "
                     f"= inflation {sum(ra)/len(ra) - sum(bl)/len(bl):+.2f} points (descriptive; n too small for a test)")
    L.append("")

    # ---- variance components
    L.append("## 5. Variance components (for planning the confirmatory round)\n")
    def sd(v):
        m = sum(v) / len(v)
        return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else float("nan")
    for key in (("val_best",) if blind else ("val_best", "test5_oa")):
        part = [r[key] for r in rows if r["round"] == "r6b" and r.get("model_seed") == 42 and r.get(key) is not None]
        seed = [r[key] for r in rows if r["round"] == "r6b" and r.get("split_seed") == 42 and r.get(key) is not None]
        L.append(f"- {key}: SD across partitions (seed 42) = {fnum(sd(part)) if len(part) > 1 else '–'} (n={len(part)}); "
                 f"SD across model seeds (partition 42) = {fnum(sd(seed)) if len(seed) > 1 else '–'} (n={len(seed)})")
    b42, b7 = pick(rows, "r9", "b8", 42), pick(rows, "r9", "b8", 7)
    vk = "val_best" if blind else "test5_oa"
    d = [b42[s][vk] - b7[s][vk] for s in sorted(set(b42) & set(b7))
         if b42[s].get(vk) is not None and b7[s].get(vk) is not None]
    if len(d) > 1:
        L.append(f"- B8 seed 42 − seed 7 on the same partition ({vk}): SD of difference {fnum(sd(d))} (n={len(d)}) "
                 f"→ single-run seed SD ≈ {fnum(sd(d) / math.sqrt(2))}")
    L.append("")

    # ---- patch size
    p9, p7 = pick(rows, "r7", "p9", 42), pick(rows, "r6b", "base", 42)
    parts = sorted(set(p9) & set(p7))
    if parts:
        L.append("## 6. Patch 9×9 vs 7×7 (Round 7 p9 vs Round 6b; different code commits – check equivalence)\n")
        for key in (("val_best",) if blind else ("val_best", "test5_oa")):
            st = paired_summary([p9[s][key] - p7[s][key] for s in parts if p9[s].get(key) is not None and p7[s].get(key) is not None])
            L.append(f"- {key}: mean Δ {fnum(st['mean'])} [{fnum(st['lo'])}, {fnum(st['hi'])}], wins {st['wins']}/{st['n']} → {verdict(st)}")
        L.append("")

    # ---- support-free
    L.append("## 7. Support-free classification (Round 11, Stage 1 only, GloVe anchors)\n")
    if blind:
        L.append("Test-region numbers: hidden in blind mode.\n")
    else:
        L.append("| branch | OA | AA | κ | majority-class OA of test region | chance AA (9 classes) | manuscript Table 13 (PU cross-scene) |")
        L.append("|---|---|---|---|---|---|---|")
    for r in ([] if blind else rows):
        if r.get("sf_oa") is not None:
            L.append(f"| {r['branch']} | {fnum(r['sf_oa'])} | {fnum(r['sf_aa'])} | {fnum(r['sf_kappa'], 3)} | "
                     f"{fnum(r['test_majority'])} | 11.11 | {'82.4 / 79.7 (PC→PU)' if 'pc2pu' in r['branch'] else '–'} |")
    L.append("")

    if warnings:
        L.append("## Integrity warnings\n")
        L.extend(f"- {w}" for w in warnings)
    else:
        L.append("## Integrity warnings\n\n- none detected (fingerprints, run conclusions, skipped updates)")
    text = "\n".join(L) + "\n"
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")   # Windows consoles
        except (AttributeError, ValueError):
            pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("check"); k.add_argument("--repo", required=True)
    f = sub.add_parser("fetch"); f.add_argument("--repo", required=True); f.add_argument("--out", default="artifacts")
    f.add_argument("--logs", action="store_true", help="also save each run's full log")
    c = sub.add_parser("collect"); c.add_argument("--in", dest="inp", default="artifacts"); c.add_argument("--out", default="runs.csv")
    a = sub.add_parser("analyze"); a.add_argument("--runs", default="runs.csv"); a.add_argument("--out", default=None)
    a.add_argument("--blind", action="store_true", help="validation numbers only (no test-region numbers)")
    args = p.parse_args()
    if args.cmd == "analyze" and args.out is None:
        args.out = "forensic_report_validation.md" if args.blind else "forensic_report.md"
    {"check": check, "fetch": fetch, "collect": collect, "analyze": analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
