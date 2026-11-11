#!/usr/bin/env python3
"""Extract run results from GitHub Actions job logs (artifact downloads are blocked here).

  python logx.py fetch            # save the full logs of every target branch to logs/
  python logx.py blind runs_blind.csv   # validation + integrity fields only
  python logx.py full  runs_full.csv    # everything, including test-region numbers

The CSV uses the column names of dgm_forensics.py, so `dgm_forensics.py analyze --runs <csv>`
works on it unchanged (blind: add --blind).
"""
import ast, csv, json, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dgm_forensics import FIELDS, parse_branch  # noqa: E402

REPO = "zarrinmahdavipour/DGM-HMamba"
LOGS = Path(__file__).parent / "logs"
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def targets():
    b = [f"long/r9-b8-g{g}-m{m}" for g in range(5, 10) for m in (42, 7)]
    b += [f"long/r10-{v}-g{g}" for v in ("diff", "static", "noadv", "cnn") for g in range(5, 10)]
    b += [f"long/r11-{v}-g{g}" for v in ("glove", "none", "feature") for g in range(5, 10)]
    b += ["exp/r11-sf-pc2pu", "exp/r11-sf-ip2pu"]
    b += ["exp/fixed2", "exp/fixed2-nodiff", "exp/fixed4", "exp/fixed4-nodiff", "exp/r1-a-way9-aug", "exp/r1-f-nodiff",
          "exp/r3-a-nodiff-ema", "exp/r3-b-ddim-ema", "exp/r3-c-ddpm-ema", "exp/r4-a-nodiff", "exp/r4-b-ddpm",
          "exp/r4-c-ddim", "exp/r5-a-nodiff-1400", "exp/r5-c-ddim-1400"]
    b += [f"exp/r6-random-s{s}" for s in (1, 2, 3, 4, 42)] + ["exp/r6-block-s1", "exp/r6-block-s42"]
    b += [f"exp/r6b-g{g}-m42" for g in (1, 2, 3, 4, 42)] + [f"exp/r6b-g42-m{m}" for m in (1, 2, 3, 4)]
    b += [f"long/r7-p9-g{g}" for g in (1, 2, 3, 4, 42)]
    return b


def gh(*args):
    r = subprocess.run(["gh", *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, r.stdout, r.stderr


def fetch_one(branch):
    out = LOGS / (branch.replace("/", "__") + ".log")
    meta_p = out.with_suffix(".json")
    if out.exists() and meta_p.exists():
        return branch, json.loads(meta_p.read_text())
    rc, so, _ = gh("run", "list", "-R", REPO, "-b", branch, "-L", "20",
                   "--json", "databaseId,status,conclusion,createdAt,headSha,workflowName")
    runs = [r for r in json.loads(so or "[]") if r.get("workflowName", "").startswith("Experiment")
            and r.get("status") == "completed"] if rc == 0 else []
    runs.sort(key=lambda r: (r.get("conclusion") == "success", r.get("createdAt", "")), reverse=True)
    meta = {"branch": branch, "run": runs[0] if runs else None, "jobs": []}
    text = []
    if runs:
        rid = runs[0]["databaseId"]
        rc, so, _ = gh("api", f"repos/{REPO}/actions/runs/{rid}/jobs", "-q", ".jobs[] | [.id, .name, .conclusion] | @tsv")
        for line in so.splitlines():
            jid, name, concl = (line.split("\t") + ["", ""])[:3]
            meta["jobs"].append({"id": jid, "name": name, "conclusion": concl})
            if concl in ("skipped", ""):
                continue
            rc, log, _ = gh("api", f"repos/{REPO}/actions/jobs/{jid}/logs")
            text.append(f"##### JOB {name} ({concl})\n" + log)
    LOGS.mkdir(exist_ok=True)
    out.write_text(ANSI.sub("", "\n".join(text)), encoding="utf-8")
    meta_p.write_text(json.dumps(meta, indent=2))
    return branch, meta


def fetch():
    bs = targets()
    with ThreadPoolExecutor(4) as ex:
        for i, (b, meta) in enumerate(ex.map(fetch_one, bs), 1):
            run = meta.get("run") or {}
            jobs = ", ".join(f"{j['name']}:{j['conclusion']}" for j in meta.get("jobs", []))
            print(f"[{i}/{len(bs)}] {b:26s} {run.get('conclusion', 'NO RUN')}  {jobs}", flush=True)


def last(pattern, text):
    m = None
    for m in re.finditer(pattern, text, flags=re.M):
        pass
    return m


def lit(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def parse(branch, blind):
    log_p = LOGS / (branch.replace("/", "__") + ".log")
    meta = json.loads(log_p.with_suffix(".json").read_text())
    text = log_p.read_text(encoding="utf-8")
    run = meta.get("run") or {}
    rnd, arm = parse_branch(branch)
    row = {k: None for k in FIELDS}
    row.update(branch=branch, round=rnd, arm=arm, run_id=run.get("databaseId"),
               conclusion=run.get("conclusion"), head_sha=run.get("headSha"))
    ts = r"^\S+Z "
    m = last(ts + r"RESULT seeds (\{.*\})\s*$", text)
    seeds = lit(m.group(1)) if m else None
    if seeds:
        row.update(split_seed=seeds.get("split"), model_seed=seeds.get("model"), eval_seed=seeds.get("eval"))
    m = last(ts + r"RESULT split (\{.*\})\s*$", text)
    sp = lit(m.group(1)) if m else None
    if sp:
        row["index_fingerprint"] = sp.get("index_fingerprint")
        row["split"] = "block" if sp.get("block_size") else "random"
    m = last(ts + r"RESULT validation (\{.*\})\s*$", text)
    val = lit(m.group(1)) if m else None
    if val:
        eps = sorted(((int(k.split()[1]), v) for k, v in val.items() if k.startswith("episode")), key=lambda t: t[0])
        sel = [v for _, v in eps] + ([val["after Stage 2"]] if "after Stage 2" in val else [])
        row.update(val_after_s1=val.get("after Stage 1"), val_best=max(sel) if sel else None,
                   val_last5=round(sum(v for _, v in eps[-5:]) / len(eps[-5:]), 3) if eps else None,
                   val_after_s2=val.get("after Stage 2"))
    m = last(ts + r"RESULT train acc .*?\| selected episode: (\S+) \| skipped non-finite updates: (\{.*\})\s*$", text)
    if m:
        row["selected_episode"] = m.group(1)
        sk = lit(m.group(2)) or {}
        row["skipped_updates"] = sum(v for v in sk.values() if isinstance(v, (int, float)))
    m = last(ts + r"RESULT params (\d+)", text)
    if m:
        row["params"] = int(m.group(1))
    if not blind:
        for key, pre in (("5", "5way_5shot"), ("1", "5way_1shot")):
            m = last(ts + rf"RESULT {pre}: OA ([\d.]+) \+/- [\d.]+\s+AA ([\d.]+)\s+kappa ([\d.\-]+)", text)
            if m:
                row[f"test{key}_oa"], row[f"test{key}_aa"], row[f"test{key}_kappa"] = (float(m.group(i)) for i in (1, 2, 3))
        m = last(ts + r"RESULT map \(\d+ draws\): OA ([\d.]+) \+/- [\d.]+\s+AA ([\d.]+) \+/- [\d.]+\s+kappa ([\d.\-]+)", text)
        if m:
            row["map_oa"], row["map_aa"], row["map_kappa"] = (float(m.group(i)) for i in (1, 2, 3))
        m = last(ts + r"RESULT map, queries sharing no pixels with supports: OA ([\d.]+)", text)
        if m:
            row["mapsep_oa"] = float(m.group(1))
        m = last(ts + r"RESULT support-free: OA ([\d.]+)\s+AA ([\d.]+)\s+kappa ([\d.\-]+)\s+\((.*)\)\s*$", text)
        if m:
            row.update(sf_oa=float(m.group(1)), sf_aa=float(m.group(2)), sf_kappa=float(m.group(3)), sf_protocol=m.group(4))
        m = last(ts + r"RESULT pool class counts (\{.*\})\s*$", text)
        pc = lit(m.group(1)) if m else None
        if pc:
            test = next((v for k, v in pc.items() if "test" in k.lower()), None)
            if test:
                row["test_majority"] = round(100 * max(test) / sum(test), 2)
        m = last(ts + r"RESULT split leakage train-test (\{.*?\}) \|", text)
        lk = lit(m.group(1)) if m else None
        if lk:
            row["leak_patches_share"] = round(100 * lk.get("patches_share_pixels", 0), 2)
            row["leak_any_train_in_patch"] = round(100 * lk.get("any_train_in_patch", 0), 2)
    return row


def write(path, blind):
    rows = [parse(b, blind) for b in targets() if (LOGS / (b.replace("/", "__") + ".json")).exists()]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} runs -> {path}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fetch":
        fetch()
    else:
        write(sys.argv[2], blind=(cmd == "blind"))
