"""Aggregate the standard-split benchmark (PraNet split, 1,450 training images) from raw run logs.

Reads BENCH_SEED_SUMMARY_JSON (per seed, per cohort) and BENCH_PER_IMAGE lines, pools each model's seeds across runs,
and writes figs/bench_metrics.json with per-cohort mean / SD / min / max and paired tests of DANG against every
baseline:
  * seed level  : two-sided paired t-test over matched seeds (n <= 5) of the per-seed mean Dice;
  * image level : Wilcoxon signed-rank test over images of the per-image Dice averaged over the matched seeds.
Holm's correction is applied per baseline over the three primary cohorts (ColonDB, ETIS, PolypGen).
Runs that have not finished simply contribute fewer seeds; rerun this script after they land.

    uv run --isolated --no-project --with numpy --with scipy python figs/aggregate_bench.py [RUN_LOG_DIR]
"""

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

LOG_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "~/.local/share/openresearch/run-logs").expanduser()
OUT = Path(__file__).with_name("bench_metrics.json")

# model -> run-id prefixes (Kaggle T4 unless noted); seeds are de-duplicated in this order
GROUPS = {
    "PraNet": ["2709562b"],
    "Polyp-PVT": ["48489f3c", "7b7d6418", "e41198e5"],                 # 99 epochs: A100 (fp32)
    "PraNet-V2": ["e733ff21", "65d742b1", "0ddfed1b"],
    "SegDINO": ["5fa721d1"],
    "V8-b": ["54edf050"],                                               # dual stream, seed 42
    "V12-b": ["fb86a0e1", "8f89680e"],
    "R2-a": ["dcfac57f"],
    "R2-c": ["28d364fe", "77e8b315"],
    "A1-noproto": ["87b2758d"],                                         # DANG without the prototype head
    "A2-nolora": ["27a7c9d3"],                                          # DANG without Conv-LoRA (fully frozen)
    "DANG": ["842016dd", "9963f524", "7e80a4ed"],                       # R2-b, the selected model; E1 adds PolypDB
}
DEPLOY_RUNS = ["55037939", "73f86fbe", "ba33170f"]   # one-host T4 latency: D3, D4, D5 (D5 adds SegDINO; later run wins)
BASELINES = ["PraNet", "Polyp-PVT", "PraNet-V2", "SegDINO", "V12-b", "R2-c", "A1-noproto", "A2-nolora"]   # last two: five-seed ablation variants
COHORTS = ["Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "CVC-300", "PolypGen", "PolypDB"]
PRIMARY = ["CVC-ColonDB", "ETIS-LaribPolypDB", "PolypGen"]
METRICS = {"dice": "dice_native", "mdice_tb": "mdice_tb", "iou": "iou_native", "bf1": "boundary_f1", "hd95": "hd95",
           "recall": "recall_native", "precision": "precision_native"}
SEED_RE = re.compile(r"BENCH_SEED_SUMMARY_JSON:(\d+): (\{.*\})\s*$")
IMG_RE = re.compile(r"BENCH_PER_IMAGE: (\{.*\})\s*$")
DEPLOY_RE = re.compile(r"BENCH_JSON: (\{.*\})\s*$")


def load_deploy():
    """Latest deploy run wins per model; latency = mean of the two passes."""
    res = {}
    for pre in DEPLOY_RUNS:
        files = [f for f in glob.glob(str(LOG_DIR / f"{pre}*.log")) if not f.endswith(".supervisor.log")]
        if not files:
            continue
        rows = {}
        for line in open(files[0], errors="replace"):
            if m := DEPLOY_RE.search(line):
                d = json.loads(m.group(1))
                if "error" not in d:                     # a model that failed to build or time is left out
                    rows.setdefault(d["model"], []).append(d)
        for name, ds in rows.items():
            r = {"run": pre, "params_m": ds[0]["params_m"], "gmacs": ds[0]["gmacs_bs1"]}
            for k in ("eager_fp16_bs1_ms", "half_bs1_ms", "graph_half_bs1_ms", "eager_fp16_bs8_img_per_s", "peak_mem_mb_bs1"):
                v = [d[k] for d in ds if k in d]
                if v:
                    r[k] = round(sum(v) / len(v), 2)
            res[name] = r
    return res


def load(prefixes):
    seeds, images = {}, {}
    for pre in prefixes:
        files = [f for f in glob.glob(str(LOG_DIR / f"{pre}*.log")) if not f.endswith(".supervisor.log")]
        if not files:
            continue
        run_seeds, run_imgs = {}, {}
        for line in open(files[0], errors="replace"):
            if m := SEED_RE.search(line):
                run_seeds[int(m.group(1))] = json.loads(m.group(2))
            elif m := IMG_RE.search(line):
                d = json.loads(m.group(1))
                run_imgs.setdefault(d["seed"], {}).setdefault(d["cohort"], {}).update({r["u"]: r["dn"] for r in d["recs"]})
        for s, summ in run_seeds.items():
            if s not in seeds:
                seeds[s] = summ
                images[s] = run_imgs.get(s, {})
            else:                                       # evaluation-only runs add cohorts the training run skipped
                for c, v in summ["cohorts"].items():
                    if c not in seeds[s]["cohorts"]:
                        seeds[s]["cohorts"][c] = v
                        images[s][c] = run_imgs.get(s, {}).get(c, {})
    return seeds, images


def describe(vals):
    a = np.asarray(vals, dtype=float)
    return {"n": len(a), "mean": round(float(a.mean()), 2), "sd": round(float(a.std(ddof=1)), 2) if len(a) > 1 else None,
            "min": round(float(a.min()), 2), "max": round(float(a.max()), 2), "seeds": [round(float(v), 2) for v in a]}


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, run = [None] * len(pvals), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = run
    return adj


def main():
    data = {g: load(p) for g, p in GROUPS.items()}
    out = {"log_dir": str(LOG_DIR), "groups": {}, "tests": {}, "deploy": load_deploy()}
    for g, (seeds, _) in data.items():
        if not seeds:
            continue
        ss = sorted(seeds)
        entry = {"runs": GROUPS[g], "seeds": ss, "params_m": seeds[ss[0]].get("params_m"), "cohorts": {}}
        for c in COHORTS:
            have = [s for s in ss if c in seeds[s]["cohorts"]]
            if have:
                entry["cohorts"][c] = {k: describe([seeds[s]["cohorts"][c]["micro"][v] for s in have])
                                       for k, v in METRICS.items()}
                doms = sorted(set.intersection(*[set(seeds[s]["cohorts"][c].get("per_domain", {})) for s in have]))
                if len(doms) > 1:                       # multi-centre cohorts: Dice per centre / modality group
                    entry["cohorts"][c]["per_domain"] = {
                        d: dict(describe([seeds[s]["cohorts"][c]["per_domain"][d]["dice_native"] for s in have]),
                                n_images=seeds[have[0]]["cohorts"][c]["per_domain"][d]["n"]) for d in doms}
        out["groups"][g] = entry

    d_seeds, d_imgs = data["DANG"]
    for b in BASELINES:
        b_seeds, b_imgs = data[b]
        res = {}
        for c in COHORTS:
            common = sorted(s for s in d_seeds if s in b_seeds and c in d_seeds[s]["cohorts"] and c in b_seeds[s]["cohorts"])
            if not common:
                continue
            dv = np.array([d_seeds[s]["cohorts"][c]["micro"]["dice_native"] for s in common])
            bv = np.array([b_seeds[s]["cohorts"][c]["micro"]["dice_native"] for s in common])
            r = {"seeds": common, "delta": round(float((dv - bv).mean()), 2)}
            if len(common) >= 3:
                r["p_seed"] = float(stats.ttest_rel(dv, bv).pvalue)
            uids = set.intersection(*[set(d_imgs[s].get(c, {})) & set(b_imgs[s].get(c, {})) for s in common])
            if uids:
                u = sorted(uids)
                di = np.array([[d_imgs[s][c][k] for k in u] for s in common]).mean(0)
                bi = np.array([[b_imgs[s][c][k] for k in u] for s in common]).mean(0)
                r.update({"n_images": len(u), "delta_image": round(float((di - bi).mean()), 2),
                          "p_image": float(stats.wilcoxon(di, bi).pvalue), "wins": int((di > bi).sum()),
                          "losses": int((di < bi).sum())})
            res[c] = r
        for key in ("p_seed", "p_image"):
            prim = [c for c in PRIMARY if c in res and key in res[c]]
            for c, a in zip(prim, holm([res[c][key] for c in prim])):
                res[c][key + "_holm"] = a
        out["tests"][b] = res

    OUT.write_text(json.dumps(out, indent=1))
    for g, e in out["groups"].items():
        row = " ".join(f"{c[:8]}={e['cohorts'][c]['dice']['mean']:.2f}" for c in COHORTS if c in e["cohorts"])
        print(f"{g:10s} seeds={e['seeds']} {row}")
    for b, res in out["tests"].items():
        print(b, {c[:8]: (r["delta"], round(r.get("p_seed_holm", r.get("p_seed", float('nan'))), 3),
                          round(r.get("p_image_holm", r.get("p_image", float('nan'))), 4)) for c, r in res.items()})


if __name__ == "__main__":
    main()
