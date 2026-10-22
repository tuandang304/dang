"""Aggregate the standard-split benchmark (PraNet split, 1,450 training images) from raw run logs.

Reads BENCH_SEED_SUMMARY_JSON (per seed, per cohort) and BENCH_PER_IMAGE lines, pools each model's seeds across runs,
and writes results/bench_metrics.json with per-cohort mean / SD / min / max and paired tests of DANG against every
baseline:
  * seed level  : two-sided paired t-test over matched seeds (n <= 5) of the per-seed mean Dice;
  * image level : Wilcoxon signed-rank test over images of the per-image Dice averaged over the matched seeds
                  (descriptive: frames from one video are not independent);
  * sensitivity : Welch's unpaired t-test over seeds (runs of different codebases share no randomness).
Holm's correction is applied per baseline over the three primary cohorts (ColonDB, ETIS, PolypGen), and, as a
sensitivity check, jointly over all published baselines x primary cohorts (p_seed_holm_all).
Statistics are stored at full precision; rounding happens only for display (scripts/bench_tables.py).
The logs are the evaluation lines of the paper's runs, extracted to results/logs/<model>__<run id>.log.gz.

    uv run --locked python scripts/aggregate_bench.py [LOG_DIR]
"""

import glob
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "logs"
OUT = ROOT / "results" / "bench_metrics.json"

# model -> run-id prefixes (Kaggle T4 unless noted); seeds are de-duplicated in this order.
# "DANG" is AdaPolyp (the project's working name; R2-b in the logs), V12-b / R2-c the decoder variants of Table 8.
GROUPS = {
    "PraNet": ["2709562b"],
    "Polyp-PVT": ["48489f3c", "7b7d6418", "e41198e5"],                 # 99 epochs: A100 (fp32)
    "PraNet-V2": ["e733ff21", "65d742b1", "0ddfed1b"],
    "SegDINO": ["5fa721d1"],
    "V12-b": ["fb86a0e1", "8f89680e"],
    "R2-c": ["28d364fe", "77e8b315"],
    "A1-noproto": ["87b2758d"],                                         # DANG without the prototype head
    "A2-nolora": ["27a7c9d3"],                                          # DANG without Conv-LoRA (fully frozen)
    "A3-pyramid-only": ["0afcc5a7"],                                    # neither Conv-LoRA nor prototype head
    "DANG": ["842016dd", "9963f524", "7e80a4ed"],                       # R2-b, the selected model; E1 adds PolypDB
}
DEPLOY_RUNS = ["55037939", "73f86fbe", "ba33170f"]   # one-host T4 latency: D3, D4, D5 (D5 adds SegDINO; later run wins)
PUBLISHED = ["PraNet", "Polyp-PVT", "PraNet-V2", "SegDINO"]
BASELINES = PUBLISHED + ["V12-b", "R2-c", "A1-noproto", "A2-nolora", "A3-pyramid-only"]
FAIL = 50.0                                         # a frame "fails" when its Dice is below 50 %
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
        files = glob.glob(str(LOG_DIR / f"*__{pre}.log.gz"))
        if not files:
            continue
        rows = {}
        for line in gzip.open(files[0], "rt", errors="replace"):
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
        files = glob.glob(str(LOG_DIR / f"*__{pre}.log.gz"))
        if not files:
            continue
        run_seeds, run_imgs = {}, {}
        for line in gzip.open(files[0], "rt", errors="replace"):
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
    return {"n": len(a), "mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else None,
            "min": float(a.min()), "max": float(a.max()), "seeds": [float(v) for v in a]}


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, run = [None] * len(pvals), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = run
    return adj


def main():
    data = {g: load(p) for g, p in GROUPS.items()}
    out = {"log_dir": "results/logs", "groups": {}, "tests": {}, "deploy": load_deploy()}
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
        imgs = data[g][1]
        for c in COHORTS:
            have = [s for s in ss if c in entry["cohorts"] and imgs.get(s, {}).get(c)]
            if c in entry["cohorts"] and have:
                per = [np.array(list(imgs[s][c].values())) for s in have]
                entry["cohorts"][c]["median_dice"] = describe([np.median(v) for v in per])
                entry["cohorts"][c]["fail_rate"] = describe([100.0 * (v < FAIL).mean() for v in per])
            if c in entry["cohorts"] and "per_domain" in entry["cohorts"][c]:
                hv = [s for s in ss if c in seeds[s]["cohorts"]]
                entry["cohorts"][c]["dice_macro"] = describe([seeds[s]["cohorts"][c]["macro"]["dice_native"] for s in hv])
        if any("deploy_half" in seeds[s] for s in ss):          # FP16-weight accuracy check logged by the DANG runs
            entry["deploy_half"] = {c: describe([seeds[s]["deploy_half"][c]["micro_dice"] for s in ss
                                                 if c in seeds[s].get("deploy_half", {})])
                                    for c in seeds[ss[0]].get("deploy_half", {})}
        mins = [seeds[s]["train_minutes"] for s in ss if seeds[s].get("train_minutes")]
        if mins:
            entry["train_minutes"] = describe(mins)
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
            r = {"seeds": common, "delta": float((dv - bv).mean())}
            if len(common) >= 3:
                r["p_seed"] = float(stats.ttest_rel(dv, bv).pvalue)
                r["p_welch"] = float(stats.ttest_ind(dv, bv, equal_var=False).pvalue)
                ci = stats.t.ppf(0.975, len(common) - 1) * (dv - bv).std(ddof=1) / np.sqrt(len(common))
                r["ci95"] = [float((dv - bv).mean() - ci), float((dv - bv).mean() + ci)]
            uids = set.intersection(*[set(d_imgs[s].get(c, {})) & set(b_imgs[s].get(c, {})) for s in common])
            if uids:
                u = sorted(uids)
                di = np.array([[d_imgs[s][c][k] for k in u] for s in common]).mean(0)
                bi = np.array([[b_imgs[s][c][k] for k in u] for s in common]).mean(0)
                r.update({"n_images": len(u), "delta_image": float((di - bi).mean()),
                          "median_diff_image": float(np.median(di - bi)),
                          "p_image": float(stats.wilcoxon(di, bi).pvalue), "wins": int((di > bi).sum()),
                          "losses": int((di < bi).sum())})
            res[c] = r
        for key in ("p_seed", "p_image"):
            prim = [c for c in PRIMARY if c in res and key in res[c]]
            for c, a in zip(prim, holm([res[c][key] for c in prim])):
                res[c][key + "_holm"] = a
        out["tests"][b] = res

    # sensitivity: one Holm family over all published baselines x primary cohorts (seed-level paired t-tests)
    fam = [(b, c) for b in PUBLISHED for c in PRIMARY if "p_seed" in out["tests"].get(b, {}).get(c, {})]
    for (b, c), a in zip(fam, holm([out["tests"][b][c]["p_seed"] for b, c in fam])):
        out["tests"][b][c]["p_seed_holm_all"] = a
    OUT.write_text(json.dumps(out, indent=1))
    for g, e in out["groups"].items():
        row = " ".join(f"{c[:8]}={e['cohorts'][c]['dice']['mean']:.2f}" for c in COHORTS if c in e["cohorts"])
        print(f"{g:10s} seeds={e['seeds']} {row}")
    for b, res in out["tests"].items():
        print(b, {c[:8]: (r["delta"], round(r.get("p_seed_holm", r.get("p_seed", float('nan'))), 3),
                          round(r.get("p_image_holm", r.get("p_image", float('nan'))), 4)) for c, r in res.items()})


if __name__ == "__main__":
    main()
