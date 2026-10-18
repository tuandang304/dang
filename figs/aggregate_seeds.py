"""Aggregate per-seed SEED_SUMMARY_JSON records for the Round 9/10 suite (seeds 42-46).

Writes figs/seed_metrics.json (raw per-seed values) and prints mean +- sample SD and
paired two-sided t-tests used in Table 3 of paper.tex.
Regenerate: uv run --no-project --with numpy --with scipy python figs/aggregate_seeds.py
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

LOGS = Path("/home/tuan/.local/share/openresearch/run-logs")
OUT = Path(__file__).resolve().parent / "seed_metrics.json"

# arm -> list of run ids whose SEED_SUMMARY_JSON lines are pooled (seed 42 from R9, 43-46 from R10)
ARMS = {
    "baseline": ["f328e771-e3d0-4ae5-8a74-f4285e5886fa", "540e36c9-2121-44cd-becb-2064335e7aa7"],
    "safeconcat": ["e770b25c-18cd-43d0-b924-66c4e2c2819d", "85f4db55-4f1b-4ad0-8ae2-873e9b4bfb2f"],
    "protodpt_passe": ["f676224f-598d-4143-8e4f-f04373be94de", "c171badb-b5cb-4c82-86af-bcd82c1e9409"],
    "flagship": [
        "db1786be-b7b7-4c1c-9a4b-c87bda43501d",
        "01713a3a-6643-4a5a-b407-54adf85a067e",
        "7091cc66-ec02-41b7-96f9-231a40c8a7e6",
    ],
    "retinex_style": ["5356b752-c8f3-47f4-9224-bb6c466b356b"],
}

# metric key -> path into a seed summary
METRICS = {
    "s1s2_dice": ("s1_s2_tiny_camo_micro", "dice"),
    "s1s2_iou": ("s1_s2_tiny_camo_micro", "iou"),
    "s1s2_bf1": ("s1_s2_tiny_camo_micro", "boundary_f1"),
    "s1s2_hd95": ("s1_s2_tiny_camo_micro", "hd95"),
    "s1s2_ep": ("s1_s2_tiny_camo_micro", "empty_pred"),
    "s1s3_dice": ("s1_s3_adverse_macro", "dice"),
    "s1s3_bf1": ("s1_s3_adverse_macro", "boundary_f1"),
    "s1s3_hd95": ("s1_s3_adverse_macro", "hd95"),
    "s1pra_dice": ("s1_pranet_external_macro", "dice"),
    "s1pra_bf1": ("s1_pranet_external_macro", "boundary_f1"),
    "s1pra_hd95": ("s1_pranet_external_macro", "hd95"),
    "s1pgdev_dice": ("s1_ood_polypgen_macro", "dice"),
    "s1_kvasir_dice": ("s1_in_domain_kvasir", "dice"),
    "full_kvasir_dice": ("in_domain_kvasir", "dice"),
    "full_kvasir_bf1": ("in_domain_kvasir", "boundary_f1"),
    "full_etis_dice": ("pranet_external_per_dataset", "Ext_ETIS_LaribPolypDB", "dice"),
    "full_etis_bf1": ("pranet_external_per_dataset", "Ext_ETIS_LaribPolypDB", "boundary_f1"),
    "full_colondb_dice": ("pranet_external_per_dataset", "Ext_CVC_ColonDB", "dice"),
    "full_pra_dice": ("pranet_external_macro", "dice"),
    "full_pra_bf1": ("pranet_external_macro", "boundary_f1"),
    "full_pra_hd95": ("pranet_external_macro", "hd95"),
    "full_pgext_dice": ("ood_polypgen_ext_macro", "dice"),
    "full_pgext_micro_dice": ("ood_polypgen_ext_micro", "dice"),
    "full_pgext_sd_dice": ("ood_polypgen_ext_seq_disjoint_macro", "dice"),
    "full_pgext_sd_micro_dice": ("ood_polypgen_ext_seq_disjoint_micro", "dice"),
    "full_pgext_dhash8_dice": ("ood_polypgen_ext_dhash8_macro", "dice"),
    "full_pgdev_dice": ("ood_polypgen_macro", "dice"),
    "full_s2_dice": ("s2_tiny_camo_micro", "dice"),
    "full_s2_bf1": ("s2_tiny_camo_micro", "boundary_f1"),
    "full_s3_dice": ("s3_adverse_macro", "dice"),
    "full_s3_bf1": ("s3_adverse_macro", "boundary_f1"),
}
# per-corruption S3 Dice (S1 and Full tracks)
for _c in ["Adverse_NBI_Spectral", "Adverse_LowLight_Lumen", "Adverse_Specular_Glare_Blur", "Adverse_Combined_Severe"]:
    METRICS[f"s1s3_{_c}"] = ("s1_s3_adverse_per_corruption", _c, "dice")
    METRICS[f"full_s3_{_c}"] = ("s3_adverse_per_corruption", _c, "dice")


def dig(d, path):
    for p in path:
        d = d[p]
    return float(d)


def load():
    data = {}
    for arm, runs in ARMS.items():
        per_seed = {}
        for rid in runs:
            for line in open(LOGS / f"{rid}.log"):
                if line.startswith("SEED_SUMMARY_JSON:"):
                    rec = json.loads(line.split(": ", 1)[1])
                    per_seed[rec["seed"]] = {k: dig(rec, p) for k, p in METRICS.items()}
                    per_seed[rec["seed"]]["run"] = rid
        data[arm] = {str(s): per_seed[s] for s in sorted(per_seed)}
    return data


def summarize(data):
    seeds = ["42", "43", "44", "45", "46"]
    print(f"{'metric':22s} " + " ".join(f"{a:>18s}" for a in ["baseline", "safeconcat", "protodpt_passe", "flagship"]))
    for m in METRICS:
        row = []
        for arm in ["baseline", "safeconcat", "protodpt_passe", "flagship"]:
            v = np.array([data[arm][s][m] for s in seeds])
            row.append(f"{v.mean():8.2f} +- {v.std(ddof=1):5.2f}")
        b = np.array([data["baseline"][s][m] for s in seeds])
        f = np.array([data["flagship"][s][m] for s in seeds])
        c = np.array([data["protodpt_passe"][s][m] for s in seeds])
        t_fb = stats.ttest_rel(f, b)
        t_fc = stats.ttest_rel(f, c)
        t_cb = stats.ttest_rel(c, b)
        print(f"{m:22s} " + " ".join(f"{r:>18s}" for r in row)
              + f" | F-B {np.mean(f-b):+6.2f} p={t_fb.pvalue:.3f}"
              + f" | C-B {np.mean(c-b):+6.2f} p={t_cb.pvalue:.3f}"
              + f" | F-C {np.mean(f-c):+6.2f} p={t_fc.pvalue:.3f}")


if __name__ == "__main__":
    d = load()
    OUT.write_text(json.dumps(d, indent=1))
    summarize(d)


TABLE_COLS = ["s1s2_dice", "s1s2_bf1", "s1s2_hd95", "s1s3_dice", "s1s3_bf1", "s1s3_hd95",
              "s1pra_dice", "s1pra_bf1", "s1pra_hd95", "full_etis_dice", "full_pra_dice", "full_pgext_dice"]


def holm(pvals):
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def table_rows(data):
    """Paired differences for the Table 3 columns, with raw and Holm-adjusted p (12 columns per family)."""
    seeds = ["42", "43", "44", "45", "46"]
    for a, b in [("flagship", "baseline"), ("protodpt_passe", "baseline"), ("flagship", "protodpt_passe")]:
        d = [np.array([data[a][s][m] - data[b][s][m] for s in seeds]) for m in TABLE_COLS]
        p = np.array([stats.ttest_1samp(x, 0.0).pvalue for x in d])
        print(f"{a} - {b}:")
        for m, x, pr, ph in zip(TABLE_COLS, d, p, holm(p)):
            print(f"  {m:16s} {x.mean():+7.2f}  p={pr:.4f}  holm={ph:.4f}")


if __name__ == "__main__":
    table_rows(load())
