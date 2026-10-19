"""Aggregate the retrained SOTA baselines (Round 11) and the standard-split runs (Round 12).

Round 11: PraNet, Polyp-PVT, EMCAD, PraNet-V2 (official losses) and Polyp-PVT with the DANG shared loss, trained
under the DANG protocol (seeds 42-46, same splits, same evaluation). Seed s of every arm sees the same
S1 subset and data order, so differences to DANG-Med are paired by seed.
Round 12: DANG-Med and Polyp-PVT on the PraNet training split (1450 images), seeds 42-44.

Writes figs/sota_metrics.json and prints the Table rows (mean +- SD; paired t vs DANG-Med with Holm over
the 12 table columns).
Regenerate: uv run --no-project --with numpy --with scipy python figs/aggregate_sota.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from aggregate_seeds import LOGS, METRICS, TABLE_COLS, dig, holm  # noqa: E402

OUT = HERE / "sota_metrics.json"
SEEDS = ["42", "43", "44", "45", "46"]
R11 = {
    "pranet": "e00b4190-193d-4a54-8af5-f11727d5866c",
    "polyp_pvt": "6993f62a-9521-483e-99c1-708d599557ad",
    "emcad": "90a8f6ab-b3e7-4f94-b4d4-c79162b6f63c",
    "polyp_pvt_shared": "d8a6af92-29aa-4203-b7af-e695430bd9fc",
    "pranet_v2": "2faf0859-97c3-4412-8d33-6fd012f362c9",
}
R12 = {
    "dang_std": "665158b8-3242-4b57-bde4-b13741106f26",
    "polyp_pvt_std": "02792b5e-9354-40e1-a05d-37b260e1950b",
}
STD_SETS = ["Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "CVC-300"]


def lines(rid, prefix):
    path = LOGS / f"{rid}.log"
    if not path.exists():
        return []
    return [json.loads(ln.split(": ", 1)[1]) for ln in open(path, errors="ignore") if ln.startswith(prefix)]


def efficiency(rid):
    recs = lines(rid, "METRICS_JSON_SINGLE_LINE:")
    return recs[-1]["efficiency"] if recs else None


def load():
    out = {"r11": {}, "r11_eff": {}, "r12": {}, "r12_eff": {}}
    for arm, rid in R11.items():
        per = {str(r["seed"]): {k: dig(r, p) for k, p in METRICS.items()} for r in lines(rid, "SEED_SUMMARY_JSON:")}
        if per:
            out["r11"][arm] = per
            out["r11_eff"][arm] = efficiency(rid)
    for arm, rid in R12.items():
        per = {}
        for r in lines(rid, "STD_SEED_SUMMARY_JSON:"):
            seed = str(r.pop("seed")) if "seed" in r else None
            per[seed] = {d: {k: r[d][k] for k in ("dice", "dice_native", "iou", "boundary_f1", "hd95")} for d in STD_SETS}
        if per:
            out["r12"][arm] = per
            out["r12_eff"][arm] = efficiency(rid)
    return out


def report(data):
    dang = json.loads((HERE / "seed_metrics.json").read_text())["flagship"]
    print("== Round 11 (DANG protocol, 5 seeds); paired vs DANG-Med flagship")
    for arm, per in data["r11"].items():
        seeds = [s for s in SEEDS if s in per]
        print(f"-- {arm} (seeds {','.join(seeds)}) eff={data['r11_eff'][arm]}")
        d = {m: np.array([per[s][m] - dang[s][m] for s in seeds]) for m in TABLE_COLS}
        p = np.array([stats.ttest_1samp(d[m], 0.0).pvalue if len(seeds) > 1 else np.nan for m in TABLE_COLS])
        ph = holm(p) if len(seeds) > 1 else p
        for m, pr, pa in zip(TABLE_COLS, p, ph):
            v = np.array([per[s][m] for s in seeds])
            print(f"  {m:16s} {v.mean():7.2f} +- {v.std(ddof=1) if len(v) > 1 else 0:5.2f}"
                  f"   DANG-arm {d[m].mean():+7.2f}  p={pr:.4f} holm={pa:.4f}")
    print("== Round 12 (PraNet split, mDice at native resolution)")
    for arm, per in data["r12"].items():
        cells = []
        for ds in STD_SETS:
            v = np.array([per[s][ds]["dice_native"] for s in per])
            cells.append(f"{ds}={v.mean():.2f}+-{v.std(ddof=1) if len(v) > 1 else 0:.2f}")
        print(f"  {arm} seeds={list(per)}: " + " | ".join(cells))


if __name__ == "__main__":
    d = load()
    OUT.write_text(json.dumps(d, indent=1))
    report(d)
