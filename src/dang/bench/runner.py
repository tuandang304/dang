"""Benchmark runner (cfg.protocol == "bench").

cfg.bench_model == ""         build and audit the cohorts only (no GPU work)
cfg.bench_model in RECIPES    train with the official recipe (dang/bench/recipes.py), fp32, final-epoch weights
cfg.bench_model == "dang"     train the configured DANG model with its own recipe, final EMA weights (no val split)

Seeds are spread over all visible GPUs (one forked worker per GPU, e.g. Kaggle's 2x T4).
"""

import copy
import json
import multiprocessing as mp
import os
import random
import subprocess
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np

from dang.bench.data import lesion_size_breakdown, per_domain_counts, prepare_bench

KEY_METRICS = ["dice_native", "iou_native", "mdice_tb", "miou_tb", "recall_native", "precision_native", "hd95",
               "boundary_f1"]


def _n_gpus() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout
        return sum(1 for line in out.splitlines() if line.startswith("GPU "))
    except Exception:
        return 0


def _out_dir() -> Path:
    p = Path(os.environ.get("ORX_OUTPUT_DIR", "outputs"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _emit_per_image(model_name: str, seed: int, cohort: str, records: list[dict], chunk: int = 120):
    for i in range(0, len(records), chunk):
        recs = [{"u": r["uid"], "dn": round(r["dice_native"], 3), "tb": round(r["mdice_tb"], 3),
                 "in": round(r["iou_native"], 3), "rc": round(r["recall_native"], 3), "hd": round(r["hd95"], 2)}
                for r in records[i:i + chunk]]
        print("BENCH_PER_IMAGE: " + json.dumps({"model": model_name, "seed": seed, "cohort": cohort,
                                                 "chunk": i // chunk, "recs": recs}), flush=True)


def run_one_seed(cfg, seed: int, data: dict, gpu_tag: str) -> dict:
    import torch
    from dang.bench.evaluate import evaluate_cohort, summarize
    from dang.bench.recipes import RECIPES, train_official

    def log(msg):
        print(f"[{gpu_tag}|{cfg.bench_model}|seed {seed}] {msg}", flush=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nw = int(getattr(cfg, "bench_num_workers", 2))
    t0 = time.time()

    if cfg.bench_model in RECIPES:
        recipe = RECIPES[cfg.bench_model]
        model = recipe.build().to(device)
        model = train_official(recipe, model, data["train"], device, log, num_workers=nw)
        logits_fn, amp_dtype = recipe.test_logits, None                     # official code runs fp32
    elif cfg.bench_model == "dang":
        from torch.utils.data import DataLoader
        from dang.data import MedicalSegDataset
        from dang.train import build_model, get_amp_config, train_and_select_best
        amp_dtype, use_scaler = get_amp_config(device)
        model = build_model(cfg).to(device)
        g0 = torch.Generator()
        g0.manual_seed(seed)
        if hasattr(model, "prepare"):   # data-dependent set-up on training images (e.g. v12 DWConv block selection)
            model.prepare(DataLoader(MedicalSegDataset(data["train"], img_size=cfg.img_size, is_train=False),
                                     batch_size=16, shuffle=True, num_workers=nw, generator=g0), device)
        ema = copy.deepcopy(model).to(device)
        ema.eval()
        g = torch.Generator()
        g.manual_seed(seed)
        loader = DataLoader(MedicalSegDataset(data["train"], img_size=cfg.img_size, is_train=True),
                            batch_size=cfg.batch_size, shuffle=True, num_workers=nw, pin_memory=True,
                            drop_last=True, generator=g)
        log(f"DANG recipe: epochs={cfg.bench_dang_epochs} warmup={cfg.warmup_epochs} batch={cfg.batch_size} "
            f"amp={amp_dtype} amp_mix={getattr(cfg, 'amp_mix_prob', 0.0)} final EMA weights")
        model, _, _ = train_and_select_best(model, ema, loader, None, cfg, device, amp_dtype=amp_dtype,
                                            use_scaler=use_scaler, epochs=cfg.bench_dang_epochs,
                                            warmup_epochs=cfg.warmup_epochs, tag=f"{gpu_tag} DANG seed {seed}")
        logits_fn = lambda out: out["logits"]
    else:
        raise ValueError(f"unknown bench_model {cfg.bench_model!r}")
    train_min = (time.time() - t0) / 60.0
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    log(f"trained in {train_min:.1f} min | params={params_m:.2f}M")

    out = _out_dir()
    torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()},
               out / f"weights_{cfg.bench_model}_seed{seed}_fp16.pth")
    dump_zip = None
    if getattr(cfg, "bench_dump_seed", None) == seed:
        dump_zip = zipfile.ZipFile(out / f"pred_masks_{cfg.bench_model}_seed{seed}.zip", "w", zipfile.ZIP_DEFLATED)

    summary = {"model": cfg.bench_model, "variant": cfg.variant_name, "seed": seed, "train_minutes": round(train_min, 2),
               "params_m": round(params_m, 3), "manifest": data["audit"]["manifest_sha256_16"], "cohorts": {}}
    for cohort, samples in data["cohorts"].items():
        te = time.time()
        recs = evaluate_cohort(model, samples, logits_fn, device, amp_dtype=amp_dtype,
                               dump=(dump_zip, cohort) if dump_zip is not None else None,
                               num_workers=nw)
        summ = summarize(recs)
        summary["cohorts"][cohort] = summ
        m = summ["macro"]
        log(f"[{cohort} n={len(recs)}] Dice(native,0.5)={summ['micro']['dice_native']:.2f} "
            f"(macro {m['dice_native']:.2f}) | toolbox mDice={summ['micro']['mdice_tb']:.2f} "
            f"| IoU={summ['micro']['iou_native']:.2f} | recall={summ['micro']['recall_native']:.2f} "
            f"| HD95(352)={summ['micro']['hd95']:.1f}px | {time.time() - te:.0f}s")
        _emit_per_image(cfg.bench_model, seed, cohort, recs)
    if dump_zip is not None:
        dump_zip.close()
        log(f"PRED_MASK_DUMP {dump_zip.filename}")
    if getattr(cfg, "v12_eval_half", False) and cfg.bench_model == "dang" and device.type == "cuda":
        # deployment precision: fp16 weights and input, no autocast (the "half" mode of dang.deploy_bench)
        hm = copy.deepcopy(model).half().eval()
        hm.cfg = copy.deepcopy(hm.cfg)
        hm.cfg.v5_vit_fp32 = False
        half_fn = lambda out: out["logits"]
        class _HalfInput(torch.nn.Module):
            def forward(self, x):
                return hm(x.half())

        wrapped = _HalfInput()
        half = {}
        for cohort in ("CVC-ColonDB", "ETIS-LaribPolypDB", "PolypGen"):
            if cohort in data["cohorts"]:
                with torch.no_grad():
                    hs = summarize(evaluate_cohort(wrapped, data["cohorts"][cohort], half_fn, device, amp_dtype=None,
                                                   num_workers=nw))
                half[cohort] = {"micro_dice": round(hs["micro"]["dice_native"], 2),
                                "macro_dice": round(hs["macro"]["dice_native"], 2),
                                "recall": round(hs["micro"]["recall_native"], 2), "hd95": round(hs["micro"]["hd95"], 1)}
        summary["deploy_half"] = half
        print("DEPLOY_HALF_JSON: " + json.dumps(half), flush=True)
    print(f"BENCH_SEED_SUMMARY_JSON:{seed}: " + json.dumps(summary), flush=True)
    with open(out / f"bench_summary_{cfg.bench_model}_seed{seed}.json", "w") as f:
        json.dump(summary, f)
    return summary


def _worker(gpu: int, seeds: list[int], cfg, data: dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        for seed in seeds:
            run_one_seed(cfg, seed, data, f"gpu{gpu}")
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)


def _aggregate(cfg, summaries: list[dict]):
    agg = {"model": cfg.bench_model, "variant": cfg.variant_name, "seeds": [s["seed"] for s in summaries], "cohorts": {}}
    for cohort in summaries[0]["cohorts"]:
        row = {}
        for level in ("micro", "macro"):
            for k in KEY_METRICS:
                vals = [s["cohorts"][cohort][level][k] for s in summaries if k in s["cohorts"][cohort][level]]
                if vals:
                    row[f"{level}_{k}_mean"] = round(float(np.mean(vals)), 3)
                    row[f"{level}_{k}_sd"] = round(float(np.std(vals, ddof=1)), 3) if len(vals) > 1 else 0.0
        agg["cohorts"][cohort] = row
    print("BENCH_AGG_JSON: " + json.dumps(agg), flush=True)
    print("\n" + "=" * 100)
    print(f"{cfg.bench_model} | seeds {agg['seeds']} | mean (sd) over seeds")
    print(f"{'cohort':<20}{'Dice@0.5 micro':>18}{'Dice@0.5 macro':>18}{'toolbox mDice':>18}{'IoU':>14}{'recall':>14}")
    for c, r in agg["cohorts"].items():
        print(f"{c:<20}{r['micro_dice_native_mean']:>11.2f} ({r['micro_dice_native_sd']:.2f})"
              f"{r['macro_dice_native_mean']:>11.2f} ({r['macro_dice_native_sd']:.2f})"
              f"{r['micro_mdice_tb_mean']:>11.2f} ({r['micro_mdice_tb_sd']:.2f})"
              f"{r['micro_iou_native_mean']:>14.2f}{r['micro_recall_native_mean']:>14.2f}")
    print("=" * 100, flush=True)


def main_bench(cfg):
    seeds = list(cfg.bench_seeds)
    print("=" * 100)
    print(f"SOTA benchmark | Variant: {cfg.variant_name} | model: {cfg.bench_model or '(data audit only)'} | seeds {seeds}")
    print(f"Effective Config: {json.dumps(cfg.to_dict(), sort_keys=True)}")
    print("=" * 100, flush=True)
    if getattr(cfg, "bench_only", False):
        import torch
        from dang.deploy_bench import run_benchmark  # same-host deployment benchmark: inference only, no data
        from dang.train import build_model
        return run_benchmark(cfg, build_model, torch.device("cuda"))

    data = prepare_bench(include_polypdb=cfg.bench_polypdb)
    audit = data["audit"]
    audit["cohort_counts"] = {c: len(v) for c, v in data["cohorts"].items()}
    audit["cohort_domains"] = {c: per_domain_counts(v) for c, v in data["cohorts"].items()}
    print("BENCH_AUDIT_JSON: " + json.dumps(audit), flush=True)
    print(f"BENCH_MANIFEST_SHA256_16: {audit['manifest_sha256_16']}", flush=True)
    if not cfg.bench_model:
        sizes = {c: lesion_size_breakdown(v) for c, v in data["cohorts"].items()}
        print("BENCH_LESION_AREA_PCT_JSON: " + json.dumps(sizes), flush=True)
        with open(_out_dir() / "bench_manifest.json", "w") as f:
            json.dump({"audit": audit, "lesion_area_pct": sizes, "manifest": data["manifest"]}, f)
        print("Data audit finished; no model trained.", flush=True)
        return

    n_gpu = _n_gpus()
    print(f"Visible GPUs: {n_gpu}", flush=True)
    out = _out_dir()
    if n_gpu >= 2 and len(seeds) > 1:
        parts = [seeds[i::n_gpu] for i in range(n_gpu)]
        ctx = mp.get_context("fork")
        procs = [ctx.Process(target=_worker, args=(g, part, cfg, data)) for g, part in enumerate(parts) if part]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        bad = [p.exitcode for p in procs if p.exitcode != 0]
        if bad:
            raise RuntimeError(f"benchmark worker(s) failed with exit codes {bad}")
    else:
        for seed in seeds:
            run_one_seed(cfg, seed, data, "gpu0" if n_gpu else "cpu")

    summaries = [json.load(open(out / f"bench_summary_{cfg.bench_model}_seed{s}.json")) for s in seeds]
    _aggregate(cfg, summaries)
