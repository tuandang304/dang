"""Literature-comparable protocol: the standard PraNet split (Fan et al., MICCAI 2020).

Train on TrainDataset (1,450 = 900 Kvasir-SEG + 550 CVC-ClinicDB); test on the five TestDataset folders
(Kvasir 100, CVC-ClinicDB 62, CVC-ColonDB 380, ETIS-LaribPolypDB 196, CVC-300 60) exactly as published
work does, without deduplication. There is no validation split, so the final EMA checkpoint is used;
no checkpoint is selected on test data. Dice is reported per dataset as the mean over images, both at
352x352 ("dice") and at native ground-truth resolution ("dice_native", the convention of published tables).
"""

import copy
import json
import os
import time
import zipfile
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader

from dang.data import MedicalSegDataset, decode_binary_mask_np
from dang.train import (
    build_model,
    emit_chunked_records,
    evaluate_loader,
    get_amp_config,
    measure_efficiency,
    seed_everything,
    train_and_select_best,
)

TEST_SETS = ["Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "CVC-300"]
IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".bmp")


def _read_pairs(zf: zipfile.ZipFile, img_dir: str, mask_dir: str, domain: str) -> list[dict]:
    names = set(zf.namelist())
    files = sorted(
        n[len(img_dir):] for n in names
        if n.startswith(img_dir) and n.lower().endswith(IMG_EXT) and "/" not in n[len(img_dir):]
    )
    out = []
    for fn in files:
        mask_entry = mask_dir + fn
        assert mask_entry in names, f"missing mask {mask_entry}"
        img_b, mask_b = zf.read(img_dir + fn), zf.read(mask_entry)
        out.append({
            "img_bytes": img_b,
            "mask_bytes": mask_b,
            "native_gt_bin": decode_binary_mask_np(mask_b),
            "name": fn,
            "domain": domain,
            "uid": f"{domain}::{fn}",
            "seq_id": f"{domain}_{fn}",
            "c4_seq_in_dev": False,
            "min_kvasir_dhash": 99,
        })
    return out


def prepare_standard() -> dict:
    train_zip = zipfile.ZipFile(hf_hub_download("Zyna1121/lightpranet-polyp-data", "TrainDataset.zip", repo_type="dataset"))
    test_zip = zipfile.ZipFile(hf_hub_download("Zyna1121/lightpranet-polyp-data", "TestDataset.zip", repo_type="dataset"))
    train = _read_pairs(train_zip, "TrainDataset/image/", "TrainDataset/masks/", "Train_PraNet1450")
    tests = {d: _read_pairs(test_zip, f"TestDataset/{d}/images/", f"TestDataset/{d}/masks/", d) for d in TEST_SETS}
    counts = {"train": len(train), **{d: len(v) for d, v in tests.items()}}
    assert counts == {"train": 1450, "Kvasir": 100, "CVC-ClinicDB": 62, "CVC-ColonDB": 380,
                      "ETIS-LaribPolypDB": 196, "CVC-300": 60}, counts
    uid_lookup = {s["uid"]: s for v in tests.values() for s in v}
    return {"train": train, "tests": tests, "uid_lookup": uid_lookup, "counts": counts}


def main_standard(cfg):
    seeds = list(getattr(cfg, "seeds", (cfg.seed,)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    amp_dtype, use_scaler = get_amp_config(device)
    epochs = getattr(cfg, "std_epochs", 50)

    print("=" * 88)
    print(f"Standard PraNet-split protocol | Variant: {cfg.variant_name} | Seeds: {seeds} | Device: {gpu_name}")
    print(f"Effective Config: {json.dumps(cfg.to_dict(), sort_keys=True)}")
    print("=" * 88, flush=True)

    data = prepare_standard()
    print("STD_DATA_COUNTS_JSON: " + json.dumps(data["counts"]), flush=True)
    top_out = Path(os.environ.get("ORX_OUTPUT_DIR", "outputs"))
    top_out.mkdir(parents=True, exist_ok=True)

    def eval_loader(samples):
        return DataLoader(MedicalSegDataset(samples, img_size=cfg.img_size, is_train=False),
                          batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    test_loaders = {d: eval_loader(v) for d, v in data["tests"].items()}
    eff, per_seed = None, {}
    for seed in seeds:
        print(f"\n{'#' * 88}\nRUNNING SEED {seed} | standard protocol, {epochs} epochs\n{'#' * 88}", flush=True)
        seed_everything(seed)
        g = torch.Generator()
        g.manual_seed(seed)
        train_loader = DataLoader(MedicalSegDataset(data["train"], img_size=cfg.img_size, is_train=True),
                                  batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                                  pin_memory=True, drop_last=True, generator=g)
        model = build_model(cfg).to(device)
        ema = copy.deepcopy(model).to(device)
        ema.eval()
        if eff is None:
            eff = measure_efficiency(model, cfg, device, amp_dtype=amp_dtype)
            print(f"Model Efficiency: Total={eff['total_params_m']:.2f}M | Trainable={eff['trainable_params_m']:.2f}M | "
                  f"Speed={eff['fps']:.2f} FPS", flush=True)
        t0 = time.time()
        model, _, _ = train_and_select_best(model, ema, train_loader, None, cfg, device, amp_dtype=amp_dtype,
                                            use_scaler=use_scaler, epochs=epochs,
                                            warmup_epochs=getattr(cfg, "warmup_epochs", 2), tag=f"STD Seed {seed}")
        print(f"Trained seed {seed} in {(time.time() - t0) / 60:.1f} min", flush=True)

        dump_zip = None
        if getattr(cfg, "dump_pred_seed", None) == seed:
            dump_path = top_out / f"std_pred_masks_seed{seed}.zip"
            dump_zip = zipfile.ZipFile(dump_path, "w", compression=zipfile.ZIP_DEFLATED)
        summary = {"seed": seed, "epochs": epochs}
        for d, loader in test_loaders.items():
            res = evaluate_loader(model, loader, device, amp_dtype=amp_dtype, uid_lookup=data["uid_lookup"],
                                  full_metrics=True, dump=(dump_zip, f"std/{d}") if dump_zip else None)
            summary[d] = res["micro"]
            emit_chunked_records(seed, f"std_{d}", res["records"])
            m = res["micro"]
            print(f"  [STD {d} (n={res['num_samples']})] mDice(native)={m['dice_native']:.2f}% | mDice(352)={m['dice']:.2f}% | "
                  f"mIoU={m['iou']:.2f}% | BF1={m['boundary_f1']:.2f}% | HD95={m['hd95']:.2f}px", flush=True)
        if dump_zip is not None:
            print(f"PRED_MASK_DUMP: seed={seed} files={len(dump_zip.namelist())} path={dump_path}", flush=True)
            dump_zip.close()
        per_seed[str(seed)] = summary
        print(f"STD_SEED_SUMMARY_JSON:{seed}: " + json.dumps(summary), flush=True)
        del model, ema
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("STD_METRICS_JSON: " + json.dumps({"variant": cfg.variant_name, "gpu": gpu_name, "efficiency": eff,
                                             "seeds": seeds, "per_seed": per_seed}), flush=True)
    with open(top_out / "std_summary.json", "w") as f:
        json.dump({"variant": cfg.variant_name, "config": cfg.to_dict(), "per_seed": per_seed, "efficiency": eff}, f, indent=2)
