import copy
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dang.config import get_config
from dang.data import prepare_all_cohorts, build_dataloaders_for_seed, decode_binary_mask_np
from dang.metrics import compute_sample_metrics, aggregate_metrics, compute_binary_dice
from dang.model import DANGModel


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_morphological_boundary_gt(mask: torch.Tensor) -> torch.Tensor:
    """Compute 3-pixel morphological gradient boundary band on GPU tensor [B, 1, H, W]."""
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits_f = logits.float()
    targets_f = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits_f, targets_f)
    probs = torch.sigmoid(logits_f)
    inter = (probs * targets_f).sum(dim=(2, 3))
    union = (probs + targets_f).sum(dim=(2, 3))
    dice_loss = 1.0 - ((2.0 * inter + 1.0) / (union + 1.0)).mean()
    return 0.5 * bce + 0.5 * dice_loss


def asymmetric_tversky_loss(logits: torch.Tensor, targets: torch.Tensor, alpha_fp: float = 0.6, beta_fn: float = 0.4) -> torch.Tensor:
    """Precision-aware Tversky overlap loss (alpha_fp > beta_fn) to suppress false-positive blobs on small polyps."""
    probs = torch.sigmoid(logits.float())
    y = targets.float()
    dims = (1, 2, 3)
    tp = (probs * y).sum(dims)
    fp = (probs * (1.0 - y)).sum(dims)
    fn = ((1.0 - probs) * y).sum(dims)
    tversky = (tp + 1.0) / (tp + alpha_fp * fp + beta_fn * fn + 1.0)
    return (1.0 - tversky).mean()


def compute_total_loss(outputs: dict, masks: torch.Tensor, cfg, epoch: int = 1) -> torch.Tensor:
    logits = outputs["logits"]
    region_loss = bce_dice_loss(logits, masks)
    tversky_loss = asymmetric_tversky_loss(logits, masks, alpha_fp=0.60, beta_fn=0.40)
    loss = region_loss + 0.20 * tversky_loss

    if cfg.use_boundary_loss:
        edge_gt = extract_morphological_boundary_gt(masks)
        edge_loss = bce_dice_loss(outputs["boundary_logits"], edge_gt)
        ramp = min(1.0, max(0.1, (epoch - 1) / 5.0))
        loss = loss + cfg.boundary_loss_weight * ramp * edge_loss

    if cfg.deep_supervision:
        aux_loss = 0.0
        for aux_logit in outputs["aux_logits"]:
            aux_loss = aux_loss + bce_dice_loss(aux_logit, masks)
        loss = loss + 0.25 * (aux_loss / len(outputs["aux_logits"]))
    return loss


def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    cnn_lr = getattr(cfg, "cnn_lr", 1.6e-4)
    new_lr = getattr(cfg, "lr", 4.0e-4)
    peft_lr = getattr(cfg, "peft_lr", 1.0e-3)

    groups = defaultdict(list)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_decay = (
            p.ndim <= 1
            or name.endswith(".bias")
            or "gamma" in name
            or "scale_" in name
            or "reassembly_scale" in name
            or "boundary_alpha" in name
            or "norm" in name
        )
        if name.startswith("cnn."):
            family = "cnn"
            base_lr = cnn_lr
            wd = 0.0 if no_decay else cfg.weight_decay
        elif name.startswith("vit."):
            family = "peft"
            base_lr = peft_lr
            wd = 0.0
        else:
            family = "new"
            base_lr = new_lr
            wd = 0.0 if no_decay else cfg.weight_decay
        groups[(family, base_lr, wd)].append(p)

    param_groups = [
        {"params": params, "lr": base_lr, "base_lr": base_lr, "weight_decay": wd, "family": fam}
        for (fam, base_lr, wd), params in groups.items()
    ]
    return torch.optim.AdamW(param_groups)


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, step: int, max_decay: float = 0.992):
    decay = min(max_decay, (1.0 + step) / (10.0 + step))
    ema_sd = ema_model.state_dict()
    model_sd = model.state_dict()
    for k, v in ema_sd.items():
        mv = model_sd[k]
        if v.dtype.is_floating_point:
            v.mul_(decay).add_(mv.detach(), alpha=1.0 - decay)
        else:
            v.copy_(mv)


def summarize_record_collection(all_records: list[dict]) -> dict:
    if not all_records:
        return {}
    domain_records = defaultdict(list)
    for r in all_records:
        domain_records[r["domain"]].append(r)

    micro = aggregate_metrics(all_records)
    per_domain = {dom: aggregate_metrics(recs) for dom, recs in sorted(domain_records.items())}
    if len(per_domain) > 1:
        keys = [k for k in micro.keys() if k not in ("n", "empty_preds")]
        macro = {k: float(np.mean([per_domain[d][k] for d in per_domain if k in per_domain[d]])) for k in keys}
        macro["n"] = len(all_records)
        macro["num_domains"] = len(per_domain)
        macro["empty_preds"] = int(sum(per_domain[d].get("empty_preds", 0) for d in per_domain))
    else:
        macro = dict(micro)

    small_recs = [r for r in all_records if r.get("gt_area_pct", 100.0) < 3.0]
    med_large_recs = [r for r in all_records if r.get("gt_area_pct", 0.0) >= 3.0]
    return {
        "micro": micro,
        "macro": macro,
        "per_domain": per_domain,
        "num_samples": len(all_records),
        "small_lesions": aggregate_metrics(small_recs) if small_recs else {},
        "small_lesions_count": len(small_recs),
        "med_large_lesions": aggregate_metrics(med_large_recs) if med_large_recs else {},
        "med_large_lesions_count": len(med_large_recs),
    }


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader,
    device: torch.device,
    uid_lookup: dict | None = None,
    full_metrics: bool = False,
) -> dict:
    model.eval()
    all_records = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].numpy()
        domains = batch["domain"]
        names = batch["name"]
        uids = batch["uid"]

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32):
            out = model(images)
            probs_t = torch.sigmoid(out["logits"]).float()
            probs = probs_t.cpu().numpy()

        B = probs.shape[0]
        for i in range(B):
            p_i = probs[i, 0]
            m_i = masks[i, 0]
            if full_metrics:
                uid = uids[i]
                meta = uid_lookup[uid] if uid_lookup and uid in uid_lookup else None
                dice_nat = None
                if meta is not None:
                    gt_nat = decode_binary_mask_np(meta["mask_bytes"])
                    h_nat, w_nat = gt_nat.shape
                    prob_up = F.interpolate(
                        probs_t[i : i + 1],
                        size=(h_nat, w_nat),
                        mode="bilinear",
                        align_corners=False,
                    )
                    pred_nat = (prob_up[0, 0] >= 0.5).cpu().numpy()
                    dice_nat = compute_binary_dice(pred_nat, gt_nat)

                rec = compute_sample_metrics(p_i, m_i, dice_native=dice_nat)
                rec["name"] = names[i]
                rec["domain"] = domains[i]
                rec["uid"] = uid
                if meta is not None:
                    rec["seq_id"] = meta["seq_id"]
                    rec["c4_seq_in_dev"] = bool(meta["c4_seq_in_dev"])
                    rec["min_kvasir_dhash"] = int(meta["min_kvasir_dhash"])
            else:
                pred_bin = p_i >= 0.5
                gt_bin = m_i >= 0.5
                inter = np.logical_and(pred_bin, gt_bin).sum()
                pred_s = pred_bin.sum()
                gt_s = gt_bin.sum()
                union = pred_s + gt_s - inter
                rec = {
                    "dice": float((2.0 * inter + 1e-6) / (pred_s + gt_s + 1e-6) * 100.0),
                    "iou": float((inter + 1e-6) / (union + 1e-6) * 100.0),
                    "domain": domains[i],
                }
            all_records.append(rec)

    if not full_metrics:
        micro = aggregate_metrics(all_records)
        return {"micro": micro, "num_samples": len(all_records)}

    res = summarize_record_collection(all_records)
    res["records"] = all_records
    return res


@torch.no_grad()
def measure_efficiency(model: nn.Module, cfg, device: torch.device) -> dict:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    model.eval()
    dummy = torch.randn(1, 3, cfg.img_size, cfg.img_size, device=device)
    if device.type == "cuda":
        for _ in range(50):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _ = model(dummy)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_iters = 200
        for _ in range(n_iters):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _ = model(dummy)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        fps = n_iters / elapsed
    else:
        fps = 0.0

    return {
        "total_params_m": round(total_params / 1e6, 3),
        "trainable_params_m": round(trainable_params / 1e6, 3),
        "frozen_params_m": round(frozen_params / 1e6, 3),
        "trainable_pct": round(100.0 * trainable_params / max(total_params, 1), 2),
        "fps": round(fps, 2),
        "warmup_iters": 50,
        "timed_iters": 200,
    }


def emit_chunked_records(seed: int, cohort_tag: str, records: list[dict], chunk_size: int = 35):
    """Emit per-image records in compact <=12KB JSON lines so Modal's 64KB line buffer never truncates."""
    total_chunks = max(1, (len(records) + chunk_size - 1) // chunk_size)
    for c_idx in range(total_chunks):
        sub = records[c_idx * chunk_size : (c_idx + 1) * chunk_size]
        compact = [
            {
                "u": r["uid"],
                "d": round(r["dice"], 3),
                "dn": round(r["dice_native"], 3),
                "i": round(r["iou"], 3),
                "bf1": round(r["boundary_f1"], 3),
                "hd": round(r["hd95"], 3),
                "hd_ne": None if np.isnan(r["hd95_non_empty"]) else round(r["hd95_non_empty"], 3),
                "ep": int(r["empty_pred"] > 50.0),
                "a": round(r["gt_area_pct"], 3),
                "sq": r.get("seq_id", ""),
                "c4s": int(r.get("c4_seq_in_dev", False)),
                "kd": int(r.get("min_kvasir_dhash", 99)),
            }
            for r in sub
        ]
        line = json.dumps({"seed": seed, "cohort": cohort_tag, "chunk": c_idx, "total": total_chunks, "recs": compact})
        print(f"PER_IMAGE_CHUNK: {line}", flush=True)


def aggregate_across_seeds(seed_dicts: list[dict]) -> dict:
    """Compute mean, std (ddof=1), and per-seed values across all numeric keys in a list of metric dicts."""
    if not seed_dicts:
        return {}
    out = {}
    keys = [k for k in seed_dicts[0].keys() if isinstance(seed_dicts[0][k], (int, float)) and not np.isnan(seed_dicts[0][k])]
    for k in keys:
        vals = [float(d[k]) for d in seed_dicts if k in d and not np.isnan(d[k])]
        if vals:
            out[f"{k}_mean"] = round(float(np.mean(vals)), 4)
            out[f"{k}_std"] = round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4)
            out[f"{k}_seeds"] = [round(v, 4) for v in vals]
    return out


def main():
    cfg = get_config()
    seeds = list(getattr(cfg, "seeds", (cfg.seed,)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print("=" * 88)
    print(f"DANG-Med v3 Multi-Seed & Expanded-OOD Runner | Variant: {cfg.variant_name}")
    print(f"Seeds: {seeds} | Device: {gpu_name} | Resolution: {cfg.img_size}x{cfg.img_size} | Epochs: {cfg.epochs}")
    print(f"Effective Config: {json.dumps(cfg.to_dict(), sort_keys=True)}")
    print("=" * 88, flush=True)

    t_data = time.time()
    cohorts = prepare_all_cohorts()
    hygiene = cohorts["hygiene_manifest"]
    print(
        f"Prepared All Cohorts in {time.time() - t_data:.1f}s | "
        f"Train(Kvasir)={hygiene['kvasir_train_kept']} | "
        f"Val(Kvasir)={hygiene['kvasir_val_kept']} | "
        f"ID-Test(Kvasir)={hygiene['kvasir_id_test_kept']} | "
        f"PolypGen-Dev={hygiene['polypgen_dev']['kept']} | "
        f"PolypGen-Extended={hygiene['polypgen_extended']['kept']} | "
        f"PraNet-External={hygiene['pranet_external_total_kept']}",
        flush=True,
    )
    print("HYGIENE_MANIFEST_JSON: " + json.dumps(hygiene), flush=True)

    eff = None
    per_seed_summaries = {}
    per_seed_records = {}

    out_dir = Path(os.environ.get("ORX_OUTPUT_DIR", "outputs")) / cfg.variant_name
    out_dir.mkdir(parents=True, exist_ok=True)
    top_out = Path(os.environ.get("ORX_OUTPUT_DIR", "outputs"))

    for s_idx, seed in enumerate(seeds):
        print("\n" + "#" * 88)
        print(f"RUNNING SEED {seed} ({s_idx + 1}/{len(seeds)}) | Variant: {cfg.variant_name}")
        print("#" * 88, flush=True)

        seed_everything(seed)
        loaders = build_dataloaders_for_seed(cohorts, cfg, seed)

        model = DANGModel(cfg).to(device)
        ema_model = copy.deepcopy(model).to(device)
        ema_model.eval()

        if eff is None:
            eff = measure_efficiency(model, cfg, device)
            print(
                f"Model Efficiency (50 warmup + 200 timed iters): Total={eff['total_params_m']:.2f}M | "
                f"Trainable={eff['trainable_params_m']:.2f}M ({eff['trainable_pct']:.2f}%) | "
                f"Frozen={eff['frozen_params_m']:.2f}M | Speed={eff['fps']:.2f} FPS",
                flush=True,
            )

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = build_optimizer(model, cfg)

        total_steps = cfg.epochs * len(loaders["train"])
        warmup_steps = max(1, cfg.warmup_epochs * len(loaders["train"]))
        global_step = 0

        best_val_dice = -1.0
        best_state = None
        best_epoch = 0

        for epoch in range(1, cfg.epochs + 1):
            ep_t0 = time.time()
            model.train()
            if cfg.use_dinov2:
                model.vit.eval()
                for m in model.vit.modules():
                    if hasattr(m, "down") and hasattr(m, "up"):
                        m.train()
                        m.block.eval()

            running_loss = 0.0
            n_batches = 0
            for batch in loaders["train"]:
                if global_step < warmup_steps:
                    lr_scale = float(global_step + 1) / float(warmup_steps)
                else:
                    progress = float(global_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                    lr_scale = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
                for pg in optimizer.param_groups:
                    pg["lr"] = pg["base_lr"] * lr_scale

                images = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32):
                    outputs = model(images)
                    loss = compute_total_loss(outputs, masks, cfg, epoch=epoch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=3.0)
                optimizer.step()

                update_ema(ema_model, model, global_step)
                global_step += 1

                running_loss += float(loss.item())
                n_batches += 1

            train_loss = running_loss / max(1, n_batches)
            val_raw = evaluate_loader(model, loaders["val"], device, full_metrics=False)
            val_ema = evaluate_loader(ema_model, loaders["val"], device, full_metrics=False)

            if val_ema["micro"]["dice"] >= val_raw["micro"]["dice"]:
                val_dice = val_ema["micro"]["dice"]
                val_iou = val_ema["micro"]["iou"]
                cand_model = ema_model
                src_tag = "EMA"
            else:
                val_dice = val_raw["micro"]["dice"]
                val_iou = val_raw["micro"]["iou"]
                cand_model = model
                src_tag = "RAW"

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in cand_model.state_dict().items()}

            ep_time = time.time() - ep_t0
            cur_new_lr = max(pg["lr"] for pg in optimizer.param_groups)
            print(
                f"[Seed {seed} | Ep {epoch:02d}/{cfg.epochs:02d}] "
                f"loss={train_loss:.4f} | val_dice={val_dice:.2f}% ({src_tag}) | val_iou={val_iou:.2f}% "
                f"| lr={cur_new_lr:.2e} | time={ep_time:.1f}s",
                flush=True,
            )

        if best_state is not None:
            model.load_state_dict(best_state)

        uid_lookup = cohorts["uid_lookup"]
        val_final = evaluate_loader(model, loaders["val"], device, uid_lookup=uid_lookup, full_metrics=True)
        id_final = evaluate_loader(model, loaders["id_test"], device, uid_lookup=uid_lookup, full_metrics=True)
        ood_dev_final = evaluate_loader(model, loaders["ood_dev"], device, uid_lookup=uid_lookup, full_metrics=True)
        ood_ext_final = evaluate_loader(model, loaders["ood_ext"], device, uid_lookup=uid_lookup, full_metrics=True)
        pranet_final = evaluate_loader(model, loaders["pranet_ext"], device, uid_lookup=uid_lookup, full_metrics=True)

        # Also compute Sequence-Disjoint PolypGen-Extended and dHash>8 sensitivity subsets
        ext_seq_disjoint_recs = [r for r in ood_ext_final["records"] if not r.get("c4_seq_in_dev", False)]
        ext_dhash8_recs = [r for r in ood_ext_final["records"] if r.get("min_kvasir_dhash", 99) > 8]
        ood_ext_seq_disjoint = summarize_record_collection(ext_seq_disjoint_recs)
        ood_ext_dhash8 = summarize_record_collection(ext_dhash8_recs)

        print(f"\n--- SEED {seed} SUMMARY (Best Val Epoch {best_epoch}) ---")
        print(
            f"  Kvasir-Val (n={val_final['num_samples']}): "
            f"Dice={val_final['micro']['dice']:.2f}% (Nat={val_final['micro']['dice_native']:.2f}%) | "
            f"BF1={val_final['micro']['boundary_f1']:.2f}% | HD95={val_final['micro']['hd95']:.2f}px"
        )
        print(
            f"  Kvasir-Test (n={id_final['num_samples']}): "
            f"Dice={id_final['micro']['dice']:.2f}% (Nat={id_final['micro']['dice_native']:.2f}%) | "
            f"BF1={id_final['micro']['boundary_f1']:.2f}% | HD95={id_final['micro']['hd95']:.2f}px"
        )
        print(
            f"  PolypGen-Dev Macro (6 centers, n={ood_dev_final['num_samples']}): "
            f"Dice={ood_dev_final['macro']['dice']:.2f}% (Nat={ood_dev_final['macro']['dice_native']:.2f}%) | "
            f"BF1={ood_dev_final['macro']['boundary_f1']:.2f}% | HD95={ood_dev_final['macro']['hd95']:.2f}px "
            f"(non-empty={ood_dev_final['macro']['hd95_non_empty']:.2f}px, empty={ood_dev_final['macro']['empty_preds']}) | "
            f"Small(n={ood_dev_final['small_lesions_count']})={ood_dev_final['small_lesions']['dice']:.2f}%"
        )
        print(
            f"  PolypGen-Extended Holdout Macro (6 centers, n={ood_ext_final['num_samples']}): "
            f"Dice={ood_ext_final['macro']['dice']:.2f}% (Nat={ood_ext_final['macro']['dice_native']:.2f}%) | "
            f"BF1={ood_ext_final['macro']['boundary_f1']:.2f}% | HD95={ood_ext_final['macro']['hd95']:.2f}px "
            f"(non-empty={ood_ext_final['macro']['hd95_non_empty']:.2f}px, empty={ood_ext_final['macro']['empty_preds']}) | "
            f"Small(n={ood_ext_final['small_lesions_count']})={ood_ext_final['small_lesions']['dice']:.2f}%"
        )
        print(
            f"  PolypGen-Extended Seq-Disjoint Macro (n={ood_ext_seq_disjoint['num_samples']}): "
            f"Dice={ood_ext_seq_disjoint['macro']['dice']:.2f}% | BF1={ood_ext_seq_disjoint['macro']['boundary_f1']:.2f}%"
        )
        print(
            f"  PraNet External Macro (4 benchmarks, n={pranet_final['num_samples']}): "
            f"Dice={pranet_final['macro']['dice']:.2f}% (Nat={pranet_final['macro']['dice_native']:.2f}%) | "
            f"BF1={pranet_final['macro']['boundary_f1']:.2f}% | HD95={pranet_final['macro']['hd95']:.2f}px | "
            + " | ".join(f"{k}: {v['dice']:.2f}%" for k, v in pranet_final["per_domain"].items()),
            flush=True,
        )

        seed_summary = {
            "seed": seed,
            "best_epoch": best_epoch,
            "in_domain_kvasir_val": val_final["micro"],
            "in_domain_kvasir": id_final["micro"],
            "ood_polypgen_macro": ood_dev_final["macro"],
            "ood_polypgen_micro": ood_dev_final["micro"],
            "ood_small_lesions": ood_dev_final["small_lesions"],
            "ood_small_lesions_count": ood_dev_final["small_lesions_count"],
            "ood_med_large_lesions": ood_dev_final["med_large_lesions"],
            "ood_med_large_lesions_count": ood_dev_final["med_large_lesions_count"],
            "ood_per_center": ood_dev_final["per_domain"],
            "ood_polypgen_ext_macro": ood_ext_final["macro"],
            "ood_polypgen_ext_micro": ood_ext_final["micro"],
            "ood_polypgen_ext_small_lesions": ood_ext_final["small_lesions"],
            "ood_polypgen_ext_small_lesions_count": ood_ext_final["small_lesions_count"],
            "ood_polypgen_ext_med_large_lesions": ood_ext_final["med_large_lesions"],
            "ood_polypgen_ext_med_large_lesions_count": ood_ext_final["med_large_lesions_count"],
            "ood_polypgen_ext_per_center": ood_ext_final["per_domain"],
            "ood_polypgen_ext_seq_disjoint_macro": ood_ext_seq_disjoint["macro"],
            "ood_polypgen_ext_seq_disjoint_micro": ood_ext_seq_disjoint["micro"],
            "ood_polypgen_ext_dhash8_macro": ood_ext_dhash8["macro"],
            "pranet_external_macro": pranet_final["macro"],
            "pranet_external_micro": pranet_final["micro"],
            "pranet_external_small_lesions": pranet_final["small_lesions"],
            "pranet_external_small_lesions_count": pranet_final["small_lesions_count"],
            "pranet_external_per_dataset": pranet_final["per_domain"],
        }
        per_seed_summaries[str(seed)] = seed_summary
        print(f"SEED_SUMMARY_JSON:{seed}: " + json.dumps(seed_summary), flush=True)

        # Emit chunked per-image records (<12KB per line, never truncated on Modal or Colab)
        emit_chunked_records(seed, "val", val_final["records"])
        emit_chunked_records(seed, "id_test", id_final["records"])
        emit_chunked_records(seed, "ood_dev", ood_dev_final["records"])
        emit_chunked_records(seed, "ood_ext", ood_ext_final["records"])
        emit_chunked_records(seed, "pranet_ext", pranet_final["records"])

        seed_rec_payload = {
            "val": val_final["records"],
            "id_test": id_final["records"],
            "ood_dev": ood_dev_final["records"],
            "ood_ext": ood_ext_final["records"],
            "pranet_ext": pranet_final["records"],
        }
        per_seed_records[str(seed)] = seed_rec_payload
        with open(out_dir / f"per_image_seed_{seed}.json", "w") as f:
            json.dump(seed_rec_payload, f)

        del model, ema_model, optimizer, best_state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Aggregate across all seeds
    seed_list = [per_seed_summaries[str(s)] for s in seeds]
    top_keys = [
        "in_domain_kvasir_val",
        "in_domain_kvasir",
        "ood_polypgen_macro",
        "ood_polypgen_micro",
        "ood_small_lesions",
        "ood_med_large_lesions",
        "ood_polypgen_ext_macro",
        "ood_polypgen_ext_micro",
        "ood_polypgen_ext_small_lesions",
        "ood_polypgen_ext_med_large_lesions",
        "ood_polypgen_ext_seq_disjoint_macro",
        "ood_polypgen_ext_seq_disjoint_micro",
        "ood_polypgen_ext_dhash8_macro",
        "pranet_external_macro",
        "pranet_external_micro",
        "pranet_external_small_lesions",
    ]
    multi_seed_agg = {tk: aggregate_across_seeds([sd[tk] for sd in seed_list]) for tk in top_keys}
    multi_seed_agg["ood_per_center"] = {
        c: aggregate_across_seeds([sd["ood_per_center"][c] for sd in seed_list])
        for c in seed_list[0]["ood_per_center"]
    }
    multi_seed_agg["ood_polypgen_ext_per_center"] = {
        c: aggregate_across_seeds([sd["ood_polypgen_ext_per_center"][c] for sd in seed_list])
        for c in seed_list[0]["ood_polypgen_ext_per_center"]
    }
    multi_seed_agg["pranet_external_per_dataset"] = {
        d: aggregate_across_seeds([sd["pranet_external_per_dataset"][d] for sd in seed_list])
        for d in seed_list[0]["pranet_external_per_dataset"]
    }

    final_payload = {
        "variant": cfg.variant_name,
        "config": cfg.to_dict(),
        "gpu": gpu_name,
        "seeds": seeds,
        "efficiency": eff,
        "dataset_hygiene": hygiene,
        "multi_seed_aggregate": multi_seed_agg,
        "per_seed": per_seed_summaries,
        # Keep top-level seed-42 fields for backward compatibility with single-seed parsers
        **per_seed_summaries[str(seeds[0])],
    }

    print("\n" + "=" * 88)
    print(f"MULTI-SEED ({len(seeds)} SEEDS: {seeds}) FINAL SUMMARY | Variant: {cfg.variant_name}")
    print("-" * 88)
    for label, key in [
        ("Kvasir-Val (n=95)", "in_domain_kvasir_val"),
        ("Kvasir-Test (n=90)", "in_domain_kvasir"),
        ("PolypGen-Dev 6-Center Macro (n=272)", "ood_polypgen_macro"),
        ("PolypGen-Dev Small Polyps (n=82)", "ood_small_lesions"),
        ("PolypGen-Extended Holdout Macro (6-Center)", "ood_polypgen_ext_macro"),
        ("PolypGen-Extended Small Polyps", "ood_polypgen_ext_small_lesions"),
        ("PolypGen-Extended Seq-Disjoint Macro", "ood_polypgen_ext_seq_disjoint_macro"),
        ("PraNet External 4-Dataset Macro", "pranet_external_macro"),
    ]:
        m = multi_seed_agg[key]
        print(
            f"  {label:42s}: Dice={m['dice_mean']:.2f}±{m['dice_std']:.2f}% "
            f"(Nat={m['dice_native_mean']:.2f}±{m['dice_native_std']:.2f}%) | "
            f"BF1={m['boundary_f1_mean']:.2f}±{m['boundary_f1_std']:.2f}% | "
            f"HD95={m['hd95_mean']:.2f}±{m['hd95_std']:.2f}px"
        )
    for ds_name, m in multi_seed_agg["pranet_external_per_dataset"].items():
        print(
            f"    -> {ds_name:37s}: Dice={m['dice_mean']:.2f}±{m['dice_std']:.2f}% "
            f"(Nat={m['dice_native_mean']:.2f}±{m['dice_native_std']:.2f}%) | "
            f"BF1={m['boundary_f1_mean']:.2f}±{m['boundary_f1_std']:.2f}%"
        )

    print("METRICS_JSON_SINGLE_LINE: " + json.dumps({
        "variant": cfg.variant_name,
        "seeds": seeds,
        "efficiency": eff,
        "dataset_hygiene": hygiene,
        "multi_seed_aggregate": multi_seed_agg,
    }), flush=True)
    print("=" * 88, flush=True)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(final_payload, f, indent=2)
    with open(top_out / "summary.json", "w") as f:
        json.dump(final_payload, f, indent=2)


if __name__ == "__main__":
    main()
