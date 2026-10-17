import json
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
from dang.data import build_dataloaders
from dang.metrics import compute_sample_metrics, aggregate_metrics
from dang.model import DANGModel


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_morphological_boundary_gt(mask: torch.Tensor) -> torch.Tensor:
    """Compute morphological gradient boundary band on GPU tensor [B, 1, H, W]."""
    dilated = F.max_pool2d(mask, kernel_size=5, stride=1, padding=2)
    eroded = -F.max_pool2d(-mask, kernel_size=5, stride=1, padding=2)
    return (dilated - eroded).clamp(0.0, 1.0)


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor, weight_map: torch.Tensor | None = None) -> torch.Tensor:
    if weight_map is not None:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        wbce = (bce * weight_map).sum(dim=(2, 3)) / weight_map.sum(dim=(2, 3)).clamp(min=1e-5)
        wbce = wbce.mean()
    else:
        wbce = F.binary_cross_entropy_with_logits(logits, targets)

    probs = torch.sigmoid(logits)
    if weight_map is not None:
        inter = (probs * targets * weight_map).sum(dim=(2, 3))
        union = ((probs + targets) * weight_map).sum(dim=(2, 3))
    else:
        inter = (probs * targets).sum(dim=(2, 3))
        union = (probs + targets).sum(dim=(2, 3))
    dice_loss = 1.0 - ((2.0 * inter + 1.0) / (union + 1.0)).mean()
    return 0.5 * wbce + 0.5 * dice_loss


def compute_total_loss(outputs: dict, masks: torch.Tensor, cfg) -> torch.Tensor:
    logits = outputs["logits"]
    if cfg.use_boundary_loss:
        edge_gt = extract_morphological_boundary_gt(masks)
        weight_map = 1.0 + 4.0 * F.avg_pool2d(edge_gt, kernel_size=15, stride=1, padding=7)
        main_loss = bce_dice_loss(logits, masks, weight_map=weight_map)
        edge_loss = bce_dice_loss(outputs["boundary_logits"], edge_gt)
        loss = main_loss + cfg.boundary_loss_weight * edge_loss
    else:
        weight_map = None
        loss = bce_dice_loss(logits, masks, weight_map=None)

    if cfg.deep_supervision:
        aux_loss = 0.0
        for aux_logit in outputs["aux_logits"]:
            aux_loss = aux_loss + bce_dice_loss(aux_logit, masks, weight_map=weight_map)
        loss = loss + 0.3 * (aux_loss / len(outputs["aux_logits"]))
    return loss


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader, device: torch.device, full_metrics: bool = False) -> dict:
    model.eval()
    all_records = []
    domain_records = defaultdict(list)

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].numpy()
        domains = batch["domain"]
        names = batch["name"]

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32):
            out = model(images)
            probs = torch.sigmoid(out["logits"]).float().cpu().numpy()

        B = probs.shape[0]
        for i in range(B):
            p_i = probs[i, 0]
            m_i = masks[i, 0]
            if full_metrics:
                rec = compute_sample_metrics(p_i, m_i)
                rec["name"] = names[i]
                rec["domain"] = domains[i]
            else:
                pred_bin = (p_i >= 0.5)
                gt_bin = (m_i >= 0.5)
                inter = np.logical_and(pred_bin, gt_bin).sum()
                pred_s = pred_bin.sum()
                gt_s = gt_bin.sum()
                union = pred_s + gt_s - inter
                rec = {
                    "dice": float((2.0 * inter + 1e-6) / (pred_s + gt_s + 1e-6) * 100.0),
                    "iou": float((inter + 1e-6) / (union + 1e-6) * 100.0),
                }
            all_records.append(rec)
            domain_records[domains[i]].append(rec)

    micro = aggregate_metrics(all_records)
    per_domain = {dom: aggregate_metrics(recs) for dom, recs in sorted(domain_records.items())}
    if len(per_domain) > 1:
        keys = list(micro.keys())
        macro = {k: float(np.mean([per_domain[d][k] for d in per_domain])) for k in keys}
    else:
        macro = dict(micro)

    res = {
        "micro": micro,
        "macro": macro,
        "per_domain": per_domain,
        "num_samples": len(all_records),
    }
    if full_metrics:
        small_recs = [r for r in all_records if r.get("gt_area_pct", 100.0) < 3.0]
        med_large_recs = [r for r in all_records if r.get("gt_area_pct", 0.0) >= 3.0]
        res["small_lesions"] = aggregate_metrics(small_recs) if small_recs else {}
        res["small_lesions_count"] = len(small_recs)
        res["med_large_lesions"] = aggregate_metrics(med_large_recs) if med_large_recs else {}
        res["med_large_lesions_count"] = len(med_large_recs)
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
        for _ in range(10):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _ = model(dummy)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_iters = 40
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
    }


def main():
    cfg = get_config()
    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print("=" * 80)
    print(f"DANG-Med Clean Protocol Runner | Variant: {cfg.variant_name} | Seed: {cfg.seed}")
    print(f"Device: {gpu_name} | Resolution: {cfg.img_size}x{cfg.img_size} | Epochs: {cfg.epochs}")
    print(f"Effective Config: {json.dumps(cfg.to_dict(), sort_keys=True)}")
    print("=" * 80, flush=True)

    t_start = time.time()
    train_loader, val_loader, id_test_loader, ood_loader = build_dataloaders(cfg)
    print(
        f"Loaded Clean Datasets in {time.time() - t_start:.1f}s | "
        f"Train(Kvasir)={len(train_loader.dataset)} | "
        f"Val(Kvasir-ckpt)={len(val_loader.dataset)} | "
        f"Test(Kvasir-heldout)={len(id_test_loader.dataset)} | "
        f"OOD(PolypGen-6Centers-Clean)={len(ood_loader.dataset)}",
        flush=True,
    )

    model = DANGModel(cfg).to(device)
    eff = measure_efficiency(model, cfg, device)
    print(
        f"Model Efficiency: Total={eff['total_params_m']:.2f}M | "
        f"Trainable={eff['trainable_params_m']:.2f}M ({eff['trainable_pct']:.2f}%) | "
        f"Frozen={eff['frozen_params_m']:.2f}M | Speed={eff['fps']:.1f} FPS",
        flush=True,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_dice = -1.0
    best_state = None
    best_epoch = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        ep_t0 = time.time()
        if epoch <= cfg.warmup_epochs:
            lr_scale = epoch / max(1, cfg.warmup_epochs)
        else:
            progress = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
            lr_scale = 0.05 + 0.95 * 0.5 * (1.0 + np.cos(np.pi * progress))
        cur_lr = cfg.lr * lr_scale
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        model.train()
        if cfg.use_dinov2:
            model.vit.eval()
            for m in model.vit.modules():
                if hasattr(m, "down") and hasattr(m, "up"):
                    m.train()
                    m.block.eval()

        running_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32):
                outputs = model(images)
                loss = compute_total_loss(outputs, masks, cfg)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        train_loss = running_loss / max(1, n_batches)
        val_res = evaluate_loader(model, val_loader, device, full_metrics=False)
        val_dice = val_res["micro"]["dice"]
        val_iou = val_res["micro"]["iou"]

        if epoch % 5 == 0 or epoch == cfg.epochs:
            ood_quick = evaluate_loader(model, ood_loader, device, full_metrics=False)
            ood_macro_dice = ood_quick["macro"]["dice"]
            ood_micro_dice = ood_quick["micro"]["dice"]
        else:
            ood_macro_dice = None
            ood_micro_dice = None

        # Strictly select checkpoint on held-out Kvasir-Val (never on Test or OOD)
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        ep_time = time.time() - ep_t0
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_dice": round(val_dice, 2),
            "val_iou": round(val_iou, 2),
            "ood_macro_dice": round(ood_macro_dice, 2) if ood_macro_dice is not None else None,
            "ood_micro_dice": round(ood_micro_dice, 2) if ood_micro_dice is not None else None,
            "lr": round(cur_lr, 6),
            "time_s": round(ep_time, 2),
        }
        history.append(row)

        ood_str = f" | ood_macro_dice={ood_macro_dice:.2f}%" if ood_macro_dice is not None else ""
        print(
            f"[Epoch {epoch:02d}/{cfg.epochs:02d}] "
            f"loss={train_loss:.4f} | val_dice={val_dice:.2f}% | val_iou={val_iou:.2f}%"
            f"{ood_str} | lr={cur_lr:.2e} | time={ep_time:.1f}s",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val = evaluate_loader(model, val_loader, device, full_metrics=True)
    final_test = evaluate_loader(model, id_test_loader, device, full_metrics=True)
    final_ood = evaluate_loader(model, ood_loader, device, full_metrics=True)

    summary = {
        "variant": cfg.variant_name,
        "config": cfg.to_dict(),
        "gpu": gpu_name,
        "best_epoch": best_epoch,
        "efficiency": eff,
        "in_domain_kvasir_val": final_val["micro"],
        "in_domain_kvasir": final_test["micro"],
        "ood_polypgen_macro": final_ood["macro"],
        "ood_polypgen_micro": final_ood["micro"],
        "ood_small_lesions": final_ood["small_lesions"],
        "ood_small_lesions_count": final_ood["small_lesions_count"],
        "ood_med_large_lesions": final_ood["med_large_lesions"],
        "ood_med_large_lesions_count": final_ood["med_large_lesions_count"],
        "ood_per_center": final_ood["per_domain"],
        "history": history,
    }

    out_dir = os.environ.get("ORX_OUTPUT_DIR", "outputs")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(Path(out_dir) / "per_image_metrics.json", "w") as f:
        json.dump({
            "in_domain_test": final_test["records"],
            "ood_polypgen": final_ood["records"],
        }, f, indent=2)

    print("\n" + "=" * 80)
    print(f"FINAL SUMMARY | Variant: {cfg.variant_name} (Best Val Epoch {best_epoch})")
    print("-" * 80)
    val_m = final_val["micro"]
    in_m = final_test["micro"]
    ood_mac = final_ood["macro"]
    ood_mic = final_ood["micro"]
    ood_sm = final_ood["small_lesions"]
    print(
        f"In-Domain Val (Kvasir-Val, n=100): "
        f"mDice={val_m['dice']:.2f}% | mIoU={val_m['iou']:.2f}% | "
        f"Boundary_F1={val_m['boundary_f1']:.2f}% | HD95={val_m['hd95']:.2f}px"
    )
    print(
        f"In-Domain Held-Out Test (Kvasir-Test, n=100): "
        f"mDice={in_m['dice']:.2f}% | mIoU={in_m['iou']:.2f}% | "
        f"Boundary_F1={in_m['boundary_f1']:.2f}% | HD95={in_m['hd95']:.2f}px | MAE={in_m['mae']:.2f}%"
    )
    print(
        f"OOD Clean 6-Center Macro (n={final_ood['num_samples']}): "
        f"mDice={ood_mac['dice']:.2f}% | mIoU={ood_mac['iou']:.2f}% | "
        f"Boundary_F1={ood_mac['boundary_f1']:.2f}% | HD95={ood_mac['hd95']:.2f}px | MAE={ood_mac['mae']:.2f}%"
    )
    print(
        f"OOD Clean 6-Center Micro (All {final_ood['num_samples']} frames): "
        f"mDice={ood_mic['dice']:.2f}% | mIoU={ood_mic['iou']:.2f}% | "
        f"Boundary_F1={ood_mic['boundary_f1']:.2f}% | HD95={ood_mic['hd95']:.2f}px | MAE={ood_mic['mae']:.2f}%"
    )
    if ood_sm:
        print(
            f"OOD Small/Flat Polyps (<3% area, n={final_ood['small_lesions_count']}): "
            f"mDice={ood_sm['dice']:.2f}% | mIoU={ood_sm['iou']:.2f}% | "
            f"Boundary_F1={ood_sm['boundary_f1']:.2f}% | HD95={ood_sm['hd95']:.2f}px"
        )
    for cname, cm in final_ood["per_domain"].items():
        print(
            f"  -> {cname:18s}: mDice={cm['dice']:.2f}% | mIoU={cm['iou']:.2f}% | "
            f"Boundary_F1={cm['boundary_f1']:.2f}% | HD95={cm['hd95']:.2f}px"
        )
    print(f"METRICS_JSON_SINGLE_LINE: {json.dumps({k: v for k, v in summary.items() if k != 'history'})}")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
