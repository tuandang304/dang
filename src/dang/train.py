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
    # Unweighted region BCE/Dice + Precision-aware Asymmetric Tversky loss
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
    """
    Separate learning rates and weight decay across:
      - Pretrained ConvNeXt backbone (cnn_lr, gentle fine-tuning to prevent forgetting)
      - DINOv2 Freq-MoE Conv-LoRA adapters (peft_lr, fast low-rank adaptation)
      - Newly initialized fusion & decoder modules (new_lr)
    """
    cnn_lr = getattr(cfg, "cnn_lr", 6e-5)
    new_lr = getattr(cfg, "lr", 5e-4)
    peft_lr = getattr(cfg, "peft_lr", 1e-3)

    groups = defaultdict(list)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_decay = (
            p.ndim <= 1
            or name.endswith(".bias")
            or "gamma" in name
            or "scale_" in name
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
    print(f"DANG-Med v2 Clean Protocol Runner | Variant: {cfg.variant_name} | Seed: {cfg.seed}")
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
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()

    eff = measure_efficiency(model, cfg, device)
    print(
        f"Model Efficiency: Total={eff['total_params_m']:.2f}M | "
        f"Trainable={eff['trainable_params_m']:.2f}M ({eff['trainable_pct']:.2f}%) | "
        f"Frozen={eff['frozen_params_m']:.2f}M | Speed={eff['fps']:.1f} FPS",
        flush=True,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(model, cfg)

    total_steps = cfg.epochs * len(train_loader)
    warmup_steps = max(1, cfg.warmup_epochs * len(train_loader))
    global_step = 0

    best_val_dice = -1.0
    best_state = None
    best_epoch = 0
    history = []

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
        for batch in train_loader:
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
        val_raw = evaluate_loader(model, val_loader, device, full_metrics=False)
        val_ema = evaluate_loader(ema_model, val_loader, device, full_metrics=False)

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

        # Strictly select checkpoint on held-out Kvasir-Val (never on Test or OOD)
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in cand_model.state_dict().items()}

        ep_time = time.time() - ep_t0
        cur_new_lr = max(pg["lr"] for pg in optimizer.param_groups)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_dice": round(val_dice, 2),
            "val_iou": round(val_iou, 2),
            "val_src": src_tag,
            "lr": round(cur_new_lr, 6),
            "time_s": round(ep_time, 2),
        }
        history.append(row)

        print(
            f"[Epoch {epoch:02d}/{cfg.epochs:02d}] "
            f"loss={train_loss:.4f} | val_dice={val_dice:.2f}% ({src_tag}) | val_iou={val_iou:.2f}% "
            f"| lr={cur_new_lr:.2e} | time={ep_time:.1f}s",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    val_final = evaluate_loader(model, val_loader, device, full_metrics=True)
    id_final = evaluate_loader(model, id_test_loader, device, full_metrics=True)
    ood_final = evaluate_loader(model, ood_loader, device, full_metrics=True)

    print("\n" + "=" * 80)
    print(f"FINAL SUMMARY | Variant: {cfg.variant_name} (Best Val Epoch {best_epoch})")
    print("-" * 80)
    print(
        f"In-Domain Val (Kvasir-Val, n=100): "
        f"mDice={val_final['micro']['dice']:.2f}% | "
        f"mIoU={val_final['micro']['iou']:.2f}% | "
        f"Boundary_F1={val_final['micro']['boundary_f1']:.2f}% | "
        f"HD95={val_final['micro']['hd95']:.2f}px"
    )
    print(
        f"In-Domain Held-Out Test (Kvasir-Test, n=100): "
        f"mDice={id_final['micro']['dice']:.2f}% | "
        f"mIoU={id_final['micro']['iou']:.2f}% | "
        f"Boundary_F1={id_final['micro']['boundary_f1']:.2f}% | "
        f"HD95={id_final['micro']['hd95']:.2f}px (non-empty={id_final['micro']['hd95_non_empty']:.2f}px) | "
        f"MAE={id_final['micro']['mae']:.2f}%"
    )
    print(
        f"OOD Clean 6-Center Macro (n=272): "
        f"mDice={ood_final['macro']['dice']:.2f}% | "
        f"mIoU={ood_final['macro']['iou']:.2f}% | "
        f"Boundary_F1={ood_final['macro']['boundary_f1']:.2f}% | "
        f"HD95={ood_final['macro']['hd95']:.2f}px (non-empty={ood_final['macro']['hd95_non_empty']:.2f}px) | "
        f"MAE={ood_final['macro']['mae']:.2f}%"
    )
    print(
        f"OOD Clean 6-Center Micro (All 272 frames): "
        f"mDice={ood_final['micro']['dice']:.2f}% | "
        f"mIoU={ood_final['micro']['iou']:.2f}% | "
        f"Boundary_F1={ood_final['micro']['boundary_f1']:.2f}% | "
        f"HD95={ood_final['micro']['hd95']:.2f}px (non-empty={ood_final['micro']['hd95_non_empty']:.2f}px) | "
        f"EmptyPred={ood_final['micro']['empty_pred']:.2f}%"
    )
    if ood_final.get("small_lesions"):
        sm = ood_final["small_lesions"]
        print(
            f"OOD Small/Flat Polyps (<3% area, n={ood_final['small_lesions_count']}): "
            f"mDice={sm['dice']:.2f}% | mIoU={sm['iou']:.2f}% | "
            f"Boundary_F1={sm['boundary_f1']:.2f}% | Precision={sm['precision']:.2f}% | Recall={sm['recall']:.2f}% | "
            f"HD95={sm['hd95']:.2f}px"
        )
    if ood_final.get("med_large_lesions"):
        ml = ood_final["med_large_lesions"]
        print(
            f"OOD Med/Large Polyps (>=3% area, n={ood_final['med_large_lesions_count']}): "
            f"mDice={ml['dice']:.2f}% | mIoU={ml['iou']:.2f}% | "
            f"Boundary_F1={ml['boundary_f1']:.2f}% | Precision={ml['precision']:.2f}% | Recall={ml['recall']:.2f}% | "
            f"HD95={ml['hd95']:.2f}px"
        )
    for dom, dm in ood_final["per_domain"].items():
        print(
            f"  -> {dom:17s}: mDice={dm['dice']:.2f}% | mIoU={dm['iou']:.2f}% | "
            f"Boundary_F1={dm['boundary_f1']:.2f}% | HD95={dm['hd95']:.2f}px (non-empty={dm['hd95_non_empty']:.2f}px)"
        )

    summary_payload = {
        "variant": cfg.variant_name,
        "config": cfg.to_dict(),
        "gpu": gpu_name,
        "best_epoch": best_epoch,
        "efficiency": eff,
        "in_domain_kvasir_val": val_final["micro"],
        "in_domain_kvasir": id_final["micro"],
        "ood_polypgen_macro": ood_final["macro"],
        "ood_polypgen_micro": ood_final["micro"],
        "ood_small_lesions": ood_final.get("small_lesions", {}),
        "ood_small_lesions_count": ood_final.get("small_lesions_count", 0),
        "ood_med_large_lesions": ood_final.get("med_large_lesions", {}),
        "ood_med_large_lesions_count": ood_final.get("med_large_lesions_count", 0),
        "ood_per_center": ood_final["per_domain"],
    }
    print("METRICS_JSON_SINGLE_LINE: " + json.dumps(summary_payload), flush=True)
    print("PER_IMAGE_JSON_SINGLE_LINE: " + json.dumps({"val": val_final["records"], "id_test": id_final["records"], "ood_polypgen": ood_final["records"]}), flush=True)
    print("=" * 80, flush=True)

    out_dir = Path(os.environ.get("ORX_OUTPUT_DIR", "outputs")) / cfg.variant_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary_payload, f, indent=2)
    with open(out_dir / "per_image_metrics.json", "w") as f:
        json.dump({
            "val": val_final["records"],
            "id_test": id_final["records"],
            "ood_polypgen": ood_final["records"],
        }, f, indent=2)
    # Also save at top-level ORX_OUTPUT_DIR for easy retrieval
    top_out = Path(os.environ.get("ORX_OUTPUT_DIR", "outputs"))
    with open(top_out / "per_image_metrics.json", "w") as f:
        json.dump({
            "val": val_final["records"],
            "id_test": id_final["records"],
            "ood_polypgen": ood_final["records"],
        }, f, indent=2)


if __name__ == "__main__":
    main()
