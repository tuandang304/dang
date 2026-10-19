import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion


def compute_boundary_mask(mask_bin: np.ndarray) -> np.ndarray:
    """Extract 1-pixel morphological contour: mask XOR binary_erosion(mask)."""
    if mask_bin.max() == 0:
        return np.zeros_like(mask_bin, dtype=bool)
    eroded = binary_erosion(mask_bin, iterations=1, border_value=0)
    return np.logical_xor(mask_bin, eroded)


def compute_binary_dice(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    inter = float(np.logical_and(pred_bin, gt_bin).sum())
    pred_sum = float(pred_bin.sum())
    gt_sum = float(gt_bin.sum())
    if pred_sum == 0.0 and gt_sum == 0.0:
        return 100.0
    if pred_sum == 0.0 or gt_sum == 0.0:
        return 0.0
    return (2.0 * inter / (pred_sum + gt_sum)) * 100.0


def compute_sample_metrics(
    pred_prob: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float = 0.5,
    dice_native: float | None = None,
) -> dict:
    """
    Compute standard medical segmentation metrics on a single 2D image:
    - dice: Dice similarity coefficient at 352x352 (%)
    - dice_native: Dice similarity coefficient at original native resolution (%)
    - iou: Intersection over Union (%)
    - boundary_f1: Boundary F1-score on 1-px contours within 2px tolerance (%)
    - hd95: Two-way 95th percentile Hausdorff Distance with 0.25*diag empty penalty (px)
    - hd95_non_empty: Two-way HD95 on non-empty predictions only (NaN if empty)
    - hd95_norm_pct: HD95 normalized by image diagonal (%)
    - empty_pred: 100.0 if prediction is empty on non-empty GT, else 0.0 (%)
    - mae: Mean Absolute Error (0-100 scale, %)
    - precision: Precision (%, 0.0 when prediction is empty on non-empty GT)
    - recall: Sensitivity / Recall (%)
    - gt_area_pct: Foreground lesion area percentage of frame (%)
    """
    pred_bin = (pred_prob >= threshold).astype(bool)
    gt_bin = (gt_mask >= 0.5).astype(bool)

    inter = float(np.logical_and(pred_bin, gt_bin).sum())
    pred_sum = float(pred_bin.sum())
    gt_sum = float(gt_bin.sum())
    union = pred_sum + gt_sum - inter

    if pred_sum == 0.0 and gt_sum == 0.0:
        dice = 100.0
        iou = 100.0
        precision = 100.0
        recall = 100.0
    elif pred_sum == 0.0 or gt_sum == 0.0:
        dice = 0.0
        iou = 0.0
        precision = 0.0
        recall = 0.0
    else:
        dice = (2.0 * inter / (pred_sum + gt_sum)) * 100.0
        iou = (inter / union) * 100.0
        precision = (inter / pred_sum) * 100.0
        recall = (inter / gt_sum) * 100.0

    mae = float(np.abs(pred_prob - gt_bin.astype(np.float32)).mean()) * 100.0
    gt_area_pct = float(gt_bin.mean()) * 100.0

    pred_b = compute_boundary_mask(pred_bin)
    gt_b = compute_boundary_mask(gt_bin)

    H, W = gt_bin.shape
    max_dist = float(np.hypot(H, W))

    if pred_b.sum() == 0 and gt_b.sum() == 0:
        boundary_f1 = 100.0
        hd95 = 0.0
        hd95_non_empty = 0.0
        empty_pred = 0.0
    elif pred_b.sum() == 0 or gt_b.sum() == 0:
        boundary_f1 = 0.0
        hd95 = max_dist * 0.25
        hd95_non_empty = float("nan")
        empty_pred = 100.0 if (pred_b.sum() == 0 and gt_b.sum() > 0) else 0.0
    else:
        dist_to_gt = distance_transform_edt(~gt_b)
        dist_to_pred = distance_transform_edt(~pred_b)

        d_pred_to_gt = dist_to_gt[pred_b]
        d_gt_to_pred = dist_to_pred[gt_b]

        tol = 2.0
        b_prec = float((d_pred_to_gt <= tol).mean())
        b_rec = float((d_gt_to_pred <= tol).mean())
        denom = b_prec + b_rec
        boundary_f1 = (2.0 * b_prec * b_rec / denom) * 100.0 if denom > 0.0 else 0.0

        hd95_p2g = float(np.percentile(d_pred_to_gt, 95))
        hd95_g2p = float(np.percentile(d_gt_to_pred, 95))
        hd95 = max(hd95_p2g, hd95_g2p)
        hd95_non_empty = hd95
        empty_pred = 0.0

    hd95_norm_pct = (hd95 / max_dist) * 100.0 if max_dist > 0 else 0.0

    return {
        "dice": float(dice),
        "dice_native": float(dice if dice_native is None else dice_native),
        "iou": float(iou),
        "boundary_f1": float(boundary_f1),
        "hd95": float(hd95),
        "hd95_non_empty": float(hd95_non_empty),
        "hd95_norm_pct": float(hd95_norm_pct),
        "empty_pred": float(empty_pred),
        "mae": float(mae),
        "precision": float(precision),
        "recall": float(recall),
        "gt_area_pct": float(gt_area_pct),
    }


def aggregate_metrics(records: list[dict]) -> dict:
    if not records:
        return {}
    numeric_keys = [
        "dice",
        "dice_native",
        "iou",
        "boundary_f1",
        "hd95",
        "hd95_norm_pct",
        "mae",
        "precision",
        "recall",
        "empty_pred",
    ]
    out = {"n": len(records)}
    for k in numeric_keys:
        if k in records[0]:
            out[k] = float(np.mean([r[k] for r in records]))
    if "hd95_non_empty" in records[0]:
        valid_hd = [r["hd95_non_empty"] for r in records if not np.isnan(r["hd95_non_empty"])]
        out["hd95_non_empty"] = float(np.mean(valid_hd)) if valid_hd else float("nan")
        if valid_hd:
            max_dist = float(np.hypot(352, 352))
            out["hd95_non_empty_norm_pct"] = float(np.mean(valid_hd) / max_dist * 100.0)
    out["empty_preds"] = int(sum(1 for r in records if r.get("empty_pred", 0.0) > 50.0))
    return out
