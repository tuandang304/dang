"""One evaluator for every model: identical preprocessing, thresholds and metric code.

Primary metrics are computed at native ground-truth resolution with the logits upsampled bilinearly (as the official
test scripts do) and thresholded at probability 0.5. `mdice_tb` / `miou_tb` reproduce the evaluation toolbox behind
the published PraNet/Polyp-PVT/PraNet-V2 tables: the probability map is min-max normalised, quantised to 8 bits, and
Dice/IoU are averaged over the 256 thresholds k/255 (pred >= threshold, score 0 when there is no true positive).
Ground truth is binarised at 127.5 for both.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dang.data import MedicalSegDataset, decode_binary_mask_np
from dang.metrics import aggregate_metrics, compute_sample_metrics
from dang.train import dump_pred_mask

EXTRA_KEYS = ["iou_native", "recall_native", "precision_native", "mdice_tb", "miou_tb"]


def toolbox_scores(prob: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    norm = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)
    q = np.rint(norm * 255.0).astype(np.int64).ravel()
    g = gt.ravel()
    num_rec = np.cumsum(np.bincount(q, minlength=256)[::-1])[::-1].astype(np.float64)   # #pred >= k/255
    tp = np.cumsum(np.bincount(q[g], minlength=256)[::-1])[::-1].astype(np.float64)
    n_gt = float(g.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        dice = np.where(tp > 0, 2.0 * tp / (n_gt + num_rec), 0.0)
        iou = np.where(tp > 0, tp / (n_gt - tp + num_rec), 0.0)
    return float(dice.mean() * 100.0), float(iou.mean() * 100.0)


def _native_scores(pred: np.ndarray, gt: np.ndarray) -> dict:
    inter = float(np.logical_and(pred, gt).sum())
    ps, gs = float(pred.sum()), float(gt.sum())
    if ps == 0 and gs == 0:
        return {"dice_native": 100.0, "iou_native": 100.0, "recall_native": 100.0, "precision_native": 100.0}
    if ps == 0 or gs == 0 or inter == 0:
        return {"dice_native": 0.0, "iou_native": 0.0, "recall_native": 0.0, "precision_native": 0.0}
    return {"dice_native": 200.0 * inter / (ps + gs), "iou_native": 100.0 * inter / (ps + gs - inter),
            "recall_native": 100.0 * inter / gs, "precision_native": 100.0 * inter / ps}


@torch.no_grad()
def evaluate_cohort(model, samples, logits_fn, device, amp_dtype=None, dump=None, batch_size=16, num_workers=2):
    loader = DataLoader(MedicalSegDataset(samples, img_size=352, is_train=False), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, pin_memory=True)
    by_uid = {s["uid"]: s for s in samples}
    model.eval()
    records = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        if amp_dtype is not None and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                logits = logits_fn(model(images)).float()
        else:
            logits = logits_fn(model(images)).float()
        probs352 = torch.sigmoid(logits).cpu().numpy()
        masks352 = batch["mask"].numpy()
        for i, uid in enumerate(batch["uid"]):
            gt = decode_binary_mask_np(by_uid[uid]["mask_bytes"])
            up = F.interpolate(logits[i:i + 1], size=gt.shape, mode="bilinear", align_corners=False)
            prob = torch.sigmoid(up)[0, 0].cpu().numpy()
            nat = _native_scores(prob >= 0.5, gt)
            rec = compute_sample_metrics(probs352[i, 0], masks352[i, 0], dice_native=nat["dice_native"])
            rec.update(nat)
            rec["mdice_tb"], rec["miou_tb"] = toolbox_scores(prob, gt)
            rec.update({"uid": uid, "name": batch["name"][i], "domain": batch["domain"][i]})
            if dump is not None:
                dump_pred_mask(dump, uid, probs352[i, 0] >= 0.5)
            records.append(rec)
    return records


def aggregate(records: list[dict]) -> dict:
    out = aggregate_metrics(records)
    for k in EXTRA_KEYS:
        out[k] = float(np.mean([r[k] for r in records]))
    return out


def summarize(records: list[dict]) -> dict:
    by_dom = {}
    for r in records:
        by_dom.setdefault(r["domain"], []).append(r)
    per_domain = {d: aggregate(v) for d, v in sorted(by_dom.items())}
    micro = aggregate(records)
    if len(per_domain) > 1:
        keys = [k for k in micro if k not in ("n", "empty_preds") and all(k in v for v in per_domain.values())]
        macro = {k: float(np.mean([v[k] for v in per_domain.values()])) for k in keys}
        macro["n"], macro["num_domains"] = len(records), len(per_domain)
    else:
        macro = dict(micro)
    return {"micro": micro, "macro": macro, "per_domain": per_domain}
