"""Official training recipes of the reference models, re-implemented line by line from the authors' scripts.

Sources (commits in dang/bench/official/PROVENANCE.md):
  PraNet     MyTrain.py        Adam 1e-4, --epoch 20 (loop range(1, 20) = 19 epochs), batch 16, clip 0.5
  Polyp-PVT  Train.py          AdamW 1e-4 / wd 1e-4, --epoch 100 (99 epochs), batch 16, clip 0.5, adjust_lr(.., 0.1, 200)
  PraNet-V2  MyTrain_med.py    PVT-PraNet-V2, Adam 1e-4, --epoch 30 (29 epochs), batch 8, clip 0.5
All three: 352x352 Resize-only data pipeline (no augmentation), multi-scale {0.75, 1, 1.25} with one optimiser step
per scale, fp32, and the authors' structure losses copied verbatim (including PraNet's legacy `reduce='none'`).
Deliberate deviations, identical for every model: the final-epoch weights are evaluated (Polyp-PVT and PraNet-V2
keep the best checkpoint on test data); PraNet-V2's `cudnn.enabled = False` speed setting is not applied.
"""

import contextlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import io

PVT_V2_B2_URL = "https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth"


# ---------------------------------------------------------------------------------------------------------------------
# Losses, verbatim from the official training scripts
# ---------------------------------------------------------------------------------------------------------------------
def structure_loss(pred, mask):
    """PraNet MyTrain.py / Polyp-PVT Train.py (identical)."""
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def structure_loss_v2(pred, pred_bg, mask_fg, mask_bg):
    """PraNet-V2 MyTrain_med.py."""
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask_fg, kernel_size=31, stride=1, padding=15) - mask_fg)
    wbce = F.binary_cross_entropy_with_logits(pred, mask_fg, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    wbce2 = F.binary_cross_entropy_with_logits(pred_bg, mask_bg, reduction='none')
    wbce2 = (weit * wbce2).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask_fg) * weit).sum(dim=(2, 3))
    union = ((pred + mask_fg) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou + 0.8 * wbce2).mean()


def pranet_loss(outputs, gts):
    lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = outputs
    return (structure_loss(lateral_map_2, gts) + structure_loss(lateral_map_3, gts)
            + structure_loss(lateral_map_4, gts) + structure_loss(lateral_map_5, gts))


def polyp_pvt_loss(outputs, gts):
    P1, P2 = outputs
    return structure_loss(P1, gts) + structure_loss(P2, gts)


def pranet_v2_loss(outputs, gts):
    l2f, l3f, l4f, l5f, l2b, l3b, l4b, l5b = outputs
    bg_mask = 1 - gts
    loss5 = structure_loss_v2(l2f, l2b, gts, bg_mask)
    loss4 = structure_loss_v2(l3f, l3b, gts, bg_mask)
    loss3 = structure_loss_v2(l4f, l4b, gts, bg_mask)
    loss2 = structure_loss_v2(l5f, l5b, gts, bg_mask)
    return loss2 + loss3 + loss4 + loss5


# ---------------------------------------------------------------------------------------------------------------------
# Test-time output fusion, from the official test scripts (logits, before the sigmoid)
# ---------------------------------------------------------------------------------------------------------------------
def pranet_test_logits(outputs):          # MyTest.py: res5, res4, res3, res2 = model(image); res = res2
    return outputs[3]


def polyp_pvt_test_logits(outputs):       # Test.py: P1, P2 = model(image); res = P1 + P2
    return outputs[0] + outputs[1]


def pranet_v2_test_logits(outputs):       # MyTest_med.py (non-PraNet branch): res = res2 + res3 + res4 + res5
    return outputs[0] + outputs[1] + outputs[2] + outputs[3]


# ---------------------------------------------------------------------------------------------------------------------
# utils/utils.py (identical in the three repositories)
# ---------------------------------------------------------------------------------------------------------------------
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


# ---------------------------------------------------------------------------------------------------------------------
# Model construction with the ImageNet weights the official constructors expect
# ---------------------------------------------------------------------------------------------------------------------
def _pvt_weights() -> str:
    cache = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch")) / "bench_weights"
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / "pvt_v2_b2.pth"
    if not dst.exists():
        torch.hub.download_url_to_file(PVT_V2_B2_URL, str(dst), progress=False)
    return str(dst)


@contextlib.contextmanager
def _official_cwd():
    """Run an official constructor from a directory holding ./pretrained_pth/ and ./models/ (its relative paths)."""
    src = _pvt_weights()
    old = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        for sub in ("pretrained_pth", "models"):
            os.makedirs(os.path.join(tmp, sub))
            os.symlink(src, os.path.join(tmp, sub, "pvt_v2_b2.pth"))
        os.chdir(tmp)
        try:
            yield
        finally:
            os.chdir(old)


def build_pranet():
    from dang.bench.official.pranet.PraNet_Res2Net import PraNet
    return PraNet()


def build_polyp_pvt():
    from dang.bench.official.polyp_pvt.pvt import PolypPVT
    with _official_cwd():
        return PolypPVT()


def build_pranet_v2():
    from dang.bench.official.pranet_v2.pranet import PVT_PraNet_V2
    with _official_cwd():
        return PVT_PraNet_V2(num_class=1)


@dataclass(frozen=True)
class Recipe:
    name: str
    paper: str
    build: Callable
    optimizer: str              # "Adam" | "AdamW"
    lr: float
    weight_decay: float
    epoch_arg: int              # the script's --epoch; the loop is range(1, epoch_arg)
    batch_size: int
    clip: float
    decay_rate: float
    decay_epoch: int
    loss: Callable
    test_logits: Callable
    size_rates: tuple = (0.75, 1, 1.25)
    trainsize: int = 352


RECIPES = {
    "pranet": Recipe("PraNet (Res2Net-50)", "arXiv:2006.11392", build_pranet, "Adam", 1e-4, 0.0,
                     20, 16, 0.5, 0.1, 50, pranet_loss, pranet_test_logits),
    "polyp_pvt": Recipe("Polyp-PVT (PVTv2-B2)", "arXiv:2108.06932", build_polyp_pvt, "AdamW", 1e-4, 1e-4,
                        100, 16, 0.5, 0.1, 200, polyp_pvt_loss, polyp_pvt_test_logits),
    "pranet_v2": Recipe("PraNet-V2 (PVT-PraNet-V2, PVTv2-B2)", "arXiv:2504.10986", build_pranet_v2, "Adam", 1e-4, 0.0,
                        30, 8, 0.5, 0.1, 50, pranet_v2_loss, pranet_v2_test_logits),
}


def make_optimizer(recipe: Recipe, model):
    params = model.parameters()
    if recipe.optimizer == "AdamW":
        return torch.optim.AdamW(params, recipe.lr, weight_decay=recipe.weight_decay)
    return torch.optim.Adam(params, recipe.lr)


# ---------------------------------------------------------------------------------------------------------------------
# utils/dataloader.py PolypDataset (augmentation off): Resize((352, 352)) + ToTensor (+ ImageNet Normalize for images)
# ---------------------------------------------------------------------------------------------------------------------
class OfficialTrainDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[dict], trainsize: int = 352):
        # filter_files(): keep pairs whose image and mask sizes agree
        self.samples = [s for s in samples
                        if Image.open(io.BytesIO(s["img_bytes"])).size == Image.open(io.BytesIO(s["mask_bytes"])).size]
        self.dropped = len(samples) - len(self.samples)
        self.img_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor()])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]
        image = Image.open(io.BytesIO(s["img_bytes"])).convert('RGB')
        gt = Image.open(io.BytesIO(s["mask_bytes"])).convert('L')
        return self.img_transform(image), self.gt_transform(gt)


def train_official(recipe: Recipe, model, train_samples, device, log, num_workers: int = 2, amp: bool = False):
    """One seed of the official loop: for epoch in range(1, epoch_arg): adjust_lr; for each batch and scale: step."""
    ds = OfficialTrainDataset(train_samples, recipe.trainsize)
    loader = torch.utils.data.DataLoader(ds, batch_size=recipe.batch_size, shuffle=True,
                                         num_workers=num_workers, pin_memory=True)
    optimizer = make_optimizer(recipe, model)
    # amp=True: fp16 autocast + GradScaler (gradients unscaled before the official value clipping). The only change
    # to the official loop; used when the fp32 recipe does not fit the compute (Polyp-PVT: ~14 h/seed on a T4).
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    log(f"official recipe {recipe.name}: optimizer={optimizer.__class__.__name__} lr={recipe.lr} "
        f"wd={recipe.weight_decay} epochs={recipe.epoch_arg - 1} batch={recipe.batch_size} clip={recipe.clip} "
        f"adjust_lr({recipe.decay_rate}, {recipe.decay_epoch}) scales={recipe.size_rates} precision={'fp16-AMP' if amp else 'fp32'} "
        f"train_images={len(ds)} (size-mismatch dropped={ds.dropped}) steps/epoch={len(loader)}x{len(recipe.size_rates)}")
    import time
    for epoch in range(1, recipe.epoch_arg):
        t0 = time.time()
        adjust_lr(optimizer, recipe.lr, epoch, recipe.decay_rate, recipe.decay_epoch)
        model.train()
        tot, n = 0.0, 0
        for images0, gts0 in loader:
            for rate in recipe.size_rates:
                optimizer.zero_grad()
                images = images0.to(device, non_blocking=True)
                gts = gts0.to(device, non_blocking=True)
                trainsize = int(round(recipe.trainsize * rate / 32) * 32)
                if rate != 1:
                    images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                    outputs = model(images)
                    loss = recipe.loss(outputs, gts)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_gradient(optimizer, recipe.clip)
                scaler.step(optimizer)
                scaler.update()
                if rate == 1:
                    tot += float(loss.item())
                    n += 1
        if not torch.isfinite(torch.tensor(tot)):
            raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
        lr_now = optimizer.param_groups[0]["lr"]
        log(f"Ep {epoch:03d}/{recipe.epoch_arg - 1:03d} loss(scale 1)={tot / max(1, n):.4f} lr={lr_now:.2e} "
            f"time={time.time() - t0:.1f}s")
    return model
