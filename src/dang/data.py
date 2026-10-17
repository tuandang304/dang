import hashlib
import io
import random
import re
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
from huggingface_hub import hf_hub_download


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-", re.IGNORECASE)


def compute_dhash64(img_bytes: bytes) -> int:
    """Compute 64-bit perceptual difference hash (dHash) on 9x8 grayscale image."""
    im = Image.open(io.BytesIO(img_bytes)).convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    arr = np.asarray(im, dtype=np.float32)
    diff = (arr[:, 1:] > arr[:, :-1]).flatten()
    val = 0
    for bit in diff:
        val = (val << 1) | int(bit)
    return val


def hamming_dist64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def classify_polypgen_official(filename: str) -> str:
    """
    Map PolypGen filenames deterministically to the 6 official PolypGen centers
    (Jha et al., Scientific Data 2023, arXiv:2106.04463, Table 2):
      - C1: Ambroise Pare Hospital, Paris, France (OLCV / 100H / 100S)
      - C2: Istituto Oncologico Veneto, Padova, Italy (EndoCV2021_<digits>.jpg)
      - C3: Centro di Riferimento Oncologico, Aviano, Italy (C3_EndoCV2021_...)
      - C4: Oslo University Hospital, Oslo, Norway (<seq>_endocv2021_positive_...)
      - C5: John Radcliffe Hospital, University of Oxford, UK (EndoCV2021_C5_...)
      - C6: University of Alexandria, Alexandria, Egypt (EndoCV2021_C6_...)
    UUID-named files belong to Kvasir-SEG and are strictly excluded from OOD.
    """
    base = filename.rsplit("/", 1)[-1]
    if UUID_RE.match(base):
        return "InDomain_Kvasir_UUID"
    fl = base.lower()
    if fl.startswith("c3_endocv"):
        return "OOD_C3_Aviano"
    if "endocv" in fl and "_c5_" in fl:
        return "OOD_C5_Oxford"
    if "endocv" in fl and "_c6_" in fl:
        return "OOD_C6_Alexandria"
    if "_endocv2021_positive" in fl or "_endocv2021_negative" in fl:
        return "OOD_C4_Oslo"
    if fl.startswith("endocv2021_"):
        return "OOD_C2_Padova"
    if "olcv" in fl or re.match(r"^\d+[hs]\d+\.jpg$", fl):
        return "OOD_C1_Paris"
    return "Unknown"


class MedicalSegDataset(Dataset):
    def __init__(self, samples: list[dict], img_size: int = 352, is_train: bool = False):
        self.samples = samples
        self.img_size = img_size
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]
        img = Image.open(io.BytesIO(item["img_bytes"])).convert("RGB")
        mask_raw = Image.open(io.BytesIO(item["mask_bytes"])).convert("L")
        mask_np = np.array(mask_raw, dtype=np.float32)
        if mask_np.max() > 1.0:
            mask_bin = (mask_np >= 127.5).astype(np.uint8) * 255
        else:
            mask_bin = (mask_np > 0.0).astype(np.uint8) * 255
        mask = Image.fromarray(mask_bin, mode="L")

        if self.is_train:
            # P2: 30% probability of true shrink-and-black-pad zoom-out (realistic endoscopy black border, no mirrored polyps)
            s = random.uniform(0.65, 0.90) if random.random() < 0.30 else random.uniform(0.95, 1.20)
            scale_size = int(self.img_size * s)
            img = TF.resize(img, [scale_size, scale_size], interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, [scale_size, scale_size], interpolation=TF.InterpolationMode.NEAREST)

            if scale_size > self.img_size:
                top = random.randint(0, scale_size - self.img_size)
                left = random.randint(0, scale_size - self.img_size)
                img = TF.crop(img, top, left, self.img_size, self.img_size)
                mask = TF.crop(mask, top, left, self.img_size, self.img_size)
            elif scale_size < self.img_size:
                pad = self.img_size - scale_size
                left = random.randint(0, pad)
                top = random.randint(0, pad)
                padding = [left, top, pad - left, pad - top]
                img = TF.pad(img, padding, fill=0, padding_mode="constant")
                mask = TF.pad(mask, padding, fill=0, padding_mode="constant")

            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() < 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
            if random.random() < 0.5:
                angle = random.choice([90, 180, 270])
                img = TF.rotate(img, angle)
                mask = TF.rotate(mask, angle)
            if random.random() < 0.4:
                img = TF.adjust_brightness(img, random.uniform(0.85, 1.15))
                img = TF.adjust_contrast(img, random.uniform(0.85, 1.15))
                img = TF.adjust_saturation(img, random.uniform(0.85, 1.15))
            if random.random() < 0.3:
                img = TF.adjust_hue(img, random.uniform(-0.04, 0.04))
            if random.random() < 0.3:
                img = TF.adjust_gamma(img, random.uniform(0.80, 1.25))
        else:
            img = TF.resize(img, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.NEAREST)

        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        mask_t = (TF.to_tensor(mask) > 0.5).float()

        return {
            "image": img_t,
            "mask": mask_t,
            "name": item["name"],
            "domain": item["domain"],
        }


def build_dataloaders(cfg):
    train_pq = hf_hub_download(
        "kowndinya23/Kvasir-SEG",
        "data/train-00000-of-00001-7a33e3aba9046bad.parquet",
        repo_type="dataset",
    )
    val_pq = hf_hub_download(
        "kowndinya23/Kvasir-SEG",
        "data/validation-00000-of-00001-4d65c2d2950bdc96.parquet",
        repo_type="dataset",
    )
    ood_pq = hf_hub_download(
        "mzschwartz88/combo_polyp",
        "data/validation-00000-of-00001.parquet",
        repo_type="dataset",
    )

    raw_kvasir = pq.read_table(train_pq).to_pylist() + pq.read_table(val_pq).to_pylist()
    raw_kvasir.sort(key=lambda r: r["name"])
    split_rng = random.Random(20260101)
    split_rng.shuffle(raw_kvasir)

    kvasir_md5 = set()
    all_kvasir_samples = []
    for r in raw_kvasir:
        img_b = r["image"]["bytes"]
        kvasir_md5.add(hashlib.md5(img_b).hexdigest())
        all_kvasir_samples.append({
            "img_bytes": img_b,
            "mask_bytes": r["annotation"]["bytes"],
            "name": r["name"],
            "domain": "Kvasir_InDomain",
            "dhash": compute_dhash64(img_b),
        })

    # P1: Exact R4 800-image training split; drop only true near-duplicates (dHash <= 4) from Val/Test without swapping
    train_samples = list(all_kvasir_samples[:800])
    train_hashes = [s["dhash"] for s in train_samples]
    is_near_dup = lambda s: any(hamming_dist64(s["dhash"], th) <= 4 for th in train_hashes)

    val_samples = [s for s in all_kvasir_samples[800:900] if not is_near_dup(s)]
    val_hashes = [s["dhash"] for s in val_samples]
    id_test_samples = [
        s for s in all_kvasir_samples[900:1000]
        if not is_near_dup(s) and not any(hamming_dist64(s["dhash"], vh) <= 4 for vh in val_hashes)
    ]

    ood_rows = pq.read_table(ood_pq).to_pylist()
    ood_samples = []
    seen_ood_md5 = set()
    for r in ood_rows:
        fn = r["image"].get("path", "unknown.jpg")
        dom = classify_polypgen_official(fn)
        if not dom.startswith("OOD_"):
            continue
        img_b = r["image"]["bytes"]
        digest = hashlib.md5(img_b).hexdigest()
        if digest in kvasir_md5 or digest in seen_ood_md5:
            continue
        arr = np.array(Image.open(io.BytesIO(r["label"]["bytes"])).convert("L"))
        if arr.max() == 0:
            continue
        seen_ood_md5.add(digest)
        ood_samples.append({
            "img_bytes": img_b,
            "mask_bytes": r["label"]["bytes"],
            "name": fn,
            "domain": dom,
        })

    assert len(train_samples) == 800 and len(val_samples) >= 90 and len(id_test_samples) >= 90, "Kvasir split size mismatch"
    assert len(ood_samples) == 272, f"Expected 272 clean OOD frames, got {len(ood_samples)}"

    train_ds = MedicalSegDataset(train_samples, img_size=cfg.img_size, is_train=True)
    val_ds = MedicalSegDataset(val_samples, img_size=cfg.img_size, is_train=False)
    id_test_ds = MedicalSegDataset(id_test_samples, img_size=cfg.img_size, is_train=False)
    ood_ds = MedicalSegDataset(ood_samples, img_size=cfg.img_size, is_train=False)

    g = torch.Generator()
    g.manual_seed(cfg.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    id_test_loader = DataLoader(
        id_test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    ood_loader = DataLoader(
        ood_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, id_test_loader, ood_loader
