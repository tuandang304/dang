import hashlib
import io
import random
import re
import zipfile
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


def extract_seq_id(filename: str, domain: str) -> str:
    base = filename.rsplit("/", 1)[-1]
    if domain == "OOD_C4_Oslo":
        return f"C4_seq_{base.split('_')[0]}"
    return f"{domain}_{base}"


def decode_binary_mask_np(mask_bytes: bytes) -> np.ndarray:
    mask_raw = Image.open(io.BytesIO(mask_bytes)).convert("L")
    mask_np = np.array(mask_raw, dtype=np.float32)
    if mask_np.max() > 1.0:
        return mask_np >= 127.5
    return mask_np > 0.0


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
        mask_bool = item["native_gt_bin"] if "native_gt_bin" in item else decode_binary_mask_np(item["mask_bytes"])
        mask = Image.fromarray(mask_bool.astype(np.uint8) * 255, mode="L")

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
            "uid": item["uid"],
        }


def prepare_all_cohorts() -> dict:
    """
    Download and prepare all In-Domain and External OOD cohorts once with strict
    perceptual (dHash) and cryptographic (MD5) hygiene.
    """
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
    ood_dev_pq = hf_hub_download(
        "mzschwartz88/combo_polyp",
        "data/validation-00000-of-00001.parquet",
        repo_type="dataset",
    )
    ood_ext_pq = hf_hub_download(
        "mzschwartz88/combo_polyp",
        "data/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    pranet_zip_path = hf_hub_download(
        "Zyna1121/lightpranet-polyp-data",
        "TestDataset.zip",
        repo_type="dataset",
    )

    raw_kvasir = pq.read_table(train_pq).to_pylist() + pq.read_table(val_pq).to_pylist()
    raw_kvasir.sort(key=lambda r: r["name"])
    split_rng = random.Random(20260101)
    split_rng.shuffle(raw_kvasir)

    all_kvasir_md5 = set()
    all_kvasir_samples = []
    for r in raw_kvasir:
        img_b = r["image"]["bytes"]
        md5_hex = hashlib.md5(img_b).hexdigest()
        all_kvasir_md5.add(md5_hex)
        dh = compute_dhash64(img_b)
        all_kvasir_samples.append({
            "img_bytes": img_b,
            "mask_bytes": r["annotation"]["bytes"],
            "name": r["name"],
            "domain": "Kvasir_InDomain",
            "uid": f"Kvasir_InDomain::{r['name']}",
            "dhash": dh,
            "md5": md5_hex,
            "seq_id": r["name"],
            "c4_seq_in_dev": False,
            "min_kvasir_dhash": 0,
        })

    all_kvasir_dhashes = [s["dhash"] for s in all_kvasir_samples]

    def min_dist_to_kvasir(dh: int) -> int:
        return min(hamming_dist64(dh, kh) for kh in all_kvasir_dhashes)

    # P1: Exact 800-image training split; drop only true near-duplicates (dHash <= 4) from Val/Test without swapping
    train_samples = list(all_kvasir_samples[:800])
    train_hashes = [s["dhash"] for s in train_samples]
    is_near_train = lambda s: any(hamming_dist64(s["dhash"], th) <= 4 for th in train_hashes)

    val_samples = [s for s in all_kvasir_samples[800:900] if not is_near_train(s)]
    val_hashes = [s["dhash"] for s in val_samples]
    id_test_samples = [
        s for s in all_kvasir_samples[900:1000]
        if not is_near_train(s) and not any(hamming_dist64(s["dhash"], vh) <= 4 for vh in val_hashes)
    ]

    # Cohort 1: PolypGen-Dev (n=272 from combo_polyp validation split)
    ood_dev_rows = pq.read_table(ood_dev_pq).to_pylist()
    ood_dev_samples = []
    seen_dev_md5 = set()
    dev_dhashes = []
    dev_c4_seq_ids = set()
    dev_stats = {"raw_rows": len(ood_dev_rows), "kvasir_uuid_dropped": 0, "empty_mask_dropped": 0, "dup_dropped": 0, "dhash_le4_kvasir": 0}

    for r in ood_dev_rows:
        fn = r["image"].get("path", "unknown.jpg")
        dom = classify_polypgen_official(fn)
        assert dom != "Unknown", f"Unexpected unknown PolypGen filename in validation split: {fn}"
        if not dom.startswith("OOD_"):
            dev_stats["kvasir_uuid_dropped"] += 1
            continue
        img_b = r["image"]["bytes"]
        digest = hashlib.md5(img_b).hexdigest()
        if digest in all_kvasir_md5 or digest in seen_dev_md5:
            dev_stats["dup_dropped"] += 1
            continue
        mask_b = r["label"]["bytes"]
        if not decode_binary_mask_np(mask_b).any():
            dev_stats["empty_mask_dropped"] += 1
            continue
        dh = compute_dhash64(img_b)
        min_kd = min_dist_to_kvasir(dh)
        if min_kd <= 4:
            dev_stats["dhash_le4_kvasir"] += 1
        seen_dev_md5.add(digest)
        dev_dhashes.append(dh)
        seq_id = extract_seq_id(fn, dom)
        if dom == "OOD_C4_Oslo":
            dev_c4_seq_ids.add(seq_id)
        ood_dev_samples.append({
            "img_bytes": img_b,
            "mask_bytes": mask_b,
            "name": fn,
            "domain": dom,
            "uid": f"PolypGenDev::{dom}::{fn}",
            "dhash": dh,
            "md5": digest,
            "seq_id": seq_id,
            "c4_seq_in_dev": False,
            "min_kvasir_dhash": min_kd,
        })

    # Cohort 2: PolypGen-Extended Holdout (from combo_polyp train split, strictly disjoint from Kvasir and PolypGen-Dev)
    ood_ext_rows = pq.read_table(ood_ext_pq).to_pylist()
    ood_ext_samples = []
    seen_ext_md5 = set()
    ext_dhashes = []
    ext_stats = {
        "raw_rows": len(ood_ext_rows),
        "kvasir_uuid_dropped": 0,
        "empty_mask_dropped": 0,
        "kvasir_dup_or_dhash_le4_dropped": 0,
        "dev_dup_or_dhash_le4_dropped": 0,
        "intra_ext_dhash_le4_dropped": 0,
        "c4_seq_shared_with_dev_count": 0,
    }

    for r in ood_ext_rows:
        fn = r["image"].get("path", "unknown.jpg")
        dom = classify_polypgen_official(fn)
        assert dom != "Unknown", f"Unexpected unknown PolypGen filename in train split: {fn}"
        if not dom.startswith("OOD_"):
            ext_stats["kvasir_uuid_dropped"] += 1
            continue
        img_b = r["image"]["bytes"]
        digest = hashlib.md5(img_b).hexdigest()
        dh = compute_dhash64(img_b)
        min_kd = min_dist_to_kvasir(dh)
        if digest in all_kvasir_md5 or min_kd <= 4:
            ext_stats["kvasir_dup_or_dhash_le4_dropped"] += 1
            continue
        if digest in seen_dev_md5 or any(hamming_dist64(dh, ddh) <= 4 for ddh in dev_dhashes):
            ext_stats["dev_dup_or_dhash_le4_dropped"] += 1
            continue
        if digest in seen_ext_md5 or any(hamming_dist64(dh, edh) <= 4 for edh in ext_dhashes):
            ext_stats["intra_ext_dhash_le4_dropped"] += 1
            continue
        mask_b = r["label"]["bytes"]
        if not decode_binary_mask_np(mask_b).any():
            ext_stats["empty_mask_dropped"] += 1
            continue
        seen_ext_md5.add(digest)
        ext_dhashes.append(dh)
        seq_id = extract_seq_id(fn, dom)
        c4_shared = (dom == "OOD_C4_Oslo") and (seq_id in dev_c4_seq_ids)
        if c4_shared:
            ext_stats["c4_seq_shared_with_dev_count"] += 1
        ood_ext_samples.append({
            "img_bytes": img_b,
            "mask_bytes": mask_b,
            "name": fn,
            "domain": dom,
            "uid": f"PolypGenExt::{dom}::{fn}",
            "dhash": dh,
            "md5": digest,
            "seq_id": seq_id,
            "c4_seq_in_dev": c4_shared,
            "min_kvasir_dhash": min_kd,
        })

    # Cohort 3: PraNet External Benchmarks (CVC-300, CVC-ClinicDB, CVC-ColonDB, ETIS-LaribPolypDB)
    pranet_benchmarks = [
        ("CVC-300", "Ext_CVC_300"),
        ("CVC-ClinicDB", "Ext_CVC_ClinicDB"),
        ("CVC-ColonDB", "Ext_CVC_ColonDB"),
        ("ETIS-LaribPolypDB", "Ext_ETIS_LaribPolypDB"),
    ]
    pranet_samples = []
    seen_pranet_md5 = set()
    pranet_dhashes = []
    pranet_stats = {}

    with zipfile.ZipFile(pranet_zip_path, "r") as zf:
        namelist = set(zf.namelist())
        for folder_name, dom_label in pranet_benchmarks:
            img_prefix = f"TestDataset/{folder_name}/images/"
            mask_prefix = f"TestDataset/{folder_name}/masks/"
            img_files = sorted([
                n[len(img_prefix):] for n in namelist
                if n.startswith(img_prefix) and not n.endswith("/") and not n[len(img_prefix):].startswith(".")
            ])
            ds_stat = {
                "raw_files": len(img_files),
                "kvasir_dhash_le4_dropped": 0,
                "cross_dup_dhash_le4_dropped": 0,
                "empty_mask_dropped": 0,
                "kept": 0,
            }
            for fname in img_files:
                img_entry = img_prefix + fname
                mask_entry = mask_prefix + fname
                assert mask_entry in namelist, f"Missing mask {mask_entry} for {img_entry}"
                img_b = zf.read(img_entry)
                mask_b = zf.read(mask_entry)
                im_pil = Image.open(io.BytesIO(img_b))
                mk_pil = Image.open(io.BytesIO(mask_b))
                assert im_pil.size == mk_pil.size, f"Size mismatch in {img_entry}: {im_pil.size} vs {mk_pil.size}"
                if not decode_binary_mask_np(mask_b).any():
                    ds_stat["empty_mask_dropped"] += 1
                    continue
                digest = hashlib.md5(img_b).hexdigest()
                dh = compute_dhash64(img_b)
                min_kd = min_dist_to_kvasir(dh)
                if digest in all_kvasir_md5 or min_kd <= 4:
                    ds_stat["kvasir_dhash_le4_dropped"] += 1
                    continue
                if digest in seen_pranet_md5 or any(hamming_dist64(dh, pdh) <= 4 for pdh in pranet_dhashes):
                    ds_stat["cross_dup_dhash_le4_dropped"] += 1
                    continue
                seen_pranet_md5.add(digest)
                pranet_dhashes.append(dh)
                ds_stat["kept"] += 1
                pranet_samples.append({
                    "img_bytes": img_b,
                    "mask_bytes": mask_b,
                    "name": f"{folder_name}/{fname}",
                    "domain": dom_label,
                    "uid": f"{dom_label}::{fname}",
                    "dhash": dh,
                    "md5": digest,
                    "seq_id": f"{dom_label}_{fname}",
                    "c4_seq_in_dev": False,
                    "min_kvasir_dhash": min_kd,
                })
            pranet_stats[dom_label] = ds_stat

    assert len(train_samples) == 800, f"Expected 800 train samples, got {len(train_samples)}"
    assert len(val_samples) == 95, f"Expected 95 val samples, got {len(val_samples)}"
    assert len(id_test_samples) == 90, f"Expected 90 id_test samples, got {len(id_test_samples)}"
    assert len(ood_dev_samples) == 272, f"Expected 272 PolypGen-Dev samples, got {len(ood_dev_samples)}"
    assert len(ood_ext_samples) == 967, f"Expected 967 PolypGen-Extended samples, got {len(ood_ext_samples)}"

    # Build lookup map from uid -> sample metadata (for native-resolution mask evaluation & sequence clustering)
    uid_lookup = {}
    for s in val_samples + id_test_samples + ood_dev_samples + ood_ext_samples + pranet_samples:
        uid_lookup[s["uid"]] = s

    hygiene_manifest = {
        "kvasir_total_raw": len(all_kvasir_samples),
        "kvasir_train_kept": len(train_samples),
        "kvasir_val_kept": len(val_samples),
        "kvasir_id_test_kept": len(id_test_samples),
        "polypgen_dev": {**dev_stats, "kept": len(ood_dev_samples)},
        "polypgen_extended": {**ext_stats, "kept": len(ood_ext_samples)},
        "pranet_external": pranet_stats,
        "pranet_external_total_kept": len(pranet_samples),
    }

    return {
        "train_samples": train_samples,
        "val_samples": val_samples,
        "id_test_samples": id_test_samples,
        "ood_dev_samples": ood_dev_samples,
        "ood_ext_samples": ood_ext_samples,
        "pranet_samples": pranet_samples,
        "uid_lookup": uid_lookup,
        "hygiene_manifest": hygiene_manifest,
    }


def build_dataloaders_for_seed(cohorts: dict, cfg, seed: int) -> dict:
    train_ds = MedicalSegDataset(cohorts["train_samples"], img_size=cfg.img_size, is_train=True)
    val_ds = MedicalSegDataset(cohorts["val_samples"], img_size=cfg.img_size, is_train=False)
    id_test_ds = MedicalSegDataset(cohorts["id_test_samples"], img_size=cfg.img_size, is_train=False)
    ood_dev_ds = MedicalSegDataset(cohorts["ood_dev_samples"], img_size=cfg.img_size, is_train=False)
    ood_ext_ds = MedicalSegDataset(cohorts["ood_ext_samples"], img_size=cfg.img_size, is_train=False)
    pranet_ds = MedicalSegDataset(cohorts["pranet_samples"], img_size=cfg.img_size, is_train=False)

    g = torch.Generator()
    g.manual_seed(seed)

    def make_eval_loader(ds):
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        generator=g,
    )

    return {
        "train": train_loader,
        "val": make_eval_loader(val_ds),
        "id_test": make_eval_loader(id_test_ds),
        "ood_dev": make_eval_loader(ood_dev_ds),
        "ood_ext": make_eval_loader(ood_ext_ds),
        "pranet_ext": make_eval_loader(pranet_ds),
    }
