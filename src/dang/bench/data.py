"""Benchmark cohorts: the official PraNet split plus two hard, filtered OOD test sets (PolypGen, PolypDB).

Duplicate filter (checked visually before freezing): two frames are duplicates when their md5 matches, or when they are
the same picture of the same lesion -- Pearson correlation >= 0.97 between 32x32 grayscale thumbnails taken after
cropping the dark endoscope border, AND mask IoU >= 0.8 at 64x64. A 64-bit dHash was tried first and rejected: on
endoscopy frames the black octagonal border dominates it and it flagged unrelated polyps as duplicates. Every hard-set
frame is checked against all 1,450 training images and against the frames already kept from its own set
(deterministic order, first occurrence kept). Frames with an empty mask or a mask/image size mismatch are dropped.
The five standard test sets are kept exactly as published so numbers stay comparable with the literature.
"""

import hashlib
import json
import os
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import io
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

from dang.data import classify_polypgen_official, crop_fov_pil, decode_binary_mask_np

STD_TEST_SETS = ["Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "CVC-300"]
IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

POLYPDB_URL = "https://osf.io/download/jg32z/"          # https://osf.io/pr7ms/ (Jha et al., arXiv:2409.00045)
POLYPDB_BYTES = 1392073356
# Centre-wise groups kept (domain tag -> folder). Simula/WLI (2,558 Norwegian WLI frames from HyperKvasir, the
# hospital Kvasir-SEG comes from) is dropped as the easy, leakage-prone set.
POLYPDB_GROUPS = {
    "VN_WLI": "BKAI/WLI",            # Hanoi (HMU + IGH), Fujinon
    "VN_BLI": "BKAI/BLI",
    "VN_FICE": "BKAI/FICE",
    "VN_LCI": "BKAI/LCI",
    "NBI": "Simula/NBI",             # narrow-band imaging (Norway + Sweden per the paper)
    "SE_WLI": "Karolinska/WLI",      # Stockholm, Olympus
}
POLYPDB_DROPPED = {"NO_WLI": "Simula/WLI"}
DUP_CORR = 0.97
DUP_MASK_IOU = 0.8


def _sample(img_b: bytes, mask_b: bytes, name: str, domain: str, uid: str) -> dict:
    return {"img_bytes": img_b, "mask_bytes": mask_b, "name": name, "domain": domain, "uid": uid,
            "seq_id": uid, "c4_seq_in_dev": False, "min_kvasir_dhash": 99}


def _read_zip_pairs(zf: zipfile.ZipFile, img_dir: str, mask_dir: str, domain: str, uid_prefix: str) -> list[dict]:
    names = set(zf.namelist())
    files = sorted(n[len(img_dir):] for n in names
                   if n.startswith(img_dir) and n.lower().endswith(IMG_EXT) and "/" not in n[len(img_dir):])
    out = []
    for fn in files:
        assert mask_dir + fn in names, f"missing mask {mask_dir + fn}"
        out.append(_sample(zf.read(img_dir + fn), zf.read(mask_dir + fn), fn, domain, f"{uid_prefix}::{fn}"))
    return out


def _signature(s: dict) -> tuple[str, np.ndarray, np.ndarray]:
    img = Image.open(io.BytesIO(s["img_bytes"])).convert("RGB")
    v = np.asarray(crop_fov_pil(img).convert("L").resize((32, 32), Image.Resampling.BILINEAR), np.float32).ravel()
    v = v - v.mean()
    v = v / (np.linalg.norm(v) + 1e-6)
    m = Image.fromarray(decode_binary_mask_np(s["mask_bytes"]).astype(np.uint8) * 255).resize((64, 64), Image.Resampling.NEAREST)
    return hashlib.md5(s["img_bytes"]).hexdigest(), v, np.asarray(m) > 127


class _DupIndex:
    def __init__(self):
        self.md5, self.vecs, self.masks = set(), [], []

    def add(self, sig):
        self.md5.add(sig[0])
        self.vecs.append(sig[1])
        self.masks.append(sig[2])

    def match(self, sig) -> tuple[bool, float]:
        """(is duplicate, highest thumbnail correlation)."""
        if sig[0] in self.md5:
            return True, 1.0
        if not self.vecs:
            return False, -1.0
        corr = np.stack(self.vecs) @ sig[1]
        for j in np.where(corr >= DUP_CORR)[0]:
            inter = np.logical_and(self.masks[j], sig[2]).sum()
            union = np.logical_or(self.masks[j], sig[2]).sum()
            if union and inter / union >= DUP_MASK_IOU:
                return True, float(corr[j])
        return False, float(corr.max())


def _filter_hard(rows: list[dict], train_idx: _DupIndex, cohort: str) -> tuple[list[dict], dict]:
    """rows: dicts from _sample (deterministic order). Returns kept samples and drop statistics."""
    stats = Counter(raw=len(rows))
    kept, own, max_corr_train = [], _DupIndex(), []
    for s in rows:
        img = Image.open(io.BytesIO(s["img_bytes"]))
        msk = Image.open(io.BytesIO(s["mask_bytes"]))
        if img.size != msk.size:
            stats["size_mismatch_dropped"] += 1
            continue
        if not decode_binary_mask_np(s["mask_bytes"]).any():
            stats["empty_mask_dropped"] += 1
            continue
        sig = _signature(s)
        dup, c_train = train_idx.match(sig)
        if dup:
            stats["duplicate_of_train_dropped"] += 1
            continue
        if own.match(sig)[0]:
            stats["duplicate_within_set_dropped"] += 1
            continue
        own.add(sig)
        max_corr_train.append(c_train)
        s["min_kvasir_dhash"] = 99
        s["max_corr_to_train"] = round(c_train, 4)
        s["md5"] = sig[0]
        kept.append(s)
    stats["kept"] = len(kept)
    stats["max_corr_to_train_of_kept"] = round(float(max(max_corr_train)), 4) if max_corr_train else None
    stats["per_domain_kept"] = dict(sorted(Counter(s["domain"] for s in kept).items()))
    print(f"[bench data] {cohort}: " + json.dumps(stats), flush=True)
    return kept, dict(stats)


def _download_polypdb() -> Path:
    cache = Path(os.environ.get("BENCH_DATA_CACHE", Path.home() / ".cache" / "dang_bench"))
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / "PolypDB.zip"
    for attempt in range(4):
        if dst.exists() and dst.stat().st_size == POLYPDB_BYTES:
            return dst
        t0 = time.time()
        tmp = dst.with_suffix(".part")
        try:
            urllib.request.urlretrieve(POLYPDB_URL, tmp)
            tmp.rename(dst)
        except Exception as e:  # network hiccup: retry
            print(f"[bench data] PolypDB download attempt {attempt + 1} failed: {e!r}", flush=True)
            continue
        print(f"[bench data] PolypDB downloaded {dst.stat().st_size} bytes in {time.time() - t0:.0f}s", flush=True)
    assert dst.exists() and dst.stat().st_size == POLYPDB_BYTES, "PolypDB.zip missing or truncated"
    return dst


def _polypgen_rows() -> list[dict]:
    rows = []
    for split in ("validation", "train"):
        path = hf_hub_download("mzschwartz88/combo_polyp", f"data/{split}-00000-of-00001.parquet", repo_type="dataset")
        for r in pq.read_table(path).to_pylist():
            fn = r["image"].get("path", "unknown.jpg")
            dom = classify_polypgen_official(fn)
            assert dom != "Unknown", f"unexpected filename in combo_polyp/{split}: {fn}"
            if not dom.startswith("OOD_"):
                continue                                 # Kvasir-SEG rows mixed into combo_polyp
            rows.append(_sample(r["image"]["bytes"], r["label"]["bytes"], fn, dom.replace("OOD_", ""),
                                f"PolypGen::{dom}::{split}::{fn}"))
    return rows


def _polypdb_rows() -> list[dict]:
    zf = zipfile.ZipFile(_download_polypdb())
    rows = []
    root = "PolypDB/PolypDB_center_wise/"
    for dom, folder in POLYPDB_GROUPS.items():
        img_dir, mask_dir = f"{root}{folder}/images/", f"{root}{folder}/masks/"
        names = set(zf.namelist())
        imgs = {os.path.splitext(n[len(img_dir):])[0]: n for n in names
                if n.startswith(img_dir) and n.lower().endswith(IMG_EXT)}
        masks = {os.path.splitext(n[len(mask_dir):])[0]: n for n in names
                 if n.startswith(mask_dir) and n.lower().endswith(IMG_EXT)}
        assert set(imgs) == set(masks), f"unpaired files in {folder}"
        for stem in sorted(imgs):
            rows.append(_sample(zf.read(imgs[stem]), zf.read(masks[stem]), stem, dom, f"PolypDB::{dom}::{stem}"))
    return rows


def prepare_bench(include_polypdb: bool = True) -> dict:
    t0 = time.time()
    train_zip = zipfile.ZipFile(hf_hub_download("Zyna1121/lightpranet-polyp-data", "TrainDataset.zip", repo_type="dataset"))
    test_zip = zipfile.ZipFile(hf_hub_download("Zyna1121/lightpranet-polyp-data", "TestDataset.zip", repo_type="dataset"))
    train = _read_zip_pairs(train_zip, "TrainDataset/image/", "TrainDataset/masks/", "Train_PraNet1450", "Train")
    cohorts = {d: _read_zip_pairs(test_zip, f"TestDataset/{d}/images/", f"TestDataset/{d}/masks/", d, d)
               for d in STD_TEST_SETS}
    counts = {"train": len(train), **{d: len(v) for d, v in cohorts.items()}}
    assert counts == {"train": 1450, "Kvasir": 100, "CVC-ClinicDB": 62, "CVC-ColonDB": 380,
                      "ETIS-LaribPolypDB": 196, "CVC-300": 60}, counts

    train_idx = _DupIndex()
    for s in train:
        train_idx.add(_signature(s))

    audit = {"standard_counts": counts, "dup_rule": {"thumb_corr": DUP_CORR, "mask_iou": DUP_MASK_IOU}}
    cohorts["PolypGen"], audit["PolypGen"] = _filter_hard(_polypgen_rows(), train_idx, "PolypGen")
    if include_polypdb:
        cohorts["PolypDB"], audit["PolypDB"] = _filter_hard(_polypdb_rows(), train_idx, "PolypDB")
        audit["PolypDB_dropped_groups"] = POLYPDB_DROPPED

    manifest = {c: sorted(s["uid"] for s in v) for c, v in cohorts.items()}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    audit["manifest_sha256_16"] = digest
    audit["prep_seconds"] = round(time.time() - t0, 1)
    uid_lookup = {s["uid"]: s for v in cohorts.values() for s in v}
    return {"train": train, "cohorts": cohorts, "uid_lookup": uid_lookup, "audit": audit, "manifest": manifest}


def lesion_size_breakdown(samples: list[dict]) -> dict:
    """Foreground-area quartiles per cohort, for the audit printout."""
    areas = [float(decode_binary_mask_np(s["mask_bytes"]).mean() * 100.0) for s in samples]
    q = np.percentile(areas, [5, 25, 50, 75]) if areas else [0, 0, 0, 0]
    return {"p5": round(q[0], 2), "p25": round(q[1], 2), "median": round(q[2], 2), "p75": round(q[3], 2),
            "lt1pct": int(sum(a < 1.0 for a in areas))}


def per_domain_counts(samples: list[dict]) -> dict:
    c = defaultdict(int)
    for s in samples:
        c[s["domain"]] += 1
    return dict(sorted(c.items()))
