"""Figure: predicted masks of the baseline, three SOTA models and DANG on stress-cohort frames.

All models are seed 42, S1 track (40 labelled Kvasir images). Masks come from the pred_masks_seed42.zip
written by each run (threshold 0.5, 352x352); the Dice under each panel is that run's logged per-image
Dice (PER_IMAGE_CHUNK) for the same mask. Frames are drawn at random (seed 1) inside fixed strata, not
curated. Input images and S3 corruptions are rebuilt exactly as in src/dang/data.py.
Regenerate:
  uv run --no-project --with matplotlib --with numpy --with pillow --with pyarrow --with huggingface_hub \
      python figs/qualitative_predictions.py
"""

import ast
import io
import json
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from orx_figstyle import WIDE, save, use_style  # noqa: E402

ROOT = HERE.parent
LOGS = Path("/home/tuan/.local/share/openresearch/run-logs")
MASKS = Path("/home/tuan/.local/share/openresearch/files/dang/dang-med-ablation/pred_masks")
SIZE = 352
GT = "#FFE500"
PRED = np.array([0.34, 0.71, 0.91])  # sky blue fill for the prediction

# column label -> run id (seed-42 run that dumped pred_masks_seed42.zip)
ARMS = [
    ("ConvNeXt-T+FPN", "7713fc08-3533-4fdd-96f7-8a87637bf7b0"),
    ("PraNet", "e00b4190-193d-4a54-8af5-f11727d5866c"),
    ("Polyp-PVT", "6993f62a-9521-483e-99c1-708d599557ad"),
    ("EMCAD", "90a8f6ab-b3e7-4f94-b4d4-c79162b6f63c"),
    ("DANG", "58097f2d-0dc9-4dc2-98c8-e6699805cce2"),
]
MODES = ["Adverse_NBI_Spectral", "Adverse_LowLight_Lumen", "Adverse_Specular_Glare_Blur", "Adverse_Combined_Severe"]
# (row label, cohort tag in logs/zip, uid filter)
STRATA = [
    ("Tiny", "s1_s2_tiny_camo", lambda u: "S2_Tiny" in u),
    ("Camouflaged", "s1_s2_tiny_camo", lambda u: "S2_Camouflaged" in u),
    ("Adverse: NBI", "s1_s3_adverse", lambda u: "::Adverse_NBI_Spectral::" in u),
    ("Adverse: low light", "s1_s3_adverse", lambda u: "::Adverse_LowLight_Lumen::" in u),
    ("Adverse: combined", "s1_s3_adverse", lambda u: "::Adverse_Combined_Severe::" in u),
    ("ETIS", "s1_pranet_ext", lambda u: u.startswith("Ext_ETIS")),
]
ZIP_TAG = {"s1_s2_tiny_camo": "s1/s2_tiny_camo", "s1_s3_adverse": "s1/s3_adverse", "s1_pranet_ext": "s1/pranet_ext"}


def data_helpers():
    src = (ROOT / "src/dang/data.py").read_text() if (ROOT / "src/dang/data.py").exists() else None
    ns = {"np": np, "Image": Image, "ImageFilter": ImageFilter, "io": io}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in ("apply_adverse_endoscopy_corruption", "decode_binary_mask_np"):
            exec(ast.get_source_segment(src, node), ns)
    return ns["apply_adverse_endoscopy_corruption"], ns["decode_binary_mask_np"]


def per_image(run_id, cohort):
    dice, order = {}, []
    for line in open(LOGS / f"{run_id}.log", errors="ignore"):
        if line.startswith("PER_IMAGE_CHUNK:"):
            rec = json.loads(line.split(": ", 1)[1])
            if rec["seed"] == 42 and rec["cohort"] == cohort:
                for r in rec["recs"]:
                    dice[r["u"]] = r["d"]
                    order.append(r["u"])
    return dice, order


def main():
    use_style()
    corrupt, decode = data_helpers()
    dev = {r["image"]["path"]: r for r in pq.read_table(hf_hub_download(
        "mzschwartz88/combo_polyp", "data/validation-00000-of-00001.parquet", repo_type="dataset")).to_pylist()}
    pz = zipfile.ZipFile(hf_hub_download("Zyna1121/lightpranet-polyp-data", "TestDataset.zip", repo_type="dataset"))
    folder = {"Ext_ETIS_LaribPolypDB": "ETIS-LaribPolypDB", "Ext_CVC_ColonDB": "CVC-ColonDB",
              "Ext_CVC_300": "CVC-300", "Ext_CVC_ClinicDB": "CVC-ClinicDB"}
    zips = {name: zipfile.ZipFile(MASKS / f"{rid}.zip") for name, rid in ARMS}
    logs = {(name, c): per_image(rid, c) for name, rid in ARMS for c in ZIP_TAG}

    def load(uid):
        parts = uid.split("::")
        if parts[0] == "S3":  # S3::<mode>::<dev filename>; seed_idx = position in the S3 list
            order = logs[(ARMS[0][0], "s1_s3_adverse")][1]
            idx = order.index(uid)
            r = dev[parts[2]]
            return corrupt(r["image"]["bytes"], mode=parts[1], seed_idx=idx), decode(r["label"]["bytes"])
        if parts[0] == "S2":
            parts = parts[2:]
        if parts[0] == "PolypGenDev":
            r = dev[parts[2]]
            return r["image"]["bytes"], decode(r["label"]["bytes"])
        f = folder[parts[0]]
        return pz.read(f"TestDataset/{f}/images/{parts[1]}"), decode(pz.read(f"TestDataset/{f}/masks/{parts[1]}"))

    rng = np.random.default_rng(1)
    rows = []
    for label, cohort, keep in STRATA:
        pool = [u for u in logs[(ARMS[0][0], cohort)][1] if keep(u)]
        rows.append((label, cohort, str(rng.choice(pool))))

    ncol = len(ARMS) + 1
    fig, axes = plt.subplots(len(rows), ncol, figsize=(WIDE, 1.13 * len(rows) + 0.25))
    fig.subplots_adjust(left=0.035, right=0.998, top=0.965, bottom=0.01, wspace=0.03, hspace=0.24)
    for i, (label, cohort, uid) in enumerate(rows):
        img_b, gt = load(uid)
        img = np.asarray(Image.open(io.BytesIO(img_b)).convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)) / 255.0
        gt = np.asarray(Image.fromarray(gt.astype(np.uint8) * 255).resize((SIZE, SIZE), Image.Resampling.NEAREST)) > 127
        for j in range(ncol):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if j == 0:
                ax.imshow(img)
                ax.contour(gt.astype(float), levels=[0.5], colors=[GT], linewidths=0.8)
                ax.text(-0.06, 0.5, label, transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=6.5)
                cap = "Input + GT"
            else:
                name, _ = ARMS[j - 1]
                key = f"{ZIP_TAG[cohort]}/{uid.replace('/', '__')}.png"
                pred = np.asarray(Image.open(io.BytesIO(zips[name].read(key)))) > 127
                over = img * 0.6
                over[pred] = 0.5 * img[pred] + 0.5 * PRED
                ax.imshow(np.clip(over, 0, 1))
                ax.contour(gt.astype(float), levels=[0.5], colors=[GT], linewidths=0.8)
                d = logs[(name, cohort)][0][uid]
                ax.text(0.03, 0.04, f"{d:.0f}", transform=ax.transAxes, fontsize=6.5, color="white",
                        fontweight="bold", va="bottom", ha="left",
                        bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.6))
                cap = name
            if i == 0:
                ax.text(0.5, 1.04, cap, transform=ax.transAxes, ha="center", va="bottom", fontsize=6.5)
    save(fig, str(HERE / "qualitative_predictions"), formats=("pdf",))
    print("rows:", rows)


if __name__ == "__main__":
    main()
