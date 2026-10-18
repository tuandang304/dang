"""Figure: real examples from the S3 (synthetic adverse) and S2 (natural tiny/camouflaged) stress cohorts.

Images come from the same public sources that src/dang/data.py downloads; the S3 corruption is the
exact `apply_adverse_endoscopy_corruption` from src/dang/data.py (extracted with ast, not re-implemented).
Dice annotations are 5-seed means (seeds 42-46, S1 5% few-shot track) read from PER_IMAGE_CHUNK /
SEED_SUMMARY_JSON logs via figs/seed_metrics.json and the run logs.
Regenerate (after figs/aggregate_seeds.py has written figs/seed_metrics.json):
  uv run --no-project --with matplotlib --with numpy --with pillow --with pyarrow --with huggingface_hub \
      python figs/stress_examples.py
"""

import ast
import io
import json
import sys
import zipfile
from collections import defaultdict
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
SEEDS = json.loads((HERE / "seed_metrics.json").read_text())
MODES = ["Adverse_NBI_Spectral", "Adverse_LowLight_Lumen", "Adverse_Specular_Glare_Blur", "Adverse_Combined_Severe"]
MODE_LABEL = {
    "Adverse_NBI_Spectral": "NBI spectral shift",
    "Adverse_LowLight_Lumen": "Low-light lumen",
    "Adverse_Specular_Glare_Blur": "Glare + defocus blur",
    "Adverse_Combined_Severe": "Combined severe",
}
GT = "#FFE500"  # ground-truth contour: saturated yellow reads on red/pink mucosa and on NBI green
SIZE = 352


def load_data_helpers():
    src = (ROOT / "src/dang/data.py").read_text()
    ns = {"np": np, "Image": Image, "ImageFilter": ImageFilter, "io": io}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in (
            "apply_adverse_endoscopy_corruption",
            "decode_binary_mask_np",
        ):
            exec(ast.get_source_segment(src, node), ns)
    return ns["apply_adverse_endoscopy_corruption"], ns["decode_binary_mask_np"]


def per_image(arm_runs, cohort):
    """uid -> list of (seed, dice) and the record order of the first seed seen."""
    vals, order, area = defaultdict(list), [], {}
    for rid in arm_runs:
        for line in open(LOGS / f"{rid}.log"):
            if not line.startswith("PER_IMAGE_CHUNK:"):
                continue
            rec = json.loads(line.split(": ", 1)[1])
            if rec["cohort"] != cohort:
                continue
            for r in rec["recs"]:
                if rec["seed"] == 42:
                    order.append(r["u"])
                vals[r["u"]].append(r["d"])
                area[r["u"]] = r["a"]
    return {u: float(np.mean(v)) for u, v in vals.items()}, order, area


def resize_pair(img_b, mask):
    im = Image.open(io.BytesIO(img_b)).convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    mk = Image.fromarray(mask.astype(np.uint8) * 255).resize((SIZE, SIZE), Image.Resampling.NEAREST)
    return np.asarray(im), np.asarray(mk) > 127


def show(ax, img, mask, top, bottom):
    ax.imshow(img, interpolation="lanczos")
    ax.contour(mask.astype(float), levels=[0.5], colors=[GT], linewidths=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(0.5, 1.03, top, transform=ax.transAxes, ha="center", va="bottom", fontsize=6.5)
    ax.text(0.5, -0.04, bottom, transform=ax.transAxes, ha="center", va="top", fontsize=6, linespacing=1.25)


def run_ids(arm):
    return sorted({SEEDS[arm][s]["run"] for s in SEEDS[arm]})


def main():
    use_style()
    corrupt, decode = load_data_helpers()

    dev_pq = hf_hub_download("mzschwartz88/combo_polyp", "data/validation-00000-of-00001.parquet", repo_type="dataset")
    dev = {r["image"]["path"]: r for r in pq.read_table(dev_pq).to_pylist()}
    zf = zipfile.ZipFile(hf_hub_download("Zyna1121/lightpranet-polyp-data", "TestDataset.zip", repo_type="dataset"))
    pranet_dir = {"Ext_ETIS_LaribPolypDB": "ETIS-LaribPolypDB", "Ext_CVC_ColonDB": "CVC-ColonDB"}

    # ---- Row 1: one PolypGen-Dev frame under the four S3 regimes --------------------------------------
    _, s3_order, s3_area = per_image(run_ids("baseline"), "s1_s3_adverse")
    # The S3 list is built as adverse_modes[idx % 4] over PolypGen-Dev in order; check the log agrees,
    # so `idx` below is the exact seed_idx the benchmark used for that frame.
    assert all(u.split("::")[1] == MODES[i % 4] for i, u in enumerate(s3_order)), "S3 order mismatch"
    # Illustrative frame: first Dev frame (benchmark order) with a clearly visible 6-12% lesion.
    idx, uid = next((i, u) for i, u in enumerate(s3_order) if 6.0 <= s3_area[u] <= 12.0)
    name = uid.split("::", 2)[2]
    row = dev[name]
    mask = decode(row["label"]["bytes"])

    fig, axes = plt.subplots(2, 5, figsize=(WIDE, 3.55))
    fig.subplots_adjust(left=0.035, right=0.995, top=0.94, bottom=0.075, wspace=0.06, hspace=0.42)

    img, mk = resize_pair(row["image"]["bytes"], mask)
    show(axes[0, 0], img, mk, "Clean source frame", f"PolypGen-Dev, lesion {s3_area[uid]:.1f}% of FOV")
    for j, mode in enumerate(MODES, start=1):
        img, mk = resize_pair(corrupt(row["image"]["bytes"], mode=mode, seed_idx=idx), mask)
        b = np.mean([SEEDS["baseline"][s][f"s1s3_{mode}"] for s in SEEDS["baseline"]])
        f = np.mean([SEEDS["flagship"][s][f"s1s3_{mode}"] for s in SEEDS["flagship"]])
        show(axes[0, j], img, mk, MODE_LABEL[mode], f"family mDice  B {b:.1f} | F {f:.1f}")

    # ---- Row 2: natural S2 lesions, sampled at random (not curated) --------------------------------
    b_img, s2_order, s2_area = per_image(run_ids("baseline"), "s1_s2_tiny_camo")
    f_img, _, _ = per_image(run_ids("flagship"), "s1_s2_tiny_camo")
    rng = np.random.default_rng(0)
    tiny = [u for u in s2_order if "S2_Tiny" in u]
    camo = [u for u in s2_order if "S2_Camouflaged" in u]
    picks = list(rng.choice(tiny, 3, replace=False)) + list(rng.choice(camo, 2, replace=False))
    for j, u in enumerate(picks):
        parts = u.split("::")
        kind = "Tiny (<1.5%)" if "Tiny" in parts[1] else "Camouflaged"
        if parts[2] == "PolypGenDev":
            r = dev[parts[4]]
            img_b, mask = r["image"]["bytes"], decode(r["label"]["bytes"])
            src = "PolypGen " + parts[3].split("_")[1]
        else:
            folder = pranet_dir[parts[2]]
            img_b = zf.read(f"TestDataset/{folder}/images/{parts[3]}")
            mask = decode(zf.read(f"TestDataset/{folder}/masks/{parts[3]}"))
            src = folder.replace("-LaribPolypDB", "-Larib")
        img, mk = resize_pair(img_b, mask)
        show(axes[1, j], img, mk, f"{kind}: {src}",
             f"area {s2_area[u]:.2f}%   Dice  B {b_img[u]:.1f} | F {f_img[u]:.1f}")

    fig.text(0.012, 0.735, "S3 adverse (synthetic)", rotation=90, ha="center", va="center", fontsize=7)
    fig.text(0.012, 0.27, "S2 tiny / camouflaged", rotation=90, ha="center", va="center", fontsize=7)
    fig.text(0.004, 0.985, "(a)", fontsize=8, fontweight="bold", va="top")
    fig.text(0.004, 0.50, "(b)", fontsize=8, fontweight="bold", va="top")
    # Photographs: PDF only; an SVG of this grid is several megabytes.
    save(fig, str(HERE / "stress_examples"), formats=("pdf",))
    print("S3 frame:", uid, "seed_idx", idx, "| S2 picks:", picks)


if __name__ == "__main__":
    main()
