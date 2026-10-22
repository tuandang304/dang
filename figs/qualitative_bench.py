"""Qualitative predictions on the primary cohorts (seed 42). Regenerate:

    uv run --isolated --no-project --with numpy --with matplotlib --with pillow --with pyarrow --with huggingface_hub \
        python figs/qualitative_bench.py

Frames are drawn at random (fixed generator, not curated): two from each primary cohort among the frames for which every
shown model's seed-42 prediction masks were saved. Predictions are the binary 352x352 masks the benchmark dumped at
threshold 0.5; the printed Dice is the per-image Dice the evaluator logged at native resolution for the same seed.
Polyp-PVT's prediction masks were not retained, so its row is a strip with the per-image Dice it logged for seed 42.
"""

import glob
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from matplotlib import pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from orx_figstyle import PALETTE, WIDE, use_style  # noqa: E402

RUNS = Path("~/.local/share/openresearch").expanduser()
DICE_ONLY = ("Polyp-PVT", "48489f3c")      # masks not retained: show the logged per-image Dice only
MODELS = [("PraNet", "2709562b", "pranet"), ("PraNet-V2", "e733ff21", "pranet_v2"), ("SegDINO", "5fa721d1", "segdino"),
          ("AdaPolyp (ours)", "842016dd", "dang")]
COHORTS = [("CVC-ColonDB", "ColonDB"), ("ETIS-LaribPolypDB", "ETIS"), ("PolypGen", "PolypGen")]
PER_COHORT = 2
SIZE = 352
IMG_RE = re.compile(r"BENCH_PER_IMAGE: (\{.*\})\s*$")


def mask_zip(run, model):
    path = glob.glob(str(RUNS / "local-runs" / f"{run}*" / "kaggle-output" / f"pred_masks_{model}_seed42.zip"))[0]
    z = zipfile.ZipFile(path)
    return z, {n.split("/", 1)[1][:-4]: n for n in z.namelist()}       # uid -> member ("<uid>.png")


def logged_dice(run):
    log = [f for f in glob.glob(str(RUNS / "run-logs" / f"{run}*.log")) if not f.endswith(".supervisor.log")][0]
    out = {}
    for line in open(log, errors="replace"):
        if m := IMG_RE.search(line):
            d = json.loads(m.group(1))
            if d["seed"] == 42:
                out.update({r["u"]: r["dn"] for r in d["recs"]})
    return out


def source_images():
    """uid -> (image bytes, mask bytes) for the primary cohorts, from the same files the benchmark reads."""
    src = {}
    test = zipfile.ZipFile(hf_hub_download("Zyna1121/lightpranet-polyp-data", "TestDataset.zip", repo_type="dataset"))
    for cohort in ("CVC-ColonDB", "ETIS-LaribPolypDB"):
        for n in test.namelist():
            pre = f"TestDataset/{cohort}/images/"
            if n.startswith(pre) and n != pre:
                fn = n[len(pre):]
                src[f"{cohort}::{fn}"] = (test.read(n), test.read(f"TestDataset/{cohort}/masks/{fn}"))
    for split in ("validation", "train"):
        path = hf_hub_download("mzschwartz88/combo_polyp", f"data/{split}-00000-of-00001.parquet", repo_type="dataset")
        for r in pq.read_table(path).to_pylist():
            fn = r["image"].get("path", "unknown.jpg")
            for key in [k for k in WANTED if k.endswith(f"::{split}::{fn}")]:
                src[key] = (r["image"]["bytes"], r["label"]["bytes"])
    return src


def load_rgb(b):
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)) / 255.0


def load_mask(b):
    m = np.asarray(Image.open(io.BytesIO(b)).convert("L").resize((SIZE, SIZE), Image.NEAREST))
    return m > (127 if m.max() > 1 else 0)          # PolypGen masks are stored as 0/1, the others as 0/255


def contour(ax, m, color):
    ax.contour(m.astype(float), levels=[0.5], colors=[color], linewidths=0.7)


def main():
    global WANTED
    use_style()
    zips = {name: mask_zip(run, model) for name, run, model in MODELS}
    dice = {name: logged_dice(run) for name, run, _ in MODELS}
    rng = np.random.default_rng(2026)
    picks = []
    for cohort, short in COHORTS:
        common = sorted(set.intersection(*[{u for u in z[1] if u.startswith(cohort + "::")} for z in zips.values()]))
        picks += [(short, u) for u in rng.choice(common, size=PER_COHORT, replace=False)]
    WANTED = [u for _, u in picks]
    src = source_images()

    rows = ["Input + ground truth"] + [name for name, _, _ in MODELS]
    rows.insert(rows.index("PraNet-V2") + 1, DICE_ONLY[0])
    dice[DICE_ONLY[0]] = logged_dice(DICE_ONLY[1])
    ratios = [0.2 if r == DICE_ONLY[0] else 1.0 for r in rows]
    fig, axes = plt.subplots(len(rows), len(picks), figsize=(WIDE, WIDE * sum(ratios) / len(picks) * 0.93 + 0.2),
                             gridspec_kw={"height_ratios": ratios})
    fig.subplots_adjust(left=0.085, right=0.995, top=0.955, bottom=0.005, wspace=0.03, hspace=0.04)
    for j, (short, uid) in enumerate(picks):
        img = load_rgb(src[uid][0])
        gt = load_mask(src[uid][1])
        for i, row in enumerate(rows):
            ax = axes[i, j]
            if row == DICE_ONLY[0]:
                ax.text(0.5, 0.5, f"Dice {dice[row][uid]:.0f}", transform=ax.transAxes, ha="center", va="center",
                        fontsize=6.5, fontweight="bold", color="black")
                ax.set_facecolor("#f2f2f2")
            else:
                ax.imshow(img, interpolation="bilinear")
            if row == DICE_ONLY[0]:
                pass
            elif i == 0:
                contour(ax, gt, PALETTE["orange"])
            else:
                z, members = zips[row]
                pm = Image.open(io.BytesIO(z.read(members[uid]))).convert("L")
                pred = np.asarray(pm.resize((SIZE, SIZE), Image.NEAREST)) > 127   # SegDINO masks are 256x256
                overlay = np.zeros((SIZE, SIZE, 4))
                overlay[pred] = (0.0, 0.45, 0.70, 0.45)
                ax.imshow(overlay, interpolation="nearest")
                contour(ax, gt, PALETTE["orange"])
                ax.text(0.03, 0.97, f"{dice[row][uid]:.0f}", transform=ax.transAxes, ha="left", va="top",
                        fontsize=6.5, color="white", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.55))
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_xlabel(short, fontsize=7)
                ax.xaxis.set_label_position("top")
            if j == 0:
                if row == DICE_ONLY[0]:
                    ax.set_ylabel("Polyp-PVT\n(no mask)", fontsize=5.5, rotation=0, labelpad=3, ha="right", va="center")
                else:
                    ax.set_ylabel(row.replace(" + ", " +\n"), fontsize=6.5, rotation=90, labelpad=3)
    # a photo grid: a 300-dpi JPEG keeps the paper small (a vector PDF of 24 photos is ~8 MB)
    out = Path(__file__).with_name("qualitative_bench")
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    Image.open(out.with_suffix(".png")).convert("RGB").save(out.with_suffix(".jpg"), quality=90, dpi=(300, 300))
    out.with_suffix(".png").unlink()
    print("wrote", out.with_suffix(".jpg"))


if __name__ == "__main__":
    main()
