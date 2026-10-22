"""Primary-cohort Dice against T4 latency. Regenerate (after scripts/aggregate_bench.py):

    uv run --locked --with matplotlib python scripts/accuracy_latency.py

y: per seed, the mean Dice over ColonDB, ETIS and PolypGen; point = mean over 5 seeds, bar = 95% t-interval.
x: batch-1 latency with FP16 weights in a CUDA graph on one NVIDIA T4 (all models timed in one process).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from figstyle import COLUMN, PALETTE, family, figure, mean_ci, save, use_style  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "results" / "figures"
M = json.loads((ROOT / "results" / "bench_metrics.json").read_text())
G, D = M["groups"], M["deploy"]
PRIMARY = ["CVC-ColonDB", "ETIS-LaribPolypDB", "PolypGen"]
POINTS = [  # group, label, deploy name, colour
    ("SegDINO", "SegDINO (256$\\times$256)", "SegDINO (DINOv3 ViT-S/16 + DPT head, 256x256)", PALETTE["green"]),
    ("PraNet", "PraNet", "PraNet (Res2Net-50)", family("blue", 3)[0]),
    ("PraNet-V2", "PraNet-V2", "PraNet-V2 (PVTv2-B2)", family("blue", 3)[1]),
    ("Polyp-PVT", "Polyp-PVT", "Polyp-PVT (PVTv2-B2)", family("blue", 3)[2]),
    ("V12-b", "fg/bg + PASSE", "V12-b: token pyramid, fg/bg decoder, PASSE", family("red", 3)[0]),
    ("R2-c", "query", "R2-c: token pyramid, fg/bg query decoder", family("red", 3)[1]),
    ("DANG", "AdaPolyp (ours)", "DANG (R2-b): token pyramid, SAD", PALETTE["red"]),
]

OFFSET = {"SegDINO": (6, 0), "PraNet": (6, -2), "PraNet-V2": (6, -4), "Polyp-PVT": (6, -2), "V12-b": (2, -9),
          "R2-c": (1, 8), "DANG": (-6, 0)}


def main():
    use_style()
    fig, ax = figure(width=COLUMN, ratio=0.72)
    for g, label, dep, color in POINTS:
        if g not in G or dep not in D or D[dep].get("graph_half_bs1_ms") is None:
            continue
        per_seed = np.mean([G[g]["cohorts"][c]["dice"]["seeds"] for c in PRIMARY], axis=0)
        mean, lo, hi = mean_ci([[v] for v in per_seed])
        x = D[dep]["graph_half_bs1_ms"]
        ax.errorbar(x, mean[0], yerr=[[mean[0] - lo[0]], [hi[0] - mean[0]]], fmt="o", ms=4, color=color,
                    elinewidth=0.9, capsize=0, zorder=3)
        dx, dy = OFFSET[g]
        ax.annotate(label, (x, mean[0]), xytext=(dx, dy), textcoords="offset points", fontsize=6.5,
                    ha="right" if dx < 0 else ("center" if g == "V12-b" else "left"), va="center", color="black")
    ax.set_xlabel("T4 latency, batch 1, FP16 CUDA graph (ms)")
    ax.set_ylabel("Mean primary-cohort Dice (%)")
    ax.grid(alpha=0.3)
    ax.set_ylim(top=86.5)                       # room for the labels above the top cluster
    save(fig, str(HERE / "accuracy_latency"))


if __name__ == "__main__":
    main()
