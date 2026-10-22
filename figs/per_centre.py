"""Dice per centre of PolypGen and per group of PolypDB, five seeds. Regenerate (after figs/aggregate_bench.py):

    uv run --isolated --no-project --with numpy --with matplotlib python figs/per_centre.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from orx_figstyle import PALETTE, WIDE, family, figure_grid, mean_ci, panel_labels, save, use_style  # noqa: E402

HERE = Path(__file__).parent
G = json.loads((HERE / "bench_metrics.json").read_text())["groups"]
MODELS = ["SegDINO", "PraNet", "PraNet-V2", "Polyp-PVT", "DANG"]
COLORS = {"SegDINO": PALETTE["green"]} | dict(zip(MODELS[1:4], family("blue", 3))) | {"DANG": PALETTE["red"]}
LABELS = {"DANG": "AdaPolyp (ours)"}
DOMAINS = {
    "PolypGen": [("C1_Paris", "C1 Paris"), ("C2_Padova", "C2 Padova"), ("C3_Aviano", "C3 Aviano"),
                 ("C4_Oslo", "C4 Oslo"), ("C5_Oxford", "C5 Oxford"), ("C6_Alexandria", "C6 Alexandria")],
    "PolypDB": [("VN_WLI", "WLI, Vietnam"), ("SE_WLI", "WLI, Sweden"), ("VN_BLI", "BLI, Vietnam"),
                ("VN_FICE", "FICE, Vietnam"), ("VN_LCI", "LCI, Vietnam"), ("NBI", "NBI, Norway/Sweden")],
}


def main():
    use_style()
    fig, axes = figure_grid(1, 2, width=WIDE, ratio=0.36, sharey=False)
    offsets = np.linspace(-0.27, 0.27, len(MODELS))
    for ax, (cohort, doms) in zip(axes, DOMAINS.items()):
        for k, (m, off) in enumerate(zip(MODELS, offsets)):
            pd = G.get(m, {}).get("cohorts", {}).get(cohort, {}).get("per_domain", {})
            for i, (d, _) in enumerate(doms):
                if d not in pd:
                    continue
                mean, lo, hi = mean_ci([[v] for v in pd[d]["seeds"]])
                y = i + off
                ax.errorbar(mean[0], y, xerr=[[mean[0] - lo[0]], [hi[0] - mean[0]]], fmt="o", ms=3.2,
                            color=COLORS[m], elinewidth=0.9, capsize=0,
                            label=LABELS.get(m, m) if i == 0 else None, zorder=3 if m == "DANG" else 2)
        names = [f"{lbl} ({G['PraNet']['cohorts'][cohort]['per_domain'][d]['n_images']})" for d, lbl in doms]
        ax.set_yticks(range(len(doms)), names)
        ax.set_ylim(len(doms) - 0.5, -0.5)
        for i in range(len(doms)):
            if i % 2:
                ax.axhspan(i - 0.5, i + 0.5, color="black", alpha=0.035, lw=0, zorder=0)
        ax.set_xlabel(f"Dice on {cohort} (%)")
        ax.grid(axis="x", alpha=0.3)
        ax.tick_params(axis="y", length=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(MODELS), frameon=False)
    panel_labels(axes)
    save(fig, str(HERE / "per_centre"))


if __name__ == "__main__":
    main()
