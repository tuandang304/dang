"""Generate static publication figures for paper.tex and interactive HTML chart for artifacts.
Regenerate: uv run --no-project --with matplotlib --with numpy python figs/generate_paper_figures.py
"""

import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from orx_figstyle import COLUMN, MUTED, PALETTE, TEXT, WIDE, family, figure, figure_grid, panel_labels, save, use_style


RUN_LOGS = {
    "R8a": "/home/tuan/.local/share/openresearch/run-logs/4ff780b9-76c8-4eca-a145-833d159a6636.log",
    "R8b": "/home/tuan/.local/share/openresearch/run-logs/eed6d338-3868-4889-91ba-c2ea667d75d2.log",
    "R8c": "/home/tuan/.local/share/openresearch/run-logs/02df9e4a-2d58-4124-92d1-bb28e5b4092c.log",
    "R8d": "/home/tuan/.local/share/openresearch/run-logs/d47b9a4b-f898-4ae0-99c5-7479431e6565.log",
    "R8e": "/home/tuan/.local/share/openresearch/run-logs/a8ecd157-acf4-4cfb-81a3-86236999eef0.log",
    "R9a": "/home/tuan/.local/share/openresearch/run-logs/f328e771-e3d0-4ae5-8a74-f4285e5886fa.log",
    "R9b": "/home/tuan/.local/share/openresearch/run-logs/e770b25c-18cd-43d0-b924-66c4e2c2819d.log",
    "R9c": "/home/tuan/.local/share/openresearch/run-logs/f676224f-598d-4143-8e4f-f04373be94de.log",
    "R9d": "/home/tuan/.local/share/openresearch/run-logs/5356b752-c8f3-47f4-9224-bb6c466b356b.log",
    "R9e": "/home/tuan/.local/share/openresearch/run-logs/db1786be-b7b7-4c1c-9a4b-c87bda43501d.log",
}


def parse_all_logs() -> dict:
    out = {}
    for k, path in RUN_LOGS.items():
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                if line.startswith("SEED_SUMMARY_JSON:42: "):
                    out[k] = json.loads(line.split("SEED_SUMMARY_JSON:42: ", 1)[1])
                elif line.startswith("METRICS_JSON_SINGLE_LINE: "):
                    meta = json.loads(line.split("METRICS_JSON_SINGLE_LINE: ", 1)[1])
                    if k in out:
                        out[k]["efficiency"] = meta["efficiency"]
    return out


def make_stress_benchmarks_fig(data: dict):
    """Figure 2: Grouped bar chart across 6 hard OOD & clinical stress benchmarks (R9a-R9e + R8e)."""
    use_style()
    methods = [
        ("ConvNeXt-T + FPN (R9a)", "base", "R9a"),
        ("Dual-Stream SafeConcat (R9b)", "prior", "R9b"),
        ("DANG-v4.1 ProtoDPT+PASSE (R9c)", "ours", "R9c"),
        ("DANG-Med v4.1 Zero-Shot (R9e)", "ours", "R9e"),
        ("DANG-Med v4 Adv-Trained (R8e)", "ours", "R8e"),
    ]
    benchmarks = [
        ("S1xS2\nTiny/Camo", lambda d: d["s1_s2_tiny_camo_micro"]["dice"]),
        ("S1xS3\nAdverse", lambda d: d["s1_s3_adverse_macro"]["dice"]),
        ("S1 (5%)\nPraNet-Ext", lambda d: d["s1_pranet_external_macro"]["dice"]),
        ("Full\nS3 Adverse", lambda d: d["s3_adverse_macro"]["dice"]),
        ("Full\nS2 Tiny/Camo", lambda d: d["s2_tiny_camo_micro"]["dice"]),
        ("Full\nETIS-Larib", lambda d: d["pranet_external_per_dataset"]["Ext_ETIS_LaribPolypDB"]["dice"]),
    ]

    ours_shades = family("blue", 3)
    colors = {
        "ConvNeXt-T + FPN (R9a)": MUTED,
        "Dual-Stream SafeConcat (R9b)": PALETTE["orange"],
        "DANG-v4.1 ProtoDPT+PASSE (R9c)": ours_shades[0],
        "DANG-Med v4.1 Zero-Shot (R9e)": ours_shades[1],
        "DANG-Med v4 Adv-Trained (R8e)": ours_shades[2],
    }

    fig, ax = figure(width=WIDE, ratio=0.34)
    x = np.arange(len(benchmarks))
    width = 0.85 / len(methods)

    for i, (m_label, _, m_key) in enumerate(methods):
        offset = (i - (len(methods) - 1) / 2.0) * width
        vals = [fn(data[m_key]) for _, fn in benchmarks]
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=m_label,
            color=colors[m_label],
            edgecolor="black",
            linewidth=0.35,
        )
        ax.bar_label(bars, fmt="%.1f", fontsize=5.0, padding=1.2)

    ax.set_xticks(x, [b[0] for b in benchmarks])
    ax.set_ylabel("mDice (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.tick_params(axis="x", length=0)
    ax.legend(
        ncol=3,
        loc="upper left",
        fontsize=6.0,
        frameon=True,
        framealpha=0.95,
        edgecolor="black",
        borderpad=0.4,
    )
    save(fig, "figs/stress_benchmarks")


def make_ablation_gains_fig(data: dict):
    """Figure 3: Two-panel comparison of (a) Boundary F1 (%) and (b) HD95 distance error (px) on S1/S2/S3/ETIS."""
    use_style()
    fig, axes = figure_grid(1, 2, width=WIDE, ratio=0.35, sharey=False)
    ax_bf1, ax_hd = axes[0], axes[1]

    cohorts = [
        ("S1xS2\nTiny", "s1_s2_tiny_camo_micro"),
        ("S1xS3\nAdv.", "s1_s3_adverse_macro"),
        ("S1\nPraNet", "s1_pranet_external_macro"),
        ("Full\nS3 Adv.", "s3_adverse_macro"),
        ("Full\nETIS", "ETIS"),
    ]
    models = [
        ("Baseline (R9a)", "R9a", MUTED),
        ("DANG-Med v4.1 (R9e)", "R9e", PALETTE["blue"]),
        ("DANG-Med v4 (R8e)", "R8e", PALETTE["green"]),
    ]

    x = np.arange(len(cohorts))
    width = 0.78 / len(models)

    for i, (m_label, m_key, col) in enumerate(models):
        offset = (i - (len(models) - 1) / 2.0) * width
        bf1_vals = []
        hd_vals = []
        for _, c_key in cohorts:
            if c_key == "ETIS":
                rec = data[m_key]["pranet_external_per_dataset"]["Ext_ETIS_LaribPolypDB"]
            else:
                rec = data[m_key][c_key]
            bf1_vals.append(rec["boundary_f1"])
            hd_vals.append(rec["hd95"])

        b1 = ax_bf1.bar(x + offset, bf1_vals, width, label=m_label, color=col, edgecolor="black", linewidth=0.35)
        b2 = ax_hd.bar(x + offset, hd_vals, width, label=m_label, color=col, edgecolor="black", linewidth=0.35)
        ax_bf1.bar_label(b1, fmt="%.0f", fontsize=5.0, padding=1.2)
        ax_hd.bar_label(b2, fmt="%.0f", fontsize=5.0, padding=1.2)

    ax_bf1.set_xticks(x, [c[0] for c in cohorts])
    ax_bf1.set_ylabel("Boundary F1 (%)")
    ax_bf1.set_ylim(0, 82)
    ax_bf1.tick_params(axis="x", length=0)

    ax_hd.set_xticks(x, [c[0] for c in cohorts])
    ax_hd.set_ylabel("HD95 Error (px)")
    ax_hd.set_ylim(0, 108)
    ax_hd.tick_params(axis="x", length=0)
    ax_hd.legend(loc="upper right", fontsize=5.8, frameon=True, framealpha=0.95, edgecolor="black")

    panel_labels(axes)
    save(fig, "figs/boundary_hd95_comparison")


def make_interactive_html(data: dict):
    art_dir = Path("/home/tuan/.local/share/openresearch/files/dang")
    sys.path.insert(0, str(art_dir))
    from orx_chart import render_chart

    # Export CSV evidence
    csv_path = art_dir / "dang_med_v4_1_results_summary.csv"
    rows = []
    for m_key in ["R8a", "R8b", "R8c", "R8d", "R8e", "R9a", "R9b", "R9c", "R9d", "R9e"]:
        if m_key not in data:
            continue
        sd = data[m_key]
        rows.append({
            "model": m_key,
            "kvasir_test_dice": round(sd["in_domain_kvasir"]["dice"], 2),
            "polypgen_dev_dice": round(sd["ood_polypgen_macro"]["dice"], 2),
            "polypgen_ext_dice": round(sd["ood_polypgen_ext_macro"]["dice"], 2),
            "pranet_ext_dice": round(sd["pranet_external_macro"]["dice"], 2),
            "etis_larib_dice": round(sd["pranet_external_per_dataset"]["Ext_ETIS_LaribPolypDB"]["dice"], 2),
            "s2_tiny_camo_dice": round(sd["s2_tiny_camo_micro"]["dice"], 2),
            "s3_adverse_dice": round(sd["s3_adverse_macro"]["dice"], 2),
            "s1_pranet_dice": round(sd["s1_pranet_external_macro"]["dice"], 2),
            "s1_s2_tiny_dice": round(sd["s1_s2_tiny_camo_micro"]["dice"], 2),
            "s1_s3_adverse_dice": round(sd["s1_s3_adverse_macro"]["dice"], 2),
            "fps": sd["efficiency"]["fps"],
        })
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    bench_labels = [
        ("S1xS2 Tiny/Camo", lambda d: d["s1_s2_tiny_camo_micro"]),
        ("S1xS3 Adverse", lambda d: d["s1_s3_adverse_macro"]),
        ("S1 PraNet-Ext", lambda d: d["s1_pranet_external_macro"]),
        ("Full S3 Adverse", lambda d: d["s3_adverse_macro"]),
        ("Full S2 Tiny/Camo", lambda d: d["s2_tiny_camo_micro"]),
        ("Full ETIS-Larib", lambda d: d["pranet_external_per_dataset"]["Ext_ETIS_LaribPolypDB"]),
        ("Full PraNet-Ext", lambda d: d["pranet_external_macro"]),
        ("Full Kvasir-Test", lambda d: d["in_domain_kvasir"]),
    ]

    series = []
    for name, key in [
        ("Single-Stream Baseline (R9a)", "R9a"),
        ("Dual-Stream SafeConcat (R9b)", "R9b"),
        ("DANG-v4.1 ProtoDPT+PASSE (R9c)", "R9c"),
        ("DANG-Med v4.1 Flagship Zero-Shot (R9e)", "R9e"),
        ("DANG-Med v4 Flagship Adv-Trained (R8e)", "R8e"),
    ]:
        pts = []
        for idx, (blabel, fn) in enumerate(bench_labels):
            m = fn(data[key])
            pts.append({
                "step": idx + 1,
                "label": blabel,
                "dice": round(m["dice"], 2),
                "iou": round(m["iou"], 2),
                "bf1": round(m["boundary_f1"], 2),
                "hd95": round(m["hd95"], 2),
            })
        series.append({"name": name, "points": pts})

    render_chart(
        str(art_dir / "figures/dang_med_v4_1_benchmark_chart.html"),
        title="DANG-Med v4 / v4.1 vs. Baselines Across Hard OOD & Clinical Stress Benchmarks",
        subtitle="Seed 42 completed on Kaggle T4 (5-seed runs R10a-R10e in progress) · 12 cohorts",
        metrics=[
            dict(key="dice", label="mDice", type="bar", unit="%", yLabel="mDice (%)"),
            dict(key="iou", label="mIoU", type="bar", unit="%", yLabel="mIoU (%)"),
            dict(key="bf1", label="Boundary F1", type="bar", unit="%", yLabel="Boundary F1 (%)"),
            dict(key="hd95", label="HD95 Distance Error", type="bar", unit="px", yLabel="HD95 (px, lower is better)"),
        ],
        series=series,
    )


if __name__ == "__main__":
    all_data = parse_all_logs()
    make_stress_benchmarks_fig(all_data)
    make_ablation_gains_fig(all_data)
    make_interactive_html(all_data)
    print("Generated all paper figures and interactive HTML chart successfully.")
