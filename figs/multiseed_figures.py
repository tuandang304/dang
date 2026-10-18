"""5-seed publication figures for paper.tex (Round 9/10 suite, seeds 42-46).

Reads figs/seed_metrics.json written by figs/aggregate_seeds.py from the raw run logs.
Regenerate:
  uv run --no-project --with numpy --with scipy python figs/aggregate_seeds.py
  uv run --no-project --with matplotlib --with numpy python figs/multiseed_figures.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from orx_figstyle import BASELINE, MUTED, PALETTE, WIDE, family, figure_grid, panel_labels, save, use_style  # noqa: E402

DATA = json.loads((HERE / "seed_metrics.json").read_text())
SEEDS = ["42", "43", "44", "45", "46"]
T95_DF4 = 2.776  # two-sided 95% t critical value, df = 4 (5 paired seeds)

BLUES = family("blue", 2)
ARMS = [
    ("baseline", "ConvNeXt-T + FPN baseline", MUTED),
    ("safeconcat", "Dual-stream SafeConcat control", PALETTE["orange"]),
    ("protodpt_passe", "ProtoDPT + PASSE + UncStrip (no stem)", BLUES[0]),
    ("flagship", "DANG-Med v4.1 (full)", BLUES[1]),
]
COHORTS = [
    ("s1s2", "S1×S2\ntiny/camo"),
    ("s1s3", "S1×S3\nadverse"),
    ("s1pra", "S1→PraNet\nexternal"),
    ("full_etis", "Full→ETIS\n(n=196)"),
    ("full_s2", "Full→S2\ntiny/camo"),
    ("full_s3", "Full→S3\nadverse"),
]


def vals(arm, key):
    return np.array([DATA[arm][s][key] for s in SEEDS])


def ci(x):
    return T95_DF4 * x.std(ddof=1) / math.sqrt(len(x))


def benchmarks():
    """Grouped bars: 5-seed mean, 95% CI whisker, and each seed as a dot."""
    use_style()
    fig, axes = figure_grid(2, 1, width=WIDE, ratio=0.62, sharex=True)
    width = 0.2
    x = np.arange(len(COHORTS))
    for ax, metric, ylabel in [(axes[0], "dice", "mDice (%)"), (axes[1], "bf1", "Boundary F1 (%)")]:
        for i, (arm, label, color) in enumerate(ARMS):
            off = (i - 1.5) * width
            means, errs, tops = [], [], []
            for j, (c, _) in enumerate(COHORTS):
                v = vals(arm, f"{c}_{metric}")
                means.append(v.mean())
                errs.append(ci(v))
                tops.append(max(v.max(), v.mean() + ci(v)))
                jitter = np.linspace(-0.055, 0.055, len(v))
                ax.scatter(x[j] + off + jitter, v, s=3.5, color="black", alpha=0.55, linewidths=0, zorder=4)
            bars = ax.bar(x + off, means, width, color=color, edgecolor="black", linewidth=0.4,
                          label=label, zorder=2)
            ax.errorbar(x + off, means, yerr=errs, fmt="none", ecolor="black", elinewidth=0.6,
                        capsize=1.2, capthick=0.6, zorder=3)
            # Value label sits above the whisker and the highest seed so neither covers it.
            for xx, m, t in zip(x + off, means, tops):
                ax.text(xx, t + 1.2, f"{m:.1f}", ha="center", va="bottom", fontsize=5)
        top = 10 * math.ceil(max(vals(a, f"{c}_{metric}").max() for a, *_ in ARMS for c, _ in COHORTS) * 1.15 / 10)
        ax.set_ylim(0, top)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="x", visible=False)
    axes[1].set_xticks(x, [lbl for _, lbl in COHORTS])
    axes[0].legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.07), fontsize=6.5, frameon=False,
                   handlelength=1.2, columnspacing=1.2)
    panel_labels(axes)
    save(fig, str(HERE / "multiseed_benchmarks"))


DELTA_ROWS = [
    ("s1s2_dice", "S1×S2 tiny/camo"),
    ("s1s3_dice", "S1×S3 adverse (all)"),
    ("s1s3_Adverse_NBI_Spectral", "S1×S3 · NBI spectral"),
    ("s1s3_Adverse_LowLight_Lumen", "S1×S3 · low-light lumen"),
    ("s1s3_Adverse_Specular_Glare_Blur", "S1×S3 · glare + blur"),
    ("s1s3_Adverse_Combined_Severe", "S1×S3 · combined severe"),
    ("s1pra_dice", "S1→PraNet external"),
    ("s1pgdev_dice", "S1→PolypGen-Dev"),
    ("full_etis_dice", "Full→ETIS-Larib"),
    ("full_pra_dice", "Full→PraNet external"),
    ("full_pgext_sd_micro_dice", "Full→PolypGen-Ext (seq.-disj.)"),
    ("full_s2_dice", "Full→S2 tiny/camo"),
    ("full_s3_dice", "Full→S3 adverse"),
]


def paired(a, b, key):
    d = vals(a, key) - vals(b, key)
    return d.mean(), ci(d)


def deltas():
    """Paired seed differences in mDice with 95% t-intervals (n=5 paired seeds)."""
    use_style()
    fig, axes = figure_grid(1, 2, width=WIDE, ratio=0.54, sharey=True)
    y = np.arange(len(DELTA_ROWS))[::-1]
    comps_a = [("protodpt_passe", "ProtoDPT + PASSE + UncStrip (no stem)", BLUES[0], 0.17),
               ("flagship", "DANG-Med v4.1 (full)", BLUES[1], -0.17)]
    for arm, label, color, dy in comps_a:
        for yi, (key, _) in zip(y, DELTA_ROWS):
            m, h = paired(arm, "baseline", key)
            axes[0].plot([m - h, m + h], [yi + dy] * 2, color=color, lw=1.1, solid_capstyle="round")
            axes[0].plot([m], [yi + dy], "o", ms=3.2, color=color, mec="black", mew=0.3)
        axes[0].plot([], [], "o-", color=color, ms=3.2, lw=1.1, label=label)
    axes[0].set_xlabel("Paired Δ mDice vs. ConvNeXt-T baseline (pp)")
    axes[0].legend(ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.06), fontsize=6.5, frameon=False,
                   handlelength=1.6, columnspacing=1.2)

    for yi, (key, _) in zip(y, DELTA_ROWS):
        m, h = paired("flagship", "protodpt_passe", key)
        hit = (m - h) > 0 or (m + h) < 0
        color = PALETTE["red"] if hit else "#4D4D4D"
        axes[1].plot([m - h, m + h], [yi] * 2, color=color, lw=1.1, solid_capstyle="round")
        axes[1].plot([m], [yi], "o", ms=3.2, color=color)
        axes[1].annotate(f"{m:+.2f}", xy=(max(m + h, 0.0), yi), xytext=(4, 0), textcoords="offset points",
                         va="center", fontsize=5.5, color=color)
    axes[1].set_xlabel("Paired Δ mDice, full − no-stem (pp)")

    for ax in axes:
        ax.axvline(0, color=BASELINE, lw=0.8, zorder=0)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    axes[1].set_xlim(-9.5, 7.5)
    axes[0].set_yticks(y, [lbl for _, lbl in DELTA_ROWS])
    axes[0].set_ylim(-0.6, len(DELTA_ROWS) - 0.4)
    panel_labels(axes)
    save(fig, str(HERE / "multiseed_deltas"))


if __name__ == "__main__":
    benchmarks()
    deltas()
