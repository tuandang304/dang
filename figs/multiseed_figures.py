"""5-seed publication figures for paper.tex (seeds 42-46).

Reads figs/seed_metrics.json written by figs/aggregate_seeds.py from the raw run logs.
Regenerate:
  uv run --no-project --with numpy --with scipy python figs/aggregate_seeds.py
  uv run --no-project --with matplotlib --with numpy python figs/multiseed_figures.py
"""

import json
import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
    ("safeconcat", "Concatenation control", PALETTE["orange"]),
    ("protodpt_passe", "DANG without stem", BLUES[0]),
    ("flagship", "DANG", BLUES[1]),
]
# darker companions for the mean bar and interval, so light fills stay legible
DARK = ["#6E6E6E", "#B07800", "#3C8FC0", "#004F7C"]
COHORTS = [
    ("s1s2", "40 → tiny/camouflaged"),
    ("s1s3", "40 → adverse"),
    ("s1pra", "40 → PraNet-Ext"),
    ("full_etis", "800 → ETIS"),
    ("full_s2", "800 → tiny/camouflaged"),
    ("full_s3", "800 → adverse"),
]


def vals(arm, key):
    return np.array([DATA[arm][s][key] for s in SEEDS])


def ci(x):
    return T95_DF4 * x.std(ddof=1) / math.sqrt(len(x))


def benchmarks():
    """Dot-and-interval rows: each cohort is a row, each arm a sub-row with seed dots, mean and 95% CI."""
    use_style()
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 3.3), sharey=True, layout="constrained")
    n = len(COHORTS)
    y = np.arange(n)[::-1].astype(float)
    offs = np.linspace(0.27, -0.27, len(ARMS))
    split = next(i for i, (c, _) in enumerate(COHORTS) if c.startswith("full"))
    rng = np.random.default_rng(0)
    for ax, (metric, xlabel, xlim) in zip(axes, [("dice", "mDice (%)", (38, 86)),
                                                 ("bf1", "Boundary F1 (%)", (8, 68))]):
        for i in range(n):
            if i % 2 == 0:
                ax.axhspan(y[i] - 0.5, y[i] + 0.5, color="#F3F3F3", lw=0, zorder=0)
        ax.axhline(y[split] + 0.5, color="#9A9A9A", lw=0.6, zorder=1)
        for yi, (c, _) in zip(y, COHORTS):
            for k, (arm, _, color) in enumerate(ARMS):
                v = vals(arm, f"{c}_{metric}")
                m, h = v.mean(), ci(v)
                yy = yi + offs[k]
                ax.scatter(v, yy + rng.uniform(-0.03, 0.03, len(v)), s=5, color=color, alpha=0.5,
                           edgecolors="none", zorder=2)
                ax.plot([m - h, m + h], [yy, yy], color=DARK[k], lw=1.1, solid_capstyle="round", zorder=3)
                ax.plot([m], [yy], "o", ms=3.4, color=DARK[k], mec="white", mew=0.4, zorder=4)
                ax.text(max(m + h, v.max()) + 0.6, yy, f"{m:.1f}", va="center", ha="left", fontsize=5,
                        color=DARK[k], zorder=5)
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [lbl for _, lbl in COHORTS])
    axes[0].set_ylim(-0.5, n - 0.5)
    handles = [mpl.lines.Line2D([], [], color=DARK[k], lw=1.1, marker="o", ms=3.4, mec="white", mew=0.4)
               for k in range(len(ARMS))]
    fig.legend(handles, [lbl for _, lbl, _ in ARMS], ncol=4, loc="outside upper center", fontsize=6.5,
               frameon=False, handlelength=1.6, columnspacing=1.4)
    panel_labels(axes)
    save(fig, str(HERE / "multiseed_benchmarks"))


DELTA_ROWS = [
    ("s1s2_dice", "40 → tiny/camouflaged"),
    ("s1s3_dice", "40 → adverse (all)"),
    ("s1s3_Adverse_NBI_Spectral", "40 → adverse · NBI"),
    ("s1s3_Adverse_LowLight_Lumen", "40 → adverse · low light"),
    ("s1s3_Adverse_Specular_Glare_Blur", "40 → adverse · glare + blur"),
    ("s1s3_Adverse_Combined_Severe", "40 → adverse · combined"),
    ("s1pra_dice", "40 → PraNet-Ext"),
    ("s1pgdev_dice", "40 → PolypGen-Dev"),
    ("full_etis_dice", "800 → ETIS"),
    ("full_pra_dice", "800 → PraNet-Ext"),
    ("full_pgext_sd_micro_dice", "800 → PolypGen-Ext-SD"),
    ("full_s2_dice", "800 → tiny/camouflaged"),
    ("full_s3_dice", "800 → adverse"),
]


def paired(a, b, key):
    d = vals(a, key) - vals(b, key)
    return d.mean(), ci(d)


def deltas():
    """Paired seed differences in mDice with 95% t-intervals (n=5 paired seeds)."""
    use_style()
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 3.45), sharey=True, layout="constrained",
                             gridspec_kw={"width_ratios": [1.55, 1]})
    n = len(DELTA_ROWS)
    y = np.arange(n)[::-1].astype(float)
    split = next(i for i, (k, _) in enumerate(DELTA_ROWS) if k.startswith("full"))
    comps_a = [("protodpt_passe", "DANG without stem", BLUES[0], 0.16),
               ("flagship", "DANG", BLUES[1], -0.16)]
    for ax in axes:
        for i in range(n):  # zebra rows
            if i % 2 == 0:
                ax.axhspan(y[i] - 0.5, y[i] + 0.5, color="#F3F3F3", lw=0, zorder=0)
        ax.axhline(y[split] + 0.5, color="#9A9A9A", lw=0.6, zorder=1)
        ax.axvline(0, color=BASELINE, lw=0.8, zorder=1)
        ax.grid(False)
        ax.tick_params(axis="y", length=0)
    for arm, label, color, dy in comps_a:
        for yi, (key, _) in zip(y, DELTA_ROWS):
            m, h = paired(arm, "baseline", key)
            axes[0].plot([m - h, m + h], [yi + dy] * 2, color=color, lw=1.3, solid_capstyle="round", zorder=3)
            axes[0].plot([m], [yi + dy], "o", ms=3.6, color=color, mec="white", mew=0.5, zorder=4)
        axes[0].plot([], [], "o-", color=color, ms=3.6, lw=1.3, mec="white", mew=0.5, label=label)
    axes[0].set_xlabel("Paired Δ mDice vs. ConvNeXt-T baseline (pp)")
    axes[0].set_xlim(-6.5, 27.5)
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=2, loc="outside upper center", fontsize=6.5,
               frameon=False, handlelength=1.8, columnspacing=1.6)

    xmax = 7.0
    for yi, (key, _) in zip(y, DELTA_ROWS):
        m, h = paired("flagship", "protodpt_passe", key)
        hit = (m - h) > 0 or (m + h) < 0
        color = PALETTE["red"] if hit else "#555555"
        axes[1].plot([m - h, m + h], [yi] * 2, color=color, lw=1.3, solid_capstyle="round", zorder=3)
        axes[1].plot([m], [yi], "o", ms=3.6, color=color, mec="white", mew=0.5, zorder=4)
        axes[1].text(xmax + 0.3, yi, f"{m:+.2f}", va="center", ha="left", fontsize=6, color=color,
                     fontweight="bold" if hit else "normal")
    axes[1].set_xlabel("Paired Δ mDice, DANG − DANG without stem (pp)")
    axes[1].set_xlim(-8.0, xmax)
    axes[1].text(xmax + 0.3, y[0] + 0.75, "mean", va="bottom", ha="left", fontsize=6, color="#555555")
    for ax, left, right in [(axes[0], "← worse than baseline", "better than baseline →"),
                            (axes[1], "← stem hurts", "stem helps →")]:
        for txt, dx, ha in [(left, -4, "right"), (right, 4, "left")]:
            ax.annotate(txt, xy=(0, -0.95), xytext=(dx, 0), textcoords="offset points", ha=ha, va="center",
                        fontsize=5.5, color="#7F7F7F")
    axes[0].set_yticks(y, [lbl for _, lbl in DELTA_ROWS])
    axes[0].set_ylim(-1.3, n - 0.5)
    panel_labels(axes)
    save(fig, str(HERE / "multiseed_deltas"))


if __name__ == "__main__":
    benchmarks()
    deltas()
