"""Write the LaTeX table rows of the paper from figs/bench_metrics.json (made by figs/aggregate_bench.py).

Cells whose data do not exist yet (fewer than five seeds for a five-seed row, no PolypDB run, no latency
measurement) are written as "--" so the paper compiles and the gap is visible. Rerun both scripts after new runs land:

    uv run --isolated --no-project --with numpy --with scipy python figs/aggregate_bench.py
    python3 figs/bench_tables.py
"""

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

HERE = Path(__file__).parent
M = json.loads((HERE / "bench_metrics.json").read_text())
G, T, D = M["groups"], M["tests"], M["deploy"]
# display order: in domain, primary external, other external, final test
COHORTS = ["Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "PolypGen", "CVC-300", "PolypDB"]
STD = ["Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "CVC-300"]
PRIMARY = ["CVC-ColonDB", "ETIS-LaribPolypDB", "PolypGen"]
NA = "--"

# display name, group key, trainable params (M; all weights are trained in the published models)
MAIN = [("PraNet~\\cite{fan2020pranet}", "PraNet", None), ("Polyp-PVT~\\cite{dong2021polyp}", "Polyp-PVT", None),
        ("PraNet-V2~\\cite{hu2026pranetv2}", "PraNet-V2", None), ("SegDINO~\\cite{yang2025segdino}", "SegDINO", 2.48),
        ("\\textbf{AdaPolyp (ours)}", "DANG", 1.94)]
DEPLOY_NAME = {"PraNet": "PraNet (Res2Net-50)", "Polyp-PVT": "Polyp-PVT (PVTv2-B2)", "PraNet-V2": "PraNet-V2 (PVTv2-B2)",
               "DANG": ["DANG (R2-b): token pyramid, SAD", "DANG (R2-b, selected): token pyramid + SAD"], "V8-b": "V8-b: dual stream, fg/bg decoder, PASSE",
               "V12-b": "V12-b: token pyramid, fg/bg decoder, PASSE", "R2-a": "R2-a: token pyramid, fg/bg decoder, no PASSE",
               "SegDINO": "SegDINO (DINOv3 ViT-S/16 + DPT head, 256x256)",
               "R2-c": ["R2-c: token pyramid, fg/bg query decoder", "R2-c: token pyramid + fg/bg query decoder"]}


def deploy(k):
    names = DEPLOY_NAME.get(k, [])
    for n in ([names] if isinstance(names, str) else names):
        if n in D:
            return D[n]
    return {}
# published toolbox mDice (%) on the five standard test sets
PUBLISHED = [("PraNet (published)~\\cite{fan2020pranet}", [89.8, 89.9, 70.9, 62.8, 87.1]),
             ("Polyp-PVT (published)$^{\\ddagger}$~\\cite{dong2021polyp}", [91.7, 93.7, 80.8, 78.7, 90.0])]


INTERIM = ()   # groups whose five-seed runs are still landing (shown with a dagger); none left


def cell(group, cohort, metric="dice", need=5, bold=False, mark=""):
    c = G.get(group, {}).get("cohorts", {}).get(cohort, {}).get(metric)
    if c is not None and c["n"] < need and group in INTERIM and need == 5:
        sd = f"_{{\\pm {c['sd']:.1f}}}" if c.get("sd") is not None else ""
        return f"${c['mean']:.1f}{sd}{{}}^{{\\dagger}}$"
    if c is None or c["n"] < need:
        return NA
    v = f"{c['mean']:.1f}"
    v = f"\\mathbf{{{v}}}" if bold else v
    sd = f"_{{\\pm {c['sd']:.1f}}}" if c.get("sd") is not None and need > 1 else ""
    return f"${v}{sd}${mark}"


def best(groups, cohort, metric="dice", need=5):
    vals = {g: G[g]["cohorts"][cohort][metric]["mean"] for g in groups
            if g in G and cohort in G[g]["cohorts"] and G[g]["cohorts"][cohort][metric]["n"] >= need}
    if not vals:
        return set()
    top = round(max(vals.values()), 1)
    return {g for g, v in vals.items() if round(v, 1) == top}     # ties at display precision are all bold


def sig_mark(baseline, cohort):
    """* when DANG is better than this baseline after Holm correction (seed-level paired t-test, 5 seeds)."""
    r = T.get(baseline, {}).get(cohort, {})
    if len(r.get("seeds", [])) >= 5 and r.get("p_seed_holm", 1.0) < 0.05 and r["delta"] > 0:
        return "$^{*}$"
    return ""


def fmt(v, nd=1):
    # half-up from the decimal value (latencies are logged at 2 dp, so binary-float rounding would turn 9.35 into 9.3)
    return NA if v is None else str(Decimal(str(v)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def main_rows():
    keys = [k for _, k, _ in MAIN]
    bests = {c: best(keys, c) for c in COHORTS}
    out = []
    for name, k, trainable in MAIN:
        params = G.get(k, {}).get("params_m")
        tr = trainable if trainable is not None else params
        cells = [cell(k, c, bold=(k in bests[c]), mark=sig_mark(k, c) if k != "DANG" and c in PRIMARY else "")
                 for c in COHORTS]
        if k == "DANG":
            out.append("\\midrule")
        out.append(f"{name} & {fmt(params)} & {fmt(tr, 2 if tr and tr < 10 else 1)} & " + " & ".join(cells) + " \\\\")
    return out


def toolbox_rows():
    keys = [k for _, k, _ in MAIN]
    out = [f"{n} & " + " & ".join(f"${v:.1f}$" for v in vals) + " \\\\" for n, vals in PUBLISHED]
    out.append("\\midrule")
    for metric in ("mdice_tb",):
        bests = {c: best(keys, c, metric) for c in STD}
        for name, k, _ in MAIN:
            out.append(f"{name} & " + " & ".join(cell(k, c, metric, bold=(k in bests[c])) for c in STD) + " \\\\")
    return out


ABLATION = [("fg/bg decoder + PASSE", "V12-b"),
            ("fg/bg query decoder", "R2-c"),
            ("\\textbf{AdaPolyp}: scale-aware decoder", "DANG")]


def ablation_rows():
    """Five-seed decoder comparison on the same token pyramid (single-seed variants are not reported)."""
    six = COHORTS[:6]
    bests = {c: best([k for _, k in ABLATION], c) for c in six}
    out = []
    for name, k in ABLATION:
        dep = deploy(k)
        cells = [cell(k, c, bold=(k in bests[c])) for c in six]
        out.append(f"{name} & {fmt(G.get(k, {}).get('params_m'))} & {fmt(dep.get('gmacs'))} & "
                   f"{fmt(dep.get('graph_half_bs1_ms'))} & " + " & ".join(cells) + " \\\\")
    return out


COMPONENTS = [("\\textbf{AdaPolyp}", "DANG", 1.94), ("\\quad without prototype head", "A1-noproto", 0.92),
              ("\\quad without adapters (frozen encoder)", "A2-nolora", 1.79),
              # trained = total minus the 21.58M frozen encoder shared by all rows (A3 logs only the total)
              ("\\quad without adapters and prototype head", "A3-pyramid-only", 0.77)]


def component_rows():
    """Five-seed component ablation of DANG; the Delta columns are DANG minus the variant on the primary cohorts."""
    six = COHORTS[:6]
    out = []
    for name, k, trained in COMPONENTS:
        cells = [cell(k, c) for c in six]
        if k == "DANG":
            deltas = ["--"] * 3
        elif any(len(T.get(k, {}).get(c, {}).get("seeds", [])) < 5 for c in PRIMARY):
            deltas = [NA] * 3                                   # runs still landing
        else:
            deltas = []
            for c in PRIMARY:
                r = T[k][c]
                star = "^{*}" if r.get("p_seed_holm", 1) < 0.05 else ""      # seed level only (frames are clustered)
                deltas.append(f"${r['delta']:+.1f}{star}$")
        out.append(f"{name} & {fmt(G.get(k, {}).get('params_m'))} & {trained:.2f} & " + " & ".join(cells) + " & "
                   + " & ".join(deltas) + " \\\\")
    return out


def stats_rows():
    names = {"PraNet": "PraNet", "Polyp-PVT": "Polyp-PVT", "PraNet-V2": "PraNet-V2", "SegDINO": "SegDINO"}
    out = []
    for b, nm in names.items():
        cells = []
        for c in PRIMARY:
            r = T.get(b, {}).get(c, {})
            if len(r.get("seeds", [])) < 5:
                cells += [NA] * 3
                continue
            p = lambda x: "$<$0.001" if x < 0.001 else f"{x:.3f}"
            cells += [f"${r['delta']:+.1f}$", p(r["p_seed_holm"]), f"{p(r['p_image_holm'])} ({r['wins']}/{r['losses']})"]
        out.append(f"{nm} & " + " & ".join(cells) + " \\\\")
    return out


def eff_rows():
    """Inference cost from the one-host deploy run (inference-time parameter count of the timed model) and the mean
    training time per seed from the accuracy runs."""
    rows = [("PraNet$^{\\|}$", "PraNet"), ("Polyp-PVT$^{\\P}$", "Polyp-PVT"), ("PraNet-V2", "PraNet-V2"),
            ("SegDINO$^{\\S}$", "SegDINO"), ("\\textbf{AdaPolyp (ours)}", "DANG")]
    trainable = {"SegDINO": 2.48, "DANG": 1.94}
    out = []
    for name, k in rows:
        d = deploy(k)
        params = d.get("params_m", G.get(k, {}).get("params_m"))
        tr = trainable.get(k, params)
        tmin = G.get(k, {}).get("train_minutes", {}).get("mean")
        out.append(f"{name} & {fmt(params)} & {fmt(tr, 2 if tr and tr < 10 else 1)} & {fmt(d.get('gmacs'))} & "
                   f"{fmt(d.get('eager_fp16_bs1_ms'))} & {fmt(d.get('half_bs1_ms'))} & {fmt(d.get('graph_half_bs1_ms'))} & "
                   f"{fmt(d.get('eager_fp16_bs8_img_per_s'), 0)} & {fmt(tmin, 0)} \\\\")
    return out


SECONDARY = [("PraNet", "PraNet"), ("Polyp-PVT", "Polyp-PVT"), ("PraNet-V2", "PraNet-V2"), ("SegDINO", "SegDINO"),
             ("\\textbf{AdaPolyp (ours)}", "DANG")]
EXT4 = ["CVC-ColonDB", "ETIS-LaribPolypDB", "PolypGen", "PolypDB"]


def secondary_rows():
    """Boundary F1 (higher is better), HD95 in pixels at 352x352 (lower is better) and the share of frames with
    Dice below 50 % (lower is better) on the external cohorts; five-seed means."""
    out = []
    spec = [("bf1", max), ("hd95", min), ("fail_rate", min)]
    best = {(m, c): f([G[k]["cohorts"][c][m]["mean"] for _, k in SECONDARY if c in G[k]["cohorts"]])
            for m, f in spec for c in EXT4}
    for name, k in SECONDARY:
        cells = []
        for m, _ in spec:
            for c in EXT4:
                v = G[k]["cohorts"].get(c, {}).get(m, {}).get("mean")
                cells.append(NA if v is None else (f"$\\mathbf{{{v:.1f}}}$" if abs(v - best[(m, c)]) < 1e-9 else f"${v:.1f}$"))
        out.append(f"{name} & " + " & ".join(cells) + " \\\\")
    return out


def main():
    for name, rows in (("tab_main_rows", main_rows()), ("tab_toolbox_rows", toolbox_rows()),
                       ("tab_ablation_rows", ablation_rows()), ("tab_stats_rows", stats_rows()),
                       ("tab_eff_rows", eff_rows()), ("tab_component_rows", component_rows()),
                       ("tab_secondary_rows", secondary_rows())):
        (HERE / f"{name}.tex").write_text("% generated by figs/bench_tables.py -- do not edit\n" + "\n".join(rows) + "\n")
        print(name, len(rows), "rows")


if __name__ == "__main__":
    main()
