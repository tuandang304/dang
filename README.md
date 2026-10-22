# AdaPolyp: Adapting a Frozen Vision Foundation Model for Generalizable Polyp Segmentation

Code, evaluation logs and table scripts for the paper by Thanh-Tuan Dang and Phu-Nguyen Le (submitted to IEEE Access).

AdaPolyp adapts a frozen DINOv3 ViT-S/16 encoder with convolutional bottleneck adapters, decodes a four-scale token
pyramid with a scale-aware decoder, and adds a dual-prototype margin head. It trains 1.94M of its 23.5M parameters.
Trained on the standard Kvasir-SEG + CVC-ClinicDB split, it reaches 89.8% mean Dice on 1,376 held-out PolypDB frames,
against 86.7% for the strongest retrained model, Polyp-PVT. A frame takes 9.4 ms on an NVIDIA T4 with FP16 weights and
CUDA graphs.

The Python package is called `dang`, the project's working name; in the code and logs AdaPolyp appears as `dang` or
`DANG (R2-b)`.

## Repository layout

| Path | Contents |
|---|---|
| `src/dang/dang_v5.py` | AdaPolyp: frozen encoder with adapters (`EvaConvLoRA`), token pyramid, scale-aware decoder, dual-prototype margin head, and the decoder variants of Table 7 |
| `src/dang/train.py` | entry point, AdaPolyp training loop, losses and augmentations |
| `src/dang/config.py` | every setting, with the paper's AdaPolyp as the default; override any field with `key=value` |
| `src/dang/bench/` | benchmark: cohorts and deduplication (`data.py`), evaluator (`evaluate.py`), multi-GPU runner (`runner.py`), official recipes of the compared models (`recipes.py`) |
| `src/dang/bench/official/` | the compared models' network code, copied from the authors' repositories (commits in `PROVENANCE.md`) |
| `src/dang/deploy_bench.py` | one-process latency benchmark of all models on one GPU (Table 8) |
| `results/logs/` | evaluation lines (per seed, per image, configuration) of every run reported in the paper, one file per run |
| `results/manifests/` | the frames kept in each evaluation cohort after deduplication |
| `results/bench_metrics.json`, `results/tables/`, `results/figures/` | aggregated statistics, table rows and figures regenerated from `results/logs/` |
| `scripts/` | aggregation, statistical tests, table and figure scripts |

`model.py` and the `protocol="dang"` path of `train.py` (an earlier Kvasir stress-test protocol) come from earlier
development and are not used by any experiment in the paper.

## Setup

Python 3.10 or later, [uv](https://docs.astral.sh/uv/) and a CUDA GPU (the paper used NVIDIA T4 16 GB).

```bash
uv sync --locked
```

All datasets are public and are downloaded on first use:

- **Training split and the five standard test sets** (Kvasir, CVC-ClinicDB, CVC-ColonDB, ETIS-LaribPolypDB, CVC-300):
  the PraNet `TrainDataset.zip` / `TestDataset.zip`, from the Hugging Face mirror `Zyna1121/lightpranet-polyp-data`.
- **PolypGen:** from the Hugging Face dataset `mzschwartz88/combo_polyp`.
- **PolypDB:** from its official [OSF release](https://osf.io/pr7ms/).

The deduplication against the training images runs every time the cohorts are built. The run logs the SHA-256 prefix
of the kept-frame list (`BENCH_MANIFEST_SHA256_16`); the lists themselves are in `results/manifests/`.

## Running the experiments

Every run is `uv run --locked python -m dang.train [key=value ...]`. Outputs (fp16 weights per seed, `summary.json`,
optional prediction masks) go to `$ORX_OUTPUT_DIR` or to `outputs/`. Seeds are spread over all visible GPUs. All
reported numbers come from the log lines `BENCH_SEED_SUMMARY_JSON` and `BENCH_PER_IMAGE`.

```bash
S=bench_seeds=42,43,44,45,46

# AdaPolyp, five seeds (Tables 2-6). Add bench_polypdb=true to also score the held-out PolypDB.
uv run --locked python -m dang.train $S

# Held-out PolypDB from the saved weights of the run above, without retraining (how the paper's final test was run)
uv run --locked python -m dang.train $S bench_polypdb=true bench_eval_cohorts=PolypDB \
    bench_eval_weights=outputs bench_dump_seed=none

# Component ablation (Table 6)
uv run --locked python -m dang.train $S v12_proto=false                    # without prototype head
uv run --locked python -m dang.train $S v5_lora_blocks=0                   # without adapters (fully frozen encoder)
uv run --locked python -m dang.train $S v12_proto=false v5_lora_blocks=0   # without both

# Decoder comparison (Table 7)
uv run --locked python -m dang.train $S v12_decoder=dual v12_passe=true    # fg/bg decoder with PASSE
uv run --locked python -m dang.train $S v12_decoder=query                  # query decoder

# Compared models, retrained with their official recipes (Tables 2-5)
for m in pranet polyp_pvt pranet_v2 segdino; do
  uv run --locked python -m dang.train $S bench_model=$m bench_polypdb=true
done

# Latency, MACs and parameters of every model on one GPU (Table 8, Fig. 5)
uv run --locked python -m dang.train bench_only=true v12_cache_rope=true
```

Under the official recipes, Polyp-PVT trains for 99 epochs in fp32. That took longer than a Kaggle T4 session, so the
paper's Polyp-PVT seeds were trained on cloud GPUs of another type.

## Reproducing the tables from the logs

`results/logs/` holds the evaluation output of every run behind the paper's tables. These commands rebuild the
statistics, the LaTeX table rows and Figs. 4 and 5 from those logs:

```bash
uv run --locked python scripts/aggregate_bench.py      # -> results/bench_metrics.json (paired t, Welch, Wilcoxon, Holm)
uv run --locked python scripts/bench_tables.py         # -> results/tables/tab_*_rows.tex
uv run --locked python scripts/export_manifests.py     # -> results/manifests/<cohort>.txt
uv run --locked --with matplotlib python scripts/per_centre.py         # -> results/figures/per_centre.pdf (Fig. 4)
uv run --locked --with matplotlib python scripts/accuracy_latency.py   # -> results/figures/accuracy_latency.pdf (Fig. 5)
```

`aggregate_bench.py` also accepts a directory of your own run logs, named `<model>__<run id>.log.gz`. Edit the run IDs
in its `GROUPS` table to match.

| Table | Rows file |
|---|---|
| 2: comparison with retrained models | `tab_main_rows.tex` |
| 3: paired tests | `tab_stats_rows.tex` |
| 4: boundary F1, HD95, failure rate | `tab_secondary_rows.tex` |
| 5: toolbox mean Dice | `tab_toolbox_rows.tex` |
| 6: component ablation | `tab_component_rows.tex` |
| 7: decoder comparison | `tab_ablation_rows.tex` |
| 8: inference and training cost | `tab_eff_rows.tex` |

## Citation

```bibtex
@article{dang2026adapolyp,
  title   = {AdaPolyp: Adapting a Frozen Vision Foundation Model for Generalizable Polyp Segmentation},
  author  = {Dang, Thanh-Tuan and Le, Phu-Nguyen},
  journal = {IEEE Access},
  note    = {under review},
  year    = {2026}
}
```

The compared models' code in `src/dang/bench/official/` remains under its authors' licenses (see the `LICENSE` files and
`PROVENANCE.md`).
