# Pipeline Guide

Full reproduction instructions for the BFFM-XGB training pipeline — from raw data through trained models, exported ONNX artifacts, and publication figures.

## Prerequisites

- Python 3.10+
- Node.js 22+ (for TypeScript inference tests via `npx vitest run`)
- ~2 GB disk space (dataset + trained models)

## Setup and Run

```bash
# Create virtual environment and install Python, TypeScript, and web dependencies
make setup

# Run the full pipeline (download -> load -> norms-check -> ... -> figures)
make all
```

`make setup` runs `setup-python`, `setup-typescript`, and `setup-web` to install all three ecosystems.

`make all` runs all local stages in order (through figure generation), including hyperparameter tuning, strict norm drift checks, cross-variant evaluation (`research-eval`), ONNX export via `export-all`, and research notes generation. You can also run individual stages (see [Pipeline Stages](#pipeline-stages) below).

## Running Tests

```bash
# Run all tests (lib/ unit tests + Python/TypeScript inference tests + web tests)
make test

# Run only the lib/ unit tests
make test-lib

# Run only the inference tests (Python + TypeScript)
make test-inference

# Run only the web tests
make test-web
```

`make test` runs `test-lib`, `test-inference`, and `test-web`.

## Pipeline Stages

The pipeline consists of 13 numbered scripts, executed in order. Each script is self-contained and reads/writes from well-defined paths.

| #   | Script                       | Make target(s)                                                    | Description                                                                                                                             |
| --- | ---------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| 01  | `01_download.py`             | `download`                                                        | Downloads the IPIP-FFM dataset ZIP from openpsychometrics.org                                                                           |
| 02  | `02_load_sqlite.py`          | `load`                                                            | Loads raw CSV, filters valid responses and duplicate IPs (IPC=1), reverse-scores items, writes to SQLite                                |
| 03  | `03_compute_norms.py`        | `norms`, `norms-check`                                            | Computes deterministic full-50 and Mini-IPIP norm stats from stage-02 SQLite; writes lock+meta artifacts; `norms-check` validates drift |
| 04  | `04_prepare_data.py`         | `prepare`, `prepare-default`, `prepare-stratified`                | Builds isolated train/val/test splits for two regimes (`ext-est`, `ext-est-opn`) and writes Parquet                                     |
| 05  | `05_compute_correlations.py` | `correlations`, `correlations-default`, `correlations-stratified` | Computes ranking artifacts per isolated data regime (`item_info.json`, `first_item.json`, correlations)                                 |
| 06  | `06_tune.py`                 | `tune`                                                            | Runs Optuna TPE hyperparameter search (optional; results locked)                                                                        |
| 07  | `07_train.py`                | `train`                                                           | Trains XGBoost quantile models with sparsity augmentation (4 configs); accepts `PARAMS=` override                                       |
| 08  | `08_validate.py`             | `validate` (+ `check-model-data-pairing`)                         | Validates at two sparsity levels (full 50-item, sparse 20-item) with bootstrap CIs                                                      |
| 09  | `09_baselines.py`            | `baselines` (+ `check-model-data-pairing`)                        | Evaluates 8 item-selection strategies at K=5,10,15,20,25,30,40,50; includes standalone Mini-IPIP baseline at K=20                       |
| 10  | `10_simulate.py`             | `simulate` (+ `check-model-data-pairing`)                         | Simulates adaptive assessment with SEM-based stopping on held-out respondents                                                           |
| 11  | `11_export_onnx.py`          | `export` (+ `check-model-data-pairing`)                           | Exports XGBoost models to ONNX, validates numerical parity, generates config.json                                                       |
| 12  | `12_generate_figures.py`     | `figures`                                                         | Generates publication figures from artifacts (efficiency curves, heatmaps, etc.)                                                        |
| 13  | `13_upload_hf.py`            | `upload-hf`                                                       | Uploads exported model and model card to HuggingFace Hub (requires `HF_TOKEN`)                                                          |

`make all` runs: download, load, norms, norms-check, prepare, correlations, tune, train, research-eval, export-all, notes, and figures. It excludes stage 13 (upload-hf). Evaluation stages (08-10) run via `research-eval`, which evaluates all four model variants in parallel by default and writes results to `artifacts/variants/<variant>/`.

## Hyperparameter Tuning

Tuning runs as part of `make all`. To re-tune independently:

```bash
# Run Optuna hyperparameter search (~2-4 hours)
make tune

# Train all model variants (reference first, then 3 ablations in parallel)
make train

# Use explicit XGBoost parallelism (recommended for reproducible thread config)
make tune N_JOBS=16
make train N_JOBS=16

# Train only one variant (N in 1..4)
make train 1
make train 4

# Control ablation fan-out after train-1 (inherits outer make parallelism by default)
make train TRAIN_PARALLEL=3
make train CV_PARALLEL_FOLDS=2
make -j1 train

# Force research-eval serially if needed
make research-eval RESEARCH_EVAL_PARALLEL=1

# Override training split paths
make train DATA_DIR=data/processed/ext_est
make train TRAIN_DATA_DIR=data/processed/ext_est TRAIN_DATA_DIR_STRATIFIED=data/processed/ext_est_opn
```

**Make variables for training:**

| Variable                    | Default                                                                                                     | Description                                                                             |
| --------------------------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `PARAMS`                    | *(none)*                                                                                                    | Hyperparameter JSON override; when unset, each config uses its own `locked_params` path |
| `N_JOBS`                    | *(none)*                                                                                                    | XGBoost thread count; when unset, uses `training.n_jobs` from config                    |
| `CV_PARALLEL_FOLDS`         | `1`                                                                                                         | Number of stage-07 CV folds to run concurrently on CPU (`1` on GPU)                     |
| `TRAIN_DATA_DIR`            | `DATA_DIR`                                                                                                  | Training data for runs 1--3                                                             |
| `TRAIN_DATA_DIR_STRATIFIED` | `data/processed/ext_est_opn` (falls back to `TRAIN_DATA_DIR` if it differs from the default `ext_est` path) | Training data for run 4 (stratified split)                                              |
| `TRAIN_PARALLEL`            | *(inherit outer make)*                                                                                      | Fan-out for ablation runs 2--4                                                          |

Thread count precedence: `N_JOBS` > `training.n_jobs` in config > `$BFFM_XGB_N_JOBS` > `os.cpu_count()`.
The committed configs pin `training.n_jobs: 16` so defaults are machine-independent.

## Model/Data Selection

Post-training stage targets (`validate`, `baselines`, `simulate`, `export`) use `MODEL_DIR` and `DATA_DIR` make variables. Evaluation output is always written to `artifacts/variants/<model_name>/` (derived automatically from `MODEL_DIR`).

```bash
# Default: reference model + ext_est data -> artifacts/variants/reference/
make validate

# Auto-selects stratified data path from model dir -> artifacts/variants/ablation_stratified/
make validate MODEL_DIR=models/ablation_stratified

# Explicit data override (must match model regime) -> artifacts/variants/ablation_none/
make baselines MODEL_DIR=models/ablation_none DATA_DIR=data/processed/ext_est
```

The `check-model-data-pairing` guard runs before these targets and fails closed if a known model bundle is paired with the wrong data regime.

## Training Variants

The `train` stage runs four model variants with a strict lock policy:

- `train-1` (`reference.yaml`) runs first.
- `train-2`, `train-3`, and `train-4` then run in parallel.
- All runs use the same tuned hyperparameters from `artifacts/tuned_params.json`.
- Ablations fail closed unless `models/reference/training_report.json` exists and the hyperparameter hash matches the reference model.

| Config                     | Sparsity                         | Description                                                  |
| -------------------------- | -------------------------------- | ------------------------------------------------------------ |
| `reference.yaml`           | Focused + Mini-IPIP + Imbalanced | Published model (exported to ONNX)                           |
| `ablation_none.yaml`       | None                             | Baseline: no sparsity augmentation                           |
| `ablation_focused.yaml`    | Focused + Mini-IPIP              | Focused + Mini-IPIP (no imbalanced patterns)                 |
| `ablation_stratified.yaml` | Focused + Mini-IPIP              | Ablation: ext-est-opn stratified data regime                 |

Run a single training configuration in isolation with `make train N` where `N` is `1`, `2`, `3`, or `4`.

Only the reference model is exported to ONNX.

## Cross-Variant Research Evaluation

After training all 4 model variants, `make research-eval` runs the full evaluation pipeline (validate + baselines + simulate) for each variant, writing results to isolated artifact directories:

```
make research-eval          # runs all 4 variants in parallel by default
make research-eval-reference
make research-eval-ablation-none
make research-eval-ablation-focused
make research-eval-ablation-stratified
```

Each `research-eval-*` target runs `validate`, `baselines`, and `simulate` with the correct model/data pairing. Output is automatically routed to `artifacts/variants/<variant>/` via the `EVAL_DIR` Makefile variable, and each variant also writes a labeled logfile under `logs/`.
Override variant parallelism with `RESEARCH_EVAL_PARALLEL=<n>` if needed. Use `RESEARCH_EVAL_PARALLEL=1` for explicit serial execution.

After all variants complete:

```
make notes                  # builds research_summary.json (strict), then refreshes NOTES.md
make figures                # generates publication figures from artifacts
```

The `notes` target runs `research-summary-strict` first, which fails closed unless all four variants have complete, provenance-consistent evaluation bundles under `artifacts/variants/`. The aggregated `artifacts/research_summary.json` serves as the single canonical manifest for all auto-generated data sections in `notes/NOTES.md`.

| Target                    | Inputs                                                                              | Outputs                                                                                                             |
| ------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `research-eval`           | `models/*/`, `data/processed/ext_est{,_opn}/`                                       | `artifacts/variants/*/validation_results.json`, `baseline_comparison_results.json`, `simulation_results.json`, etc. |
| `research-summary-strict` | `artifacts/variants/*/`, `models/*/training_report.json`                            | `artifacts/research_summary.json`                                                                                   |
| `notes`                   | `artifacts/research_summary.json`                                                   | `notes/NOTES.md` (data sections refreshed)                                                                          |
| `figures`                 | `artifacts/variants/<model>/` (defaults to `reference`; override with `MODEL_DIR=`) | `figures/*.png`                                                                                                     |

Parallel train/eval targets also write prefixed per-task logs under `logs/`
(`train-*.log`, `eval-*.log`) to make concurrent AWS runs easier to follow.

## Norm Reproducibility

```bash
# Recompute and write deterministic norms lock file + provenance sidecar
make norms

# Strict check against committed artifacts/ipip_bffm_norms.json (fail-closed)
make norms-check
```

All downstream provenance uses a stable `data_snapshot_id` derived from
`artifacts/ipip_bffm_norms.json` SHA-256 (`norms_sha256:<hash>`), so run identity
is not coupled to wall-clock dates during long multi-day pipeline runs.

The metadata omits per-run timestamps. Key fields stored by `build_provenance()` include
the git hash, `data_snapshot_id`, `preprocessing_version`, `script`,
and any RNG seeds or bootstrap config provided by each pipeline stage. This keeps the
provenance chain intact while avoiding launch-time differences between successive runs.

## Uploading to HuggingFace

```bash
# Copy .env.example to .env and add your HuggingFace token
cp .env.example .env
# Edit .env to set HF_TOKEN=hf_...
make upload-hf
```

## Remote Training

A 96-vCPU AWS CPU instance completes the full pipeline much quicker than on a local machine. See [`docs/infrastructure.md`](infrastructure.md) for provisioning details, spot/on-demand configuration, and the `remote-all` workflow.

## Cleaning Up

```bash
# Move generated pipeline outputs into .backup/
make clean

# Copy the most recent .backup/ payload back into place
make restore
```

`make clean` preserves repo-relative structure under `.backup/` instead of
deleting generated outputs outright. Running `make restore` copies that backup
back into place. `make restore` fails on existing destination conflicts unless
you pass `FORCE=1`.

## Reference Artifacts

The `artifacts/` directory contains global pipeline artifacts. Per-model evaluation results live under `artifacts/variants/<variant>/`.

| File                                               | Contents                                                |
| -------------------------------------------------- | ------------------------------------------------------- |
| `tuned_params.json`                                | Tuned hyperparameters from Optuna (used by all configs) |
| `ipip_bffm_norms.json`                             | Deterministic full-50 + Mini-IPIP norm lock file        |
| `mini_ipip_mapping.json`                           | Mini-IPIP to IPIP-BFFM item mapping                     |
| `research_summary.json`                            | Aggregated cross-variant research summary               |
| `variants/<name>/validation_results.json`          | Validation suite output per variant                     |
| `variants/<name>/baseline_comparison_results.json` | Baseline comparison with bootstrap CIs per variant      |
| `variants/<name>/simulation_results.json`          | Adaptive assessment simulation metrics per variant      |

These artifacts allow `12_generate_figures.py` to produce publication figures without retraining. They also serve as regression tests; the pipeline validates that newly trained models reproduce these numbers within tolerance.
