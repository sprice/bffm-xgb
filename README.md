# BFFM-XGB: Big Five Sparse Quantile Model

**[Take the assessment](https://big5.shawnprice.com/)** — try the live Big Five personality assessment powered by this model.

Standalone pipeline for training, evaluating, and exporting sparse-input XGBoost quantile regression models for the 50-item [IPIP Big-Five Factor Markers](https://ipip.ori.org/newBigFive5broadKey.htm) (BFFM) personality assessment.

The exported models predict accurate Big Five personality scores from partial item responses (NaN = unanswered), enabling short-form assessments of 20 items or fewer with calibrated 90% confidence intervals.

### Comparison with the Mini-IPIP

Both approaches use 20 items (4 per domain) to recover the full 50-item IPIP-BFFM scale scores. The [Mini-IPIP](https://ipip.ori.org/MiniIPIPTable.htm) (Donnellan et al., 2006) uses expert-curated items scored by simple averaging. BFFM-XGB-20 uses correlation-ranked items scored by XGBoost cross-domain ML. All *r* values below are Pearson correlations with the full 50-item scale on a held-out test set (*N* = TBD).

|                              | Mini-IPIP                   | BFFM-XGB-20                       |
|------------------------------|-----------------------------|------------------------------------|
| **Items**                    | 20 (4 per domain)           | 20 (4 per domain)                  |
| **Item selection**           | Expert-curated brevity      | Top-4 by within-domain *r*         |
| **Scoring**                  | Simple scale averaging      | XGBoost quantile regression        |
| **Overall *r***              | .906                        | **.926**                           |
| **MAE (percentile pts)**     | 9.2                         | **8.2**                            |
| **90% prediction intervals** | —                           | ✓ (90.1% coverage)                |

**Per-domain accuracy at *K* = 20:**

| Domain              | Mini-IPIP *α* | Mini-IPIP *r* | BFFM-XGB-20 *r* |
|---------------------|---------------|---------------|------------------|
| Extraversion        | .77           | .937          | **.946**         |
| Agreeableness       | .70           | .910          | **.918**         |
| Conscientiousness   | .69           | .910          | **.919**         |
| Emotional Stability | .68           | .925          | **.936**         |
| Intellect/Openness  | .65           | .844          | **.910**         |

Mini-IPIP *α* from [Donnellan et al. (2006)](https://doi.org/10.1037/1040-3590.18.2.192). Mini-IPIP *r* uses ML scoring on Mini-IPIP items (a drop-in upgrade; simple averaging yields overall *r* = .906). The largest gain is on Intellect/Openness (+.066), where the Mini-IPIP items were selected for brevity rather than discrimination.

At 15 items, BFFM-XGB already matches the Mini-IPIP's 20-item accuracy (*r* = .907 vs .906).

## Quick Start: Inference Only

If you just want to **run the pre-trained models** without reproducing the full pipeline, see the inference packages:

| Language   | Directory                    | Install                                      | Run tests             |
| ---------- | ---------------------------- | -------------------------------------------- | --------------------- |
| Python     | [`python/`](python/)         | `pip install onnxruntime numpy scipy pytest` | `python -m pytest -v` |
| TypeScript | [`typescript/`](typescript/) | `npm ci`                                     | `npm test`            |

**Python example:**

```python
from inference import IPIPBFFMPredictor

predictor = IPIPBFFMPredictor()

result = predictor.predict({
    "ext3": 4.0, "ext5": 5.0, "agr1": 3.0, "agr7": 4.0,
    "csn1": 5.0, "csn4": 3.0, "est9": 4.0, "est10": 3.0,
    "opn5": 3.0, "opn10": 4.0,
})

for domain in ["ext", "agr", "csn", "est", "opn"]:
    r = result[domain]
    print(f"{domain}: {r['percentile']['q50']}th pct "
          f"(90% CI: {r['percentile']['q05']}--{r['percentile']['q95']})")
```

**TypeScript example:**

```typescript
import { IPIPBFFMPredictor } from "./inference.js";

const predictor = await IPIPBFFMPredictor.create();

const result = await predictor.predict({
  ext3: 4.0, ext5: 5.0, agr1: 3.0, agr7: 4.0,
  csn1: 5.0, csn4: 3.0, est9: 4.0, est10: 3.0,
  opn5: 3.0, opn10: 4.0,
});

for (const domain of ["ext", "agr", "csn", "est", "opn"] as const) {
  const r = result[domain];
  console.log(`${domain}: ${r.percentile.q50}th pct `
    + `(90% CI: ${r.percentile.q05}--${r.percentile.q95})`);
}

predictor.dispose();
```

The pre-trained ONNX model and `config.json` are in [`output/`](output/). The config file contains feature names, norms, calibration factors, and all metadata needed for inference.

## Full Reproduction

To reproduce the entire pipeline from raw data through trained model, exported ONNX artifacts, and publication figures:

### Prerequisites

- Python 3.10+
- Node.js 22+ (for TypeScript inference tests via `npx vitest run`)
- ~2 GB disk space (dataset + trained models)

### Setup and Run

```bash
# Create virtual environment and install dependencies
make setup

# Run the full pipeline (download -> load -> norms-check -> ... -> figures)
make all
```

The `make all` target runs all local stages in order (through figure generation), including hyperparameter tuning, strict norm drift checks, cross-variant evaluation (`research-eval`), and research notes generation. You can also run individual stages (see [Pipeline Stages](#pipeline-stages) below).

### Running Tests

```bash
# Run all tests (lib/ unit tests + Python/TypeScript inference tests)
make test

# Run only the lib/ unit tests
make test-lib

# Run only the inference tests (Python + TypeScript)
make test-inference
```

### Custom Hyperparameter Tuning

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
make -j1 train

# Override training split paths
make train DATA_DIR=data/processed/ext_est
make train TRAIN_DATA_DIR=data/processed/ext_est TRAIN_DATA_DIR_STRATIFIED=data/processed/ext_est_opn
```

**Make variables for training:**

| Variable                    | Default                                                                                                     | Description                                                                             |
| --------------------------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `PARAMS`                    | *(none)*                                                                                                    | Hyperparameter JSON override; when unset, each config uses its own `locked_params` path |
| `N_JOBS`                    | *(none)*                                                                                                    | XGBoost thread count; when unset, uses `training.n_jobs` from config                    |
| `TRAIN_DATA_DIR`            | `DATA_DIR`                                                                                                  | Training data for runs 1--3                                                             |
| `TRAIN_DATA_DIR_STRATIFIED` | `data/processed/ext_est_opn` (falls back to `TRAIN_DATA_DIR` if it differs from the default `ext_est` path) | Training data for run 4 (stratified split)                                              |
| `TRAIN_PARALLEL`            | *(inherit outer make)*                                                                                      | Fan-out for ablation runs 2--4                                                          |

Thread count precedence: `N_JOBS` > `training.n_jobs` in config > `$BFFM_XGB_N_JOBS` > `os.cpu_count()`.
The committed configs pin `training.n_jobs: 16` so defaults are machine-independent.

### Model/Data Selection For Validation, Baselines, Simulation, Export

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

### Cross-Variant Research Artifacts and NOTES

`make all` includes `research-eval` and `notes`, so a full pipeline run automatically evaluates all four model variants and generates the research summary. You can also run these targets independently:

```bash
# Run validate/baselines/simulate for each model variant -> artifacts/variants/<variant>/
make research-eval

# Build aggregated research summary JSON
make research-summary
make research-summary-strict

# Refresh NOTES.md data sections (runs research-summary first)
make notes
```

`make notes` runs `make research-summary-strict` first and fails closed
unless all four variants have complete, provenance-consistent bundles.
All auto-generated NOTES data sections read from
`artifacts/research_summary.json` (single canonical manifest), whose inputs are
`models/*/training_report.json` plus per-variant evaluation outputs under
`artifacts/variants/*/`.

### Norm Reproducibility

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

### Uploading to HuggingFace

```bash
# Copy .env.example to .env and add your HuggingFace token
cp .env.example .env
# Edit .env to set HF_TOKEN=hf_...
make upload-hf
```

### Remote Training on AWS

A 96-vCPU AWS spot instance completes the full pipeline much quicker than on a local machine. Terraform provisions the instance and `make remote-all` handles the entire run unattended:

```bash
make infra-up      # provision spot instance (~60s)
make remote-all    # push, setup, run full pipeline, pull results, tear down
```

`remote-all` attaches to a live tmux view of the pipeline. If you detach (Ctrl+B D) or lose connection, the run continues remotely and the local process polls until completion, pulls results, and destroys the infrastructure.

See [`infra/README.md`](infra/README.md)

### Cleaning Up

```bash
# Remove downloaded data, processed data, and trained models
make clean
```

## Directory Structure

```
bffm-xgb/
├── artifacts/          Pipeline artifacts and static reference data
├── configs/            YAML training configurations (reference + ablation variants)
│   ├── reference.yaml          Published model configuration
│   ├── ablation_none.yaml      Ablation: no sparsity augmentation
│   ├── ablation_focused.yaml   Ablation: focused sparsity only
│   └── ablation_stratified.yaml Ablation: focused sparsity on ext-est-opn split
├── data/               Downloaded and processed data (gitignored)
├── figures/            Generated publication figures (gitignored, regenerable)
├── lib/                Shared Python library
│   ├── bootstrap.py       Bootstrap confidence interval functions
│   ├── constants.py       Domains, items, reverse-keyed items, default params
│   ├── item_info.py       Item info loading and validation
│   ├── mini_ipip.py       Mini-IPIP subset utilities
│   ├── norms.py           Strict norms loader for artifacts/ipip_bffm_norms.json
│   ├── parallelism.py     XGBoost thread-count resolution
│   ├── provenance.py      Git hash detection and build metadata
│   ├── provenance_checks.py  Provenance validation checks
│   ├── scoring.py         Raw score -> percentile conversion (z-score + CDF)
│   └── sparsity.py        Sparsity augmentation strategies
├── models/             Trained model checkpoints (gitignored)
├── notes/              Research notes (auto-generated data sections)
├── output/             Exported ONNX model, config.json, model card (generated)
├── pipeline/           Numbered pipeline scripts (01 through 13)
├── python/             Python inference package with tests
├── scripts/            Research summary and notes generation scripts
├── tests/              Unit tests for lib/ modules (pytest)
├── typescript/         TypeScript inference package with tests (vitest)
├── .env.example        Template for HF_TOKEN (required for `make upload-hf`)
├── .gitignore
├── LICENSE.md          MIT License
├── Makefile            Orchestrates the full pipeline
├── NOTICES.md          Third-party attributions (CC0, IPIP, OSPP)
├── package.json        Workspace metadata
├── pyproject.toml      pytest configuration (pythonpath, testpaths)
├── requirements.txt    Python dependencies
└── README.md           This file
```

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
| 09  | `09_baselines.py`            | `baselines` (+ `check-model-data-pairing`)                        | Evaluates 7 item-selection strategies at K=5,10,15,20,25,30,40,50; includes standalone Mini-IPIP baseline at K=20                       |
| 10  | `10_simulate.py`             | `simulate` (+ `check-model-data-pairing`)                         | Simulates adaptive assessment with SEM-based stopping on held-out respondents                                                           |
| 11  | `11_export_onnx.py`          | `export` (+ `check-model-data-pairing`)                           | Exports XGBoost models to ONNX, validates numerical parity, generates config.json                                                       |
| 12  | `12_generate_figures.py`     | `figures`                                                         | Generates publication figures from artifacts (efficiency curves, heatmaps, etc.)                                                        |
| 13  | `13_upload_hf.py`            | `upload-hf`                                                       | Uploads exported model and model card to HuggingFace Hub (requires `HF_TOKEN`)                                                          |

`make all` runs: download, load, norms, norms-check, prepare, correlations, tune, train, research-eval, export, notes, and figures. It excludes stage 13 (upload-hf). Evaluation stages (08-10) run via `research-eval`, which evaluates all four model variants and writes results to `artifacts/variants/<variant>/`.

### Cross-Variant Research Evaluation

After training all 4 model variants, `make research-eval` runs the full evaluation pipeline (validate + baselines + simulate) for each variant, writing results to isolated artifact directories:

```
make research-eval          # runs all 4 variants sequentially
make research-eval-reference
make research-eval-ablation-none
make research-eval-ablation-focused
make research-eval-ablation-stratified
```

Each `research-eval-*` target runs `validate`, `baselines`, and `simulate` with the correct model/data pairing. Output is automatically routed to `artifacts/variants/<variant>/` via the `EVAL_DIR` Makefile variable.

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

### Training Variants

The `train` stage runs four model variants with a strict lock policy:

- `train-1` (`reference.yaml`) runs first.
- `train-2`, `train-3`, and `train-4` then run in parallel.
- All runs use the same tuned hyperparameters from `artifacts/tuned_params.json`.
- Ablations fail closed unless `models/reference/training_report.json` exists and the hyperparameter hash matches the reference model.

| Config                     | Sparsity                         | Description                                  |
| -------------------------- | -------------------------------- | -------------------------------------------- |
| `reference.yaml`           | Focused + Mini-IPIP + Imbalanced | Published model (exported to ONNX)           |
| `ablation_none.yaml`       | None                             | Baseline: no sparsity augmentation           |
| `ablation_focused.yaml`    | Focused + Mini-IPIP              | Ablation: no imbalanced patterns             |
| `ablation_stratified.yaml` | Focused + Mini-IPIP              | Ablation: ext-est-opn stratified data regime |

Run a single training configuration in isolation with `make train N` where `N` is `1`, `2`, `3`, or `4`.

Only the reference model is exported to ONNX.

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

## Model Architecture

- **Algorithm:** XGBoost quantile regression with pinball loss
- **Models:** 15 total (5 domains x 3 quantiles: q05, q50, q95)
- **Input:** 50 features (float32 at inference; float64 during training), one per IPIP-BFFM item; NaN for unanswered items
- **Output:** Raw domain score (1--5 scale), converted to percentile via z-score norms
- **Cross-validation:** 5-fold nested CV with evaluation split before augmentation
- **Hyperparameters:** Tuned via Optuna TPE search (stage 06); stored in `artifacts/tuned_params.json`

### Sparsity Augmentation

The key idea is **sparsity augmentation**: each training respondent (who answered all 50 items) is augmented 3 times (`n_augmentation_passes=3`), each time with a different random mask that sets a subset of items to NaN. The model only trains on masked data and learns to predict accurately regardless of which items are present.

The reference model uses **focused sparsity** with the A0.1 distribution, which assigns each augmented row to one of five masking buckets:

| Bucket | Proportion | Items Kept | Description                                       |
| ------ | ---------- | ---------- | ------------------------------------------------- |
| 0      | 40%        | 10--20     | Domain-balanced masking (min 2 items per domain)  |
| 1      | 10%        | 20         | Mini-IPIP subset (fixed 4 items x 5 domains)      |
| 2      | 20%        | 21--35     | Moderate sparsity (min 4 items per domain)        |
| 3      | 15%        | 36--50     | Light sparsity (min 4 items per domain)           |
| 4      | 15%        | varies     | Imbalanced patterns (some domains get zero items) |

Within buckets 0--3, item selection is weighted by cross-domain information scores (from step 05), so more informative items are retained more frequently.

Bucket 4 (imbalanced patterns) simulates real-world adaptive behavior where some domains receive many items while others receive none:

- **Greedy-mimicking** (50%): selects exactly the top-K items from the ranked item pool
- **Random-skewed** (30%): drops 1--2 random domains entirely
- **Extreme-skewed** (20%): concentrates items in 1--2 domains, 0--1 items in others

This teaches the model to handle arbitrary missing-item patterns, enabling accurate predictions from as few as 20 items.

## Norms

Raw-score to percentile conversion uses z-score transformation with norms derived from the full cleaned stage-02 SQLite response table (`responses`, OSPP dataset). The single source of truth is `artifacts/ipip_bffm_norms.json` and includes both `norms` (full-50 scoring) and `mini_ipip_norms` (standalone Mini-IPIP scoring); regenerate with `make norms` and validate with `make norms-check`.

## Data

Training data comes from the [Open-Source Psychometrics Project](https://openpsychometrics.org/) (OSPP) dataset:

- **Split:** Stratified train/val/test split (default 70/15/15) using EXT x EST quintile strata
- **Augmentation:** Training set is augmented via 3 sparsity passes (see [Sparsity Augmentation](#sparsity-augmentation))
- **Split before augmentation:** Train/val/test split is performed before augmentation to prevent data leakage
- **RNG seed:** 42

## Limitations

- Norms are derived from self-selected online respondents (OSPP); they may not represent the general population
- Models are trained on English-language IPIP items only
- Accuracy degrades with fewer items; 20 items is the recommended minimum for reliable scoring
- Not intended for clinical diagnosis or high-stakes selection decisions

## License

[MIT](LICENSE.md). See [NOTICES.md](NOTICES.md) for third-party attributions.
