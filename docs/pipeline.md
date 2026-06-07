# Pipeline Guide

Full reproduction instructions for the BFFM-XGB training pipeline — from raw data through trained models, exported ONNX artifacts, and publication figures.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) — manages the Python interpreter (pinned to 3.14 via `.python-version`; `requires-python` is `>=3.11`) and all dependencies. **All Python is run through `uv` (`uv run …`), never a bare `python`.**
- Node.js 22+ (for TypeScript inference tests via `npx vitest run`)
- ~2 GB disk space (dataset + trained models)

## Setup and Run

```bash
# Install Python deps (uv sync), plus TypeScript and web dependencies
make setup

# Run the full pipeline (download -> load -> norms-check -> ... -> figures)
make all
```

`make setup` runs `setup-python`, `setup-typescript`, and `setup-web` to install all three ecosystems. `setup-python` runs `uv sync`, which creates `.venv` and installs the locked dependencies from `pyproject.toml` + `uv.lock` (including the `dev` group: pytest, ruff, basedpyright).

`make all` runs all local stages in order (through figure generation), including hyperparameter tuning, strict norm drift checks, cross-variant evaluation (`research-eval`), ONNX export via `export-all`, and research notes generation. You can also run individual stages (see [Pipeline Stages](#pipeline-stages) below).

## Regenerating model-derived docs

Every model-derived number in the docs and web app is **generated from artifacts**, never hand-typed — the model cards, `output/README.md`, `notes/NOTES.md`, the `<!-- BEGIN/END GENERATED -->` fences in `README.md`/`docs/*.md`, and `web/.../repo-facts.generated.ts` (imported by the web app). `make all` regenerates all of them at the end. After a *partial* re-run (or after pulling fresh artifacts, or editing a generator), refresh every surface from the current artifacts in one shot — **no retrain, no re-eval**:

```bash
make refresh-docs                   # reference + both ablations
make refresh-docs REFERENCE_ONLY=1  # only the reference variant has been trained
```

Then `git diff` and commit the result. CI runs `make check-docs`, which fails if any committed generated surface is stale relative to the artifacts — so a forgotten regeneration is caught automatically. (Figures regenerate too; matplotlib embeds non-deterministic PDF metadata, so `figures/manifest.json` PDF SHAs may change even when the underlying data did not.)

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

## Smoke test (de-risk the full run)

```bash
# Tiny sampled end-to-end run: stages 03-12 + all analysis code, in minutes
make smoke

# Remove the isolated smoke outputs
make smoke-clean
```

`make smoke` runs the *entire* pipeline (norms → prepare → correlations → tune →
train → validate → baselines → simulate → export → figures) on a small sample
(`SMOKE_SAMPLE`, default 8000 respondents) with a tiny model, exercising every
stage and all the analysis paths (reliability, raw quantile-crossing, paired /
subset bootstraps, the SEM simulation, cross-validation robustness, ONNX export)
in a few minutes on CPU. It is the cheap way to catch a stage crash or a
shape/logic bug *before* committing to a full multi-hour/day run.

Everything is namespaced under `smoke_v1` / `models/smoke` / `output/smoke` /
`artifacts/variants/smoke` / `artifacts/smoke_*.json` / `figures/smoke`, so it
never touches the canonical artifacts. It needs the SQLite DB (`make load`)
present. The smoke config (`configs/smoke.yaml` + `configs/smoke_params.json`)
relaxes the validation gates to 0.0 and uses tiny hyperparameters — it is **not**
a publication config. Sampling is leakage-safe: running the stage-03 norms step
with `--sample N` (`uv run python pipeline/03_compute_norms.py --sample N`) fits
norms on the same first-N respondents stage 04 splits, and the stage-04
`population_signature` guard fails closed if the norms population differs.

`make fixtures` is a separate, deterministic generator for the committed
tri-runtime test fixture (`tests/fixtures/golden/`); re-run it only after an
intentional `xgboost`/`onnx`/`onnxmltools` version bump.

## Linting & Formatting

```bash
make lint     # ruff check . (whole tree, incl. tests/)
make format   # ruff format (apply)
```

Linting uses [ruff](https://docs.astral.sh/ruff/) (config in `[tool.ruff]` in
`pyproject.toml`; ruleset `E4/E7/E9/F/I/UP`). The CI `lint` job runs `ruff
check .` over the whole tree (including `tests/`) on every push/PR and it is kept
at **zero** findings (no baseline). `E402` (import-not-at-top) is ignored for
`pipeline/`, `scripts/`, and `tests/`, where a `sys.path.insert(...)` precedes the
`lib.*` imports by design.

## Type Checking

```bash
# Static type-check the Python sources (pipeline/, lib/, scripts/, python/)
make typecheck
```

Type checking uses [basedpyright](https://docs.basedpyright.com/) in `standard`
mode (configured in `[tool.basedpyright]` in `pyproject.toml`, pinned in
`pyproject.toml`'s `[dependency-groups] dev`). The CI `typecheck` job runs it on
every push/PR.

The checker is gated against a committed baseline at
`.basedpyright/baseline.json`: it **fails only on new diagnostics**, so new code
must be type-clean. New **error- and warning**-severity diagnostics both fail
`make typecheck` — basedpyright's own exit code ignores warnings, so the target
pipes `--outputjson` through `scripts/typecheck_gate.py` to gate on both (green ==
0 errors **and** 0 warnings). Keep that gate; a plain `basedpyright` invocation
would let new warnings through CI. pandas is fully typed via the `pandas-stubs` dev dependency;
the baseline grandfathers the residual false-positives from libraries that ship
no/incomplete type stubs (xgboost, onnxmltools, scipy, matplotlib) and should
only ever shrink. To inspect or re-snapshot it:

```bash
uv run basedpyright --writebaseline   # re-record current diagnostics
```

Regenerate the baseline only after intentionally reducing it; never grow it to
hide a genuine new error. (The committed baseline is generated locally; if a CI
run surfaces an environment-specific stub diagnostic, refresh it there.)

## Pipeline Stages

The pipeline consists of 13 numbered scripts, executed in order. Each script is self-contained and reads/writes from well-defined paths.

| #   | Script                       | Make target(s)                                                    | Description                                                                                                                             |
| --- | ---------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| 01  | `01_download.py`             | `download`                                                        | Downloads the IPIP-FFM dataset ZIP from openpsychometrics.org                                                                           |
| 02  | `02_load_sqlite.py`          | `load`                                                            | Loads raw CSV, filters valid responses and duplicate IPs (IPC=1), reverse-scores items, writes to SQLite                                |
| 03  | `03_compute_norms.py`        | `norms`, `norms-check`                                            | Computes deterministic full-50 and Mini-IPIP norm stats from stage-02 SQLite; writes lock+meta artifacts; `norms-check` validates drift |
| 04  | `04_prepare_data.py`         | `prepare`                                                         | Builds the single plain random train/val/test split (`canonical_v1`, 70/15/15) with train-only norms; writes Parquet                    |
| 05  | `05_compute_correlations.py` | `correlations`                                                    | Computes ranking artifacts from the canonical split (`item_info.json`, `first_item.json`, correlations)                                 |
| 06  | `06_tune.py`                 | `tune`                                                            | Runs Optuna TPE hyperparameter search (optional; results locked)                                                                        |
| 07  | `07_train.py`                | `train`                                                           | Trains XGBoost quantile models with sparsity augmentation (3 configs); accepts `PARAMS=` override                                       |
| 08  | `08_validate.py`             | `validate`                                                        | Validates at two sparsity levels (full 50-item, sparse 20-item) with bootstrap CIs                                                      |
| 09  | `09_baselines.py`            | `baselines`                                                       | Evaluates 8 item-selection strategies at K=5,10,15,20,25,30,40,50; includes standalone Mini-IPIP baseline at K=20                       |
| 10  | `10_simulate.py`             | `simulate`                                                        | Simulates adaptive assessment with SEM-based stopping on held-out respondents                                                           |
| 11  | `11_export_onnx.py`          | `export`                                                          | Exports XGBoost models to ONNX, validates numerical parity, generates config.json                                                       |
| 12  | `12_generate_figures.py`     | `figures`                                                         | Generates publication figures from artifacts (efficiency curves, heatmaps, etc.)                                                        |
| 13  | `13_upload_hf.py`            | `upload-hf`                                                       | Uploads exported model and model card to HuggingFace Hub (requires `HF_TOKEN`)                                                          |

`make all` runs: download, load, norms, norms-check, prepare, correlations, tune, train, research-eval, export-all, notes, gen-docs, and figures. It excludes stage 13 (upload-hf). Evaluation stages (08-10) run via `research-eval`, which evaluates all three model variants in parallel by default and writes results to `artifacts/variants/<variant>/`.

## Hyperparameter Tuning

Tuning runs as part of `make all`. To re-tune independently:

```bash
# Run Optuna hyperparameter search (~2-4 hours)
make tune

# Train all model variants (reference first, then 2 ablations in parallel)
make train

# Use explicit XGBoost parallelism (recommended for reproducible thread config)
make tune N_JOBS=16
make train N_JOBS=16

# Train only one variant (N in 1..3)
make train 1
make train 3

# Control ablation fan-out after train-1 (inherits outer make parallelism by default)
make train TRAIN_PARALLEL=3
make train CV_PARALLEL_FOLDS=2
make -j1 train

# Force research-eval serially if needed
make research-eval RESEARCH_EVAL_PARALLEL=1

# Override the training split path
make train DATA_DIR=data/processed/canonical_v1
```

**Make variables for training:**

| Variable                    | Default                                                                                                     | Description                                                                             |
| --------------------------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `PARAMS`                    | *(none)*                                                                                                    | Hyperparameter JSON override; when unset, each config uses its own `locked_params` path |
| `N_JOBS`                    | *(none)*                                                                                                    | XGBoost thread count; when unset, uses `training.n_jobs` from config                    |
| `CV_PARALLEL_FOLDS`         | `1`                                                                                                         | Number of stage-07 CV folds to run concurrently on CPU (`1` on GPU)                     |
| `TRAIN_DATA_DIR`            | `DATA_DIR`                                                                                                  | Training data for runs 1--3                                                             |
| `TRAIN_PARALLEL`            | *(inherit outer make)*                                                                                      | Fan-out for ablation runs 2--3                                                          |
| `NO_GATE`                   | *(none)*                                                                                                    | `NO_GATE=1` records the quality-gate outcome but saves the bundle even on a threshold miss |

Thread count precedence: `N_JOBS` > `training.n_jobs` in config > `$BFFM_XGB_N_JOBS` > `os.cpu_count()`.
The committed configs pin `training.n_jobs: 16` so the default does not silently follow `os.cpu_count()` on whatever machine runs the pipeline. **Bit-exact reproduction, however, requires matching the *recorded* thread count, not the config default:** the published reference bundle was trained with a CLI override (recorded as `xgb_n_jobs_source: "cli"` in `models/reference/training_report.json`). Because `tree_method=hist` accumulates gradients across threads in a non-associative (non-deterministic) order, the trained trees — and therefore the exported `model.onnx` bytes — are **not** bit-identical across thread counts. To reproduce the published bundle byte-for-byte, match the recorded thread count below:

<!-- BEGIN GENERATED: training-config -->
- **Optuna tuning budget:** 200 trials.
- **Cross-validation:** 3-fold.
- **Recorded XGBoost thread count (published bundle):** `xgb_n_jobs = 10` (CLI override) — reproduce byte-for-byte with `make train N_JOBS=10`.
<!-- END GENERATED: training-config -->

**Quality gate (`NO_GATE`).** Stage 07 evaluates the trained models against the config's `validation` thresholds *after* training and, by default, aborts (`return 1`, saving nothing) if any threshold is missed. For the first run on a new split — where a near-miss should not discard multi-day compute — pass `NO_GATE=1` (e.g. `make train NO_GATE=1`, or `NO_GATE=1 bash scripts/run-pipeline.sh --reference-only`). The gate still runs and its outcome is recorded honestly in `training_report.json` as `quality_gates: {passed, enforced}`, but the bundle is saved regardless. **A bundle with `enforced: false` (or `passed: false`) must have its `validation_metrics` reviewed manually before it is published** — `NO_GATE` only suppresses the abort, not the check. Only `NO_GATE=1` enables it; any other value (including `0`) leaves the gate enforcing.

## Model/Data Selection

Post-training stage targets (`validate`, `baselines`, `simulate`, `export`) use `MODEL_DIR` and `DATA_DIR` make variables. Evaluation output is always written to `artifacts/variants/<model_name>/` (derived automatically from `MODEL_DIR`).

```bash
# Default: reference model + canonical_v1 data -> artifacts/variants/reference/
make validate

# Explicit data override (must match model regime) -> artifacts/variants/ablation_none/
make baselines MODEL_DIR=models/ablation_none DATA_DIR=data/processed/canonical_v1
```

## Training Variants

The `train` stage runs three model variants with a strict lock policy:

- `train-1` (`reference.yaml`) runs first.
- `train-2` and `train-3` then run in parallel.
- All runs use the same tuned hyperparameters from `artifacts/tuned_params.json`.
- Ablations fail closed unless `models/reference/training_report.json` exists and the hyperparameter hash matches the reference model.

| Config                     | Sparsity                         | Description                                                  |
| -------------------------- | -------------------------------- | ------------------------------------------------------------ |
| `reference.yaml`           | Focused + Mini-IPIP + Imbalanced | Published model (exported to ONNX)                           |
| `ablation_none.yaml`       | None                             | Baseline: no sparsity augmentation                           |
| `ablation_focused.yaml`    | Focused + Mini-IPIP              | Focused + Mini-IPIP (no imbalanced patterns)                 |

Run a single training configuration in isolation with `make train N` where `N` is `1`, `2`, or `3`.

Only the reference model is exported to ONNX.

## Cross-Variant Research Evaluation

After training all 3 model variants, `make research-eval` runs the full evaluation pipeline (validate + baselines + simulate) for each variant, writing results to isolated artifact directories:

```
make research-eval          # runs all 3 variants in parallel by default
make research-eval-reference
make research-eval-ablation-none
make research-eval-ablation-focused
```

Each `research-eval-*` target runs `validate`, `baselines`, and `simulate` with the correct model/data pairing. Output is automatically routed to `artifacts/variants/<variant>/` via the `EVAL_DIR` Makefile variable, and each variant also writes a labeled logfile under `logs/`.
Override variant parallelism with `RESEARCH_EVAL_PARALLEL=<n>` if needed. Use `RESEARCH_EVAL_PARALLEL=1` for explicit serial execution.

After all variants complete:

```
make notes                  # builds research_summary.json (strict), then refreshes NOTES.md
make figures                # generates publication figures from artifacts
```

The `notes` target runs `research-summary-strict` first, which fails closed unless all three variants have complete, provenance-consistent evaluation bundles under `artifacts/variants/`. The aggregated `artifacts/research_summary.json` serves as the single canonical manifest for all auto-generated data sections in `notes/NOTES.md`.

For a **reference-only** pipeline run (which produces only the `reference` variant), pass `REFERENCE_ONLY=1` to scope these targets to that single variant: `make research-summary-strict REFERENCE_ONLY=1`, `make notes REFERENCE_ONLY=1`, and `make provenance-check REFERENCE_ONLY=1`. The summary and completeness gate then require only the reference bundle (still fail-closed on *it*), and the generated `NOTES.md` renders the reference run with a disclosure that the ablation variants were not run. `scripts/run-pipeline.sh --reference-only` passes this flag automatically. The committed full-run `NOTES.md` is always produced without the flag (all three variants).

| Target                    | Inputs                                                                              | Outputs                                                                                                             |
| ------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `research-eval`           | `models/*/`, `data/processed/canonical_v1/`                                       | `artifacts/variants/*/validation_results.json`, `baseline_comparison_results.json`, `simulation_results.json`, etc. |
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

Note that `data/processed/load_metadata.json`'s `data_snapshot_id` may *lag* the
canonical locked-norms SHA used downstream (e.g. a metadata-only edit to the norms
file re-keyed its whole-file SHA while the norm *values* stayed byte-identical).
This is benign: downstream binding uses the split's `split_signature` plus the locked
`norms_sha256`, not `load_metadata`'s snapshot field.

For the same reason, a provenance sidecar's `git_hash` / `preprocessing_version` can
point at an earlier or rewritten commit — e.g. `artifacts/ipip_bffm_norms.meta.json` is
stamped at the commit that locked the norms, which later history may have rebased so the
hash is no longer reachable from the current branch. The binding identity is the content
SHA (`norms_sha256`, `split_signature`), not the `git_hash`, so an unreachable sidecar
`git_hash` does not affect reproducibility; regenerate the sidecar with `make norms` if a
reachable hash is wanted (the locked norm values are byte-identical, so `norms_sha256` is
unchanged).

The metadata omits per-run timestamps. Key fields stored by `build_provenance()` include
the git hash, `data_snapshot_id`, `preprocessing_version`, `script`,
and any RNG seeds or bootstrap config provided by each pipeline stage. This keeps the
provenance chain intact while avoiding launch-time differences between successive runs.

## Verifying a published release (fresh clone)

A clone can verify the published reference bundle **without retraining** and without the
SQLite DB. The small provenance/coupling artifacts (`output/reference/{config,provenance}.json`,
the model card, `research_summary.json`, `figures/manifest.json`, the norms meta sidecar) are
tracked in git; only the ~248 MB `model.onnx` lives on Hugging Face.

```bash
git clone <repo> && cd bffm-xgb && git checkout next
make pull-reference            # fetch model.onnx from the pinned HF revision (sha256-verified, fail-closed)
make verify-release            # provenance checks only — no norms-check, no DB, no retrain
make verify-release STRICT_HEAD=1   # additionally enforce git-freshness
```

`verify-release` runs `scripts/check_provenance.py` directly (it does **not** depend on
`norms-check`, so no SQLite DB is needed) and auto-detects reference-only mode from the
summary. Freshness is **merge-robust**: a bundle is fresh when its stamped `git_hash` is
HEAD, *or an ancestor of HEAD with no `pipeline/`, `lib/`, or `configs/` changes in between*
— so the release/refresh commits on top of the generation commit, and a later merge of
`next` into `main`, keep `--strict-head` green (the generation code at HEAD is identical to
what produced the bundle). Where git is unavailable (shallow clone / tarball) the git check
degrades and the content-sha chain stands alone. `pull-reference` is fail-closed against the
`.env` `HF_SHA256_MODEL` pin, so a stale pin or wrong model is rejected rather than installed.

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

**Reproduction expectations.** `artifacts/research_summary.json` is the source of truth for the cited numbers. A full `make all` reproduces them only *up to* XGBoost thread-count nondeterminism — `tree_method=hist` accumulates gradients across threads in a non-associative order — so retrained metrics match within tolerance, not bit-for-bit. Bit-exact reproduction of the published `model.onnx` additionally requires matching the recorded `xgb_n_jobs` (see [Hyperparameter Tuning](#hyperparameter-tuning)).
