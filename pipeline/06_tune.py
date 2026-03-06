#!/usr/bin/env python3
"""
Optuna hyperparameter tuning for IPIP-BFFM XGBoost quantile models.

Runs Optuna TPE optimization over the XGBoost search space, maximizing sparse
20-item validation correlation with a full-50 guardrail. Tuning applies the
same sparsity regime configured for training (focused, imbalanced, Mini-IPIP).
Only q50 models are trained during tuning for efficiency; q05/q95 models are
trained with the winning params in 07_train.py.

Output is saved to artifacts/tuned_params.json and consumed by 07_train.py
via the locked_params key in each YAML config.

Usage:
    python pipeline/06_tune.py --trials 200 --config configs/reference.yaml
    python pipeline/06_tune.py --trials 200 --data-dir data/processed/ext_est --config configs/reference.yaml
    python pipeline/06_tune.py --trials 200 --parallel-trials 4 --config configs/reference.yaml
    python pipeline/06_tune.py --trials 50 --output artifacts/tuned_params.json
"""

import sys
import gc
import json
import logging
import shutil
import time
import argparse
import warnings
from pathlib import Path
from typing import Any, Optional

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ConstantInputWarning
import xgboost as xgb

from lib.constants import (
    DOMAINS,
    ITEM_COLUMNS,
    DEFAULT_PARAMS,
)
from lib.scoring import raw_score_to_percentile
from lib.provenance import build_provenance, add_provenance_args
from lib.item_info import file_sha256, load_item_info_strict
from lib.mini_ipip import load_mini_ipip_mapping
from lib.parallelism import coerce_positive_int, resolve_default_xgb_n_jobs
from lib.provenance_checks import build_split_signature as _build_split_signature
from lib.sparsity import apply_adaptive_sparsity_balanced, apply_sparsity_single

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data/processed/ext_est")
DEFAULT_ARTIFACTS_DIR = Path("artifacts")

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_parquet(path: Path) -> pd.DataFrame:
    """Load a parquet file and validate it exists."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_parquet(path)


def _prepare_features_targets(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into X (items), y (raw scores), y_pct (percentiles)."""
    item_cols = list(ITEM_COLUMNS)
    target_cols = [f"{d}_score" for d in DOMAINS]
    pct_cols = [f"{d}_percentile" for d in DOMAINS]

    missing_item_cols = [c for c in item_cols if c not in df.columns]
    missing_target_cols = [c for c in target_cols if c not in df.columns]
    missing_pct_cols = [c for c in pct_cols if c not in df.columns]
    if missing_item_cols or missing_target_cols or missing_pct_cols:
        details = []
        if missing_item_cols:
            details.append("items=" + ",".join(missing_item_cols))
        if missing_target_cols:
            details.append("scores=" + ",".join(missing_target_cols))
        if missing_pct_cols:
            details.append("percentiles=" + ",".join(missing_pct_cols))
        raise ValueError(
            "Tuning requires full Big-5 schema (50 items + 5 score + 5 percentile columns). "
            + "; ".join(details)
        )

    X = df[item_cols].astype(np.float64).copy()
    y = df[target_cols].astype(np.float64).copy()
    y_pct = df[pct_cols].astype(np.float64).copy()

    valid_mask = y.notna().all(axis=1) & y_pct.notna().all(axis=1)
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)
    y_pct = y_pct[valid_mask].reset_index(drop=True)
    if X.empty:
        raise ValueError("Tuning data has zero valid rows after target/percentile filtering.")

    return X, y, y_pct


def _load_item_info(
    data_dir: Path,
    *,
    expected_source_sha256: str | None = None,
) -> tuple[dict, str]:
    """Load stage-05 item info JSON with strict provenance."""
    item_info_path = data_dir / "item_info.json"
    item_info = load_item_info_strict(
        item_info_path,
        require_first_item=True,
        expected_source_sha256=expected_source_sha256,
    )
    return item_info, file_sha256(item_info_path)


def _load_mini_ipip_mapping(artifacts_dir: Path) -> dict[str, list[str]]:
    """Load Mini-IPIP mapping from artifacts directory (fail closed)."""
    mapping_path = artifacts_dir / "mini_ipip_mapping.json"
    return load_mini_ipip_mapping(mapping_path)


def _safe_pearson(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    floor: float = -1.0,
) -> float:
    """Compute Pearson r with a deterministic floor for invalid outputs."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            r, _ = stats.pearsonr(y_true, y_pred)
    except (ValueError, TypeError, FloatingPointError):
        return floor
    if not np.isfinite(r):
        return floor
    return float(r)


# ---------------------------------------------------------------------------
# Sparsity helpers
# ---------------------------------------------------------------------------

def _apply_sparsity_for_tuning(
    X: pd.DataFrame,
    item_info: dict,
    config: dict,
    rng: np.random.Generator,
    mini_ipip_items: Optional[dict[str, list[str]]] = None,
) -> pd.DataFrame:
    """Apply training-consistent sparsity to data for tuning."""
    sparsity_cfg = config.get("sparsity", {})
    if not sparsity_cfg.get("enabled", False):
        return X

    return apply_sparsity_single(
        X.copy(),
        item_info,
        balanced=sparsity_cfg.get("balanced", True),
        focused=sparsity_cfg.get("focused", False),
        mini_ipip_items=mini_ipip_items,
        include_mini_ipip=sparsity_cfg.get("include_mini_ipip", True),
        include_imbalanced=sparsity_cfg.get("include_imbalanced", False),
        min_items_per_domain=sparsity_cfg.get("min_items_per_domain", 4),
        min_total_items=sparsity_cfg.get("min_total_items", 20),
        max_total_items=sparsity_cfg.get("max_total_items", 40),
        rng=rng,
    )


# ---------------------------------------------------------------------------
# XGBoost model creation
# ---------------------------------------------------------------------------

def _create_xgb_model(
    quantile: float,
    params: dict,
    n_jobs: int = 1,
    early_stopping_rounds: Optional[int] = None,
) -> xgb.XGBRegressor:
    """Create an XGBoost quantile regression model."""
    kwargs = dict(
        objective="reg:quantileerror",
        quantile_alpha=quantile,
        n_estimators=params.get("n_estimators", DEFAULT_PARAMS["n_estimators"]),
        max_depth=params.get("max_depth", DEFAULT_PARAMS["max_depth"]),
        learning_rate=params.get("learning_rate", DEFAULT_PARAMS["learning_rate"]),
        reg_alpha=params.get("reg_alpha", DEFAULT_PARAMS["reg_alpha"]),
        reg_lambda=params.get("reg_lambda", DEFAULT_PARAMS["reg_lambda"]),
        subsample=params.get("subsample", DEFAULT_PARAMS["subsample"]),
        colsample_bytree=params.get("colsample_bytree", DEFAULT_PARAMS["colsample_bytree"]),
        min_child_weight=params.get("min_child_weight", DEFAULT_PARAMS["min_child_weight"]),
        random_state=42,
        n_jobs=n_jobs,
        missing=np.nan,
    )
    if early_stopping_rounds is not None:
        kwargs["early_stopping_rounds"] = early_stopping_rounds
    return xgb.XGBRegressor(**kwargs)


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def _run_optuna_tuning(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    y_val_pct: pd.DataFrame,
    n_trials: int,
    item_info: dict,
    config: dict,
    mini_ipip_items: Optional[dict[str, list[str]]] = None,
    parallel_trials: int = 1,
) -> dict:
    """Run Optuna hyperparameter optimization.

    Objective: weighted blend of sparse-20 and full-50 q50 correlations.
    Only trains q50 models during tuning for efficiency.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as exc:
        raise RuntimeError(
            "Optuna not installed. Install with: pip install optuna "
            "before running stage 06 tuning."
        ) from exc

    configured_n_jobs = coerce_positive_int(
        config.get("_xgb_n_jobs", 1),
        label="config._xgb_n_jobs",
    )
    xgb_n_jobs = max(1, configured_n_jobs // max(parallel_trials, 1))
    if parallel_trials > 1:
        log.info(
            "Parallel trials: %d (XGBoost n_jobs per trial: %d, total cores: %d)",
            parallel_trials, xgb_n_jobs, configured_n_jobs,
        )

    # Subsample training data for tuning speed
    tune_max_rows = 50_000
    if len(X_train) > tune_max_rows:
        subsample_idx = np.random.default_rng(42).choice(
            len(X_train), size=tune_max_rows, replace=False
        )
        X_train_tune = X_train.iloc[subsample_idx].reset_index(drop=True)
        y_train_tune = y_train.iloc[subsample_idx].reset_index(drop=True)
        log.info("Subsampled training data for tuning: %d -> %d", len(X_train), tune_max_rows)
    else:
        X_train_tune = X_train
        y_train_tune = y_train

    # Pre-compute sparsity masks to avoid recomputing each trial
    n_cached_masks = 5
    cached_train_masks: list[pd.DataFrame] = []
    cached_val_es_masks: list[pd.DataFrame] = []
    cached_val_sparse20_masks: list[pd.DataFrame] = []

    sparsity_enabled = config.get("sparsity", {}).get("enabled", False)
    sparse20_eval_enabled = bool(item_info.get("item_pool"))
    if sparsity_enabled:
        log.info("Pre-computing %d sparsity masks for tuning...", n_cached_masks)
        for mask_idx in range(n_cached_masks):
            train_rng = np.random.default_rng(42 + mask_idx * 2)
            val_rng = np.random.default_rng(42 + mask_idx * 2 + 1)
            cached_train_masks.append(
                _apply_sparsity_for_tuning(
                    X_train_tune.copy(),
                    item_info,
                    config,
                    train_rng,
                    mini_ipip_items=mini_ipip_items,
                )
            )
            cached_val_es_masks.append(
                _apply_sparsity_for_tuning(
                    X_val.copy(),
                    item_info,
                    config,
                    val_rng,
                    mini_ipip_items=mini_ipip_items,
                )
            )
    else:
        cached_train_masks = [X_train_tune]
        cached_val_es_masks = [X_val]

    if sparse20_eval_enabled:
        for mask_idx in range(n_cached_masks):
            sparse20_rng = np.random.default_rng(9_000 + mask_idx)
            cached_val_sparse20_masks.append(
                apply_adaptive_sparsity_balanced(
                    X_val.copy(),
                    item_info,
                    min_items_per_domain=4,
                    min_total_items=20,
                    max_total_items=20,
                    rng=sparse20_rng,
                )
            )

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }

        # Use pre-computed masks (cycle through cached variants)
        train_mask_idx = trial.number % max(len(cached_train_masks), 1)
        X_tr = cached_train_masks[train_mask_idx]
        X_vl_es = cached_val_es_masks[train_mask_idx]
        X_vl_sparse20 = (
            cached_val_sparse20_masks[trial.number % len(cached_val_sparse20_masks)]
            if sparse20_eval_enabled and cached_val_sparse20_masks
            else X_val
        )

        # Train q50 model per domain; score at sparse-20 and full-50.
        correlations_sparse: list[float] = []
        correlations_full: list[float] = []

        for domain in DOMAINS:
            score_col = f"{domain}_score"
            pct_col = f"{domain}_percentile"
            if score_col not in y_train_tune.columns or pct_col not in y_val_pct.columns:
                continue

            model = _create_xgb_model(
                0.5,
                params,
                n_jobs=xgb_n_jobs,
                early_stopping_rounds=25,
            )
            model.fit(
                X_tr, y_train_tune[score_col],
                eval_set=[(X_vl_es, y_val[score_col])],
                verbose=False,
            )

            y_true_pct = y_val_pct[pct_col].values

            y_pred_sparse_raw = model.predict(X_vl_sparse20)
            y_pred_sparse_pct = raw_score_to_percentile(y_pred_sparse_raw, domain)
            r_sparse = _safe_pearson(y_true_pct, y_pred_sparse_pct)
            correlations_sparse.append(r_sparse)

            y_pred_full_raw = model.predict(X_val)
            y_pred_full_pct = raw_score_to_percentile(y_pred_full_raw, domain)
            r_full = _safe_pearson(y_true_pct, y_pred_full_pct)
            correlations_full.append(r_full)

        if not correlations_sparse and not correlations_full:
            return float("-inf")

        # Deployment-aligned objective:
        # - sparse20 available: prioritize sparse-20 with full-50 guardrail
        # - sparse20 unavailable (explicit override): full-50 only objective
        if sparse20_eval_enabled and correlations_sparse:
            mean_sparse = float(np.mean(correlations_sparse))
            min_sparse = float(min(correlations_sparse))
            mean_full = float(np.mean(correlations_full)) if correlations_full else mean_sparse

            sparse_penalty = 2.0 * max(0.0, 0.85 - min_sparse)
            full_penalty = 1.0 * max(0.0, 0.95 - mean_full)
            composite = (0.80 * mean_sparse + 0.20 * mean_full) - sparse_penalty - full_penalty
        else:
            if not correlations_full:
                return float("-inf")
            mean_full = float(np.mean(correlations_full))
            min_full = float(min(correlations_full))

            # Conservative full-50-only fallback objective.
            full_penalty = 1.5 * max(0.0, 0.90 - min_full)
            composite = mean_full - full_penalty

        if trial.number % 10 == 0:
            gc.collect()

        return composite

    # Run optimization
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    log.info("Starting Optuna optimization with %d trials (parallel=%d)...", n_trials, parallel_trials)
    if sparse20_eval_enabled:
        log.info(
            "Objective: 0.80*mean_r_sparse20 + 0.20*mean_r_full "
            "- 2.0*max(0,0.85-min_r_sparse20) - 1.0*max(0,0.95-mean_r_full)"
        )
    else:
        log.info(
            "Objective (full-50 fallback): mean_r_full - 1.5*max(0,0.90-min_r_full)"
        )
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, n_jobs=parallel_trials, show_progress_bar=True)
    elapsed = time.time() - t0

    log.info("Optimization complete in %.1f seconds", elapsed)
    log.info("Best composite score: %.4f", study.best_value)
    log.info("Best params: %s", study.best_params)

    # Log trial history
    log.info("Trial history (top 10):")
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float("-inf"), reverse=True)
    for i, trial in enumerate(sorted_trials[:10]):
        log.info("  #%d (trial %d): objective=%.4f", i + 1, trial.number, trial.value if trial.value is not None else float("nan"))

    return {**DEFAULT_PARAMS, **study.best_params}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for IPIP-BFFM XGBoost models"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=200,
        help="Number of Optuna trials (default: 200)",
    )
    parser.add_argument(
        "--parallel-trials",
        type=int,
        default=1,
        help=(
            "Number of Optuna trials to run concurrently. Each trial gets "
            "n_jobs/parallel_trials XGBoost threads. (default: 1)"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file for sparsity settings (relative to PACKAGE_ROOT)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for tuned params JSON (default: artifacts/tuned_params.json)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data directory with train/val/item_info (overrides config.data_dir)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Artifacts directory for Mini-IPIP mapping and default output path",
    )
    parser.add_argument(
        "--allow-no-sparse20-objective",
        action="store_true",
        help=(
            "Allow tuning to continue with full-50-only objective when sparse-20 "
            "objective inputs are unavailable (not recommended for production tuning)"
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help=(
            "Explicit XGBoost thread count for tuning. If omitted: "
            "config.training.n_jobs, then $BFFM_XGB_N_JOBS, then os.cpu_count()."
        ),
    )
    add_provenance_args(parser)
    args = parser.parse_args()

    # Load config
    config: dict[str, Any] = {}
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = PACKAGE_ROOT / config_path
        if not config_path.exists():
            log.error("Config file not found: %s", config_path)
            return 1
        try:
            import yaml
        except ImportError:
            log.error("PyYAML not installed. Install with: pip install pyyaml")
            return 1
        with open(config_path) as f:
            config = yaml.safe_load(f)
        log.info("Loaded config: %s", config.get("name", config_path.name))
    else:
        log.info("No config specified; using default sparsity settings (disabled)")

    config_data_dir = Path(config.get("data_dir", DEFAULT_DATA_DIR))
    config_artifacts_dir = Path(config.get("artifacts_dir", DEFAULT_ARTIFACTS_DIR))

    data_dir_raw = args.data_dir if args.data_dir is not None else config_data_dir
    artifacts_dir_raw = (
        args.artifacts_dir if args.artifacts_dir is not None else config_artifacts_dir
    )
    data_dir = data_dir_raw if data_dir_raw.is_absolute() else PACKAGE_ROOT / data_dir_raw
    artifacts_dir = (
        artifacts_dir_raw
        if artifacts_dir_raw.is_absolute()
        else PACKAGE_ROOT / artifacts_dir_raw
    )

    output_path = (
        Path(args.output) if args.output else artifacts_dir / "tuned_params.json"
    )
    if not output_path.is_absolute():
        output_path = PACKAGE_ROOT / output_path

    log.info("=" * 60)
    log.info("IPIP-BFFM Hyperparameter Tuning")
    log.info("=" * 60)
    log.info("Trials: %d (parallel: %d)", args.trials, args.parallel_trials)
    log.info("Config: %s", args.config or "none (defaults)")
    log.info("Data dir: %s", data_dir)
    log.info("Artifacts dir: %s", artifacts_dir)
    log.info("Output: %s", output_path)
    sparsity_cfg = config.get("sparsity", {})
    log.info("Sparsity enabled: %s", sparsity_cfg.get("enabled", False))

    training_cfg_raw = config.get("training", {})
    training_cfg = training_cfg_raw if isinstance(training_cfg_raw, dict) else {}
    n_jobs_source = "cpu_count"
    try:
        if args.n_jobs is not None:
            xgb_n_jobs = coerce_positive_int(args.n_jobs, label="--n-jobs")
            n_jobs_source = "cli"
        elif training_cfg.get("n_jobs") is not None:
            xgb_n_jobs = coerce_positive_int(
                training_cfg.get("n_jobs"),
                label="training.n_jobs",
            )
            n_jobs_source = "config.training.n_jobs"
        else:
            xgb_n_jobs, n_jobs_source = resolve_default_xgb_n_jobs()
    except ValueError as e:
        log.error("Invalid XGBoost n_jobs setting: %s", e)
        return 1
    config["_xgb_n_jobs"] = xgb_n_jobs
    log.info("XGBoost n_jobs: %d (source=%s)", xgb_n_jobs, n_jobs_source)

    # Load data
    log.info("Loading data...")
    train_path = data_dir / "train.parquet"
    val_path = data_dir / "val.parquet"
    test_path = data_dir / "test.parquet"

    if not train_path.exists() or not val_path.exists():
        log.error("Data files not found in %s", data_dir)
        log.error("Expected: train.parquet, val.parquet")
        log.error("Run the data preparation pipeline first.")
        return 1

    try:
        train_sha256 = file_sha256(train_path)
        val_sha256 = file_sha256(val_path)
        test_sha256: str | None = file_sha256(test_path) if test_path.exists() else None
    except OSError as e:
        log.error("Failed to hash tuning inputs: %s", e)
        return 1

    split_signature: str | None = None
    if test_sha256 is not None:
        try:
            split_signature = _build_split_signature(
                train_sha256=train_sha256,
                val_sha256=val_sha256,
                test_sha256=test_sha256,
            )
        except ValueError as e:
            log.error("Failed to compute split signature for tuning inputs: %s", e)
            return 1

    train_df = _load_parquet(train_path)
    val_df = _load_parquet(val_path)
    log.info("Train: %d rows", len(train_df))
    log.info("Val: %d rows", len(val_df))

    try:
        X_train, y_train, _ = _prepare_features_targets(train_df)
        X_val, y_val, y_val_pct = _prepare_features_targets(val_df)
    except ValueError as e:
        log.error("Invalid train/val data schema for tuning: %s", e)
        return 1
    log.info("Features: %d items", X_train.shape[1])

    # Load item info for sparse-20 objective and sparsity masking
    item_info: dict = {}
    item_info_sha256: str | None = None
    mini_ipip_items: Optional[dict[str, list[str]]] = None
    try:
        item_info, item_info_sha256 = _load_item_info(
            data_dir,
            expected_source_sha256=train_sha256,
        )
        log.info("Loaded item info (%d items in pool)", len(item_info.get("item_pool", [])))
    except (FileNotFoundError, ValueError) as e:
        if isinstance(e, ValueError):
            log.error("Item info unavailable: %s", e)
            log.error(
                "Aborting tuning because item ranking metadata is malformed. "
                "Re-run stage 05 (make correlations) for this data split."
            )
            return 1
        if not args.allow_no_sparse20_objective:
            log.error("Item info unavailable: %s", e)
            log.error(
                "Aborting tuning because sparse-20 objective cannot be evaluated. "
                "Run stage 05 (make correlations), or pass --allow-no-sparse20-objective "
                "for explicit full-50-only fallback."
            )
            return 1
        log.warning("Item info unavailable: %s", e)
        log.warning("Proceeding with explicit full-50-only fallback objective.")
        if sparsity_cfg.get("enabled", False):
            log.warning("Disabling sparsity masks for tuning due to missing item info.")
            config["sparsity"] = {"enabled": False}

    sparsity_cfg = config.get("sparsity", {})
    if sparsity_cfg.get("enabled", False) and sparsity_cfg.get("include_mini_ipip", True):
        try:
            mini_ipip_items = _load_mini_ipip_mapping(artifacts_dir)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
            log.error("Mini-IPIP mapping unavailable: %s", e)
            log.error(
                "Aborting tuning because sparsity.include_mini_ipip=true requires a valid "
                "artifacts/mini_ipip_mapping.json."
            )
            return 1
        log.info("Loaded Mini-IPIP mapping (%d domains)", len(mini_ipip_items))

    # Run tuning
    try:
        best_params = _run_optuna_tuning(
            X_train, y_train,
            X_val, y_val, y_val_pct,
            n_trials=args.trials,
            item_info=item_info,
            config=config,
            mini_ipip_items=mini_ipip_items,
            parallel_trials=args.parallel_trials,
        )
    except RuntimeError as e:
        log.error("%s", e)
        return 1

    # Save results
    log.info("Saving tuned parameters to %s", output_path)
    provenance = build_provenance(
        Path(__file__).name,
        args=args,
        rng_seed=42,
        extra={
            "n_trials": args.trials,
            "parallel_trials": args.parallel_trials,
            "config": args.config,
            "xgb_n_jobs": xgb_n_jobs,
            "xgb_n_jobs_source": n_jobs_source,
            "train_sha256": train_sha256,
            "val_sha256": val_sha256,
            "test_sha256": test_sha256,
            "split_signature": split_signature,
            "item_info_sha256": item_info_sha256,
        },
    )

    payload = {
        "hyperparameters": best_params,
        "provenance": provenance,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    original_path = output_path.parent / (output_path.stem + ".original.json")
    shutil.copy2(output_path, original_path)
    log.info("Saved original tuned parameters to %s", original_path)

    log.info("Tuned parameters:")
    for k, v in best_params.items():
        log.info("  %s: %s", k, v)

    log.info("=" * 60)
    log.info("Tuning complete!")
    log.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
