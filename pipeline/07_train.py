#!/usr/bin/env python3
"""
Train 15 XGBoost quantile regression models (5 domains x 3 quantiles) for
IPIP-BFFM adaptive personality assessment.

Each model predicts a raw domain score (1-5 scale) from a [n_samples, 50]
float32 feature array where NaN indicates missing (unanswered) items.
XGBoost natively handles NaN as missing values.

Sparsity augmentation randomly masks items to NaN during training so the
models learn to predict from partial responses. Multiple augmentation passes
produce different sparsity patterns for each copy of the training data.

Usage:
    python pipeline/07_train.py --config configs/reference.yaml
    python pipeline/07_train.py --config configs/ablation_none.yaml
"""

import sys
import gc
import json
import hashlib
import logging
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import xgboost as xgb

from lib.constants import (
    DOMAINS,
    DOMAIN_LABELS,
    ITEM_COLUMNS,
    ITEMS_PER_DOMAIN,
    QUANTILES,
    QUANTILE_NAMES,
    DEFAULT_PARAMS,
    DEFAULT_EARLY_STOPPING_ROUNDS,
    DEFAULT_STAGE07_CV_FOLDS,
    DEFAULT_LOCAL_CV_PARALLEL_FOLDS,
)
from lib.scoring import raw_score_to_percentile
from lib.provenance import build_provenance, add_provenance_args, relative_to_root
from lib.item_info import load_item_info_strict, file_sha256, load_training_report
from lib.mini_ipip import load_mini_ipip_mapping
from lib.parallelism import coerce_positive_int, resolve_default_xgb_n_jobs
from lib.provenance_checks import (
    build_split_signature as _build_split_signature,
    verify_split_metadata_hash_lock,
)
from lib.sparsity import apply_sparsity_single

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_ARTIFACTS_DIR = Path("artifacts")


# ============================================================================
# Data loading
# ============================================================================

def _load_parquet(path: Path) -> pd.DataFrame:
    """Load a parquet file, raising if it does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_parquet(path)


def _prepare_features_targets(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into X (items), y (raw scores), y_pct (percentiles).

    Returns:
        X: Feature matrix [n_samples, 50] with item responses (float64, NaN for missing)
        y: Target matrix with columns {domain}_score (raw 1-5 scale)
        y_pct: Percentile matrix with columns {domain}_percentile (0-100 scale)
    """
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
            "Training requires full Big-5 schema (50 items + 5 score + 5 percentile columns). "
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
        raise ValueError("Training data has zero valid rows after target/percentile filtering.")

    n_missing = X.isna().sum().sum()
    total = X.shape[0] * X.shape[1]
    log.info(
        "  Features: %d items, Targets: %d domains, Samples: %d, Missing: %d (%.1f%%)",
        len(item_cols), len(target_cols), len(X), n_missing, 100 * n_missing / max(total, 1),
    )

    return X, y, y_pct


def _load_item_info(
    data_dir: Path,
    *,
    expected_source_sha256: str | None = None,
) -> dict:
    """Load item info JSON for sparsity augmentation.

    Requires data/processed/item_info.json (stage 05 output, camelCase)
    to enforce strict provenance and avoid stale exported fallbacks.
    """
    primary = data_dir / "item_info.json"
    return load_item_info_strict(
        primary,
        require_first_item=True,
        expected_source_sha256=expected_source_sha256,
    )


def _load_mini_ipip_mapping(artifacts_dir: Path) -> dict[str, list[str]]:
    """Load Mini-IPIP item mapping from artifacts (fail closed)."""
    mapping_path = artifacts_dir / "mini_ipip_mapping.json"
    return load_mini_ipip_mapping(mapping_path)


def _extract_split_strata(df: pd.DataFrame) -> Optional[pd.Series]:
    """Return row-aligned split strata after the same target validity filtering."""
    if "split_stratum" not in df.columns:
        return None

    target_cols = [f"{d}_score" for d in DOMAINS if f"{d}_score" in df.columns]
    pct_cols = [f"{d}_percentile" for d in DOMAINS if f"{d}_percentile" in df.columns]
    if not target_cols or not pct_cols:
        return None

    valid_mask = df[target_cols].notna().all(axis=1) & df[pct_cols].notna().all(axis=1)
    return df.loc[valid_mask, "split_stratum"].reset_index(drop=True)


def _stable_json_sha256(payload: Any) -> str:
    """Compute deterministic SHA-256 of a JSON-serializable payload."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_locked_params(
    path: Path,
    *,
    require_provenance: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load locked hyperparameters + payload metadata from JSON file."""
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Locked params file must be a JSON object: {path}")
    if "hyperparameters" not in payload:
        raise ValueError(
            f"Locked params file {path} is missing 'hyperparameters' key."
        )
    params = payload["hyperparameters"]
    if not isinstance(params, dict) or not params:
        raise ValueError(
            f"Locked params file {path} has invalid 'hyperparameters' payload (expected non-empty object)."
        )
    if require_provenance and not isinstance(payload.get("provenance"), dict):
        raise ValueError(
            f"Locked params file {path} is missing required 'provenance' object. "
            "Run `make tune` to regenerate artifacts/tuned_params.json."
        )
    return params, payload


def _normalize_sha256_hex_strict(
    value: Any,
    *,
    label: str,
    allow_none: bool = False,
) -> str | None:
    """Normalize and validate SHA-256 hex values used for hash-lock checks."""
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{label} is required and must be a 64-char SHA-256 hex value.")
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string SHA-256 value.")
    normalized = value.strip().lower()
    if not normalized:
        if allow_none:
            return None
        raise ValueError(f"{label} is required and must be non-empty.")
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError(f"{label} must be a 64-char SHA-256 hex value.")
    return normalized


def _verify_locked_params_hash_lock(
    params_source: dict[str, Any],
    *,
    lock_policy: str,
    reference_model_dir: Path | None,
    train_sha256: str,
    val_sha256: str,
    split_signature: str | None,
    item_info_sha256: str | None,
) -> None:
    """Fail closed when hyperparameter lock policy requirements are not met."""
    if lock_policy == "strict_data_hash":
        if params_source.get("mode") != "config_locked_params":
            return

        provenance = params_source.get("payload_provenance")
        if not isinstance(provenance, dict):
            raise ValueError(
                "Locked params are missing provenance hash lock data. "
                "Run `make tune` to regenerate artifacts/tuned_params.json."
            )

        expected_train = _normalize_sha256_hex_strict(
            provenance.get("train_sha256"),
            label="tuned_params.provenance.train_sha256",
        )
        expected_val = _normalize_sha256_hex_strict(
            provenance.get("val_sha256"),
            label="tuned_params.provenance.val_sha256",
        )
        expected_item_info = _normalize_sha256_hex_strict(
            provenance.get("item_info_sha256"),
            label="tuned_params.provenance.item_info_sha256",
        )
        expected_split = _normalize_sha256_hex_strict(
            provenance.get("split_signature"),
            label="tuned_params.provenance.split_signature",
            allow_none=True,
        )

        actual_train = _normalize_sha256_hex_strict(train_sha256, label="train_sha256")
        actual_val = _normalize_sha256_hex_strict(val_sha256, label="val_sha256")
        actual_item_info = _normalize_sha256_hex_strict(
            item_info_sha256,
            label="item_info_sha256",
        )
        actual_split = _normalize_sha256_hex_strict(
            split_signature,
            label="split_signature",
            allow_none=True,
        )

        mismatches: list[str] = []
        if expected_train != actual_train:
            mismatches.append("train_sha256 mismatch")
        if expected_val != actual_val:
            mismatches.append("val_sha256 mismatch")
        if expected_item_info != actual_item_info:
            mismatches.append("item_info_sha256 mismatch")
        if expected_split is not None and expected_split != actual_split:
            mismatches.append("split_signature mismatch")

        if mismatches:
            details = ", ".join(mismatches)
            raise ValueError(
                "Locked params provenance does not match current training inputs "
                f"({details}). Re-run `make tune` for this data split."
            )
        return

    if lock_policy == "reference_model_hash":
        if reference_model_dir is None:
            raise ValueError(
                "hyperparameters.lock_policy=reference_model_hash requires "
                "hyperparameters.reference_model_dir in the training config."
            )

        actual_params_sha256 = _normalize_sha256_hex_strict(
            params_source.get("hyperparameters_sha256"),
            label="hyperparameters_sha256",
        )
        actual_source_sha256 = _normalize_sha256_hex_strict(
            params_source.get("file_sha256"),
            label="hyperparameters_source_sha256",
            allow_none=True,
        )

        try:
            reference_report, reference_report_path = load_training_report(reference_model_dir)
        except FileNotFoundError as e:
            raise ValueError(
                "Reference model training report not found for hyperparameter lock policy "
                f"at {reference_model_dir}. Run `make train 1` first."
            ) from e
        except (OSError, json.JSONDecodeError, ValueError) as e:
            raise ValueError(
                "Failed to load reference model training report "
                f"from {reference_model_dir}: {e}"
            ) from e

        reference_data = reference_report.get("data", {})
        if not isinstance(reference_data, dict):
            reference_data = {}

        expected_params_sha256 = _normalize_sha256_hex_strict(
            reference_data.get("hyperparameters_sha256"),
            label=f"{reference_report_path}.data.hyperparameters_sha256",
        )
        expected_source_sha256 = _normalize_sha256_hex_strict(
            reference_data.get("hyperparameters_source_sha256"),
            label=f"{reference_report_path}.data.hyperparameters_source_sha256",
            allow_none=True,
        )

        mismatches: list[str] = []
        if expected_params_sha256 != actual_params_sha256:
            mismatches.append("hyperparameters_sha256 mismatch")
        if (
            expected_source_sha256 is not None
            and actual_source_sha256 is not None
            and expected_source_sha256 != actual_source_sha256
        ):
            mismatches.append("hyperparameters_source_sha256 mismatch")

        if mismatches:
            details = ", ".join(mismatches)
            raise ValueError(
                "Hyperparameter lock policy check failed against reference model "
                f"({details}). reference_model_dir={reference_model_dir}. "
                "Run `make train 1` and ensure all runs use the same params source."
            )
        return

    raise ValueError(
        "Unsupported hyperparameter lock policy "
        f"{lock_policy!r}. Expected one of: strict_data_hash, reference_model_hash."
    )


# ============================================================================
# Sparsity augmentation (config-aware wrappers around lib.sparsity)
# ============================================================================

def _apply_sparsity_single(
    X: pd.DataFrame,
    item_info: dict,
    config: dict,
    mini_ipip_items: Optional[dict[str, list[str]]] = None,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Dispatch wrapper: apply the appropriate sparsity method to X based on config."""
    sparsity_cfg = config.get("sparsity", {})
    return apply_sparsity_single(
        X, item_info,
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


def _apply_multipass_sparsity(
    X: pd.DataFrame,
    y: pd.DataFrame,
    item_info: dict,
    config: dict,
    n_passes: int = 3,
    base_seed: int = 42,
    mini_ipip_items: Optional[dict[str, list[str]]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply sparsity augmentation with multiple passes.

    Each pass uses a different random seed, producing different sparsity masks.
    All passes are concatenated, effectively multiplying the training data.
    """
    X_parts: list[pd.DataFrame] = []

    for pass_idx in range(n_passes):
        rng = np.random.default_rng(base_seed + pass_idx)
        X_aug = _apply_sparsity_single(
            X.copy(), item_info, config,
            mini_ipip_items=mini_ipip_items,
            rng=rng,
        )
        X_parts.append(X_aug)

    X_combined = pd.concat(X_parts, ignore_index=True)
    y_combined = pd.concat([y] * n_passes, ignore_index=True)

    # Shuffle
    shuffle_idx = np.random.default_rng(base_seed).permutation(len(X_combined))
    X_combined = X_combined.iloc[shuffle_idx].reset_index(drop=True)
    y_combined = y_combined.iloc[shuffle_idx].reset_index(drop=True)

    return X_combined, y_combined


# ============================================================================
# XGBoost model creation and training
# ============================================================================

def _create_xgb_model(
    quantile: float,
    params: dict,
    n_jobs: int = 1,
    early_stopping_rounds: Optional[int] = None,
    gpu: bool = False,
) -> xgb.XGBRegressor:
    """Create XGBoost quantile regression model."""
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
    if gpu:
        kwargs["device"] = "cuda"
        kwargs["n_jobs"] = 1
    if early_stopping_rounds is not None:
        kwargs["early_stopping_rounds"] = early_stopping_rounds
    return xgb.XGBRegressor(**kwargs)


def _train_single_domain(
    domain: str,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    params: dict,
    n_jobs: int = 1,
    X_eval: Optional[pd.DataFrame] = None,
    y_eval: Optional[pd.DataFrame] = None,
    gpu: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Train q05/q50/q95 models for a single domain. Thread-safe."""
    use_early_stopping = X_eval is not None and y_eval is not None
    score_col = f"{domain}_score"
    y_domain = y_train[score_col]
    models: dict[str, Any] = {}

    log.info("  Training %s (raw score prediction)...", DOMAIN_LABELS[domain])
    t0 = time.time()

    for q in QUANTILES:
        q_name = QUANTILE_NAMES.get(q, f"q{int(q*100)}")
        model = _create_xgb_model(
            q, params,
            n_jobs=n_jobs,
            early_stopping_rounds=DEFAULT_EARLY_STOPPING_ROUNDS if use_early_stopping else None,
            gpu=gpu,
        )

        if use_early_stopping:
            y_domain_eval = y_eval[score_col]
            model.fit(
                X_train, y_domain,
                eval_set=[(X_eval, y_domain_eval)],
                verbose=False,
            )
            best_iter = getattr(model, "best_iteration", params.get("n_estimators", DEFAULT_PARAMS["n_estimators"]))
            log.info("    %s/%s trained (best_iteration=%d)", DOMAIN_LABELS[domain], q_name, best_iter)
        else:
            model.fit(X_train, y_domain)
            log.info("    %s/%s trained", DOMAIN_LABELS[domain], q_name)

        models[q_name] = model

    elapsed = time.time() - t0
    log.info("    %s complete in %.1fs", DOMAIN_LABELS[domain], elapsed)
    return domain, models


def _train_domain_models(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    params: dict,
    n_jobs: int = 1,
    X_eval: Optional[pd.DataFrame] = None,
    y_eval: Optional[pd.DataFrame] = None,
    parallel_domains: int = 1,
    gpu: bool = False,
) -> dict[str, dict[str, Any]]:
    """Train XGBoost models for each domain on raw scores (1-5 scale).

    Returns dict mapping domain -> {q05: model, q50: model, q95: model}.
    When parallel_domains > 1, domains train concurrently using threads
    (XGBoost releases the GIL). Each domain gets n_jobs // parallel_domains
    XGBoost threads.
    """
    use_early_stopping = X_eval is not None and y_eval is not None
    if use_early_stopping:
        log.info("  Early stopping: %d train, %d eval (external)", len(X_train), len(X_eval))

    missing_score_cols = [f"{d}_score" for d in DOMAINS if f"{d}_score" not in y_train.columns]
    if missing_score_cols:
        raise ValueError(
            "Training targets missing required domain score columns: "
            + ", ".join(missing_score_cols)
        )

    n_parallel = min(parallel_domains, len(DOMAINS), max(1, n_jobs))
    if gpu:
        n_parallel = 1
    xgb_threads_per_domain = max(1, n_jobs // max(n_parallel, 1))

    if n_parallel > 1:
        log.info(
            "  Parallel domains: %d (%d XGBoost threads each, %d total cores)",
            n_parallel, xgb_threads_per_domain, n_jobs,
        )
        domain_models: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(
            max_workers=n_parallel,
            thread_name_prefix="train-domain",
        ) as executor:
            futures = {
                executor.submit(
                    _train_single_domain,
                    domain, X_train, y_train, params,
                    n_jobs=xgb_threads_per_domain,
                    X_eval=X_eval, y_eval=y_eval,
                    gpu=gpu,
                ): domain
                for domain in DOMAINS
            }
            errors: list[tuple[str, Exception]] = []
            for future in as_completed(futures):
                domain = futures[future]
                try:
                    _, models = future.result()
                    domain_models[domain] = models
                except Exception as exc:
                    errors.append((domain, exc))
                    log.error("  %s training failed: %s", DOMAIN_LABELS[domain], exc)
            if errors:
                failed = ", ".join(
                    f"{DOMAIN_LABELS[d]} ({type(e).__name__}: {e})"
                    for d, e in errors
                )
                raise RuntimeError(
                    f"{len(errors)}/{len(DOMAINS)} domain(s) failed to train: {failed}"
                )
        return domain_models

    domain_models = {}
    for domain in DOMAINS:
        _, models = _train_single_domain(
            domain, X_train, y_train, params,
            n_jobs=n_jobs, X_eval=X_eval, y_eval=y_eval,
            gpu=gpu,
        )
        domain_models[domain] = models

    return domain_models


# ============================================================================
# Evaluation
# ============================================================================

def _evaluate_domain_models(
    domain_models: dict[str, dict[str, Any]],
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    calibration_params: Optional[dict[str, dict[str, float]]] = None,
) -> dict[str, dict[str, float]]:
    """Evaluate models on test set (percentile space)."""
    metrics: dict[str, dict[str, float]] = {}

    all_true_parts: list[np.ndarray] = []
    all_pred_parts: list[np.ndarray] = []
    all_lower_parts: list[np.ndarray] = []
    all_upper_parts: list[np.ndarray] = []

    missing_pct_cols = [f"{d}_percentile" for d in DOMAINS if f"{d}_percentile" not in y_test.columns]
    if missing_pct_cols:
        raise ValueError(
            "Evaluation targets missing required domain percentile columns: "
            + ", ".join(missing_pct_cols)
        )

    for domain, models in domain_models.items():
        pct_col = f"{domain}_percentile"

        y_true = y_test[pct_col].values

        q_lower_raw = models["q05"].predict(X_test)
        q50_raw = models["q50"].predict(X_test)
        q_upper_raw = models["q95"].predict(X_test)

        q_lower_pred = raw_score_to_percentile(q_lower_raw, domain)
        q50_pred = raw_score_to_percentile(q50_raw, domain)
        q_upper_pred = raw_score_to_percentile(q_upper_raw, domain)

        # Enforce monotonicity
        stacked = np.sort(np.stack([q_lower_pred, q50_pred, q_upper_pred], axis=0), axis=0)
        q_lower_pred, q50_pred, q_upper_pred = stacked[0], stacked[1], stacked[2]

        # Apply calibration
        if calibration_params is not None and domain in calibration_params:
            scale = float(calibration_params[domain].get("scale_factor", 1.0))
            if scale != 1.0:
                half_width = 0.5 * (q_upper_pred - q_lower_pred) * scale
                q_lower_pred = np.clip(q50_pred - half_width, 0.0, 100.0)
                q_upper_pred = np.clip(q50_pred + half_width, 0.0, 100.0)
                calibrated = np.sort(np.stack([q_lower_pred, q50_pred, q_upper_pred], axis=0), axis=0)
                q_lower_pred, q50_pred, q_upper_pred = calibrated[0], calibrated[1], calibrated[2]

        all_true_parts.append(y_true)
        all_pred_parts.append(q50_pred)
        all_lower_parts.append(q_lower_pred)
        all_upper_parts.append(q_upper_pred)

        r, p = stats.pearsonr(y_true, q50_pred)
        mse = mean_squared_error(y_true, q50_pred)
        mae = mean_absolute_error(y_true, q50_pred)
        coverage = float(np.mean((y_true >= q_lower_pred) & (y_true <= q_upper_pred)))
        interval_width = float(np.mean(q_upper_pred - q_lower_pred))
        within_5 = float(np.mean(np.abs(y_true - q50_pred) <= 5))
        within_10 = float(np.mean(np.abs(y_true - q50_pred) <= 10))

        metrics[domain] = {
            "pearson_r": float(r),
            "p_value": float(p),
            "mae": float(mae),
            "rmse": float(np.sqrt(mse)),
            "coverage_90": coverage,
            "interval_width": interval_width,
            "within_5_pct": within_5,
            "within_10_pct": within_10,
        }

    # Overall metrics
    if all_true_parts:
        all_true = np.concatenate(all_true_parts)
        all_pred = np.concatenate(all_pred_parts)
        all_lower = np.concatenate(all_lower_parts)
        all_upper = np.concatenate(all_upper_parts)

        overall_r = float(stats.pearsonr(all_true, all_pred)[0])
        overall_mae = float(mean_absolute_error(all_true, all_pred))
        overall_rmse = float(np.sqrt(mean_squared_error(all_true, all_pred)))
        overall_coverage = float(np.mean((all_true >= all_lower) & (all_true <= all_upper)))
        overall_within_5 = float(np.mean(np.abs(all_true - all_pred) <= 5))
        overall_within_10 = float(np.mean(np.abs(all_true - all_pred) <= 10))

        metrics["overall"] = {
            "pearson_r": overall_r,
            "mae": overall_mae,
            "rmse": overall_rmse,
            "coverage_90": overall_coverage,
            "within_5_pct": overall_within_5,
            "within_10_pct": overall_within_10,
        }

    return metrics


def _compute_calibration_params(
    domain_models: dict[str, dict[str, Any]],
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Compute calibration parameters (observed coverage + scale factor) per domain."""
    def _scale_for_coverage(coverage: float) -> float:
        if coverage < 0.85:
            return 0.90 / max(coverage, 0.5)
        if coverage > 0.95:
            return 0.90 / coverage
        return 1.0

    calibration: dict[str, dict[str, float]] = {}

    missing_pct_cols = [f"{d}_percentile" for d in DOMAINS if f"{d}_percentile" not in y_val.columns]
    if missing_pct_cols:
        raise ValueError(
            "Calibration targets missing required domain percentile columns: "
            + ", ".join(missing_pct_cols)
        )

    for domain, models in domain_models.items():
        pct_col = f"{domain}_percentile"

        y_true = y_val[pct_col].values

        q_lower_raw = models["q05"].predict(X_val)
        q50_raw = models["q50"].predict(X_val)
        q_upper_raw = models["q95"].predict(X_val)

        q_lower_pred = raw_score_to_percentile(q_lower_raw, domain)
        q50_pred = raw_score_to_percentile(q50_raw, domain)
        q_upper_pred = raw_score_to_percentile(q_upper_raw, domain)

        stacked = np.sort(np.stack([q_lower_pred, q50_pred, q_upper_pred], axis=0), axis=0)
        q_lower_pred, q50_pred, q_upper_pred = stacked[0], stacked[1], stacked[2]

        coverage = float(np.mean((y_true >= q_lower_pred) & (y_true <= q_upper_pred)))
        scale = _scale_for_coverage(coverage)

        calibration[domain] = {
            "observed_coverage": coverage,
            "scale_factor": float(scale),
        }

    return calibration


def _calibration_from_metrics(
    metrics: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Build per-domain calibration params from metric payloads.

    Uses observed ``coverage_90`` and the same scaling policy as
    ``_compute_calibration_params``.
    """
    calibration: dict[str, dict[str, float]] = {}
    for domain in DOMAINS:
        coverage_raw = metrics.get(domain, {}).get("coverage_90")
        if not isinstance(coverage_raw, (int, float, np.integer, np.floating)):
            continue
        coverage = float(coverage_raw)
        if not np.isfinite(coverage):
            continue
        if coverage < 0.85:
            scale = 0.90 / max(coverage, 0.5)
        elif coverage > 0.95:
            scale = 0.90 / coverage
        else:
            scale = 1.0
        calibration[domain] = {
            "observed_coverage": coverage,
            "scale_factor": float(scale),
        }
    return calibration


# ============================================================================
# Cross-validation
# ============================================================================

def _run_cv_fold(
    *,
    fold: int,
    n_folds: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X: pd.DataFrame,
    y: pd.DataFrame,
    y_pct: pd.DataFrame,
    item_info: dict,
    config: dict,
    params: dict,
    n_jobs: int,
    mini_ipip_items: Optional[dict[str, list[str]]],
    parallel_domains: int,
    random_state: int,
    augment_sparsity: bool,
    n_augmentation_passes: int,
    gpu: bool,
) -> dict:
    """Train and evaluate a single CV fold."""
    fold_label = f"[CV fold {fold + 1}/{n_folds}]"

    log.info("%s Starting", fold_label)

    X_train_fold = X.iloc[train_idx].copy()
    X_test_fold = X.iloc[test_idx].copy()
    y_train_fold = y.iloc[train_idx].copy()
    y_pct_test_fold = y_pct.iloc[test_idx].copy()

    # Split early-stopping eval set BEFORE augmentation.
    X_eval_es: Optional[pd.DataFrame] = None
    y_eval_es: Optional[pd.DataFrame] = None
    X_fit_pre = X_train_fold
    y_fit_pre = y_train_fold

    if augment_sparsity and n_augmentation_passes > 1:
        X_fit_pre, X_eval_es, y_fit_pre, y_eval_es = train_test_split(
            X_train_fold, y_train_fold, test_size=0.15, random_state=random_state + fold
        )

    # Apply sparsity augmentation.
    if augment_sparsity:
        if n_augmentation_passes > 1:
            X_train_aug, y_train_fold = _apply_multipass_sparsity(
                X_fit_pre, y_fit_pre, item_info, config,
                n_passes=n_augmentation_passes,
                base_seed=random_state,
                mini_ipip_items=mini_ipip_items,
            )
            if X_eval_es is not None:
                eval_rng = np.random.default_rng(999 + fold)
                X_eval_es = _apply_sparsity_single(
                    X_eval_es.copy(), item_info, config,
                    mini_ipip_items=mini_ipip_items,
                    rng=eval_rng,
                )
        else:
            X_train_aug = _apply_sparsity_single(
                X_train_fold.copy(), item_info, config,
                mini_ipip_items=mini_ipip_items,
            )
    else:
        X_train_aug = X_train_fold

    log.info("%s Train: %d rows (after augmentation)", fold_label, len(X_train_aug))

    domain_models = _train_domain_models(
        X_train_aug, y_train_fold, params,
        n_jobs=n_jobs,
        X_eval=X_eval_es, y_eval=y_eval_es,
        parallel_domains=parallel_domains,
        gpu=gpu,
    )

    metrics = _evaluate_domain_models(domain_models, X_test_fold, y_pct_test_fold)

    if "overall" in metrics:
        log.info(
            "%s Overall r = %.3f, Coverage = %.1f%%, Within 5 pct = %.1f%%",
            fold_label,
            metrics["overall"]["pearson_r"],
            metrics["overall"]["coverage_90"] * 100,
            metrics["overall"]["within_5_pct"] * 100,
        )

    return {"fold": fold + 1, "metrics": metrics}


def _run_cross_validation_robustness(
    X: pd.DataFrame,
    y: pd.DataFrame,
    y_pct: pd.DataFrame,
    item_info: dict,
    config: dict,
    params: dict,
    n_folds: int = DEFAULT_STAGE07_CV_FOLDS,
    n_jobs: int = 1,
    mini_ipip_items: Optional[dict[str, list[str]]] = None,
    strata: Optional[pd.Series] = None,
    parallel_domains: int = 1,
    parallel_folds: int = 1,
    gpu: bool = False,
) -> dict:
    """Run fixed-hyperparameter cross-validation for robustness estimation.

    Split BEFORE augmentation to prevent respondent leakage.
    """
    sparsity_cfg = config.get("sparsity", {})
    augment_sparsity = sparsity_cfg.get("enabled", False)
    n_augmentation_passes = sparsity_cfg.get("n_augmentation_passes", 1)
    random_state = config.get("training", {}).get("random_state", 42)

    split_iter: Any
    if strata is not None:
        strata_values = np.asarray(strata)
        if len(strata_values) != len(X):
            raise ValueError(
                "Strata length mismatch for cross-validation: "
                f"len(strata)={len(strata_values)}, len(X)={len(X)}"
            )
        unique, counts = np.unique(strata_values, return_counts=True)
        if len(unique) < 2:
            raise ValueError(
                "Cross-validation stratification requires at least 2 strata, "
                f"found {len(unique)}."
            )
        min_count = int(np.min(counts))
        if min_count < n_folds:
            raise ValueError(
                "Cross-validation stratification requires each stratum to appear at least "
                f"n_folds times (n_folds={n_folds}, min_count={min_count})."
            )
        outer_cv = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_state,
        )
        split_iter = outer_cv.split(X, strata_values)
    else:
        outer_cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        split_iter = outer_cv.split(X)

    split_plan = list(split_iter)
    requested_parallel_folds = max(parallel_folds, 1)
    n_parallel_folds = min(requested_parallel_folds, n_folds)
    if gpu:
        n_parallel_folds = 1
    fold_n_jobs = max(1, n_jobs // max(n_parallel_folds, 1))
    effective_parallel_domains = min(max(parallel_domains, 1), len(DOMAINS), fold_n_jobs)

    if requested_parallel_folds != n_parallel_folds:
        log.info(
            "  CV fold parallelism request clamped from %d to %d.",
            requested_parallel_folds,
            n_parallel_folds,
        )
    if parallel_domains > effective_parallel_domains:
        log.warning(
            "  Requested parallel_domains=%d with %d XGBoost thread(s) per fold; "
            "domain-level training will clamp to %d worker(s) per fold.",
            parallel_domains,
            fold_n_jobs,
            effective_parallel_domains,
        )

    if n_parallel_folds > 1:
        log.info(
            "  Parallel CV folds: %d (%d XGBoost threads budget per fold, %d total cores)",
            n_parallel_folds,
            fold_n_jobs,
            n_jobs,
        )
        fold_results_by_idx: dict[int, dict] = {}
        errors: list[tuple[int, Exception]] = []
        with ThreadPoolExecutor(
            max_workers=n_parallel_folds,
            thread_name_prefix="cv-fold",
        ) as executor:
            futures = {
                executor.submit(
                    _run_cv_fold,
                    fold=fold,
                    n_folds=n_folds,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    X=X,
                    y=y,
                    y_pct=y_pct,
                    item_info=item_info,
                    config=config,
                    params=params,
                    n_jobs=fold_n_jobs,
                    mini_ipip_items=mini_ipip_items,
                    parallel_domains=parallel_domains,
                    random_state=random_state,
                    augment_sparsity=augment_sparsity,
                    n_augmentation_passes=n_augmentation_passes,
                    gpu=gpu,
                ): fold
                for fold, (train_idx, test_idx) in enumerate(split_plan)
            }
            for future in as_completed(futures):
                fold = futures[future]
                try:
                    fold_results_by_idx[fold] = future.result()
                    gc.collect()
                except Exception as exc:
                    errors.append((fold, exc))
                    log.error("CV fold %d failed: %s", fold + 1, exc)
        if errors:
            failed = ", ".join(
                f"fold {fold + 1} ({type(exc).__name__}: {exc})"
                for fold, exc in errors
            )
            raise RuntimeError(
                f"{len(errors)}/{len(split_plan)} cross-validation fold(s) failed: {failed}"
            )
        fold_results = [fold_results_by_idx[idx] for idx in range(len(split_plan))]
    else:
        fold_results = []
        for fold, (train_idx, test_idx) in enumerate(split_plan):
            fold_results.append(
                _run_cv_fold(
                    fold=fold,
                    n_folds=n_folds,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    X=X,
                    y=y,
                    y_pct=y_pct,
                    item_info=item_info,
                    config=config,
                    params=params,
                    n_jobs=n_jobs,
                    mini_ipip_items=mini_ipip_items,
                    parallel_domains=parallel_domains,
                    random_state=random_state,
                    augment_sparsity=augment_sparsity,
                    n_augmentation_passes=n_augmentation_passes,
                    gpu=gpu,
                )
            )
            gc.collect()

    # Aggregate results
    aggregate: dict[str, dict[str, dict[str, float]]] = {"overall": {}}
    metric_names = ["pearson_r", "mae", "rmse", "coverage_90", "within_5_pct", "within_10_pct"]

    for metric in metric_names:
        values = [f["metrics"]["overall"][metric] for f in fold_results if "overall" in f["metrics"]]
        if values:
            aggregate["overall"][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    for domain in DOMAINS:
        aggregate[domain] = {}
        for metric in metric_names:
            values = [
                f["metrics"].get(domain, {}).get(metric, np.nan)
                for f in fold_results
            ]
            values = [v for v in values if not np.isnan(v)]
            if values:
                aggregate[domain][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }

    return {
        "fold_results": fold_results,
        "aggregate": aggregate,
        "n_folds": n_folds,
        "parallel_folds": n_parallel_folds,
        "xgb_n_jobs_total": n_jobs,
        "xgb_n_jobs_per_fold": fold_n_jobs,
    }


# ============================================================================
# Model validation
# ============================================================================

def _validate_model_outputs(
    domain_models: dict[str, dict[str, Any]],
    X_test: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Verify each model produces variable outputs (detect collapsed models)."""
    results: dict[str, dict[str, Any]] = {}
    critical_failures: list[str] = []

    n_samples = min(1000, len(X_test))
    X_sample = X_test.iloc[:n_samples]
    pred_cache: dict[str, np.ndarray] = {}

    for domain, models in domain_models.items():
        for q_name, model in models.items():
            model_key = f"{domain}_{q_name}"
            preds = model.predict(X_sample)
            pred_cache[model_key] = preds

            pred_std = float(np.std(preds))
            pred_min = float(np.min(preds))
            pred_max = float(np.max(preds))
            pred_range = pred_max - pred_min

            issues: list[str] = []

            if pred_std < 0.1:
                issues.append(f"Low variance: std={pred_std:.4f}")
                critical_failures.append(f"{model_key}: collapsed model (std={pred_std:.4f})")

            if pred_range < 0.5:
                issues.append(f"Narrow range: {pred_min:.2f} - {pred_max:.2f}")

            if pred_min < 0.5 or pred_max > 5.5:
                issues.append(f"Out of range: [{pred_min:.2f}, {pred_max:.2f}]")

            results[model_key] = {
                "passed": len(issues) == 0,
                "std": pred_std,
                "min": pred_min,
                "max": pred_max,
                "range": pred_range,
                "issues": issues,
            }

    # Check quantile ordering
    for domain, models in domain_models.items():
        if all(q in models for q in ["q05", "q50", "q95"]):
            mean_q05 = float(np.mean(pred_cache[f"{domain}_q05"]))
            mean_q50 = float(np.mean(pred_cache[f"{domain}_q50"]))
            mean_q95 = float(np.mean(pred_cache[f"{domain}_q95"]))

            if not (mean_q05 < mean_q50 < mean_q95):
                issue = f"Quantile ordering violated: q05={mean_q05:.2f}, q50={mean_q50:.2f}, q95={mean_q95:.2f}"
                results[f"{domain}_ordering"] = {"passed": False, "issue": issue}

    if critical_failures:
        error_msg = "MODEL VALIDATION FAILED - Collapsed models detected:\n"
        error_msg += "\n".join(f"  - {f}" for f in critical_failures)
        raise ValueError(error_msg)

    return results


def _threshold_for_domain(value: Any, domain: str) -> Optional[float]:
    """Resolve scalar-or-dict threshold configs for a domain."""
    if value is None:
        return None
    if isinstance(value, dict):
        raw = value.get(domain, value.get("default"))
    else:
        raw = value
    if raw is None:
        return None
    if isinstance(raw, (int, float, np.integer, np.floating)):
        return float(raw)
    raise TypeError(f"Invalid threshold value for {domain}: {raw!r}")


def _aggregate_metric_runs(
    runs: list[dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    """Average numeric metrics across repeated evaluation runs.

    If a metric is missing or non-finite in any run for a section, the
    aggregated value is set to NaN so downstream quality gates fail closed.
    """
    if not runs:
        return {}

    aggregated: dict[str, dict[str, float]] = {}
    section_names = sorted(
        {
            section
            for run in runs
            for section in run.keys()
            if isinstance(run.get(section), dict)
        }
    )

    for section in section_names:
        section_runs_all = [run.get(section) for run in runs]
        section_runs = [sr for sr in section_runs_all if isinstance(sr, dict)]
        if not section_runs:
            continue

        missing_section_in_any_run = len(section_runs) != len(runs)
        metric_names = sorted(
            {
                metric
                for section_run in section_runs
                for metric in section_run.keys()
            }
        )
        if missing_section_in_any_run:
            aggregated[section] = {metric: float("nan") for metric in metric_names}
            continue

        metric_means: dict[str, float] = {}
        for metric in metric_names:
            values: list[float] = []
            valid = True
            for section_run in section_runs:
                raw_value = section_run.get(metric)
                if not isinstance(raw_value, (int, float, np.integer, np.floating)):
                    valid = False
                    break
                value = float(raw_value)
                if not np.isfinite(value):
                    valid = False
                    break
                values.append(value)

            # Fail closed on partial/missing/non-finite metrics across masks.
            metric_means[metric] = float(np.mean(values)) if valid else float("nan")

        if metric_means:
            aggregated[section] = metric_means

    return aggregated


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train IPIP-BFFM XGBoost quantile regression models"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (relative to PACKAGE_ROOT)",
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="Override hyperparameters JSON file (takes precedence over config)",
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
        help="Artifacts directory with Mini-IPIP mapping (overrides config.artifacts_dir)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help=(
            "Explicit XGBoost thread count. If omitted: "
            "config.training.n_jobs, then $BFFM_XGB_N_JOBS, then os.cpu_count()."
        ),
    )
    parser.add_argument(
        "--parallel-domains",
        type=int,
        default=1,
        help=(
            "Number of domains to train concurrently. Each domain gets "
            "n_jobs/parallel_domains XGBoost threads. Recommended: 5 (one per domain). "
            "(default: 1)"
        ),
    )
    parser.add_argument(
        "--cv-parallel-folds",
        type=int,
        default=None,
        help=(
            "Number of CV folds to run concurrently during stage-07 cross-validation. "
            "Each fold gets a share of the total XGBoost thread budget. "
            f"(default: {DEFAULT_LOCAL_CV_PARALLEL_FOLDS} on CPU, forced to 1 on GPU)"
        ),
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Use GPU acceleration (device='cuda')",
    )
    add_provenance_args(parser)
    args = parser.parse_args()

    # Load config
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

    config_name = config.get("name", config_path.stem)
    output_dir = PACKAGE_ROOT / config.get("output_dir", f"models/{config_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

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

    hp_cfg_raw = config.get("hyperparameters", {})
    hp_cfg = hp_cfg_raw if isinstance(hp_cfg_raw, dict) else {}

    lock_policy_raw = hp_cfg.get("lock_policy", "strict_data_hash")
    if not isinstance(lock_policy_raw, str) or not lock_policy_raw.strip():
        log.error(
            "Invalid hyperparameters.lock_policy in %s. Expected a non-empty string.",
            config_path,
        )
        return 1
    lock_policy = lock_policy_raw.strip()
    valid_lock_policies = {"strict_data_hash", "reference_model_hash"}
    if lock_policy not in valid_lock_policies:
        log.error(
            "Unsupported hyperparameters.lock_policy=%r. Expected one of: %s",
            lock_policy,
            ", ".join(sorted(valid_lock_policies)),
        )
        return 1

    reference_model_dir_raw = hp_cfg.get("reference_model_dir")
    reference_model_dir: Path | None = None
    if reference_model_dir_raw is not None:
        if not isinstance(reference_model_dir_raw, str) or not reference_model_dir_raw.strip():
            log.error(
                "Invalid hyperparameters.reference_model_dir in %s. Expected a non-empty path string.",
                config_path,
            )
            return 1
        reference_model_dir = Path(reference_model_dir_raw)
        if not reference_model_dir.is_absolute():
            reference_model_dir = PACKAGE_ROOT / reference_model_dir

    if lock_policy == "reference_model_hash" and reference_model_dir is None:
        log.error(
            "hyperparameters.lock_policy=reference_model_hash requires "
            "hyperparameters.reference_model_dir."
        )
        return 1

    # Load hyperparameters (--params CLI flag takes precedence over config)
    params_source: dict[str, Any] = {
        "mode": "default_params",
        "path": None,
        "file_sha256": None,
        "payload_sha256": None,
        "hyperparameters_sha256": None,
        "payload_provenance": None,
        "lock_policy": lock_policy,
        "reference_model_dir": relative_to_root(reference_model_dir) if reference_model_dir else None,
    }
    if args.params:
        params_path = Path(args.params)
        if not params_path.is_absolute():
            params_path = PACKAGE_ROOT / params_path
        if not params_path.exists():
            log.error("Params file not found: %s", params_path)
            return 1
        try:
            params, params_payload = _load_locked_params(
                params_path,
                require_provenance=False,
            )
        except (OSError, json.JSONDecodeError, ValueError) as e:
            log.error("Invalid params file at %s: %s", params_path, e)
            return 1
        payload_provenance = params_payload.get("provenance")
        log.info("Loaded hyperparameters from --params %s", params_path)
        params_source = {
            "mode": "cli_params_override",
            "path": relative_to_root(params_path),
            "file_sha256": file_sha256(params_path),
            "payload_sha256": _stable_json_sha256(params_payload),
            "hyperparameters_sha256": _stable_json_sha256(params),
            "payload_provenance": payload_provenance if isinstance(payload_provenance, dict) else None,
            "lock_policy": lock_policy,
            "reference_model_dir": relative_to_root(reference_model_dir) if reference_model_dir else None,
        }
    else:
        locked_params_path_str = hp_cfg.get("locked_params", None)

        if locked_params_path_str:
            locked_params_path = Path(locked_params_path_str)
            if not locked_params_path.is_absolute():
                locked_params_path = PACKAGE_ROOT / locked_params_path
            if not locked_params_path.exists():
                log.error("Locked params file not found: %s", locked_params_path)
                return 1
            try:
                params, params_payload = _load_locked_params(
                    locked_params_path,
                    require_provenance=True,
                )
            except (OSError, json.JSONDecodeError, ValueError) as e:
                log.error("Invalid locked params file at %s: %s", locked_params_path, e)
                return 1
            payload_provenance = params_payload.get("provenance")
            log.info("Loaded locked hyperparameters from %s", locked_params_path)
            params_source = {
                "mode": "config_locked_params",
                "path": relative_to_root(locked_params_path),
                "file_sha256": file_sha256(locked_params_path),
                "payload_sha256": _stable_json_sha256(params_payload),
                "hyperparameters_sha256": _stable_json_sha256(params),
                "payload_provenance": payload_provenance if isinstance(payload_provenance, dict) else None,
                "lock_policy": lock_policy,
                "reference_model_dir": relative_to_root(reference_model_dir) if reference_model_dir else None,
            }

            # Detect manual overrides vs original tuned values
            original_path = locked_params_path.parent / (locked_params_path.stem + ".original.json")
            if original_path.exists():
                try:
                    with open(original_path) as f:
                        original_payload = json.load(f)
                    original_params = original_payload.get("hyperparameters", {})
                    overrides = {}
                    for key in sorted(set(params) | set(original_params)):
                        if params.get(key) != original_params.get(key):
                            overrides[key] = {"tuned": original_params.get(key), "used": params.get(key)}
                    if overrides:
                        log.info("Hyperparameter overrides detected vs tuned values:")
                        for k, v in overrides.items():
                            log.info("  %s: %s -> %s", k, v["tuned"], v["used"])
                        params_source["overrides"] = overrides
                except (OSError, json.JSONDecodeError, ValueError) as e:
                    log.warning("Could not read original params sidecar %s: %s", original_path, e)
        else:
            params = DEFAULT_PARAMS.copy()
            log.info("Using default hyperparameters")
            params_source = {
                "mode": "default_params",
                "path": None,
                "file_sha256": None,
                "payload_sha256": None,
                "hyperparameters_sha256": _stable_json_sha256(params),
                "payload_provenance": None,
                "lock_policy": lock_policy,
                "reference_model_dir": relative_to_root(reference_model_dir) if reference_model_dir else None,
            }

    sparsity_cfg = config.get("sparsity", {})
    training_cfg_raw = config.get("training", {})
    training_cfg = training_cfg_raw if isinstance(training_cfg_raw, dict) else {}
    validation_cfg = config.get("validation", {})
    cv_folds_raw = training_cfg.get("cv_folds")
    if cv_folds_raw is None:
        cv_folds = DEFAULT_STAGE07_CV_FOLDS
        cv_folds_source = "lib.constants.DEFAULT_STAGE07_CV_FOLDS"
    else:
        try:
            cv_folds = int(cv_folds_raw)
        except (TypeError, ValueError):
            log.error("Invalid training.cv_folds in %s: %r", config_path, cv_folds_raw)
            return 1
        if cv_folds < 0:
            log.error("Invalid training.cv_folds in %s: expected >= 0, got %r", config_path, cv_folds_raw)
            return 1
        cv_folds_source = "config.training.cv_folds"
    random_state = training_cfg.get("random_state", 42)
    n_augmentation_passes = sparsity_cfg.get("n_augmentation_passes", 1)
    try:
        if args.cv_parallel_folds is not None:
            cv_parallel_folds = coerce_positive_int(
                args.cv_parallel_folds,
                label="--cv-parallel-folds",
            )
            cv_parallel_folds_source = "cli"
        else:
            cv_parallel_folds = DEFAULT_LOCAL_CV_PARALLEL_FOLDS
            cv_parallel_folds_source = "lib.constants.DEFAULT_LOCAL_CV_PARALLEL_FOLDS"
    except ValueError as e:
        log.error("Invalid CV fold parallelism setting: %s", e)
        return 1
    if args.gpu and cv_parallel_folds > 1:
        log.warning("GPU mode disables fold-level CV parallelism; forcing cv_parallel_folds=1.")
        cv_parallel_folds = 1
        cv_parallel_folds_source = "forced_for_gpu"
    if cv_folds <= 1:
        cv_parallel_folds = 1
        cv_parallel_folds_source = "disabled_without_cv"
    else:
        requested_cv_parallel_folds = cv_parallel_folds
        cv_parallel_folds = min(cv_parallel_folds, cv_folds)
        if cv_parallel_folds != requested_cv_parallel_folds:
            cv_parallel_folds_source = f"{cv_parallel_folds_source}_clamped_to_cv_folds"
    training_cfg_effective = dict(training_cfg)
    training_cfg_effective["cv_folds"] = cv_folds
    training_cfg_effective["cv_parallel_folds"] = cv_parallel_folds
    xgb_n_jobs_source = "cpu_count"
    try:
        if args.n_jobs is not None:
            xgb_n_jobs = coerce_positive_int(args.n_jobs, label="--n-jobs")
            xgb_n_jobs_source = "cli"
        elif training_cfg.get("n_jobs") is not None:
            xgb_n_jobs = coerce_positive_int(
                training_cfg.get("n_jobs"),
                label="training.n_jobs",
            )
            xgb_n_jobs_source = "config.training.n_jobs"
        else:
            xgb_n_jobs, xgb_n_jobs_source = resolve_default_xgb_n_jobs()
    except ValueError as e:
        log.error("Invalid XGBoost n_jobs setting: %s", e)
        return 1

    log.info("=" * 60)
    log.info("IPIP-BFFM Model Training")
    log.info("=" * 60)
    log.info("Config: %s", config_name)
    log.info("Output: %s", output_dir)
    log.info("Data dir: %s", data_dir)
    log.info("Artifacts dir: %s", artifacts_dir)
    log.info("Sparsity: %s", "enabled" if sparsity_cfg.get("enabled", False) else "disabled")
    if sparsity_cfg.get("enabled", False):
        log.info("  Focused: %s", sparsity_cfg.get("focused", False))
        log.info("  Mini-IPIP: %s", sparsity_cfg.get("include_mini_ipip", True))
        log.info("  Imbalanced: %s", sparsity_cfg.get("include_imbalanced", False))
        log.info("  Augmentation passes: %d", n_augmentation_passes)
    log.info("CV folds: %s (source=%s)", cv_folds if cv_folds > 0 else "None", cv_folds_source)
    log.info(
        "CV parallel folds: %d (source=%s)",
        cv_parallel_folds,
        cv_parallel_folds_source,
    )
    log.info("XGBoost n_jobs: %d (source=%s)", xgb_n_jobs, xgb_n_jobs_source)
    if args.gpu:
        log.info("GPU mode: enabled (device=cuda)")
        try:
            xgb.XGBRegressor(n_estimators=1, device="cuda").fit(
                np.zeros((2, 1)), np.zeros(2),
            )
        except Exception as e:
            log.error("GPU check failed: %s", e)
            log.error("Ensure CUDA is installed and a GPU is available, or remove --gpu.")
            return 1
    if args.parallel_domains > 1:
        log.info("Parallel domains: %d (%d XGBoost threads each)", args.parallel_domains, max(1, xgb_n_jobs // args.parallel_domains))
    if cv_folds > 1:
        fold_thread_budget = max(1, xgb_n_jobs // max(cv_parallel_folds, 1))
        effective_domain_workers_per_fold = min(
            max(args.parallel_domains, 1),
            len(DOMAINS),
            fold_thread_budget,
        )
        if args.parallel_domains > effective_domain_workers_per_fold:
            log.warning(
                "Requested parallel_domains=%d with cv_parallel_folds=%d and n_jobs=%d; "
                "each fold has %d XGBoost thread(s), so domain-level training will clamp "
                "to %d worker(s) per fold.",
                args.parallel_domains,
                cv_parallel_folds,
                xgb_n_jobs,
                fold_thread_budget,
                effective_domain_workers_per_fold,
            )
    log.info("Hyperparameters:")
    for k, v in params.items():
        log.info("  %s: %s", k, v)
    log.info(
        "Hyperparameter provenance: mode=%s, params_sha256=%s",
        params_source["mode"],
        str(params_source["hyperparameters_sha256"])[:12],
    )
    log.info("Hyperparameter lock policy: %s", lock_policy)
    if reference_model_dir is not None:
        log.info("  Reference model dir: %s", reference_model_dir)
    if params_source.get("path"):
        log.info(
            "  Source file: %s (sha256=%s)",
            params_source["path"],
            str(params_source.get("file_sha256", ""))[:12],
        )

    # Load data
    log.info("Step 1: Loading data...")
    train_path = data_dir / "train.parquet"
    val_path = data_dir / "val.parquet"
    test_path = data_dir / "test.parquet"
    split_metadata_path = data_dir / "split_metadata.json"

    if not train_path.exists() or not val_path.exists():
        log.error("Data files not found in %s", data_dir)
        return 1

    train_sha256 = file_sha256(train_path)
    val_sha256 = file_sha256(val_path)
    test_sha256: Optional[str] = None
    split_signature: Optional[str] = None
    if test_path.exists():
        test_sha256 = file_sha256(test_path)
        split_signature = _build_split_signature(
            train_sha256=train_sha256,
            val_sha256=val_sha256,
            test_sha256=test_sha256,
        )
    else:
        log.warning(
            "test.parquet not found in %s; split signature will be unavailable in training report.",
            data_dir,
        )

    split_metadata_sha256: Optional[str] = None
    if split_metadata_path.exists():
        try:
            split_metadata_sha256 = file_sha256(split_metadata_path)
            verified_split_signature = verify_split_metadata_hash_lock(
                split_metadata_path,
                train_sha256=train_sha256,
                val_sha256=val_sha256,
                test_sha256=test_sha256,
            )
        except (OSError, json.JSONDecodeError, ValueError, FileNotFoundError) as e:
            log.error("Invalid split metadata at %s: %s", split_metadata_path, e)
            return 1

        if split_signature is not None and verified_split_signature != split_signature:
            log.error(
                "Internal split signature mismatch before training: computed=%s verified=%s",
                split_signature,
                verified_split_signature,
            )
            return 1
        log.info(
            "  Split metadata verified (train/val/test hashes match %s)",
            split_metadata_path,
        )
    else:
        log.warning(
            "split_metadata.json not found in %s; proceeding without stage-04 hash lock.",
            data_dir,
        )

    train_df = _load_parquet(train_path)
    val_df = _load_parquet(val_path)
    log.info("  Train: %d rows", len(train_df))
    log.info("  Val: %d rows", len(val_df))

    try:
        X_train, y_train, y_train_pct = _prepare_features_targets(train_df)
        X_val, y_val, y_val_pct = _prepare_features_targets(val_df)
        train_strata = _extract_split_strata(train_df)
        val_strata = _extract_split_strata(val_df)
    except (ValueError, KeyError) as e:
        log.error("Invalid train/val data schema for training: %s", e)
        return 1

    # Load item info for sparsity augmentation and sparse validation gates
    item_info: dict = {}
    item_info_path = data_dir / "item_info.json"
    item_info_sha256: Optional[str] = None
    sparse_gate_cfg = validation_cfg.get("sparse_20", {})
    sparse_gate_enabled = bool(sparse_gate_cfg.get("enabled", False))
    requires_item_info = bool(sparsity_cfg.get("enabled", False) or sparse_gate_enabled)
    if requires_item_info:
        try:
            item_info = _load_item_info(
                data_dir,
                expected_source_sha256=train_sha256,
            )
            log.info("  Item info loaded (%d items)", len(item_info.get("item_pool", [])))
        except (FileNotFoundError, ValueError) as e:
            log.error("Item info not available: %s", e)
            return 1

    if item_info_path.exists():
        try:
            item_info_sha256 = file_sha256(item_info_path)
        except OSError as e:
            log.error("Failed to hash item_info at %s: %s", item_info_path, e)
            return 1

    try:
        _verify_locked_params_hash_lock(
            params_source,
            lock_policy=lock_policy,
            reference_model_dir=reference_model_dir,
            train_sha256=train_sha256,
            val_sha256=val_sha256,
            split_signature=split_signature,
            item_info_sha256=item_info_sha256,
        )
    except ValueError as e:
        log.error("Locked params hash-lock check failed: %s", e)
        return 1

    # Load Mini-IPIP mapping
    mini_ipip_items: Optional[dict[str, list[str]]] = None
    include_mini_ipip = bool(sparsity_cfg.get("include_mini_ipip", True))
    if sparsity_cfg.get("enabled", False) and include_mini_ipip:
        try:
            mini_ipip_items = _load_mini_ipip_mapping(artifacts_dir)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
            log.error("Mini-IPIP mapping unavailable: %s", e)
            log.error(
                "Aborting training because sparsity.include_mini_ipip=true requires a "
                "valid artifacts/mini_ipip_mapping.json."
            )
            return 1
        log.info("  Mini-IPIP mapping loaded (%d domains)", len(mini_ipip_items))

    # Cross-validation robustness analysis
    cv_results = None
    if cv_folds > 1:
        log.info("Step 2: %d-fold cross-validation robustness analysis...", cv_folds)
        X_trainval = pd.concat([X_train, X_val]).reset_index(drop=True)
        y_trainval = pd.concat([y_train, y_val]).reset_index(drop=True)
        y_trainval_pct = pd.concat([y_train_pct, y_val_pct]).reset_index(drop=True)
        strata_trainval: Optional[pd.Series] = None
        if train_strata is not None and val_strata is not None:
            strata_trainval = pd.concat([train_strata, val_strata]).reset_index(drop=True)
            log.info(
                "  Using stratified cross-validation via split_stratum (%d strata)",
                int(strata_trainval.nunique()),
            )
        elif train_strata is None and val_strata is None:
            log.warning("  split_stratum unavailable; falling back to unstratified cross-validation.")
        else:
            log.error(
                "split_stratum presence mismatch between train/val; "
                "cannot safely run cross-validation. Re-run stage 04 prepare."
            )
            return 1

        cv_results = _run_cross_validation_robustness(
            X_trainval, y_trainval, y_trainval_pct,
            item_info, config, params,
            n_folds=cv_folds,
            n_jobs=xgb_n_jobs,
            mini_ipip_items=mini_ipip_items,
            strata=strata_trainval,
            parallel_domains=args.parallel_domains,
            parallel_folds=cv_parallel_folds,
            gpu=args.gpu,
        )

        log.info("CV Results:")
        for metric in ["pearson_r", "coverage_90", "within_5_pct"]:
            if metric in cv_results["aggregate"].get("overall", {}):
                agg = cv_results["aggregate"]["overall"][metric]
                log.info("  %s: %.3f +/- %.3f", metric, agg["mean"], agg["std"])

    # Train final models
    log.info("Step 3: Training final models...")
    t_start = time.time()

    # Apply sparsity augmentation to training data
    if sparsity_cfg.get("enabled", False):
        log.info("  Applying sparsity augmentation...")

        # Split early-stopping eval set BEFORE augmentation
        X_eval_es: Optional[pd.DataFrame] = None
        y_eval_es: Optional[pd.DataFrame] = None
        X_fit_pre = X_train
        y_fit_pre = y_train

        if n_augmentation_passes > 1:
            X_fit_pre, X_eval_es, y_fit_pre, y_eval_es = train_test_split(
                X_train, y_train, test_size=0.15, random_state=random_state
            )
            log.info("  Split early-stopping eval: %d fit, %d eval", len(X_fit_pre), len(X_eval_es))

        if n_augmentation_passes > 1:
            X_train_aug, y_train_aug = _apply_multipass_sparsity(
                X_fit_pre, y_fit_pre, item_info, config,
                n_passes=n_augmentation_passes,
                base_seed=random_state,
                mini_ipip_items=mini_ipip_items,
            )
            if X_eval_es is not None:
                eval_rng = np.random.default_rng(999)
                X_eval_es = _apply_sparsity_single(
                    X_eval_es.copy(), item_info, config,
                    mini_ipip_items=mini_ipip_items,
                    rng=eval_rng,
                )
        else:
            rng = np.random.default_rng(random_state)
            X_train_aug = _apply_sparsity_single(
                X_train.copy(), item_info, config,
                mini_ipip_items=mini_ipip_items,
                rng=rng,
            )
            y_train_aug = y_train
            X_eval_es = None
            y_eval_es = None

        # Log sparsity stats
        n_nan = X_train_aug.isna().sum().sum()
        total = X_train_aug.shape[0] * X_train_aug.shape[1]
        log.info("  Augmented train: %d rows, %.1f%% missing values", len(X_train_aug), 100 * n_nan / total)
    else:
        X_train_aug = X_train
        y_train_aug = y_train
        X_eval_es = None
        y_eval_es = None

    # Train domain models
    domain_models = _train_domain_models(
        X_train_aug, y_train_aug, params,
        n_jobs=xgb_n_jobs,
        X_eval=X_eval_es, y_eval=y_eval_es,
        parallel_domains=args.parallel_domains,
        gpu=args.gpu,
    )

    t_train = time.time() - t_start
    log.info("  Training complete in %.1f seconds", t_train)

    # Validate model outputs
    log.info("Step 4: Validating model outputs...")
    try:
        validation_results = _validate_model_outputs(domain_models, X_val)
        n_passed = sum(1 for v in validation_results.values() if isinstance(v, dict) and v.get("passed", False))
        n_total = sum(1 for v in validation_results.values() if isinstance(v, dict) and "passed" in v)
        log.info("  Model validation: %d/%d checks passed", n_passed, n_total)
    except ValueError as e:
        log.error("  %s", e)
        return 1

    # Evaluate on validation set
    log.info("Step 5: Evaluating on validation set...")
    val_metrics = _evaluate_domain_models(domain_models, X_val, y_val_pct)
    if "overall" in val_metrics:
        log.info("  Overall r = %.4f", val_metrics["overall"]["pearson_r"])
        log.info("  Overall MAE = %.2f", val_metrics["overall"]["mae"])
        log.info("  Coverage 90%% = %.1f%%", val_metrics["overall"]["coverage_90"] * 100)
        log.info("  Within 5 pct = %.1f%%", val_metrics["overall"]["within_5_pct"] * 100)

    for domain in DOMAINS:
        if domain in val_metrics:
            dm = val_metrics[domain]
            log.info("  %s: r=%.4f, MAE=%.2f, coverage=%.1f%%",
                     DOMAIN_LABELS[domain], dm["pearson_r"], dm["mae"], dm["coverage_90"] * 100)

    sparse_val_metrics: dict[str, dict[str, float]] = {}
    sparse_val_runs: list[dict[str, dict[str, float]]] = []
    if sparse_gate_enabled:
        n_sparse_masks = int(sparse_gate_cfg.get("n_masks", 5))
        if n_sparse_masks < 1:
            n_sparse_masks = 1
        log.info(
            "Step 5b: Evaluating sparse 20-item validation gate set (%d masks)...",
            n_sparse_masks,
        )
        for mask_idx in range(n_sparse_masks):
            sparse_rng = np.random.default_rng(random_state + 2024 + mask_idx)
            X_val_sparse = apply_sparsity_single(
                X_val.copy(),
                item_info,
                balanced=True,
                focused=False,
                include_mini_ipip=False,
                include_imbalanced=False,
                min_items_per_domain=4,
                min_total_items=20,
                max_total_items=20,
                rng=sparse_rng,
            )
            sparse_val_runs.append(_evaluate_domain_models(domain_models, X_val_sparse, y_val_pct))

        sparse_val_metrics = _aggregate_metric_runs(sparse_val_runs)
        if "overall" in sparse_val_metrics:
            log.info(
                "  Sparse20 overall r = %.4f, MAE = %.2f, coverage = %.1f%%",
                sparse_val_metrics["overall"]["pearson_r"],
                sparse_val_metrics["overall"]["mae"],
                sparse_val_metrics["overall"]["coverage_90"] * 100,
            )
        for domain in DOMAINS:
            if domain in sparse_val_metrics:
                dm = sparse_val_metrics[domain]
                log.info(
                    "  Sparse20 %s: r=%.4f, MAE=%.2f, coverage=%.1f%%",
                    DOMAIN_LABELS[domain], dm["pearson_r"], dm["mae"], dm["coverage_90"] * 100,
                )

    # Check config quality gates
    min_pearson_r = validation_cfg.get("min_pearson_r")
    min_coverage_90 = validation_cfg.get("min_coverage_90")
    gate_failed = False

    def _valid_float(value: Any) -> Optional[float]:
        if not isinstance(value, (int, float, np.integer, np.floating)):
            return None
        f = float(value)
        if not np.isfinite(f):
            return None
        return f

    has_gates = min_pearson_r is not None or min_coverage_90 is not None
    if has_gates and "overall" not in val_metrics:
        log.error("Quality gate FAILED: no overall validation metrics produced.")
        gate_failed = True

    if min_pearson_r is not None and "overall" in val_metrics:
        actual_r = _valid_float(val_metrics["overall"]["pearson_r"])
        if actual_r is None or actual_r < min_pearson_r:
            log.error(
                "Quality gate FAILED: pearson_r=%s < min_pearson_r=%.4f",
                actual_r, min_pearson_r,
            )
            gate_failed = True

    if min_coverage_90 is not None and "overall" in val_metrics:
        actual_cov = _valid_float(val_metrics["overall"].get("coverage_90"))
        if actual_cov is None or actual_cov < min_coverage_90:
            log.error(
                "Quality gate FAILED: coverage_90=%s < min_coverage_90=%.4f",
                actual_cov, min_coverage_90,
            )
            gate_failed = True

    per_domain_cfg = validation_cfg.get("per_domain", {})
    min_domain_r_cfg = per_domain_cfg.get("min_pearson_r")
    min_domain_cov_cfg = per_domain_cfg.get("min_coverage_90")
    for domain in DOMAINS:
        domain_metrics = val_metrics.get(domain)
        min_domain_r = _threshold_for_domain(min_domain_r_cfg, domain)
        min_domain_cov = _threshold_for_domain(min_domain_cov_cfg, domain)
        if min_domain_r is None and min_domain_cov is None:
            continue
        if domain_metrics is None:
            log.error("Quality gate FAILED: missing full-validation metrics for %s", domain)
            gate_failed = True
            continue
        if min_domain_r is not None:
            actual_r = _valid_float(domain_metrics.get("pearson_r"))
            if actual_r is None or actual_r < min_domain_r:
                log.error(
                    "Quality gate FAILED (full %s): pearson_r=%s < min=%.4f",
                    domain, actual_r, min_domain_r,
                )
                gate_failed = True
        if min_domain_cov is not None:
            actual_cov = _valid_float(domain_metrics.get("coverage_90"))
            if actual_cov is None or actual_cov < min_domain_cov:
                log.error(
                    "Quality gate FAILED (full %s): coverage_90=%s < min=%.4f",
                    domain, actual_cov, min_domain_cov,
                )
                gate_failed = True

    if sparse_gate_enabled:
        sparse_min_pearson_r = sparse_gate_cfg.get("min_pearson_r", min_pearson_r)
        sparse_min_coverage_90 = sparse_gate_cfg.get("min_coverage_90", min_coverage_90)
        if "overall" not in sparse_val_metrics:
            log.error("Quality gate FAILED: no sparse-20 overall validation metrics produced.")
            gate_failed = True
        else:
            if sparse_min_pearson_r is not None:
                actual_r = _valid_float(sparse_val_metrics["overall"].get("pearson_r"))
                if actual_r is None or actual_r < sparse_min_pearson_r:
                    log.error(
                        "Quality gate FAILED (sparse20): pearson_r=%s < min=%.4f",
                        actual_r, sparse_min_pearson_r,
                    )
                    gate_failed = True
            if sparse_min_coverage_90 is not None:
                actual_cov = _valid_float(sparse_val_metrics["overall"].get("coverage_90"))
                if actual_cov is None or actual_cov < sparse_min_coverage_90:
                    log.error(
                        "Quality gate FAILED (sparse20): coverage_90=%s < min=%.4f",
                        actual_cov, sparse_min_coverage_90,
                    )
                    gate_failed = True

        sparse_per_domain_cfg = sparse_gate_cfg.get("per_domain", {})
        sparse_min_domain_r_cfg = sparse_per_domain_cfg.get("min_pearson_r", min_domain_r_cfg)
        sparse_min_domain_cov_cfg = sparse_per_domain_cfg.get("min_coverage_90", min_domain_cov_cfg)
        for domain in DOMAINS:
            min_domain_r = _threshold_for_domain(sparse_min_domain_r_cfg, domain)
            min_domain_cov = _threshold_for_domain(sparse_min_domain_cov_cfg, domain)
            if min_domain_r is None and min_domain_cov is None:
                continue
            domain_metrics = sparse_val_metrics.get(domain)
            if domain_metrics is None:
                log.error("Quality gate FAILED: missing sparse20 metrics for %s", domain)
                gate_failed = True
                continue
            if min_domain_r is not None:
                actual_r = _valid_float(domain_metrics.get("pearson_r"))
                if actual_r is None or actual_r < min_domain_r:
                    log.error(
                        "Quality gate FAILED (sparse20 %s): pearson_r=%s < min=%.4f",
                        domain, actual_r, min_domain_r,
                    )
                    gate_failed = True
            if min_domain_cov is not None:
                actual_cov = _valid_float(domain_metrics.get("coverage_90"))
                if actual_cov is None or actual_cov < min_domain_cov:
                    log.error(
                        "Quality gate FAILED (sparse20 %s): coverage_90=%s < min=%.4f",
                        domain, actual_cov, min_domain_cov,
                    )
                    gate_failed = True

    if gate_failed:
        log.error("Quality gates not met — aborting before saving models.")
        return 1

    # Compute calibration parameters
    log.info("Step 6: Computing calibration parameters...")
    calibration_params_full_50 = _compute_calibration_params(domain_models, X_val, y_val_pct)
    for domain, cal in calibration_params_full_50.items():
        log.info(
            "  Full-50 %s: observed_coverage=%.3f, scale_factor=%.3f",
            DOMAIN_LABELS[domain], cal["observed_coverage"], cal["scale_factor"],
        )

    calibration_params_sparse_20_balanced: dict[str, dict[str, float]] = {}
    if sparse_gate_enabled and sparse_val_metrics:
        calibration_params_sparse_20_balanced = _calibration_from_metrics(sparse_val_metrics)
        for domain, cal in calibration_params_sparse_20_balanced.items():
            log.info(
                "  Sparse-20 %s: observed_coverage=%.3f, scale_factor=%.3f",
                DOMAIN_LABELS[domain], cal["observed_coverage"], cal["scale_factor"],
            )

    # Save models
    log.info("Step 7: Saving models to %s...", output_dir)
    for domain, models in domain_models.items():
        for q_name, model in models.items():
            model_path = output_dir / f"adaptive_{domain}_{q_name}.joblib"
            joblib.dump(model, model_path)
            log.info("  Saved %s", model_path.name)

    # Save calibration params (explicit sparse/full regimes + legacy alias)
    full_calibration_path = output_dir / "calibration_params_full_50.json"
    with open(full_calibration_path, "w") as f:
        json.dump(calibration_params_full_50, f, indent=2)
    log.info("  Saved calibration_params_full_50.json")

    if calibration_params_sparse_20_balanced:
        sparse_calibration_path = output_dir / "calibration_params_sparse_20_balanced.json"
        with open(sparse_calibration_path, "w") as f:
            json.dump(calibration_params_sparse_20_balanced, f, indent=2)
        log.info("  Saved calibration_params_sparse_20_balanced.json")

    # Keep legacy filename for backward compatibility, but ONLY when sparse calibration
    # was actually produced. Writing full-50 params here would silently bias downstream
    # sparse evaluations that fall back to this file when the sparse-specific file is absent.
    if calibration_params_sparse_20_balanced:
        calibration_path = output_dir / "calibration_params.json"
        with open(calibration_path, "w") as f:
            json.dump(calibration_params_sparse_20_balanced, f, indent=2)
        log.info("  Saved calibration_params.json (legacy alias -> sparse_20_balanced)")

    # Save training report
    provenance = build_provenance(
        Path(__file__).name,
        args=args,
        rng_seed=random_state,
        extra={
            "config_name": config_name,
            "split_signature": split_signature,
            "xgb_n_jobs": xgb_n_jobs,
            "xgb_n_jobs_source": xgb_n_jobs_source,
            "parallel_domains": args.parallel_domains,
            "cv_folds": cv_folds,
            "cv_parallel_folds": cv_parallel_folds,
            "gpu": args.gpu,
            "hyperparameters_source_mode": params_source.get("mode"),
            "hyperparameters_sha256": params_source.get("hyperparameters_sha256"),
            "hyperparameters_source_sha256": params_source.get("file_sha256"),
        },
    )

    report = {
        "provenance": provenance,
        "config": {
            "name": config_name,
            "config_path": relative_to_root(config_path),
            "sparsity": sparsity_cfg,
            "training": training_cfg_effective,
            "hyperparameters_lock_policy": lock_policy,
            "hyperparameters_reference_model_dir": (
                relative_to_root(reference_model_dir) if reference_model_dir is not None else None
            ),
            "xgb_n_jobs": xgb_n_jobs,
            "xgb_n_jobs_source": xgb_n_jobs_source,
            "hyperparameters": params,
            "hyperparameters_source": params_source,
        },
        "data": {
            "data_dir": relative_to_root(data_dir),
            "artifacts_dir": relative_to_root(artifacts_dir),
            "train_path": relative_to_root(train_path),
            "val_path": relative_to_root(val_path),
            "test_path": relative_to_root(test_path) if test_path.exists() else None,
            "split_metadata_path": relative_to_root(split_metadata_path) if split_metadata_path.exists() else None,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "train_rows_after_augmentation": len(X_train_aug),
            "n_features": X_train.shape[1],
            "train_sha256": train_sha256,
            "val_sha256": val_sha256,
            "test_sha256": test_sha256,
            "split_signature": split_signature,
            "split_metadata_sha256": split_metadata_sha256,
            "item_info_path": relative_to_root(item_info_path) if item_info_path.exists() else None,
            "item_info_sha256": item_info_sha256,
            "hyperparameters_source_mode": params_source.get("mode"),
            "hyperparameters_source_path": params_source.get("path"),
            "hyperparameters_source_sha256": params_source.get("file_sha256"),
            "hyperparameters_payload_sha256": params_source.get("payload_sha256"),
            "hyperparameters_sha256": params_source.get("hyperparameters_sha256"),
            "hyperparameters_lock_policy": lock_policy,
            "hyperparameters_reference_model_dir": (
                relative_to_root(reference_model_dir) if reference_model_dir is not None else None
            ),
            "xgb_n_jobs": xgb_n_jobs,
            "xgb_n_jobs_source": xgb_n_jobs_source,
        },
        "timing": {
            "training_seconds": round(t_train, 1),
        },
        "validation_metrics": val_metrics,
        "validation_metrics_sparse_20": sparse_val_metrics,
        "validation_metrics_sparse_20_runs": sparse_val_runs,
        "calibration_params": (
            calibration_params_sparse_20_balanced
            if calibration_params_sparse_20_balanced
            else calibration_params_full_50
        ),
        "calibration_params_full_50": calibration_params_full_50,
        "calibration_params_sparse_20_balanced": calibration_params_sparse_20_balanced,
        "model_validation": {
            k: {kk: vv for kk, vv in v.items() if kk != "issues" or vv}
            for k, v in validation_results.items()
            if isinstance(v, dict)
        },
    }

    if cv_results is not None:
        report["cross_validation"] = cv_results

    report_path = output_dir / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("  Saved training_report.json")

    # Summary
    log.info("=" * 60)
    log.info("Training complete!")
    log.info("=" * 60)
    log.info("Models saved to: %s", output_dir)
    log.info("15 models: 5 domains x 3 quantiles (q05, q50, q95)")
    if "overall" in val_metrics:
        log.info("Validation: r=%.4f, MAE=%.2f, coverage=%.1f%%",
                 val_metrics["overall"]["pearson_r"],
                 val_metrics["overall"]["mae"],
                 val_metrics["overall"]["coverage_90"] * 100)

    return 0


if __name__ == "__main__":
    sys.exit(main())
