#!/usr/bin/env python3
"""
Validate trained IPIP-BFFM XGBoost models against held-out test data.

Evaluates at two sparsity levels:
  1. Full 50-item (no masking)
  2. Sparse 20-item (domain-balanced sparsity)

Computes per-domain and overall metrics with bootstrap 95% CIs.
Optionally generates validation plots (scatter, residual, calibration).

Usage:
    python pipeline/08_validate.py --data-dir data/processed/ext_est
    python pipeline/08_validate.py --data-dir data/processed/ext_est --model-dir models/reference --plots --bootstrap-n 2000
"""

import sys
import json
import logging
import time
import argparse
from pathlib import Path
from typing import Any, Optional

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from lib.constants import (
    DOMAINS,
    DOMAIN_LABELS,
    ITEM_COLUMNS,
    ITEMS_PER_DOMAIN,
    QUANTILES,
    QUANTILE_NAMES,
)
from lib.bootstrap import respondent_bootstrap_multi_domain
from lib.scoring import raw_score_to_percentile
from lib.provenance import build_provenance, add_provenance_args, relative_to_root
from lib.item_info import load_item_info_for_model
from lib.provenance_checks import (
    verify_model_data_split_provenance as _verify_model_data_split_provenance,
)
from lib.sparsity import apply_adaptive_sparsity_balanced as apply_sparse_balanced

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_ARTIFACTS_DIR = Path("artifacts")


# ============================================================================
# Data and model loading
# ============================================================================

def _load_models(models_dir: Path) -> dict[str, dict[str, Any]]:
    """Load trained domain models from .joblib files."""
    domain_models: dict[str, dict[str, Any]] = {}

    for domain in DOMAINS:
        domain_models[domain] = {}
        for q_name in ["q05", "q50", "q95"]:
            model_path = models_dir / f"adaptive_{domain}_{q_name}.joblib"
            if model_path.exists():
                domain_models[domain][q_name] = joblib.load(model_path)

    return domain_models


def _check_models_complete(domain_models: dict[str, dict[str, Any]]) -> list[str]:
    """Return list of missing model keys."""
    missing: list[str] = []
    for domain in DOMAINS:
        for q_name in ["q05", "q50", "q95"]:
            if q_name not in domain_models.get(domain, {}):
                missing.append(f"{domain}_{q_name}")
    return missing


def _load_calibration_params(path: Path) -> dict[str, dict[str, float]]:
    """Load a calibration params file.

    Missing files are treated as optional and return an empty mapping.
    Present files must be complete and well-formed for all five domains.
    """
    if not path.exists():
        return {}

    with open(path) as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Calibration file must be a JSON object: {path}")

    expected_domains = set(DOMAINS)
    payload_domains = set(payload.keys())
    missing_domains = sorted(expected_domains - payload_domains)
    extra_domains = sorted(payload_domains - expected_domains)
    if missing_domains or extra_domains:
        details = []
        if missing_domains:
            details.append(f"missing={missing_domains}")
        if extra_domains:
            details.append(f"unexpected={extra_domains}")
        raise ValueError(f"Calibration file has invalid domain coverage ({', '.join(details)}): {path}")

    calibration: dict[str, dict[str, float]] = {}
    for domain in DOMAINS:
        params = payload[domain]
        if not isinstance(params, dict):
            raise ValueError(f"Calibration params for domain '{domain}' must be an object: {path}")
        if "scale_factor" not in params or "observed_coverage" not in params:
            raise ValueError(
                f"Calibration params for domain '{domain}' must include "
                f"'scale_factor' and 'observed_coverage': {path}"
            )
        try:
            scale_factor = float(params["scale_factor"])
            observed_coverage = float(params["observed_coverage"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Calibration params for domain '{domain}' must be numeric: {path}"
            ) from exc
        if not np.isfinite(scale_factor) or scale_factor <= 0.0:
            raise ValueError(
                f"Calibration scale_factor for domain '{domain}' must be finite and > 0: {path}"
            )
        if not np.isfinite(observed_coverage) or observed_coverage < 0.0 or observed_coverage > 1.0:
            raise ValueError(
                f"Calibration observed_coverage for domain '{domain}' must be finite and in [0, 1]: {path}"
            )
        calibration[domain] = {
            "scale_factor": scale_factor,
            "observed_coverage": observed_coverage,
        }
    return calibration


def _load_test_data(data_dir: Path) -> pd.DataFrame:
    """Load held-out test parquet."""
    test_path = data_dir / "test.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")
    return pd.read_parquet(test_path)


def _prepare_features_targets(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into X (items) and y (percentiles + raw scores)."""
    item_cols = list(ITEM_COLUMNS)
    pct_cols = [f"{d}_percentile" for d in DOMAINS]
    score_cols = [f"{d}_score" for d in DOMAINS]

    missing_item_cols = [c for c in item_cols if c not in df.columns]
    missing_pct_cols = [c for c in pct_cols if c not in df.columns]
    missing_score_cols = [c for c in score_cols if c not in df.columns]
    if missing_item_cols or missing_pct_cols or missing_score_cols:
        details = []
        if missing_item_cols:
            details.append("items=" + ",".join(missing_item_cols))
        if missing_score_cols:
            details.append("scores=" + ",".join(missing_score_cols))
        if missing_pct_cols:
            details.append("percentiles=" + ",".join(missing_pct_cols))
        raise ValueError(
            "Validation requires full Big-5 schema (50 items + 5 score + 5 percentile columns). "
            + "; ".join(details)
        )

    target_cols = pct_cols + score_cols
    X = df[item_cols].astype(np.float64).copy()
    y = df[target_cols].astype(np.float64).copy()

    valid_mask = y[pct_cols].notna().all(axis=1)
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)
    if X.empty:
        raise ValueError("Validation data has zero valid rows after percentile filtering.")

    return X, y


# ============================================================================
# Sparsity for sparse evaluation
# ============================================================================

def _apply_adaptive_sparsity_balanced(
    X: pd.DataFrame,
    item_info: dict,
    min_items_per_domain: int = 4,
    min_total_items: int = 20,
    max_total_items: int = 20,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Apply domain-balanced sparsity (delegates to lib.sparsity canonical impl)."""
    return apply_sparse_balanced(
        X,
        item_info,
        min_items_per_domain=min_items_per_domain,
        min_total_items=min_total_items,
        max_total_items=max_total_items,
        rng=rng,
    )


# ============================================================================
# Prediction and metrics
# ============================================================================

def _pearsonr_strict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    label: str,
) -> tuple[float, float]:
    """Compute Pearson r/p and fail closed on non-finite/degenerate inputs."""
    true_arr = np.asarray(y_true, dtype=np.float64)
    pred_arr = np.asarray(y_pred, dtype=np.float64)
    if true_arr.shape != pred_arr.shape:
        raise ValueError(f"{label}: Pearson inputs have mismatched shapes {true_arr.shape} vs {pred_arr.shape}")

    finite_mask = np.isfinite(true_arr) & np.isfinite(pred_arr)
    if int(np.sum(finite_mask)) < 2:
        raise ValueError(f"{label}: Pearson requires at least 2 finite paired samples.")

    true_f = true_arr[finite_mask]
    pred_f = pred_arr[finite_mask]
    if np.ptp(true_f) <= 0 or np.ptp(pred_f) <= 0:
        raise ValueError(f"{label}: Pearson undefined for constant input arrays.")

    r, p = stats.pearsonr(true_f, pred_f)
    r_f = float(r)
    p_f = float(p)
    if not np.isfinite(r_f) or not np.isfinite(p_f):
        raise ValueError(f"{label}: Pearson produced non-finite output (r={r_f}, p={p_f}).")
    return r_f, p_f


def _predict_all_domains(
    domain_models: dict[str, dict[str, Any]],
    X: pd.DataFrame,
    y: pd.DataFrame,
    calibration_params: Optional[dict[str, dict[str, float]]] = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Run predictions for all domains and return per-domain arrays."""
    per_domain: dict[str, dict[str, np.ndarray]] = {}
    missing_pct_cols = [f"{d}_percentile" for d in DOMAINS if f"{d}_percentile" not in y.columns]
    missing_score_cols = [f"{d}_score" for d in DOMAINS if f"{d}_score" not in y.columns]
    if missing_pct_cols or missing_score_cols:
        details = []
        if missing_score_cols:
            details.append("scores=" + ",".join(missing_score_cols))
        if missing_pct_cols:
            details.append("percentiles=" + ",".join(missing_pct_cols))
        raise ValueError(
            "Validation prediction requires complete domain targets. " + "; ".join(details)
        )

    for domain, models in domain_models.items():
        pct_col = f"{domain}_percentile"
        score_col = f"{domain}_score"

        y_true = y[pct_col].values

        raw_q05 = models["q05"].predict(X)
        raw_q50 = models["q50"].predict(X)
        raw_q95 = models["q95"].predict(X)

        pct_q05 = raw_score_to_percentile(np.asarray(raw_q05, dtype=np.float64), domain)
        pct_q50 = raw_score_to_percentile(np.asarray(raw_q50, dtype=np.float64), domain)
        pct_q95 = raw_score_to_percentile(np.asarray(raw_q95, dtype=np.float64), domain)

        # Enforce monotonicity
        stacked = np.sort(np.stack([pct_q05, pct_q50, pct_q95], axis=0), axis=0)
        pct_q05, pct_q50, pct_q95 = stacked[0], stacked[1], stacked[2]

        # Apply calibration
        if calibration_params is not None and domain in calibration_params:
            scale = float(calibration_params[domain].get("scale_factor", 1.0))
            if scale != 1.0:
                half_width = 0.5 * (pct_q95 - pct_q05) * scale
                pct_q05 = np.clip(pct_q50 - half_width, 0.0, 100.0)
                pct_q95 = np.clip(pct_q50 + half_width, 0.0, 100.0)
                calibrated = np.sort(np.stack([pct_q05, pct_q50, pct_q95], axis=0), axis=0)
                pct_q05, pct_q50, pct_q95 = calibrated[0], calibrated[1], calibrated[2]

        raw_crossing = float(np.mean(
            (raw_q05 > raw_q95) | (raw_q50 < raw_q05) | (raw_q50 > raw_q95)
        ))

        raw_true = y[score_col].values if score_col in y.columns else np.full(len(y_true), np.nan)

        per_domain[domain] = {
            "true": y_true,
            "pred": pct_q50,
            "lower": pct_q05,
            "upper": pct_q95,
            "raw_true": raw_true,
            "raw_pred": raw_q50,
            "raw_crossing_rate": raw_crossing,
        }

    return per_domain


def _compute_domain_metrics(
    per_domain: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, float]]:
    """Compute per-domain and overall metrics from prediction arrays."""
    metrics: dict[str, dict[str, float]] = {}

    all_true_parts: list[np.ndarray] = []
    all_pred_parts: list[np.ndarray] = []
    all_lower_parts: list[np.ndarray] = []
    all_upper_parts: list[np.ndarray] = []

    for domain in DOMAINS:
        if domain not in per_domain:
            continue

        d = per_domain[domain]
        y_true = d["true"]
        y_pred = d["pred"]
        y_lower = d["lower"]
        y_upper = d["upper"]

        errors = y_true - y_pred
        abs_errors = np.abs(errors)
        r, p = _pearsonr_strict(y_true, y_pred, label=f"validation-domain:{domain}")

        in_interval = (y_true >= y_lower) & (y_true <= y_upper)
        coverage = float(np.mean(in_interval))
        interval_width = y_upper - y_lower
        mean_width = float(np.mean(interval_width))
        std_width = float(np.std(interval_width))

        # Central vs tail coverage
        central_mask = (y_true >= 20) & (y_true <= 80)
        tail_mask = ~central_mask
        central_coverage = float(np.mean(in_interval[central_mask])) if central_mask.sum() > 0 else float("nan")
        tail_coverage = float(np.mean(in_interval[tail_mask])) if tail_mask.sum() > 0 else float("nan")

        domain_metrics: dict[str, Any] = {
            "n": int(len(y_true)),
            "pearson_r": float(r),
            "p_value": float(p),
            "mae": float(np.mean(abs_errors)),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "mean_error": float(np.mean(errors)),
            "median_error": float(np.median(errors)),
            "within_target_pct": float(np.mean(abs_errors <= 5)),
            "within_3_pct": float(np.mean(abs_errors <= 3)),
            "within_5_pct": float(np.mean(abs_errors <= 5)),
            "within_10_pct": float(np.mean(abs_errors <= 10)),
            "coverage_90": coverage,
            "coverage_central": central_coverage,
            "coverage_tail": tail_coverage,
            "mean_interval_width": mean_width,
            "std_interval_width": std_width,
            "quantile_crossing_rate": 0.0,  # post-sort is always 0
            "raw_crossing_rate": d.get("raw_crossing_rate", 0.0),
        }

        # Raw-scale metrics
        raw_true = d.get("raw_true", None)
        raw_pred = d.get("raw_pred", None)
        if raw_true is not None and raw_pred is not None and not np.all(np.isnan(raw_true)):
            raw_r, raw_p = _pearsonr_strict(
                np.asarray(raw_true, dtype=np.float64),
                np.asarray(raw_pred, dtype=np.float64),
                label=f"validation-domain-raw:{domain}",
            )
            raw_errors = raw_true - raw_pred
            domain_metrics["raw_pearson_r"] = float(raw_r)
            domain_metrics["raw_mae"] = float(np.mean(np.abs(raw_errors)))
            domain_metrics["raw_rmse"] = float(np.sqrt(np.mean(raw_errors ** 2)))

        metrics[domain] = domain_metrics

        all_true_parts.append(y_true)
        all_pred_parts.append(y_pred)
        all_lower_parts.append(y_lower)
        all_upper_parts.append(y_upper)

    # Overall metrics
    if all_true_parts:
        all_true = np.concatenate(all_true_parts)
        all_pred = np.concatenate(all_pred_parts)
        all_lower = np.concatenate(all_lower_parts)
        all_upper = np.concatenate(all_upper_parts)

        errors = all_true - all_pred
        abs_errors = np.abs(errors)

        in_interval = (all_true >= all_lower) & (all_true <= all_upper)
        overall_coverage = float(np.mean(in_interval))
        interval_widths = all_upper - all_lower

        # Central (20-80) vs tail coverage
        central_mask = (all_true >= 20) & (all_true <= 80)
        tail_mask = ~central_mask
        central_coverage = float(np.mean(in_interval[central_mask])) if central_mask.sum() > 0 else float("nan")
        tail_coverage = float(np.mean(in_interval[tail_mask])) if tail_mask.sum() > 0 else float("nan")

        metrics["overall"] = {
            "n": int(len(all_true)),
            "pearson_r": _pearsonr_strict(
                all_true,
                all_pred,
                label="validation-overall",
            )[0],
            "mae": float(np.mean(abs_errors)),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "within_target_pct": float(np.mean(abs_errors <= 5)),
            "within_5_pct": float(np.mean(abs_errors <= 5)),
            "within_10_pct": float(np.mean(abs_errors <= 10)),
            "coverage_90": overall_coverage,
            "coverage_central": central_coverage,
            "coverage_tail": tail_coverage,
            "mean_interval_width": float(np.mean(interval_widths)),
            "std_interval_width": float(np.std(interval_widths)),
        }

    return metrics


# ============================================================================
# Bootstrap CIs
# ============================================================================

def _percentile_metric_fn(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> dict[str, float]:
    """Percentile-scale metrics for bootstrap resampling."""
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    try:
        r, _ = _pearsonr_strict(
            y_true,
            y_pred,
            label="validation-bootstrap",
        )
    except ValueError:
        # Keep bootstrap running; CI aggregation drops NaNs from invalid resamples.
        r = float("nan")
    return {
        "pearson_r": float(r),
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "within_5_pct": float(np.mean(abs_errors <= 5)),
        "within_10_pct": float(np.mean(abs_errors <= 10)),
        "coverage_90": float(np.mean((y_true >= y_lower) & (y_true <= y_upper))),
    }


def _raw_metric_fn(
    raw_true: np.ndarray,
    raw_pred: np.ndarray,
) -> dict[str, float]:
    """Raw-scale metrics for bootstrap resampling (placeholder, not used here)."""
    return {}


def _bootstrap_metrics(
    per_domain: dict[str, dict[str, np.ndarray]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, list[float]]]:
    """Compute bootstrap 95% CIs for per-domain and overall metrics.

    Delegates to lib.bootstrap.respondent_bootstrap_multi_domain.
    Returns dict mapping domain/overall -> metric_name -> [lower, upper].
    """
    valid_domains = [d for d in DOMAINS if d in per_domain]
    if not valid_domains:
        return {}

    result = respondent_bootstrap_multi_domain(
        per_domain_data=per_domain,
        domains=valid_domains,
        percentile_metric_fn=_percentile_metric_fn,
        raw_metric_fn=_raw_metric_fn,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    # Convert lib format to downstream format:
    # lib: {"overall_cis": {metric: {"lower", "upper"}}, "per_domain_cis": {domain: {metric: {"lower", "upper"}}}}
    # needed: {domain: {metric: [lower, upper]}, "overall": {metric: [lower, upper]}}
    cis: dict[str, dict[str, list[float]]] = {}

    for d in valid_domains:
        if d in result.get("per_domain_cis", {}):
            domain_cis = result["per_domain_cis"][d]
            cis[d] = {}
            for metric, bounds in domain_cis.items():
                cis[d][metric] = [bounds["lower"], bounds["upper"]]

    overall_cis: dict[str, list[float]] = {}
    for metric, bounds in result.get("overall_cis", {}).items():
        overall_cis[metric] = [bounds["lower"], bounds["upper"]]
    cis["overall"] = overall_cis

    return cis


# ============================================================================
# Quintile analysis
# ============================================================================

def _analyze_by_quintile(
    per_domain: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Analyze performance stratified by score quintile."""
    quintile_metrics: dict[str, dict[str, dict[str, float]]] = {}

    for domain in DOMAINS:
        if domain not in per_domain:
            continue

        d = per_domain[domain]
        y_true = d["true"]
        y_pred = d["pred"]
        y_lower = d["lower"]
        y_upper = d["upper"]

        try:
            quintiles = pd.qcut(y_true, q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        except ValueError:
            continue

        quintile_metrics[domain] = {}
        for q_label in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            mask = quintiles == q_label
            if mask.sum() == 0:
                continue

            true_q = y_true[mask]
            pred_q = y_pred[mask]
            lower_q = y_lower[mask]
            upper_q = y_upper[mask]

            errors = true_q - pred_q
            in_interval = (true_q >= lower_q) & (true_q <= upper_q)

            quintile_metrics[domain][q_label] = {
                "n": int(mask.sum()),
                "mean_true": float(np.mean(true_q)),
                "mean_pred": float(np.mean(pred_q)),
                "mae": float(np.mean(np.abs(errors))),
                "coverage_90": float(np.mean(in_interval)),
                "mean_interval_width": float(np.mean(upper_q - lower_q)),
            }

    return quintile_metrics


# ============================================================================
# Plotting
# ============================================================================

def _create_validation_plots(
    per_domain: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
    prefix: str = "",
) -> None:
    """Generate scatter, residual, and calibration plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping plots")
        return

    # Scatter plot: true vs predicted per domain
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()

    for i, domain in enumerate(DOMAINS):
        if domain not in per_domain:
            continue
        ax = axes_flat[i]
        d = per_domain[domain]

        # Subsample for plotting (max 5000 points)
        n = len(d["true"])
        if n > 5000:
            idx = np.random.default_rng(42).choice(n, 5000, replace=False)
        else:
            idx = np.arange(n)

        ax.scatter(d["true"][idx], d["pred"][idx], alpha=0.1, s=2, color="steelblue")
        ax.plot([0, 100], [0, 100], "r--", linewidth=1, alpha=0.7)
        try:
            r, _ = _pearsonr_strict(
                d["true"],
                d["pred"],
                label=f"validation-plot:{domain}",
            )
            title = f"{DOMAIN_LABELS[domain]} (r={r:.3f})"
        except ValueError:
            title = f"{DOMAIN_LABELS[domain]} (r=n/a)"
        ax.set_title(title)
        ax.set_xlabel("True Percentile")
        ax.set_ylabel("Predicted Percentile")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

    # Hide unused subplot
    if len(DOMAINS) < 6:
        axes_flat[-1].set_visible(False)

    plt.suptitle(f"{prefix}Predicted vs True Percentile", fontsize=14)
    plt.tight_layout()
    scatter_path = output_dir / f"{prefix}scatter_plot.png"
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved %s", scatter_path.name)

    # Calibration plot: coverage per domain
    fig, ax = plt.subplots(figsize=(8, 5))
    coverages = []
    labels = []
    for domain in DOMAINS:
        if domain not in per_domain:
            continue
        d = per_domain[domain]
        coverage = float(np.mean((d["true"] >= d["lower"]) & (d["true"] <= d["upper"])))
        coverages.append(coverage)
        labels.append(DOMAIN_LABELS[domain])

    x = np.arange(len(labels))
    ax.bar(x, coverages, color="steelblue", alpha=0.8)
    ax.axhline(y=0.90, color="red", linestyle="--", linewidth=1.5, label="Target (90%)")
    ax.fill_between([-0.5, len(labels) - 0.5], 0.88, 0.92,
                     alpha=0.15, color="red", label="+/-2% tolerance")
    ax.set_xlabel("Domain")
    ax.set_ylabel("90% CI Coverage")
    ax.set_title(f"{prefix}Calibration: Observed vs Target Coverage")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0.7, 1.0)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    cal_path = output_dir / f"{prefix}calibration_plot.png"
    plt.savefig(cal_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved %s", cal_path.name)

    # Residual plot: errors by true percentile
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()

    for i, domain in enumerate(DOMAINS):
        if domain not in per_domain:
            continue
        ax = axes_flat[i]
        d = per_domain[domain]
        errors = d["true"] - d["pred"]

        n = len(errors)
        if n > 5000:
            idx = np.random.default_rng(42).choice(n, 5000, replace=False)
        else:
            idx = np.arange(n)

        ax.scatter(d["true"][idx], errors[idx], alpha=0.1, s=2, color="steelblue")
        ax.axhline(y=0, color="red", linestyle="--", linewidth=1)
        ax.set_title(f"{DOMAIN_LABELS[domain]}")
        ax.set_xlabel("True Percentile")
        ax.set_ylabel("Error (True - Predicted)")
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.3)

    if len(DOMAINS) < 6:
        axes_flat[-1].set_visible(False)

    plt.suptitle(f"{prefix}Residual Plot", fontsize=14)
    plt.tight_layout()
    residual_path = output_dir / f"{prefix}residual_plot.png"
    plt.savefig(residual_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved %s", residual_path.name)


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate IPIP-BFFM XGBoost models on held-out test data"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to model directory (default: models/reference, relative to PACKAGE_ROOT)",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate validation plots",
    )
    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for 95%% CIs (default: 1000, 0 to disable)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Data directory containing test.parquet/item_info",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Artifacts output directory (default: artifacts)",
    )
    add_provenance_args(parser)
    args = parser.parse_args()

    # Resolve paths
    data_dir = args.data_dir if args.data_dir.is_absolute() else PACKAGE_ROOT / args.data_dir
    artifacts_dir = (
        args.artifacts_dir
        if args.artifacts_dir.is_absolute()
        else PACKAGE_ROOT / args.artifacts_dir
    )

    if args.model_dir:
        model_dir = Path(args.model_dir)
        if not model_dir.is_absolute():
            model_dir = PACKAGE_ROOT / model_dir
    else:
        model_dir = PACKAGE_ROOT / "models" / "reference"

    if not model_dir.exists():
        log.error("Model directory not found: %s", model_dir)
        return 1

    log.info("=" * 60)
    log.info("IPIP-BFFM Model Validation")
    log.info("=" * 60)
    log.info("Model dir: %s", model_dir)
    log.info("Data dir: %s", data_dir)
    log.info("Artifacts dir: %s", artifacts_dir)
    log.info("Bootstrap: %d resamples", args.bootstrap_n)
    log.info("Plots: %s", args.plots)

    # Load models
    log.info("Step 1: Loading models...")
    domain_models = _load_models(model_dir)
    missing = _check_models_complete(domain_models)
    if missing:
        log.error("Missing models: %s", ", ".join(missing))
        return 1

    n_models = sum(len(m) for m in domain_models.values())
    log.info("  Loaded %d models", n_models)

    # Load calibration (explicit full/sparse regimes with legacy fallback)
    try:
        full_calibration = _load_calibration_params(
            model_dir / "calibration_params_full_50.json"
        )
        sparse_calibration = _load_calibration_params(
            model_dir / "calibration_params_sparse_20_balanced.json"
        )
        legacy_calibration: dict[str, dict[str, float]] = {}
        if not full_calibration or not sparse_calibration:
            legacy_calibration = _load_calibration_params(
                model_dir / "calibration_params.json"
            )
        if not full_calibration and legacy_calibration:
            full_calibration = legacy_calibration
        if not sparse_calibration and legacy_calibration:
            sparse_calibration = legacy_calibration
    except (ValueError, json.JSONDecodeError) as e:
        log.error("Invalid calibration parameters: %s", e)
        return 1
    log.info("  Calibration full-50: %d domains", len(full_calibration))
    log.info("  Calibration sparse-20: %d domains", len(sparse_calibration))

    test_sha256: str | None = None
    split_signature: str | None = None
    try:
        test_sha256, split_signature = _verify_model_data_split_provenance(
            model_dir=model_dir,
            data_dir=data_dir,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        log.error("Model/data provenance check failed: %s", e)
        return 1
    if test_sha256 is not None:
        log.info("  Test split hash verified against training report")
    if split_signature is not None:
        log.info("  Full split signature verified against training report")

    # Load test data
    log.info("Step 2: Loading test data...")
    test_df = _load_test_data(data_dir)
    log.info("  Test: %d respondents", len(test_df))

    try:
        X_test, y_test = _prepare_features_targets(test_df)
    except ValueError as e:
        log.error("Invalid test data schema for validation: %s", e)
        return 1
    log.info("  Features: %d items, Samples: %d", X_test.shape[1], len(X_test))

    # Load item info for sparse evaluation
    try:
        item_info, item_info_path, item_info_sha256 = load_item_info_for_model(
            model_dir,
            data_dir,
            require_first_item=True,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        log.error("%s", e)
        return 1
    log.info(
        "  Item info loaded (%d items, sha256=%s)",
        len(item_info["item_pool"]),
        item_info_sha256[:12],
    )

    # ===== Evaluation 1: Full 50-item =====
    log.info("Step 3: Evaluating at full 50-item...")
    t0 = time.time()
    per_domain_full = _predict_all_domains(
        domain_models, X_test, y_test,
        calibration_params=full_calibration,
    )
    metrics_full = _compute_domain_metrics(per_domain_full)
    t_full = time.time() - t0

    log.info("  Full 50-item results (%.1fs):", t_full)
    if "overall" in metrics_full:
        log.info("    Overall: r=%.4f, MAE=%.2f, within_5=%.1f%%",
                 metrics_full["overall"]["pearson_r"],
                 metrics_full["overall"]["mae"],
                 metrics_full["overall"]["within_5_pct"] * 100)
    for domain in DOMAINS:
        if domain in metrics_full:
            dm = metrics_full[domain]
            log.info("    %s: r=%.4f, MAE=%.2f, coverage=%.1f%%, width=%.1f",
                     DOMAIN_LABELS[domain], dm["pearson_r"], dm["mae"],
                     dm["coverage_90"] * 100, dm["mean_interval_width"])

    # Quintile analysis
    quintile_full = _analyze_by_quintile(per_domain_full)

    # Bootstrap CIs for full 50-item
    bootstrap_cis_full: dict = {}
    if args.bootstrap_n > 0:
        log.info("  Computing bootstrap CIs (full 50-item, %d resamples)...", args.bootstrap_n)
        bootstrap_cis_full = _bootstrap_metrics(
            per_domain_full, n_bootstrap=args.bootstrap_n, seed=42,
        )

    # ===== Evaluation 2: Sparse 20-item =====
    metrics_sparse: dict = {}
    bootstrap_cis_sparse: dict = {}
    quintile_sparse: dict = {}

    log.info("Step 4: Evaluating at sparse 20-item...")
    t0 = time.time()

    X_sparse = _apply_adaptive_sparsity_balanced(
        X_test.copy(), item_info,
        min_items_per_domain=4,
        min_total_items=20,
        max_total_items=20,
        rng=np.random.default_rng(42),
    )

    n_nan = X_sparse.isna().sum().sum()
    total = X_sparse.shape[0] * X_sparse.shape[1]
    log.info("  Sparse data: %.1f%% missing", 100 * n_nan / total)

    per_domain_sparse = _predict_all_domains(
        domain_models, X_sparse, y_test,
        calibration_params=sparse_calibration,
    )
    metrics_sparse = _compute_domain_metrics(per_domain_sparse)
    t_sparse = time.time() - t0

    log.info("  Sparse 20-item results (%.1fs):", t_sparse)
    if "overall" in metrics_sparse:
        log.info("    Overall: r=%.4f, MAE=%.2f, within_5=%.1f%%",
                 metrics_sparse["overall"]["pearson_r"],
                 metrics_sparse["overall"]["mae"],
                 metrics_sparse["overall"]["within_5_pct"] * 100)
    for domain in DOMAINS:
        if domain in metrics_sparse:
            dm = metrics_sparse[domain]
            log.info("    %s: r=%.4f, MAE=%.2f, coverage=%.1f%%",
                     DOMAIN_LABELS[domain], dm["pearson_r"], dm["mae"],
                     dm["coverage_90"] * 100)

    quintile_sparse = _analyze_by_quintile(per_domain_sparse)

    if args.bootstrap_n > 0:
        log.info("  Computing bootstrap CIs (sparse 20-item, %d resamples)...", args.bootstrap_n)
        bootstrap_cis_sparse = _bootstrap_metrics(
            per_domain_sparse, n_bootstrap=args.bootstrap_n, seed=42,
        )

    # Plots
    if args.plots:
        log.info("Step 5: Generating plots...")
        plot_dir = model_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        _create_validation_plots(per_domain_full, plot_dir, prefix="full_50_")
        _create_validation_plots(per_domain_sparse, plot_dir, prefix="sparse_20_")

    # Save results
    log.info("Step 6: Saving results...")

    provenance = build_provenance(
        Path(__file__).name,
        args=args,
        rng_seed=42,
        extra={
            "model_dir": relative_to_root(model_dir),
            "bootstrap_n": args.bootstrap_n,
            "test_sha256": test_sha256,
            "split_signature": split_signature,
        },
    )

    results = {
        "provenance": provenance,
        "metrics": metrics_full,
        "quintile_metrics": quintile_full,
    }

    if bootstrap_cis_full:
        results["bootstrap_cis"] = bootstrap_cis_full

    if metrics_sparse:
        results["sparse_20"] = {
            "metrics": metrics_sparse,
            "quintile_metrics": quintile_sparse,
        }
        if bootstrap_cis_sparse:
            results["sparse_20"]["bootstrap_cis"] = bootstrap_cis_sparse

    results_path = artifacts_dir / "validation_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("  Saved %s", results_path)

    # Summary
    log.info("=" * 60)
    log.info("Validation complete!")
    log.info("=" * 60)

    if "overall" in metrics_full:
        log.info("Full 50-item: r=%.4f, MAE=%.2f, coverage=%.1f%%",
                 metrics_full["overall"]["pearson_r"],
                 metrics_full["overall"]["mae"],
                 metrics_full["overall"].get("coverage_90", float("nan")) * 100)

    if "overall" in metrics_sparse:
        log.info("Sparse 20-item: r=%.4f, MAE=%.2f",
                 metrics_sparse["overall"]["pearson_r"],
                 metrics_sparse["overall"]["mae"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
