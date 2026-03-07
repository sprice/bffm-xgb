#!/usr/bin/env python3
"""
Compare IPIP-BFFM adaptive XGBoost model against baseline item selection
strategies at various item budgets (K = 5, 10, 15, 20, 25, 30, 40, 50).

Strategies evaluated:
  1. domain_balanced  - K/5 items per domain, highest correlation items
  2. domain_constrained_adaptive - K/5 items per domain, highest cross-domain info score
  3. mini_ipip        - Standalone Mini-IPIP scoring baseline (only at K=20)
  4. adaptive_topk    - Greedy top-K by cross-domain info score
  5. greedy_balanced  - One-per-domain cold start, then greedy top-K
  6. random           - Random K items (averaged over multiple trials)
  7. first_n          - First K items in standard interleaved order
  8. worst_k          - K items with lowest own-domain correlations (negative control)

For each strategy x K:
  - Mask test data to keep only the K selected items
  - Run ML model predictions (XGBoost quantile), except Mini-IPIP standalone
  - Compute simple averaging predictions (mean of answered items per domain)
  - Compute Pearson r, MAE, RMSE, coverage vs full-information ground truth
  - Bootstrap CIs

Output files:
  - artifacts/baseline_comparison_results.json
  - artifacts/baseline_comparison_per_domain.csv
  - artifacts/baseline_comparison_per_domain.meta.json
  - artifacts/ml_vs_averaging_comparison.json
  - artifacts/adaptive_item_order_analysis.json

Usage:
    python pipeline/09_baselines.py --data-dir data/processed/ext_est
    python pipeline/09_baselines.py --data-dir data/processed/ext_est --model-dir models/reference --bootstrap-n 2000
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
from tqdm import tqdm

from lib.constants import (
    DOMAINS,
    DOMAIN_LABELS,
    ITEM_COLUMNS,
    ITEMS_PER_DOMAIN,
)
from lib.bootstrap import respondent_bootstrap_multi_domain, vectorized_pearsonr_bootstrap
from lib.mini_ipip import flatten_mini_ipip_items, load_mini_ipip_mapping
from lib.norms import load_mini_ipip_norms
from lib.scoring import raw_score_to_percentile
from lib.provenance import build_provenance, add_provenance_args, relative_to_root
from lib.item_info import (
    file_sha256,
    load_item_info_for_model,
)
from lib.provenance_checks import (
    verify_model_data_split_provenance as _verify_model_data_split_provenance,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# K values to evaluate
K_VALUES = [5, 10, 15, 20, 25, 30, 40, 50]
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


def _load_calibration_params(
    models_dir: Path,
    filename: str = "calibration_params.json",
) -> dict[str, dict[str, float]]:
    """Load calibration parameters.

    Missing files are treated as optional and return an empty mapping.
    Present files must be complete and well-formed for all five domains.
    """
    path = models_dir / filename
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


def _choose_calibration_for_budget(
    n_items: int,
    sparse_calibration: dict[str, dict[str, float]],
    full_calibration: dict[str, dict[str, float]],
) -> tuple[Optional[dict[str, dict[str, float]]], str]:
    """Select calibration regime for a given item budget."""
    if n_items >= 50 and full_calibration:
        return full_calibration, "full_50"
    if sparse_calibration:
        return sparse_calibration, "sparse_20_balanced"
    return None, "none"


def _load_test_data(data_dir: Path) -> pd.DataFrame:
    """Load held-out test parquet."""
    test_path = data_dir / "test.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")
    return pd.read_parquet(test_path)


def _validate_test_schema(df: pd.DataFrame) -> None:
    """Fail closed unless held-out data has the full Big-5 schema."""
    required_item_cols = list(ITEM_COLUMNS)
    required_pct_cols = [f"{d}_percentile" for d in DOMAINS]
    required_score_cols = [f"{d}_score" for d in DOMAINS]

    missing_item_cols = [c for c in required_item_cols if c not in df.columns]
    missing_pct_cols = [c for c in required_pct_cols if c not in df.columns]
    missing_score_cols = [c for c in required_score_cols if c not in df.columns]
    if missing_item_cols or missing_pct_cols or missing_score_cols:
        details = []
        if missing_item_cols:
            details.append("items=" + ",".join(missing_item_cols))
        if missing_score_cols:
            details.append("scores=" + ",".join(missing_score_cols))
        if missing_pct_cols:
            details.append("percentiles=" + ",".join(missing_pct_cols))
        raise ValueError(
            "Baselines require full Big-5 schema (50 items + 5 score + 5 percentile columns). "
            + "; ".join(details)
        )


def _load_item_info(data_dir: Path, model_dir: Path) -> tuple[dict, str]:
    """Load and provenance-verify stage-05 item_info for a specific model bundle."""
    item_info, _path, item_info_sha256 = load_item_info_for_model(
        model_dir,
        data_dir,
        require_first_item=True,
    )
    return item_info, item_info_sha256


def _load_mini_ipip_mapping(
    mapping_path: Path = PACKAGE_ROOT / "artifacts" / "mini_ipip_mapping.json",
) -> dict[str, list[str]]:
    """Load Mini-IPIP item IDs from mapping file (fail closed)."""
    return load_mini_ipip_mapping(mapping_path)


# ============================================================================
# Item selection strategies
# ============================================================================

def _select_domain_balanced(item_pool: list[dict], n_per_domain: int) -> list[str]:
    """Select N items per domain (highest own-domain correlation)."""
    selected: list[str] = []
    for domain in DOMAINS:
        domain_items = [item for item in item_pool if item["home_domain"] == domain]
        domain_items.sort(key=lambda x: abs(x.get("own_domain_r", 0)), reverse=True)
        selected.extend([item["id"] for item in domain_items[:n_per_domain]])
    return selected


def _select_domain_constrained_adaptive(item_pool: list[dict], n_per_domain: int) -> list[str]:
    """Select N items per domain, highest cross-domain info score each.

    This represents the best possible adaptive strategy with domain balance
    constraints (round-robin + info-ranked), contrasted with domain_balanced
    which uses own-domain correlation for ranking within each domain.
    """
    selected: list[str] = []
    for domain in DOMAINS:
        domain_items = [item for item in item_pool if item["home_domain"] == domain]
        domain_items.sort(key=lambda x: x.get("cross_domain_info", 0), reverse=True)
        selected.extend([item["id"] for item in domain_items[:n_per_domain]])
    return selected


def _select_adaptive_topk(item_pool: list[dict], n_items: int) -> list[str]:
    """Select top N items by adaptive ranking (composite score)."""
    return [item["id"] for item in item_pool[:n_items]]


def _select_greedy_balanced(item_pool: list[dict], n_items: int) -> list[str]:
    """Select items with domain coverage cold-start, then greedy fill.

    First pass: select the highest-ranked available item from each domain.
    Second pass: fill remaining slots from the global ranking.
    """
    if n_items <= 0:
        return []

    if n_items < len(DOMAINS):
        return _select_adaptive_topk(item_pool, n_items)

    selected: list[str] = []
    selected_set: set[str] = set()

    # Cold-start coverage: one best-ranked item per domain.
    for domain in DOMAINS:
        for item in item_pool:
            item_id = item.get("id")
            if item_id and item.get("home_domain") == domain and item_id not in selected_set:
                selected.append(item_id)
                selected_set.add(item_id)
                break

    # Greedy fill from global ranking.
    for item in item_pool:
        if len(selected) >= n_items:
            break
        item_id = item.get("id")
        if item_id and item_id not in selected_set:
            selected.append(item_id)
            selected_set.add(item_id)

    return selected[:n_items]


def _select_random(item_cols: list[str], n_items: int, seed: int = 42) -> list[str]:
    """Randomly select N items."""
    rng = np.random.default_rng(seed)
    return list(rng.choice(item_cols, n_items, replace=False))


def _select_first_n(n_items: int) -> list[str]:
    """Select first K items in interleaved order (ext1, agr1, csn1, est1, opn1, ext2, ...)."""
    interleaved: list[str] = []
    for i in range(1, ITEMS_PER_DOMAIN + 1):
        for domain in DOMAINS:
            interleaved.append(f"{domain}{i}")
    return interleaved[:n_items]


def _select_worst_k(item_pool: list[dict], n_items: int) -> list[str]:
    """Select K items with the lowest own-domain correlations."""
    sorted_pool = sorted(item_pool, key=lambda x: abs(x.get("own_domain_r", 0)))
    return [item["id"] for item in sorted_pool[:n_items]]


# ============================================================================
# Sparse data creation and prediction
# ============================================================================

def _create_sparse_numpy(
    X_values: np.ndarray,
    all_columns: list[str],
    selected_items: list[str],
) -> np.ndarray:
    """Create sparse version of data (NaN for unselected items)."""
    selected_set = set(selected_items)
    mask = np.array([col not in selected_set for col in all_columns])
    X_sparse = X_values.copy()
    X_sparse[:, mask] = np.nan
    return X_sparse


def _predict_all_domains(
    domain_models: dict[str, dict[str, Any]],
    X_sparse: np.ndarray,
    all_columns: list[str],
    y_test: pd.DataFrame,
    calibration_params: Optional[dict[str, dict[str, float]]] = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Run predictions for all domains and return per-domain arrays.

    Returns {domain: {"true", "pred", "lower", "upper", "raw_true", "raw_pred"}}.
    """
    X_df = pd.DataFrame(X_sparse, columns=all_columns)

    per_domain: dict[str, dict[str, np.ndarray]] = {}
    missing_pct_cols = [f"{d}_percentile" for d in DOMAINS if f"{d}_percentile" not in y_test.columns]
    missing_score_cols = [f"{d}_score" for d in DOMAINS if f"{d}_score" not in y_test.columns]
    if missing_pct_cols or missing_score_cols:
        details = []
        if missing_score_cols:
            details.append("scores=" + ",".join(missing_score_cols))
        if missing_pct_cols:
            details.append("percentiles=" + ",".join(missing_pct_cols))
        raise ValueError(
            "Baseline prediction requires complete domain targets. " + "; ".join(details)
        )

    for domain, models in domain_models.items():
        pct_col = f"{domain}_percentile"
        score_col = f"{domain}_score"

        y_true = y_test[pct_col].values

        raw_q05 = models["q05"].predict(X_df)
        raw_q50 = models["q50"].predict(X_df)
        raw_q95 = models["q95"].predict(X_df)

        pct_q05 = raw_score_to_percentile(np.asarray(raw_q05, dtype=np.float64), domain)
        pct_q50 = raw_score_to_percentile(np.asarray(raw_q50, dtype=np.float64), domain)
        pct_q95 = raw_score_to_percentile(np.asarray(raw_q95, dtype=np.float64), domain)

        stacked = np.sort(np.stack([pct_q05, pct_q50, pct_q95], axis=0), axis=0)
        pct_q05, pct_q50, pct_q95 = stacked[0], stacked[1], stacked[2]

        if calibration_params is not None and domain in calibration_params:
            scale = float(calibration_params[domain].get("scale_factor", 1.0))
            if scale != 1.0:
                half_width = 0.5 * (pct_q95 - pct_q05) * scale
                pct_q05 = np.clip(pct_q50 - half_width, 0.0, 100.0)
                pct_q95 = np.clip(pct_q50 + half_width, 0.0, 100.0)
                calibrated = np.sort(np.stack([pct_q05, pct_q50, pct_q95], axis=0), axis=0)
                pct_q05, pct_q50, pct_q95 = calibrated[0], calibrated[1], calibrated[2]

        raw_true = y_test[score_col].values if score_col in y_test.columns else np.full(len(y_true), np.nan)

        per_domain[domain] = {
            "true": y_true,
            "pred": pct_q50,
            "lower": pct_q05,
            "upper": pct_q95,
            "raw_true": raw_true,
            "raw_pred": raw_q50,
        }

    return per_domain


# ============================================================================
# Metrics computation
# ============================================================================

def _pearsonr_strict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    label: str,
) -> float:
    """Compute Pearson r and fail closed on non-finite/degenerate inputs."""
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

    r, _ = stats.pearsonr(true_f, pred_f)
    r_f = float(r)
    if not np.isfinite(r_f):
        raise ValueError(f"{label}: Pearson produced non-finite output.")
    return r_f


def _compute_metrics(
    all_true: np.ndarray,
    all_pred: np.ndarray,
    all_lower: np.ndarray,
    all_upper: np.ndarray,
) -> dict[str, float]:
    """Compute evaluation metrics from prediction arrays."""
    errors = all_true - all_pred
    abs_errors = np.abs(errors)
    r = _pearsonr_strict(all_true, all_pred, label="baselines-overall")
    return {
        "pearson_r": float(r),
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "within_5_pct": float(np.mean(abs_errors <= 5)),
        "within_10_pct": float(np.mean(abs_errors <= 10)),
        "coverage_90": float(np.mean((all_true >= all_lower) & (all_true <= all_upper))),
    }


def _flatten_domain_arrays(
    per_domain: dict[str, dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten per-domain arrays into single arrays."""
    trues, preds, lowers, uppers = [], [], [], []
    for domain in DOMAINS:
        if domain not in per_domain:
            continue
        d = per_domain[domain]
        trues.append(d["true"])
        preds.append(d["pred"])
        lowers.append(d["lower"])
        uppers.append(d["upper"])
    return (np.concatenate(trues), np.concatenate(preds),
            np.concatenate(lowers), np.concatenate(uppers))


def _count_items_per_domain(selected_items: list[str]) -> dict[str, int]:
    """Count how many selected items belong to each domain."""
    counts = {d: 0 for d in DOMAINS}
    for item_id in selected_items:
        domain = item_id.rstrip("0123456789")
        if domain in counts:
            counts[domain] += 1
    return counts


# ============================================================================
# Bootstrap
# ============================================================================

def _noop_raw_metric_fn(raw_true: np.ndarray, raw_pred: np.ndarray) -> dict[str, float]:
    """No-op raw metric function (baselines only need percentile-scale CIs)."""
    return {}


def _bootstrap_cis(
    per_domain: dict[str, dict[str, np.ndarray]],
    n_bootstrap: int,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Compute bootstrap 95% CIs for overall metrics.

    Delegates to lib.bootstrap.respondent_bootstrap_multi_domain with
    respondent-level resampling preserving cross-domain pairing.
    """
    valid_domains = [d for d in DOMAINS if d in per_domain]
    if not valid_domains:
        return {}

    result = respondent_bootstrap_multi_domain(
        per_domain_data=per_domain,
        domains=valid_domains,
        percentile_metric_fn=_compute_metrics,
        raw_metric_fn=_noop_raw_metric_fn,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    # Convert {"metric": {"lower": x, "upper": y}} -> {"metric_ci": [x, y]}
    cis: dict[str, list[float]] = {}
    for k, bounds in result.get("overall_cis", {}).items():
        cis[f"{k}_ci"] = [bounds["lower"], bounds["upper"]]
    return cis


def _bootstrap_per_domain_r(
    per_domain: dict[str, dict[str, np.ndarray]],
    n_bootstrap: int,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Compute bootstrap 95% CIs for per-domain Pearson r.

    Delegates to lib.bootstrap.vectorized_pearsonr_bootstrap for each domain.
    """
    rng = np.random.default_rng(seed)
    valid_domains = [d for d in DOMAINS if d in per_domain]
    if not valid_domains:
        return {}

    cis: dict[str, list[float]] = {}
    for d in valid_domains:
        boot_rs = vectorized_pearsonr_bootstrap(
            per_domain[d]["true"],
            per_domain[d]["pred"],
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        valid = boot_rs[~np.isnan(boot_rs)]
        if len(valid) > 0:
            cis[d] = [
                float(np.percentile(valid, 2.5)),
                float(np.percentile(valid, 97.5)),
            ]
    return cis


# ============================================================================
# Single method evaluation
# ============================================================================

def _evaluate_method(
    domain_models: dict[str, dict[str, Any]],
    X_values: np.ndarray,
    all_columns: list[str],
    y_test: pd.DataFrame,
    selected_items: list[str],
    calibration_params: Optional[dict[str, dict[str, float]]],
    n_bootstrap: int = 0,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Evaluate a single item selection, returning overall + per-domain metrics.

    Returns (overall_metrics, per_domain_metrics).
    """
    X_sparse = _create_sparse_numpy(X_values, all_columns, selected_items)
    per_domain = _predict_all_domains(
        domain_models, X_sparse, all_columns, y_test,
        calibration_params=calibration_params,
    )

    all_true, all_pred, all_lower, all_upper = _flatten_domain_arrays(per_domain)
    overall = _compute_metrics(all_true, all_pred, all_lower, all_upper)
    overall["n_items"] = len(selected_items)

    # Per-domain metrics
    item_counts = _count_items_per_domain(selected_items)
    per_domain_results: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        if domain not in per_domain:
            continue
        d = per_domain[domain]
        r_val = _pearsonr_strict(
            d["true"],
            d["pred"],
            label=f"baselines-domain:{domain}",
        )
        domain_result: dict[str, Any] = {
            "pearson_r": float(r_val),
            "n_domain_items": item_counts.get(domain, 0),
        }
        if "raw_true" in d and "raw_pred" in d and not np.all(np.isnan(d["raw_true"])):
            raw_r = _pearsonr_strict(
                d["raw_true"],
                d["raw_pred"],
                label=f"baselines-domain-raw:{domain}",
            )
            raw_errors = d["raw_true"] - d["raw_pred"]
            domain_result["raw_pearson_r"] = float(raw_r)
            domain_result["raw_mae"] = float(np.mean(np.abs(raw_errors)))
            domain_result["raw_rmse"] = float(np.sqrt(np.mean(raw_errors ** 2)))
        per_domain_results[domain] = domain_result

    # Bootstrap CIs
    if n_bootstrap > 0:
        overall_cis = _bootstrap_cis(per_domain, n_bootstrap, seed=seed)
        overall.update(overall_cis)

        domain_r_cis = _bootstrap_per_domain_r(per_domain, n_bootstrap, seed=seed)
        for domain, ci in domain_r_cis.items():
            if domain in per_domain_results:
                per_domain_results[domain]["pearson_r_ci"] = ci

    return overall, per_domain_results


# ============================================================================
# Random method aggregation
# ============================================================================

def _evaluate_random_aggregated(
    domain_models: dict[str, dict[str, Any]],
    X_values: np.ndarray,
    all_columns: list[str],
    y_test: pd.DataFrame,
    available_items: list[str],
    n_items: int,
    n_random_trials: int,
    calibration_params: Optional[dict[str, dict[str, float]]],
    n_bootstrap: int = 0,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Evaluate random selection averaged over multiple trials."""
    metric_keys = ["pearson_r", "mae", "rmse", "within_5_pct", "within_10_pct", "coverage_90"]

    all_seed_predictions: list[dict[str, dict[str, np.ndarray]]] = []
    all_seed_per_domain: list[dict[str, dict[str, Any]]] = []

    for seed in range(n_random_trials):
        items = _select_random(available_items, n_items, seed=seed)
        X_sparse = _create_sparse_numpy(X_values, all_columns, items)
        per_domain = _predict_all_domains(
            domain_models, X_sparse, all_columns, y_test,
            calibration_params=calibration_params,
        )
        all_seed_predictions.append(per_domain)

        item_counts = _count_items_per_domain(items)
        seed_per_domain: dict[str, dict[str, Any]] = {}
        for domain in DOMAINS:
            if domain not in per_domain:
                continue
            d = per_domain[domain]
            r_val = _pearsonr_strict(
                d["true"],
                d["pred"],
                label=f"baselines-random-domain:{domain}",
            )
            domain_result: dict[str, Any] = {
                "pearson_r": float(r_val),
                "n_domain_items": item_counts.get(domain, 0),
            }
            if "raw_true" in d and "raw_pred" in d and not np.all(np.isnan(d["raw_true"])):
                raw_r = _pearsonr_strict(
                    d["raw_true"],
                    d["raw_pred"],
                    label=f"baselines-random-domain-raw:{domain}",
                )
                domain_result["raw_pearson_r"] = float(raw_r)
            seed_per_domain[domain] = domain_result
        all_seed_per_domain.append(seed_per_domain)

    # Overall: mean across seeds
    seed_metrics = []
    for per_seed in all_seed_predictions:
        at, ap, al, au = _flatten_domain_arrays(per_seed)
        seed_metrics.append(_compute_metrics(at, ap, al, au))

    overall: dict[str, Any] = {"n_items": n_items}
    for k in metric_keys:
        overall[k] = float(np.mean([m[k] for m in seed_metrics]))

    # Bootstrap CIs for the averaged estimator
    if n_bootstrap > 0 and all_seed_predictions:
        rng = np.random.default_rng(42)
        first_domain = next((d for d in DOMAINS if d in all_seed_predictions[0]), None)
        if first_domain:
            n_respondents = len(all_seed_predictions[0][first_domain]["true"])
            all_idx = rng.integers(0, n_respondents, size=(n_bootstrap, n_respondents))

            boot_metrics: dict[str, list[float]] = {k: [] for k in metric_keys}
            for i in range(n_bootstrap):
                idx = all_idx[i]
                boot_seed_points = []
                for per_seed in all_seed_predictions:
                    valid_domains = [d for d in DOMAINS if d in per_seed]
                    at = np.concatenate([per_seed[d]["true"][idx] for d in valid_domains])
                    ap = np.concatenate([per_seed[d]["pred"][idx] for d in valid_domains])
                    al = np.concatenate([per_seed[d]["lower"][idx] for d in valid_domains])
                    au = np.concatenate([per_seed[d]["upper"][idx] for d in valid_domains])
                    boot_seed_points.append(_compute_metrics(at, ap, al, au))
                for k in metric_keys:
                    boot_metrics[k].append(float(np.mean([m[k] for m in boot_seed_points])))

            for k in metric_keys:
                arr = np.array(boot_metrics[k])
                valid = arr[~np.isnan(arr)]
                if len(valid) > 0:
                    overall[f"{k}_ci"] = [
                        float(np.percentile(valid, 2.5)),
                        float(np.percentile(valid, 97.5)),
                    ]

    # Per-domain: mean across seeds
    per_domain_agg: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        rs = [s[domain]["pearson_r"] for s in all_seed_per_domain if domain in s]
        items = [s[domain].get("n_domain_items", 0) for s in all_seed_per_domain if domain in s]
        if rs:
            per_domain_agg[domain] = {
                "pearson_r": float(np.mean(rs)),
                "n_domain_items": float(np.mean(items)) if items else 0.0,
            }

    return overall, per_domain_agg


# ============================================================================
# Simple averaging scoring
# ============================================================================

def _compute_simple_averaging_scores(
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    selected_items: list[str],
    norms: Optional[dict[str, dict[str, float]]] = None,
) -> dict[str, Any]:
    """Score using simple domain averaging (traditional psychometrics)."""
    missing_pct_cols = [f"{d}_percentile" for d in DOMAINS if f"{d}_percentile" not in y_test.columns]
    if missing_pct_cols:
        raise ValueError(
            "Simple averaging requires complete domain percentile targets: "
            + ", ".join(missing_pct_cols)
        )

    item_cols_by_domain: dict[str, list[str]] = {d: [] for d in DOMAINS}
    for item_id in selected_items:
        for domain in DOMAINS:
            if item_id.startswith(domain):
                item_cols_by_domain[domain].append(item_id)
                break

    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    per_domain_results: dict[str, dict[str, Any]] = {}

    for domain in DOMAINS:
        domain_items = item_cols_by_domain[domain]
        pct_col = f"{domain}_percentile"

        y_true = y_test[pct_col].values

        if len(domain_items) == 0:
            y_pred = np.full(len(y_true), 50.0)
        else:
            raw_avg = X_test[domain_items].mean(axis=1).values
            y_pred = np.asarray(raw_score_to_percentile(raw_avg, domain, norms=norms))

        all_true.append(y_true)
        all_pred.append(y_pred)

        r_val = _pearsonr_strict(
            y_true,
            y_pred,
            label=f"baselines-averaging-domain:{domain}",
        )
        per_domain_results[domain] = {
            "pearson_r": float(r_val),
            "n_domain_items": len(domain_items),
        }

    all_true_arr = np.concatenate(all_true)
    all_pred_arr = np.concatenate(all_pred)
    errors = all_true_arr - all_pred_arr
    abs_errors = np.abs(errors)
    r = _pearsonr_strict(all_true_arr, all_pred_arr, label="baselines-averaging-overall")

    overall: dict[str, Any] = {
        "n_items": len(selected_items),
        "pearson_r": float(r),
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "within_5_pct": float(np.mean(abs_errors <= 5)),
        "within_10_pct": float(np.mean(abs_errors <= 10)),
        "coverage_90": None,
    }

    return {"overall": overall, "per_domain": per_domain_results}


def _evaluate_mini_ipip_standalone(
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    mini_ipip_mapping: dict[str, list[str]],
    mini_ipip_norms: dict[str, dict[str, float]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Evaluate standalone Mini-IPIP scoring with Mini-IPIP-specific norms."""
    selected_items = flatten_mini_ipip_items(mini_ipip_mapping)
    missing_items = [item_id for item_id in selected_items if item_id not in X_test.columns]
    if missing_items:
        raise ValueError(
            "Mini-IPIP standalone scoring is missing required test columns: "
            + ", ".join(missing_items)
        )

    result = _compute_simple_averaging_scores(
        X_test,
        y_test,
        selected_items,
        norms=mini_ipip_norms,
    )

    for domain in DOMAINS:
        domain_result = result["per_domain"].get(domain)
        expected_n = len(mini_ipip_mapping[domain])
        if not isinstance(domain_result, dict) or domain_result.get("n_domain_items") != expected_n:
            raise ValueError(
                f"Mini-IPIP standalone scoring produced invalid domain coverage for {domain}: "
                f"expected {expected_n} items."
            )

    return result["overall"], result["per_domain"]


# ============================================================================
# Run comparisons at a given K
# ============================================================================

def _run_comparisons_at_k(
    domain_models: dict[str, dict[str, Any]],
    X_values: np.ndarray,
    all_columns: list[str],
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    item_pool: list[dict],
    available_items: list[str],
    n_items: int,
    mini_ipip_mapping: dict[str, list[str]],
    mini_ipip_norms: dict[str, dict[str, float]],
    calibration_params: Optional[dict[str, dict[str, float]]],
    calibration_regime: str,
    n_bootstrap: int,
    n_random_trials: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    """Run all strategies at a given K value."""
    overall_results: dict[str, dict[str, Any]] = {}
    per_domain_results: dict[str, dict[str, dict[str, Any]]] = {}

    def _eval(items: list[str], seed: int = 42) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        return _evaluate_method(
            domain_models, X_values, all_columns, y_test,
            items, calibration_params, n_bootstrap=n_bootstrap, seed=seed,
        )

    # --- Random (averaged over trials) ---
    ov, pd_res = _evaluate_random_aggregated(
        domain_models, X_values, all_columns, y_test,
        available_items, n_items, n_random_trials,
        calibration_params, n_bootstrap=n_bootstrap,
    )
    overall_results["random"] = ov
    per_domain_results["random"] = pd_res

    # --- Adaptive top-k ---
    items_adaptive = _select_adaptive_topk(item_pool, n_items)
    items_adaptive = [i for i in items_adaptive if i in available_items][:n_items]
    if len(items_adaptive) == n_items:
        ov, pd_res = _eval(items_adaptive)
        overall_results["adaptive_topk"] = ov
        per_domain_results["adaptive_topk"] = pd_res

    # --- Greedy balanced ---
    items_greedy_balanced = _select_greedy_balanced(item_pool, n_items)
    items_greedy_balanced = [
        i for i in items_greedy_balanced if i in available_items
    ][:n_items]
    if len(items_greedy_balanced) == n_items:
        ov, pd_res = _eval(items_greedy_balanced)
        overall_results["greedy_balanced"] = ov
        per_domain_results["greedy_balanced"] = pd_res

    # --- Domain balanced (only if divisible by 5) ---
    if n_items % 5 == 0:
        n_per_domain = n_items // 5
        items_balanced = _select_domain_balanced(item_pool, n_per_domain)
        items_balanced = [i for i in items_balanced if i in available_items]
        if len(items_balanced) >= n_items:
            ov, pd_res = _eval(items_balanced[:n_items])
            overall_results["domain_balanced"] = ov
            per_domain_results["domain_balanced"] = pd_res

        # --- Domain constrained adaptive (only if divisible by 5) ---
        items_ca = _select_domain_constrained_adaptive(item_pool, n_per_domain)
        items_ca = [i for i in items_ca if i in available_items]
        if len(items_ca) >= n_items:
            ov, pd_res = _eval(items_ca[:n_items])
            overall_results["domain_constrained_adaptive"] = ov
            per_domain_results["domain_constrained_adaptive"] = pd_res

    # --- First K ---
    items_first = _select_first_n(n_items)
    items_first = [i for i in items_first if i in available_items][:n_items]
    if len(items_first) == n_items:
        ov, pd_res = _eval(items_first)
        overall_results["first_n"] = ov
        per_domain_results["first_n"] = pd_res

    # --- Worst K ---
    items_worst = _select_worst_k(item_pool, n_items)
    items_worst = [i for i in items_worst if i in available_items][:n_items]
    if len(items_worst) == n_items:
        ov, pd_res = _eval(items_worst)
        overall_results["worst_k"] = ov
        per_domain_results["worst_k"] = pd_res

    # --- Mini-IPIP (only at K=20) ---
    if n_items == 20:
        mini_items = flatten_mini_ipip_items(mini_ipip_mapping)
        missing_items = sorted(set(mini_items) - set(available_items))
        if missing_items:
            raise ValueError(
                "Mini-IPIP mapping contains items absent from test data: "
                + ", ".join(missing_items)
            )
        ov, pd_res = _evaluate_mini_ipip_standalone(
            X_test,
            y_test,
            mini_ipip_mapping=mini_ipip_mapping,
            mini_ipip_norms=mini_ipip_norms,
        )
        overall_results["mini_ipip"] = ov
        per_domain_results["mini_ipip"] = pd_res

    # --- Full assessment (K=50) ---
    if n_items >= 50:
        ov, pd_res = _eval(available_items)
        overall_results["full_50"] = ov
        per_domain_results["full_50"] = pd_res

    # Add calibration regime to all results
    for method_metrics in overall_results.values():
        method_metrics["calibration_regime"] = calibration_regime

    return overall_results, per_domain_results


# ============================================================================
# ML vs averaging comparison
# ============================================================================

def _run_ml_vs_averaging_comparison(
    domain_models: dict[str, dict[str, Any]],
    X_values: np.ndarray,
    all_columns: list[str],
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    item_pool: list[dict],
    available_items: list[str],
    mini_ipip_mapping: dict[str, list[str]],
    mini_ipip_norms: dict[str, dict[str, float]],
    sparse_calibration: dict[str, dict[str, float]],
    full_calibration: dict[str, dict[str, float]],
    n_bootstrap: int = 0,
) -> dict:
    """Compare ML scoring vs simple averaging for domain_balanced and mini_ipip."""
    results: list[dict[str, Any]] = []

    for n_items in [10, 15, 20, 25]:
        strategies: dict[str, list[str]] = {}

        # Domain balanced
        if n_items % 5 == 0:
            n_per_domain = n_items // 5
            items = _select_domain_balanced(item_pool, n_per_domain)
            items = [i for i in items if i in available_items][:n_items]
            if len(items) == n_items:
                strategies["domain_balanced"] = items

        # Mini-IPIP (only at 20)
        if n_items == 20:
            mini_items = flatten_mini_ipip_items(mini_ipip_mapping)
            valid_mini = [i for i in mini_items if i in available_items]
            if len(valid_mini) != len(mini_items):
                missing = sorted(set(mini_items) - set(valid_mini))
                raise ValueError(
                    "Mini-IPIP mapping contains items absent from test data: "
                    + ", ".join(missing)
                )
            strategies["mini_ipip"] = valid_mini

        selected_calibration, calibration_regime = _choose_calibration_for_budget(
            n_items, sparse_calibration, full_calibration,
        )

        for method, selected_items in strategies.items():
            # ML scoring
            X_sparse = _create_sparse_numpy(X_values, all_columns, selected_items)
            per_domain_preds = _predict_all_domains(
                domain_models, X_sparse, all_columns, y_test,
                calibration_params=selected_calibration,
            )
            all_true, all_pred, all_lower, all_upper = _flatten_domain_arrays(per_domain_preds)
            ml_metrics = _compute_metrics(all_true, all_pred, all_lower, all_upper)

            ml_per_domain: dict[str, float] = {}
            for domain in DOMAINS:
                if domain in per_domain_preds:
                    d = per_domain_preds[domain]
                    r_val = _pearsonr_strict(
                        d["true"],
                        d["pred"],
                        label=f"baselines-ml-vs-avg-domain:{domain}",
                    )
                    ml_per_domain[domain] = float(r_val)

            # Simple averaging
            avg_norms = mini_ipip_norms if method == "mini_ipip" else None
            avg_result = _compute_simple_averaging_scores(
                X_test,
                y_test,
                selected_items,
                norms=avg_norms,
            )

            avg_per_domain: dict[str, float] = {}
            for domain in DOMAINS:
                if domain in avg_result["per_domain"]:
                    avg_per_domain[domain] = avg_result["per_domain"][domain]["pearson_r"]

            row_result: dict[str, Any] = {
                "method": method,
                "n_items": n_items,
                "calibration_regime": calibration_regime,
                "ml_r": ml_metrics["pearson_r"],
                "avg_r": avg_result["overall"]["pearson_r"],
                "delta_r": ml_metrics["pearson_r"] - avg_result["overall"]["pearson_r"],
                "ml_mae": ml_metrics["mae"],
                "avg_mae": avg_result["overall"]["mae"],
                "delta_mae": ml_metrics["mae"] - avg_result["overall"]["mae"],
                "ml_within5": ml_metrics["within_5_pct"],
                "avg_within5": avg_result["overall"]["within_5_pct"],
                "ml_per_domain": ml_per_domain,
                "avg_per_domain": avg_per_domain,
            }

            # Bootstrap CIs for delta_r
            if n_bootstrap > 0:
                rng = np.random.default_rng(42)
                active_domains = [
                    d for d in DOMAINS
                    if d in per_domain_preds and f"{d}_percentile" in y_test.columns
                ]
                if not active_domains:
                    raise ValueError(
                        "ML vs averaging bootstrap requires at least one active domain."
                    )

                n_resp = len(per_domain_preds[active_domains[0]]["true"])
                if n_resp <= 0:
                    raise ValueError(
                        "ML vs averaging bootstrap requires non-empty respondent arrays."
                    )
                for domain in active_domains[1:]:
                    if len(per_domain_preds[domain]["true"]) != n_resp:
                        raise ValueError(
                            "ML vs averaging bootstrap requires equal respondent counts "
                            "across active domains."
                        )
                all_boot_idx = rng.integers(0, n_resp, size=(n_bootstrap, n_resp))

                # Rebuild averaging predictions per domain for bootstrap
                avg_stacked_true: dict[str, np.ndarray] = {}
                avg_stacked_pred: dict[str, np.ndarray] = {}
                for domain in active_domains:
                    pct_col = f"{domain}_percentile"
                    avg_stacked_true[domain] = y_test[pct_col].values
                    d_items = [it for it in selected_items if it.startswith(domain)]
                    if len(d_items) == 0:
                        avg_stacked_pred[domain] = np.full(n_resp, 50.0)
                    else:
                        raw_avg = X_test[d_items].mean(axis=1).values
                        avg_stacked_pred[domain] = np.asarray(
                            raw_score_to_percentile(
                                raw_avg,
                                domain,
                                norms=avg_norms,
                            )
                        )

                boot_delta_r: list[float] = []
                boot_delta_mae: list[float] = []

                for b_idx in range(n_bootstrap):
                    idx = all_boot_idx[b_idx]
                    # ML
                    ml_true_b = np.concatenate(
                        [per_domain_preds[d]["true"][idx] for d in active_domains]
                    )
                    ml_pred_b = np.concatenate(
                        [per_domain_preds[d]["pred"][idx] for d in active_domains]
                    )
                    ml_r_b = _pearsonr_strict(
                        ml_true_b,
                        ml_pred_b,
                        label="baselines-ml-vs-avg-bootstrap-ml",
                    )
                    ml_mae_b = float(np.mean(np.abs(ml_true_b - ml_pred_b)))

                    # Averaging
                    avg_true_b = np.concatenate(
                        [avg_stacked_true[d][idx] for d in active_domains]
                    )
                    avg_pred_b = np.concatenate(
                        [avg_stacked_pred[d][idx] for d in active_domains]
                    )
                    avg_r_b = _pearsonr_strict(
                        avg_true_b,
                        avg_pred_b,
                        label="baselines-ml-vs-avg-bootstrap-avg",
                    )
                    avg_mae_b = float(np.mean(np.abs(avg_true_b - avg_pred_b)))

                    boot_delta_r.append(float(ml_r_b - avg_r_b))
                    boot_delta_mae.append(float(ml_mae_b - avg_mae_b))

                arr_r = np.array(boot_delta_r)
                arr_mae = np.array(boot_delta_mae)
                row_result["delta_r_ci"] = [
                    float(np.percentile(arr_r, 2.5)),
                    float(np.percentile(arr_r, 97.5)),
                ]
                row_result["delta_mae_ci"] = [
                    float(np.percentile(arr_mae, 2.5)),
                    float(np.percentile(arr_mae, 97.5)),
                ]

            results.append(row_result)

    return {"comparisons": results}


# ============================================================================
# Output formatting
# ============================================================================

def _create_per_domain_csv(
    all_per_domain: dict[int, dict[str, dict[str, dict[str, Any]]]],
) -> pd.DataFrame:
    """Create flat DataFrame with per-domain metrics for CSV export."""
    rows: list[dict[str, Any]] = []

    for n_items in sorted(all_per_domain.keys()):
        methods = all_per_domain[n_items]
        for method, per_dom in methods.items():
            row: dict[str, Any] = {"n_items": n_items, "method": method}

            for domain in DOMAINS:
                label = DOMAIN_LABELS[domain]
                if domain in per_dom:
                    dm = per_dom[domain]
                    row[f"r_{label}"] = dm.get("pearson_r", None)
                    row[f"items_{label}"] = dm.get("n_domain_items", 0)
                    if "pearson_r_ci" in dm:
                        row[f"r_{label}_ci_lower"] = dm["pearson_r_ci"][0]
                        row[f"r_{label}_ci_upper"] = dm["pearson_r_ci"][1]
                else:
                    row[f"r_{label}"] = None
                    row[f"items_{label}"] = 0

            rows.append(row)

    return pd.DataFrame(rows)


def _write_per_domain_csv_with_metadata(
    df: pd.DataFrame,
    csv_path: Path,
    metadata_path: Path,
    provenance: dict[str, Any],
) -> None:
    """Write per-domain CSV plus fail-closed provenance metadata sidecar."""
    df.to_csv(csv_path, index=False)
    metadata = {
        "provenance": provenance,
        "artifact": {
            "file": csv_path.name,
            "sha256": file_sha256(csv_path),
            "n_rows": int(len(df)),
            "n_columns": int(len(df.columns)),
        },
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def _print_summary(
    all_results: dict[int, dict[str, dict[str, Any]]],
) -> None:
    """Print comparison summary to log."""
    methods_order = [
        "random",
        "first_n",
        "domain_balanced",
        "domain_constrained_adaptive",
        "adaptive_topk",
        "greedy_balanced",
        "worst_k",
        "mini_ipip",
        "full_50",
    ]

    for n_items in sorted(all_results.keys()):
        results = all_results[n_items]
        log.info("--- %d Items ---", n_items)
        for method in methods_order:
            if method not in results:
                continue
            m = results[method]
            r_str = f"r={m['pearson_r']:.3f}"
            if f"pearson_r_ci" in m:
                ci = m["pearson_r_ci"]
                r_str += f" [{ci[0]:.3f},{ci[1]:.3f}]"
            coverage_raw = m.get("coverage_90")
            if isinstance(coverage_raw, (int, float, np.integer, np.floating)) and np.isfinite(float(coverage_raw)):
                coverage_str = f"{float(coverage_raw) * 100:.1f}%"
            else:
                coverage_str = "n/a"
            log.info(
                "  %-20s %s  MAE=%.2f  <=5%%=%.1f%%  cov=%s",
                method, r_str, m["mae"], m["within_5_pct"] * 100, coverage_str,
            )


def _print_ml_vs_avg_summary(comparison: dict) -> None:
    """Print ML vs averaging summary to log."""
    log.info("ML vs Averaging Comparison:")
    for row in comparison["comparisons"]:
        log.info(
            "  %-20s K=%-3d  ML r=%.4f  Avg r=%.4f  delta=%.4f",
            row["method"], row["n_items"], row["ml_r"], row["avg_r"], row["delta_r"],
        )
        if "delta_r_ci" in row:
            ci = row["delta_r_ci"]
            log.info("    delta CI: [%.4f, %.4f]", ci[0], ci[1])


# ============================================================================
# Adaptive item order analysis (starvation proof artifact)
# ============================================================================

def _generate_adaptive_item_order_analysis(
    item_pool: list[dict],
    artifacts_dir: Path,
    provenance: dict[str, Any],
) -> None:
    """Produce adaptive item ordering analysis showing domain starvation.

    Generates cumulative domain counts at various K checkpoints to prove that
    pure adaptive (greedy) selection starves low-information domains.
    """
    checkpoints = [5, 10, 15, 20, 25]

    # Full ranking by composite score (item_pool is already sorted by rank)
    ranking = []
    for item in item_pool:
        ranking.append({
            "rank": item.get("rank"),
            "id": item["id"],
            "home_domain": item["home_domain"],
            "cross_domain_info": item.get("cross_domain_info", 0.0),
            "own_domain_r": item.get("own_domain_r", 0.0),
        })

    # Cumulative domain counts at each checkpoint
    cumulative_counts: dict[str, dict[str, int]] = {}
    for k in checkpoints:
        top_k = item_pool[:k]
        counts = {d: 0 for d in DOMAINS}
        for item in top_k:
            d = item["home_domain"]
            if d in counts:
                counts[d] += 1
        cumulative_counts[str(k)] = counts

    # Starved domains at K=20
    k20_counts = cumulative_counts.get("20", {})
    starved_domains = [d for d in DOMAINS if k20_counts.get(d, 0) == 0]

    # First appearance rank per domain
    first_appearance: dict[str, int | None] = {d: None for d in DOMAINS}
    for i, item in enumerate(item_pool):
        d = item["home_domain"]
        if d in first_appearance and first_appearance[d] is None:
            first_appearance[d] = i + 1  # 1-indexed rank

    analysis = {
        "provenance": provenance,
        "description": "Adaptive (greedy) item ordering analysis showing domain starvation",
        "item_ranking": ranking,
        "cumulative_domain_counts": cumulative_counts,
        "starved_domains_at_k20": starved_domains,
        "first_appearance_rank_per_domain": first_appearance,
    }

    path = artifacts_dir / "adaptive_item_order_analysis.json"
    with open(path, "w") as f:
        json.dump(analysis, f, indent=2)
    log.info("  Saved %s", path)


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare IPIP-BFFM adaptive model against baseline strategies"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to model directory (default: models/reference, relative to PACKAGE_ROOT)",
    )
    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for 95%% CIs (default: 1000, 0 to disable)",
    )
    parser.add_argument(
        "--random-trials",
        type=int,
        default=10,
        help="Number of random selection trials to average (default: 10)",
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
    log.info("IPIP-BFFM Baseline Comparisons")
    log.info("=" * 60)
    log.info("Model dir: %s", model_dir)
    log.info("Data dir: %s", data_dir)
    log.info("Artifacts dir: %s", artifacts_dir)
    log.info("Bootstrap: %d resamples", args.bootstrap_n)
    log.info("Random trials: %d", args.random_trials)

    # Load models
    log.info("Step 1: Loading models and data...")
    domain_models = _load_models(model_dir)
    missing = _check_models_complete(domain_models)
    if missing:
        log.error("Missing models: %s", ", ".join(missing))
        return 1
    n_models = sum(len(m) for m in domain_models.values())
    log.info("  Loaded %d models", n_models)

    # Load calibration
    try:
        sparse_calibration = _load_calibration_params(
            model_dir, "calibration_params_sparse_20_balanced.json"
        )
        if not sparse_calibration:
            # Legacy fallback for older model bundles.
            sparse_calibration = _load_calibration_params(
                model_dir, "calibration_params.json"
            )
        full_calibration = _load_calibration_params(
            model_dir, "calibration_params_full_50.json"
        )
    except (ValueError, json.JSONDecodeError) as e:
        log.error("Invalid calibration parameters: %s", e)
        return 1
    log.info("  Sparse calibration: %d domains", len(sparse_calibration))
    log.info("  Full calibration: %d domains", len(full_calibration))

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
    try:
        test_df = _load_test_data(data_dir)
        _validate_test_schema(test_df)
    except (FileNotFoundError, ValueError) as e:
        log.error("Invalid test data for baselines: %s", e)
        return 1
    log.info("  Test data: %d respondents", len(test_df))

    # Load item info
    try:
        item_info, item_info_sha256 = _load_item_info(data_dir, model_dir)
    except (FileNotFoundError, ValueError) as e:
        log.error("Item info not available: %s", e)
        return 1
    item_pool = item_info["item_pool"]
    log.info("  Item pool: %d items (sha256=%s)", len(item_pool), item_info_sha256[:12])

    # Load Mini-IPIP mapping
    try:
        mini_ipip_mapping = _load_mini_ipip_mapping()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        log.error("%s", e)
        return 1
    mini_ipip_items = flatten_mini_ipip_items(mini_ipip_mapping)
    log.info("  Mini-IPIP items: %d", len(mini_ipip_items))

    # Load Mini-IPIP standalone norms
    try:
        mini_ipip_norms = load_mini_ipip_norms()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        log.error("Mini-IPIP norms unavailable in stage-03 lock file: %s", e)
        log.error(
            "Run stage 03 (make norms) to regenerate artifacts/ipip_bffm_norms.json "
            "with mini_ipip_norms."
        )
        return 1
    log.info("  Mini-IPIP norms loaded for %d domains", len(mini_ipip_norms))

    # Prepare data arrays
    available_items = list(ITEM_COLUMNS)
    X_test = test_df[available_items].astype(np.float64)
    y_cols = [f"{d}_percentile" for d in DOMAINS] + [f"{d}_score" for d in DOMAINS]
    y_test = test_df[y_cols]
    X_values = X_test.values
    all_columns = list(X_test.columns)

    # Run comparisons
    log.info("Step 2: Running baseline comparisons...")
    all_results: dict[int, dict[str, dict[str, Any]]] = {}
    all_per_domain: dict[int, dict[str, dict[str, dict[str, Any]]]] = {}

    for n_items in tqdm(K_VALUES, desc="K values"):
        log.info("  Evaluating at K=%d...", n_items)
        t0 = time.time()

        selected_calibration, calibration_regime = _choose_calibration_for_budget(
            n_items, sparse_calibration, full_calibration,
        )

        overall, per_domain = _run_comparisons_at_k(
            domain_models, X_values, all_columns, X_test, y_test,
            item_pool, available_items, n_items,
            mini_ipip_mapping, mini_ipip_norms, selected_calibration, calibration_regime,
            n_bootstrap=args.bootstrap_n,
            n_random_trials=args.random_trials,
        )

        all_results[n_items] = overall
        all_per_domain[n_items] = per_domain

        elapsed = time.time() - t0
        n_methods = len(overall)
        log.info("    %d methods evaluated in %.1fs (calibration: %s)",
                 n_methods, elapsed, calibration_regime)

    # Print summary
    log.info("Step 3: Results summary...")
    _print_summary(all_results)

    # Per-domain summary at K=20
    if 20 in all_per_domain:
        log.info("Per-domain Pearson r at K=20:")
        for method, per_dom in all_per_domain[20].items():
            domain_rs = []
            for domain in DOMAINS:
                if domain in per_dom:
                    domain_rs.append(f"{DOMAIN_LABELS[domain][:5]}={per_dom[domain]['pearson_r']:.3f}")
            if domain_rs:
                log.info("  %-20s %s", method, "  ".join(domain_rs))

    # ML vs averaging comparison
    log.info("Step 4: ML vs averaging comparison...")
    ml_vs_avg = _run_ml_vs_averaging_comparison(
        domain_models, X_values, all_columns, X_test, y_test,
        item_pool, available_items, mini_ipip_mapping, mini_ipip_norms,
        sparse_calibration, full_calibration,
        n_bootstrap=args.bootstrap_n,
    )
    _print_ml_vs_avg_summary(ml_vs_avg)

    # Save results
    log.info("Step 5: Saving results...")

    provenance = build_provenance(
        Path(__file__).name,
        args=args,
        rng_seed=42,
        extra={
            "model_dir": relative_to_root(model_dir),
            "bootstrap_n": args.bootstrap_n,
            "random_trials": args.random_trials,
            "test_sha256": test_sha256,
            "split_signature": split_signature,
        },
    )

    # Save baseline comparison results
    combined_output: dict[str, Any] = {
        "provenance": provenance,
        "config": {
            "n_sample": None,
            "sample_seed": None,
            "bootstrap_n": args.bootstrap_n,
            "calibration_applied_sparse": bool(sparse_calibration),
            "calibration_applied_full_50": bool(full_calibration),
            "calibration_policy": {
                "n_items_50_or_more": "full_50" if full_calibration else (
                    "sparse_20_balanced" if sparse_calibration else "none"
                ),
                "n_items_below_50": "sparse_20_balanced" if sparse_calibration else "none",
                "fallback_without_sparse_calibration": "none",
            },
        },
        "overall": {str(k): v for k, v in all_results.items()},
        "per_domain": {str(k): v for k, v in all_per_domain.items()},
    }

    results_path = artifacts_dir / "baseline_comparison_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(combined_output, f, indent=2)
    log.info("  Saved %s", results_path)

    # Save per-domain CSV
    per_domain_df = _create_per_domain_csv(all_per_domain)
    csv_path = artifacts_dir / "baseline_comparison_per_domain.csv"
    csv_meta_path = artifacts_dir / "baseline_comparison_per_domain.meta.json"
    _write_per_domain_csv_with_metadata(
        per_domain_df,
        csv_path,
        csv_meta_path,
        provenance,
    )
    log.info("  Saved %s", csv_path)
    log.info("  Saved %s", csv_meta_path)

    # Save ML vs averaging comparison
    ml_avg_output: dict[str, Any] = {
        "provenance": provenance,
        **ml_vs_avg,
    }
    ml_avg_path = artifacts_dir / "ml_vs_averaging_comparison.json"
    with open(ml_avg_path, "w") as f:
        json.dump(ml_avg_output, f, indent=2)
    log.info("  Saved %s", ml_avg_path)

    # Adaptive item order analysis (starvation proof artifact)
    _generate_adaptive_item_order_analysis(item_pool, artifacts_dir, provenance)

    # Final summary
    log.info("=" * 60)
    log.info("Baseline comparison complete!")
    log.info("=" * 60)

    # Highlight key findings
    if 20 in all_results:
        methods_at_20 = all_results[20]
        if "domain_balanced" in methods_at_20:
            log.info("Key result: domain_balanced@20 r=%.3f", methods_at_20["domain_balanced"]["pearson_r"])
        if "mini_ipip" in methods_at_20:
            log.info("Key result: mini_ipip@20 r=%.3f", methods_at_20["mini_ipip"]["pearson_r"])
        if "adaptive_topk" in methods_at_20:
            log.info("Key result: adaptive_topk@20 r=%.3f", methods_at_20["adaptive_topk"]["pearson_r"])

    for row in ml_vs_avg.get("comparisons", []):
        if row["n_items"] == 20:
            log.info("ML vs Avg (%s@%d): delta_r=%+.3f", row["method"], row["n_items"], row["delta_r"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
