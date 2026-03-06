#!/usr/bin/env python3
"""Simulate adaptive assessment on held-out IPIP-BFFM test data.

This script simulates the actual adaptive testing process:
1. Start with the universal first item
2. Use cross-domain correlation utility to select subsequent items
3. Stop when SEM threshold is met for all domains (or max items reached)
4. Compare predicted vs actual percentiles

Reads models from models/reference/, item info from
data/processed/item_info.json, and test data from data/processed/test.parquet.

Usage:
    python pipeline/10_simulate.py --data-dir data/processed/ext_est [options]
    python pipeline/10_simulate.py --data-dir data/processed/ext_est --n-sample 5000 --sem-threshold 0.45
    python pipeline/10_simulate.py --data-dir data/processed/ext_est --sweep-sem-thresholds
"""

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

import argparse
import json
import logging
from dataclasses import dataclass, field, replace
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy.special
from tqdm import tqdm

from lib.constants import DOMAINS, DOMAIN_LABELS, ITEM_COLUMNS
from lib.item_info import (
    load_item_info_for_model,
    load_item_info_strict,
)
from lib.norms import load_norms
from lib.provenance import add_provenance_args, build_provenance, relative_to_root
from lib.provenance_checks import (
    verify_model_data_split_provenance as _verify_model_data_split_provenance,
)
from lib.scoring import raw_score_to_percentile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_QUANTILES = ("q05", "q50", "q95")
DOMAIN_LABELS_DISPLAY = DOMAIN_LABELS
DEFAULT_ARTIFACTS_DIR = Path("artifacts")


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SelectionWeights:
    """Weights for item selection scoring."""

    alpha: float = 0.5  # Cross-domain correlation utility weight
    beta: float = 0.3  # Coverage need weight
    gamma: float = 0.2  # Uncertainty weight


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive assessment."""

    min_items: int = 8
    max_items: int = 50
    ci_width_target: float = 0.5  # Target 90% CI width in RAW SCORE units (1-5 scale)
    min_items_per_domain: int = 4  # At least 4 items per domain before stopping
    target_items_per_domain: dict[str, int] | int = 1
    use_ci_stopping: bool = False
    use_sem_stopping: bool = True
    sem_threshold: float = 0.45
    selection_weights: SelectionWeights = field(default_factory=SelectionWeights)
    selection_strategy: str = "correlation_ranked"


@dataclass
class DomainPrediction:
    """Prediction for a single domain."""

    domain: str
    # Raw score predictions (1-5 scale) - used for stopping criterion
    raw_lower: float
    raw_median: float
    raw_upper: float
    ci_width_raw: float
    # Percentile predictions (0-100 scale) - used for final output
    percentile_lower: float
    percentile_median: float
    percentile_upper: float
    ci_width_pct: float
    # Metadata
    items_used: int
    converged: bool


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def get_item_columns() -> list[str]:
    """Get all item column names."""
    return list(ITEM_COLUMNS)


def load_models(models_dir: Path) -> dict[str, dict[str, Any]]:
    """Load trained domain models (15 .joblib files)."""
    domain_models: dict[str, dict[str, Any]] = {}

    for domain in DOMAINS:
        domain_models[domain] = {}
        for q_name in REQUIRED_QUANTILES:
            model_path = models_dir / f"adaptive_{domain}_{q_name}.joblib"
            if model_path.exists():
                domain_models[domain][q_name] = joblib.load(model_path)

    return domain_models


def get_missing_required_models(
    domain_models: dict[str, dict[str, Any]],
) -> list[str]:
    """Return missing domain/quantile model keys (e.g., 'ext_q05')."""
    missing: list[str] = []
    for domain in DOMAINS:
        models = domain_models.get(domain, {})
        for q_name in REQUIRED_QUANTILES:
            if q_name not in models:
                missing.append(f"{domain}_{q_name}")
    return missing


def load_item_info_for_model_bundle(
    models_dir: Path,
    data_dir: Path,
) -> tuple[dict, Path, str]:
    """Load and provenance-verify stage-05 item_info for the selected model bundle."""
    return load_item_info_for_model(
        models_dir,
        data_dir,
        require_first_item=True,
    )


def load_calibration_params(
    models_dir: Path,
) -> tuple[dict[str, dict[str, float]], str]:
    """Load sparse runtime calibration parameters if available.

    Prefers explicit sparse calibration artifact when present.
    """
    candidate_paths = [
        models_dir / "calibration_params_sparse_20_balanced.json",
        models_dir / "calibration_params.json",
    ]

    calibration_path = next((p for p in candidate_paths if p.exists()), None)
    if calibration_path is None:
        return {}, "none"

    with open(calibration_path) as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Calibration file must be a JSON object: {calibration_path}")

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
        raise ValueError(
            "Calibration file has invalid domain coverage "
            f"({', '.join(details)}): {calibration_path}"
        )

    calibration: dict[str, dict[str, float]] = {}
    for domain in DOMAINS:
        params = payload[domain]
        if not isinstance(params, dict):
            raise ValueError(
                f"Calibration params for domain '{domain}' must be an object: {calibration_path}"
            )
        if "scale_factor" not in params or "observed_coverage" not in params:
            raise ValueError(
                f"Calibration params for domain '{domain}' must include "
                f"'scale_factor' and 'observed_coverage': {calibration_path}"
            )
        try:
            scale = float(params["scale_factor"])
            observed = float(params["observed_coverage"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Calibration params for domain '{domain}' must be numeric: {calibration_path}"
            ) from exc
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"Calibration scale_factor for domain '{domain}' must be finite and > 0: {calibration_path}"
            )
        if not np.isfinite(observed) or observed < 0.0 or observed > 1.0:
            raise ValueError(
                "Calibration observed_coverage for domain "
                f"'{domain}' must be finite and in [0, 1]: {calibration_path}"
            )
        calibration[domain] = {
            "scale_factor": scale,
            "observed_coverage": observed,
        }

    return calibration, calibration_path.name


def calibration_matches_runtime_defaults(config: AdaptiveConfig) -> bool:
    """Sparse calibration is fit for the canonical runtime operating point.

    Only apply it when simulation matches that operating point; otherwise
    leave intervals unscaled to avoid silent cross-regime calibration drift.
    """
    return (
        config.min_items == 8
        and config.max_items == 50
        and config.min_items_per_domain == 4
        and not config.use_ci_stopping
        and config.use_sem_stopping
        and config.selection_strategy in {"correlation_ranked", "domain_balanced"}
    )


def load_test_data(test_path: Path) -> pd.DataFrame:
    """Load held-out test data."""
    return pd.read_parquet(test_path)


def _validate_test_data_schema(test_df: pd.DataFrame) -> None:
    """Fail closed unless simulation input has complete Big-5 response/target columns."""
    required_item_cols = list(ITEM_COLUMNS)
    required_pct_cols = [f"{d}_percentile" for d in DOMAINS]

    missing_item_cols = [c for c in required_item_cols if c not in test_df.columns]
    missing_pct_cols = [c for c in required_pct_cols if c not in test_df.columns]
    if missing_item_cols or missing_pct_cols:
        details = []
        if missing_item_cols:
            details.append("items=" + ",".join(missing_item_cols))
        if missing_pct_cols:
            details.append("percentiles=" + ",".join(missing_pct_cols))
        raise ValueError(
            "Simulation requires full Big-5 schema (50 item responses + 5 percentile columns). "
            + "; ".join(details)
        )


def _safe_pearson_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float | None:
    """Compute Pearson r with finite/constant guards for JSON-safe output."""
    true_arr = np.asarray(y_true, dtype=np.float64)
    pred_arr = np.asarray(y_pred, dtype=np.float64)
    if true_arr.shape != pred_arr.shape:
        raise ValueError(
            "Cannot compute Pearson r for mismatched arrays: "
            f"{true_arr.shape} vs {pred_arr.shape}"
        )

    finite_mask = np.isfinite(true_arr) & np.isfinite(pred_arr)
    if int(np.sum(finite_mask)) < 2:
        return None

    true_f = true_arr[finite_mask]
    pred_f = pred_arr[finite_mask]
    if np.ptp(true_f) <= 0 or np.ptp(pred_f) <= 0:
        return None

    r = float(np.corrcoef(true_f, pred_f)[0, 1])
    if not np.isfinite(r):
        return None
    return r


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_single(
    domain_models: dict[str, dict[str, Any]],
    responses: dict[str, int],
    feature_names: list[str],
    ci_width_target: float = 0.5,
    calibration_params: dict[str, dict[str, float]] | None = None,
    X_reuse: pd.DataFrame | None = None,
    col_to_idx: dict[str, int] | None = None,
) -> dict[str, DomainPrediction]:
    """Get predictions for a single respondent from current responses.

    Models predict RAW SCORES (1-5 scale). Raw score CIs are used for
    stopping decisions. Percentiles are computed for final output only.

    Args:
        domain_models: Trained models for each domain
        responses: Dict mapping item_id -> response (1-5)
        feature_names: List of feature column names
        ci_width_target: Target CI width in raw score units (default: 0.5)
        calibration_params: Optional per-domain percentile CI scale factors
        X_reuse: Optional pre-allocated single-row DataFrame to reuse
        col_to_idx: Optional dict mapping column name -> positional index

    Returns:
        Dict mapping domain -> DomainPrediction
    """
    # Build feature vector with NaN for unanswered items
    if X_reuse is not None and col_to_idx is not None:
        X = X_reuse
        X.iloc[0, :] = np.nan
        for item_id, value in responses.items():
            idx = col_to_idx.get(item_id)
            if idx is not None:
                X.iloc[0, idx] = value
    else:
        X = pd.DataFrame({col: [np.nan] for col in feature_names})
        for item_id, value in responses.items():
            if item_id in X.columns:
                X[item_id] = value

    predictions = {}

    for domain, models in domain_models.items():
        # Get RAW SCORE predictions from models (1-5 scale)
        q_lower_raw = float(models["q05"].predict(X)[0])
        q_median_raw = float(models["q50"].predict(X)[0])
        q_upper_raw = float(models["q95"].predict(X)[0])

        # Enforce monotonicity for stable raw-space CI width used in stopping.
        q_lower_raw, q_median_raw, q_upper_raw = sorted(
            [q_lower_raw, q_median_raw, q_upper_raw]
        )

        # CI width in RAW SCORE units - this is what we use for stopping
        ci_width_raw = q_upper_raw - q_lower_raw

        # Transform raw scores to percentiles (for final output only)
        q_lower_pct = float(raw_score_to_percentile(q_lower_raw, domain))
        q_median_pct = float(raw_score_to_percentile(q_median_raw, domain))
        q_upper_pct = float(raw_score_to_percentile(q_upper_raw, domain))

        # Enforce monotonicity in percentile space for stable reported intervals.
        q_lower_pct, q_median_pct, q_upper_pct = sorted(
            [q_lower_pct, q_median_pct, q_upper_pct]
        )

        # Apply learned calibration in percentile space (when available).
        if calibration_params is not None and domain in calibration_params:
            scale = float(calibration_params[domain].get("scale_factor", 1.0))
            if scale != 1.0:
                half_width = 0.5 * (q_upper_pct - q_lower_pct) * scale
                q_lower_pct = max(0.0, q_median_pct - half_width)
                q_upper_pct = min(100.0, q_median_pct + half_width)
                q_lower_pct, q_median_pct, q_upper_pct = sorted(
                    [q_lower_pct, q_median_pct, q_upper_pct]
                )

        ci_width_pct = q_upper_pct - q_lower_pct
        items_in_domain = sum(1 for item_id in responses if item_id.startswith(domain))

        predictions[domain] = DomainPrediction(
            domain=domain,
            raw_lower=q_lower_raw,
            raw_median=q_median_raw,
            raw_upper=q_upper_raw,
            ci_width_raw=ci_width_raw,
            percentile_lower=q_lower_pct,
            percentile_median=q_median_pct,
            percentile_upper=q_upper_pct,
            ci_width_pct=ci_width_pct,
            items_used=items_in_domain,
            converged=ci_width_raw <= ci_width_target,
        )

    return predictions


# ---------------------------------------------------------------------------
# Item selection utilities
# ---------------------------------------------------------------------------


def compute_correlation_utility(
    item_id: str,
    current_predictions: dict[str, DomainPrediction],
    item_pool: list[dict],
    item_pool_dict: dict[str, dict] | None = None,
) -> float:
    """Compute cross-domain correlation utility for an item.

    Uses the item's domain correlations and current uncertainty (in raw score units)
    to estimate how much the item will reduce uncertainty.
    """
    if item_pool_dict is not None:
        item_info = item_pool_dict.get(item_id)
    else:
        item_info = None
        for item in item_pool:
            if item["id"] == item_id:
                item_info = item
                break

    if item_info is None:
        return 0.0

    total_gain = 0.0
    domain_corrs = item_info.get("domain_correlations", {})

    for domain, pred in current_predictions.items():
        if domain not in domain_corrs:
            continue

        r = abs(domain_corrs[domain])
        current_width = pred.ci_width_raw

        # Expected variance reduction based on correlation
        expected_reduction = current_width * r * r

        # Weight by current uncertainty
        weight = current_width / 4.0

        total_gain += expected_reduction * weight

    return total_gain


def _get_domain_target(domain: str, target: dict[str, int] | int) -> int:
    """Get target items for a specific domain."""
    if isinstance(target, dict):
        return target.get(domain, 4)
    return target


def compute_sem_reduction(
    item_domain: str,
    domain_item_counts: dict[str, int],
    inter_item_r_bars: dict[str, float],
    norms: dict[str, dict[str, float]] | None = None,
) -> float:
    """Estimate SEM reduction from adding one more item to this domain.

    Uses Spearman-Brown prophecy formula:
        alpha_k = (k * r_bar) / (1 + (k - 1) * r_bar)
        SEM     = domain_sd * sqrt(1 - alpha_k)

    Returns:
        delta_SEM = current_SEM - new_SEM  (positive means improvement)
    """
    norms_map = norms if norms is not None else load_norms()

    if item_domain not in norms_map:
        return 0.0

    current_k = domain_item_counts.get(item_domain, 0)
    r_bar = inter_item_r_bars.get(item_domain, 0.3)
    domain_sd = norms_map[item_domain]["sd"]

    # Current SEM
    if current_k <= 0:
        current_sem = float("inf")
    else:
        alpha_k = (current_k * r_bar) / (1 + (current_k - 1) * r_bar)
        alpha_k = max(0.0, min(1.0, alpha_k))
        current_sem = domain_sd * np.sqrt(1 - alpha_k)

    # Projected SEM after adding one more item
    new_k = current_k + 1
    alpha_new = (new_k * r_bar) / (1 + (new_k - 1) * r_bar)
    alpha_new = max(0.0, min(1.0, alpha_new))
    new_sem = domain_sd * np.sqrt(1 - alpha_new)

    return current_sem - new_sem


def compute_item_score(
    item_id: str,
    current_predictions: dict[str, DomainPrediction],
    item_pool: list[dict],
    domain_coverage: dict[str, int],
    config: AdaptiveConfig,
    answered_items: set[str] | None = None,
    inter_item_r_bars: dict[str, float] | None = None,
    item_pool_dict: dict[str, dict] | None = None,
    norms: dict[str, dict[str, float]] | None = None,
) -> float:
    """Compute weighted item score combining correlation utility, coverage need,
    and uncertainty.

    Score = alpha * info_component + beta * (coverage_need/6.0) + gamma * (ci_width_raw/4.0)

    Within-domain dampening (0.75^n) is applied to info_component to discourage
    over-sampling a single domain.
    """
    if item_pool_dict is not None:
        item_info = item_pool_dict.get(item_id)
    else:
        item_info = None
        for item in item_pool:
            if item["id"] == item_id:
                item_info = item
                break

    if item_info is None:
        return -float("inf")

    home_domain = item_info.get("home_domain", item_id.rstrip("0123456789"))
    weights = config.selection_weights

    # Component 1: Information utility
    if config.use_sem_stopping and inter_item_r_bars is not None:
        sem_delta = compute_sem_reduction(
            home_domain, domain_coverage, inter_item_r_bars, norms=norms
        )
        info_component = min(sem_delta, 1.0) / 0.3
    else:
        correlation_utility = compute_correlation_utility(
            item_id, current_predictions, item_pool, item_pool_dict=item_pool_dict
        )
        info_component = correlation_utility / 2.0

    # Within-domain dampening
    n_same_domain = domain_coverage.get(home_domain, 0)
    dampening = 0.75**n_same_domain
    info_component *= dampening

    # Component 2: Coverage need
    current_count = domain_coverage.get(home_domain, 0)
    target_count = _get_domain_target(home_domain, config.target_items_per_domain)
    coverage_need = max(0, target_count - current_count)
    coverage_component = coverage_need / 6.0

    # Component 3: Uncertainty (CI width for this domain in RAW SCORE units)
    if home_domain in current_predictions:
        ci_width = current_predictions[home_domain].ci_width_raw
    else:
        ci_width = 4.0
    uncertainty_component = ci_width / 4.0

    score = (
        weights.alpha * info_component
        + weights.beta * coverage_component
        + weights.gamma * uncertainty_component
    )

    return score


def _pick_target_domain(
    domain_coverage: dict[str, int],
    current_predictions: dict[str, DomainPrediction],
) -> str:
    """Pick the domain with fewest items answered; break ties by highest uncertainty.

    Used by both domain_balanced and correlation_ranked strategies.
    """
    min_count = min(domain_coverage.get(d, 0) for d in DOMAINS)
    tied_domains = [d for d in DOMAINS if domain_coverage.get(d, 0) == min_count]

    if len(tied_domains) == 1:
        return tied_domains[0]

    best_domain = tied_domains[0]
    best_uncertainty = -1.0
    for d in tied_domains:
        if d in current_predictions:
            uncertainty = current_predictions[d].ci_width_raw
        else:
            uncertainty = 4.0
        if uncertainty > best_uncertainty:
            best_uncertainty = uncertainty
            best_domain = d
    return best_domain


def select_next_item_correlation_ranked(
    answered_items: set[str],
    current_predictions: dict[str, DomainPrediction],
    domain_coverage: dict[str, int],
    domain_item_ranking: dict[str, list[str]],
) -> str | None:
    """Select the next item using static correlation-ranked order within domains.

    Strategy (matches baseline's select_items_domain_balanced):
    1. Find domain(s) with fewest items answered (round-robin)
    2. Among tied domains, pick highest uncertainty
    3. From that domain, pick the next item by own_domain_r rank (no scoring)
    4. Fall back to least-covered domain with remaining items

    Args:
        answered_items: Set of already answered item IDs
        current_predictions: Current predictions for each domain
        domain_coverage: Current count of items answered per domain
        domain_item_ranking: Pre-sorted item IDs per domain by own_domain_r (desc)

    Returns:
        item_id or None if no items available
    """
    target_domain = _pick_target_domain(domain_coverage, current_predictions)

    # Pick the first unanswered item from this domain's ranked list
    for item_id in domain_item_ranking.get(target_domain, []):
        if item_id not in answered_items:
            return item_id

    # Fallback: try other domains in order of fewest items
    domain_counts = [
        (d, domain_coverage.get(d, 0)) for d in DOMAINS if d != target_domain
    ]
    domain_counts.sort(key=lambda x: x[1])
    for d, _ in domain_counts:
        for item_id in domain_item_ranking.get(d, []):
            if item_id not in answered_items:
                return item_id

    return None


def select_next_item_balanced(
    answered_items: set[str],
    current_predictions: dict[str, DomainPrediction],
    item_pool: list[dict],
    config: AdaptiveConfig,
    domain_coverage: dict[str, int],
    inter_item_r_bars: dict[str, float] | None = None,
    item_pool_dict: dict[str, dict] | None = None,
    norms: dict[str, dict[str, float]] | None = None,
) -> str | None:
    """Select the next item using strict domain-balanced (round-robin) selection.

    Strategy:
    1. Find domain(s) with the fewest items answered so far
    2. Among tied domains, pick the one with highest uncertainty (ci_width_raw)
    3. From that domain's remaining items, select the one with highest compute_item_score
    4. Fall back to any unanswered item if the selected domain has no items left
    """
    target_domain = _pick_target_domain(domain_coverage, current_predictions)

    best_item = None
    best_score = -float("inf")

    for item in item_pool:
        item_id = item["id"]
        if item_id in answered_items:
            continue

        item_domain = item.get("home_domain", item_id.rstrip("0123456789"))
        if item_domain != target_domain:
            continue

        score = compute_item_score(
            item_id,
            current_predictions,
            item_pool,
            domain_coverage,
            config,
            answered_items=answered_items,
            inter_item_r_bars=inter_item_r_bars,
            item_pool_dict=item_pool_dict,
            norms=norms,
        )

        if score > best_score:
            best_score = score
            best_item = item_id

    # Fallback to any unanswered item
    if best_item is None:
        best_score = -float("inf")
        for item in item_pool:
            item_id = item["id"]
            if item_id in answered_items:
                continue

            score = compute_item_score(
                item_id,
                current_predictions,
                item_pool,
                domain_coverage,
                config,
                answered_items=answered_items,
                inter_item_r_bars=inter_item_r_bars,
                item_pool_dict=item_pool_dict,
                norms=norms,
            )

            if score > best_score:
                best_score = score
                best_item = item_id

    return best_item


def select_next_item(
    answered_items: set[str],
    current_predictions: dict[str, DomainPrediction],
    item_pool: list[dict],
    config: AdaptiveConfig,
    domain_coverage: dict[str, int],
    inter_item_r_bars: dict[str, float] | None = None,
    item_pool_dict: dict[str, dict] | None = None,
    norms: dict[str, dict[str, float]] | None = None,
) -> str | None:
    """Select the next item to administer using weighted scoring (original strategy)."""
    best_item = None
    best_score = -float("inf")

    for item in item_pool:
        item_id = item["id"]
        if item_id in answered_items:
            continue

        score = compute_item_score(
            item_id,
            current_predictions,
            item_pool,
            domain_coverage,
            config,
            answered_items=answered_items,
            inter_item_r_bars=inter_item_r_bars,
            item_pool_dict=item_pool_dict,
            norms=norms,
        )

        if score > best_score:
            best_score = score
            best_item = item_id

    return best_item


# ---------------------------------------------------------------------------
# Stopping criteria
# ---------------------------------------------------------------------------


def compute_domain_sem(
    domain: str,
    n_items_in_domain: int,
    inter_item_r_bars: dict[str, float],
    norms: dict[str, dict[str, float]] | None = None,
) -> float:
    """Compute Standard Error of Measurement for a domain using Spearman-Brown.

    Args:
        domain: Domain key (ext, agr, csn, est, opn)
        n_items_in_domain: Number of items answered in this domain (k)
        inter_item_r_bars: Dict mapping domain -> mean inter-item correlation (r_bar)

    Returns:
        SEM value (in raw score units)
    """
    if domain not in inter_item_r_bars:
        raise ValueError(
            f"Missing inter_item_r_bar for domain '{domain}'. "
            "Re-run stage 05 to regenerate item_info.json."
        )
    r_bar = float(inter_item_r_bars[domain])
    k = n_items_in_domain

    if not np.isfinite(r_bar) or r_bar <= 0:
        raise ValueError(
            f"Invalid inter_item_r_bar for domain '{domain}': {r_bar!r}. "
            "Expected a finite value > 0 from stage 05."
        )

    if k <= 0:
        return float("inf")

    alpha_k = (k * r_bar) / (1 + (k - 1) * r_bar)
    alpha_k = max(0.0, min(1.0, alpha_k))

    norms_map = norms if norms is not None else load_norms()
    domain_sd = norms_map[domain]["sd"]
    sem = domain_sd * np.sqrt(1 - alpha_k)

    return float(sem)


def _validate_inter_item_r_bars(inter_item_r_bars: Any) -> dict[str, float]:
    """Validate SEM metadata and fail closed on missing/invalid domains."""
    if not isinstance(inter_item_r_bars, dict):
        raise ValueError(
            "SEM stopping requires inter_item_r_bar data in item_info as a domain->value mapping. "
            "Re-run compute_correlations to generate it."
        )

    validated: dict[str, float] = {}
    missing: list[str] = []
    invalid: list[str] = []

    for domain in DOMAINS:
        raw = inter_item_r_bars.get(domain)
        if raw is None:
            missing.append(domain)
            continue
        if not isinstance(raw, (int, float, np.integer, np.floating)):
            invalid.append(domain)
            continue
        value = float(raw)
        if not np.isfinite(value) or value <= 0:
            invalid.append(domain)
            continue
        validated[domain] = value

    if missing or invalid:
        details = []
        if missing:
            details.append(f"missing={','.join(missing)}")
        if invalid:
            details.append(f"invalid={','.join(invalid)}")
        raise ValueError(
            "SEM stopping requires complete inter_item_r_bar for all domains. "
            + "; ".join(details)
            + ". Re-run stage 05 (make correlations)."
        )

    return validated


def check_stopping_criteria(
    n_items: int,
    predictions: dict[str, DomainPrediction],
    config: AdaptiveConfig,
    domain_coverage: dict[str, int],
    inter_item_r_bars: dict[str, float] | None = None,
    norms: dict[str, dict[str, float]] | None = None,
) -> tuple[bool, str]:
    """Check if stopping criteria are met.

    Control flow (after min/max gates):
    1. If use_sem_stopping=True: SEM convergence is the ONLY stopping criterion.
    2. If use_ci_stopping=True (and SEM off): CI-width convergence.
    3. Otherwise: Fall back to domain coverage stopping.
    """
    if n_items < config.min_items:
        return False, "below_min_items"

    if n_items >= config.max_items:
        return True, "max_items_reached"

    # SEM-based stopping
    if config.use_sem_stopping and inter_item_r_bars is not None:
        all_sem_met = True
        for domain in DOMAINS:
            n_domain = domain_coverage.get(domain, 0)
            if n_domain < config.min_items_per_domain:
                all_sem_met = False
                break
            sem = compute_domain_sem(
                domain,
                n_domain,
                inter_item_r_bars,
                norms=norms,
            )
            if sem > config.sem_threshold:
                all_sem_met = False
                break

        # Domain balance gate: prevent SEM from triggering when domains are
        # severely imbalanced (max-min <= 2)
        if all_sem_met:
            domain_counts = [domain_coverage.get(d, 0) for d in DOMAINS]
            if max(domain_counts) - min(domain_counts) > 2:
                all_sem_met = False

        if all_sem_met:
            return True, "sem_threshold_met"
        return False, "continuing"

    # CI-based stopping
    if config.use_ci_stopping:
        all_converged = all(
            pred.ci_width_raw <= config.ci_width_target
            for pred in predictions.values()
        )
        if all_converged:
            return True, "confidence_met"
        return False, "continuing"

    # Fallback: domain coverage stopping
    all_domains_met = True
    for domain in DOMAINS:
        current = domain_coverage.get(domain, 0)
        target = _get_domain_target(domain, config.target_items_per_domain)
        if current < target:
            all_domains_met = False
            break

    if all_domains_met:
        return True, "domain_coverage_met"

    return False, "continuing"


# ---------------------------------------------------------------------------
# Simulation core
# ---------------------------------------------------------------------------


def simulate_single_respondent(
    true_responses: dict[str, int],
    true_percentiles: dict[str, float],
    domain_models: dict[str, dict[str, Any]],
    item_pool: list[dict],
    first_item_id: str,
    feature_names: list[str],
    config: AdaptiveConfig,
    calibration_params: dict[str, dict[str, float]] | None = None,
    inter_item_r_bars: dict[str, float] | None = None,
    item_pool_dict: dict[str, dict] | None = None,
    X_reuse: pd.DataFrame | None = None,
    col_to_idx: dict[str, int] | None = None,
    domain_item_ranking: dict[str, list[str]] | None = None,
    norms: dict[str, dict[str, float]] | None = None,
) -> dict | None:
    """Simulate adaptive assessment for a single respondent.

    Returns simulation results including items administered,
    final predictions, stopping reason, and domain coverage.
    Returns None if the first item is missing from true_responses.
    """
    answered_items: dict[str, int] = {}
    item_sequence: list[str] = []
    domain_coverage = {d: 0 for d in DOMAINS}

    # Start with first item
    if first_item_id not in true_responses:
        return None
    answered_items[first_item_id] = true_responses[first_item_id]
    item_sequence.append(first_item_id)
    first_domain = first_item_id.rstrip("0123456789")
    if first_domain in domain_coverage:
        domain_coverage[first_domain] += 1

    reason = "continuing"

    # Adaptive loop
    while len(answered_items) < config.max_items:
        # Get current predictions
        predictions = predict_single(
            domain_models,
            answered_items,
            feature_names,
            config.ci_width_target,
            calibration_params=calibration_params,
            X_reuse=X_reuse,
            col_to_idx=col_to_idx,
        )

        # Check stopping criteria
        should_stop, reason = check_stopping_criteria(
            len(answered_items),
            predictions,
            config,
            domain_coverage,
            inter_item_r_bars,
            norms=norms,
        )

        if should_stop:
            break

        # Select next item using configured strategy
        if config.selection_strategy == "correlation_ranked":
            next_item = select_next_item_correlation_ranked(
                set(answered_items.keys()),
                predictions,
                domain_coverage,
                domain_item_ranking or {},
            )
        elif config.selection_strategy == "domain_balanced":
            next_item = select_next_item_balanced(
                set(answered_items.keys()),
                predictions,
                item_pool,
                config,
                domain_coverage,
                inter_item_r_bars=inter_item_r_bars,
                item_pool_dict=item_pool_dict,
                norms=norms,
            )
        else:
            next_item = select_next_item(
                set(answered_items.keys()),
                predictions,
                item_pool,
                config,
                domain_coverage,
                inter_item_r_bars=inter_item_r_bars,
                item_pool_dict=item_pool_dict,
                norms=norms,
            )

        if next_item is None:
            reason = "no_items_available"
            break

        # Administer item (get true response)
        if next_item in true_responses:
            answered_items[next_item] = true_responses[next_item]
            item_sequence.append(next_item)
            next_domain = next_item.rstrip("0123456789")
            if next_domain in domain_coverage:
                domain_coverage[next_domain] += 1
        else:
            reason = "missing_response"
            break

    # Override reason when max items hit naturally
    if len(answered_items) >= config.max_items and reason not in (
        "no_items_available",
        "max_items_reached",
    ):
        reason = "max_items_reached"

    # Final predictions
    final_predictions = predict_single(
        domain_models,
        answered_items,
        feature_names,
        config.ci_width_target,
        calibration_params=calibration_params,
        X_reuse=X_reuse,
        col_to_idx=col_to_idx,
    )

    # Compute accuracy
    domain_errors: dict[str, dict[str, Any]] = {}
    for domain, pred in final_predictions.items():
        pct_col = f"{domain}_percentile"
        if pct_col in true_percentiles:
            true_pct = float(true_percentiles[pct_col])
            error = float(abs(true_pct - pred.percentile_median))
            in_interval = pred.percentile_lower <= true_pct <= pred.percentile_upper
            domain_errors[domain] = {
                "true_percentile": true_pct,
                "predicted_percentile": pred.percentile_median,
                "error": error,
                "ci_lower": pred.percentile_lower,
                "ci_upper": pred.percentile_upper,
                "ci_width_pct": pred.ci_width_pct,
                "ci_width_raw": pred.ci_width_raw,
                "in_interval": bool(in_interval),
                "within_5": bool(error <= 5),
                "items_in_domain": domain_coverage.get(domain, 0),
            }

    return {
        "n_items": len(answered_items),
        "stop_reason": reason,
        "item_sequence": item_sequence,
        "domain_errors": domain_errors,
        "domain_coverage": domain_coverage,
        "predictions": {
            d: {
                "median": p.percentile_median,
                "lower": p.percentile_lower,
                "upper": p.percentile_upper,
                "ci_width_pct": p.ci_width_pct,
                "ci_width_raw": p.ci_width_raw,
                "raw_median": p.raw_median,
            }
            for d, p in final_predictions.items()
        },
    }


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------


def run_simulation(
    test_df: pd.DataFrame,
    domain_models: dict[str, dict[str, Any]],
    item_info: dict,
    config: AdaptiveConfig,
    calibration_params: dict[str, dict[str, float]] | None = None,
    n_sample: int | None = None,
    seed: int = 42,
    norms: dict[str, dict[str, float]] | None = None,
) -> list[dict]:
    """Run simulation on test dataset.

    Args:
        test_df: Test data with responses and true percentiles
        domain_models: Trained models
        item_info: Item correlation info (normalised to snake_case)
        config: Adaptive configuration
        calibration_params: Optional per-domain percentile CI scale factors
        n_sample: Number of respondents to sample (None = all)
        seed: Random seed for sampling

    Returns:
        List of simulation results
    """
    _validate_test_data_schema(test_df)
    norms_map = norms if norms is not None else load_norms()

    missing_models = get_missing_required_models(domain_models)
    if missing_models:
        missing_str = ", ".join(sorted(missing_models))
        raise ValueError(
            "Incomplete model bundle: missing required quantile models "
            f"({missing_str}). Re-run training before simulation."
        )

    item_cols = get_item_columns()
    feature_names = list(item_cols)

    first_item_id = item_info["first_item"]["id"]
    item_pool = item_info["item_pool"]
    inter_item_r_bars = item_info.get("inter_item_r_bar", None)

    if config.use_sem_stopping:
        inter_item_r_bars = _validate_inter_item_r_bars(inter_item_r_bars)

    # Build item_pool_dict for O(1) lookup
    item_pool_dict = {item["id"]: item for item in item_pool}

    # Build domain_item_ranking: items per domain sorted by own_domain_r descending
    domain_item_ranking: dict[str, list[str]] = {}
    for domain in DOMAINS:
        domain_items = [
            item for item in item_pool if item.get("home_domain") == domain
        ]
        domain_items.sort(key=lambda x: abs(x.get("own_domain_r", 0)), reverse=True)
        domain_item_ranking[domain] = [item["id"] for item in domain_items]

    # Pre-allocate reusable single-row DataFrame and column index map
    X_reuse = pd.DataFrame({col: [np.nan] for col in feature_names})
    col_to_idx = {col: i for i, col in enumerate(feature_names)}

    # Sample if requested
    if n_sample and n_sample < len(test_df):
        sample_df = test_df.sample(n=n_sample, random_state=seed)
    else:
        sample_df = test_df

    # Pre-extract response and percentile arrays as numpy for fast indexing
    response_arr = sample_df[feature_names].values
    pct_cols = [f"{d}_percentile" for d in DOMAINS]
    pct_arr = sample_df[pct_cols].values

    results: list[dict] = []

    for row_i in tqdm(range(len(sample_df)), desc="Simulating respondents"):
        # Extract true responses from pre-extracted numpy array
        row_vals = response_arr[row_i]
        true_responses: dict[str, int] = {}
        for col_i, col in enumerate(feature_names):
            val = row_vals[col_i]
            if not np.isnan(val):
                true_responses[col] = int(val)

        # Extract true percentiles
        true_percentiles: dict[str, float] = {}
        pct_vals = pct_arr[row_i]
        for col_i, col in enumerate(pct_cols):
            val = pct_vals[col_i]
            if not np.isnan(val):
                true_percentiles[col] = float(val)

        # Run simulation
        result = simulate_single_respondent(
            true_responses,
            true_percentiles,
            domain_models,
            item_pool,
            first_item_id,
            feature_names,
            config,
            calibration_params=calibration_params,
            inter_item_r_bars=inter_item_r_bars,
            item_pool_dict=item_pool_dict,
            X_reuse=X_reuse,
            col_to_idx=col_to_idx,
            domain_item_ranking=domain_item_ranking,
            norms=norms_map,
        )

        if result is not None:
            results.append(result)

    if not results:
        raise ValueError(
            "Simulation produced zero valid respondents. "
            "Check first-item coverage and test data integrity."
        )

    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze_simulation_results(results: list[dict]) -> dict:
    """Analyze simulation results and compute aggregate metrics."""
    if not results:
        raise ValueError("Cannot analyze simulation results: results list is empty.")

    n_items_list = [r["n_items"] for r in results]
    stop_reasons = [r["stop_reason"] for r in results]

    # Items to convergence statistics
    items_stats = {
        "mean": float(np.mean(n_items_list)),
        "std": float(np.std(n_items_list)),
        "median": float(np.median(n_items_list)),
        "min": int(np.min(n_items_list)),
        "max": int(np.max(n_items_list)),
        "percentile_25": float(np.percentile(n_items_list, 25)),
        "percentile_75": float(np.percentile(n_items_list, 75)),
        "percentile_90": float(np.percentile(n_items_list, 90)),
    }

    # Stop reason counts
    reason_counts: dict[str, int] = {}
    for reason in stop_reasons:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    # Per-domain coverage statistics
    domain_coverage_stats: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        coverage_counts = []
        for r in results:
            if "domain_coverage" in r:
                coverage_counts.append(r["domain_coverage"].get(domain, 0))
            elif domain in r["domain_errors"]:
                coverage_counts.append(
                    r["domain_errors"][domain].get("items_in_domain", 0)
                )

        if coverage_counts:
            domain_coverage_stats[domain] = {
                "mean_items": float(np.mean(coverage_counts)),
                "std_items": float(np.std(coverage_counts)),
                "min_items": int(np.min(coverage_counts)),
                "max_items": int(np.max(coverage_counts)),
            }

    # Per-domain metrics (including Pearson r)
    domain_metrics: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        errors = []
        within_5 = []
        in_interval = []
        ci_widths_pct = []
        ci_widths_raw = []
        items_in_domain = []
        true_pcts = []
        pred_pcts = []

        for r in results:
            if domain in r["domain_errors"]:
                de = r["domain_errors"][domain]
                errors.append(de["error"])
                within_5.append(de["within_5"])
                in_interval.append(de["in_interval"])
                ci_widths_pct.append(de["ci_width_pct"])
                ci_widths_raw.append(de["ci_width_raw"])
                true_pcts.append(de["true_percentile"])
                pred_pcts.append(de["predicted_percentile"])
                if "items_in_domain" in de:
                    items_in_domain.append(de["items_in_domain"])

        if errors:
            pearson_r = _safe_pearson_correlation(
                np.asarray(true_pcts, dtype=np.float64),
                np.asarray(pred_pcts, dtype=np.float64),
            )

            domain_metrics[domain] = {
                "pearson_r": pearson_r,
                "mae": float(np.mean(errors)),
                "rmse": float(np.sqrt(np.mean(np.array(errors) ** 2))),
                "within_5_pct": float(np.mean(within_5)),
                "coverage_90": float(np.mean(in_interval)),
                "mean_ci_width_pct": float(np.mean(ci_widths_pct)),
                "mean_ci_width_raw": float(np.mean(ci_widths_raw)),
                "mean_items_used": (
                    float(np.mean(items_in_domain)) if items_in_domain else None
                ),
            }

    # Overall accuracy (including Pearson r)
    all_errors = []
    all_within_5 = []
    all_in_interval = []
    all_true_pcts = []
    all_pred_pcts = []

    for r in results:
        for domain, de in r["domain_errors"].items():
            all_errors.append(de["error"])
            all_within_5.append(de["within_5"])
            all_in_interval.append(de["in_interval"])
            all_true_pcts.append(de["true_percentile"])
            all_pred_pcts.append(de["predicted_percentile"])

    overall_pearson_r = _safe_pearson_correlation(
        np.asarray(all_true_pcts, dtype=np.float64),
        np.asarray(all_pred_pcts, dtype=np.float64),
    )

    overall_metrics = {
        "pearson_r": overall_pearson_r,
        "mae": float(np.mean(all_errors)),
        "rmse": float(np.sqrt(np.mean(np.array(all_errors) ** 2))),
        "within_5_pct": float(np.mean(all_within_5)),
        "coverage_90": float(np.mean(all_in_interval)),
    }

    return {
        "n_respondents": len(results),
        "items_to_convergence": items_stats,
        "stop_reasons": reason_counts,
        "domain_coverage_stats": domain_coverage_stats,
        "domain_metrics": domain_metrics,
        "overall_metrics": overall_metrics,
    }


# ---------------------------------------------------------------------------
# SEM threshold sweep
# ---------------------------------------------------------------------------


def run_sem_threshold_sweep(
    test_df: pd.DataFrame,
    domain_models: dict[str, dict[str, Any]],
    item_info: dict,
    base_config: AdaptiveConfig,
    thresholds: list[float],
    calibration_params: dict[str, dict[str, float]] | None = None,
    n_sample: int | None = None,
    seed: int = 42,
    norms: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Run simulation at multiple SEM thresholds and collect comparison metrics.

    Args:
        test_df: Test data with responses and true percentiles
        domain_models: Trained models
        item_info: Item correlation info
        base_config: Base adaptive configuration (sem_threshold will be overridden)
        thresholds: List of SEM thresholds to sweep
        calibration_params: Optional per-domain percentile CI scale factors
        n_sample: Number of respondents to sample (None = all)
        seed: Random seed for sampling

    Returns:
        Dict with per-threshold results and summary
    """
    inter_item_r_bars = item_info.get("inter_item_r_bar", {})
    norms_map = norms if norms is not None else load_norms()

    sweep_results: dict[str, Any] = {
        "thresholds": thresholds,
        "per_threshold": {},
    }

    for i, threshold in enumerate(thresholds):
        log.info(
            "Running threshold %.2f (%d/%d)...", threshold, i + 1, len(thresholds)
        )

        config = replace(base_config, use_sem_stopping=True, sem_threshold=threshold)

        results = run_simulation(
            test_df,
            domain_models,
            item_info,
            config,
            calibration_params=calibration_params,
            n_sample=n_sample,
            seed=seed,
            norms=norms_map,
        )

        analysis = analyze_simulation_results(results)

        # Compute per-domain mean SEM at stopping
        domain_sem_at_stop: dict[str, float] = {}
        for domain in DOMAINS:
            sem_values = []
            for r in results:
                n_domain = r["domain_coverage"].get(domain, 0)
                sem = compute_domain_sem(
                    domain,
                    n_domain,
                    inter_item_r_bars,
                    norms=norms_map,
                )
                sem_values.append(sem)
            if sem_values:
                domain_sem_at_stop[domain] = float(np.mean(sem_values))

        # Compute SEM-stopped percentage
        total = analysis["n_respondents"]
        sem_stopped = analysis["stop_reasons"].get("sem_threshold_met", 0)
        sem_stopped_pct = sem_stopped / total * 100 if total > 0 else 0.0

        sweep_results["per_threshold"][str(threshold)] = {
            "analysis": analysis,
            "domain_sem_at_stop": domain_sem_at_stop,
            "sem_stopped_pct": sem_stopped_pct,
        }

    # Print formatted summary table
    log.info("")
    log.info("SEM Threshold Sweep Results")
    log.info("=" * 82)
    log.info(
        "%-12s%-13s%-9s%-8s%-10s%-11s%-14s",
        "Threshold",
        "Mean Items",
        "Median",
        "MAE",
        "<=5pct",
        "Coverage",
        "SEM-stopped%",
    )
    log.info("-" * 82)

    for threshold in thresholds:
        entry = sweep_results["per_threshold"][str(threshold)]
        a = entry["analysis"]
        items = a["items_to_convergence"]
        overall = a["overall_metrics"]
        w5 = f"{overall['within_5_pct'] * 100:.1f}%"
        cov = f"{overall['coverage_90'] * 100:.1f}%"
        sem_pct = f"{entry['sem_stopped_pct']:.1f}%"
        log.info(
            "%-12.2f%-13.1f%-9.0f%-8.2f%-10s%-11s%-14s",
            threshold,
            items["mean"],
            items["median"],
            overall["mae"],
            w5,
            cov,
            sem_pct,
        )

    log.info("=" * 82)

    return sweep_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simulate IPIP-BFFM adaptive assessment"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PACKAGE_ROOT / "models" / "reference",
        help="Path to directory containing .joblib models (default: models/reference/)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Data directory containing test.parquet and item_info.json",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Artifacts output directory (default: artifacts)",
    )
    parser.add_argument(
        "--n-sample",
        type=int,
        default=5000,
        help="Number of respondents to simulate (default: 5000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--sem-threshold",
        type=float,
        default=0.45,
        help="SEM threshold for stopping in raw score units (default: 0.45)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=50,
        help="Maximum items (default: 50)",
    )
    parser.add_argument(
        "--min-items",
        type=int,
        default=8,
        help="Minimum items before stopping check (default: 8)",
    )
    parser.add_argument(
        "--min-items-per-domain",
        type=int,
        default=4,
        help="Minimum items per domain before SEM stopping can trigger (default: 4)",
    )
    parser.add_argument(
        "--ci-width-target",
        type=float,
        default=0.5,
        help="Target 90%% CI width in RAW SCORE units on 1-5 scale (default: 0.5)",
    )
    parser.add_argument(
        "--use-ci-stopping",
        action="store_true",
        help=(
            "Use CI-width stopping criterion (disables default SEM stopping "
            "unless --use-sem-stopping is explicitly set)"
        ),
    )
    sem_group = parser.add_mutually_exclusive_group()
    sem_group.add_argument(
        "--use-sem-stopping",
        dest="use_sem_stopping",
        action="store_true",
        help="Use Cronbach SEM stopping criterion",
    )
    sem_group.add_argument(
        "--no-sem-stopping",
        dest="use_sem_stopping",
        action="store_false",
        help="Disable Cronbach SEM stopping",
    )
    parser.add_argument(
        "--selection-strategy",
        type=str,
        choices=["weighted", "domain_balanced", "correlation_ranked"],
        default="correlation_ranked",
        help="Item selection strategy (default: correlation_ranked)",
    )
    parser.add_argument(
        "--selection-alpha",
        type=float,
        default=0.5,
        help="Weight for cross-domain correlation utility (default: 0.5)",
    )
    parser.add_argument(
        "--selection-beta",
        type=float,
        default=0.3,
        help="Weight for coverage need (default: 0.3)",
    )
    parser.add_argument(
        "--selection-gamma",
        type=float,
        default=0.2,
        help="Weight for uncertainty (default: 0.2)",
    )
    parser.add_argument(
        "--target-items-per-domain",
        type=int,
        default=1,
        help="Target items per domain for stopping (default: 1)",
    )
    parser.add_argument(
        "--sweep-sem-thresholds",
        action="store_true",
        help="Run simulation at multiple SEM thresholds and produce comparison table",
    )
    parser.add_argument(
        "--sweep-thresholds",
        type=float,
        nargs="+",
        default=[0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        help="List of SEM thresholds to sweep (default: 0.20 0.25 ... 0.50)",
    )

    add_provenance_args(parser)
    parser.set_defaults(use_sem_stopping=None)
    args = parser.parse_args()

    # Validate arguments
    if args.n_sample is not None and args.n_sample <= 0:
        log.error("--n-sample must be > 0.")
        return 1
    if args.min_items < 1:
        log.error("--min-items must be >= 1.")
        return 1
    if args.max_items < args.min_items:
        log.error("--max-items must be >= --min-items.")
        return 1
    if args.min_items_per_domain < 0:
        log.error("--min-items-per-domain must be >= 0.")
        return 1
    if args.sem_threshold <= 0:
        log.error("--sem-threshold must be > 0.")
        return 1

    # Resolve paths
    models_dir = args.model_dir
    if not models_dir.is_absolute():
        models_dir = PACKAGE_ROOT / models_dir
    data_dir = args.data_dir
    if not data_dir.is_absolute():
        data_dir = PACKAGE_ROOT / data_dir
    test_path = data_dir / "test.parquet"
    artifacts_dir = args.artifacts_dir
    if not artifacts_dir.is_absolute():
        artifacts_dir = PACKAGE_ROOT / artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if not models_dir.exists():
        log.error("Model directory not found: %s", models_dir)
        return 1
    if not test_path.exists():
        log.error("test.parquet not found at %s. Run stage 04 (make prepare) first.", test_path)
        return 1
    try:
        norms_map = load_norms()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        log.error("Failed to load norms: %s", e)
        return 1

    # Resolve SEM stopping default behavior:
    # - sweep mode: always on
    # - explicit CLI flag: honor it
    # - otherwise: enabled by default unless CI mode requested
    if args.sweep_sem_thresholds:
        sem_stopping_enabled = True
    elif args.use_sem_stopping is None:
        sem_stopping_enabled = not args.use_ci_stopping
    else:
        sem_stopping_enabled = bool(args.use_sem_stopping)

    # Build selection weights
    selection_weights = SelectionWeights(
        alpha=args.selection_alpha,
        beta=args.selection_beta,
        gamma=args.selection_gamma,
    )

    config = AdaptiveConfig(
        min_items=args.min_items,
        max_items=args.max_items,
        ci_width_target=args.ci_width_target,
        min_items_per_domain=args.min_items_per_domain,
        target_items_per_domain=args.target_items_per_domain,
        use_ci_stopping=args.use_ci_stopping,
        use_sem_stopping=sem_stopping_enabled,
        sem_threshold=args.sem_threshold,
        selection_weights=selection_weights,
        selection_strategy=args.selection_strategy,
    )

    log.info("=" * 60)
    log.info("IPIP-BFFM Adaptive Assessment Simulation")
    log.info("=" * 60)

    log.info("Configuration:")
    log.info("  Model dir: %s", models_dir)
    log.info("  Data dir: %s", data_dir)
    log.info("  Artifacts dir: %s", artifacts_dir)
    log.info("  Min items: %d", config.min_items)
    log.info("  Max items: %d", config.max_items)
    log.info("  Min items per domain: %d", config.min_items_per_domain)
    log.info("  Target items per domain: %s", config.target_items_per_domain)
    log.info(
        "  CI width target: %.2f (raw score units, 1-5 scale)", config.ci_width_target
    )
    log.info("  Use CI stopping: %s", config.use_ci_stopping)
    if args.sweep_sem_thresholds:
        log.info("  Use SEM stopping: True (sweep mode)")
        log.info("  SEM thresholds: %s", args.sweep_thresholds)
    else:
        log.info("  Use SEM stopping: %s", config.use_sem_stopping)
        log.info("  SEM threshold: %.3f", config.sem_threshold)
    log.info(
        "  Selection weights: alpha=%.2f, beta=%.2f, gamma=%.2f",
        selection_weights.alpha,
        selection_weights.beta,
        selection_weights.gamma,
    )
    log.info("  Selection strategy: %s", config.selection_strategy)
    log.info("  Sample size: %s", args.n_sample if args.n_sample else "all")
    log.info("  Seed: %d", args.seed)

    # Load resources
    log.info("")
    log.info("1. Loading models and data...")
    domain_models = load_models(models_dir)
    n_models = sum(len(m) for m in domain_models.values())
    log.info("   Loaded %d models from %s", n_models, models_dir)

    missing_models = get_missing_required_models(domain_models)
    if missing_models:
        log.error("Incomplete model bundle. Missing required models:")
        for key in sorted(missing_models):
            log.error("   - adaptive_%s.joblib", key)
        log.error(
            "Run training pipeline to regenerate all 15 domain/quantile models."
        )
        return 1

    try:
        item_info, item_info_path, item_info_sha256 = load_item_info_for_model_bundle(
            models_dir,
            data_dir,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        log.error("Failed to load model-coupled item_info: %s", e)
        return 1
    log.info("   First item: %s", item_info["first_item"]["id"])
    log.info(
        "   Item pool: %d items (%s, sha256=%s)",
        len(item_info["item_pool"]),
        item_info_path,
        item_info_sha256[:12],
    )

    try:
        calibration_params, calibration_source = load_calibration_params(models_dir)
    except (ValueError, json.JSONDecodeError) as e:
        log.error("Failed to load calibration parameters: %s", e)
        return 1

    test_sha256: str | None = None
    split_signature: str | None = None
    try:
        test_sha256, split_signature = _verify_model_data_split_provenance(
            model_dir=models_dir,
            data_dir=data_dir,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        log.error("Model/data provenance check failed: %s", e)
        return 1
    if test_sha256 is not None:
        log.info("   Test split hash verified against training report")
    if split_signature is not None:
        log.info("   Full split signature verified against training report")

    try:
        test_df = load_test_data(test_path)
        _validate_test_data_schema(test_df)
    except (
        FileNotFoundError,
        OSError,
        ValueError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as e:
        log.error("Failed to load test data from %s: %s", test_path, e)
        return 1
    log.info("   Test data: %d respondents", len(test_df))

    if calibration_params:
        log.info(
            "   Calibration loaded: %d domains (%s)",
            len(calibration_params),
            calibration_source,
        )
        if args.sweep_sem_thresholds:
            log.info("   Calibration disabled for SEM sweeps (regime mismatch risk).")
            calibration_params = {}
            calibration_source = f"disabled-from-{calibration_source}"
        elif not calibration_matches_runtime_defaults(config):
            log.info(
                "   Calibration disabled (simulation config differs from sparse calibration regime)."
            )
            calibration_params = {}
            calibration_source = f"disabled-from-{calibration_source}"
    else:
        log.info("   Calibration loaded: none (using unscaled percentile intervals)")

    # Branch: SEM threshold sweep mode
    if args.sweep_sem_thresholds:
        log.info("")
        log.info(
            "2. Running SEM threshold sweep (%d thresholds)...",
            len(args.sweep_thresholds),
        )

        sweep_results = run_sem_threshold_sweep(
            test_df,
            domain_models,
            item_info,
            config,
            thresholds=args.sweep_thresholds,
            calibration_params=calibration_params,
            n_sample=args.n_sample,
            seed=args.seed,
            norms=norms_map,
        )
        sweep_results["provenance"] = build_provenance(
            Path(__file__).name,
            args=args,
            rng_seed=args.seed,
            extra={
                "model_dir": relative_to_root(models_dir),
                "item_info_sha256": item_info_sha256,
                "calibration_source": calibration_source,
                "n_sample": args.n_sample,
                "test_sha256": test_sha256,
                "split_signature": split_signature,
            },
        )

        # Save sweep results
        sweep_path = artifacts_dir / "sem_threshold_sweep.json"
        with open(sweep_path, "w") as f:
            json.dump(sweep_results, f, indent=2)
        log.info("   Saved sweep results to %s", sweep_path)

        log.info("")
        log.info("=" * 60)
        log.info("SEM threshold sweep complete!")
        log.info("=" * 60)
        return 0

    # Run simulation
    log.info("")
    log.info("2. Running simulation...")
    results = run_simulation(
        test_df,
        domain_models,
        item_info,
        config,
        calibration_params=calibration_params,
        n_sample=args.n_sample,
        seed=args.seed,
        norms=norms_map,
    )

    # Analyze results
    log.info("")
    log.info("3. Analyzing results...")
    analysis = analyze_simulation_results(results)

    # Print summary
    log.info("")
    log.info("=" * 60)
    log.info("SIMULATION RESULTS")
    log.info("=" * 60)

    stats = analysis["items_to_convergence"]
    log.info("")
    log.info("Items to Convergence:")
    log.info("  Mean: %.1f +/- %.1f", stats["mean"], stats["std"])
    log.info("  Median: %.0f", stats["median"])
    log.info("  Range: %d - %d", stats["min"], stats["max"])
    log.info("  90th percentile: %.0f", stats["percentile_90"])

    log.info("")
    log.info("Stop Reasons:")
    for reason, count in analysis["stop_reasons"].items():
        pct = count / analysis["n_respondents"] * 100
        log.info("  %s: %d (%.1f%%)", reason, count, pct)

    log.info("")
    log.info("Per-Domain Coverage:")
    for domain in DOMAINS:
        if domain in analysis.get("domain_coverage_stats", {}):
            cov = analysis["domain_coverage_stats"][domain]
            log.info(
                "  %s: %.1f +/- %.1f items (range: %d-%d)",
                DOMAIN_LABELS_DISPLAY.get(domain, domain),
                cov["mean_items"],
                cov["std_items"],
                cov["min_items"],
                cov["max_items"],
            )

    log.info("")
    log.info("Per-Domain Accuracy:")
    for domain, m in analysis["domain_metrics"].items():
        items_str = (
            f", items={m['mean_items_used']:.1f}" if m.get("mean_items_used") else ""
        )
        r_str = f"r={m['pearson_r']:.3f}, " if m.get("pearson_r") is not None else ""
        log.info(
            "  %s: %sMAE=%.2f, within 5=%.1f%%, coverage=%.1f%%%s",
            DOMAIN_LABELS_DISPLAY.get(domain, domain),
            r_str,
            m["mae"],
            m["within_5_pct"] * 100,
            m["coverage_90"] * 100,
            items_str,
        )

    log.info("")
    log.info("Overall Accuracy:")
    m = analysis["overall_metrics"]
    if m.get("pearson_r") is not None:
        log.info("  Pearson r: %.3f", m["pearson_r"])
    log.info("  MAE: %.2f", m["mae"])
    log.info("  Within 5 percentile points: %.1f%%", m["within_5_pct"] * 100)
    log.info("  90%% CI coverage: %.1f%%", m["coverage_90"] * 100)

    # Save results
    log.info("")
    log.info("4. Saving results...")

    # Serialize target_items_per_domain for JSON
    target_items_serialized = (
        config.target_items_per_domain
        if isinstance(config.target_items_per_domain, int)
        else dict(config.target_items_per_domain)
    )

    provenance = build_provenance(
        Path(__file__).name,
        args=args,
        rng_seed=args.seed,
        extra={
            "model_dir": relative_to_root(models_dir),
            "item_info_sha256": item_info_sha256,
            "calibration_source": calibration_source,
            "n_sample": args.n_sample,
            "test_sha256": test_sha256,
            "split_signature": split_signature,
        },
    )

    output_path = artifacts_dir / "simulation_results.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "config": {
                    "min_items": config.min_items,
                    "max_items": config.max_items,
                    "ci_width_target": config.ci_width_target,
                    "min_items_per_domain": config.min_items_per_domain,
                    "target_items_per_domain": target_items_serialized,
                    "use_ci_stopping": config.use_ci_stopping,
                    "use_sem_stopping": config.use_sem_stopping,
                    "sem_threshold": config.sem_threshold,
                    "selection_weights": {
                        "alpha": config.selection_weights.alpha,
                        "beta": config.selection_weights.beta,
                        "gamma": config.selection_weights.gamma,
                    },
                    "selection_strategy": config.selection_strategy,
                    "n_sample": args.n_sample,
                    "calibration_applied": bool(calibration_params),
                    "calibration_source": calibration_source,
                },
                "analysis": analysis,
                "provenance": provenance,
            },
            f,
            indent=2,
        )
    log.info("   Saved to %s", output_path)

    # Save detailed results for further analysis
    detailed_path = artifacts_dir / "simulation_detailed.json"
    with open(detailed_path, "w") as f:
        json.dump(results, f)
    log.info("   Saved detailed results to %s", detailed_path)

    log.info("")
    log.info("=" * 60)
    log.info("Simulation complete!")
    log.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
