"""
Centralized bootstrap confidence interval utilities for IPIP-BFFM evaluation.

Provides paired (and optionally stratified) percentile bootstrap CIs for
evaluation metrics across the adaptive personality assessment pipeline.

Key functions:
- paired_bootstrap_cis: Core paired percentile bootstrap with respondent-level resampling
- stratified_paired_bootstrap_cis: Stratified resampling by split_stratum
- bootstrap_metric_deltas: CIs for metric deltas between two prediction sets
- vectorized_pearsonr_bootstrap: Fast bootstrap for per-domain Pearson r
- respondent_bootstrap_multi_domain: Multi-domain respondent-level bootstrap
"""

from typing import Any, Callable

import numpy as np
from scipy import stats


def vectorized_pearsonr_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute bootstrap Pearson r values using vectorized index generation.

    Generates all bootstrap index arrays at once and computes correlations
    in a loop over pre-allocated arrays.

    Args:
        y_true: Ground truth array of shape (n,).
        y_pred: Prediction array of shape (n,).
        n_bootstrap: Number of bootstrap iterations.
        rng: NumPy random generator.

    Returns:
        Array of bootstrap Pearson r values of shape (n_bootstrap,).
    """
    n_obs = len(y_true)
    all_idx = rng.integers(0, n_obs, size=(n_bootstrap, n_obs))

    boot_rs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = all_idx[i]
        boot_rs[i] = stats.pearsonr(y_true[idx], y_pred[idx])[0]

    return boot_rs


def _generate_bootstrap_indices(
    n_respondents: int,
    n_bootstrap: int,
    rng: np.random.Generator,
    strata: np.ndarray | None = None,
) -> np.ndarray:
    """Generate bootstrap respondent index arrays.

    When strata is None, performs standard paired bootstrap (resample all
    respondents with replacement). When strata is provided, resamples within
    each stratum separately and concatenates.

    Args:
        n_respondents: Number of respondents in the dataset.
        n_bootstrap: Number of bootstrap iterations.
        rng: NumPy random generator.
        strata: Optional array of stratum labels (length n_respondents).
            When provided, resampling is stratified within each unique stratum.

    Returns:
        Array of shape (n_bootstrap, n_respondents) with resampled indices.
    """
    if strata is None:
        return rng.integers(0, n_respondents, size=(n_bootstrap, n_respondents))

    # Stratified: resample within each stratum, then stitch together
    unique_strata = np.unique(strata)
    # Pre-compute stratum member indices
    stratum_indices: dict[int, np.ndarray] = {}
    for s in unique_strata:
        stratum_indices[int(s)] = np.where(strata == s)[0]

    all_boot_idx = np.empty((n_bootstrap, n_respondents), dtype=np.intp)

    for i in range(n_bootstrap):
        pieces: list[np.ndarray] = []
        for s in unique_strata:
            members = stratum_indices[int(s)]
            n_s = len(members)
            resampled = members[rng.integers(0, n_s, size=n_s)]
            pieces.append(resampled)
        combined = np.concatenate(pieces)
        # Shuffle to avoid ordering artifacts
        rng.shuffle(combined)
        all_boot_idx[i] = combined

    return all_boot_idx


def paired_bootstrap_cis(
    metric_fn: Callable[..., dict[str, float]],
    *arrays: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
    ci_levels: tuple[float, float] = (2.5, 97.5),
) -> dict[str, dict[str, float]]:
    """Compute paired percentile bootstrap 95% CIs for arbitrary metrics.

    Resamples by respondent (same indices across all provided arrays) and
    computes the metric function on each bootstrap replicate.

    Args:
        metric_fn: Function that takes the same number of arrays as *arrays
            and returns a dict of metric_name -> float.
        *arrays: Arrays to resample. All must have the same first dimension
            (n_respondents). Resampling uses the same indices for all arrays.
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed.
        ci_levels: Percentile levels for the CI (default: 2.5th and 97.5th).

    Returns:
        Dict of metric_name -> {"lower": float, "upper": float}.
    """
    rng = np.random.default_rng(seed)
    n_respondents = arrays[0].shape[0]

    all_idx = _generate_bootstrap_indices(n_respondents, n_bootstrap, rng)

    # First call to discover metric keys
    point_estimate = metric_fn(*arrays)
    metric_keys = list(point_estimate.keys())
    boot_values: dict[str, list[float]] = {k: [] for k in metric_keys}

    for i in range(n_bootstrap):
        idx = all_idx[i]
        resampled = tuple(a[idx] for a in arrays)
        boot_metrics = metric_fn(*resampled)
        for k in metric_keys:
            boot_values[k].append(boot_metrics.get(k, float("nan")))

    cis: dict[str, dict[str, float]] = {}
    for k in metric_keys:
        arr = np.array(boot_values[k])
        valid = arr[~np.isnan(arr)]
        if len(valid) > 0:
            cis[k] = {
                "lower": float(np.percentile(valid, ci_levels[0])),
                "upper": float(np.percentile(valid, ci_levels[1])),
            }
        else:
            cis[k] = {"lower": float("nan"), "upper": float("nan")}

    return cis


def stratified_paired_bootstrap_cis(
    metric_fn: Callable[..., dict[str, float]],
    *arrays: np.ndarray,
    strata: np.ndarray | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
    ci_levels: tuple[float, float] = (2.5, 97.5),
) -> dict[str, dict[str, float]]:
    """Compute stratified paired percentile bootstrap 95% CIs.

    Same as paired_bootstrap_cis but resamples within each stratum separately.
    Falls back to plain paired bootstrap if strata is None.

    Args:
        metric_fn: Function that takes the same number of arrays as *arrays
            and returns a dict of metric_name -> float.
        *arrays: Arrays to resample. All must have the same first dimension.
        strata: Array of stratum labels (e.g., split_stratum column).
            If None, falls back to plain paired bootstrap.
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed.
        ci_levels: Percentile levels for the CI.

    Returns:
        Dict of metric_name -> {"lower": float, "upper": float}.
    """
    if strata is None:
        return paired_bootstrap_cis(
            metric_fn, *arrays,
            n_bootstrap=n_bootstrap, seed=seed, ci_levels=ci_levels,
        )

    rng = np.random.default_rng(seed)
    n_respondents = arrays[0].shape[0]

    all_idx = _generate_bootstrap_indices(n_respondents, n_bootstrap, rng, strata=strata)

    # Discover metric keys
    point_estimate = metric_fn(*arrays)
    metric_keys = list(point_estimate.keys())
    boot_values: dict[str, list[float]] = {k: [] for k in metric_keys}

    for i in range(n_bootstrap):
        idx = all_idx[i]
        resampled = tuple(a[idx] for a in arrays)
        boot_metrics = metric_fn(*resampled)
        for k in metric_keys:
            boot_values[k].append(boot_metrics.get(k, float("nan")))

    cis: dict[str, dict[str, float]] = {}
    for k in metric_keys:
        arr = np.array(boot_values[k])
        valid = arr[~np.isnan(arr)]
        if len(valid) > 0:
            cis[k] = {
                "lower": float(np.percentile(valid, ci_levels[0])),
                "upper": float(np.percentile(valid, ci_levels[1])),
            }
        else:
            cis[k] = {"lower": float("nan"), "upper": float("nan")}

    return cis


def bootstrap_metric_deltas(
    metric_fn: Callable[..., dict[str, float]],
    reference_arrays: tuple[np.ndarray, ...],
    comparison_arrays: tuple[np.ndarray, ...],
    n_bootstrap: int = 1000,
    seed: int = 42,
    strata: np.ndarray | None = None,
    ci_levels: tuple[float, float] = (2.5, 97.5),
) -> dict[str, dict[str, float]]:
    """Compute bootstrap CIs for metric deltas between two prediction sets.

    For each bootstrap resample, computes metrics for both reference and
    comparison using the SAME respondent indices, then takes the difference
    (comparison - reference). Returns CIs on the deltas.

    This is used for the ablation comparison protocol: comparing a new model
    against a locked reference model on the same test data.

    Args:
        metric_fn: Function that takes arrays and returns dict of metrics.
            Must accept the same number of arrays as in reference_arrays /
            comparison_arrays.
        reference_arrays: Tuple of arrays for the reference (locked) model.
        comparison_arrays: Tuple of arrays for the comparison model.
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed.
        strata: Optional stratum labels for stratified resampling.
        ci_levels: Percentile levels for the CI.

    Returns:
        Dict with keys:
        - "point_deltas": dict of metric_name -> float (comparison - reference)
        - "delta_cis": dict of metric_name -> {"lower": float, "upper": float}
    """
    rng = np.random.default_rng(seed)
    n_respondents = reference_arrays[0].shape[0]

    all_idx = _generate_bootstrap_indices(
        n_respondents, n_bootstrap, rng, strata=strata,
    )

    # Point estimates
    ref_point = metric_fn(*reference_arrays)
    cmp_point = metric_fn(*comparison_arrays)
    metric_keys = list(ref_point.keys())

    point_deltas: dict[str, float] = {}
    for k in metric_keys:
        point_deltas[k] = cmp_point.get(k, float("nan")) - ref_point.get(k, float("nan"))

    boot_deltas: dict[str, list[float]] = {k: [] for k in metric_keys}

    for i in range(n_bootstrap):
        idx = all_idx[i]
        ref_resampled = tuple(a[idx] for a in reference_arrays)
        cmp_resampled = tuple(a[idx] for a in comparison_arrays)
        ref_metrics = metric_fn(*ref_resampled)
        cmp_metrics = metric_fn(*cmp_resampled)
        for k in metric_keys:
            ref_val = ref_metrics.get(k, float("nan"))
            cmp_val = cmp_metrics.get(k, float("nan"))
            boot_deltas[k].append(cmp_val - ref_val)

    delta_cis: dict[str, dict[str, float]] = {}
    for k in metric_keys:
        arr = np.array(boot_deltas[k])
        valid = arr[~np.isnan(arr)]
        if len(valid) > 0:
            delta_cis[k] = {
                "lower": float(np.percentile(valid, ci_levels[0])),
                "upper": float(np.percentile(valid, ci_levels[1])),
            }
        else:
            delta_cis[k] = {"lower": float("nan"), "upper": float("nan")}

    return {"point_deltas": point_deltas, "delta_cis": delta_cis}


def respondent_bootstrap_multi_domain(
    per_domain_data: dict[str, dict[str, np.ndarray]],
    domains: list[str],
    percentile_metric_fn: Callable[
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray], dict[str, float]
    ],
    raw_metric_fn: Callable[[np.ndarray, np.ndarray], dict[str, float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
    strata: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run respondent-level bootstrap across multiple domains simultaneously.

    Resamples respondents (preserving cross-domain dependence) and computes
    both percentile-scale and raw-scale overall metrics, plus per-domain
    Pearson r bootstrap distributions.

    Args:
        per_domain_data: Dict of domain -> {"true", "pred", "lower", "upper",
            "raw_true", "raw_pred"} arrays.
        domains: Ordered list of domain keys to include.
        percentile_metric_fn: Computes metrics from (true, pred, lower, upper).
        raw_metric_fn: Computes metrics from (raw_true, raw_pred).
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed.
        strata: Optional stratum labels for stratified resampling.

    Returns:
        Dict with:
        - "overall_cis": dict of metric_name -> {"lower", "upper"}
        - "per_domain_cis": dict of domain -> {metric -> {"lower", "upper"}}
    """
    rng = np.random.default_rng(seed)

    valid_domains = [d for d in domains if d in per_domain_data]
    if not valid_domains:
        return {"overall_cis": {}, "per_domain_cis": {}}

    n_respondents = len(per_domain_data[valid_domains[0]]["true"])

    all_idx = _generate_bootstrap_indices(
        n_respondents, n_bootstrap, rng, strata=strata,
    )

    # Discover metric keys from point estimates
    # Flatten for overall
    flat_true = np.concatenate([per_domain_data[d]["true"] for d in valid_domains])
    flat_pred = np.concatenate([per_domain_data[d]["pred"] for d in valid_domains])
    flat_lower = np.concatenate([per_domain_data[d]["lower"] for d in valid_domains])
    flat_upper = np.concatenate([per_domain_data[d]["upper"] for d in valid_domains])
    pct_keys = list(percentile_metric_fn(flat_true, flat_pred, flat_lower, flat_upper).keys())

    # Raw domains
    raw_valid_domains = [
        d for d in valid_domains
        if "raw_true" in per_domain_data[d] and "raw_pred" in per_domain_data[d]
        and not np.all(np.isnan(per_domain_data[d]["raw_true"]))
    ]
    raw_keys: list[str] = []
    if raw_valid_domains:
        flat_raw_true = np.concatenate([per_domain_data[d]["raw_true"] for d in raw_valid_domains])
        flat_raw_pred = np.concatenate([per_domain_data[d]["raw_pred"] for d in raw_valid_domains])
        raw_keys = list(raw_metric_fn(flat_raw_true, flat_raw_pred).keys())

    all_keys = pct_keys + raw_keys
    boot_overall: dict[str, list[float]] = {k: [] for k in all_keys}

    # Per-domain bootstrap: only Pearson r (percentile and raw)
    per_domain_boot_r: dict[str, list[float]] = {d: [] for d in valid_domains}
    per_domain_boot_raw_r: dict[str, list[float]] = {d: [] for d in raw_valid_domains}

    for i in range(n_bootstrap):
        idx = all_idx[i]

        # Overall percentile metrics
        boot_true = np.concatenate([per_domain_data[d]["true"][idx] for d in valid_domains])
        boot_pred = np.concatenate([per_domain_data[d]["pred"][idx] for d in valid_domains])
        boot_lower = np.concatenate([per_domain_data[d]["lower"][idx] for d in valid_domains])
        boot_upper = np.concatenate([per_domain_data[d]["upper"][idx] for d in valid_domains])
        m = percentile_metric_fn(boot_true, boot_pred, boot_lower, boot_upper)
        for k in pct_keys:
            boot_overall[k].append(m[k])

        # Overall raw metrics
        if raw_valid_domains:
            boot_raw_true = np.concatenate([per_domain_data[d]["raw_true"][idx] for d in raw_valid_domains])
            boot_raw_pred = np.concatenate([per_domain_data[d]["raw_pred"][idx] for d in raw_valid_domains])
            rm = raw_metric_fn(boot_raw_true, boot_raw_pred)
            for k in raw_keys:
                boot_overall[k].append(rm[k])

        # Per-domain Pearson r
        for d in valid_domains:
            dt = per_domain_data[d]["true"][idx]
            dp = per_domain_data[d]["pred"][idx]
            if np.std(dt) > 0 and np.std(dp) > 0:
                r_val, _ = stats.pearsonr(dt, dp)
                per_domain_boot_r[d].append(float(r_val))
            else:
                per_domain_boot_r[d].append(float("nan"))

        for d in raw_valid_domains:
            rt = per_domain_data[d]["raw_true"][idx]
            rp = per_domain_data[d]["raw_pred"][idx]
            if np.std(rt) > 0 and np.std(rp) > 0:
                r_val, _ = stats.pearsonr(rt, rp)
                per_domain_boot_raw_r[d].append(float(r_val))
            else:
                per_domain_boot_raw_r[d].append(float("nan"))

    # Compute CIs
    overall_cis: dict[str, dict[str, float]] = {}
    for k in all_keys:
        arr = np.array(boot_overall[k])
        valid = arr[~np.isnan(arr)]
        if len(valid) > 0:
            overall_cis[k] = {
                "lower": float(np.percentile(valid, 2.5)),
                "upper": float(np.percentile(valid, 97.5)),
            }
        else:
            overall_cis[k] = {"lower": float("nan"), "upper": float("nan")}

    per_domain_cis: dict[str, dict[str, dict[str, float]]] = {}
    for d in valid_domains:
        per_domain_cis[d] = {}
        arr = np.array(per_domain_boot_r[d])
        valid = arr[~np.isnan(arr)]
        if len(valid) > 0:
            per_domain_cis[d]["pearson_r"] = {
                "lower": float(np.percentile(valid, 2.5)),
                "upper": float(np.percentile(valid, 97.5)),
            }
        else:
            per_domain_cis[d]["pearson_r"] = {"lower": float("nan"), "upper": float("nan")}

        if d in raw_valid_domains:
            arr_raw = np.array(per_domain_boot_raw_r[d])
            valid_raw = arr_raw[~np.isnan(arr_raw)]
            if len(valid_raw) > 0:
                per_domain_cis[d]["raw_pearson_r"] = {
                    "lower": float(np.percentile(valid_raw, 2.5)),
                    "upper": float(np.percentile(valid_raw, 97.5)),
                }
            else:
                per_domain_cis[d]["raw_pearson_r"] = {"lower": float("nan"), "upper": float("nan")}

    return {"overall_cis": overall_cis, "per_domain_cis": per_domain_cis}
