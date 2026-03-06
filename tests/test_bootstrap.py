"""Tests for lib/bootstrap.py — confidence interval functions."""

import numpy as np
import pytest
from scipy import stats

from lib.bootstrap import (
    _generate_bootstrap_indices,
    bootstrap_metric_deltas,
    paired_bootstrap_cis,
    respondent_bootstrap_multi_domain,
    stratified_paired_bootstrap_cis,
    vectorized_pearsonr_bootstrap,
)


def _metric_fn(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Simple metric function returning Pearson r and MAE."""
    r, _ = stats.pearsonr(y_true, y_pred)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"pearson_r": float(r), "mae": mae}


def _coverage_metric_fn(
    y_true: np.ndarray, y_pred: np.ndarray,
    lower: np.ndarray, upper: np.ndarray,
) -> dict[str, float]:
    """Metric function that uses all 4 args including lower/upper."""
    coverage = float(np.mean((lower <= y_true) & (y_true <= upper)))
    r, _ = stats.pearsonr(y_true, y_pred)
    return {"coverage": coverage, "pearson_r": float(r)}


def _pct_fn(t, p, lo, hi):
    """Percentile-scale metric function for multi-domain tests."""
    r, _ = stats.pearsonr(t, p)
    return {"pearson_r": float(r)}


def _raw_fn(t, p):
    """Raw-scale metric function for multi-domain tests."""
    r, _ = stats.pearsonr(t, p)
    return {"raw_pearson_r": float(r)}


@pytest.fixture
def correlated_arrays():
    """Pair of correlated arrays (n=50) for bootstrap tests."""
    rng = np.random.default_rng(0)
    y_true = rng.normal(50, 10, size=50)
    y_pred = y_true + rng.normal(0, 3, size=50)
    return y_true, y_pred


@pytest.fixture
def heterogeneous_strata_data():
    """Data with two strata: high-correlation and low-correlation."""
    rng = np.random.default_rng(0)
    # Stratum 0: high correlation
    y_true_hi = rng.normal(50, 10, size=50)
    y_pred_hi = y_true_hi + rng.normal(0, 1, size=50)
    # Stratum 1: low correlation
    y_true_lo = rng.normal(50, 10, size=50)
    y_pred_lo = y_true_lo + rng.normal(0, 20, size=50)

    y_true = np.concatenate([y_true_hi, y_true_lo])
    y_pred = np.concatenate([y_pred_hi, y_pred_lo])
    strata = np.array([0] * 50 + [1] * 50)
    return y_true, y_pred, strata


class TestVectorizedPearsonrBootstrap:
    def test_output_shape(self, correlated_arrays):
        y_true, y_pred = correlated_arrays
        result = vectorized_pearsonr_bootstrap(
            y_true, y_pred, n_bootstrap=500, rng=np.random.default_rng(0),
        )
        assert result.shape == (500,)

    def test_values_in_valid_range(self, correlated_arrays):
        y_true, y_pred = correlated_arrays
        result = vectorized_pearsonr_bootstrap(
            y_true, y_pred, n_bootstrap=1000, rng=np.random.default_rng(0),
        )
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_reproducible_with_seed(self, correlated_arrays):
        y_true, y_pred = correlated_arrays
        a = vectorized_pearsonr_bootstrap(
            y_true, y_pred, n_bootstrap=500, rng=np.random.default_rng(42),
        )
        b = vectorized_pearsonr_bootstrap(
            y_true, y_pred, n_bootstrap=500, rng=np.random.default_rng(42),
        )
        np.testing.assert_array_equal(a, b)

    def test_perfect_correlation(self):
        """Perfect correlation input should produce bootstrap r values near 1.0."""
        y = np.linspace(0, 100, 50)
        result = vectorized_pearsonr_bootstrap(
            y, y, n_bootstrap=500, rng=np.random.default_rng(0),
        )
        valid = result[~np.isnan(result)]
        assert np.all(valid > 0.99), f"Min bootstrap r = {valid.min():.4f}"


class TestGenerateBootstrapIndices:
    def test_output_shape(self):
        indices = _generate_bootstrap_indices(
            n_respondents=100, n_bootstrap=500,
            rng=np.random.default_rng(0),
        )
        assert indices.shape == (500, 100)

    def test_indices_in_range(self):
        n_resp = 80
        indices = _generate_bootstrap_indices(
            n_respondents=n_resp, n_bootstrap=200,
            rng=np.random.default_rng(0),
        )
        assert np.all(indices >= 0)
        assert np.all(indices < n_resp)

    def test_stratified_preserves_stratum_sizes(self):
        """Each bootstrap replicate should have the same stratum composition."""
        strata = np.array([0] * 30 + [1] * 20 + [2] * 10)
        n_resp = len(strata)
        indices = _generate_bootstrap_indices(
            n_respondents=n_resp, n_bootstrap=100,
            rng=np.random.default_rng(0),
            strata=strata,
        )
        assert indices.shape == (100, n_resp)

        for i in range(100):
            idx = indices[i]
            boot_strata = strata[idx]
            counts = np.bincount(boot_strata, minlength=3)
            assert counts[0] == 30, f"Replicate {i}: stratum 0 count = {counts[0]}, expected 30"
            assert counts[1] == 20, f"Replicate {i}: stratum 1 count = {counts[1]}, expected 20"
            assert counts[2] == 10, f"Replicate {i}: stratum 2 count = {counts[2]}, expected 10"


class TestPairedBootstrapCIs:
    def test_ci_bounds_valid(self, correlated_arrays):
        """CI lower should be <= upper, and width should be reasonable for Pearson r."""
        y_true, y_pred = correlated_arrays
        cis = paired_bootstrap_cis(
            _metric_fn, y_true, y_pred,
            n_bootstrap=1000, seed=42,
        )
        for key, ci in cis.items():
            assert ci["lower"] <= ci["upper"], f"{key}: lower > upper"
            width = ci["upper"] - ci["lower"]
            assert width > 0, f"{key}: CI width is 0"
            if "pearson_r" in key:
                assert width < 1.0, f"{key}: CI width {width} unreasonably large"

    def test_lower_le_upper(self, correlated_arrays):
        y_true, y_pred = correlated_arrays
        cis = paired_bootstrap_cis(
            _metric_fn, y_true, y_pred,
            n_bootstrap=1000, seed=42,
        )
        for key, ci in cis.items():
            assert ci["lower"] <= ci["upper"], f"{key}: lower > upper"

    def test_reproducible_with_seed(self, correlated_arrays):
        y_true, y_pred = correlated_arrays
        a = paired_bootstrap_cis(_metric_fn, y_true, y_pred, n_bootstrap=1000, seed=7)
        b = paired_bootstrap_cis(_metric_fn, y_true, y_pred, n_bootstrap=1000, seed=7)
        for key in a:
            assert a[key]["lower"] == b[key]["lower"]
            assert a[key]["upper"] == b[key]["upper"]

    def test_returns_all_metric_keys(self, correlated_arrays):
        y_true, y_pred = correlated_arrays
        cis = paired_bootstrap_cis(
            _metric_fn, y_true, y_pred, n_bootstrap=1000, seed=42,
        )
        assert "pearson_r" in cis
        assert "mae" in cis

    def test_coverage_metric_fn_with_four_args(self):
        """Metric function using all 4 args (including lower/upper) should work."""
        rng = np.random.default_rng(0)
        y_true = rng.normal(50, 10, size=100)
        y_pred = y_true + rng.normal(0, 3, size=100)
        lower = y_pred - 8
        upper = y_pred + 8

        cis = paired_bootstrap_cis(
            _coverage_metric_fn, y_true, y_pred, lower, upper,
            n_bootstrap=1000, seed=42,
        )
        assert "coverage" in cis
        assert "pearson_r" in cis
        assert cis["coverage"]["lower"] <= cis["coverage"]["upper"]
        # Coverage should be reasonably high (most true values within +/-8)
        assert cis["coverage"]["lower"] > 0.3


class TestStratifiedPairedBootstrapCIs:
    def test_fallback_to_non_stratified(self, correlated_arrays):
        """strata=None should produce exact same result as paired_bootstrap_cis."""
        y_true, y_pred = correlated_arrays
        plain = paired_bootstrap_cis(
            _metric_fn, y_true, y_pred, n_bootstrap=1000, seed=42,
        )
        strat = stratified_paired_bootstrap_cis(
            _metric_fn, y_true, y_pred, strata=None, n_bootstrap=1000, seed=42,
        )
        for key in plain:
            assert plain[key]["lower"] == strat[key]["lower"]
            assert plain[key]["upper"] == strat[key]["upper"]

    def test_valid_cis_with_strata(self, correlated_arrays):
        y_true, y_pred = correlated_arrays
        strata = np.array([0] * 25 + [1] * 25)
        cis = stratified_paired_bootstrap_cis(
            _metric_fn, y_true, y_pred,
            strata=strata, n_bootstrap=1000, seed=42,
        )
        for key, ci in cis.items():
            assert ci["lower"] <= ci["upper"], f"{key}: lower > upper"
            assert not np.isnan(ci["lower"])
            assert not np.isnan(ci["upper"])

    def test_stratified_narrows_pearson_r_ci(self, heterogeneous_strata_data):
        """Stratified bootstrap should produce narrower CIs for Pearson r
        when strata are heterogeneous (high-corr vs low-corr strata)."""
        y_true, y_pred, strata = heterogeneous_strata_data
        plain = paired_bootstrap_cis(
            _metric_fn, y_true, y_pred, n_bootstrap=2000, seed=42,
        )
        strat = stratified_paired_bootstrap_cis(
            _metric_fn, y_true, y_pred, strata=strata, n_bootstrap=2000, seed=42,
        )
        # Stratification should narrow the Pearson r CI width
        plain_width = plain["pearson_r"]["upper"] - plain["pearson_r"]["lower"]
        strat_width = strat["pearson_r"]["upper"] - strat["pearson_r"]["lower"]
        assert strat_width <= plain_width, (
            f"Stratified CI width ({strat_width:.4f}) should be <= "
            f"non-stratified ({plain_width:.4f}) for heterogeneous strata"
        )


class TestBootstrapMetricDeltas:
    def test_returns_expected_keys(self, correlated_arrays):
        y_true, y_pred = correlated_arrays
        result = bootstrap_metric_deltas(
            _metric_fn,
            reference_arrays=(y_true, y_pred),
            comparison_arrays=(y_true, y_pred),
            n_bootstrap=1000, seed=42,
        )
        assert "point_deltas" in result
        assert "delta_cis" in result

    def test_point_deltas_match_manual(self, correlated_arrays):
        y_true, y_pred = correlated_arrays
        rng = np.random.default_rng(99)
        y_pred2 = y_true + rng.normal(0, 5, size=50)

        result = bootstrap_metric_deltas(
            _metric_fn,
            reference_arrays=(y_true, y_pred),
            comparison_arrays=(y_true, y_pred2),
            n_bootstrap=1000, seed=42,
        )
        ref_metrics = _metric_fn(y_true, y_pred)
        cmp_metrics = _metric_fn(y_true, y_pred2)
        for key in ref_metrics:
            expected_delta = cmp_metrics[key] - ref_metrics[key]
            np.testing.assert_allclose(
                result["point_deltas"][key], expected_delta, rtol=1e-10,
                err_msg=f"{key} point delta mismatch",
            )
            # The CI should contain the point delta
            ci = result["delta_cis"][key]
            point_delta = result["point_deltas"][key]
            assert ci["lower"] <= point_delta <= ci["upper"], (
                f"{key}: point delta {point_delta:.4f} not in "
                f"CI [{ci['lower']:.4f}, {ci['upper']:.4f}]"
            )

    def test_identical_predictions_zero_deltas(self, correlated_arrays):
        y_true, y_pred = correlated_arrays
        result = bootstrap_metric_deltas(
            _metric_fn,
            reference_arrays=(y_true, y_pred),
            comparison_arrays=(y_true, y_pred),
            n_bootstrap=1000, seed=42,
        )
        for key, delta in result["point_deltas"].items():
            assert abs(delta) < 1e-10, f"{key}: expected 0 delta, got {delta}"
        for key, ci in result["delta_cis"].items():
            assert ci["lower"] <= 0.0 <= ci["upper"], (
                f"{key}: 0 not in delta CI [{ci['lower']}, {ci['upper']}]"
            )
            # CI width should be ~0 since both models are identical
            assert abs(ci["upper"] - ci["lower"]) < 1e-10, (
                f"{key}: CI width should be ~0, got {ci['upper'] - ci['lower']}"
            )

    def test_with_strata(self, heterogeneous_strata_data):
        """bootstrap_metric_deltas should work with strata parameter."""
        y_true, y_pred, strata = heterogeneous_strata_data
        rng = np.random.default_rng(99)
        y_pred2 = y_true + rng.normal(0, 5, size=len(y_true))

        result = bootstrap_metric_deltas(
            _metric_fn,
            reference_arrays=(y_true, y_pred),
            comparison_arrays=(y_true, y_pred2),
            strata=strata,
            n_bootstrap=1000, seed=42,
        )
        assert "point_deltas" in result
        assert "delta_cis" in result
        for key in result["point_deltas"]:
            assert key in result["delta_cis"]
            ci = result["delta_cis"][key]
            assert ci["lower"] <= ci["upper"]


class TestRespondentBootstrapMultiDomain:
    @pytest.fixture
    def multi_domain_data(self):
        """Build minimal per_domain_data for 2 domains, n=50."""
        rng = np.random.default_rng(0)
        data = {}
        for d in ["ext", "agr"]:
            true = rng.normal(50, 10, size=50)
            pred = true + rng.normal(0, 3, size=50)
            lower = pred - 5
            upper = pred + 5
            raw_true = rng.normal(3, 0.5, size=50)
            raw_pred = raw_true + rng.normal(0, 0.2, size=50)
            data[d] = {
                "true": true, "pred": pred,
                "lower": lower, "upper": upper,
                "raw_true": raw_true, "raw_pred": raw_pred,
            }
        return data

    def test_returns_expected_structure(self, multi_domain_data):
        result = respondent_bootstrap_multi_domain(
            multi_domain_data, ["ext", "agr"],
            percentile_metric_fn=_pct_fn,
            raw_metric_fn=_raw_fn,
            n_bootstrap=1000, seed=42,
        )
        assert "overall_cis" in result
        assert "per_domain_cis" in result

    def test_per_domain_cis_lower_le_upper(self, multi_domain_data):
        result = respondent_bootstrap_multi_domain(
            multi_domain_data, ["ext", "agr"],
            percentile_metric_fn=_pct_fn,
            raw_metric_fn=_raw_fn,
            n_bootstrap=1000, seed=42,
        )
        for domain, metrics in result["per_domain_cis"].items():
            for metric_name, ci in metrics.items():
                assert ci["lower"] <= ci["upper"], (
                    f"{domain}/{metric_name}: lower > upper"
                )

    def test_overall_cis_present(self, multi_domain_data):
        result = respondent_bootstrap_multi_domain(
            multi_domain_data, ["ext", "agr"],
            percentile_metric_fn=_pct_fn,
            raw_metric_fn=_raw_fn,
            n_bootstrap=1000, seed=42,
        )
        assert len(result["overall_cis"]) > 0
        for metric_name, ci in result["overall_cis"].items():
            assert ci["lower"] <= ci["upper"]

    def test_with_strata(self, multi_domain_data):
        """respondent_bootstrap_multi_domain should work with strata parameter."""
        strata = np.array([0] * 25 + [1] * 25)
        result = respondent_bootstrap_multi_domain(
            multi_domain_data, ["ext", "agr"],
            percentile_metric_fn=_pct_fn,
            raw_metric_fn=_raw_fn,
            strata=strata,
            n_bootstrap=1000, seed=42,
        )
        assert "overall_cis" in result
        assert "per_domain_cis" in result
        for domain in ["ext", "agr"]:
            assert domain in result["per_domain_cis"]
            for metric_name, ci in result["per_domain_cis"][domain].items():
                assert ci["lower"] <= ci["upper"]


class TestBootstrapNanHandling:
    def test_nan_replicates_handled_gracefully(self):
        """Bootstrap should filter NaN replicates and produce a valid tight CI.

        With y_true = y_pred = [1,1,1,1,2], resamples that include the '2' value
        produce r=1.0; constant resamples return NaN. The implementation filters
        NaN replicates and computes CIs on the valid ones (all r ≈ 1.0).
        """
        y_true = np.array([1.0, 1.0, 1.0, 1.0, 2.0])
        y_pred = np.array([1.0, 1.0, 1.0, 1.0, 2.0])

        def metric_fn_safe(yt, yp):
            if np.std(yt) < 1e-10 or np.std(yp) < 1e-10:
                return {"pearson_r": float("nan")}
            r, _ = stats.pearsonr(yt, yp)
            return {"pearson_r": float(r)}

        cis = paired_bootstrap_cis(
            metric_fn_safe, y_true, y_pred,
            n_bootstrap=1000, seed=42,
        )
        assert "pearson_r" in cis
        ci = cis["pearson_r"]
        # CI should be valid (not NaN) since many resamples include the '2'
        assert not np.isnan(ci["lower"]), "CI lower should not be NaN"
        assert not np.isnan(ci["upper"]), "CI upper should not be NaN"
        assert ci["lower"] <= ci["upper"], "CI lower should be <= upper"
        # All valid replicates produce r ≈ 1.0, so CI should be tight
        ci_width = ci["upper"] - ci["lower"]
        assert ci_width < 0.5, f"CI width should be tight, got {ci_width:.4f}"
