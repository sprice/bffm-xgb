"""Tests for lib/scoring.py — raw_score_to_percentile conversion."""

import numpy as np
import pytest

from lib.constants import DOMAINS
from lib.norms import load_norms
from lib.scoring import raw_score_to_percentile

try:
    NORMS = load_norms()
except FileNotFoundError:
    NORMS = None

pytestmark = pytest.mark.skipif(NORMS is None, reason="Norms file not generated (run make norms)")


class TestScalarConversion:
    def test_mean_score_near_50th_percentile(self):
        """Mean raw score should map to approximately the 50th percentile."""
        for domain in DOMAINS:
            mean = NORMS[domain]["mean"]
            pct = raw_score_to_percentile(mean, domain)
            assert abs(pct - 50.0) < 0.5, f"{domain}: mean -> {pct}, expected ~50"

    def test_low_score_near_zero(self):
        """Score of 1.0 should produce a low percentile (actual values are ~0-1.5%)."""
        for domain in DOMAINS:
            pct = raw_score_to_percentile(1.0, domain)
            assert pct < 5.0, f"{domain}: score 1.0 -> {pct}, expected < 5"

    def test_high_score_near_100(self):
        """Score of 5.0 should produce a high percentile (actual values are ~95-99%)."""
        for domain in DOMAINS:
            pct = raw_score_to_percentile(5.0, domain)
            assert pct > 95.0, f"{domain}: score 5.0 -> {pct}, expected > 95"

    def test_output_clipped_to_range(self):
        """Output stays in [0, 100] even for extreme inputs."""
        for domain in DOMAINS:
            assert raw_score_to_percentile(-10.0, domain) >= 0.0
            assert raw_score_to_percentile(100.0, domain) <= 100.0


class TestMonotonicity:
    def test_monotonically_increasing(self):
        """Sorted inputs should produce sorted (non-decreasing) percentiles."""
        scores = np.linspace(1.0, 5.0, 20)
        for domain in DOMAINS:
            percentiles = raw_score_to_percentile(scores, domain)
            assert np.all(np.diff(percentiles) >= 0), (
                f"{domain}: percentiles not monotonically increasing"
            )


class TestVectorized:
    def test_array_input_matches_scalar(self):
        """Vectorized array input should match element-wise scalar calls."""
        scores = np.array([1.5, 2.5, 3.5, 4.5])
        for domain in DOMAINS:
            vec_result = raw_score_to_percentile(scores, domain)
            for i, s in enumerate(scores):
                scalar_result = raw_score_to_percentile(s, domain)
                np.testing.assert_allclose(
                    vec_result[i], scalar_result, rtol=1e-10,
                    err_msg=f"{domain} mismatch at score {s}",
                )


class TestEdgeCases:
    def test_nan_propagates(self):
        """NaN input should produce NaN output."""
        pct = raw_score_to_percentile(np.nan, "ext")
        assert np.isnan(pct)

    def test_nan_in_mixed_array(self):
        """NaN should propagate per-element without corrupting neighbors."""
        scores = np.array([3.0, np.nan, 4.0])
        result = raw_score_to_percentile(scores, "ext")
        assert not np.isnan(result[0])
        assert np.isnan(result[1])
        assert not np.isnan(result[2])

    def test_invalid_domain_raises_key_error(self):
        with pytest.raises(KeyError):
            raw_score_to_percentile(3.0, "invalid_domain")

    def test_known_z_score_anchor(self):
        """mean + 1*sd should map to approximately the 84.13th percentile."""
        for domain in DOMAINS:
            norm = NORMS[domain]
            score = norm["mean"] + norm["sd"]
            pct = raw_score_to_percentile(score, domain)
            assert abs(pct - 84.13) < 0.5, (
                f"{domain}: mean+sd -> {pct}, expected ~84.13"
            )
