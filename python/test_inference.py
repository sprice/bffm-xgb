"""Tests for IPIP-BFFM inference module.

Runs against the committed tiny fixture bundle (tests/fixtures/golden) by
default, so these real-inference tests execute in CI without the gitignored
137 MB reference model. Override with the BFFM_FIXTURE_DIR env var.
"""

import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

FIXTURE_DIR = Path(
    os.environ.get("BFFM_FIXTURE_DIR")
    or Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "golden"
)
_CONFIG = FIXTURE_DIR / "config.json"

# REQUIRE_ARTIFACTS=1 (set in CI) turns a missing fixture into a hard error so
# these tests can never silently vanish into a green skip.
if os.environ.get("REQUIRE_ARTIFACTS") == "1" and not _CONFIG.exists():
    raise RuntimeError(
        f"REQUIRE_ARTIFACTS=1 but the golden fixture config is missing at {_CONFIG}"
    )

pytestmark = pytest.mark.skipif(
    not _CONFIG.exists(),
    reason=f"golden fixture config.json not found at {_CONFIG} (run `make fixtures`)",
)

from inference import IPIPBFFMPredictor

# ── Test vectors ─────────────────────────────────────────────────────────

DOMAINS = ["ext", "agr", "csn", "est", "opn"]
QUANTILES = ["q05", "q50", "q95"]

# Feature names in order
FEATURE_NAMES = [f"{d}{i}" for d in DOMAINS for i in range(1, 11)]

# Input A: Full 50-item response, repeating 1-5 pattern
INPUT_A_VALUES = [float((i % 5) + 1) for i in range(50)]

# Input B: the canonical deployed balanced-20 set (top-4 per domain), loaded
# from the committed fixture so it cannot drift from the pipeline's selection.
if _CONFIG.exists():
    _CANONICAL_20 = json.loads(
        (FIXTURE_DIR / "canonical_items.json").read_text()
    )["domain_balanced_20"]
else:
    _CANONICAL_20 = []
INPUT_B_ITEMS = {
    iid: float((i % 5) + 1) for i, iid in enumerate(_CANONICAL_20)
}


@pytest.fixture(scope="module")
def predictor():
    return IPIPBFFMPredictor(model_dir=FIXTURE_DIR)


class TestDictInput:
    def test_dict_matches_array(self, predictor):
        """predict() with item dict produces same results as predict_array()."""
        arr = np.full((1, 50), np.nan, dtype=np.float32)
        for item_id, val in INPUT_B_ITEMS.items():
            idx = FEATURE_NAMES.index(item_id)
            arr[0, idx] = val

        result_array = predictor.predict_array(arr)
        result_dict = predictor.predict(INPUT_B_ITEMS)

        for domain in DOMAINS:
            for q in QUANTILES:
                assert result_array[domain]["raw"][q] == result_dict[domain]["raw"][q], (
                    f"{domain}_{q} raw mismatch"
                )
                assert result_array[domain]["percentile"][q] == result_dict[domain]["percentile"][q], (
                    f"{domain}_{q} percentile mismatch"
                )


class TestQuantileOrdering:
    def test_percentile_ordering_full(self, predictor):
        """q05 <= q50 <= q95 in percentile space for full response."""
        arr = np.array(INPUT_A_VALUES, dtype=np.float32).reshape(1, 50)
        result = predictor.predict_array(arr)

        for domain in DOMAINS:
            p05 = result[domain]["percentile"]["q05"]
            p50 = result[domain]["percentile"]["q50"]
            p95 = result[domain]["percentile"]["q95"]
            assert p05 <= p50 <= p95, (
                f"{domain}: {p05} <= {p50} <= {p95} violated"
            )

    def test_percentile_ordering_sparse(self, predictor):
        """q05 <= q50 <= q95 in percentile space for sparse response."""
        result = predictor.predict(INPUT_B_ITEMS)

        for domain in DOMAINS:
            p05 = result[domain]["percentile"]["q05"]
            p50 = result[domain]["percentile"]["q50"]
            p95 = result[domain]["percentile"]["q95"]
            assert p05 <= p50 <= p95, (
                f"{domain}: {p05} <= {p50} <= {p95} violated"
            )


class TestPercentileRange:
    def test_percentiles_in_range_full(self, predictor):
        """All percentiles in [0, 100] for full response."""
        arr = np.array(INPUT_A_VALUES, dtype=np.float32).reshape(1, 50)
        result = predictor.predict_array(arr)

        for domain in DOMAINS:
            for q in QUANTILES:
                pct = result[domain]["percentile"][q]
                assert 0 <= pct <= 100, (
                    f"{domain}_{q}: percentile {pct} out of range"
                )

    def test_percentiles_in_range_sparse(self, predictor):
        """All percentiles in [0, 100] for sparse response."""
        result = predictor.predict(INPUT_B_ITEMS)

        for domain in DOMAINS:
            for q in QUANTILES:
                pct = result[domain]["percentile"][q]
                assert 0 <= pct <= 100, (
                    f"{domain}_{q}: percentile {pct} out of range"
                )


# ── Calibration regime boundary ──────────────────────────────────────────


class TestCalibrationRegime:
    """Lock the count-only regime dispatch and its K=49/K=50 boundary.

    The regime selects which scale_factor multiplies the prediction interval,
    so a boundary regression (>=50 -> >=49 / >50) would change deployed coverage.
    """

    def test_50_answered_is_full_50(self, predictor):
        arr = np.full(50, 3.0, dtype=np.float32)
        assert predictor._calibration_regime(arr) == "full_50"

    def test_49_answered_is_sparse(self, predictor):
        arr = np.full(50, np.nan, dtype=np.float32)
        arr[:49] = 3.0
        assert predictor._calibration_regime(arr) == "sparse_20_balanced"

    def test_nan_not_counted_as_answered(self, predictor):
        # 49 real answers + 1 NaN must stay sparse, not flip to full_50.
        arr = np.full(50, np.nan, dtype=np.float32)
        arr[:49] = 3.0
        assert predictor._calibration_regime(arr) == "sparse_20_balanced"


# ── Config schema ────────────────────────────────────────────────────────

OUTPUT_DIR = FIXTURE_DIR
EXPECTED_OUTPUTS = [f"{d}_{q}" for d in DOMAINS for q in QUANTILES]


@pytest.fixture(scope="module")
def config():
    with open(OUTPUT_DIR / "config.json") as f:
        return json.load(f)


class TestConfigSchema:
    def test_model_file_key_exists(self, config):
        """config.json has model_file pointing to a real file."""
        assert isinstance(config["model_file"], str)
        assert (OUTPUT_DIR / config["model_file"]).is_file()

    def test_outputs_key_has_15_names(self, config):
        """config.json outputs lists all 15 domain-quantile names in order."""
        assert config["outputs"] == EXPECTED_OUTPUTS

    def test_scores_output_key_exists(self, config):
        """config.json has scores_output key."""
        assert config["scores_output"] == "scores"

    def test_no_legacy_models_key(self, config):
        """config.json does not contain the old 'models' mapping."""
        assert "models" not in config


# ── ONNX model structure ────────────────────────────────────────────────


@pytest.fixture(scope="module")
def session(config):
    return ort.InferenceSession(str(OUTPUT_DIR / config["model_file"]))


class TestOnnxModelStructure:
    def test_single_input_named_input(self, session):
        """Merged model has exactly one input named 'input'."""
        inputs = session.get_inputs()
        assert len(inputs) == 1
        assert inputs[0].name == "input"

    def test_input_shape(self, session):
        """Input shape is [batch, 50]."""
        shape = session.get_inputs()[0].shape
        assert len(shape) == 2
        assert shape[1] == 50

    def test_has_15_named_outputs(self, session):
        """Model exposes all 15 domain-quantile named outputs."""
        output_names = {o.name for o in session.get_outputs()}
        for name in EXPECTED_OUTPUTS:
            assert name in output_names, f"missing output: {name}"

    def test_has_scores_concat_output(self, session):
        """Model exposes the 'scores' concat output."""
        output_names = {o.name for o in session.get_outputs()}
        assert "scores" in output_names

    def test_total_output_count(self, session):
        """Model has exactly 16 outputs (15 named + scores)."""
        assert len(session.get_outputs()) == 16

    def test_individual_output_shapes(self, session):
        """Each named output has shape [batch, 1]."""
        for o in session.get_outputs():
            if o.name == "scores":
                continue
            assert o.shape[1] == 1, f"{o.name} shape[1] = {o.shape[1]}"

    def test_scores_output_shape(self, session):
        """Scores concat output has shape [batch, 15]."""
        for o in session.get_outputs():
            if o.name == "scores":
                assert o.shape[1] == 15
                return
        pytest.fail("scores output not found")

    def test_run_produces_all_outputs(self, session):
        """Running the model returns all 15 named outputs with correct shapes."""
        arr = np.full((1, 50), 3.0, dtype=np.float32)
        results = session.run(EXPECTED_OUTPUTS, {"input": arr})
        assert len(results) == 15
        for i, name in enumerate(EXPECTED_OUTPUTS):
            assert results[i].shape == (1, 1), f"{name}: shape={results[i].shape}"

    def test_scores_tensor_matches_individual_outputs(self, session):
        """The scores concat tensor equals the 15 individual outputs stacked."""
        arr = np.full((1, 50), 3.0, dtype=np.float32)
        all_names = EXPECTED_OUTPUTS + ["scores"]
        results = session.run(all_names, {"input": arr})
        individual = np.concatenate(results[:15], axis=1)
        scores = results[15]
        np.testing.assert_array_equal(individual, scores)
