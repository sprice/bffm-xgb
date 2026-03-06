"""Tests for IPIP-BFFM inference module."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

_OUTPUT_CONFIG = Path(__file__).resolve().parent.parent / "output" / "reference" / "config.json"

pytestmark = pytest.mark.skipif(
    not _OUTPUT_CONFIG.exists(),
    reason="output/config.json not found (run `make export` first)",
)

from inference import IPIPBFFMPredictor

# ── Test vectors ─────────────────────────────────────────────────────────

# Input A: Full 50-item response, repeating 1-5 pattern
INPUT_A_VALUES = [float((i % 5) + 1) for i in range(50)]

# Input B: 20-item sparse response, top-4 per domain
INPUT_B_ITEMS = {
    "ext3": 3.0, "ext5": 4.0, "ext7": 5.0, "ext4": 2.0,
    "agr10": 3.0, "agr7": 4.0, "agr2": 5.0, "agr9": 2.0,
    "csn4": 3.0, "csn8": 4.0, "csn1": 5.0, "csn5": 2.0,
    "est10": 3.0, "est9": 4.0, "est8": 5.0, "est6": 2.0,
    "opn5": 3.0, "opn10": 4.0, "opn7": 5.0, "opn2": 2.0,
}

DOMAINS = ["ext", "agr", "csn", "est", "opn"]
QUANTILES = ["q05", "q50", "q95"]

# Feature names in order
FEATURE_NAMES = [f"{d}{i}" for d in DOMAINS for i in range(1, 11)]


@pytest.fixture(scope="module")
def predictor():
    return IPIPBFFMPredictor()


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


# ── Config schema ────────────────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "reference"
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
