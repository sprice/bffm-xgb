"""Regression lock for the committed test-fixture bundle.

Re-runs every golden-vector input through the Python reference runtime against
the committed ``tests/fixtures/golden/model.onnx`` + ``config.json`` and asserts
the predictions still match ``golden_vectors.json``. The fixture is committed, so
this runs unconditionally (no skip, no REQUIRE_ARTIFACTS gate) and fails loudly
if the model bytes or the inference math drift.

Tolerances come from the oracle itself (raw_atol = 1e-4, percentile_atol = 0.1):
the recorded percentiles were produced on one host, and single-threaded ONNX
Runtime is deterministic per host but not guaranteed bit-identical across CPU
architectures; 1e-4 raw / 0.1pp absorbs that without masking a real regression.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = PACKAGE_ROOT / "tests" / "fixtures" / "golden"

DOMAINS = ["ext", "agr", "csn", "est", "opn"]
QUANTILES = ["q05", "q50", "q95"]
N_FEATURES = 50


def _load_predictor():
    path = PACKAGE_ROOT / "python" / "inference.py"
    spec = importlib.util.spec_from_file_location("_fixture_inference_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.IPIPBFFMPredictor(model_dir=FIXTURE_DIR)


def _golden() -> dict:
    return json.loads((FIXTURE_DIR / "golden_vectors.json").read_text())


def test_fixture_files_exist():
    for name in ("model.onnx", "config.json", "golden_vectors.json", "canonical_items.json"):
        assert (FIXTURE_DIR / name).is_file(), f"missing committed fixture file: {name}"


def test_fixture_config_model_dir_is_pinned():
    """provenance.model_dir is scrubbed to "fixture" so config.json is byte-stable
    across rebuilds (it otherwise leaks the random temp-dir name)."""
    config = json.loads((FIXTURE_DIR / "config.json").read_text())
    assert config.get("provenance", {}).get("model_dir") == "fixture"


def test_fixture_reproduces_golden_vectors():
    golden = _golden()
    # Floor guard: a malformed `make fixtures` that emitted {"inputs": {}} would
    # make the loop below vacuously pass. Require the full set of fixed cases.
    assert len(golden["inputs"]) >= 6, "golden_vectors.json has too few inputs"
    raw_atol = golden["tolerance"]["raw_atol"]
    pct_atol = golden["tolerance"]["percentile_atol"]
    predictor = _load_predictor()

    for name, spec in golden["inputs"].items():
        if "array" in spec:
            values = [np.nan if v is None else float(v) for v in spec["array"]]
            arr = np.array(values, dtype=np.float32).reshape(1, N_FEATURES)
            result = predictor.predict_array(arr)
        else:
            result = predictor.predict(spec["items"])

        for domain in DOMAINS:
            for q in QUANTILES:
                exp = golden["expected"][name][domain]
                got_raw = result[domain]["raw"][q]
                got_pct = result[domain]["percentile"][q]
                assert abs(got_raw - exp["raw"][q]) <= raw_atol, (
                    f"{name}/{domain}/{q} raw drift: {got_raw} vs {exp['raw'][q]}"
                )
                assert abs(got_pct - exp["percentile"][q]) <= pct_atol, (
                    f"{name}/{domain}/{q} percentile drift: {got_pct} vs {exp['percentile'][q]}"
                )


def test_golden_vectors_preserve_quantile_ordering():
    golden = _golden()
    for name, by_domain in golden["expected"].items():
        for domain, res in by_domain.items():
            p = res["percentile"]
            assert p["q05"] <= p["q50"] <= p["q95"], (
                f"{name}/{domain} percentile ordering violated: {p}"
            )
