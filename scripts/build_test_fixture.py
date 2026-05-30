"""Generate the committed test-fixture bundle in ``tests/fixtures/golden/``.

This is a ONE-OFF generator (run via ``make fixtures``), NOT a test — it lives in
``scripts/`` so pytest (``testpaths=["tests"]``) never collects it. It produces a
tiny, deterministic ONNX model + config so the artifact-dependent tri-runtime
parity / canonical-item / calibration-boundary tests can actually run in CI
without the real 137 MB ``output/reference`` bundle (which is gitignored).

The bundle is a numerical PARITY ORACLE, not a research artifact: it is trained
on a small synthetic dataset with synthetic norms, and every figure here is
arbitrary. What matters is that all three inference runtimes (Python,
TypeScript, web) reproduce ``golden_vectors.json`` from the SAME committed
``model.onnx`` + ``config.json``.

Determinism / byte-stability contract (so the committed ``model.onnx`` does not
churn on every rebuild):
  * fixed RNG seed (20240601) for the synthetic data;
  * XGBoost ``random_state=42`` (hardcoded in ``_create_xgb_model``), ``n_jobs=1``,
    ``subsample = colsample_bytree = 1.0`` (no stochastic tie-breaking);
  * ``OMP_NUM_THREADS=1`` to avoid thread-order nondeterminism;
  * ``json.dump(..., sort_keys=True, indent=2)`` for stable text;
  * the exact, version-pinned export path (xgboost / onnx / onnxmltools /
    onnxruntime are hash-pinned in ``uv.lock``).
On an intentional dependency bump that changes ONNX serialization, re-run
``make fixtures`` and re-commit. ``tests/test_fixture_stable.py`` asserts numeric
parity (not a model hash), so a re-serialization that preserves predictions
stays green; only a true prediction change fails.
"""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np  # noqa: E402

from lib.constants import (  # noqa: E402
    DOMAIN_CSV_TO_INTERNAL,
    DOMAINS,
    ITEM_COLUMNS,
    ITEMS_PER_DOMAIN,
    QUANTILE_NAME_LIST,
    QUANTILES,
    REVERSE_KEYED,
)

FIXTURE_DIR = PACKAGE_ROOT / "tests" / "fixtures" / "golden"
SEED = 20240601
N_ROWS = 400

# Tiny, deterministic hyperparameters (~360 tree nodes total -> a small ONNX).
TINY_PARAMS = {
    "n_estimators": 8,
    "max_depth": 2,
    "learning_rate": 0.3,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "min_child_weight": 1,
}

# Verified canonical domain_balanced-20 set under canonical_v1 (top-4 per domain
# by |own_domain_r|), used as a fallback when item_info.json is absent. Matches
# web/src/client/items.ts exactly.
_CANONICAL_20_FALLBACK = [
    "ext4", "ext5", "ext7", "ext2",
    "agr4", "agr9", "agr7", "agr5",
    "csn6", "csn1", "csn5", "csn4",
    "est8", "est6", "est1", "est7",
    "opn10", "opn2", "opn1", "opn5",
]


def _load_pipeline_module(script_name: str):
    module_path = PACKAGE_ROOT / "pipeline" / script_name
    spec = importlib.util.spec_from_file_location(
        f"_fixture_{script_name.replace('.', '_')}", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_inference_module():
    path = PACKAGE_ROOT / "python" / "inference.py"
    spec = importlib.util.spec_from_file_location("_fixture_inference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _is_reverse_keyed(item_id: str) -> bool:
    """Reverse-keying per lib.constants.REVERSE_KEYED (the authoritative source)."""
    code = item_id[:3]
    number = int(item_id[3:])
    csv_domain = next(k for k, v in DOMAIN_CSV_TO_INTERNAL.items() if v == code)
    return number in REVERSE_KEYED[csv_domain]


def _canonical_20() -> list[str]:
    """Derive the canonical domain_balanced-20 set from item_info.json if present.

    Mirrors pipeline/09_baselines.py::_select_domain_balanced (top-4 per domain
    by |ownDomainR|). Falls back to the verified frozen set when the (gitignored)
    item_info.json is absent.
    """
    info_path = PACKAGE_ROOT / "data" / "processed" / "canonical_v1" / "item_info.json"
    if not info_path.exists():
        return list(_CANONICAL_20_FALLBACK)
    pool = json.loads(info_path.read_text())["itemPool"]
    selected: list[str] = []
    for domain in DOMAINS:
        items = [it for it in pool if it["homeDomain"] == domain]
        items.sort(key=lambda x: abs(x.get("ownDomainR", 0.0)), reverse=True)
        selected.extend(it["id"] for it in items[:4])
    # Hermeticity: the committed fixture must be a pure function of committed
    # inputs. item_info.json is gitignored, so when it IS present the derived
    # selection must equal the frozen fallback — else the fixture bytes would
    # silently depend on a non-committed file. Fail loud on divergence.
    if selected != _CANONICAL_20_FALLBACK:
        raise SystemExit(
            f"Derived canonical-20 {selected} != frozen _CANONICAL_20_FALLBACK "
            f"{_CANONICAL_20_FALLBACK}. The seed-locked split selection changed; "
            "update the fallback (and web/src/client/items.ts) deliberately."
        )
    return selected


def _synthesize_training_data(rng: np.random.Generator):
    """Item responses in [1,5] with a domain target = item-mean + noise.

    The additive noise gives the quantile regressors a real conditional spread,
    so q05 < q50 < q95 holds (the export's parity check and the runtime ordering
    tests both rely on a non-degenerate spread).
    """
    x = rng.integers(1, 6, size=(N_ROWS, len(ITEM_COLUMNS))).astype(np.float32)
    targets: dict[str, np.ndarray] = {}
    for d_idx, domain in enumerate(DOMAINS):
        cols = slice(d_idx * ITEMS_PER_DOMAIN, (d_idx + 1) * ITEMS_PER_DOMAIN)
        base = x[:, cols].mean(axis=1)
        noise = rng.normal(0.0, 0.5, size=N_ROWS).astype(np.float32)
        targets[domain] = np.clip(base + noise, 1.0, 5.0).astype(np.float32)
    return x, targets


def main() -> int:
    train = _load_pipeline_module("07_train.py")
    export = _load_pipeline_module("11_export_onnx.py")

    rng = np.random.default_rng(SEED)
    x, targets = _synthesize_training_data(rng)

    # Synthetic, train-only norms (mean/sd of each domain target).
    fixture_norms = {
        d: {"mean": float(targets[d].mean()), "sd": float(targets[d].std(ddof=1))}
        for d in DOMAINS
    }

    # Train 15 tiny quantile models via the real stage-07 factory.
    models: dict[str, object] = {}
    for domain in DOMAINS:
        for q_value, q_name in zip(QUANTILES, QUANTILE_NAME_LIST):
            model = train._create_xgb_model(q_value, TINY_PARAMS, n_jobs=1)
            model.fit(x, targets[domain])
            models[f"{domain}_{q_name}"] = model

    # Export via the real stage-11 functions (faithful graph + parity checks).
    export._patch_onnxmltools_xgb3()
    onnx_models = export.convert_to_onnx(models)
    merged = export.merge_onnx_models(onnx_models)
    export.validate_parity(models, onnx_models)
    export.validate_merged_parity(merged, onnx_models)

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=PACKAGE_ROOT) as tmp:
        tmp_path = Path(tmp)
        # save_onnx_file rmtree's its output dir, so export to a temp subdir of
        # PACKAGE_ROOT, then copy model.onnx into the committed fixture dir.
        onnx_out = tmp_path / "onnx_out"
        export.save_onnx_file(merged, onnx_out)
        shutil.copyfile(onnx_out / "model.onnx", FIXTURE_DIR / "model.onnx")

        # Calibration sidecars with DISTINCT per-regime scale_factors, so the
        # golden-vector parity tests actually exercise the interval-rescale
        # branch (q05/q95 widened about q50, clipped to [0,100]) in all three
        # runtimes — not just the scale==1.0 no-op. full_50 stays 1.0 (a no-op
        # reference); sparse_20_balanced uses 1.3 so its cases diverge.
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        regime_scale = {"full_50": 1.0, "sparse_20_balanced": 1.3}
        for regime, scale in regime_scale.items():
            cal = {
                d: {"observed_coverage": 0.9, "scale_factor": scale} for d in DOMAINS
            }
            (models_dir / f"calibration_params_{regime}.json").write_text(
                json.dumps(cal, indent=2)
            )

        config = export.generate_config(
            models_dir=models_dir,
            artifacts_dir=models_dir,
            provenance_dict={
                "git_hash": "fixture",
                "data_snapshot_id": "fixture",
                "preprocessing_version": "fixture",
            },
            norms_map=fixture_norms,
            variant_name="golden",
        )

    # generate_config records models_dir in provenance; here that's the random
    # TemporaryDirectory name, which would churn config.json on every rebuild.
    # Pin it so the committed config is byte-stable (the field is unused at
    # inference; the fixture is identified by provenance.git_hash == "fixture").
    if isinstance(config.get("provenance"), dict):
        config["provenance"]["model_dir"] = "fixture"

    (FIXTURE_DIR / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )

    # canonical_items.json: shared source of truth for the canonical-20 tests.
    canonical = _canonical_20()
    (FIXTURE_DIR / "canonical_items.json").write_text(
        json.dumps(
            {
                "domain_balanced_20": canonical,
                "reverse_keyed": {iid: _is_reverse_keyed(iid) for iid in canonical},
            },
            indent=2,
        )
        + "\n"
    )

    # golden_vectors.json: the cross-runtime oracle. Generated from the committed
    # fixture via the deterministic (single-thread) Python reference runtime.
    inference = _load_inference_module()
    predictor = inference.IPIPBFFMPredictor(model_dir=FIXTURE_DIR)

    def _array_case(values: list[float]) -> dict:
        arr = np.array(values, dtype=np.float32).reshape(1, len(ITEM_COLUMNS))
        return predictor.predict_array(arr)

    ramp = [float((i % 5) + 1) for i in range(len(ITEM_COLUMNS))]
    k49 = ramp.copy()
    k49[-1] = float("nan")  # 49 answered -> sparse regime
    canonical_resp = {iid: float((i % 5) + 1) for i, iid in enumerate(canonical)}

    inputs: dict[str, dict] = {
        "full_50_ramp": {"array": ramp},
        "all_3s": {"array": [3.0] * len(ITEM_COLUMNS)},
        "all_1s": {"array": [1.0] * len(ITEM_COLUMNS)},
        "all_5s": {"array": [5.0] * len(ITEM_COLUMNS)},
        "k49_sparse": {"array": k49},
        "canonical_20": {"items": canonical_resp},
    }
    expected: dict[str, dict] = {}
    for name, spec in inputs.items():
        if "array" in spec:
            expected[name] = _array_case(spec["array"])
        else:
            expected[name] = predictor.predict(spec["items"])

    # JSON cannot hold NaN: serialize unanswered items in array cases as null.
    serializable_inputs = {
        name: (
            {"array": [None if (isinstance(v, float) and np.isnan(v)) else v
                       for v in spec["array"]]}
            if "array" in spec
            else spec
        )
        for name, spec in inputs.items()
    }
    (FIXTURE_DIR / "golden_vectors.json").write_text(
        json.dumps(
            {
                "_comment": (
                    "Cross-runtime parity oracle generated by "
                    "scripts/build_test_fixture.py from the committed model.onnx. "
                    "Regenerate with `make fixtures`."
                ),
                "tolerance": {"raw_atol": 1e-4, "percentile_atol": 0.1},
                "inputs": serializable_inputs,
                "expected": expected,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    model_kb = (FIXTURE_DIR / "model.onnx").stat().st_size / 1024
    print(f"Fixture written to {FIXTURE_DIR} (model.onnx = {model_kb:.0f} KB)")
    print(f"  golden vectors: {len(expected)} inputs x {len(DOMAINS)} domains")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
