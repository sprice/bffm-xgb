#!/usr/bin/env python3
"""Export IPIP-BFFM adaptive XGBoost models to ONNX format.

Converts 15 joblib models (5 domains × 3 quantiles) to ONNX, merges
them into a single model graph, validates parity, generates config.json,
and produces a HuggingFace-ready output/ directory.

Usage:
    python pipeline/11_export_onnx.py --data-dir data/processed/ext_est
    python pipeline/11_export_onnx.py --data-dir data/processed/ext_est --model-dir models/reference
"""

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

import argparse
import json
import logging
import shutil
from typing import Any

import joblib
import numpy as np

from lib.constants import DOMAINS, DOMAIN_LABELS, ITEM_COLUMNS
from lib.norms import load_norms
from lib.provenance import add_provenance_args, build_provenance, relative_to_root, sanitize_paths, file_sha256, _resolve_norms_lock_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUANTILE_NAMES = ["q05", "q50", "q95"]
QUANTILE_VALUES = [0.05, 0.5, 0.95]
N_FEATURES = 50
FEATURE_NAMES = list(ITEM_COLUMNS)
PARITY_TOL = 1e-4
# Small relative tolerance for XGBoost -> ONNX numeric drift on larger scores.
PARITY_RTOL = 5e-5
N_TEST_SAMPLES = 100
ARTIFACT_PROVENANCE_KEYS = (
    "git_hash",
    "data_snapshot_id",
    "preprocessing_version",
)
OPTIONAL_ARTIFACT_PROVENANCE_KEYS = (
    "split_signature",
)
DEFAULT_ARTIFACTS_DIR = Path("artifacts")


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------


def load_joblib_models(models_dir: Path) -> dict[str, object]:
    """Load all 15 joblib models, keyed like 'ext_q05'."""
    models = {}
    for domain in DOMAINS:
        for q in QUANTILE_NAMES:
            key = f"{domain}_{q}"
            path = models_dir / f"adaptive_{key}.joblib"
            if not path.exists():
                log.error("Missing model file: %s", path)
                sys.exit(1)
            models[key] = joblib.load(path)
    log.info("Loaded %d joblib models from %s", len(models), models_dir)
    return models


# ---------------------------------------------------------------------------
# ONNX conversion
# ---------------------------------------------------------------------------


def _patch_onnxmltools_xgb3() -> None:
    """
    Monkey-patch onnxmltools to handle XGBoost 3.x base_score format.

    XGBoost >=3.0 stores base_score as '[3E0]' (array notation) in
    booster.save_config(), but onnxmltools <=1.13 calls float() on it
    directly, which fails. This wraps get_xgb_params to parse the
    array notation before float conversion.
    """
    import onnxmltools.convert.xgboost.common as _xgb_common

    _orig_get_xgb_params = _xgb_common.get_xgb_params

    def _patched_get_xgb_params(xgb_node):
        import json as _json

        # Temporarily patch the booster config for the duration of this call
        booster = xgb_node.get_booster()
        config = _json.loads(booster.save_config())
        bs = config["learner"]["learner_model_param"].get("base_score", "")

        if isinstance(bs, str) and bs.startswith("["):
            # Intercept: parse '[1.5E0]' -> 1.5, then call original with
            # a wrapper that returns the fixed config
            fixed_bs = str(float(bs.strip("[]")))

            # Monkey-patch save_config temporarily
            orig_save_config = booster.save_config

            def _fixed_save_config():
                raw = orig_save_config()
                cfg = _json.loads(raw)
                cfg["learner"]["learner_model_param"]["base_score"] = fixed_bs
                return _json.dumps(cfg)

            booster.save_config = _fixed_save_config
            try:
                return _orig_get_xgb_params(xgb_node)
            finally:
                booster.save_config = orig_save_config
        else:
            return _orig_get_xgb_params(xgb_node)

    _xgb_common.get_xgb_params = _patched_get_xgb_params

    # Also patch the already-imported reference in operator_converters/XGBoost.py
    import onnxmltools.convert.xgboost.operator_converters.XGBoost as _xgb_ops

    _xgb_ops.get_xgb_params = _patched_get_xgb_params

    # Patch onnx.helper.make_attribute to cast bools to ints in int-list attrs.
    # onnxmltools passes Python bools in tree node splits, but onnx >= 1.16
    # rejects them with "Expected an int, got a boolean".
    import onnx.helper as _onnx_helper

    _orig_make_attribute = _onnx_helper.make_attribute

    def _patched_make_attribute(key, value):
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], (bool, np.bool_)):
            value = [int(v) for v in value]
        return _orig_make_attribute(key, value)

    _onnx_helper.make_attribute = _patched_make_attribute

    log.info("Patched onnxmltools for XGBoost 3.x base_score compatibility")
    log.info("Patched onnx.helper.make_attribute for bool-to-int coercion")


def convert_to_onnx(models: dict) -> dict[str, object]:
    """Convert each XGBoost model to ONNX."""
    try:
        import onnx
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError:
        log.error(
            "onnxmltools, onnx, or onnxruntime not installed. "
            "Run: pip install onnxmltools onnx onnxruntime"
        )
        sys.exit(1)

    _patch_onnxmltools_xgb3()

    onnx_models = {}
    for key, model in models.items():
        initial_type = [("input", FloatTensorType([None, N_FEATURES]))]
        onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)
        onnx.checker.check_model(onnx_model)
        onnx_models[key] = onnx_model
        log.info("  Converted %s", key)
    log.info("Converted %d models to ONNX", len(onnx_models))
    return onnx_models


# ---------------------------------------------------------------------------
# Merge into single graph
# ---------------------------------------------------------------------------


def merge_onnx_models(onnx_models: dict) -> object:
    """Merge 15 individual ONNX models into a single graph with shared input.

    Constructs a single ONNX graph where:
    - All models share input ``"input"`` with shape ``[None, 50]``
    - Each model's nodes produce domain-quantile outputs (ext_q05, ext_q50, ...)
    - A ``Concat`` node combines all 15 outputs into ``"scores"`` ``[None, 15]``
    """
    import onnx
    from onnx import TensorProto, helper

    output_order = [f"{d}_{q}" for d in DOMAINS for q in QUANTILE_NAMES]

    all_nodes: list = []
    all_initializers: list = []

    for key in output_order:
        model = onnx_models[key]
        graph = model.graph

        src_input = graph.input[0].name
        src_output = graph.output[0].name

        # Build rename map: shared input, prefixed intermediates, named output
        rename: dict[str, str] = {src_input: "input"}

        for init in graph.initializer:
            rename[init.name] = f"{key}/{init.name}"

        for node in graph.node:
            for name in node.output:
                if name == src_output:
                    rename[name] = key
                elif name not in rename:
                    rename[name] = f"{key}/{name}"

        for node in graph.node:
            for name in node.input:
                if name and name not in rename:
                    rename[name] = f"{key}/{name}"

        # Copy nodes with renamed I/O and unique node names
        for node_idx, node in enumerate(graph.node):
            n = onnx.NodeProto()
            n.CopyFrom(node)
            if n.name:
                n.name = f"{key}/{n.name}"
            else:
                n.name = f"{key}/node_{node_idx}"
            for i in range(len(n.input)):
                if n.input[i] in rename:
                    n.input[i] = rename[n.input[i]]
            for i in range(len(n.output)):
                if n.output[i] in rename:
                    n.output[i] = rename[n.output[i]]
            all_nodes.append(n)

        # Copy initializers with renamed names
        for init in graph.initializer:
            t = onnx.TensorProto()
            t.CopyFrom(init)
            t.name = rename[init.name]
            all_initializers.append(t)

    # Concat node: combine 15 outputs into [None, 15]
    all_nodes.append(
        helper.make_node("Concat", inputs=output_order, outputs=["scores"], axis=1)
    )

    graph_inputs = [
        helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, N_FEATURES])
    ]
    graph_outputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, [None, 1])
        for name in output_order
    ] + [
        helper.make_tensor_value_info(
            "scores", TensorProto.FLOAT, [None, len(output_order)]
        )
    ]

    merged_graph = helper.make_graph(
        all_nodes,
        "merged_ipip_bffm",
        graph_inputs,
        graph_outputs,
        initializer=all_initializers,
    )

    merged = helper.make_model(
        merged_graph,
        opset_imports=[
            helper.make_opsetid("", 18),
            helper.make_opsetid("ai.onnx.ml", 1),
        ],
    )
    onnx.checker.check_model(merged)
    log.info(
        "Merged %d models into single graph (%d nodes)",
        len(output_order),
        len(all_nodes),
    )
    return merged


# ---------------------------------------------------------------------------
# Parity validation
# ---------------------------------------------------------------------------


def validate_parity(joblib_models: dict, onnx_models: dict) -> None:
    """Run predictions through both backends and assert numerical parity."""
    try:
        import onnxruntime as ort
    except ImportError:
        log.error("onnxruntime not installed. Run: pip install onnxruntime")
        sys.exit(1)

    rng = np.random.default_rng(42)

    # Generate test data: mix of full responses and sparse (with NaN)
    X_full = rng.uniform(1.0, 5.0, size=(N_TEST_SAMPLES // 2, N_FEATURES)).astype(
        np.float32
    )
    X_sparse = rng.uniform(1.0, 5.0, size=(N_TEST_SAMPLES // 2, N_FEATURES)).astype(
        np.float32
    )
    # Randomly mask ~60% of items as NaN in sparse samples
    mask = rng.random(X_sparse.shape) < 0.6
    X_sparse[mask] = np.nan
    X_test = np.vstack([X_full, X_sparse])

    max_diff_overall = 0.0
    max_rel_diff_overall = 0.0
    for key in joblib_models:
        # Joblib prediction
        joblib_pred = joblib_models[key].predict(X_test)

        # ONNX prediction
        onnx_bytes = onnx_models[key].SerializeToString()
        sess = ort.InferenceSession(onnx_bytes)
        input_name = sess.get_inputs()[0].name
        ort_pred = sess.run(None, {input_name: X_test})[0].flatten()

        abs_diff = np.abs(joblib_pred - ort_pred)
        max_diff = float(np.max(abs_diff))
        max_diff_overall = max(max_diff_overall, max_diff)
        rel_scale = np.maximum(np.maximum(np.abs(joblib_pred), np.abs(ort_pred)), 1.0)
        rel_diff = abs_diff / rel_scale
        max_rel_diff = float(np.max(rel_diff))
        max_rel_diff_overall = max(max_rel_diff_overall, max_rel_diff)

        if not np.allclose(joblib_pred, ort_pred, atol=PARITY_TOL, rtol=PARITY_RTOL, equal_nan=True):
            log.error(
                "  FAIL %s: max abs diff = %.2e (atol %.0e), max rel diff = %.2e (rtol %.0e)",
                key,
                max_diff,
                PARITY_TOL,
                max_rel_diff,
                PARITY_RTOL,
            )
            sys.exit(1)
        else:
            log.info("  PASS %s: max abs diff = %.2e, max rel diff = %.2e", key, max_diff, max_rel_diff)

    log.info(
        "All models pass parity check (max abs diff = %.2e, max rel diff = %.2e)",
        max_diff_overall,
        max_rel_diff_overall,
    )


def validate_merged_parity(
    merged_model: object,
    individual_models: dict,
) -> None:
    """Verify merged-model outputs are bit-for-bit identical to individual models."""
    try:
        import onnxruntime as ort
    except ImportError:
        log.error("onnxruntime not installed. Run: pip install onnxruntime")
        sys.exit(1)

    output_names = [f"{d}_{q}" for d in DOMAINS for q in QUANTILE_NAMES]

    rng = np.random.default_rng(42)
    X_full = rng.uniform(1.0, 5.0, size=(N_TEST_SAMPLES // 2, N_FEATURES)).astype(
        np.float32
    )
    X_sparse = rng.uniform(1.0, 5.0, size=(N_TEST_SAMPLES // 2, N_FEATURES)).astype(
        np.float32
    )
    mask = rng.random(X_sparse.shape) < 0.6
    X_sparse[mask] = np.nan
    X_test = np.vstack([X_full, X_sparse])

    merged_sess = ort.InferenceSession(merged_model.SerializeToString())
    merged_raw = merged_sess.run(output_names, {"input": X_test})
    merged_dict = dict(zip(output_names, merged_raw))

    max_diff_overall = 0.0
    for key in output_names:
        ind_sess = ort.InferenceSession(
            individual_models[key].SerializeToString()
        )
        input_name = ind_sess.get_inputs()[0].name
        ind_pred = ind_sess.run(None, {input_name: X_test})[0]

        diff = float(np.max(np.abs(ind_pred - merged_dict[key])))
        max_diff_overall = max(max_diff_overall, diff)

        if diff > 0:
            log.error("  FAIL %s: max diff = %.2e", key, diff)
            sys.exit(1)
        log.info("  PASS %s: bit-for-bit match", key)

    log.info(
        "All merged outputs match individual models (max diff = %.2e)",
        max_diff_overall,
    )


# ---------------------------------------------------------------------------
# Save ONNX file
# ---------------------------------------------------------------------------


def save_onnx_file(merged_model: object, output_dir: Path) -> None:
    """Write single merged model.onnx to output directory."""
    import onnx

    resolved = output_dir.resolve()
    if not resolved.is_relative_to(PACKAGE_ROOT) or resolved == PACKAGE_ROOT:
        raise ValueError(f"output_dir must be a subdirectory of the package root: {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    path = output_dir / "model.onnx"
    onnx.save(merged_model, str(path))
    size_kb = path.stat().st_size / 1024
    log.info("  %s (%.0f KB)", path.name, size_kb)


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------


def generate_config(
    models_dir: Path,
    artifacts_dir: Path,
    provenance_dict: dict,
    norms_map: dict[str, dict[str, float]],
    *,
    variant_name: str | None = None,
) -> dict:
    """Build config.json content from reference artifacts and model metadata."""
    hyperparameters: dict = {}

    # Try to load calibration params from models dir
    calibration: dict = {}

    # Sparse calibration
    sparse_cal_path = models_dir / "calibration_params_sparse_20_balanced.json"
    if not sparse_cal_path.exists():
        sparse_cal_path = models_dir / "calibration_params.json"
    if sparse_cal_path.exists():
        with open(sparse_cal_path) as f:
            cal_sparse = json.load(f)
        missing = [d for d in DOMAINS if d not in cal_sparse]
        if missing:
            raise ValueError(f"Sparse calibration missing domains: {missing}")
        calibration["sparse_20_balanced"] = {
            d: {
                "observed_coverage": cal_sparse[d].get("observed_coverage", 0.0),
                "scale_factor": cal_sparse[d].get("scale_factor", 1.0),
            }
            for d in DOMAINS
        }

    # Full-50 calibration
    full_cal_path = models_dir / "calibration_params_full_50.json"
    if full_cal_path.exists():
        with open(full_cal_path) as f:
            cal_full = json.load(f)
        missing = [d for d in DOMAINS if d not in cal_full]
        if missing:
            raise ValueError(f"Full-50 calibration missing domains: {missing}")
        calibration["full_50"] = {
            d: {
                "observed_coverage": cal_full[d].get("observed_coverage", 0.0),
                "scale_factor": cal_full[d].get("scale_factor", 1.0),
            }
            for d in DOMAINS
        }

    # Load training report for data provenance / hyperparameters
    # Try new filename first, fall back to legacy name
    report_path = models_dir / "training_report.json"
    if not report_path.exists():
        report_path = models_dir / "adaptive_training_report.json"
    data_prov: dict = {}
    report_prov: dict = {}
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        if not isinstance(report, dict):
            report = {}
        report_prov = report.get("provenance", {})
        report_data = report.get("data", {})
        report_cfg = report.get("config", {})
        if isinstance(report_cfg, dict):
            hp = report_cfg.get("hyperparameters")
            if isinstance(hp, dict):
                hyperparameters = hp
        data_prov = {
            "n_train_original": (
                report_data.get("train_rows")
                or report_data.get("n_train_original", 0)
            ),
            "n_train_augmented": (
                report_data.get("train_rows_after_augmentation")
                or report_data.get("n_train_augmented", 0)
            ),
            "n_test": (
                report_data.get("n_test")
                or report_data.get("test_rows", 0)
            ),
        }
        for key in (
            "train_sha256",
            "val_sha256",
            "test_sha256",
            "split_signature",
            "split_metadata_sha256",
        ):
            value = report_data.get(key)
            if isinstance(value, str) and value.strip():
                data_prov[key] = value.strip()

    artifact_match_signature = _build_expected_artifact_signature(
        models_dir,
        report_provenance=report_prov,
        fallback_provenance=provenance_dict,
    )
    if artifact_match_signature is None:
        log.warning(
            "Could not build strict artifact provenance signature for config export; "
            "artifact-derived backfills will be skipped."
        )

    # Load validation/simulation results for metrics
    val_path = artifacts_dir / "validation_results.json"
    sim_path = artifacts_dir / "simulation_results.json"

    # Backfill n_test from validation artifacts when training report lacks it.
    if (
        artifact_match_signature is not None
        and int(data_prov.get("n_test", 0) or 0) <= 0
        and val_path.exists()
    ):
        try:
            with open(val_path) as f:
                val_payload = json.load(f)
            if _artifact_matches_signature(val_payload, artifact_match_signature):
                overall_n = (
                    val_payload.get("metrics", {})
                    .get("overall", {})
                    .get("n")
                )
                if isinstance(overall_n, (int, float)):
                    overall_n_int = int(overall_n)
                    if overall_n_int > 0:
                        if overall_n_int % len(DOMAINS) == 0:
                            data_prov["n_test"] = overall_n_int // len(DOMAINS)
                        else:
                            data_prov["n_test"] = overall_n_int
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
            log.warning(
                "Skipping n_test backfill from validation artifact due to parse/shape error: %s",
                e,
            )

    output_names = [f"{d}_{q}" for d in DOMAINS for q in QUANTILE_NAMES]

    norms = {d: {"mean": norms_map[d]["mean"], "sd": norms_map[d]["sd"]} for d in DOMAINS}

    config: dict = {
        "model_name": "bffm-xgb",
        "model_display_name": "IPIP-BFFM Sparse Quantile Model",
        "model_type": "xgboost-quantile-regression",
        "framework": "onnx",
        "assessment": "ipip-bffm-50",
        "version": "1.0.0",
        "domains": list(DOMAINS),
        "domain_labels": dict(DOMAIN_LABELS),
        "quantiles": QUANTILE_VALUES,
        "model_file": "model.onnx",
        "outputs": output_names,
        "scores_output": "scores",
        "input": {
            "shape": [None, N_FEATURES],
            "dtype": "float32",
            "feature_names": FEATURE_NAMES,
            "missing_value": "NaN",
            "value_range": [1, 5],
        },
        "output": {
            "shape": [None, 1],
            "dtype": "float32",
            "scale": "raw_score",
            "value_range": [1, 5],
        },
        "norms": norms,
    }

    if hyperparameters:
        config["hyperparameters"] = hyperparameters

    if calibration:
        config["calibration"] = calibration

    # Build provenance section
    training_script = (
        report_prov.get("script")
        or report_prov.get("training_script")
        or "07_train.py"
    )
    preprocessing_version = (
        _provenance_token(report_prov, "preprocessing_version")
        or _provenance_token(provenance_dict, "preprocessing_version")
        or provenance_dict.get("git_hash", "unknown")
    )
    git_hash = report_prov.get(
        "git_hash",
        provenance_dict.get("git_hash", "unknown"),
    )
    data_snapshot_id = (
        _provenance_token(report_prov, "data_snapshot_id")
        or _provenance_token(provenance_dict, "data_snapshot_id")
        or f"git:{git_hash}"
    )
    config["provenance"] = {
        "source": report_prov.get("source", "ipip-bffm-adaptive-v1-reference"),
        "training_script": training_script,
        "git_hash": git_hash,
        "preprocessing_version": preprocessing_version,
        "data_snapshot_id": data_snapshot_id,
        "rng_seed": report_prov.get("rng_seed", 42),
        "model_dir": relative_to_root(models_dir),
    }
    config["provenance"].update(data_prov)

    if variant_name:
        config["variant"] = variant_name

    return config


# ---------------------------------------------------------------------------
# Provenance document generation
# ---------------------------------------------------------------------------


def generate_provenance_document(
    *,
    models_dir: Path,
    output_dir: Path,
    provenance_dict: dict,
    variant_name: str | None = None,
) -> dict:
    """Build a complete provenance audit trail document for output/provenance.json.

    This separates concerns: config.json is the runtime inference config,
    provenance.json is the full audit trail for reviewers.
    """
    # Load training report for verbatim provenance/config/data blocks
    report_path = models_dir / "training_report.json"
    if not report_path.exists():
        report_path = models_dir / "adaptive_training_report.json"

    training_block: dict = {}
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        if isinstance(report, dict):
            if isinstance(report.get("provenance"), dict):
                training_block["provenance"] = sanitize_paths(report["provenance"])
            if isinstance(report.get("config"), dict):
                training_block["config"] = sanitize_paths(report["config"])
            if isinstance(report.get("data"), dict):
                training_block["data"] = sanitize_paths(report["data"])

    # Compute artifact checksums
    artifacts_block: dict = {}
    model_onnx_path = output_dir / "model.onnx"
    if model_onnx_path.exists():
        artifacts_block["model_onnx_sha256"] = file_sha256(model_onnx_path)
    config_json_path = output_dir / "config.json"
    if config_json_path.exists():
        artifacts_block["config_json_sha256"] = file_sha256(config_json_path)

    # Norms lock checksum
    norms_lock_path = _resolve_norms_lock_path()
    norms_lock_sha256 = ""
    if norms_lock_path.exists():
        norms_lock_sha256 = file_sha256(norms_lock_path)

    report_prov = training_block.get("provenance", {})
    git_hash = report_prov.get(
        "git_hash",
        provenance_dict.get("git_hash", "unknown"),
    )
    data_snapshot_id = (
        _provenance_token(report_prov, "data_snapshot_id")
        or _provenance_token(provenance_dict, "data_snapshot_id")
        or f"git:{git_hash}"
    )
    preprocessing_version = (
        _provenance_token(report_prov, "preprocessing_version")
        or _provenance_token(provenance_dict, "preprocessing_version")
        or git_hash
    )

    export_block: dict = {
        "script": Path(__file__).name,
        "git_hash": provenance_dict.get("git_hash", "unknown"),
        "data_snapshot_id": data_snapshot_id,
        "preprocessing_version": preprocessing_version,
        "model_dir": relative_to_root(models_dir),
        "norms_lock_sha256": norms_lock_sha256,
    }
    if variant_name:
        export_block["variant"] = variant_name

    return {
        "schema_version": 1,
        "description": "Complete provenance audit trail for IPIP-BFFM sparse quantile model.",
        "export": export_block,
        "training": training_block,
        "artifacts": artifacts_block,
    }


# ---------------------------------------------------------------------------
# README generation
# ---------------------------------------------------------------------------


def _provenance_token(payload: dict, key: str) -> str | None:
    """Resolve provenance token with explicit fallback for preprocessing_version."""
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    if key == "preprocessing_version":
        git_hash = payload.get("git_hash")
        if isinstance(git_hash, str) and git_hash.strip():
            return git_hash.strip()
    return None


def _build_expected_artifact_signature(
    model_dir: Path,
    *,
    report_provenance: dict,
    fallback_provenance: dict,
) -> dict[str, str] | None:
    """Build strict artifact provenance signature for same-run matching."""
    signature: dict[str, str] = {"model_dir": relative_to_root(model_dir)}
    for key in ARTIFACT_PROVENANCE_KEYS:
        value = _provenance_token(report_provenance, key) or _provenance_token(
            fallback_provenance, key
        )
        if value is None:
            return None
        signature[key] = value

    for key in OPTIONAL_ARTIFACT_PROVENANCE_KEYS:
        value = _provenance_token(report_provenance, key) or _provenance_token(
            fallback_provenance, key
        )
        if value is not None:
            signature[key] = value
    return signature


def _artifact_matches_signature(
    payload: dict,
    expected_signature: dict[str, str] | None,
) -> bool:
    """Return True when artifact provenance matches expected strict signature."""
    if expected_signature is None:
        return False

    provenance = payload.get("provenance", {})
    if not isinstance(provenance, dict):
        return False

    model_dir_raw = provenance.get("model_dir")
    if not isinstance(model_dir_raw, str):
        return False
    artifact_model_dir = Path(model_dir_raw)
    if not artifact_model_dir.is_absolute():
        artifact_model_dir = PACKAGE_ROOT / artifact_model_dir
    expected_model_dir = Path(expected_signature["model_dir"])
    if not expected_model_dir.is_absolute():
        expected_model_dir = PACKAGE_ROOT / expected_model_dir
    if artifact_model_dir.resolve() != expected_model_dir.resolve():
        return False

    for key in ARTIFACT_PROVENANCE_KEYS:
        expected_value = expected_signature.get(key)
        actual_value = _provenance_token(provenance, key)
        if expected_value is None or actual_value != expected_value:
            return False

    for key in OPTIONAL_ARTIFACT_PROVENANCE_KEYS:
        if key not in expected_signature:
            continue
        expected_value = expected_signature.get(key)
        actual_value = _provenance_token(provenance, key)
        if expected_value is None or actual_value != expected_value:
            return False

    return True


def _format_md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Format a Markdown table with padded, equal-width columns."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    def _row(cells: list[str]) -> str:
        padded = [cell.ljust(widths[i]) for i, cell in enumerate(cells)]
        return "| " + " | ".join(padded) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([_row(headers), sep] + [_row(r) for r in rows])


def generate_readme(config: dict, artifacts_dir: Path, model_dir: Path, *, variant_name: str | None = None) -> str:
    """Build HuggingFace model card markdown.

    Reads existing metrics from artifacts to fill in performance tables.
    """
    # Require strict same-run artifact provenance for publication metrics.
    val_path = artifacts_dir / "validation_results.json"
    baseline_path = artifacts_dir / "baseline_comparison_results.json"
    ml_vs_avg_path = artifacts_dir / "ml_vs_averaging_comparison.json"
    expected_signature = _build_expected_artifact_signature(
        model_dir,
        report_provenance=config.get("provenance", {}),
        fallback_provenance={},
    )
    if expected_signature is None:
        raise ValueError(
            "Cannot build strict artifact provenance signature for README generation. "
            "Ensure config.provenance includes git_hash, data_snapshot_id, and "
            "preprocessing_version."
        )

    def _load_artifact(path: Path, label: str) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact for README generation: {path}. "
                "Run the full evaluation stack (`make validate make baselines make simulate`) "
                "before export."
            )
        with open(path) as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"{label} must contain a JSON object: {path}")
        if not _artifact_matches_signature(payload, expected_signature):
            raise ValueError(
                f"{label} provenance does not match model bundle signature. "
                "Re-run make validate/make baselines/make simulate for this model."
            )
        return payload

    def _require_metric(value: Any, *, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Missing numeric metric for {label}.")
        metric = float(value)
        if not np.isfinite(metric):
            raise ValueError(f"Non-finite metric for {label}: {metric}")
        return metric

    baselines = _load_artifact(baseline_path, "baseline_comparison_results.json")
    validation = _load_artifact(val_path, "validation_results.json")
    ml_vs_avg = _load_artifact(ml_vs_avg_path, "ml_vs_averaging_comparison.json")

    overall = baselines.get("overall")
    if not isinstance(overall, dict):
        raise ValueError("baseline_comparison_results.json missing 'overall' object.")
    k20 = overall.get("20")
    k50 = overall.get("50")
    if not isinstance(k20, dict) or not isinstance(k50, dict):
        raise ValueError(
            "baseline_comparison_results.json must contain K=20 and K=50 entries."
        )

    def _r_at(method_payload: Any, *, method: str, k: int) -> float:
        if not isinstance(method_payload, dict):
            raise ValueError(
                f"baseline_comparison_results.json missing method={method!r} at K={k}."
            )
        return _require_metric(
            method_payload.get("pearson_r"),
            label=f"baseline {method}@K={k} pearson_r",
        )

    perf_table = _format_md_table(
        ["Strategy", "Items (K)", "Correlation (r)"],
        [
            ["Full assessment", "50", f"{_r_at(k50.get('full_50'), method='full_50', k=50):.3f}"],
            ["Domain-balanced", "20", f"{_r_at(k20.get('domain_balanced'), method='domain_balanced', k=20):.3f}"],
            ["Mini-IPIP mapping", "20", f"{_r_at(k20.get('mini_ipip'), method='mini_ipip', k=20):.3f}"],
            ["Adaptive top-K", "20", f"{_r_at(k20.get('adaptive_topk'), method='adaptive_topk', k=20):.3f}"],
        ],
    )

    validation_metrics = validation.get("metrics", {})
    validation_sparse = validation.get("sparse_20", {})
    if not isinstance(validation_metrics, dict) or not isinstance(validation_sparse, dict):
        raise ValueError("validation_results.json has invalid metrics structure.")
    full_cov = _require_metric(
        validation_metrics.get("overall", {}).get("coverage_90")
        if isinstance(validation_metrics.get("overall"), dict) else None,
        label="validation full_50 coverage_90",
    )
    sparse_cov = _require_metric(
        validation_sparse.get("metrics", {}).get("overall", {}).get("coverage_90")
        if isinstance(validation_sparse.get("metrics"), dict)
        and isinstance(validation_sparse.get("metrics", {}).get("overall"), dict)
        else None,
        label="validation sparse_20 coverage_90",
    )
    coverage_line = (
        f"90% CI coverage: {sparse_cov * 100:.1f}% (sparse 20-item), "
        f"{full_cov * 100:.1f}% (full 50-item)."
    )

    comparisons = ml_vs_avg.get("comparisons")
    if not isinstance(comparisons, list):
        raise ValueError("ml_vs_averaging_comparison.json missing 'comparisons' list.")
    domain_balanced_20 = None
    for row in comparisons:
        if (
            isinstance(row, dict)
            and row.get("method") == "domain_balanced"
            and int(row.get("n_items", -1)) == 20
        ):
            domain_balanced_20 = row
            break
    if not isinstance(domain_balanced_20, dict):
        raise ValueError(
            "ml_vs_averaging_comparison.json missing domain_balanced@K=20 comparison."
        )
    delta_r = _require_metric(
        domain_balanced_20.get("delta_r"),
        label="ml_vs_avg domain_balanced@K=20 delta_r",
    )
    ml_advantage_line = (
        "ML advantage over simple averaging: "
        f"{delta_r:+.3f} r (domain-balanced K=20)."
    )

    # Hyperparameters
    hp = config.get("hyperparameters", {})
    n_est = hp.get("n_estimators", "?")
    max_d = hp.get("max_depth", "?")
    lr = hp.get("learning_rate", "?")
    if isinstance(lr, float):
        lr_str = f"{lr:.4f}"
    else:
        lr_str = str(lr)

    # Data
    prov = config.get("provenance", {})
    n_train_orig = prov.get("n_train_original", "?")
    n_train_aug = prov.get("n_train_augmented", "?")
    if isinstance(n_train_orig, int):
        n_train_orig_str = f"{n_train_orig:,}"
    else:
        n_train_orig_str = str(n_train_orig)
    if isinstance(n_train_aug, int):
        n_train_aug_str = f"{n_train_aug:,}"
    else:
        n_train_aug_str = str(n_train_aug)

    if n_train_aug == n_train_orig:
        training_data_line = (
            f"{n_train_orig_str} respondents from the Open-Source "
            "Psychometrics Project (OSPP) (no sparsity augmentation)"
        )
    else:
        training_data_line = (
            f"{n_train_orig_str} respondents from the Open-Source "
            f"Psychometrics Project (OSPP), augmented to {n_train_aug_str} "
            "via sparsity augmentation"
        )

    norms = config["norms"] if "norms" in config else load_norms()

    domain_table = _format_md_table(
        ["Domain", "Code", "Items"],
        [
            ["Extraversion", "`ext`", "ext1-ext10"],
            ["Agreeableness", "`agr`", "agr1-agr10"],
            ["Conscientiousness", "`csn`", "csn1-csn10"],
            ["Emotional Stability", "`est`", "est1-est10"],
            ["Intellect/Imagination", "`opn`", "opn1-opn10"],
        ],
    )

    norms_table = _format_md_table(
        ["Domain", "Mean", "SD"],
        [
            ["Extraversion", f"{norms['ext']['mean']:.3f}", f"{norms['ext']['sd']:.3f}"],
            ["Agreeableness", f"{norms['agr']['mean']:.3f}", f"{norms['agr']['sd']:.3f}"],
            ["Conscientiousness", f"{norms['csn']['mean']:.3f}", f"{norms['csn']['sd']:.3f}"],
            ["Emotional Stability", f"{norms['est']['mean']:.3f}", f"{norms['est']['sd']:.3f}"],
            ["Intellect/Imagination", f"{norms['opn']['mean']:.3f}", f"{norms['opn']['sd']:.3f}"],
        ],
    )

    variant_tag = ""
    if variant_name:
        variant_tag = f"\n  - variant:{variant_name}"

    if variant_name and variant_name != "reference":
        variant_note = (
            f"\n> **Note:** This is the `{variant_name}` research ablation variant. "
            "The primary model is `reference`.\n"
        )
    elif variant_name == "reference":
        variant_note = "\n> This is the primary published model.\n"
    else:
        variant_note = ""

    readme = f"""---
license: cc0-1.0
language: en
tags:
  - personality
  - psychometrics
  - big-five
  - ipip
  - xgboost
  - onnx
  - quantile-regression{variant_tag}
library_name: onnxruntime
pipeline_tag: tabular-regression
---

# IPIP-BFFM Sparse Quantile Model
{variant_note}
Sparse-input XGBoost quantile regression models for the 50-item IPIP Big-Five Factor Markers (BFFM) personality assessment, exported to ONNX format.

## Model Description

This package contains a **single merged ONNX model** with 15 outputs (5 personality domains × 3 quantiles) that predicts Big Five personality scores from item responses. The model is designed for **adaptive assessment** — it produces accurate predictions even when many items are missing (answered as NaN), enabling short-form assessments of 20 items or fewer.

{domain_table}

Each domain has three quantile models:
- **q05** -- 5th percentile (lower bound of 90% CI)
- **q50** -- median (point estimate)
- **q95** -- 95th percentile (upper bound of 90% CI)

## Input Specification

- **Shape:** `[batch_size, 50]` -- one column per IPIP-BFFM item
- **Dtype:** `float32`
- **Values:** `1.0` to `5.0` (Likert scale), or `NaN` for unanswered items
- **Feature order:** ext1, ext2, ..., ext10, agr1, ..., agr10, csn1, ..., csn10, est1, ..., est10, opn1, ..., opn10

## Output Specification

- **Shape:** `[batch_size, 1]`
- **Scale:** Raw domain score (1-5 range)
- **Percentile conversion:** Use the provided norms (z-score -> CDF)

## Quick Start (Python)

Requires `onnxruntime`, `numpy`, `scipy`.

```python
import json, numpy as np, onnxruntime as ort
from scipy.stats import norm

# Load model and config
sess = ort.InferenceSession("model.onnx")
with open("config.json") as f:
    config = json.load(f)

# Build input array (NaN = unanswered)
# Reverse-keyed items must already be transformed via `6 - raw_value`.
responses = {{
    "ext3": 4.0, "ext5": 5.0, "agr1": 3.0, "agr7": 4.0,
    "csn1": 5.0, "csn4": 3.0, "est9": 4.0, "est10": 3.0,
    "opn5": 3.0, "opn10": 4.0,
}}
features = config["input"]["feature_names"]
arr = np.full((1, len(features)), np.nan, dtype=np.float32)
for item, val in responses.items():
    arr[0, features.index(item)] = val

# Run inference (single call for all 15 outputs)
outputs = sess.run(config["outputs"], {{"input": arr}})
scores = dict(zip(config["outputs"], outputs))

# Convert to percentiles
for domain in config["domains"]:
    raw = float(scores[f"{{domain}}_q50"].flatten()[0])
    n = config["norms"][domain]
    pct = norm.cdf((raw - n["mean"]) / n["sd"]) * 100
    print(f"{{domain}}: {{pct:.1f}}th percentile (raw={{raw:.3f}})")
```

## Quick Start (TypeScript)

Requires `onnxruntime-node`.

```typescript
import {{ readFileSync }} from "node:fs";
import * as ort from "onnxruntime-node";

// Load model and config
const config = JSON.parse(readFileSync("config.json", "utf-8"));
const session = await ort.InferenceSession.create("model.onnx");

// Build input array (NaN = unanswered)
// Reverse-keyed items must already be transformed via `6 - rawValue`.
const responses: Record<string, number> = {{
  ext3: 4.0, ext5: 5.0, agr1: 3.0, agr7: 4.0,
  csn1: 5.0, csn4: 3.0, est9: 4.0, est10: 3.0,
  opn5: 3.0, opn10: 4.0,
}};
const features: string[] = config.input.feature_names;
const arr = new Float32Array(features.length).fill(NaN);
for (const [item, val] of Object.entries(responses)) {{
  arr[features.indexOf(item)] = val;
}}

// Run inference (single call for all 15 outputs)
const output = await session.run({{
  input: new ort.Tensor("float32", arr, [1, features.length]),
}});

for (const domain of config.domains) {{
  const raw = (output[`${{domain}}_q50`].data as Float32Array)[0];
  console.log(`${{domain}}: raw=${{raw.toFixed(3)}}`);
}}

session.release();
```

## Training Details

- **Algorithm:** XGBoost quantile regression with pinball loss
- **Training data:** {training_data_line}
- **Sparsity augmentation:** Training samples are randomly masked to simulate adaptive (partial) responses, teaching the model to handle missing items
- **Hyperparameters:** n_estimators={n_est}, max_depth={max_d}, learning_rate={lr_str}
- **Cross-validation:** 3-fold cross-validation robustness analysis with evaluation split before augmentation

## Performance

Evaluated on held-out test respondents:

{perf_table}

{coverage_line}

{ml_advantage_line}

## Norms

Population norms for raw-score -> percentile conversion (from OSPP dataset):

{norms_table}

## Limitations

- Norms are derived from self-selected online respondents (OSPP); they may not represent the general population
- Models are trained on English-language IPIP items only
- Standalone Python/TypeScript inference expects reverse-keyed items to be preprocessed before scoring; the web app applies that transform server-side
- Exported calibration regimes are `full_50` and `sparse_20_balanced`; arbitrary sub-50 response patterns use the sparse regime as a fallback rather than a separately fit calibration curve
- Accuracy degrades with fewer items; 20 items is the recommended minimum for reliable scoring
- Not intended for clinical diagnosis or high-stakes selection decisions

## Item Source

The IPIP-BFFM items are from the [International Personality Item Pool](https://ipip.ori.org/) and are in the **public domain**.

## License

CC0 1.0 Universal -- Public Domain Dedication
"""
    return readme


# ---------------------------------------------------------------------------
# Repo-level README (multi-variant)
# ---------------------------------------------------------------------------


def generate_repo_readme(variants: list[tuple[str, Path]]) -> str:
    """Generate a top-level README listing all variants.

    Args:
        variants: list of (variant_name, variant_path) tuples.
    """
    variant_names = [name for name, _ in variants if name]
    primary = "reference" if "reference" in variant_names else (variant_names[0] if variant_names else "unknown")

    variant_row_data = []
    for name in variant_names:
        if name == primary:
            variant_row_data.append([f"`{name}`", "Primary published model"])
        else:
            variant_row_data.append([f"`{name}`", "Research ablation variant"])
    variant_table = _format_md_table(["Variant", "Description"], variant_row_data)

    lines = [
        "---",
        "license: cc0-1.0",
        "language: en",
        "tags:",
        "  - personality",
        "  - psychometrics",
        "  - big-five",
        "  - ipip",
        "  - xgboost",
        "  - onnx",
        "  - quantile-regression",
        "library_name: onnxruntime",
        "pipeline_tag: tabular-regression",
        "---",
        "",
        "# IPIP-BFFM Sparse Quantile Models",
        "",
        "XGBoost quantile regression models for the 50-item "
        "[IPIP Big-Five Factor Markers](https://ipip.ori.org/newBigFive5broadKey.htm) "
        "(BFFM) personality assessment, exported as ONNX for cross-platform inference.",
        "",
        "## What These Models Do",
        "",
        "Each model takes up to 50 item responses (Likert 1--5) and predicts Big Five "
        "domain scores (Extraversion, Agreeableness, Conscientiousness, Emotional "
        "Stability, Intellect). The exported calibration regimes are fit for full "
        "50-item completion and the primary domain-balanced 20-item sparse regime.",
        "",
        "**Key capability: sparse input.** The models produce accurate predictions even "
        "when most items are unanswered (NaN). This allows adaptive and short-form "
        "assessments (as few as 20 items) without retraining or switching models.",
        "",
        "## How It Works",
        "",
        "- **15 models in one graph** -- 5 domains x 3 quantiles (q05, q50, q95), "
        "merged into a single ONNX file",
        "- **Sparsity augmentation** -- during training, complete responses are randomly "
        "masked to simulate missing items, teaching the model to handle arbitrary "
        "missing-item patterns",
        "- **Quantile regression** -- pinball loss at tau = 0.05, 0.50, 0.95 provides "
        "median predictions with uncertainty bounds that are explicitly calibrated "
        "for full_50 and sparse_20_balanced runtime regimes",
        "- **Norms-based percentiles** -- raw predictions are converted to population "
        "percentiles using z-score norms derived from ~603k respondents",
        "",
        "## Variants",
        "",
        variant_table,
        "",
        f"The primary model is **`{primary}`**. Other variants are research ablations "
        "that isolate the contribution of each sparsity augmentation strategy.",
        "",
        "Each variant directory contains:",
        "- `model.onnx` -- merged ONNX model (5 domains x 3 quantiles)",
        "- `config.json` -- runtime configuration, feature names, and norms",
        "- `README.md` -- variant-specific model card with performance tables",
        "- `provenance.json` -- full audit trail (git hash, data snapshot, training config)",
        "",
        "## Source Code",
        "",
        "Training pipeline, evaluation scripts, and inference packages (Python + TypeScript): "
        "[github.com/sprice/bffm-xgb](https://github.com/sprice/bffm-xgb)",
        "",
        "## License",
        "",
        "CC0 1.0 Universal -- Public Domain Dedication",
        "",
    ]
    return "\n".join(lines)


def _discover_variants(output_dir: Path) -> list[tuple[str, Path]]:
    """Scan output_dir for variant subdirectories containing config.json."""
    if not output_dir.is_dir():
        return []
    variants = []
    for child in sorted(output_dir.iterdir()):
        if child.is_dir() and (child / "config.json").is_file():
            variants.append((child.name, child))
    return variants


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export IPIP-BFFM XGBoost models to ONNX format"
    )
    parser.add_argument(
        "--repo-readme",
        action="store_true",
        help="Generate repo-level README.md from variant subdirectories in --output-dir, then exit.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PACKAGE_ROOT / "models" / "reference",
        help="Path to directory containing .joblib models (default: models/reference/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "output",
        help="Output directory for ONNX files (default: output/)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=False,
        default=None,
        help="Data directory for provenance verification",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Artifacts directory containing validation/simulation outputs (default: artifacts)",
    )
    add_provenance_args(parser)
    args = parser.parse_args()

    # --- repo-readme mode: scan output dir and write top-level README ---
    if args.repo_readme:
        output_dir = args.output_dir
        if not output_dir.is_absolute():
            output_dir = PACKAGE_ROOT / output_dir
        variants = _discover_variants(output_dir)
        if not variants:
            log.error("No variant subdirectories found in %s", output_dir)
            return 1
        readme = generate_repo_readme(variants)
        readme_path = output_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme)
        log.info("Wrote repo-level README.md (%d variants) to %s", len(variants), readme_path)
        return 0

    # --- normal export mode: require --data-dir ---
    if args.data_dir is None:
        parser.error("--data-dir is required for export (omit only with --repo-readme)")

    models_dir = args.model_dir
    if not models_dir.is_absolute():
        models_dir = PACKAGE_ROOT / models_dir
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PACKAGE_ROOT / output_dir
    artifacts_dir = (
        args.artifacts_dir
        if args.artifacts_dir.is_absolute()
        else PACKAGE_ROOT / args.artifacts_dir
    )
    data_dir = args.data_dir if args.data_dir.is_absolute() else PACKAGE_ROOT / args.data_dir
    try:
        norms_map = load_norms()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        log.error("Failed to load norms: %s", e)
        return 1
    variant_name = models_dir.name
    log.info("=" * 60)
    log.info("IPIP-BFFM ONNX Export")
    log.info("=" * 60)
    log.info("Variant: %s", variant_name)
    log.info("Data dir: %s", data_dir)
    log.info("Artifacts dir: %s", artifacts_dir)

    # 1. Load joblib models
    log.info("")
    log.info("1. Loading joblib models...")
    joblib_models = load_joblib_models(models_dir)

    # 2. Convert to ONNX (individual)
    log.info("")
    log.info("2. Converting to ONNX...")
    onnx_models = convert_to_onnx(joblib_models)

    # 3. Validate parity (joblib vs individual ONNX)
    log.info("")
    log.info("3. Validating numerical parity (joblib vs ONNX)...")
    validate_parity(joblib_models, onnx_models)

    # 4. Merge into single model
    log.info("")
    log.info("4. Merging into single ONNX model...")
    merged_model = merge_onnx_models(onnx_models)

    # 5. Validate merged parity
    log.info("")
    log.info("5. Validating merged model parity...")
    validate_merged_parity(merged_model, onnx_models)

    # 6. Save merged ONNX file
    log.info("")
    log.info("6. Saving merged ONNX file...")
    save_onnx_file(merged_model, output_dir)

    # 7. Generate and save config.json
    log.info("")
    log.info("7. Generating config.json...")
    provenance_dict = build_provenance(Path(__file__).name, args=args)
    config = generate_config(models_dir, artifacts_dir, provenance_dict, norms_map, variant_name=variant_name)
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    log.info("  Saved %s", config_path.name)

    # 7b. Generate provenance.json (full audit trail)
    log.info("")
    log.info("7b. Generating provenance.json...")
    prov_doc = generate_provenance_document(
        models_dir=models_dir,
        output_dir=output_dir,
        provenance_dict=provenance_dict,
        variant_name=variant_name,
    )
    prov_path = output_dir / "provenance.json"
    with open(prov_path, "w") as f:
        json.dump(prov_doc, f, indent=2)
    log.info("  Saved %s", prov_path.name)

    # 8. Generate README.md
    log.info("")
    log.info("8. Generating README.md...")
    try:
        readme = generate_readme(config, artifacts_dir, models_dir, variant_name=variant_name)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        log.error("README generation failed strict provenance checks: %s", e)
        return 1
    readme_path = output_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme)
    log.info("  Saved %s", readme_path.name)

    # Summary
    log.info("")
    log.info("=" * 60)
    log.info("Export complete!")
    log.info("Output: %s", output_dir)
    files = sorted(output_dir.iterdir())
    total_kb = sum(f.stat().st_size for f in files if f.is_file()) / 1024
    log.info("Files: %d (%.0f KB total)", len(files), total_kb)
    for f in files:
        if f.is_file():
            log.info("  %s (%.0f KB)", f.name, f.stat().st_size / 1024)
    log.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
