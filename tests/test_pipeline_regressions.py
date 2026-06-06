"""Regression tests for critical pipeline behaviors in stages 05/06/11."""

from __future__ import annotations

import argparse
import importlib.util
import hashlib
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lib.constants import DEFAULT_STAGE07_CV_FOLDS, DOMAINS, ITEM_COLUMNS
from lib.item_info import file_sha256
from lib.provenance_checks import build_split_signature as _build_split_signature


_MODULE_COUNTER = 0


def _load_pipeline_module(script_name: str):
    """Load a pipeline script module by filename (e.g., '06_tune.py')."""
    global _MODULE_COUNTER
    _MODULE_COUNTER += 1
    module_name = f"test_{script_name.replace('.', '_')}_{_MODULE_COUNTER}"
    module_path = Path(__file__).resolve().parent.parent / "pipeline" / script_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_paper_module(script_name: str):
    """Load a scripts utility module by filename (e.g., 'generate_notes_data.py')."""
    global _MODULE_COUNTER
    _MODULE_COUNTER += 1
    module_name = f"test_paper_{script_name.replace('.', '_')}_{_MODULE_COUNTER}"
    module_path = Path(__file__).resolve().parent.parent / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_dataset(n_rows: int = 48) -> pd.DataFrame:
    """Create synthetic frame with all item/score/percentile columns."""
    data: dict[str, Any] = {}
    for idx, col in enumerate(ITEM_COLUMNS):
        data[col] = np.full(n_rows, float((idx % 5) + 1))

    base = np.linspace(1.0, 5.0, n_rows)
    pcts = np.linspace(2.5, 97.5, n_rows)
    for offset, domain in enumerate(DOMAINS):
        data[f"{domain}_score"] = np.roll(base, offset)
        data[f"{domain}_percentile"] = np.roll(pcts, offset)

    return pd.DataFrame(data)


def _write_item_info(path: Path, *, source_sha256: str | None = None) -> None:
    """Write a valid camelCase item_info.json payload."""
    item_pool = []
    for rank, col in enumerate(ITEM_COLUMNS, start=1):
        item_pool.append(
            {
                "id": col,
                "homeDomain": col[:3],
                "ownDomainR": 0.5,
                "crossDomainInfo": 0.1,
                "domainCorrelations": {domain: 0.1 for domain in DOMAINS},
                "isReverseKeyed": False,
                "rank": rank,
            }
        )
    payload = {
        "firstItemId": ITEM_COLUMNS[0],
        "firstItemText": "test",
        "firstItemDomain": ITEM_COLUMNS[0][:3],
        "itemPool": item_pool,
        "interItemRBar": {domain: 0.3 for domain in DOMAINS},
    }
    if source_sha256 is None:
        train_path = path.parent / "train.parquet"
        if train_path.exists():
            source_sha256 = file_sha256(train_path)
    if source_sha256 is not None:
        payload["provenance"] = {"source_sha256": source_sha256}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _write_training_report(
    path: Path,
    *,
    item_info_sha256: str | None = None,
    hyperparameters: dict[str, Any] | None = None,
    data_overrides: dict[str, Any] | None = None,
) -> None:
    """Write a minimal training_report.json payload."""
    data: dict[str, Any] = {
        "train_rows": 100,
        "train_rows_after_augmentation": 300,
        "n_test": 0,
    }
    if item_info_sha256 is not None:
        data["item_info_sha256"] = item_info_sha256
    if data_overrides:
        data.update(data_overrides)

    payload = {
        "provenance": {"script": "07_train.py", "git_hash": "deadbeef"},
        "config": {"hyperparameters": hyperparameters or {}},
        "data": data,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _per_domain_row(
    n_items: int,
    method: str,
    *,
    r_value: float = 0.9,
    n_items_domain: int = 4,
) -> dict[str, Any]:
    """Build one per-domain CSV row compatible with stage-11 figure loaders."""
    return {
        "n_items": n_items,
        "method": method,
        "r_Extraversion": r_value,
        "r_Agreeableness": r_value,
        "r_Conscientiousness": r_value,
        "r_EmotionalStability": r_value,
        "r_Intellect": r_value,
        "items_Extraversion": n_items_domain,
        "items_Agreeableness": n_items_domain,
        "items_Conscientiousness": n_items_domain,
        "items_EmotionalStability": n_items_domain,
        "items_Intellect": n_items_domain,
    }


def _make_eval_metrics(r: float, coverage: float) -> dict[str, dict[str, float]]:
    """Create minimal metric payload expected by 06_train quality gates."""
    by_domain = {
        domain: {
            "pearson_r": r,
            "mae": 4.0,
            "rmse": 5.0,
            "coverage_90": coverage,
            "within_5_pct": 0.7,
            "within_10_pct": 0.9,
        }
        for domain in DOMAINS
    }
    by_domain["overall"] = {
        "pearson_r": r,
        "mae": 4.0,
        "rmse": 5.0,
        "coverage_90": coverage,
        "within_5_pct": 0.7,
        "within_10_pct": 0.9,
    }
    return by_domain


class _DummyModel:
    """Simple predict-only model object for pipeline stubs."""

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value, dtype=np.float64)


def _dummy_domain_models() -> dict[str, dict[str, _DummyModel]]:
    """Return placeholder q05/q50/q95 models for each domain."""
    return {
        domain: {
            "q05": _DummyModel(2.5),
            "q50": _DummyModel(3.0),
            "q95": _DummyModel(3.5),
        }
        for domain in DOMAINS
    }


def test_tune_safe_pearson_floors_nonfinite() -> None:
    tune = _load_pipeline_module("06_tune.py")

    y_true = np.ones(12, dtype=np.float64)
    y_pred = np.ones(12, dtype=np.float64)
    assert tune._safe_pearson(y_true, y_pred) == -1.0

    y_linear = np.linspace(0.0, 1.0, 12)
    assert tune._safe_pearson(y_linear, y_linear) > 0.99


def test_tune_prepare_features_targets_requires_full_big5_schema() -> None:
    tune = _load_pipeline_module("06_tune.py")
    frame = _make_dataset()
    frame = frame.drop(columns=["opn_percentile"])

    try:
        tune._prepare_features_targets(frame)
        raise AssertionError("Expected ValueError for missing domain percentile column")
    except ValueError as exc:
        assert "full big-5 schema" in str(exc).lower()


def test_tune_loads_item_info_for_sparse20_even_when_sparsity_disabled(
    tmp_path,
    monkeypatch,
) -> None:
    tune = _load_pipeline_module("06_tune.py")
    monkeypatch.setattr(tune, "PACKAGE_ROOT", tmp_path)

    data_dir = tmp_path / "data" / "processed"
    artifacts_dir = tmp_path / "artifacts"
    data_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)

    frame = _make_dataset()
    frame.to_parquet(data_dir / "train.parquet", index=False)
    frame.to_parquet(data_dir / "val.parquet", index=False)
    _write_item_info(data_dir / "item_info.json")

    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: test\ndata_dir: data/processed\nsparsity:\n  enabled: false\n", encoding="utf-8")

    captured: dict[str, Any] = {}

    def _fake_run_optuna_tuning(
        X_train,
        y_train,
        X_val,
        y_val,
        y_val_pct,
        n_trials,
        item_info,
        config,
        mini_ipip_items=None,
        parallel_trials=1,
        gpu=False,
    ):
        captured["item_info"] = item_info
        captured["config"] = config
        return tune.DEFAULT_PARAMS.copy()

    monkeypatch.setattr(tune, "_run_optuna_tuning", _fake_run_optuna_tuning)

    monkeypatch.setattr(
        sys,
        "argv",
        ["06_tune.py", "--trials", "1", "--config", str(config_path)],
    )

    rc = tune.main()
    assert rc == 0
    assert captured["item_info"].get("item_pool"), "Expected item_info to be loaded"
    assert captured["config"]["sparsity"]["enabled"] is False


def test_tune_item_info_requires_primary_stage05_file(
    tmp_path,
    monkeypatch,
) -> None:
    tune = _load_pipeline_module("06_tune.py")
    monkeypatch.setattr(tune, "PACKAGE_ROOT", tmp_path)

    data_dir = tmp_path / "data" / "processed"
    output_dir = tmp_path / "output"
    data_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    _write_item_info(output_dir / "item_info.json")

    try:
        tune._load_item_info(data_dir)
        raise AssertionError("Expected FileNotFoundError when primary item_info is missing")
    except FileNotFoundError as exc:
        assert "stage 05" in str(exc).lower() or "make correlations" in str(exc).lower()


def test_tune_main_aborts_when_sparse20_objective_inputs_missing(
    tmp_path,
    monkeypatch,
) -> None:
    tune = _load_pipeline_module("06_tune.py")
    monkeypatch.setattr(tune, "PACKAGE_ROOT", tmp_path)

    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)

    frame = _make_dataset()
    frame.to_parquet(data_dir / "train.parquet", index=False)
    frame.to_parquet(data_dir / "val.parquet", index=False)

    # No item_info in data/processed or output -> should hard fail by default.
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: test\ndata_dir: data/processed\nsparsity:\n  enabled: true\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        ["06_tune.py", "--trials", "1", "--config", str(config_path)],
    )
    rc = tune.main()
    assert rc == 1


def test_tune_main_aborts_on_malformed_item_info_even_with_full50_override(
    tmp_path,
    monkeypatch,
) -> None:
    tune = _load_pipeline_module("06_tune.py")
    monkeypatch.setattr(tune, "PACKAGE_ROOT", tmp_path)

    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)

    frame = _make_dataset()
    frame.to_parquet(data_dir / "train.parquet", index=False)
    frame.to_parquet(data_dir / "val.parquet", index=False)
    with open(data_dir / "item_info.json", "w") as f:
        json.dump({"itemPool": []}, f)

    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: test\ndata_dir: data/processed\nsparsity:\n  enabled: false\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "06_tune.py",
            "--trials",
            "1",
            "--config",
            str(config_path),
            "--allow-no-sparse20-objective",
        ],
    )
    rc = tune.main()
    assert rc == 1


def test_train_sparse_gate_defaults_to_disabled_without_config_block(
    tmp_path,
    monkeypatch,
) -> None:
    train = _load_pipeline_module("07_train.py")
    monkeypatch.setattr(train, "PACKAGE_ROOT", tmp_path)

    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    frame = _make_dataset()
    frame.to_parquet(data_dir / "train.parquet", index=False)
    frame.to_parquet(data_dir / "val.parquet", index=False)

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "name: unit",
                "output_dir: models/unit",
                "sparsity:",
                "  enabled: false",
                "training:",
                "  cv_folds: 0",
                "  random_state: 42",
                "validation:",
                "  min_pearson_r: 0.0",
                "  min_coverage_90: 0.0",
            ]
        ),
        encoding="utf-8",
    )

    eval_calls: list[int] = []

    monkeypatch.setattr(
        train,
        "_load_item_info",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not load item_info")),
    )
    monkeypatch.setattr(train, "_load_mini_ipip_mapping", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train, "_train_domain_models", lambda *_args, **_kwargs: _dummy_domain_models())
    monkeypatch.setattr(train, "_validate_model_outputs", lambda *_args, **_kwargs: {"ok": {"passed": True}})

    def _fake_eval(*_args, **_kwargs):
        eval_calls.append(1)
        return _make_eval_metrics(r=0.92, coverage=0.9)

    monkeypatch.setattr(train, "_evaluate_domain_models", _fake_eval)
    monkeypatch.setattr(
        train,
        "_compute_calibration_params",
        lambda *_args, **_kwargs: {
            domain: {"observed_coverage": 0.9, "scale_factor": 1.0}
            for domain in DOMAINS
        },
    )
    monkeypatch.setattr(train.joblib, "dump", lambda *_args, **_kwargs: None)

    monkeypatch.setattr(sys, "argv", ["07_train.py", "--config", str(cfg_path)])
    rc = train.main()
    assert rc == 0
    assert len(eval_calls) == 1, "Expected only full-validation evaluation call"

    report_path = tmp_path / "models" / "unit" / "training_report.json"
    with open(report_path) as f:
        report = json.load(f)
    assert report["validation_metrics_sparse_20"] == {}
    assert report["validation_metrics_sparse_20_runs"] == []
    # Default path: gate passes and is enforced -> honest record on the happy path.
    assert report["quality_gates"] == {"passed": True, "enforced": True}


def test_train_no_gate_saves_despite_failing_quality_gate(
    tmp_path,
    monkeypatch,
) -> None:
    """By default a failing quality gate aborts before saving anything; with
    --no-gate the run still saves and honestly records the (failed, unenforced)
    gate outcome in training_report.json — so a multi-day run is not discarded on a
    near-miss, but a saved-despite-failure bundle is distinguishable from a clean one."""
    train = _load_pipeline_module("07_train.py")
    monkeypatch.setattr(train, "PACKAGE_ROOT", tmp_path)

    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    frame = _make_dataset()
    frame.to_parquet(data_dir / "train.parquet", index=False)
    frame.to_parquet(data_dir / "val.parquet", index=False)

    cfg_path = tmp_path / "cfg_gate.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "name: unit_gate",
                "output_dir: models/unit_gate",
                "sparsity:",
                "  enabled: false",
                "training:",
                "  cv_folds: 0",
                "  random_state: 42",
                "validation:",
                "  min_pearson_r: 0.99",  # achieved overall r=0.92 -> gate fails
                "  min_coverage_90: 0.0",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        train,
        "_load_item_info",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not load item_info")),
    )
    monkeypatch.setattr(train, "_load_mini_ipip_mapping", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train, "_train_domain_models", lambda *_args, **_kwargs: _dummy_domain_models())
    monkeypatch.setattr(train, "_validate_model_outputs", lambda *_args, **_kwargs: {"ok": {"passed": True}})
    monkeypatch.setattr(
        train,
        "_evaluate_domain_models",
        lambda *_args, **_kwargs: _make_eval_metrics(r=0.92, coverage=0.9),
    )
    monkeypatch.setattr(
        train,
        "_compute_calibration_params",
        lambda *_args, **_kwargs: {
            domain: {"observed_coverage": 0.9, "scale_factor": 1.0} for domain in DOMAINS
        },
    )
    monkeypatch.setattr(train.joblib, "dump", lambda *_args, **_kwargs: None)

    report_path = tmp_path / "models" / "unit_gate" / "training_report.json"

    # Default: a failing gate aborts before saving — no report written.
    monkeypatch.setattr(sys, "argv", ["07_train.py", "--config", str(cfg_path)])
    assert train.main() == 1
    assert not report_path.exists(), "Failing gate must not write a training report by default"

    # --no-gate: the run saves and records the failed/unenforced gate outcome.
    monkeypatch.setattr(sys, "argv", ["07_train.py", "--config", str(cfg_path), "--no-gate"])
    assert train.main() == 0
    with open(report_path) as f:
        report = json.load(f)
    assert report["quality_gates"] == {"passed": False, "enforced": False}


def test_train_prepare_features_targets_requires_full_big5_schema() -> None:
    train = _load_pipeline_module("07_train.py")
    frame = _make_dataset()
    frame = frame.drop(columns=["opn_score"])

    try:
        train._prepare_features_targets(frame)
        raise AssertionError("Expected ValueError for missing domain score column")
    except ValueError as exc:
        assert "full big-5 schema" in str(exc).lower()


def test_train_main_fails_closed_on_split_metadata_hash_mismatch(
    tmp_path,
    monkeypatch,
) -> None:
    train = _load_pipeline_module("07_train.py")
    monkeypatch.setattr(train, "PACKAGE_ROOT", tmp_path)

    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    frame = _make_dataset(n_rows=24)
    frame.iloc[:14].to_parquet(data_dir / "train.parquet", index=False)
    frame.iloc[14:19].to_parquet(data_dir / "val.parquet", index=False)
    frame.iloc[19:].to_parquet(data_dir / "test.parquet", index=False)

    with open(data_dir / "split_metadata.json", "w") as f:
        json.dump(
            {
                "train_sha256": "0" * 64,
                "val_sha256": file_sha256(data_dir / "val.parquet"),
                "test_sha256": file_sha256(data_dir / "test.parquet"),
            },
            f,
            indent=2,
        )

    cfg_path = tmp_path / "cfg_split_mismatch.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "name: unit_split_mismatch",
                "output_dir: models/unit_split_mismatch",
                "sparsity:",
                "  enabled: false",
                "training:",
                "  cv_folds: 0",
                "  random_state: 42",
                "validation:",
                "  min_pearson_r: 0.0",
                "  min_coverage_90: 0.0",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["07_train.py", "--config", str(cfg_path)])
    rc = train.main()
    assert rc == 1


def test_train_sparse_gate_uses_multiple_masks_and_aggregates(
    tmp_path,
    monkeypatch,
) -> None:
    train = _load_pipeline_module("07_train.py")
    monkeypatch.setattr(train, "PACKAGE_ROOT", tmp_path)

    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    frame = _make_dataset()
    frame.to_parquet(data_dir / "train.parquet", index=False)
    frame.to_parquet(data_dir / "val.parquet", index=False)
    _write_item_info(data_dir / "item_info.json")

    cfg_path = tmp_path / "cfg_sparse.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "name: unit_sparse",
                "output_dir: models/unit_sparse",
                "sparsity:",
                "  enabled: false",
                "training:",
                "  cv_folds: 0",
                "  random_state: 42",
                "validation:",
                "  min_pearson_r: 0.0",
                "  min_coverage_90: 0.0",
                "  sparse_20:",
                "    enabled: true",
                "    n_masks: 3",
                "    min_pearson_r: 0.0",
                "    min_coverage_90: 0.0",
            ]
        ),
        encoding="utf-8",
    )

    eval_metrics = [
        _make_eval_metrics(r=0.95, coverage=0.90),  # full validation
        _make_eval_metrics(r=0.80, coverage=0.80),  # sparse mask 1
        _make_eval_metrics(r=0.85, coverage=0.82),  # sparse mask 2
        _make_eval_metrics(r=0.90, coverage=0.84),  # sparse mask 3
    ]
    eval_calls: list[int] = []

    monkeypatch.setattr(train, "_load_mini_ipip_mapping", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train, "apply_sparsity_single", lambda X, *_args, **_kwargs: X)
    monkeypatch.setattr(train, "_train_domain_models", lambda *_args, **_kwargs: _dummy_domain_models())
    monkeypatch.setattr(train, "_validate_model_outputs", lambda *_args, **_kwargs: {"ok": {"passed": True}})

    def _fake_eval(*_args, **_kwargs):
        idx = len(eval_calls)
        eval_calls.append(idx)
        return eval_metrics[idx]

    monkeypatch.setattr(train, "_evaluate_domain_models", _fake_eval)
    monkeypatch.setattr(
        train,
        "_compute_calibration_params",
        lambda *_args, **_kwargs: {
            domain: {"observed_coverage": 0.9, "scale_factor": 1.0}
            for domain in DOMAINS
        },
    )
    monkeypatch.setattr(train.joblib, "dump", lambda *_args, **_kwargs: None)

    monkeypatch.setattr(sys, "argv", ["07_train.py", "--config", str(cfg_path)])
    rc = train.main()
    assert rc == 0
    assert len(eval_calls) == 4, "Expected full + 3 sparse-mask evaluations"

    report_path = tmp_path / "models" / "unit_sparse" / "training_report.json"
    with open(report_path) as f:
        report = json.load(f)
    sparse = report["validation_metrics_sparse_20"]["overall"]
    assert abs(sparse["pearson_r"] - 0.85) < 1e-9
    assert abs(sparse["coverage_90"] - 0.82) < 1e-9
    assert len(report["validation_metrics_sparse_20_runs"]) == 3
    assert report.get("data", {}).get("item_info_sha256") == file_sha256(data_dir / "item_info.json")

    model_dir = tmp_path / "models" / "unit_sparse"
    full_cal_path = model_dir / "calibration_params_full_50.json"
    sparse_cal_path = model_dir / "calibration_params_sparse_20_balanced.json"
    legacy_cal_path = model_dir / "calibration_params.json"
    assert full_cal_path.exists()
    assert sparse_cal_path.exists()
    assert legacy_cal_path.exists()

    with open(sparse_cal_path) as f:
        sparse_cal = json.load(f)
    with open(legacy_cal_path) as f:
        legacy_cal = json.load(f)
    assert legacy_cal == sparse_cal


def test_train_sparse_gate_fails_closed_on_invalid_mask_metrics(
    tmp_path,
    monkeypatch,
) -> None:
    train = _load_pipeline_module("07_train.py")
    monkeypatch.setattr(train, "PACKAGE_ROOT", tmp_path)

    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    frame = _make_dataset()
    frame.to_parquet(data_dir / "train.parquet", index=False)
    frame.to_parquet(data_dir / "val.parquet", index=False)
    _write_item_info(data_dir / "item_info.json")

    cfg_path = tmp_path / "cfg_sparse_invalid.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "name: unit_sparse_invalid",
                "output_dir: models/unit_sparse_invalid",
                "sparsity:",
                "  enabled: false",
                "training:",
                "  cv_folds: 0",
                "  random_state: 42",
                "validation:",
                "  min_pearson_r: 0.0",
                "  min_coverage_90: 0.0",
                "  sparse_20:",
                "    enabled: true",
                "    n_masks: 2",
                "    min_pearson_r: 0.0",
                "    min_coverage_90: 0.0",
            ]
        ),
        encoding="utf-8",
    )

    invalid_sparse = _make_eval_metrics(r=0.85, coverage=0.82)
    invalid_sparse["overall"]["pearson_r"] = float("nan")

    eval_metrics = [
        _make_eval_metrics(r=0.95, coverage=0.90),  # full validation
        invalid_sparse,  # sparse mask 1 invalid
        _make_eval_metrics(r=0.90, coverage=0.84),  # sparse mask 2 valid
    ]
    eval_calls: list[int] = []

    monkeypatch.setattr(train, "_load_mini_ipip_mapping", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train, "apply_sparsity_single", lambda X, *_args, **_kwargs: X)
    monkeypatch.setattr(train, "_train_domain_models", lambda *_args, **_kwargs: _dummy_domain_models())
    monkeypatch.setattr(train, "_validate_model_outputs", lambda *_args, **_kwargs: {"ok": {"passed": True}})

    def _fake_eval(*_args, **_kwargs):
        idx = len(eval_calls)
        eval_calls.append(idx)
        return eval_metrics[idx]

    monkeypatch.setattr(train, "_evaluate_domain_models", _fake_eval)
    monkeypatch.setattr(
        train,
        "_compute_calibration_params",
        lambda *_args, **_kwargs: {
            domain: {"observed_coverage": 0.9, "scale_factor": 1.0}
            for domain in DOMAINS
        },
    )
    monkeypatch.setattr(train.joblib, "dump", lambda *_args, **_kwargs: None)

    monkeypatch.setattr(sys, "argv", ["07_train.py", "--config", str(cfg_path)])
    rc = train.main()
    assert rc == 1, "Expected training to fail when any sparse mask yields invalid metrics"
    assert len(eval_calls) == 3


def test_aggregate_metric_runs_marks_nan_when_section_missing() -> None:
    train = _load_pipeline_module("07_train.py")
    runs = [
        {"overall": {"pearson_r": 0.90, "coverage_90": 0.85}},
        {},
    ]
    aggregated = train._aggregate_metric_runs(runs)
    assert np.isnan(aggregated["overall"]["pearson_r"])
    assert np.isnan(aggregated["overall"]["coverage_90"])


def test_validate_prepare_features_targets_requires_full_big5_schema() -> None:
    validate = _load_pipeline_module("08_validate.py")
    frame = _make_dataset()
    frame = frame.drop(columns=["opn_percentile"])

    try:
        validate._prepare_features_targets(frame)
        raise AssertionError("Expected ValueError for missing domain percentile column")
    except ValueError as exc:
        assert "full big-5 schema" in str(exc).lower()


def test_validate_calibration_loader_fails_closed_for_partial_payload(tmp_path) -> None:
    validate = _load_pipeline_module("08_validate.py")
    path = tmp_path / "calibration_params_sparse_20_balanced.json"
    with open(path, "w") as f:
        json.dump(
            {"ext": {"scale_factor": 1.1, "observed_coverage": 0.9}},
            f,
            indent=2,
        )

    with pytest.raises(ValueError) as exc_info:
        validate._load_calibration_params(path)

    assert "domain coverage" in str(exc_info.value).lower()


def test_validate_percentile_metric_fn_uses_nan_for_degenerate_pearson() -> None:
    validate = _load_pipeline_module("08_validate.py")
    y_true = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    y_pred = np.array([50.0, 50.0, 50.0, 50.0], dtype=np.float64)
    y_lower = np.array([45.0, 45.0, 45.0, 45.0], dtype=np.float64)
    y_upper = np.array([55.0, 55.0, 55.0, 55.0], dtype=np.float64)

    metrics = validate._percentile_metric_fn(y_true, y_pred, y_lower, y_upper)
    assert np.isnan(metrics["pearson_r"])
    assert np.isfinite(metrics["mae"])


def test_validate_domain_metrics_reports_raw_crossing_and_no_fake_zero() -> None:
    """A4.3: raw_crossing_rate is the headline metric; the misleading constant
    quantile_crossing_rate=0.0 is gone, replaced by monotonized_crossing_rate."""
    validate = _load_pipeline_module("08_validate.py")
    rng = np.random.default_rng(0)
    true = rng.uniform(0, 100, size=400)
    pred = true + rng.normal(0, 5, size=400)
    pred = np.clip(pred, 0, 100)
    per_domain = {
        "ext": {
            "true": true,
            "pred": pred,
            "lower": np.clip(pred - 10, 0, 100),
            "upper": np.clip(pred + 10, 0, 100),
            "raw_crossing_rate": 0.25,
        }
    }
    metrics = validate._compute_domain_metrics(per_domain)
    ext = metrics["ext"]
    assert ext["raw_crossing_rate"] == 0.25
    assert ext["monotonized_crossing_rate"] == 0.0
    assert "quantile_crossing_rate" not in ext  # guard against the misleading key
    # A4.4: tail vs central coverage surfaced per-domain and overall.
    assert "coverage_central" in ext and "coverage_tail" in ext
    overall = metrics["overall"]
    assert overall["raw_crossing_rate"] == 0.25  # mean over the one domain
    assert overall["monotonized_crossing_rate"] == 0.0
    assert "coverage_central" in overall and "coverage_tail" in overall


def test_figures_include_worst_k_with_distinct_color() -> None:
    figures = _load_pipeline_module("12_generate_figures.py")

    assert "worst_k" in figures.STRATEGY_ORDER
    assert figures.COLORS["worst_k"] != figures.COLORS["greedy_balanced"]


def test_baselines_write_per_domain_csv_metadata_sidecar(tmp_path) -> None:
    baselines = _load_pipeline_module("09_baselines.py")
    df = pd.DataFrame(
        [
            _per_domain_row(20, "domain_balanced", n_items_domain=4),
            _per_domain_row(20, "adaptive_topk", n_items_domain=0),
        ]
    )
    csv_path = tmp_path / "baseline_comparison_per_domain.csv"
    meta_path = tmp_path / "baseline_comparison_per_domain.meta.json"
    provenance = {"model_dir": str(tmp_path / "models" / "reference")}

    baselines._write_per_domain_csv_with_metadata(df, csv_path, meta_path, provenance)

    with open(meta_path) as f:
        meta = json.load(f)
    assert meta["provenance"]["model_dir"] == provenance["model_dir"]
    assert meta["artifact"]["file"] == csv_path.name
    assert meta["artifact"]["sha256"] == file_sha256(csv_path)
    assert meta["artifact"]["n_rows"] == len(df)
    assert meta["artifact"]["n_columns"] == len(df.columns)


def test_figures_per_domain_csv_requires_metadata_sidecar(tmp_path) -> None:
    figures = _load_pipeline_module("12_generate_figures.py")
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    pd.DataFrame([_per_domain_row(20, "domain_balanced")]).to_csv(
        artifacts_dir / "baseline_comparison_per_domain.csv",
        index=False,
    )

    try:
        figures.load_per_domain_csv(artifacts_dir)
        raise AssertionError("Expected missing metadata sidecar to fail closed")
    except FileNotFoundError as exc:
        assert "metadata" in str(exc).lower()


def test_figures_per_domain_csv_sha_mismatch_fails_closed(tmp_path) -> None:
    figures = _load_pipeline_module("12_generate_figures.py")
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    csv_path = artifacts_dir / "baseline_comparison_per_domain.csv"
    pd.DataFrame([_per_domain_row(20, "domain_balanced")]).to_csv(csv_path, index=False)

    with open(artifacts_dir / "baseline_comparison_per_domain.meta.json", "w") as f:
        json.dump(
            {
                "provenance": {"model_dir": str(tmp_path / "models" / "reference")},
                "artifact": {
                    "file": csv_path.name,
                    "sha256": "0" * 64,
                    "n_rows": 1,
                    "n_columns": 12,
                },
            },
            f,
            indent=2,
        )

    try:
        figures.load_per_domain_csv(artifacts_dir)
        raise AssertionError("Expected SHA mismatch to fail closed")
    except ValueError as exc:
        assert "sha-256 mismatch" in str(exc).lower()


def test_figures_fail_when_model_dir_provenance_mismatches() -> None:
    figures = _load_pipeline_module("12_generate_figures.py")
    model_dirs = {
        "baseline_comparison_results.json": Path("/tmp/a"),
        "baseline_comparison_per_domain.meta.json": Path("/tmp/b"),
        "ml_vs_averaging_comparison.json": Path("/tmp/a"),
    }
    try:
        figures._assert_common_model_dir(model_dirs)
        raise AssertionError("Expected provenance mismatch to fail closed")
    except ValueError as exc:
        assert "provenance mismatch" in str(exc).lower()


def test_figures_fail_when_run_provenance_signatures_mismatch() -> None:
    figures = _load_pipeline_module("12_generate_figures.py")
    signatures = {
        "baseline_comparison_results.json": "sig-a",
        "baseline_comparison_per_domain.meta.json": "sig-a",
        "ml_vs_averaging_comparison.json": "sig-b",
    }
    try:
        figures._assert_common_run_signature(signatures)
        raise AssertionError("Expected run-signature mismatch to fail closed")
    except ValueError as exc:
        assert "run metadata" in str(exc).lower()


def test_figure2_fails_when_method_row_missing_for_k(tmp_path) -> None:
    figures = _load_pipeline_module("12_generate_figures.py")
    rows = []
    for k in [5, 10, 15, 20]:
        for method in figures.STRATEGY_ORDER:
            if k == 10 and method == "random":
                continue
            rows.append(_per_domain_row(k, method, n_items_domain=4))
    df = pd.DataFrame(rows)
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir(parents=True)

    try:
        figures.figure_2_domain_starvation(df, fig_dir)
        raise AssertionError("Expected missing per-K method row to fail closed")
    except ValueError as exc:
        msg = str(exc).lower()
        assert "n_items=10" in msg
        assert "random" in msg


def test_figure4_annotation_uses_actual_intellect_item_count(tmp_path, monkeypatch) -> None:
    figures = _load_pipeline_module("12_generate_figures.py")
    rows = [
        _per_domain_row(20, "domain_balanced", r_value=0.92, n_items_domain=4),
        _per_domain_row(20, "mini_ipip", r_value=0.90, n_items_domain=4),
        _per_domain_row(20, "adaptive_topk", r_value=0.55, n_items_domain=2),
    ]
    # Force adaptive_topk to show a non-zero Intellect item count in annotation.
    rows[2]["items_Intellect"] = 2
    rows[2]["r_Intellect"] = 0.41
    df = pd.DataFrame(rows)
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir(parents=True)

    import matplotlib.axes._axes as maxes

    captured: dict[str, str] = {}
    original_annotate = maxes.Axes.annotate

    def _capture_annotate(self, text, *args, **kwargs):  # type: ignore[no-untyped-def]
        text_str = str(text)
        if "Intellect items" in text_str:
            captured["text"] = text_str
        return original_annotate(self, text, *args, **kwargs)

    monkeypatch.setattr(maxes.Axes, "annotate", _capture_annotate)

    figures.figure_4_per_domain_k20(df, fig_dir)
    assert captured.get("text") == "r = 0.410\n(2 Intellect items)"


def test_figure1_requires_complete_strategy_coverage() -> None:
    figures = _load_pipeline_module("12_generate_figures.py")
    payload = {
        "overall": {
            str(k): {
                "domain_balanced": {
                    "pearson_r": 0.9,
                    "pearson_r_ci": [0.89, 0.91],
                }
            }
            for k in figures.EXPECTED_K_VALUES
        }
    }
    try:
        figures.figure_1_efficiency_curves(payload, Path("."))
        raise AssertionError("Expected missing strategies to fail closed")
    except ValueError as exc:
        assert "missing required strategies" in str(exc).lower()


def test_figure3_fails_closed_when_required_comparison_missing(tmp_path) -> None:
    figures = _load_pipeline_module("12_generate_figures.py")
    payload = {
        "comparisons": [
            {"method": "domain_balanced", "n_items": 10, "ml_r": 0.90, "avg_r": 0.88, "delta_r": 0.02},
            {"method": "domain_balanced", "n_items": 15, "ml_r": 0.91, "avg_r": 0.89, "delta_r": 0.02},
            {"method": "domain_balanced", "n_items": 20, "ml_r": 0.92, "avg_r": 0.90, "delta_r": 0.02},
            {"method": "mini_ipip", "n_items": 20, "ml_r": 0.91, "avg_r": 0.89, "delta_r": 0.02},
        ]
    }
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir(parents=True)

    try:
        figures.figure_3_ml_vs_averaging(payload, fig_dir)
        raise AssertionError("Expected Figure 3 generation to fail on incomplete comparisons")
    except ValueError as exc:
        msg = str(exc).lower()
        assert "missing required comparisons" in msg
        assert "domain_balanced:25" in msg


def test_figure3_fails_closed_on_unexpected_comparison_pair(tmp_path) -> None:
    figures = _load_pipeline_module("12_generate_figures.py")
    payload = {
        "comparisons": [
            {"method": "domain_balanced", "n_items": 10, "ml_r": 0.90, "avg_r": 0.88, "delta_r": 0.02},
            {"method": "domain_balanced", "n_items": 15, "ml_r": 0.91, "avg_r": 0.89, "delta_r": 0.02},
            {"method": "domain_balanced", "n_items": 20, "ml_r": 0.92, "avg_r": 0.90, "delta_r": 0.02},
            {"method": "domain_balanced", "n_items": 25, "ml_r": 0.93, "avg_r": 0.91, "delta_r": 0.02},
            {"method": "mini_ipip", "n_items": 20, "ml_r": 0.91, "avg_r": 0.89, "delta_r": 0.02},
            {"method": "adaptive_topk", "n_items": 20, "ml_r": 0.85, "avg_r": 0.84, "delta_r": 0.01},
        ]
    }
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir(parents=True)

    try:
        figures.figure_3_ml_vs_averaging(payload, fig_dir)
        raise AssertionError("Expected Figure 3 generation to fail on unexpected comparison")
    except ValueError as exc:
        msg = str(exc).lower()
        assert "unexpected comparison" in msg
        assert "adaptive_topk" in msg


def test_correlations_item_info_embeds_provenance_metadata(tmp_path) -> None:
    correlations = _load_pipeline_module("05_compute_correlations.py")

    source_path = tmp_path / "train.parquet"
    source_path.write_bytes(b"synthetic-train")
    source_sha = file_sha256(source_path)

    output_path = tmp_path / "item_info.json"
    item_pool = [
        {
            "id": "ext1",
            "rank": 1,
            "home_domain": "ext",
            "cross_domain_info": 0.5,
            "own_domain_r": 0.4,
            "domain_correlations": {domain: 0.1 for domain in DOMAINS},
            "is_reverse_keyed": False,
        }
    ]
    first_item = {
        "id": "ext1",
        "text": "test",
        "home_domain": "ext",
    }
    correlations.write_item_info(
        output_path,
        item_pool,
        first_item,
        inter_item_r_bars={domain: 0.3 for domain in DOMAINS},
        source_path=source_path,
        source_sha256=source_sha,
    )

    with open(output_path) as f:
        payload = json.load(f)
    provenance = payload.get("provenance", {})
    assert provenance.get("script") == "05_compute_correlations.py"
    assert provenance.get("source") == str(source_path)
    assert provenance.get("source_sha256") == source_sha


def test_cronbach_alpha_distinguishes_correlated_from_independent() -> None:
    """A4.5: alpha is high for a coherent scale, low for independent items, and
    None when there are too few rows/items to define it."""
    correlations = _load_pipeline_module("05_compute_correlations.py")
    rng = np.random.default_rng(7)
    n = 500
    latent = rng.normal(size=n)
    coherent = pd.DataFrame(
        {f"ext{i}": latent + rng.normal(scale=0.4, size=n) for i in range(1, 5)}
    )
    a_coherent = correlations.cronbach_alpha(coherent, [f"ext{i}" for i in range(1, 5)])
    assert a_coherent["k"] == 4 and a_coherent["n"] == n
    assert a_coherent["alpha"] is not None and a_coherent["alpha"] > 0.7

    independent = pd.DataFrame({f"agr{i}": rng.normal(size=n) for i in range(1, 5)})
    a_indep = correlations.cronbach_alpha(independent, [f"agr{i}" for i in range(1, 5)])
    assert a_indep["alpha"] is not None and a_indep["alpha"] < 0.3

    a_small = correlations.cronbach_alpha(coherent.head(10), [f"ext{i}" for i in range(1, 5)])
    assert a_small["alpha"] is None and a_small["n"] == 10


def test_write_reliability_embeds_provenance(tmp_path) -> None:
    """A4.5: reliability.json carries stage-05 provenance, split=train, and the three forms."""
    correlations = _load_pipeline_module("05_compute_correlations.py")
    source_path = tmp_path / "train.parquet"
    source_path.write_bytes(b"synthetic-train")
    reliability = {
        "method": {"cronbach_alpha": "...", "omega": "..."},
        "full_50": {
            "ext": {"alpha": 0.8, "alpha_std": 0.81, "r_bar": 0.3, "omega": 0.79, "n": 500, "k": 10}
        },
        "domain_balanced_20": {},
        "mini_ipip_20": {},
    }
    out_path = tmp_path / "reliability.json"
    correlations.write_reliability(out_path, reliability, source_path)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["split"] == "train"
    assert payload["provenance"]["script"] == "05_compute_correlations.py"
    assert "source_sha256" in payload
    assert {"full_50", "domain_balanced_20", "mini_ipip_20"} <= set(payload)


def test_correlations_item_correlations_embeds_standard_provenance(tmp_path) -> None:
    correlations = _load_pipeline_module("05_compute_correlations.py")
    source_path = tmp_path / "train.parquet"
    source_path.write_bytes(b"synthetic-train")
    source_sha = file_sha256(source_path)

    output_path = tmp_path / "item_correlations.json"
    correlations.write_item_correlations(
        output_path,
        {"ext1": {domain: 0.1 for domain in DOMAINS}},
        source_path=source_path,
        n_samples=10,
        provenance={
            "script": "05_compute_correlations.py",
            "git_hash": "deadbeef",
            "data_snapshot_id": "snapshot-a",
            "preprocessing_version": "prep-a",
        },
    )

    with open(output_path) as f:
        payload = json.load(f)

    assert payload.get("source_sha256") == source_sha
    provenance = payload.get("provenance", {})
    assert provenance.get("data_snapshot_id") == "snapshot-a"
    assert provenance.get("git_hash") == "deadbeef"
    assert provenance.get("source") == str(source_path)
    assert provenance.get("source_sha256") == source_sha


def test_train_item_info_requires_primary_stage05_file(tmp_path) -> None:
    train = _load_pipeline_module("07_train.py")
    data_dir = tmp_path / "data" / "processed"
    output_dir = tmp_path / "output"
    data_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    _write_item_info(output_dir / "item_info.json")

    try:
        train._load_item_info(data_dir)
        raise AssertionError("Expected FileNotFoundError when primary item_info is missing")
    except FileNotFoundError as exc:
        assert "stage 05" in str(exc).lower() or "make correlations" in str(exc).lower()


def test_baselines_item_info_requires_primary_stage05_file(tmp_path) -> None:
    baselines = _load_pipeline_module("09_baselines.py")
    data_dir = tmp_path / "data" / "processed"
    output_dir = tmp_path / "output"
    model_dir = tmp_path / "models" / "reference"
    data_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    _write_item_info(output_dir / "item_info.json")
    _write_training_report(
        model_dir / "training_report.json",
        item_info_sha256="a" * 64,
    )

    try:
        baselines._load_item_info(data_dir, model_dir)
        raise AssertionError("Expected FileNotFoundError when primary item_info is missing")
    except FileNotFoundError as exc:
        assert "stage 05" in str(exc).lower() or "make correlations" in str(exc).lower()


def test_baselines_item_info_hash_mismatch_fails_closed(tmp_path) -> None:
    baselines = _load_pipeline_module("09_baselines.py")
    data_dir = tmp_path / "data" / "processed"
    model_dir = tmp_path / "models" / "reference"
    data_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    item_info_path = data_dir / "item_info.json"
    _write_item_info(item_info_path)
    _write_training_report(
        model_dir / "training_report.json",
        item_info_sha256="0" * 64,
    )

    try:
        baselines._load_item_info(data_dir, model_dir)
        raise AssertionError("Expected ValueError for item_info hash mismatch")
    except ValueError as exc:
        assert "provenance mismatch" in str(exc).lower()


def test_baselines_mini_ipip_mapping_rejects_duplicates(tmp_path) -> None:
    baselines = _load_pipeline_module("09_baselines.py")
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)

    dup_items = ITEM_COLUMNS[:19] + [ITEM_COLUMNS[0]]
    payload = {
        "domains": {
            "Extraversion": dup_items[0:4],
            "Agreeableness": dup_items[4:8],
            "Conscientiousness": dup_items[8:12],
            "EmotionalStability": dup_items[12:16],
            "Intellect": dup_items[16:20],
        }
    }
    with open(artifacts_dir / "mini_ipip_mapping.json", "w") as f:
        json.dump(payload, f, indent=2)

    try:
        baselines._load_mini_ipip_mapping(artifacts_dir / "mini_ipip_mapping.json")
        raise AssertionError("Expected ValueError for duplicate Mini-IPIP items")
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()


def test_baselines_validate_test_schema_requires_full_big5_schema() -> None:
    baselines = _load_pipeline_module("09_baselines.py")
    frame = _make_dataset()
    frame = frame.drop(columns=["opn_score"])

    try:
        baselines._validate_test_schema(frame)
        raise AssertionError("Expected ValueError for missing domain score column")
    except ValueError as exc:
        assert "full big-5 schema" in str(exc).lower()


def test_baselines_calibration_loader_fails_closed_for_partial_payload(tmp_path) -> None:
    baselines = _load_pipeline_module("09_baselines.py")
    models_dir = tmp_path / "models" / "reference"
    models_dir.mkdir(parents=True)
    with open(models_dir / "calibration_params_sparse_20_balanced.json", "w") as f:
        json.dump(
            {"ext": {"scale_factor": 1.0, "observed_coverage": 0.9}},
            f,
            indent=2,
        )

    with pytest.raises(ValueError) as exc_info:
        baselines._load_calibration_params(
            models_dir, "calibration_params_sparse_20_balanced.json"
        )

    assert "domain coverage" in str(exc_info.value).lower()


def test_baselines_ml_vs_avg_fails_closed_when_any_domain_percentile_missing() -> None:
    baselines = _load_pipeline_module("09_baselines.py")

    n_rows = 24
    X_test = pd.DataFrame(
        np.linspace(1.0, 5.0, n_rows * len(ITEM_COLUMNS)).reshape(n_rows, len(ITEM_COLUMNS)),
        columns=ITEM_COLUMNS,
    )
    y_test = pd.DataFrame(
        {
            "ext_percentile": np.linspace(5.0, 95.0, n_rows),
            "agr_percentile": np.linspace(7.0, 97.0, n_rows),
            "csn_percentile": np.linspace(9.0, 99.0, n_rows),
            "est_percentile": np.linspace(11.0, 91.0, n_rows),
            "ext_score": np.linspace(1.5, 4.5, n_rows),
            "agr_score": np.linspace(1.7, 4.7, n_rows),
            "csn_score": np.linspace(1.9, 4.9, n_rows),
            "est_score": np.linspace(2.1, 4.8, n_rows),
            "opn_score": np.linspace(2.3, 4.6, n_rows),
        }
    )

    item_pool = [
        {
            "id": item_id,
            "home_domain": item_id[:3],
            "own_domain_r": 0.3 + rank * 1e-4,
            "rank": rank,
        }
        for rank, item_id in enumerate(ITEM_COLUMNS, start=1)
    ]

    class _LinearModel:
        def __init__(self, offset: float) -> None:
            self.offset = float(offset)

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            arr = X.to_numpy(dtype=np.float64, copy=False)
            base = np.nanmean(arr, axis=1)
            return np.clip(0.6 * base + self.offset, 1.0, 5.0)

    domain_models = {
        domain: {
            "q05": _LinearModel(0.00),
            "q50": _LinearModel(0.10),
            "q95": _LinearModel(0.20),
        }
        for domain in DOMAINS
    }

    try:
        baselines._run_ml_vs_averaging_comparison(
            domain_models=domain_models,
            X_values=X_test.values,
            all_columns=list(X_test.columns),
            X_test=X_test,
            y_test=y_test,
            item_pool=item_pool,
            available_items=list(X_test.columns),
            mini_ipip_mapping={domain: [f"{domain}{i}" for i in range(1, 5)] for domain in DOMAINS},
            mini_ipip_norms={domain: {"mean": 3.0, "sd": 1.0} for domain in DOMAINS},
            sparse_calibration={},
            full_calibration={},
            train_df=X_test,
            n_bootstrap=3,
        )
        raise AssertionError("Expected fail-closed error for missing domain percentile columns")
    except ValueError as exc:
        assert "complete domain targets" in str(exc).lower() or "percentile" in str(exc).lower()


def test_baselines_subset_norms_match_train_subset_average() -> None:
    """A4.7: subset norms = mean/sd(ddof=1) of the per-domain subset average over train."""
    baselines = _load_pipeline_module("09_baselines.py")
    rng = np.random.default_rng(1)
    train = pd.DataFrame(
        rng.uniform(1.0, 5.0, size=(200, len(ITEM_COLUMNS))),
        columns=ITEM_COLUMNS,
    )
    cols_by_domain = {"ext": ["ext1", "ext2", "ext3", "ext4"], "agr": ["agr1", "agr2"]}
    norms = baselines._compute_subset_norms(train, cols_by_domain)
    expected_ext = train[["ext1", "ext2", "ext3", "ext4"]].mean(axis=1)
    assert norms["ext"]["mean"] == pytest.approx(float(expected_ext.mean()))
    assert norms["ext"]["sd"] == pytest.approx(float(expected_ext.std(ddof=1)))
    # Only requested domains are present; full-domain default is NOT used.
    assert "agr" in norms and "csn" not in norms


def test_baselines_subset_norms_fail_closed_on_degenerate_sd() -> None:
    """A4.7: a constant subset (sd<=0) must fail closed rather than emit NaN percentiles."""
    baselines = _load_pipeline_module("09_baselines.py")
    train = pd.DataFrame({"ext1": [3.0] * 50, "ext2": [3.0] * 50})
    with pytest.raises(ValueError):
        baselines._compute_subset_norms(train, {"ext": ["ext1", "ext2"]})


def test_baselines_mini_ipip_standalone_bootstrap_attaches_cis() -> None:
    """A4.1: the Mini-IPIP comparator gets respondent-level bootstrap CIs like every
    XGBoost method, but NO coverage CI (averaging has no prediction intervals)."""
    baselines = _load_pipeline_module("09_baselines.py")
    rng = np.random.default_rng(2)
    n = 300
    mapping = {d: [f"{d}{i}" for i in range(1, 5)] for d in DOMAINS}
    item_cols = [it for d in DOMAINS for it in mapping[d]]
    X_test = pd.DataFrame(rng.uniform(1.0, 5.0, size=(n, len(item_cols))), columns=item_cols)
    y_test = pd.DataFrame({f"{d}_percentile": rng.uniform(0, 100, size=n) for d in DOMAINS})
    norms = {d: {"mean": 3.0, "sd": 1.0} for d in DOMAINS}

    overall, per_domain = baselines._evaluate_mini_ipip_standalone(
        X_test, y_test, mini_ipip_mapping=mapping, mini_ipip_norms=norms, n_bootstrap=64,
    )
    assert "pearson_r_ci" in overall and len(overall["pearson_r_ci"]) == 2
    assert overall["pearson_r_ci"][0] <= overall["pearson_r_ci"][1]
    assert "coverage_90_ci" not in overall  # averaging has no intervals
    for d in DOMAINS:
        assert "pearson_r_ci" in per_domain[d]

    overall0, _ = baselines._evaluate_mini_ipip_standalone(
        X_test, y_test, mini_ipip_mapping=mapping, mini_ipip_norms=norms, n_bootstrap=0,
    )
    assert "pearson_r_ci" not in overall0


def test_baselines_ml_vs_avg_emits_paired_xgb_vs_mini_ipip() -> None:
    """A4.1 (part B): a paired domain_balanced-ML vs Mini-IPIP-averaging bootstrap
    with a 95% CI is emitted on the shared test respondents."""
    baselines = _load_pipeline_module("09_baselines.py")
    rng = np.random.default_rng(3)
    n_rows = 200
    X_test = pd.DataFrame(
        rng.uniform(1.0, 5.0, size=(n_rows, len(ITEM_COLUMNS))),
        columns=ITEM_COLUMNS,
    )
    cols = {f"{d}_percentile": rng.uniform(0, 100, size=n_rows) for d in DOMAINS}
    cols.update({f"{d}_score": rng.uniform(1.5, 4.5, size=n_rows) for d in DOMAINS})
    y_test = pd.DataFrame(cols)
    item_pool = [
        {"id": item_id, "home_domain": item_id[:3], "own_domain_r": 0.3 + rank * 1e-4, "rank": rank}
        for rank, item_id in enumerate(ITEM_COLUMNS, start=1)
    ]

    class _LinearModel:
        def __init__(self, offset: float) -> None:
            self.offset = float(offset)

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            arr = X.to_numpy(dtype=np.float64, copy=False)
            base = np.nanmean(arr, axis=1)
            return np.clip(0.6 * base + self.offset, 1.0, 5.0)

    domain_models = {
        d: {"q05": _LinearModel(0.0), "q50": _LinearModel(0.1), "q95": _LinearModel(0.2)}
        for d in DOMAINS
    }

    out = baselines._run_ml_vs_averaging_comparison(
        domain_models=domain_models,
        X_values=X_test.values,
        all_columns=list(X_test.columns),
        X_test=X_test,
        y_test=y_test,
        item_pool=item_pool,
        available_items=list(X_test.columns),
        mini_ipip_mapping={d: [f"{d}{i}" for i in range(1, 5)] for d in DOMAINS},
        mini_ipip_norms={d: {"mean": 3.0, "sd": 1.0} for d in DOMAINS},
        sparse_calibration={},
        full_calibration={},
        train_df=X_test,
        n_bootstrap=64,
    )
    paired = out["xgb_vs_mini_ipip_paired"]
    assert paired["n_items"] == 20
    assert paired["comparison"] == "domain_balanced_ml"
    assert paired["reference"] == "mini_ipip_averaging"
    assert len(paired["delta_pearson_r_ci"]) == 2
    assert paired["delta_pearson_r_ci"][0] <= paired["delta_pearson_r_ci"][1]
    # Mini-IPIP arm now also carries CIs within the same comparison artifact.
    assert "comparisons" in out

    # A4.7: the domain_balanced averaging arm must use train-fit SUBSET norms, not
    # full-domain norms (norms=None). Guards a revert of the avg_norms wiring.
    db_row = next(
        r for r in out["comparisons"]
        if r["method"] == "domain_balanced" and r["n_items"] == 20
    )
    db_items = baselines._select_domain_balanced(item_pool, 4)
    subset_cols = {d: [it for it in db_items if it.startswith(d)] for d in DOMAINS}
    subset = baselines._compute_simple_averaging_scores(
        X_test, y_test, db_items,
        norms=baselines._compute_subset_norms(X_test, subset_cols),
    )["overall"]
    full_domain = baselines._compute_simple_averaging_scores(
        X_test, y_test, db_items, norms=None,
    )["overall"]
    assert db_row["avg_mae"] == pytest.approx(subset["mae"])
    assert subset["mae"] != pytest.approx(full_domain["mae"])  # subset vs full-domain diverge


def test_paired_xgb_vs_mini_ipip_delta_sign_follows_comparison_minus_reference() -> None:
    """A4.1: delta = domain_balanced_ML minus mini_ipip_averaging; swapping the arms
    must flip the sign (the comparison/reference labels alone are hardcoded strings)."""
    baselines = _load_pipeline_module("09_baselines.py")
    rng = np.random.default_rng(11)
    n = 300
    truth = {d: rng.uniform(0, 100, n) for d in DOMAINS}
    high_r = {d: {"true": truth[d], "pred": truth[d] + rng.normal(0, 2, n)} for d in DOMAINS}
    low_r = {d: {"true": truth[d], "pred": rng.normal(50, 5, n)} for d in DOMAINS}

    res = baselines._paired_xgb_vs_mini_ipip(high_r, low_r, n_bootstrap=64, seed=42)
    assert res["comparison"] == "domain_balanced_ml" and res["reference"] == "mini_ipip_averaging"
    assert res["delta_pearson_r"] > 0  # comparison (high r) minus reference (low r)
    lo, hi = res["delta_pearson_r_ci"]
    assert lo <= res["delta_pearson_r"] <= hi
    assert len(res["delta_mae_ci"]) == 2

    swapped = baselines._paired_xgb_vs_mini_ipip(low_r, high_r, n_bootstrap=64, seed=42)
    assert swapped["delta_pearson_r"] < 0  # sign flips when the arms are swapped


def test_figures_attach_checksums_matches_files(tmp_path) -> None:
    """A5.5: _attach_figure_checksums hashes the rendered files per format."""
    figures = _load_pipeline_module("12_generate_figures.py")
    (tmp_path / "figX.png").write_bytes(b"png-bytes")
    (tmp_path / "figX.pdf").write_bytes(b"pdf-bytes")
    entries = [{"filename": "figX", "formats": ["png", "pdf"], "source_artifacts": []}]
    figures._attach_figure_checksums(entries, tmp_path)
    assert entries[0]["sha256"]["png"] == file_sha256(tmp_path / "figX.png")
    assert entries[0]["sha256"]["pdf"] == file_sha256(tmp_path / "figX.pdf")
    assert len(entries[0]["sha256"]["png"]) == 64


def test_baselines_compute_metrics_fails_closed_on_constant_inputs() -> None:
    baselines = _load_pipeline_module("09_baselines.py")
    y_true = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    y_pred = np.array([50.0, 50.0, 50.0, 50.0], dtype=np.float64)
    y_lower = np.array([45.0, 45.0, 45.0, 45.0], dtype=np.float64)
    y_upper = np.array([55.0, 55.0, 55.0, 55.0], dtype=np.float64)

    with pytest.raises(ValueError):
        baselines._compute_metrics(y_true, y_pred, y_lower, y_upper)


def test_simulate_defaults_to_sem_stopping(tmp_path, monkeypatch) -> None:
    simulate = _load_pipeline_module("10_simulate.py")
    monkeypatch.setattr(simulate, "PACKAGE_ROOT", tmp_path)

    model_dir = tmp_path / "models" / "reference"
    model_dir.mkdir(parents=True)
    _write_training_report(model_dir / "training_report.json")
    (tmp_path / "artifacts").mkdir(parents=True)
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    _write_item_info(data_dir / "item_info.json")
    pd.DataFrame({col: [3.0] for col in ITEM_COLUMNS}).to_parquet(
        data_dir / "test.parquet", index=False
    )

    dummy_models = {
        domain: {"q05": object(), "q50": object(), "q95": object()}
        for domain in DOMAINS
    }
    monkeypatch.setattr(simulate, "load_domain_models", lambda *_args, **_kwargs: dummy_models)
    monkeypatch.setattr(
        simulate,
        "load_norms",
        lambda *_args, **_kwargs: {d: {"mean": 3.0, "sd": 0.8} for d in DOMAINS},
    )
    monkeypatch.setattr(
        simulate,
        "load_item_info_for_model_bundle",
        lambda *_args, **_kwargs: (
            {
                "first_item": {"id": "ext1"},
                "item_pool": [
                    {"id": col, "home_domain": col[:3], "own_domain_r": 0.5, "domain_correlations": {}}
                    for col in ITEM_COLUMNS
                ],
                "inter_item_r_bar": {domain: 0.3 for domain in DOMAINS},
            },
            data_dir / "item_info.json",
            "a" * 64,
        ),
    )
    monkeypatch.setattr(
        simulate,
        "load_test_data",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                **{col: [3.0] for col in ITEM_COLUMNS},
                **{f"{domain}_percentile": [50.0] for domain in DOMAINS},
            }
        ),
    )
    monkeypatch.setattr(
        simulate,
        "load_calibration_params",
        lambda *_args, **_kwargs: (
            {
                domain: {"observed_coverage": 0.90, "scale_factor": 1.0}
                for domain in DOMAINS
            },
            "calibration_params_sparse_20_balanced.json",
        ),
    )

    captured: dict[str, Any] = {}

    def _fake_run_simulation(*args, **kwargs):
        config = args[3] if len(args) > 3 else kwargs["config"]
        captured["use_sem_stopping"] = config.use_sem_stopping
        captured["sem_threshold"] = config.sem_threshold
        captured["calibration_params"] = kwargs.get("calibration_params")
        return [{"n_items": 20, "stop_reason": "sem_threshold_met", "domain_errors": {}, "domain_coverage": {d: 4 for d in DOMAINS}}]

    monkeypatch.setattr(simulate, "run_simulation", _fake_run_simulation)
    monkeypatch.setattr(
        simulate,
        "analyze_simulation_results",
        lambda _results: {
            "n_respondents": 1,
            "items_to_convergence": {
                "mean": 20.0,
                "std": 0.0,
                "median": 20.0,
                "min": 20,
                "max": 20,
                "percentile_90": 20.0,
            },
            "stop_reasons": {"sem_threshold_met": 1},
            "domain_coverage_stats": {},
            "domain_metrics": {},
            "overall_metrics": {
                "pearson_r": 0.9,
                "mae": 3.0,
                "within_5_pct": 0.8,
                "coverage_90": 0.9,
            },
        },
    )

    monkeypatch.setattr(sys, "argv", ["10_simulate.py", "--data-dir", "data/processed", "--n-sample", "1"])
    rc = simulate.main()
    assert rc == 0
    assert captured.get("use_sem_stopping") is True
    assert abs(float(captured.get("sem_threshold")) - 0.45) < 1e-9
    assert captured.get("calibration_params")


def test_simulate_load_calibration_params_fails_closed_for_partial_payload(tmp_path) -> None:
    simulate = _load_pipeline_module("10_simulate.py")
    models_dir = tmp_path / "models" / "reference"
    models_dir.mkdir(parents=True)
    with open(models_dir / "calibration_params_sparse_20_balanced.json", "w") as f:
        json.dump(
            {"ext": {"scale_factor": 1.0, "observed_coverage": 0.9}},
            f,
            indent=2,
        )

    with pytest.raises(ValueError) as exc_info:
        simulate.load_calibration_params(models_dir)

    assert "domain coverage" in str(exc_info.value).lower()


def test_simulate_safe_pearson_returns_none_for_constant_inputs() -> None:
    simulate = _load_pipeline_module("10_simulate.py")

    r_const = simulate._safe_pearson_correlation(
        np.array([1.0, 1.0, 1.0], dtype=np.float64),
        np.array([2.0, 2.0, 2.0], dtype=np.float64),
    )
    assert r_const is None

    r_good = simulate._safe_pearson_correlation(
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
    )
    assert r_good is not None
    assert r_good > 0.99


def test_simulate_sem_requires_complete_inter_item_r_bar(monkeypatch) -> None:
    simulate = _load_pipeline_module("10_simulate.py")
    monkeypatch.setattr(simulate, "load_norms", lambda *_args, **_kwargs: {d: {"mean": 3.0, "sd": 0.8} for d in DOMAINS})

    test_df = pd.DataFrame(
        {
            **{col: [3.0] for col in ITEM_COLUMNS},
            **{f"{domain}_percentile": [50.0] for domain in DOMAINS},
        }
    )
    domain_models = _dummy_domain_models()
    item_info = {
        "first_item": {"id": "ext1"},
        "item_pool": [{"id": col, "home_domain": col[:3], "own_domain_r": 0.5} for col in ITEM_COLUMNS],
        # Missing 'opn' on purpose.
        "inter_item_r_bar": {domain: 0.30 for domain in DOMAINS if domain != "opn"},
    }
    config = simulate.AdaptiveConfig(use_sem_stopping=True, use_ci_stopping=False)

    try:
        simulate.run_simulation(
            test_df=test_df,
            domain_models=domain_models,
            item_info=item_info,
            config=config,
            n_sample=1,
            seed=42,
        )
        raise AssertionError("Expected ValueError for incomplete inter_item_r_bar")
    except ValueError as exc:
        assert "inter_item_r_bar" in str(exc)
        assert "missing=opn" in str(exc)


def test_simulate_run_simulation_requires_full_big5_schema() -> None:
    simulate = _load_pipeline_module("10_simulate.py")

    test_df = pd.DataFrame(
        {
            **{col: [3.0] for col in ITEM_COLUMNS},
            "ext_percentile": [50.0],
            "agr_percentile": [50.0],
            "csn_percentile": [50.0],
            "est_percentile": [50.0],
        }
    )
    domain_models = _dummy_domain_models()
    item_info = {
        "first_item": {"id": "ext1"},
        "item_pool": [
            {
                "id": col,
                "home_domain": col[:3],
                "own_domain_r": 0.5,
                "domain_correlations": {domain: 0.1 for domain in DOMAINS},
            }
            for col in ITEM_COLUMNS
        ],
        "inter_item_r_bar": {domain: 0.30 for domain in DOMAINS},
    }
    config = simulate.AdaptiveConfig(use_sem_stopping=False, use_ci_stopping=False)

    try:
        simulate.run_simulation(
            test_df=test_df,
            domain_models=domain_models,
            item_info=item_info,
            config=config,
            n_sample=1,
            seed=42,
        )
        raise AssertionError("Expected ValueError for missing domain percentile column")
    except ValueError as exc:
        assert "full big-5 schema" in str(exc).lower()


def test_simulate_run_simulation_accepts_explicit_norms(monkeypatch) -> None:
    import lib.scoring
    simulate = _load_pipeline_module("10_simulate.py")
    monkeypatch.setattr(simulate, "load_norms", lambda *_args, **_kwargs: {d: {"mean": 3.0, "sd": 0.8} for d in DOMAINS})
    monkeypatch.setattr(lib.scoring, "load_norms", lambda *_args, **_kwargs: {d: {"mean": 3.0, "sd": 0.8} for d in DOMAINS})

    test_df = pd.DataFrame(
        {
            **{col: [3.0] for col in ITEM_COLUMNS},
            **{f"{domain}_percentile": [50.0] for domain in DOMAINS},
        }
    )
    domain_models = _dummy_domain_models()
    item_info = {
        "first_item": {"id": "ext1"},
        "item_pool": [
            {
                "id": col,
                "home_domain": col[:3],
                "own_domain_r": 0.5,
                "domain_correlations": {domain: 0.1 for domain in DOMAINS},
            }
            for col in ITEM_COLUMNS
        ],
        "inter_item_r_bar": {domain: 0.30 for domain in DOMAINS},
    }
    config = simulate.AdaptiveConfig(
        min_items=1,
        max_items=1,
        use_ci_stopping=False,
        use_sem_stopping=False,
    )
    norms = {domain: {"mean": 3.0, "sd": 1.0} for domain in DOMAINS}

    results = simulate.run_simulation(
        test_df=test_df,
        domain_models=domain_models,
        item_info=item_info,
        config=config,
        n_sample=1,
        seed=42,
        norms=norms,
    )
    assert len(results) == 1
    assert results[0]["n_items"] == 1


def test_simulate_model_dir_relative_to_package_root(tmp_path, monkeypatch) -> None:
    simulate = _load_pipeline_module("10_simulate.py")
    monkeypatch.setattr(simulate, "PACKAGE_ROOT", tmp_path)

    model_dir = tmp_path / "models" / "custom"
    data_dir = tmp_path / "data" / "processed"
    artifacts_dir = tmp_path / "artifacts"
    model_dir.mkdir(parents=True)
    _write_training_report(model_dir / "training_report.json")
    data_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)
    _write_item_info(data_dir / "item_info.json")
    pd.DataFrame({col: [3.0] for col in ITEM_COLUMNS}).to_parquet(
        data_dir / "test.parquet", index=False
    )

    captured: dict[str, Any] = {}
    dummy_models = {
        domain: {"q05": object(), "q50": object(), "q95": object()}
        for domain in DOMAINS
    }

    def _fake_load_models(path):
        captured["models_dir"] = path
        return dummy_models

    monkeypatch.setattr(simulate, "load_domain_models", _fake_load_models)
    monkeypatch.setattr(simulate, "load_norms", lambda *_args, **_kwargs: {d: {"mean": 3.0, "sd": 0.8} for d in DOMAINS})
    monkeypatch.setattr(
        simulate,
        "load_item_info_for_model_bundle",
        lambda models_dir, _data_dir: (
            {
                "first_item": {"id": "ext1"},
                "item_pool": [
                    {"id": col, "home_domain": col[:3], "own_domain_r": 0.5, "domain_correlations": {}}
                    for col in ITEM_COLUMNS
                ],
                "inter_item_r_bar": {domain: 0.3 for domain in DOMAINS},
            },
            data_dir / "item_info.json",
            "a" * 64,
        ),
    )
    monkeypatch.setattr(
        simulate,
        "load_test_data",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                **{col: [3.0] for col in ITEM_COLUMNS},
                **{f"{domain}_percentile": [50.0] for domain in DOMAINS},
            }
        ),
    )
    monkeypatch.setattr(simulate, "load_calibration_params", lambda *_args, **_kwargs: ({}, "none"))
    monkeypatch.setattr(
        simulate,
        "run_simulation",
        lambda *_args, **_kwargs: [
            {
                "n_items": 20,
                "stop_reason": "sem_threshold_met",
                "domain_errors": {},
                "domain_coverage": {d: 4 for d in DOMAINS},
            }
        ],
    )
    monkeypatch.setattr(
        simulate,
        "analyze_simulation_results",
        lambda _results: {
            "n_respondents": 1,
            "items_to_convergence": {
                "mean": 20.0,
                "std": 0.0,
                "median": 20.0,
                "min": 20,
                "max": 20,
                "percentile_90": 20.0,
            },
            "stop_reasons": {"sem_threshold_met": 1},
            "domain_coverage_stats": {},
            "domain_metrics": {},
            "overall_metrics": {
                "pearson_r": 0.9,
                "mae": 3.0,
                "within_5_pct": 0.8,
                "coverage_90": 0.9,
            },
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["10_simulate.py", "--model-dir", "models/custom", "--data-dir", "data/processed", "--n-sample", "1"],
    )
    rc = simulate.main()
    assert rc == 0
    assert captured["models_dir"] == model_dir


def test_tune_main_allows_explicit_full50_only_fallback(
    tmp_path,
    monkeypatch,
) -> None:
    tune = _load_pipeline_module("06_tune.py")
    monkeypatch.setattr(tune, "PACKAGE_ROOT", tmp_path)

    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)

    frame = _make_dataset()
    frame.to_parquet(data_dir / "train.parquet", index=False)
    frame.to_parquet(data_dir / "val.parquet", index=False)

    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: test\ndata_dir: data/processed\nsparsity:\n  enabled: true\n", encoding="utf-8")

    captured: dict[str, Any] = {}

    def _fake_run_optuna_tuning(
        X_train,
        y_train,
        X_val,
        y_val,
        y_val_pct,
        n_trials,
        item_info,
        config,
        mini_ipip_items=None,
        parallel_trials=1,
        gpu=False,
    ):
        captured["item_info"] = item_info
        captured["sparsity_enabled"] = config.get("sparsity", {}).get("enabled", None)
        return tune.DEFAULT_PARAMS.copy()

    monkeypatch.setattr(tune, "_run_optuna_tuning", _fake_run_optuna_tuning)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "06_tune.py",
            "--trials",
            "1",
            "--config",
            str(config_path),
            "--allow-no-sparse20-objective",
        ],
    )

    rc = tune.main()
    assert rc == 0
    assert captured.get("item_info", {}) == {}
    assert captured.get("sparsity_enabled") is False


def test_tune_main_honors_data_dir_cli_override(tmp_path, monkeypatch) -> None:
    tune = _load_pipeline_module("06_tune.py")
    monkeypatch.setattr(tune, "PACKAGE_ROOT", tmp_path)

    default_data_dir = tmp_path / "data" / "processed"
    override_data_dir = tmp_path / "data" / "override"
    default_data_dir.mkdir(parents=True)
    override_data_dir.mkdir(parents=True)
    (tmp_path / "artifacts").mkdir(parents=True)

    frame = _make_dataset()
    frame.to_parquet(override_data_dir / "train.parquet", index=False)
    frame.to_parquet(override_data_dir / "val.parquet", index=False)
    _write_item_info(override_data_dir / "item_info.json")

    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: test\nsparsity:\n  enabled: false\n", encoding="utf-8")

    captured: dict[str, Any] = {}

    def _fake_run_optuna_tuning(
        X_train,
        y_train,
        X_val,
        y_val,
        y_val_pct,
        n_trials,
        item_info,
        config,
        mini_ipip_items=None,
        parallel_trials=1,
        gpu=False,
    ):
        captured["train_rows"] = len(X_train)
        captured["val_rows"] = len(X_val)
        captured["item_pool_n"] = len(item_info.get("item_pool", []))
        return tune.DEFAULT_PARAMS.copy()

    monkeypatch.setattr(tune, "_run_optuna_tuning", _fake_run_optuna_tuning)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "06_tune.py",
            "--trials",
            "1",
            "--config",
            str(config_path),
            "--data-dir",
            "data/override",
        ],
    )

    rc = tune.main()
    assert rc == 0
    assert captured["train_rows"] == len(frame)
    assert captured["val_rows"] == len(frame)
    assert captured["item_pool_n"] == len(ITEM_COLUMNS)


def test_tune_main_resolves_xgb_n_jobs_from_config(tmp_path, monkeypatch) -> None:
    tune = _load_pipeline_module("06_tune.py")
    monkeypatch.setattr(tune, "PACKAGE_ROOT", tmp_path)

    data_dir = tmp_path / "data" / "processed"
    artifacts_dir = tmp_path / "artifacts"
    data_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)

    frame = _make_dataset()
    frame.to_parquet(data_dir / "train.parquet", index=False)
    frame.to_parquet(data_dir / "val.parquet", index=False)
    _write_item_info(data_dir / "item_info.json")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: test",
                "data_dir: data/processed",
                "sparsity:",
                "  enabled: false",
                "training:",
                "  n_jobs: 6",
            ]
        ),
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    def _fake_run_optuna_tuning(
        X_train,
        y_train,
        X_val,
        y_val,
        y_val_pct,
        n_trials,
        item_info,
        config,
        mini_ipip_items=None,
        parallel_trials=1,
        gpu=False,
    ):
        captured["xgb_n_jobs"] = config.get("_xgb_n_jobs")
        return tune.DEFAULT_PARAMS.copy()

    monkeypatch.setattr(tune, "_run_optuna_tuning", _fake_run_optuna_tuning)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "06_tune.py",
            "--trials",
            "1",
            "--config",
            str(config_path),
        ],
    )

    rc = tune.main()
    assert rc == 0
    assert captured["xgb_n_jobs"] == 6

    with open(artifacts_dir / "tuned_params.json") as f:
        payload = json.load(f)
    assert payload["provenance"]["xgb_n_jobs"] == 6
    assert payload["provenance"]["xgb_n_jobs_source"] == "config.training.n_jobs"


def test_tune_optuna_missing_fails_closed(monkeypatch) -> None:
    tune = _load_pipeline_module("06_tune.py")

    n_rows = 12
    X_train = pd.DataFrame(
        np.linspace(1.0, 5.0, n_rows * len(ITEM_COLUMNS)).reshape(n_rows, len(ITEM_COLUMNS)),
        columns=ITEM_COLUMNS,
    )
    y_train = pd.DataFrame(
        {f"{domain}_score": np.linspace(2.0, 4.0, n_rows) for domain in DOMAINS}
    )
    X_val = X_train.copy()
    y_val = y_train.copy()
    y_val_pct = pd.DataFrame(
        {f"{domain}_percentile": np.linspace(5.0, 95.0, n_rows) for domain in DOMAINS}
    )

    monkeypatch.setitem(sys.modules, "optuna", None)

    try:
        tune._run_optuna_tuning(
            X_train,
            y_train,
            X_val,
            y_val,
            y_val_pct,
            n_trials=1,
            item_info={},
            config={},
            mini_ipip_items=None,
        )
        raise AssertionError("Expected tuning to fail closed when optuna is unavailable")
    except RuntimeError as exc:
        assert "optuna not installed" in str(exc).lower()


def test_tune_main_fails_closed_when_mini_ipip_mapping_missing(
    tmp_path,
    monkeypatch,
) -> None:
    tune = _load_pipeline_module("06_tune.py")
    monkeypatch.setattr(tune, "PACKAGE_ROOT", tmp_path)

    data_dir = tmp_path / "data" / "processed"
    artifacts_dir = tmp_path / "artifacts"
    data_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)

    frame = _make_dataset()
    frame.to_parquet(data_dir / "train.parquet", index=False)
    frame.to_parquet(data_dir / "val.parquet", index=False)
    _write_item_info(data_dir / "item_info.json")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: test",
                "data_dir: data/processed",
                "sparsity:",
                "  enabled: true",
                "  include_mini_ipip: true",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["06_tune.py", "--trials", "1", "--config", str(config_path)],
    )
    rc = tune.main()
    assert rc == 1


def test_tune_main_fails_closed_when_mini_ipip_mapping_malformed(
    tmp_path,
    monkeypatch,
) -> None:
    tune = _load_pipeline_module("06_tune.py")
    monkeypatch.setattr(tune, "PACKAGE_ROOT", tmp_path)

    data_dir = tmp_path / "data" / "processed"
    artifacts_dir = tmp_path / "artifacts"
    data_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)

    frame = _make_dataset()
    frame.to_parquet(data_dir / "train.parquet", index=False)
    frame.to_parquet(data_dir / "val.parquet", index=False)
    _write_item_info(data_dir / "item_info.json")

    with open(artifacts_dir / "mini_ipip_mapping.json", "w") as f:
        json.dump({"domains": {"Extraversion": ["ext1"]}}, f, indent=2)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: test",
                "data_dir: data/processed",
                "sparsity:",
                "  enabled: true",
                "  include_mini_ipip: true",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["06_tune.py", "--trials", "1", "--config", str(config_path)],
    )
    rc = tune.main()
    assert rc == 1


def test_tune_main_fails_closed_when_item_info_source_hash_mismatch(
    tmp_path,
    monkeypatch,
) -> None:
    tune = _load_pipeline_module("06_tune.py")
    monkeypatch.setattr(tune, "PACKAGE_ROOT", tmp_path)

    data_dir = tmp_path / "data" / "processed"
    artifacts_dir = tmp_path / "artifacts"
    data_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)

    frame = _make_dataset()
    frame.to_parquet(data_dir / "train.parquet", index=False)
    frame.to_parquet(data_dir / "val.parquet", index=False)
    _write_item_info(data_dir / "item_info.json", source_sha256="0" * 64)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: test",
                "data_dir: data/processed",
                "sparsity:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["06_tune.py", "--trials", "1", "--config", str(config_path)],
    )
    rc = tune.main()
    assert rc == 1


def test_train_main_fails_closed_when_mini_ipip_mapping_missing(
    tmp_path,
    monkeypatch,
) -> None:
    train = _load_pipeline_module("07_train.py")
    monkeypatch.setattr(train, "PACKAGE_ROOT", tmp_path)

    data_dir = tmp_path / "data" / "processed"
    artifacts_dir = tmp_path / "artifacts"
    data_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)

    frame = _make_dataset()
    frame.to_parquet(data_dir / "train.parquet", index=False)
    frame.to_parquet(data_dir / "val.parquet", index=False)
    _write_item_info(data_dir / "item_info.json")

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "name: unit_missing_mini_ipip",
                "output_dir: models/unit_missing_mini_ipip",
                "sparsity:",
                "  enabled: true",
                "  include_mini_ipip: true",
                "training:",
                "  cv_folds: 0",
                "  random_state: 42",
                "validation:",
                "  min_pearson_r: 0.0",
                "  min_coverage_90: 0.0",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["07_train.py", "--config", str(cfg_path)])
    rc = train.main()
    assert rc == 1


def test_train_main_fails_closed_when_mini_ipip_mapping_malformed(
    tmp_path,
    monkeypatch,
) -> None:
    train = _load_pipeline_module("07_train.py")
    monkeypatch.setattr(train, "PACKAGE_ROOT", tmp_path)

    data_dir = tmp_path / "data" / "processed"
    artifacts_dir = tmp_path / "artifacts"
    data_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)

    frame = _make_dataset()
    frame.to_parquet(data_dir / "train.parquet", index=False)
    frame.to_parquet(data_dir / "val.parquet", index=False)
    _write_item_info(data_dir / "item_info.json")

    with open(artifacts_dir / "mini_ipip_mapping.json", "w") as f:
        json.dump({"domains": {"Extraversion": ["ext1"]}}, f, indent=2)

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "name: unit_bad_mini_ipip",
                "output_dir: models/unit_bad_mini_ipip",
                "sparsity:",
                "  enabled: true",
                "  include_mini_ipip: true",
                "training:",
                "  cv_folds: 0",
                "  random_state: 42",
                "validation:",
                "  min_pearson_r: 0.0",
                "  min_coverage_90: 0.0",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["07_train.py", "--config", str(cfg_path)])
    rc = train.main()
    assert rc == 1


def test_train_main_fails_closed_when_item_info_source_hash_mismatch(
    tmp_path,
    monkeypatch,
) -> None:
    train = _load_pipeline_module("07_train.py")
    monkeypatch.setattr(train, "PACKAGE_ROOT", tmp_path)

    data_dir = tmp_path / "data" / "processed"
    artifacts_dir = tmp_path / "artifacts"
    data_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)

    frame = _make_dataset()
    frame.to_parquet(data_dir / "train.parquet", index=False)
    frame.to_parquet(data_dir / "val.parquet", index=False)
    _write_item_info(data_dir / "item_info.json", source_sha256="0" * 64)

    cfg_path = tmp_path / "cfg_iteminfo_source_mismatch.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "name: unit_iteminfo_source_mismatch",
                "output_dir: models/unit_iteminfo_source_mismatch",
                "sparsity:",
                "  enabled: true",
                "  include_mini_ipip: false",
                "training:",
                "  cv_folds: 0",
                "  random_state: 42",
                "validation:",
                "  min_pearson_r: 0.0",
                "  min_coverage_90: 0.0",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["07_train.py", "--config", str(cfg_path)])
    rc = train.main()
    assert rc == 1


def test_train_main_records_xgb_n_jobs_from_cli(
    tmp_path,
    monkeypatch,
) -> None:
    train = _load_pipeline_module("07_train.py")
    monkeypatch.setattr(train, "PACKAGE_ROOT", tmp_path)

    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    frame = _make_dataset()
    frame.to_parquet(data_dir / "train.parquet", index=False)
    frame.to_parquet(data_dir / "val.parquet", index=False)

    cfg_path = tmp_path / "cfg_n_jobs.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "name: unit_n_jobs",
                "output_dir: models/unit_n_jobs",
                "sparsity:",
                "  enabled: false",
                "training:",
                "  cv_folds: 0",
                "  random_state: 42",
                "validation:",
                "  min_pearson_r: 0.0",
                "  min_coverage_90: 0.0",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(train, "_load_mini_ipip_mapping", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train, "_train_domain_models", lambda *_args, **_kwargs: _dummy_domain_models())
    monkeypatch.setattr(train, "_validate_model_outputs", lambda *_args, **_kwargs: {"ok": {"passed": True}})
    monkeypatch.setattr(train, "_evaluate_domain_models", lambda *_args, **_kwargs: _make_eval_metrics(r=0.92, coverage=0.9))
    monkeypatch.setattr(
        train,
        "_compute_calibration_params",
        lambda *_args, **_kwargs: {
            domain: {"observed_coverage": 0.9, "scale_factor": 1.0}
            for domain in DOMAINS
        },
    )
    monkeypatch.setattr(train.joblib, "dump", lambda *_args, **_kwargs: None)

    monkeypatch.setattr(
        sys,
        "argv",
        ["07_train.py", "--config", str(cfg_path), "--n-jobs", "5"],
    )
    rc = train.main()
    assert rc == 0

    report_path = tmp_path / "models" / "unit_n_jobs" / "training_report.json"
    with open(report_path) as f:
        report = json.load(f)
    assert report["config"]["xgb_n_jobs"] == 5
    assert report["config"]["xgb_n_jobs_source"] == "cli"
    assert report["provenance"]["xgb_n_jobs"] == 5


def test_train_main_defaults_cv_folds_from_constants_when_config_omits_it(
    tmp_path,
    monkeypatch,
) -> None:
    train = _load_pipeline_module("07_train.py")
    monkeypatch.setattr(train, "PACKAGE_ROOT", tmp_path)

    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    frame = _make_dataset()
    frame.to_parquet(data_dir / "train.parquet", index=False)
    frame.to_parquet(data_dir / "val.parquet", index=False)

    cfg_path = tmp_path / "cfg_default_cv.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "name: unit_default_cv",
                "output_dir: models/unit_default_cv",
                "sparsity:",
                "  enabled: false",
                "validation:",
                "  min_pearson_r: 0.0",
                "  min_coverage_90: 0.0",
            ]
        ),
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    def _fake_cv(*_args, **kwargs):
        captured["n_folds"] = kwargs["n_folds"]
        return {
            "fold_results": [],
            "aggregate": {
                "overall": {
                    "pearson_r": {"mean": 0.9, "std": 0.0, "min": 0.9, "max": 0.9},
                    "coverage_90": {"mean": 0.9, "std": 0.0, "min": 0.9, "max": 0.9},
                    "within_5_pct": {"mean": 0.9, "std": 0.0, "min": 0.9, "max": 0.9},
                }
            },
            "n_folds": DEFAULT_STAGE07_CV_FOLDS,
            "parallel_folds": 1,
            "xgb_n_jobs_total": 1,
            "xgb_n_jobs_per_fold": 1,
        }

    monkeypatch.setattr(train, "_run_cross_validation_robustness", _fake_cv)
    monkeypatch.setattr(train, "_load_item_info", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(train, "_load_mini_ipip_mapping", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train, "_train_domain_models", lambda *_args, **_kwargs: _dummy_domain_models())
    monkeypatch.setattr(train, "_validate_model_outputs", lambda *_args, **_kwargs: {"ok": {"passed": True}})
    monkeypatch.setattr(train, "_evaluate_domain_models", lambda *_args, **_kwargs: _make_eval_metrics(r=0.92, coverage=0.9))
    monkeypatch.setattr(
        train,
        "_compute_calibration_params",
        lambda *_args, **_kwargs: {
            domain: {"observed_coverage": 0.9, "scale_factor": 1.0}
            for domain in DOMAINS
        },
    )
    monkeypatch.setattr(train.joblib, "dump", lambda *_args, **_kwargs: None)

    monkeypatch.setattr(sys, "argv", ["07_train.py", "--config", str(cfg_path)])
    rc = train.main()
    assert rc == 0
    assert captured["n_folds"] == DEFAULT_STAGE07_CV_FOLDS

    report_path = tmp_path / "models" / "unit_default_cv" / "training_report.json"
    with open(report_path) as f:
        report = json.load(f)
    assert report["config"]["training"]["cv_folds"] == DEFAULT_STAGE07_CV_FOLDS
    assert report["provenance"]["cv_folds"] == DEFAULT_STAGE07_CV_FOLDS


def test_baselines_loader_fails_closed_when_mini_ipip_mapping_missing(tmp_path) -> None:
    baselines = _load_pipeline_module("09_baselines.py")
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        baselines._load_mini_ipip_mapping(artifacts_dir / "mini_ipip_mapping.json")


def test_baselines_calibration_policy_uses_sparse_for_all_sparse_budgets() -> None:
    baselines = _load_pipeline_module("09_baselines.py")
    sparse = {domain: {"scale_factor": 1.1, "observed_coverage": 0.84} for domain in DOMAINS}
    full = {domain: {"scale_factor": 1.0, "observed_coverage": 0.90} for domain in DOMAINS}

    selected_10, regime_10 = baselines._choose_calibration_for_budget(10, sparse, full)
    assert selected_10 == sparse
    assert regime_10 == "sparse_20_balanced"

    selected_50, regime_50 = baselines._choose_calibration_for_budget(50, sparse, full)
    assert selected_50 == full
    assert regime_50 == "full_50"


def test_baselines_run_comparisons_routes_mini_ipip_to_standalone(
    monkeypatch,
) -> None:
    baselines = _load_pipeline_module("09_baselines.py")

    n_rows = 16
    X_test = pd.DataFrame(
        np.linspace(1.0, 5.0, n_rows * len(ITEM_COLUMNS)).reshape(n_rows, len(ITEM_COLUMNS)),
        columns=ITEM_COLUMNS,
    )
    y_test = pd.DataFrame(
        {
            **{f"{d}_percentile": np.linspace(5.0, 95.0, n_rows) for d in DOMAINS},
            **{f"{d}_score": np.linspace(1.5, 4.5, n_rows) for d in DOMAINS},
        }
    )
    item_pool = [
        {"id": item_id, "home_domain": item_id[:3], "own_domain_r": 0.5, "rank": rank}
        for rank, item_id in enumerate(ITEM_COLUMNS, start=1)
    ]

    def _fake_overall(score: float) -> dict[str, Any]:
        return {
            "pearson_r": score,
            "mae": 1.0,
            "rmse": 1.1,
            "within_5_pct": 0.8,
            "within_10_pct": 0.9,
            "coverage_90": 0.5,
            "n_items": 20,
        }

    fake_domain = {domain: {"pearson_r": 0.5, "n_domain_items": 4} for domain in DOMAINS}
    monkeypatch.setattr(
        baselines,
        "_evaluate_random_aggregated",
        lambda *_args, **_kwargs: (_fake_overall(0.2), fake_domain),
    )
    monkeypatch.setattr(
        baselines,
        "_evaluate_method",
        lambda *_args, **_kwargs: (_fake_overall(0.3), fake_domain),
    )

    captured: dict[str, Any] = {}

    def _fake_standalone(X_arg, y_arg, mini_ipip_mapping, mini_ipip_norms, n_bootstrap=0, seed=42):  # type: ignore[no-untyped-def]
        captured["mapping"] = mini_ipip_mapping
        captured["norms"] = mini_ipip_norms
        captured["n_bootstrap"] = n_bootstrap
        return _fake_overall(0.77), fake_domain

    monkeypatch.setattr(baselines, "_evaluate_mini_ipip_standalone", _fake_standalone)

    mini_mapping = {domain: [f"{domain}{i}" for i in range(1, 5)] for domain in DOMAINS}
    mini_norms = {domain: {"mean": 3.0, "sd": 1.0} for domain in DOMAINS}

    overall, per_domain = baselines._run_comparisons_at_k(
        domain_models=_dummy_domain_models(),
        X_values=X_test.values,
        all_columns=list(X_test.columns),
        X_test=X_test,
        y_test=y_test,
        item_pool=item_pool,
        available_items=list(X_test.columns),
        n_items=20,
        mini_ipip_mapping=mini_mapping,
        mini_ipip_norms=mini_norms,
        calibration_params=None,
        calibration_regime="none",
        n_bootstrap=0,
        n_random_trials=1,
    )

    assert "mini_ipip" in overall
    assert abs(float(overall["mini_ipip"]["pearson_r"]) - 0.77) < 1e-9
    assert captured["mapping"] == mini_mapping
    assert captured["norms"] == mini_norms
    assert "mini_ipip" in per_domain


def test_makefile_train_runs_reference_then_parallel_ablations() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        ["make", "-n", "train"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    out = result.stdout
    assert "pipeline/07_train.py --config configs/reference.yaml" in out
    assert "pipeline/07_train.py --config configs/ablation_none.yaml" in out
    assert "pipeline/07_train.py --config configs/ablation_focused.yaml" in out
    assert out.index("configs/reference.yaml") < out.index("configs/ablation_none.yaml")
    assert "train-2 train-3" in out


def test_makefile_research_eval_defaults_to_parallel_submake(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    calls_path = tmp_path / "make_calls.log"
    fake_make = tmp_path / "fake-make.sh"
    fake_make.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                f'printf "%s\\n" "$*" >> "{calls_path}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_make.chmod(0o755)
    env = os.environ.copy()
    env["MAKE"] = str(fake_make)
    result = subprocess.run(
        ["make", "research-eval"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0
    calls = calls_path.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 1
    assert calls[0].startswith("-j4 research-eval-reference research-eval-ablation-none")


def test_makefile_research_eval_respects_explicit_parent_parallelism(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    calls_path = tmp_path / "make_calls.log"
    fake_make = tmp_path / "fake-make.sh"
    fake_make.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                f'printf "%s\\n" "$*" >> "{calls_path}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_make.chmod(0o755)
    env = os.environ.copy()
    env["MAKE"] = str(fake_make)
    result = subprocess.run(
        ["make", "-j2", "research-eval"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0
    calls = calls_path.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 1
    assert calls[0].startswith("research-eval-reference research-eval-ablation-none")
    assert not calls[0].startswith("-j4")


def test_makefile_train_invalid_run_reports_single_actionable_error() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        ["make", "train", "10"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "Invalid train run index" in combined
    assert "No rule to make target" not in combined


def test_run_pipeline_omits_empty_make_overrides(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True)
    calls_path = tmp_path / "make_calls.log"
    make_stub = fake_bin / "make"
    make_stub.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                f'printf \"%s\\n\" \"$*\" >> \"{calls_path}\"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    make_stub.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    result = subprocess.run(
        ["bash", "scripts/run-pipeline.sh", "--end-stage", "research-eval"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0
    calls = calls_path.read_text(encoding="utf-8").splitlines()
    train_call = next(line for line in calls if line.startswith("train"))
    research_eval_call = next(line for line in calls if line.startswith("research-eval"))
    assert "CV_PARALLEL_FOLDS=" not in train_call
    assert "RESEARCH_EVAL_PARALLEL=" not in research_eval_call


def test_run_pipeline_writes_checkpoint_markers_for_major_stages(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "scripts" / "run-pipeline.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True)
    make_stub = fake_bin / "make"
    make_stub.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    make_stub.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    result = subprocess.run(
        ["bash", str(script_path), "--end-stage", "correlations"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0

    checkpoint_dir = tmp_path / ".pipeline-checkpoints"
    # Every stage that actually RUNS drops a marker (run-pipeline.sh marks every
    # stage so --resume can skip any completed stage, incl. the expensive download
    # — see the "marking every stage" change in commit 106db88, which replaced the
    # old CHECKPOINT_STAGES allowlist). With --end-stage correlations, download
    # through correlations run; tune (and everything after) does not.
    assert (checkpoint_dir / "download.done").exists()
    assert (checkpoint_dir / "norms.done").exists()
    assert (checkpoint_dir / "prepare.done").exists()
    assert (checkpoint_dir / "correlations.done").exists()
    assert not (checkpoint_dir / "tune.done").exists()


def test_run_pipeline_reference_only_uses_reference_targets_and_runs_notes(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True)
    calls_path = tmp_path / "make_calls.log"
    make_stub = fake_bin / "make"
    make_stub.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                f'printf \"%s\\n\" \"$*\" >> \"{calls_path}\"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    make_stub.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    result = subprocess.run(
        ["bash", "scripts/run-pipeline.sh", "--reference-only", "--end-stage", "figures"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0
    calls = calls_path.read_text(encoding="utf-8").splitlines()
    assert any(line == "prepare" or line.startswith("prepare ") for line in calls)
    assert any(line == "correlations" or line.startswith("correlations ") for line in calls)
    assert any(line.startswith("train 1") for line in calls)
    assert any(line.startswith("research-eval-reference") for line in calls)
    assert any(line.startswith("export-reference export-repo-readme") for line in calls)
    assert not any(line == "prepare-default" for line in calls)
    assert not any(line == "correlations-default" for line in calls)
    assert not any(line.startswith("research-eval ") for line in calls)
    # Reference-only now RUNS notes scoped to the reference variant (was: skipped).
    assert any(line.startswith("notes REFERENCE_ONLY=1") for line in calls)


def test_build_research_summary_reference_only_filters_to_reference(tmp_path, monkeypatch) -> None:
    """--reference-only iterates only the reference variant: the summary's variants and
    provenance narrow to reference, and --strict still fails closed if reference is incomplete."""
    brs = _load_paper_module("build_research_summary.py")

    # Keep the real PACKAGE_ROOT so the variant configs resolve; an empty tmp
    # variants dir (+ tmp output) makes every variant incomplete without touching
    # or overwriting the committed artifacts/research_summary.json.
    variants_dir = tmp_path / "artifacts" / "variants"
    variants_dir.mkdir(parents=True)  # empty -> every variant is incomplete
    out_full = tmp_path / "rs_full.json"
    out_ref = tmp_path / "rs_ref.json"

    # Full (default): the summary contains all three variant keys.
    monkeypatch.setattr(sys, "argv", [
        "build_research_summary.py", "--output", str(out_full),
        "--artifacts-variants-dir", str(variants_dir),
    ])
    assert brs.main() == 0
    full = json.loads(out_full.read_text())
    assert set(full["variants"]) == set(brs.VARIANTS)
    assert full["provenance"]["reference_only"] is False

    # Reference-only: the summary contains ONLY reference; provenance records the mode.
    monkeypatch.setattr(sys, "argv", [
        "build_research_summary.py", "--reference-only", "--output", str(out_ref),
        "--artifacts-variants-dir", str(variants_dir),
    ])
    assert brs.main() == 0
    ref = json.loads(out_ref.read_text())
    assert set(ref["variants"]) == {"reference"}
    assert ref["provenance"]["variants_included"] == ["reference"]
    assert ref["provenance"]["reference_only"] is True

    # Fail-closed preserved: an INCOMPLETE reference still trips --strict under --reference-only.
    monkeypatch.setattr(sys, "argv", [
        "build_research_summary.py", "--reference-only", "--strict", "--output", str(out_ref),
        "--artifacts-variants-dir", str(variants_dir),
    ])
    assert brs.main() == 2


def test_generate_notes_reference_only_scopes_variant_iteration(tmp_path, monkeypatch) -> None:
    """In reference-only mode the cross-variant iteration is scoped to ['reference'] so it does
    NOT KeyError on absent ablation bundles, and the honesty disclosure is emitted."""
    notes = _load_paper_module("generate_notes_data.py")
    summary_path = tmp_path / "research_summary.json"
    summary_path.write_text(
        json.dumps({
            "variants": {"reference": {"notes_inputs": {}}},  # ONLY reference present
            "reference_notes_inputs": {},
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(notes, "RESEARCH_SUMMARY_PATH", summary_path)

    # Default all-variants order hits the absent ablation keys -> KeyError; no disclosure.
    monkeypatch.setattr(notes, "_ACTIVE_VARIANT_ORDER", list(notes.VARIANT_ORDER))
    assert notes._reference_only_disclosure() == ""
    with pytest.raises(KeyError):
        notes._iter_variant_notes_inputs()

    # Reference-only order iterates only reference -> single record, no KeyError; disclosure present.
    monkeypatch.setattr(notes, "_ACTIVE_VARIANT_ORDER", [notes.REFERENCE_VARIANT])
    records = notes._iter_variant_notes_inputs()
    assert [r[0] for r in records] == ["reference"]
    assert "were not run in this reference-only build" in notes._reference_only_disclosure()


def test_update_notes_reference_only_writes_and_restores_active_order(tmp_path, monkeypatch) -> None:
    """update_notes(reference_only=True) writes NOTES.md (no abort), threads the reference-only
    signal into the generators, and ALWAYS restores _ACTIVE_VARIANT_ORDER afterward (finally)."""
    notes = _load_paper_module("generate_notes_data.py")
    template = tmp_path / "NOTES.template.md"
    template.write_text("<!-- BEGIN:probe -->\nPLACEHOLDER\n<!-- END:probe -->\n", encoding="utf-8")
    out = tmp_path / "NOTES.md"
    monkeypatch.setattr(notes, "NOTES_TEMPLATE_PATH", template)
    monkeypatch.setattr(notes, "NOTES_PATH", out)
    monkeypatch.setattr(notes, "PACKAGE_ROOT", tmp_path)
    monkeypatch.setattr(
        notes,
        "SECTION_GENERATORS",
        {"probe": lambda: notes._reference_only_disclosure() + "PROBE_BODY"},
    )

    notes.update_notes(reference_only=True)
    written = out.read_text()
    assert "PROBE_BODY" in written
    assert "were not run in this reference-only build" in written  # disclosure threaded through
    assert notes._ACTIVE_VARIANT_ORDER == notes.VARIANT_ORDER  # restored in finally

    notes.update_notes(reference_only=False)
    assert "were not run in this reference-only build" not in out.read_text()


def test_generate_notes_reference_only_renders_real_ablation_details(tmp_path, monkeypatch) -> None:
    """End-to-end guard: the 6 cross-variant DETAIL generators (which the empty-notes_inputs
    scoping test cannot exercise) render on the REAL reference bundle in reference-only mode —
    disclosure first, Reference block present, no KeyError. Skips when the gitignored reference
    bundle is absent (e.g. in CI)."""
    repo_root = Path(__file__).resolve().parent.parent
    if not (repo_root / "artifacts" / "variants" / "reference" / "validation_results.json").exists():
        pytest.skip("reference bundle (artifacts/variants/reference) not present")

    # Build a reference-only research_summary from the real bundle into tmp.
    brs = _load_paper_module("build_research_summary.py")
    summary_path = tmp_path / "research_summary.json"
    monkeypatch.setattr(
        sys, "argv",
        ["build_research_summary.py", "--reference-only", "--output", str(summary_path)],
    )
    assert brs.main() == 0
    assert summary_path.exists()

    # Render the 6 detail generators in reference-only mode against it.
    notes = _load_paper_module("generate_notes_data.py")
    monkeypatch.setattr(notes, "RESEARCH_SUMMARY_PATH", summary_path)
    monkeypatch.setattr(notes, "_ACTIVE_VARIANT_ORDER", [notes.REFERENCE_VARIANT])
    detail_gens = [
        notes.gen_ablation_validation_details,
        notes.gen_ablation_baselines_details,
        notes.gen_ablation_per_domain_k20_details,
        notes.gen_ablation_domain_starvation_details,
        notes.gen_ablation_ml_vs_averaging_details,
        notes.gen_ablation_simulation_details,
    ]
    for gen in detail_gens:
        out = gen()  # must not KeyError on the absent ablation variants
        assert out.lstrip().startswith(">"), f"{gen.__name__} missing leading disclosure"
        assert "were not run in this reference-only build" in out
        assert "Reference" in out


def test_run_pipeline_reference_only_accepts_export_stage_aliases(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True)
    calls_path = tmp_path / "make_calls.log"
    make_stub = fake_bin / "make"
    make_stub.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                f'printf \"%s\\n\" \"$*\" >> \"{calls_path}\"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    make_stub.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    result = subprocess.run(
        ["bash", "scripts/run-pipeline.sh", "--reference-only", "--start-stage", "export-all", "--end-stage", "export-reference"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0
    calls = calls_path.read_text(encoding="utf-8").splitlines()
    assert calls == ["export-reference export-repo-readme"]
    assert "--- export done" in result.stdout


def test_manage_reference_only_workspace_fails_closed_and_force_cleans(tmp_path, monkeypatch, capsys) -> None:
    workspace = _load_paper_module("manage_reference_only_workspace.py")
    monkeypatch.setattr(workspace, "PACKAGE_ROOT", tmp_path)

    stale_paths = [
        "artifacts/variants/ablation_none/validation_results.json",
        "notes/NOTES.md",
        "output/ablation_focused/model.onnx",
        "logs/eval-ablation-focused.log",
    ]
    for rel in stale_paths:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["manage_reference_only_workspace.py"])
    rc = workspace.main()
    assert rc == 1
    captured = capsys.readouterr()
    assert "remote-reference refuses to run" in captured.out
    assert "artifacts/variants/ablation_none" in captured.out

    monkeypatch.setattr(sys, "argv", ["manage_reference_only_workspace.py", "--force"])
    rc = workspace.main()
    assert rc == 0
    assert not (tmp_path / "artifacts/variants/ablation_none").exists()
    assert not (tmp_path / "notes/NOTES.md").exists()
    assert not (tmp_path / "output/ablation_focused").exists()
    assert not (tmp_path / "logs/eval-ablation-focused.log").exists()


def test_manage_backup_clean_moves_outputs_and_writes_manifest(tmp_path, monkeypatch, capsys) -> None:
    backup = _load_paper_module("manage_backup.py")
    monkeypatch.setattr(backup, "PACKAGE_ROOT", tmp_path)
    monkeypatch.setattr(backup, "BACKUP_ROOT", tmp_path / ".backup")
    monkeypatch.setattr(backup, "MANIFEST_PATH", tmp_path / ".backup" / "manifest.json")

    for rel in [
        "data/.gitkeep",
        "models/.gitkeep",
        "figures/.gitkeep",
        "output/.gitkeep",
        "data/raw/source.zip",
        "data/processed/train.parquet",
        "artifacts/ipip_bffm_norms.json",
        "artifacts/tuned_params.json",
        "artifacts/variants/reference/validation_results.json",
        "models/reference/training_report.json",
        "output/reference/model.onnx",
        "notes/NOTES.md",
        "logs/train-reference.log",
        "figures/plot.png",
        "pipeline.log",
    ]:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"payload:{rel}", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["manage_backup.py", "clean", "--force"])
    rc = backup.main()
    assert rc == 0

    captured = capsys.readouterr()
    assert "Clean complete." in captured.out

    assert not (tmp_path / "data/raw").exists()
    assert not (tmp_path / "models/reference").exists()
    assert not (tmp_path / "notes/NOTES.md").exists()
    assert (tmp_path / ".backup" / "data/raw/source.zip").exists()
    assert (tmp_path / ".backup" / "models/reference/training_report.json").exists()
    assert (tmp_path / ".backup" / "notes/NOTES.md").exists()
    assert (tmp_path / "data/.gitkeep").exists()
    assert (tmp_path / "models/.gitkeep").exists()

    manifest = json.loads((tmp_path / ".backup" / "manifest.json").read_text(encoding="utf-8"))
    assert "data/raw" in manifest["paths"]
    assert "models/reference" in manifest["paths"]
    assert "notes/NOTES.md" in manifest["paths"]


def test_manage_backup_clean_noop_preserves_existing_backup(tmp_path, monkeypatch, capsys) -> None:
    backup = _load_paper_module("manage_backup.py")
    monkeypatch.setattr(backup, "PACKAGE_ROOT", tmp_path)
    monkeypatch.setattr(backup, "BACKUP_ROOT", tmp_path / ".backup")
    monkeypatch.setattr(backup, "MANIFEST_PATH", tmp_path / ".backup" / "manifest.json")

    marker = tmp_path / ".backup" / "marker.txt"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("keep", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["manage_backup.py", "clean", "--force"])
    rc = backup.main()
    assert rc == 0

    captured = capsys.readouterr()
    assert "Existing .backup preserved." in captured.out
    assert marker.exists()


def test_manage_backup_restore_fails_on_conflicts_and_force_restores(tmp_path, monkeypatch, capsys) -> None:
    backup = _load_paper_module("manage_backup.py")
    monkeypatch.setattr(backup, "PACKAGE_ROOT", tmp_path)
    monkeypatch.setattr(backup, "BACKUP_ROOT", tmp_path / ".backup")
    monkeypatch.setattr(backup, "MANIFEST_PATH", tmp_path / ".backup" / "manifest.json")

    src = tmp_path / "artifacts" / "tuned_params.json"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text('{"n_estimators": 100}', encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["manage_backup.py", "clean", "--force"])
    assert backup.main() == 0

    conflict = tmp_path / "artifacts" / "tuned_params.json"
    conflict.parent.mkdir(parents=True, exist_ok=True)
    conflict.write_text("conflict", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["manage_backup.py", "restore"])
    rc = backup.main()
    assert rc == 1
    captured = capsys.readouterr()
    assert "ERROR: Restore conflict at artifacts/tuned_params.json" in captured.out

    monkeypatch.setattr(sys, "argv", ["manage_backup.py", "restore", "--force"])
    rc = backup.main()
    assert rc == 0
    assert conflict.read_text(encoding="utf-8") == '{"n_estimators": 100}'
    assert (tmp_path / ".backup" / "artifacts" / "tuned_params.json").exists()


def test_makefile_remote_all_includes_checkpoint_pull_loop() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        ["make", "-n", "remote-all"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    out = result.stdout
    assert ".pipeline-checkpoints" in out
    assert "Checkpoint '$STAGE' reached. Pulling intermediate results..." in out
    assert '"${MAKE:-make}" remote-pull' in out


def test_makefile_remote_reference_runs_reference_only_pipeline() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        ["make", "-n", "remote-reference"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    out = result.stdout
    assert "scripts/manage_reference_only_workspace.py" in out
    assert "bash scripts/run-pipeline.sh --reference-only" in out
    assert ".pipeline-checkpoints" in out
    assert "TRAIN_N_JOBS=96" in out
    assert '"${MAKE:-make}" remote-pull-reference' in out


def test_makefile_remote_pull_reference_scopes_results() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        ["make", "-n", "remote-pull-reference"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    out = result.stdout
    assert "data/processed/canonical_v1/" in out
    assert "models/reference/" in out
    assert "artifacts/variants/reference/" in out
    assert "output/reference/" in out
    assert "notes/" not in out
    assert "artifacts/research_summary.json" not in out


def test_makefile_remote_push_excludes_backup_dir() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        ["make", "-n", "remote-push"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    out = result.stdout
    assert "--exclude='.backup/'" in out


def test_makefile_remote_push_data_syncs_data_tree() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        ["make", "-n", "remote-push-data"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    out = result.stdout
    assert "Uploading data/" in out
    assert "./data/ " in out
    assert ":$(REMOTE_DIR)/data/" not in out
    assert "/data/" in out


def test_makefile_remote_gpu_push_excludes_backup_dir() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        ["make", "-n", "remote-1-gpu"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    out = result.stdout
    assert "--exclude='.backup/'" in out


def _make_fake_onnxruntime(session_cls):
    """Build a fake `onnxruntime` module for the parity tests.

    Stage 11 now creates single-threaded sessions via SessionOptions /
    ExecutionMode (see `_deterministic_session`), so the stub must expose those
    alongside InferenceSession.
    """
    return type(
        "_Ort",
        (),
        {
            "InferenceSession": session_cls,
            "SessionOptions": type("_Opts", (), {}),
            "ExecutionMode": type("_EM", (), {"ORT_SEQUENTIAL": 0}),
        },
    )()


def test_export_validate_parity_allows_tiny_relative_drift(monkeypatch) -> None:
    export = _load_pipeline_module("11_export_onnx.py")

    class _FakeJoblibModel:
        def predict(self, X):
            return np.full(X.shape[0], 4.413000106811523, dtype=np.float32)

    class _FakeOnnxModel:
        def SerializeToString(self):
            return b"fake-onnx"

    class _FakeSession:
        def __init__(self, _bytes, sess_options=None, providers=None):
            pass

        def get_inputs(self):
            return [type("_Input", (), {"name": "input"})()]

        def run(self, _outputs, inputs):
            X = next(iter(inputs.values()))
            pred = np.full(X.shape[0], 4.413000106811523, dtype=np.float32)
            pred[71] = np.float32(4.413167953491211)
            return [pred]

    monkeypatch.setitem(sys.modules, "onnxruntime", _make_fake_onnxruntime(_FakeSession))

    export.validate_parity({"agr_q50": _FakeJoblibModel()}, {"agr_q50": _FakeOnnxModel()})


def test_export_validate_parity_still_fails_on_material_drift(monkeypatch) -> None:
    export = _load_pipeline_module("11_export_onnx.py")

    class _FakeJoblibModel:
        def predict(self, X):
            return np.full(X.shape[0], 4.0, dtype=np.float32)

    class _FakeOnnxModel:
        def SerializeToString(self):
            return b"fake-onnx"

    class _FakeSession:
        def __init__(self, _bytes, sess_options=None, providers=None):
            pass

        def get_inputs(self):
            return [type("_Input", (), {"name": "input"})()]

        def run(self, _outputs, inputs):
            X = next(iter(inputs.values()))
            pred = np.full(X.shape[0], 4.0, dtype=np.float32)
            pred[71] = np.float32(4.0005)
            return [pred]

    monkeypatch.setitem(sys.modules, "onnxruntime", _make_fake_onnxruntime(_FakeSession))

    with pytest.raises(SystemExit) as exc_info:
        export.validate_parity({"agr_q50": _FakeJoblibModel()}, {"agr_q50": _FakeOnnxModel()})
    assert exc_info.value.code == 1


def test_export_requires_stage05_item_info(tmp_path, monkeypatch) -> None:
    export = _load_pipeline_module("11_export_onnx.py")
    monkeypatch.setattr(export, "PACKAGE_ROOT", tmp_path)

    (tmp_path / "models" / "reference").mkdir(parents=True)
    (tmp_path / "output").mkdir(parents=True)
    (tmp_path / "artifacts").mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir(parents=True)

    monkeypatch.setattr(export, "load_norms", lambda *_args, **_kwargs: {d: {"mean": 3.0, "sd": 0.8} for d in DOMAINS})
    monkeypatch.setattr(sys, "argv", ["11_export_onnx.py", "--data-dir", "data/processed"])
    # load_joblib_models calls sys.exit(1) when model files are missing,
    # so main() raises SystemExit rather than returning 1.
    with pytest.raises(SystemExit) as exc_info:
        export.main()
    assert exc_info.value.code == 1


def test_export_main_resolves_relative_model_and_output_dirs(tmp_path, monkeypatch) -> None:
    export = _load_pipeline_module("11_export_onnx.py")
    monkeypatch.setattr(export, "PACKAGE_ROOT", tmp_path)

    models_dir = tmp_path / "models" / "custom"
    data_dir = tmp_path / "data" / "processed"
    artifacts_dir = tmp_path / "artifacts"
    models_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)

    _write_training_report(models_dir / "training_report.json")

    captured: dict[str, Path] = {}

    def _fake_load_joblib_models(model_dir):
        captured["model_dir"] = model_dir
        return {}

    def _fake_save_onnx_file(_merged_model, output_dir):
        captured["output_dir"] = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(export, "load_norms", lambda *_args, **_kwargs: {d: {"mean": 3.0, "sd": 0.5} for d in DOMAINS})
    monkeypatch.setattr(export, "load_joblib_models", _fake_load_joblib_models)
    monkeypatch.setattr(export, "convert_to_onnx", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(export, "validate_parity", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(export, "merge_onnx_models", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(export, "validate_merged_parity", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(export, "save_onnx_file", _fake_save_onnx_file)
    monkeypatch.setattr(export, "generate_config", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(export, "generate_readme", lambda *_args, **_kwargs: "ok")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "11_export_onnx.py",
            "--model-dir",
            "models/custom",
            "--data-dir",
            "data/processed",
            "--output-dir",
            "output/custom",
        ],
    )
    rc = export.main()
    assert rc == 0
    assert captured["model_dir"] == models_dir
    assert captured["output_dir"] == tmp_path / "output" / "custom"


def test_export_readme_fails_closed_without_strict_signature(tmp_path) -> None:
    export = _load_pipeline_module("11_export_onnx.py")
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    model_dir = tmp_path / "models" / "reference"
    model_dir.mkdir(parents=True)

    with pytest.raises(ValueError) as exc_info:
        export.generate_readme({}, artifacts_dir, model_dir)
    assert "strict artifact provenance signature" in str(exc_info.value).lower()


def test_export_readme_fails_closed_on_mismatched_model_dir(tmp_path) -> None:
    export = _load_pipeline_module("11_export_onnx.py")
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)

    target_model_dir = tmp_path / "models" / "target"
    other_model_dir = tmp_path / "models" / "other"
    target_model_dir.mkdir(parents=True)
    other_model_dir.mkdir(parents=True)

    baseline_payload = {
        "provenance": {
            "model_dir": str(other_model_dir),
            "git_hash": "deadbeef",
            "data_snapshot_id": "snapshot-a",
            "preprocessing_version": "train-pre",
        },
        "overall": {"50": {"full_50": {"pearson_r": 0.12}}},
    }
    with open(artifacts_dir / "baseline_comparison_results.json", "w") as f:
        json.dump(baseline_payload, f, indent=2)

    config = {
        "provenance": {
            "git_hash": "deadbeef",
            "data_snapshot_id": "snapshot-a",
            "preprocessing_version": "train-pre",
        }
    }
    with pytest.raises(ValueError) as exc_info:
        export.generate_readme(config, artifacts_dir, target_model_dir)
    assert "provenance does not match" in str(exc_info.value).lower()


def test_export_readme_fails_closed_on_mismatched_signature(tmp_path) -> None:
    export = _load_pipeline_module("11_export_onnx.py")
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    model_dir = tmp_path / "models" / "target"
    model_dir.mkdir(parents=True)

    with open(artifacts_dir / "baseline_comparison_results.json", "w") as f:
        json.dump(
            {
                "provenance": {
                    "model_dir": str(model_dir),
                    "git_hash": "otherhash",
                    "data_snapshot_id": "snapshot-other",
                    "preprocessing_version": "other-pre",
                },
                "overall": {"50": {"full_50": {"pearson_r": 0.88}}},
            },
            f,
            indent=2,
        )

    config = {
        "provenance": {
            "git_hash": "targethash",
            "data_snapshot_id": "snapshot-target",
            "preprocessing_version": "target-pre",
        }
    }
    with pytest.raises(ValueError) as exc_info:
        export.generate_readme(config, artifacts_dir, model_dir)
    assert "provenance does not match" in str(exc_info.value).lower()


def test_export_generate_config_ignores_validation_from_other_model_dir(tmp_path) -> None:
    export = _load_pipeline_module("11_export_onnx.py")
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    model_dir = tmp_path / "models" / "target"
    other_model_dir = tmp_path / "models" / "other"
    model_dir.mkdir(parents=True)
    other_model_dir.mkdir(parents=True)

    _write_training_report(
        model_dir / "training_report.json",
        item_info_sha256="a" * 64,
        hyperparameters={"n_estimators": 111},
    )

    with open(artifacts_dir / "validation_results.json", "w") as f:
        json.dump(
            {
                "provenance": {"model_dir": str(other_model_dir)},
                "metrics": {"overall": {"n": 99999}},
            },
            f,
            indent=2,
        )

    config = export.generate_config(
        model_dir,
        artifacts_dir,
        {
            "git_hash": "deadbeef",
            "data_snapshot_id": "snapshot-a",
        },
        {domain: {"mean": 3.0, "sd": 1.0} for domain in DOMAINS},
    )
    assert config.get("hyperparameters", {}).get("n_estimators") == 111
    assert config.get("provenance", {}).get("n_test", 0) in (0, None)


def test_export_generate_config_ignores_validation_from_mismatched_signature(tmp_path) -> None:
    export = _load_pipeline_module("11_export_onnx.py")
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    model_dir = tmp_path / "models" / "target"
    model_dir.mkdir(parents=True)

    with open(model_dir / "training_report.json", "w") as f:
        json.dump(
            {
                "provenance": {
                    "script": "07_train.py",
                    "git_hash": "trainhash123",
                    "data_snapshot_id": "snapshot-a",
                    "preprocessing_version": "train-pre",
                },
                "config": {"hyperparameters": {"n_estimators": 100}},
                "data": {"item_info_sha256": "a" * 64, "n_test": 0},
            },
            f,
            indent=2,
        )

    with open(artifacts_dir / "validation_results.json", "w") as f:
        json.dump(
            {
                "provenance": {
                    "model_dir": str(model_dir),
                    "git_hash": "different-hash",
                    "data_snapshot_id": "snapshot-a",
                    "preprocessing_version": "train-pre",
                },
                "metrics": {"overall": {"n": 43210}},
            },
            f,
            indent=2,
        )

    config = export.generate_config(
        model_dir,
        artifacts_dir,
        {
            "git_hash": "trainhash123",
            "data_snapshot_id": "snapshot-a",
            "preprocessing_version": "train-pre",
        },
        {domain: {"mean": 3.0, "sd": 1.0} for domain in DOMAINS},
    )
    assert config.get("provenance", {}).get("n_test", 0) in (0, None)


def test_export_generate_config_logs_warning_on_invalid_validation_payload(
    tmp_path,
    caplog,
) -> None:
    export = _load_pipeline_module("11_export_onnx.py")
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    model_dir = tmp_path / "models" / "target"
    model_dir.mkdir(parents=True)

    with open(model_dir / "training_report.json", "w") as f:
        json.dump(
            {
                "provenance": {
                    "script": "07_train.py",
                    "git_hash": "trainhash123",
                    "data_snapshot_id": "snapshot-a",
                    "preprocessing_version": "train-pre",
                },
                "config": {"hyperparameters": {"n_estimators": 100}},
                "data": {"item_info_sha256": "a" * 64, "n_test": 0},
            },
            f,
            indent=2,
        )

    # Malformed artifact should not crash generate_config, but should emit a warning.
    (artifacts_dir / "validation_results.json").write_text("{invalid-json", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        config = export.generate_config(
            model_dir,
            artifacts_dir,
            {
                "git_hash": "trainhash123",
                "data_snapshot_id": "snapshot-a",
                "preprocessing_version": "train-pre",
            },
            {domain: {"mean": 3.0, "sd": 1.0} for domain in DOMAINS},
        )

    assert config.get("provenance", {}).get("n_test", 0) in (0, None)
    assert any(
        "Skipping n_test backfill" in record.message for record in caplog.records
    )


def test_export_generate_config_uses_training_report_hyperparams_not_global_tune(tmp_path) -> None:
    export = _load_pipeline_module("11_export_onnx.py")
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    model_dir = tmp_path / "models" / "target"
    model_dir.mkdir(parents=True)

    _write_training_report(
        model_dir / "training_report.json",
        item_info_sha256="a" * 64,
        hyperparameters={"n_estimators": 321, "max_depth": 7},
    )
    with open(artifacts_dir / "tuned_params.json", "w") as f:
        json.dump({"hyperparameters": {"n_estimators": 9999, "max_depth": 1}}, f, indent=2)

    config = export.generate_config(
        model_dir,
        artifacts_dir,
        {
            "git_hash": "deadbeef",
            "data_snapshot_id": "snapshot-a",
        },
        {domain: {"mean": 3.0, "sd": 1.0} for domain in DOMAINS},
    )
    assert config.get("hyperparameters", {}).get("n_estimators") == 321
    assert config.get("hyperparameters", {}).get("max_depth") == 7


def test_export_generate_config_prefers_training_report_git_hash(tmp_path) -> None:
    export = _load_pipeline_module("11_export_onnx.py")
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    model_dir = tmp_path / "models" / "target"
    model_dir.mkdir(parents=True)

    with open(model_dir / "training_report.json", "w") as f:
        json.dump(
            {
                "provenance": {
                    "script": "07_train.py",
                    "git_hash": "trainhash123",
                    "data_snapshot_id": "snapshot-a",
                },
                "config": {"hyperparameters": {"n_estimators": 100}},
                "data": {"item_info_sha256": "a" * 64},
            },
            f,
            indent=2,
        )

    config = export.generate_config(
        model_dir,
        artifacts_dir,
        {
            "git_hash": "exporthash999",
            "data_snapshot_id": "snapshot-a",
        },
        {domain: {"mean": 3.0, "sd": 1.0} for domain in DOMAINS},
    )
    assert config.get("provenance", {}).get("git_hash") == "trainhash123"


def test_simulate_main_handles_missing_inputs_gracefully(tmp_path, monkeypatch) -> None:
    simulate = _load_pipeline_module("10_simulate.py")
    monkeypatch.setattr(simulate, "PACKAGE_ROOT", tmp_path)
    (tmp_path / "artifacts").mkdir(parents=True)

    # Missing data/processed/item_info.json and test.parquet should not traceback.
    monkeypatch.setattr(sys, "argv", ["10_simulate.py", "--data-dir", "data/processed", "--n-sample", "1"])
    rc = simulate.main()
    assert rc == 1


def test_simulate_calibration_gate_not_brittle_to_sem_threshold() -> None:
    simulate = _load_pipeline_module("10_simulate.py")
    cfg = simulate.AdaptiveConfig(
        min_items=8,
        max_items=50,
        min_items_per_domain=4,
        use_ci_stopping=False,
        use_sem_stopping=True,
        sem_threshold=0.40,
        selection_strategy="correlation_ranked",
    )
    assert simulate.calibration_matches_runtime_defaults(cfg) is True


def test_simulate_sem_sweep_writes_provenance(tmp_path, monkeypatch) -> None:
    simulate = _load_pipeline_module("10_simulate.py")
    monkeypatch.setattr(simulate, "PACKAGE_ROOT", tmp_path)

    model_dir = tmp_path / "models" / "reference"
    data_dir = tmp_path / "data" / "processed"
    artifacts_dir = tmp_path / "artifacts"
    model_dir.mkdir(parents=True)
    _write_training_report(model_dir / "training_report.json")
    data_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)

    pd.DataFrame({"x": [1]}).to_parquet(data_dir / "test.parquet", index=False)
    dummy_models = {
        domain: {"q05": object(), "q50": object(), "q95": object()}
        for domain in DOMAINS
    }
    monkeypatch.setattr(simulate, "load_domain_models", lambda *_args, **_kwargs: dummy_models)
    monkeypatch.setattr(simulate, "load_norms", lambda *_args, **_kwargs: {d: {"mean": 3.0, "sd": 0.8} for d in DOMAINS})
    monkeypatch.setattr(
        simulate,
        "load_item_info_for_model_bundle",
        lambda *_args, **_kwargs: (
            {
                "first_item": {"id": "ext1"},
                "item_pool": [{"id": c, "home_domain": c[:3]} for c in ITEM_COLUMNS],
                "inter_item_r_bar": {d: 0.3 for d in DOMAINS},
            },
            data_dir / "item_info.json",
            "f" * 64,
        ),
    )
    monkeypatch.setattr(
        simulate,
        "load_test_data",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                **{col: [3.0] for col in ITEM_COLUMNS},
                **{f"{domain}_percentile": [50.0] for domain in DOMAINS},
            }
        ),
    )
    monkeypatch.setattr(simulate, "load_calibration_params", lambda *_args, **_kwargs: ({}, "none"))
    monkeypatch.setattr(
        simulate,
        "run_sem_threshold_sweep",
        lambda *_args, **_kwargs: {
            "thresholds": [0.45],
            "per_threshold": {
                "0.45": {
                    "analysis": {
                        "n_respondents": 1,
                        "items_to_convergence": {
                            "mean": 20.0,
                            "median": 20.0,
                            "std": 0.0,
                            "min": 20,
                            "max": 20,
                            "percentile_90": 20.0,
                        },
                        "overall_metrics": {
                            "mae": 4.0,
                            "within_5_pct": 0.8,
                            "coverage_90": 0.9,
                        },
                        "stop_reasons": {"sem_threshold_met": 1},
                    },
                    "domain_sem_at_stop": {d: 0.45 for d in DOMAINS},
                    "sem_stopped_pct": 100.0,
                }
            },
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["10_simulate.py", "--data-dir", "data/processed", "--sweep-sem-thresholds", "--n-sample", "1"],
    )
    rc = simulate.main()
    assert rc == 0

    with open(artifacts_dir / "sem_threshold_sweep.json") as f:
        payload = json.load(f)
    provenance = payload.get("provenance", {})
    assert provenance.get("model_dir") == str(model_dir)
    assert provenance.get("item_info_sha256") == "f" * 64


def test_download_verify_zip_integrity_checks_sha(tmp_path, monkeypatch) -> None:
    download = _load_pipeline_module("01_download.py")
    zip_path = tmp_path / "dataset.zip"
    zip_path.write_bytes(b"zip-bytes")

    good_hash = download._file_sha256(zip_path)
    monkeypatch.setattr(download, "EXPECTED_ZIP_SHA256", good_hash)
    download._verify_zip_integrity(zip_path)

    monkeypatch.setattr(download, "EXPECTED_ZIP_SHA256", "0" * 64)
    try:
        download._verify_zip_integrity(zip_path)
        raise AssertionError("Expected checksum mismatch to fail closed")
    except ValueError as exc:
        assert "sha-256 mismatch" in str(exc).lower()


def test_download_main_verifies_zip_even_when_already_extracted(tmp_path, monkeypatch) -> None:
    download = _load_pipeline_module("01_download.py")
    raw_dir = tmp_path / "data" / "raw"
    extract_dir = raw_dir / "IPIP-FFM-data-8Nov2018"
    extract_dir.mkdir(parents=True)

    zip_path = raw_dir / "IPIP-FFM-data-8Nov2018.zip"
    csv_path = extract_dir / "data-final.csv"
    zip_path.write_bytes(b"zip-bytes")
    csv_path.write_text("already-extracted", encoding="utf-8")

    monkeypatch.setattr(download, "RAW_DIR", raw_dir)
    monkeypatch.setattr(download, "EXTRACT_DIR", extract_dir)
    monkeypatch.setattr(download, "ZIP_PATH", zip_path)
    monkeypatch.setattr(download, "CSV_PATH", csv_path)
    monkeypatch.setattr(download, "EXPECTED_ZIP_SHA256", "0" * 64)

    rc = download.main()
    assert rc == 1


def test_download_verify_extracted_csv_matches_zip_member(tmp_path) -> None:
    download = _load_pipeline_module("01_download.py")
    zip_path = tmp_path / "dataset.zip"
    csv_path = tmp_path / "data-final.csv"
    payload = b"a\tb\n1\t2\n"

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("IPIP-FFM-data-8Nov2018/data-final.csv", payload)
    csv_path.write_bytes(payload)

    download._verify_extracted_csv_matches_zip(zip_path, csv_path)

    csv_path.write_bytes(payload + b"3\t4\n")
    with pytest.raises(ValueError) as exc_info:
        download._verify_extracted_csv_matches_zip(zip_path, csv_path)
    assert "mismatch" in str(exc_info.value).lower()


def test_download_safe_extract_rejects_path_traversal(tmp_path) -> None:
    download = _load_pipeline_module("01_download.py")
    zip_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../escape.txt", "x")

    try:
        download._safe_extract_all(zip_path, tmp_path / "dest")
        raise AssertionError("Expected unsafe ZIP path to fail closed")
    except ValueError as exc:
        assert "unsafe zip member path" in str(exc).lower()


def test_load_sqlite_main_writes_provenance_metadata(tmp_path, monkeypatch) -> None:
    load = _load_pipeline_module("02_load_sqlite.py")

    raw_dir = tmp_path / "data" / "raw"
    extract_dir = raw_dir / "IPIP-FFM-data-8Nov2018"
    processed_dir = tmp_path / "data" / "processed"
    extract_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    zip_path = raw_dir / "IPIP-FFM-data-8Nov2018.zip"
    csv_path = extract_dir / "data-final.csv"
    db_path = processed_dir / "ipip_bffm.db"
    meta_path = processed_dir / "load_metadata.json"

    zip_path.write_bytes(b"verified-source-zip")

    csv_item_cols = load.get_csv_item_columns()
    n_rows = 3
    data: dict[str, list[Any]] = {}
    for col_idx, col in enumerate(csv_item_cols):
        data[col] = [int(((row_idx + col_idx) % 5) + 1) for row_idx in range(n_rows)]
    # Force one invalid response so item filtering is exercised.
    data[csv_item_cols[0]][-1] = 0
    # IPC column: row 1 has duplicate IP so IPC filter drops it.
    data["IPC"] = [1, 2, 1]
    data["country"] = ["US", "CA", "GB"]
    pd.DataFrame(data).to_csv(csv_path, sep="\t", index=False)

    monkeypatch.setattr(load, "ZIP_PATH", zip_path)
    monkeypatch.setattr(load, "CSV_PATH", csv_path)
    monkeypatch.setattr(load, "DB_DIR", processed_dir)
    monkeypatch.setattr(load, "DB_PATH", db_path)
    monkeypatch.setattr(load, "LOAD_METADATA_PATH", meta_path)

    rc = load.main()
    assert rc == 0
    assert db_path.exists()
    assert meta_path.exists()

    with open(meta_path) as f:
        payload = json.load(f)

    assert payload["row_counts"]["n_raw"] == 3
    assert payload["row_counts"]["n_valid_items"] == 2
    assert payload["row_counts"]["n_valid"] == 1
    assert payload["row_counts"]["n_dropped_invalid_items"] == 1
    assert payload["row_counts"]["n_dropped_ipc_filter"] == 1
    assert payload["row_counts"]["n_dropped"] == 2
    assert payload["inputs"]["zip_sha256"] == load._file_sha256(zip_path)
    assert payload["inputs"]["csv_sha256"] == load._file_sha256(csv_path)
    assert payload["outputs"]["db_sha256"] == load._file_sha256(db_path)
    provenance = payload.get("provenance", {})
    assert provenance.get("script") == "02_load_sqlite.py"
    assert isinstance(provenance.get("data_snapshot_id"), str)
    assert provenance.get("data_snapshot_id")


def test_prepare_write_metadata_embeds_provenance_and_split_hashes(tmp_path) -> None:
    prepare = _load_pipeline_module("04_prepare_data.py")

    output_dir = tmp_path / "data" / "processed" / "canonical_v1"
    output_dir.mkdir(parents=True)
    db_path = tmp_path / "data" / "processed" / "ipip_bffm.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"sqlite-bytes")

    df_all = _make_dataset(n_rows=20)
    train_df = df_all.iloc[:12].copy()
    val_df = df_all.iloc[12:16].copy()
    test_df = df_all.iloc[16:].copy()

    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    args = argparse.Namespace(
        data_snapshot=None,
        data_snapshot_id=None,
        preprocess_tag=None,
        bootstrap_b=None,
        bootstrap_seed=None,
    )
    validation = {domain: {"max_mean_diff": 0.01, "ks_statistic": 0.01, "ks_pvalue": 0.5} for domain in DOMAINS}

    prepare.write_metadata(
        output_dir=output_dir,
        df_all=df_all,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        validation=validation,
        seed=42,
        test_size=0.15,
        val_size=0.15,
        norms_path=db_path,
        norms_sha256=file_sha256(db_path),
        db_path=db_path,
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        args=args,
    )

    meta_path = output_dir / "split_metadata.json"
    with open(meta_path) as f:
        payload = json.load(f)

    train_sha = file_sha256(train_path)
    val_sha = file_sha256(val_path)
    test_sha = file_sha256(test_path)
    expected_sig = hashlib.sha256(
        f"train={train_sha}\nval={val_sha}\ntest={test_sha}\n".encode("utf-8")
    ).hexdigest()

    assert payload["provenance"]["script"] == "04_prepare_data.py"
    assert payload["inputs"]["sqlite_db"]["sha256"] == file_sha256(db_path)
    assert payload["outputs"]["train"]["sha256"] == train_sha
    assert payload["outputs"]["val"]["sha256"] == val_sha
    assert payload["outputs"]["test"]["sha256"] == test_sha
    assert payload["split_signature"] == expected_sig
    assert payload["train_sha256"] == train_sha
    assert payload["val_sha256"] == val_sha
    assert payload["test_sha256"] == test_sha
    assert payload["total_valid"] == len(df_all)

    # canonical_v1 schema: new split identity + train-only norms provenance.
    assert payload["split_id"] == prepare.CANONICAL_SPLIT_ID == "canonical_v1"
    assert payload["split_scheme"] == prepare.SPLIT_SCHEME == "random"
    assert payload["norms_sha256"] == file_sha256(db_path)
    assert payload["inputs"]["norms"]["sha256"] == file_sha256(db_path)
    assert payload["norms_path"]
    # Removed stratified-regime fields must not reappear.
    assert "stratification_scheme" not in payload
    assert "n_strata" not in payload


def test_prepare_load_from_sqlite_enforces_stable_row_order(tmp_path, monkeypatch) -> None:
    prepare = _load_pipeline_module("04_prepare_data.py")
    db_path = tmp_path / "ipip_bffm.db"
    sqlite3.connect(str(db_path)).close()

    captured_queries: list[str] = []

    def _fake_read_sql_query(query, conn):  # type: ignore[no-untyped-def]
        captured_queries.append(query)
        return pd.DataFrame({"respondent_id": [2, 1]})

    monkeypatch.setattr(prepare.pd, "read_sql_query", _fake_read_sql_query)

    prepare.load_from_sqlite(db_path, sample=None)
    prepare.load_from_sqlite(db_path, sample=5)

    assert len(captured_queries) == 2
    assert "ORDER BY respondent_id" in captured_queries[0]
    assert "ORDER BY respondent_id" in captured_queries[1]
    assert "LIMIT 5" in captured_queries[1]


def test_validate_model_data_split_provenance_fails_on_test_hash_mismatch(tmp_path) -> None:
    validate = _load_pipeline_module("08_validate.py")
    model_dir = tmp_path / "models" / "reference"
    data_dir = tmp_path / "data" / "processed"
    model_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    frame = _make_dataset(n_rows=16)
    frame.iloc[:8].to_parquet(data_dir / "train.parquet", index=False)
    frame.iloc[8:12].to_parquet(data_dir / "val.parquet", index=False)
    frame.iloc[12:].to_parquet(data_dir / "test.parquet", index=False)

    train_sha = file_sha256(data_dir / "train.parquet")
    val_sha = file_sha256(data_dir / "val.parquet")
    test_sha = file_sha256(data_dir / "test.parquet")
    split_signature = _build_split_signature(
        train_sha256=train_sha,
        val_sha256=val_sha,
        test_sha256=test_sha,
    )
    _write_training_report(
        model_dir / "training_report.json",
        data_overrides={
            "train_sha256": train_sha,
            "val_sha256": val_sha,
            "test_sha256": test_sha,
            "split_signature": split_signature,
        },
    )

    observed_test_sha, observed_sig = validate._verify_model_data_split_provenance(
        model_dir=model_dir,
        data_dir=data_dir,
    )
    assert observed_test_sha == test_sha
    assert observed_sig == split_signature

    frame.iloc[:4].to_parquet(data_dir / "test.parquet", index=False)
    with pytest.raises(ValueError) as exc_info:
        validate._verify_model_data_split_provenance(model_dir=model_dir, data_dir=data_dir)
    assert "test split" in str(exc_info.value).lower()


def test_baselines_model_data_split_provenance_fails_on_signature_mismatch(tmp_path) -> None:
    baselines = _load_pipeline_module("09_baselines.py")
    model_dir = tmp_path / "models" / "reference"
    data_dir = tmp_path / "data" / "processed"
    model_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    frame = _make_dataset(n_rows=18)
    frame.iloc[:9].to_parquet(data_dir / "train.parquet", index=False)
    frame.iloc[9:13].to_parquet(data_dir / "val.parquet", index=False)
    frame.iloc[13:].to_parquet(data_dir / "test.parquet", index=False)

    test_sha = file_sha256(data_dir / "test.parquet")
    _write_training_report(
        model_dir / "training_report.json",
        data_overrides={
            "test_sha256": test_sha,
            "split_signature": "0" * 64,
        },
    )

    with pytest.raises(ValueError) as exc_info:
        baselines._verify_model_data_split_provenance(model_dir=model_dir, data_dir=data_dir)
    assert "split signature mismatch" in str(exc_info.value).lower()


def test_simulate_model_data_split_provenance_accepts_matching_hashes(tmp_path) -> None:
    simulate = _load_pipeline_module("10_simulate.py")
    model_dir = tmp_path / "models" / "reference"
    data_dir = tmp_path / "data" / "processed"
    model_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    frame = _make_dataset(n_rows=18)
    frame.iloc[:9].to_parquet(data_dir / "train.parquet", index=False)
    frame.iloc[9:13].to_parquet(data_dir / "val.parquet", index=False)
    frame.iloc[13:].to_parquet(data_dir / "test.parquet", index=False)

    train_sha = file_sha256(data_dir / "train.parquet")
    val_sha = file_sha256(data_dir / "val.parquet")
    test_sha = file_sha256(data_dir / "test.parquet")
    split_signature = _build_split_signature(
        train_sha256=train_sha,
        val_sha256=val_sha,
        test_sha256=test_sha,
    )
    _write_training_report(
        model_dir / "training_report.json",
        data_overrides={
            "train_sha256": train_sha,
            "val_sha256": val_sha,
            "test_sha256": test_sha,
            "split_signature": split_signature,
        },
    )

    observed_test_sha, observed_sig = simulate._verify_model_data_split_provenance(
        model_dir=model_dir,
        data_dir=data_dir,
    )
    assert observed_test_sha == test_sha
    assert observed_sig == split_signature


def test_upload_bundle_validation_rejects_unexpected_files(tmp_path) -> None:
    upload = _load_pipeline_module("13_upload_hf.py")
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True)

    outputs = [
        f"{domain}_{q}"
        for domain in DOMAINS
        for q in ("q05", "q50", "q95")
    ]
    config_payload = {
        "model_file": "model.onnx",
        "outputs": outputs,
        "scores_output": "scores",
        "provenance": {
            "git_hash": "deadbeef",
            "data_snapshot_id": "snapshot-a",
            "preprocessing_version": "deadbeef",
            "model_dir": "models/reference",
            "training_script": "07_train.py",
        },
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_payload, f, indent=2)
    (output_dir / "README.md").write_text("ok", encoding="utf-8")
    (output_dir / "model.onnx").write_bytes(b"onnx")

    # Compute actual SHA-256s to satisfy cross-check validation
    from lib.provenance import file_sha256 as _sha256
    config_sha = _sha256(output_dir / "config.json")
    model_sha = _sha256(output_dir / "model.onnx")

    with open(output_dir / "provenance.json", "w") as f:
        json.dump(
            {
                "schema_version": 1,
                "export": {
                    "script": "11_export_onnx.py",
                    "git_hash": "deadbeef",
                    "data_snapshot_id": "snapshot-a",
                    "preprocessing_version": "deadbeef",
                    "model_dir": "models/reference",
                },
                "training": {"provenance": {"script": "07_train.py"}},
                "artifacts": {
                    "config_json_sha256": config_sha,
                    "model_onnx_sha256": model_sha,
                },
            },
            f,
        )

    files = upload._validate_output_bundle(output_dir)
    assert len(files) == 4

    (output_dir / "notes.txt").write_text("unexpected", encoding="utf-8")
    try:
        upload._validate_output_bundle(output_dir)
        raise AssertionError("Expected unexpected output file to fail closed")
    except ValueError as exc:
        assert "unexpected files" in str(exc).lower()


def test_upload_main_resolves_relative_output_dir_to_package_root(tmp_path, monkeypatch) -> None:
    import types

    upload = _load_pipeline_module("13_upload_hf.py")
    monkeypatch.setattr(upload, "PACKAGE_ROOT", tmp_path)
    monkeypatch.setenv("HF_TOKEN", "test-token")

    captured: dict[str, Any] = {}

    class _FakeApi:
        def __init__(self, token: str) -> None:
            captured["token"] = token

        def create_repo(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            captured["repo"] = kwargs.get("repo_id")

        def upload_file(self, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            raise AssertionError("No files should be uploaded in this unit test")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(
        HfApi=_FakeApi, CommitOperationAdd=None, CommitOperationDelete=None,
    ))
    def _fake_validate_output_bundle(output_dir):  # type: ignore[no-untyped-def]
        captured["output_dir"] = output_dir
        return []

    def _fake_discover_variants(output_dir):  # type: ignore[no-untyped-def]
        captured["output_dir"] = output_dir
        return [("", output_dir)]

    monkeypatch.setattr(upload, "_validate_output_bundle", _fake_validate_output_bundle)
    monkeypatch.setattr(upload, "_discover_variants", _fake_discover_variants)
    monkeypatch.setattr(
        sys,
        "argv",
        ["13_upload_hf.py", "--repo-id", "org/repo", "--output-dir", "output/custom"],
    )

    upload.main()
    assert captured.get("repo") == "org/repo"
    assert captured.get("output_dir") == tmp_path / "output" / "custom"


def test_upload_revision_creates_branch_and_scopes_commit(tmp_path, monkeypatch) -> None:
    """--revision <branch> creates the branch (if absent) and routes the commit + the
    stale-file listing to that revision, so a release can be staged on e.g. `next`."""
    import types

    upload = _load_pipeline_module("13_upload_hf.py")
    monkeypatch.setattr(upload, "PACKAGE_ROOT", tmp_path)
    monkeypatch.setenv("HF_TOKEN", "test-token")

    # A minimal on-disk reference variant so `--variant reference` resolves a real dir.
    variant_path = tmp_path / "output" / "reference"
    variant_path.mkdir(parents=True)
    model_file = variant_path / "model.onnx"
    model_file.write_text("onnx-bytes")

    captured: dict[str, Any] = {}

    class _FakeApi:
        def __init__(self, token: str) -> None:
            captured["token"] = token

        def create_repo(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            captured["repo"] = kwargs.get("repo_id")

        def create_branch(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            captured["branch"] = (kwargs.get("branch"), kwargs.get("exist_ok"))

        def list_repo_files(self, **kwargs) -> list[str]:  # type: ignore[no-untyped-def]
            captured["list_revision"] = kwargs.get("revision")
            return []

        def create_commit(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            captured["commit_revision"] = kwargs.get("revision")
            captured["n_ops"] = len(kwargs.get("operations", []))

    # CommitOperationAdd is constructed with kwargs (path_in_repo=, path_or_fileobj=) and
    # later read via op.path_in_repo + isinstance(op, CommitOperationAdd) -> SimpleNamespace
    # satisfies both (a SimpleNamespace built from those kwargs is an instance of the class).
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(
        HfApi=_FakeApi, CommitOperationAdd=types.SimpleNamespace,
        CommitOperationDelete=types.SimpleNamespace,
    ))
    monkeypatch.setattr(upload, "_validate_output_bundle", lambda _p: [model_file])
    monkeypatch.setattr(
        sys,
        "argv",
        ["13_upload_hf.py", "--repo-id", "org/repo", "--variant", "reference", "--revision", "next"],
    )

    upload.main()
    assert captured.get("branch") == ("next", True)
    assert captured.get("list_revision") == "next"
    assert captured.get("commit_revision") == "next"
    assert captured.get("n_ops") == 1


def _write_responses_sqlite(db_path: Path, df: pd.DataFrame) -> None:
    """Write synthetic responses table with *_score columns for norms stage tests."""
    df_out = df.copy()
    n_rows = len(df_out)
    if "respondent_id" not in df_out.columns:
        df_out.insert(0, "respondent_id", range(1, n_rows + 1))
    for item_idx, item_id in enumerate(ITEM_COLUMNS):
        if item_id not in df_out.columns:
            df_out[item_id] = [int(((row_idx + item_idx) % 5) + 1) for row_idx in range(n_rows)]

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        df_out.to_sql("responses", conn, index=False, if_exists="replace")


def _write_mini_ipip_mapping(path: Path) -> None:
    """Write a valid 20-item Mini-IPIP mapping artifact for tests."""
    payload = {
        "domains": {domain: [f"{domain}{i}" for i in range(1, 5)] for domain in DOMAINS},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def test_norms_stage_computes_expected_means_and_sds_from_sqlite(tmp_path) -> None:
    norms_stage = _load_pipeline_module("03_compute_norms.py")

    df = pd.DataFrame(
        {
            "ext_score": [1.0, 2.0, 3.0, 4.0],
            "agr_score": [2.0, 3.0, 4.0, 5.0],
            "csn_score": [1.5, 2.5, 3.5, 4.5],
            "est_score": [2.0, 2.0, 4.0, 4.0],
            "opn_score": [3.0, 3.5, 4.0, 4.5],
        }
    )
    db_path = tmp_path / "ipip_bffm.db"
    _write_responses_sqlite(db_path, df)

    loaded = norms_stage._load_domain_scores_from_sqlite(db_path)
    computed = norms_stage._compute_norms(loaded)

    for domain in DOMAINS:
        col = f"{domain}_score"
        expected_mean = float(df[col].mean())
        expected_sd = float(df[col].std(ddof=1))
        assert abs(float(computed[domain]["mean"]) - expected_mean) < 1e-12
        assert abs(float(computed[domain]["sd"]) - expected_sd) < 1e-12
        assert int(computed[domain]["n"]) == len(df)


def test_norms_stage_main_writes_lock_and_meta(tmp_path, monkeypatch) -> None:
    norms_stage = _load_pipeline_module("03_compute_norms.py")
    monkeypatch.setattr(norms_stage, "PACKAGE_ROOT", tmp_path)

    _b = np.linspace(1.0, 5.0, 24)
    df = pd.DataFrame(
        {
            "ext_score": _b,
            "agr_score": np.clip(_b + 0.3, 1.0, 5.0),
            "csn_score": np.clip(5.3 - _b, 1.0, 5.0),
            "est_score": np.clip(_b * 0.8 + 0.5, 1.0, 5.0),
            "opn_score": np.clip(_b + 0.1, 1.0, 5.0),
        }
    )
    db_path = tmp_path / "data" / "processed" / "ipip_bffm.db"
    _write_responses_sqlite(db_path, df)
    _write_mini_ipip_mapping(tmp_path / "artifacts" / "mini_ipip_mapping.json")

    output_path = tmp_path / "artifacts" / "ipip_bffm_norms.json"
    meta_path = tmp_path / "artifacts" / "ipip_bffm_norms.meta.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "03_compute_norms.py",
            "--db-path",
            "data/processed/ipip_bffm.db",
            "--output",
            "artifacts/ipip_bffm_norms.json",
        ],
    )
    rc = norms_stage.main()
    assert rc == 0
    assert output_path.exists()
    assert meta_path.exists()

    with open(output_path) as f:
        payload = json.load(f)
    from lib.splits import assign_splits, CANONICAL_SEED, CANONICAL_TEST_SIZE, CANONICAL_VAL_SIZE

    labels = assign_splits(
        np.arange(1, len(df) + 1),
        seed=CANONICAL_SEED,
        test_size=CANONICAL_TEST_SIZE,
        val_size=CANONICAL_VAL_SIZE,
    )
    train_df = df.loc[labels == "train"].reset_index(drop=True)
    assert payload["n_respondents"] == len(train_df)
    assert payload["schema_version"] == 3
    assert payload["split"]["fit_on"] == "train"
    for domain in DOMAINS:
        col = f"{domain}_score"
        assert abs(payload["norms"][domain]["mean"] - float(train_df[col].mean())) < 1e-12
        assert abs(payload["norms"][domain]["sd"] - float(train_df[col].std(ddof=1))) < 1e-12
        assert payload["mini_ipip_norms"][domain]["sd"] > 0

    with open(meta_path) as f:
        meta = json.load(f)
    assert meta["check_passed"] is True
    norms_sha = file_sha256(output_path)
    assert meta["provenance"]["norms_lock_sha256"] == norms_sha
    assert meta["provenance"]["data_snapshot_id"] == f"norms_sha256:{norms_sha}"


# ---------------------------------------------------------------------------
# lib/splits.assign_splits — the leakage-critical shared split function.
# ---------------------------------------------------------------------------

def test_assign_splits_is_order_independent() -> None:
    """Same id set in any order yields the same per-id label.

    This is the property that lets stage 03 (norms) and stage 04 (prepare) agree
    on which respondents are ``train`` even though they load rows independently.
    """
    from lib.splits import assign_splits

    ids = np.arange(1, 101)
    labels_sorted = assign_splits(ids, seed=42)

    shuffled = ids.copy()
    np.random.default_rng(0).shuffle(shuffled)
    labels_shuffled = assign_splits(shuffled, seed=42)

    mapping_sorted = dict(zip(ids.tolist(), labels_sorted.tolist()))
    mapping_shuffled = dict(zip(shuffled.tolist(), labels_shuffled.tolist()))
    assert mapping_sorted == mapping_shuffled


def test_assign_splits_is_deterministic_for_fixed_seed() -> None:
    from lib.splits import assign_splits

    ids = np.arange(1, 101)
    assert np.array_equal(assign_splits(ids, seed=42), assign_splits(ids, seed=42))


def test_assign_splits_partitions_are_disjoint_and_cover_all_ids() -> None:
    from lib.splits import assign_splits

    ids = np.arange(1, 101)
    labels = assign_splits(ids, seed=42)
    assert set(labels.tolist()) == {"train", "val", "test"}
    assert len(labels) == len(ids)


def test_assign_splits_rejects_empty_id_set() -> None:
    from lib.splits import assign_splits

    with pytest.raises(ValueError):
        assign_splits(np.array([], dtype=int))


def test_assign_splits_rejects_non_unique_ids() -> None:
    from lib.splits import assign_splits

    with pytest.raises(ValueError):
        assign_splits(np.array([1, 2, 2, 3, 4, 5, 6, 7]))


def test_assign_splits_rejects_invalid_sizes() -> None:
    from lib.splits import assign_splits

    ids = np.arange(1, 101)
    with pytest.raises(ValueError):
        assign_splits(ids, test_size=0.6, val_size=0.6)
    with pytest.raises(ValueError):
        assign_splits(ids, test_size=0.0, val_size=0.15)


def test_assign_splits_rejects_small_n_with_empty_partition() -> None:
    from lib.splits import assign_splits

    with pytest.raises(ValueError):
        assign_splits(np.array([1, 2, 3]))


def test_assign_splits_proportions_are_70_15_15_and_train_is_majority() -> None:
    """Pin the split SIZES so a slice-swap regression (e.g. train collapsing to
    15% / test ballooning to 70%) fails loudly instead of passing the suite."""
    import collections

    from lib.splits import assign_splits

    ids = np.arange(1, 10001)
    counts = collections.Counter(assign_splits(ids, seed=42).tolist())
    n = len(ids)
    assert abs(counts["train"] / n - 0.70) < 0.01
    assert abs(counts["val"] / n - 0.15) < 0.01
    assert abs(counts["test"] / n - 0.15) < 0.01
    # Load-bearing: train must be the majority partition (catches a train/test swap).
    assert counts["train"] > counts["val"] and counts["train"] > counts["test"]


def test_population_signature_is_stable_and_population_sensitive() -> None:
    """The population signature is order-independent, integer-only, and changes
    when the respondent set changes (the binding stage 04 relies on)."""
    from lib.splits import population_signature

    ids = np.arange(1, 101)
    sig = population_signature(ids)
    assert sig == population_signature(ids[::-1])  # order-independent
    assert sig != population_signature(np.arange(1, 100))  # dropping one id changes it
    with pytest.raises(ValueError):
        population_signature(np.array(["a", "b"]))  # integer ids required


def _write_norms_with_split(tmp_path, **split_overrides):
    """Write a minimal norms artifact whose 'split' block can be tampered per-test."""
    from lib.splits import (
        CANONICAL_SEED,
        CANONICAL_SPLIT_ID,
        CANONICAL_TEST_SIZE,
        CANONICAL_VAL_SIZE,
    )

    split = {
        "id": CANONICAL_SPLIT_ID,
        "scheme": "random",
        "seed": CANONICAL_SEED,
        "test_size": CANONICAL_TEST_SIZE,
        "val_size": CANONICAL_VAL_SIZE,
        "fit_on": "train",
    }
    split.update(split_overrides)
    path = tmp_path / "norms.json"
    path.write_text(json.dumps({"split": split, "norms": {}}), encoding="utf-8")
    return path


def test_assert_norms_match_split_rejects_each_leaky_branch(tmp_path) -> None:
    """Lock all rejection branches of the stage-04 leakage guard so a future
    weakening (e.g. dropping the fit_on check) turns the suite red."""
    prepare = _load_pipeline_module("04_prepare_data.py")
    from lib.splits import CANONICAL_SEED, CANONICAL_TEST_SIZE, CANONICAL_VAL_SIZE

    kw = dict(seed=CANONICAL_SEED, test_size=CANONICAL_TEST_SIZE, val_size=CANONICAL_VAL_SIZE)

    # (a) no 'split' block at all
    no_split = tmp_path / "no_split.json"
    no_split.write_text(json.dumps({"norms": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        prepare.assert_norms_match_split(no_split, **kw)

    # (b) fit_on != 'train'  (c) wrong split id  (d) each param mismatch
    with pytest.raises(ValueError):
        prepare.assert_norms_match_split(_write_norms_with_split(tmp_path, fit_on="full"), **kw)
    with pytest.raises(ValueError):
        prepare.assert_norms_match_split(_write_norms_with_split(tmp_path, id="ext_est"), **kw)
    with pytest.raises(ValueError):
        prepare.assert_norms_match_split(_write_norms_with_split(tmp_path, seed=CANONICAL_SEED + 1), **kw)
    with pytest.raises(ValueError):
        prepare.assert_norms_match_split(
            _write_norms_with_split(tmp_path, test_size=CANONICAL_TEST_SIZE + 0.05), **kw
        )

    # Happy path: a matching split block does NOT raise.
    prepare.assert_norms_match_split(_write_norms_with_split(tmp_path), **kw)


def test_assert_norms_match_split_rejects_population_change(tmp_path) -> None:
    """The population-signature binding rejects a norms artifact fit on a different
    respondent set (the defense-in-depth leakage gap fix)."""
    prepare = _load_pipeline_module("04_prepare_data.py")
    from lib.splits import (
        CANONICAL_SEED,
        CANONICAL_TEST_SIZE,
        CANONICAL_VAL_SIZE,
        population_signature,
    )

    kw = dict(seed=CANONICAL_SEED, test_size=CANONICAL_TEST_SIZE, val_size=CANONICAL_VAL_SIZE)
    pop_a = np.arange(1, 1001)
    pop_b = np.arange(1, 996)  # 5 respondents removed (an upstream re-ingest)
    artifact = _write_norms_with_split(tmp_path, population_signature=population_signature(pop_a))

    # Same population -> passes; changed population -> fails closed.
    prepare.assert_norms_match_split(artifact, respondent_ids=pop_a, **kw)
    with pytest.raises(ValueError, match="population"):
        prepare.assert_norms_match_split(artifact, respondent_ids=pop_b, **kw)

    # Backward-compat: an older artifact with no signature skips the population check.
    legacy = _write_norms_with_split(tmp_path)
    prepare.assert_norms_match_split(legacy, respondent_ids=pop_b, **kw)


def test_stage04_sample_relies_on_population_guard_not_a_hard_refusal(tmp_path) -> None:
    """`make smoke` passes --sample to stage 04 (the old unconditional refusal was
    removed). Leakage-safety now rests entirely on the population_signature guard:
    a sampled split against FULL-population norms must still be rejected, while a
    sample-matched smoke norms file is accepted. This locks the smoke relaxation."""
    prepare = _load_pipeline_module("04_prepare_data.py")
    from lib.splits import (
        CANONICAL_SEED,
        CANONICAL_TEST_SIZE,
        CANONICAL_VAL_SIZE,
        population_signature,
    )

    kw = dict(seed=CANONICAL_SEED, test_size=CANONICAL_TEST_SIZE, val_size=CANONICAL_VAL_SIZE)
    full_pop = np.arange(1, 10001)  # stage 03 without --sample
    sample_pop = np.arange(1, 801)  # stage 04 --sample 800 (first N respondents)

    # Full-population norms vs a sampled split -> rejected (would be leaky/incoherent).
    full_norms = _write_norms_with_split(
        tmp_path, population_signature=population_signature(full_pop)
    )
    with pytest.raises(ValueError, match="population"):
        prepare.assert_norms_match_split(full_norms, respondent_ids=sample_pop, **kw)

    # Sample-matched smoke norms (stage 03 --sample 800) vs the same sampled split -> accepted.
    smoke_norms = _write_norms_with_split(
        tmp_path, population_signature=population_signature(sample_pop)
    )
    prepare.assert_norms_match_split(smoke_norms, respondent_ids=sample_pop, **kw)


# ---------------------------------------------------------------------------
# Stage 04 prepare — random_split / add_percentile_columns / main fail-closed.
# ---------------------------------------------------------------------------

def test_prepare_random_split_partitions_by_respondent_id() -> None:
    """random_split must match assign_splits exactly and produce disjoint partitions."""
    prepare = _load_pipeline_module("04_prepare_data.py")
    from lib.splits import assign_splits, CANONICAL_SEED, CANONICAL_TEST_SIZE, CANONICAL_VAL_SIZE

    df = _make_dataset(n_rows=40)
    df.insert(0, "respondent_id", np.arange(1, len(df) + 1))

    train_df, val_df, test_df = prepare.random_split(
        df, test_size=CANONICAL_TEST_SIZE, val_size=CANONICAL_VAL_SIZE, seed=CANONICAL_SEED
    )

    labels = assign_splits(
        df["respondent_id"].to_numpy(),
        seed=CANONICAL_SEED,
        test_size=CANONICAL_TEST_SIZE,
        val_size=CANONICAL_VAL_SIZE,
    )
    expected_train = set(df.loc[labels == "train", "respondent_id"].tolist())
    expected_val = set(df.loc[labels == "val", "respondent_id"].tolist())
    expected_test = set(df.loc[labels == "test", "respondent_id"].tolist())

    got_train = set(train_df["respondent_id"].tolist())
    got_val = set(val_df["respondent_id"].tolist())
    got_test = set(test_df["respondent_id"].tolist())

    assert got_train == expected_train
    assert got_val == expected_val
    assert got_test == expected_test
    # Disjoint and exhaustive.
    assert got_train.isdisjoint(got_val)
    assert got_train.isdisjoint(got_test)
    assert got_val.isdisjoint(got_test)
    assert got_train | got_val | got_test == set(df["respondent_id"].tolist())


def test_prepare_random_split_requires_respondent_id() -> None:
    prepare = _load_pipeline_module("04_prepare_data.py")
    df = _make_dataset(n_rows=40)  # no respondent_id column
    with pytest.raises(ValueError):
        prepare.random_split(df, test_size=0.15, val_size=0.15, seed=42)


def test_prepare_add_percentile_columns_uses_provided_norms() -> None:
    prepare = _load_pipeline_module("04_prepare_data.py")
    from lib.scoring import raw_score_to_percentile

    df = _make_dataset(n_rows=20)
    norms = {domain: {"mean": 3.0, "sd": 0.8} for domain in DOMAINS}

    out = prepare.add_percentile_columns(df, norms)
    for domain in DOMAINS:
        expected = raw_score_to_percentile(df[f"{domain}_score"].values, domain, norms=norms)
        assert np.allclose(out[f"{domain}_percentile"].values, expected)


def test_prepare_main_fails_closed_without_norms(tmp_path, monkeypatch) -> None:
    """Stage 04 returns 1 (not a leaky run) when the --norms artifact is absent."""
    prepare = _load_pipeline_module("04_prepare_data.py")
    monkeypatch.setattr(prepare, "PACKAGE_ROOT", tmp_path)

    df = _make_dataset(n_rows=40)
    db_path = tmp_path / "data" / "processed" / "ipip_bffm.db"
    _write_responses_sqlite(db_path, df)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "04_prepare_data.py",
            "--db-path",
            "data/processed/ipip_bffm.db",
            "--norms",
            "artifacts/does_not_exist.json",
            "--output-dir",
            "data/processed/canonical_v1",
        ],
    )
    assert prepare.main() == 1


def test_stage03_train_set_equals_stage04_train_parquet(tmp_path, monkeypatch) -> None:
    """End-to-end leakage guard: stage 03 fits norms on exactly the rows stage 04
    writes as ``train``.

    The redesign's headline invariant is that both stages compute the same train
    set. The two stages load rows via independent queries and only agree because
    they share :func:`lib.splits.assign_splits`. We run both against one db and
    assert that the per-domain train means recorded by stage 03 equal the
    train.parquet score means produced by stage 04 (train.parquet drops
    respondent_id, so mean equality is the available proxy for row-set equality).
    """
    norms_stage = _load_pipeline_module("03_compute_norms.py")
    monkeypatch.setattr(norms_stage, "PACKAGE_ROOT", tmp_path)

    _b = np.linspace(1.0, 5.0, 40)
    df = pd.DataFrame(
        {
            "ext_score": _b,
            "agr_score": np.clip(_b + 0.3, 1.0, 5.0),
            "csn_score": np.clip(5.3 - _b, 1.0, 5.0),
            "est_score": np.clip(_b * 0.8 + 0.5, 1.0, 5.0),
            "opn_score": np.clip(_b + 0.1, 1.0, 5.0),
        }
    )
    db_path = tmp_path / "data" / "processed" / "ipip_bffm.db"
    _write_responses_sqlite(db_path, df)
    _write_mini_ipip_mapping(tmp_path / "artifacts" / "mini_ipip_mapping.json")

    norms_output = tmp_path / "artifacts" / "ipip_bffm_norms.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "03_compute_norms.py",
            "--db-path",
            "data/processed/ipip_bffm.db",
            "--output",
            "artifacts/ipip_bffm_norms.json",
        ],
    )
    assert norms_stage.main() == 0
    with open(norms_output) as f:
        norms_payload = json.load(f)

    prepare = _load_pipeline_module("04_prepare_data.py")
    monkeypatch.setattr(prepare, "PACKAGE_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "04_prepare_data.py",
            "--db-path",
            "data/processed/ipip_bffm.db",
            "--norms",
            "artifacts/ipip_bffm_norms.json",
            "--output-dir",
            "data/processed/canonical_v1",
        ],
    )
    assert prepare.main() == 0

    train_parquet = tmp_path / "data" / "processed" / "canonical_v1" / "train.parquet"
    train = pd.read_parquet(train_parquet)
    for domain in DOMAINS:
        norm_mean = norms_payload["norms"][domain]["mean"]
        parquet_mean = float(train[f"{domain}_score"].mean())
        assert abs(norm_mean - parquet_mean) < 1e-9, domain


def test_norms_stage_main_check_passes_against_existing_lock(tmp_path, monkeypatch) -> None:
    norms_stage = _load_pipeline_module("03_compute_norms.py")
    monkeypatch.setattr(norms_stage, "PACKAGE_ROOT", tmp_path)

    _b = np.linspace(1.0, 5.0, 24)
    df = pd.DataFrame(
        {
            "ext_score": _b,
            "agr_score": np.clip(_b + 0.3, 1.0, 5.0),
            "csn_score": np.clip(5.3 - _b, 1.0, 5.0),
            "est_score": np.clip(_b * 0.8 + 0.5, 1.0, 5.0),
            "opn_score": np.clip(_b + 0.1, 1.0, 5.0),
        }
    )
    db_path = tmp_path / "data" / "processed" / "ipip_bffm.db"
    _write_responses_sqlite(db_path, df)
    _write_mini_ipip_mapping(tmp_path / "artifacts" / "mini_ipip_mapping.json")

    # First run writes the lock file.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "03_compute_norms.py",
            "--db-path",
            "data/processed/ipip_bffm.db",
            "--output",
            "artifacts/ipip_bffm_norms.json",
        ],
    )
    assert norms_stage.main() == 0

    # Second run checks against the lock.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "03_compute_norms.py",
            "--db-path",
            "data/processed/ipip_bffm.db",
            "--output",
            "artifacts/ipip_bffm_norms.json",
            "--check",
        ],
    )
    assert norms_stage.main() == 0


def test_norms_stage_main_check_fails_when_lock_missing(tmp_path, monkeypatch) -> None:
    norms_stage = _load_pipeline_module("03_compute_norms.py")
    monkeypatch.setattr(norms_stage, "PACKAGE_ROOT", tmp_path)

    _b = np.linspace(1.0, 5.0, 24)
    df = pd.DataFrame(
        {
            "ext_score": _b,
            "agr_score": np.clip(_b + 0.3, 1.0, 5.0),
            "csn_score": np.clip(5.3 - _b, 1.0, 5.0),
            "est_score": np.clip(_b * 0.8 + 0.5, 1.0, 5.0),
            "opn_score": np.clip(_b + 0.1, 1.0, 5.0),
        }
    )
    db_path = tmp_path / "data" / "processed" / "ipip_bffm.db"
    _write_responses_sqlite(db_path, df)
    _write_mini_ipip_mapping(tmp_path / "artifacts" / "mini_ipip_mapping.json")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "03_compute_norms.py",
            "--db-path",
            "data/processed/ipip_bffm.db",
            "--output",
            "artifacts/ipip_bffm_norms.json",
            "--check",
        ],
    )
    assert norms_stage.main() == 1


def test_norms_stage_main_check_fails_on_drift(tmp_path, monkeypatch) -> None:
    norms_stage = _load_pipeline_module("03_compute_norms.py")
    monkeypatch.setattr(norms_stage, "PACKAGE_ROOT", tmp_path)

    _b = np.linspace(1.0, 5.0, 24)
    df = pd.DataFrame(
        {
            "ext_score": _b,
            "agr_score": np.clip(_b + 0.3, 1.0, 5.0),
            "csn_score": np.clip(5.3 - _b, 1.0, 5.0),
            "est_score": np.clip(_b * 0.8 + 0.5, 1.0, 5.0),
            "opn_score": np.clip(_b + 0.1, 1.0, 5.0),
        }
    )
    db_path = tmp_path / "data" / "processed" / "ipip_bffm.db"
    _write_responses_sqlite(db_path, df)
    _write_mini_ipip_mapping(tmp_path / "artifacts" / "mini_ipip_mapping.json")

    output_path = tmp_path / "artifacts" / "ipip_bffm_norms.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "schema_version": 2,
                "n_respondents": len(df),
                "norms": {domain: {"mean": 0.0, "sd": 1.0} for domain in DOMAINS},
                "mini_ipip_norms": {domain: {"mean": 0.0, "sd": 1.0} for domain in DOMAINS},
            },
            f,
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "03_compute_norms.py",
            "--db-path",
            "data/processed/ipip_bffm.db",
            "--output",
            "artifacts/ipip_bffm_norms.json",
            "--check",
            "--tolerance",
            "1e-12",
        ],
    )
    rc = norms_stage.main()
    assert rc == 1


def test_notes_calibration_policy_parses_current_baselines_schema(
    tmp_path,
    monkeypatch,
) -> None:
    notes = _load_paper_module("generate_notes_data.py")
    monkeypatch.setattr(notes, "PACKAGE_ROOT", tmp_path)
    monkeypatch.setattr(notes, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(notes, "RESEARCH_SUMMARY_PATH", tmp_path / "artifacts" / "research_summary.json")

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)

    # gen_calibration_policy reads data via load_reference_notes_inputs()
    # which loads from research_summary.json, not directly from
    # baseline_comparison_results.json.
    with open(artifacts_dir / "research_summary.json", "w") as f:
        json.dump(
            {
                "variants": {},
                "reference_notes_inputs": {
                    "baseline_comparison_results": {
                        "config": {
                            "calibration_policy": {
                                "n_items_50_or_more": "full_50",
                                "n_items_below_50": "sparse_20_balanced",
                                "fallback_without_sparse_calibration": "none",
                            }
                        }
                    }
                },
            },
            f,
            indent=2,
        )

    table = notes.gen_calibration_policy()
    assert "50+ items (full scale)" in table
    assert "Below 50 items" in table
    assert "Fallback (if sparse calibration missing)" in table
    assert "`sparse_20_balanced`" in table


def test_notes_ml_vs_averaging_per_domain_decomposition() -> None:
    """A4.2: per-domain matched-item table isolates the scoring gain and shows
    Emotional Stability is a Mini-IPIP-item win under ML scoring."""
    notes = _load_paper_module("generate_notes_data.py")
    notes_inputs = {
        "ml_vs_averaging_comparison": {
            "comparisons": [
                {
                    "method": "domain_balanced",
                    "n_items": 20,
                    "ml_per_domain": {"ext": 0.947, "agr": 0.920, "csn": 0.919, "est": 0.9366, "opn": 0.910},
                    "avg_per_domain": {"ext": 0.939, "agr": 0.911, "csn": 0.909, "est": 0.929, "opn": 0.842},
                },
                {
                    "method": "mini_ipip",
                    "n_items": 20,
                    "ml_per_domain": {"ext": 0.945, "agr": 0.918, "csn": 0.915, "est": 0.9434, "opn": 0.860},
                    "avg_per_domain": {"ext": 0.939, "agr": 0.911, "csn": 0.909, "est": 0.929, "opn": 0.842},
                },
            ]
        }
    }
    table = notes._gen_ml_vs_averaging_per_domain_from_notes_inputs(notes_inputs)
    assert "scoring" in table  # the scoring-gain columns are present
    # EST: the Mini-IPIP item set under ML (0.9434) beats the domain-balanced set (0.9366).
    assert "0.9434" in table and "0.9366" in table


def test_notes_reliability_renders_and_degrades_gracefully() -> None:
    """A4.5: the reliability table renders per-domain alpha, and degrades to a
    placeholder (never raises) when reliability.json is absent from the bundle."""
    notes = _load_paper_module("generate_notes_data.py")

    rel = {
        "full_50": {d: {"alpha": 0.80 + 0.01 * i} for i, d in enumerate(DOMAINS)},
        "domain_balanced_20": {d: {"alpha": 0.70 + 0.01 * i} for i, d in enumerate(DOMAINS)},
        "mini_ipip_20": {d: {"alpha": 0.65 + 0.01 * i} for i, d in enumerate(DOMAINS)},
    }
    table = notes._gen_reliability_from_notes_inputs({"reliability": rel})
    assert "Full 50-item" in table and "Domain-balanced 20" in table and "Mini-IPIP 20" in table
    assert "0.800" in table  # ext full-50 alpha

    # Missing/absent reliability -> placeholder, no exception.
    placeholder = notes._gen_reliability_from_notes_inputs({})
    assert "not available" in placeholder.lower()


def test_notes_data_splits_renders_current_split_schema(
    tmp_path,
    monkeypatch,
) -> None:
    notes = _load_paper_module("generate_notes_data.py")
    monkeypatch.setattr(notes, "PACKAGE_ROOT", tmp_path)
    monkeypatch.setattr(notes, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(
        notes, "RESEARCH_SUMMARY_PATH", tmp_path / "artifacts" / "research_summary.json"
    )

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)

    # gen_data_splits reads reference_notes_inputs.split_metadata from
    # research_summary.json, exercising the canonical_v1 split schema.
    with open(artifacts_dir / "research_summary.json", "w") as f:
        json.dump(
            {
                "variants": {},
                "reference_notes_inputs": {
                    "split_metadata": {
                        "split_id": "canonical_v1",
                        "seed": 42,
                        "total_valid": 1000,
                        "train_rows": 700,
                        "val_rows": 150,
                        "test_rows": 150,
                        "train_frac": 0.7,
                        "val_frac": 0.15,
                        "test_frac": 0.15,
                    }
                },
            },
            f,
            indent=2,
        )

    table = notes.gen_data_splits()
    assert "Split: canonical_v1 — plain random partition (70/15/15, seed=42)." in table
    assert "700" in table
    assert "150" in table


# ── A6.9 generated-document drift guards ─────────────────────────────────────


def test_notes_section_generators_match_template_markers() -> None:
    """Every NOTES section generator must have matching BEGIN/END markers in the
    committed template, and every marker pair must have a generator. A mismatch
    means `make notes` silently SKIPs that section, so stale numbers survive a
    regeneration — exactly the drift this guards against."""
    notes = _load_paper_module("generate_notes_data.py")
    template = notes.NOTES_TEMPLATE_PATH.read_text()

    generator_names = set(notes.SECTION_GENERATORS)
    # Only standalone marker LINES are real section markers (this matches what
    # update_notes substitutes); an inline `<!-- BEGIN:section_name -->` in the
    # header comment documenting the format is not a section.
    begin_markers = set(re.findall(r"(?m)^<!-- BEGIN:([a-z0-9_]+) -->$", template))
    end_markers = set(re.findall(r"(?m)^<!-- END:([a-z0-9_]+) -->$", template))

    # Every generator has a complete marker pair in the template.
    missing = sorted(n for n in generator_names if n not in begin_markers or n not in end_markers)
    assert not missing, f"SECTION_GENERATORS without BEGIN/END markers in the template: {missing}"

    # Every BEGIN marker has a matching END and a registered generator (no orphans).
    assert begin_markers == end_markers, (
        f"Unbalanced markers: BEGIN-only={begin_markers - end_markers}, "
        f"END-only={end_markers - begin_markers}"
    )
    orphan_markers = sorted(begin_markers - generator_names)
    assert not orphan_markers, f"Template markers with no SECTION_GENERATORS entry: {orphan_markers}"


@pytest.mark.skipif(
    os.environ.get("BFFM_STRICT_DRIFT") != "1",
    reason=(
        "Strict generated-document drift is checked in Part D after the re-run "
        "regenerates the committed docs; the committed notes/NOTES.md is "
        "disclosed-stale (pre-canonical_v1) until then. Set BFFM_STRICT_DRIFT=1 to run."
    ),
)
def test_notes_md_has_no_drift_from_generators() -> None:
    """Part-D gate: regenerating NOTES.md from the committed template + artifacts
    must reproduce the committed notes/NOTES.md byte-for-byte (no stale sections)."""
    notes = _load_paper_module("generate_notes_data.py")
    # Regenerate the SAME way the committed NOTES.md was produced: a reference-only
    # bundle (research_summary provenance.reference_only) must be rendered with the
    # cross-variant sections scoped to the reference variant, else the ablation
    # generators KeyError on the absent variants. Matches `make notes REFERENCE_ONLY=1`.
    summary = notes.load_research_summary()
    if summary.get("provenance", {}).get("reference_only"):
        notes._ACTIVE_VARIANT_ORDER = [notes.REFERENCE_VARIANT]
    template = notes.NOTES_TEMPLATE_PATH.read_text()
    regenerated = template
    for name, gen_fn in notes.SECTION_GENERATORS.items():
        pattern = rf"(<!-- BEGIN:{name} -->\n).*?(\n<!-- END:{name} -->)"
        if re.search(pattern, regenerated, flags=re.DOTALL):
            regenerated = re.sub(
                pattern, rf"\g<1>{gen_fn()}\g<2>", regenerated, flags=re.DOTALL
            )
    assert regenerated == notes.NOTES_PATH.read_text(), (
        "notes/NOTES.md drifted from its generators — run `make notes`."
    )


def test_train_main_fails_closed_when_locked_params_lack_provenance(
    tmp_path,
    monkeypatch,
) -> None:
    train = _load_pipeline_module("07_train.py")
    monkeypatch.setattr(train, "PACKAGE_ROOT", tmp_path)

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    with open(artifacts_dir / "tuned_params.json", "w") as f:
        json.dump({"hyperparameters": {"n_estimators": 100}}, f, indent=2)

    cfg_path = tmp_path / "cfg_locked.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "name: unit_locked",
                "output_dir: models/unit_locked",
                "hyperparameters:",
                "  locked_params: artifacts/tuned_params.json",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["07_train.py", "--config", str(cfg_path)])
    rc = train.main()
    assert rc == 1


def test_train_main_fails_closed_when_locked_params_hash_lock_mismatches_data(
    tmp_path,
    monkeypatch,
) -> None:
    train = _load_pipeline_module("07_train.py")
    monkeypatch.setattr(train, "PACKAGE_ROOT", tmp_path)

    data_dir = tmp_path / "data" / "processed"
    artifacts_dir = tmp_path / "artifacts"
    data_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)

    frame = _make_dataset()
    frame.to_parquet(data_dir / "train.parquet", index=False)
    frame.to_parquet(data_dir / "val.parquet", index=False)
    _write_item_info(data_dir / "item_info.json")

    tuned_payload = {
        "hyperparameters": {"n_estimators": 321, "max_depth": 7},
        "provenance": {
            "script": "06_tune.py",
            "git_hash": "feedbeef",
            "data_snapshot_id": "norms_sha256:abc123",
            "preprocessing_version": "feedbeef",
            "train_sha256": "0" * 64,
            "val_sha256": file_sha256(data_dir / "val.parquet"),
            "item_info_sha256": file_sha256(data_dir / "item_info.json"),
        },
    }
    tuned_path = artifacts_dir / "tuned_params.json"
    with open(tuned_path, "w") as f:
        json.dump(tuned_payload, f, indent=2)

    cfg_path = tmp_path / "cfg_locked_mismatch.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "name: unit_locked_mismatch",
                "output_dir: models/unit_locked_mismatch",
                "sparsity:",
                "  enabled: true",
                "  include_mini_ipip: false",
                "hyperparameters:",
                "  locked_params: artifacts/tuned_params.json",
                "training:",
                "  cv_folds: 0",
                "  random_state: 42",
                "validation:",
                "  min_pearson_r: 0.0",
                "  min_coverage_90: 0.0",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["07_train.py", "--config", str(cfg_path)])
    rc = train.main()
    assert rc == 1


def test_train_main_reference_lock_policy_requires_reference_report(
    tmp_path,
    monkeypatch,
) -> None:
    train = _load_pipeline_module("07_train.py")
    monkeypatch.setattr(train, "PACKAGE_ROOT", tmp_path)

    data_dir = tmp_path / "data" / "processed"
    artifacts_dir = tmp_path / "artifacts"
    data_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)

    frame = _make_dataset()
    frame.to_parquet(data_dir / "train.parquet", index=False)
    frame.to_parquet(data_dir / "val.parquet", index=False)

    tuned_payload = {
        "hyperparameters": {"n_estimators": 321, "max_depth": 7},
        "provenance": {"script": "06_tune.py"},
    }
    with open(artifacts_dir / "tuned_params.json", "w") as f:
        json.dump(tuned_payload, f, indent=2)

    cfg_path = tmp_path / "cfg_reference_lock_missing_report.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "name: unit_reference_lock_missing_report",
                "output_dir: models/unit_reference_lock_missing_report",
                "sparsity:",
                "  enabled: false",
                "hyperparameters:",
                "  locked_params: artifacts/tuned_params.json",
                "  lock_policy: reference_model_hash",
                "  reference_model_dir: models/reference",
                "training:",
                "  cv_folds: 0",
                "  random_state: 42",
                "validation:",
                "  min_pearson_r: 0.0",
                "  min_coverage_90: 0.0",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["07_train.py", "--config", str(cfg_path)])
    rc = train.main()
    assert rc == 1


def test_train_main_reference_lock_policy_fails_on_reference_hash_mismatch(
    tmp_path,
    monkeypatch,
) -> None:
    train = _load_pipeline_module("07_train.py")
    monkeypatch.setattr(train, "PACKAGE_ROOT", tmp_path)

    data_dir = tmp_path / "data" / "processed"
    artifacts_dir = tmp_path / "artifacts"
    reference_dir = tmp_path / "models" / "reference"
    data_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)
    reference_dir.mkdir(parents=True)

    frame = _make_dataset()
    frame.to_parquet(data_dir / "train.parquet", index=False)
    frame.to_parquet(data_dir / "val.parquet", index=False)

    tuned_payload = {
        "hyperparameters": {"n_estimators": 321, "max_depth": 7},
        "provenance": {"script": "06_tune.py"},
    }
    tuned_path = artifacts_dir / "tuned_params.json"
    with open(tuned_path, "w") as f:
        json.dump(tuned_payload, f, indent=2)

    _write_training_report(
        reference_dir / "training_report.json",
        data_overrides={
            "hyperparameters_sha256": "0" * 64,
            "hyperparameters_source_sha256": file_sha256(tuned_path),
        },
    )

    cfg_path = tmp_path / "cfg_reference_lock_mismatch.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "name: unit_reference_lock_mismatch",
                "output_dir: models/unit_reference_lock_mismatch",
                "sparsity:",
                "  enabled: false",
                "hyperparameters:",
                "  locked_params: artifacts/tuned_params.json",
                "  lock_policy: reference_model_hash",
                "  reference_model_dir: models/reference",
                "training:",
                "  cv_folds: 0",
                "  random_state: 42",
                "validation:",
                "  min_pearson_r: 0.0",
                "  min_coverage_90: 0.0",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["07_train.py", "--config", str(cfg_path)])
    rc = train.main()
    assert rc == 1


def test_train_main_reference_lock_policy_allows_canonical_data_with_matching_reference(
    tmp_path,
    monkeypatch,
) -> None:
    train = _load_pipeline_module("07_train.py")
    monkeypatch.setattr(train, "PACKAGE_ROOT", tmp_path)

    data_dir = tmp_path / "data" / "processed" / "canonical_v1"
    artifacts_dir = tmp_path / "artifacts"
    reference_dir = tmp_path / "models" / "reference"
    data_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)
    reference_dir.mkdir(parents=True)

    frame = _make_dataset()
    frame.to_parquet(data_dir / "train.parquet", index=False)
    frame.to_parquet(data_dir / "val.parquet", index=False)

    tuned_payload = {
        "hyperparameters": {"n_estimators": 321, "max_depth": 7},
        "provenance": {
            "script": "06_tune.py",
            "train_sha256": "0" * 64,
            "val_sha256": "0" * 64,
            "item_info_sha256": "0" * 64,
        },
    }
    tuned_path = artifacts_dir / "tuned_params.json"
    with open(tuned_path, "w") as f:
        json.dump(tuned_payload, f, indent=2)

    tuned_hp_sha = train._stable_json_sha256(tuned_payload["hyperparameters"])
    _write_training_report(
        reference_dir / "training_report.json",
        data_overrides={
            "hyperparameters_sha256": tuned_hp_sha,
            "hyperparameters_source_sha256": file_sha256(tuned_path),
        },
    )

    cfg_path = tmp_path / "cfg_reference_lock_success.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "name: unit_reference_lock_success",
                "output_dir: models/unit_reference_lock_success",
                "data_dir: data/processed/canonical_v1",
                "artifacts_dir: artifacts",
                "sparsity:",
                "  enabled: false",
                "hyperparameters:",
                "  locked_params: artifacts/tuned_params.json",
                "  lock_policy: reference_model_hash",
                "  reference_model_dir: models/reference",
                "training:",
                "  cv_folds: 0",
                "  random_state: 42",
                "validation:",
                "  min_pearson_r: 0.0",
                "  min_coverage_90: 0.0",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(train, "_train_domain_models", lambda *_args, **_kwargs: _dummy_domain_models())
    monkeypatch.setattr(train, "_validate_model_outputs", lambda *_args, **_kwargs: {"ok": {"passed": True}})
    monkeypatch.setattr(
        train,
        "_evaluate_domain_models",
        lambda *_args, **_kwargs: _make_eval_metrics(r=0.95, coverage=0.90),
    )
    monkeypatch.setattr(
        train,
        "_compute_calibration_params",
        lambda *_args, **_kwargs: {
            domain: {"observed_coverage": 0.9, "scale_factor": 1.0}
            for domain in DOMAINS
        },
    )
    monkeypatch.setattr(train.joblib, "dump", lambda *_args, **_kwargs: None)

    monkeypatch.setattr(sys, "argv", ["07_train.py", "--config", str(cfg_path)])
    rc = train.main()
    assert rc == 0

    with open(tmp_path / "models" / "unit_reference_lock_success" / "training_report.json") as f:
        report = json.load(f)
    assert report["data"]["hyperparameters_lock_policy"] == "reference_model_hash"
    assert report["data"]["hyperparameters_reference_model_dir"] == str(reference_dir)


def test_train_report_records_locked_params_provenance_chain(
    tmp_path,
    monkeypatch,
) -> None:
    train = _load_pipeline_module("07_train.py")
    monkeypatch.setattr(train, "PACKAGE_ROOT", tmp_path)

    data_dir = tmp_path / "data" / "processed"
    artifacts_dir = tmp_path / "artifacts"
    data_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)

    frame = _make_dataset()
    frame.to_parquet(data_dir / "train.parquet", index=False)
    frame.to_parquet(data_dir / "val.parquet", index=False)
    _write_item_info(data_dir / "item_info.json")

    tuned_payload = {
        "hyperparameters": {"n_estimators": 321, "max_depth": 7},
        "provenance": {
            "script": "06_tune.py",
            "git_hash": "feedbeef",
            "data_snapshot_id": "norms_sha256:abc123",
            "preprocessing_version": "feedbeef",
            "train_sha256": file_sha256(data_dir / "train.parquet"),
            "val_sha256": file_sha256(data_dir / "val.parquet"),
            "item_info_sha256": file_sha256(data_dir / "item_info.json"),
        },
    }
    tuned_path = artifacts_dir / "tuned_params.json"
    with open(tuned_path, "w") as f:
        json.dump(tuned_payload, f, indent=2)

    cfg_path = tmp_path / "cfg_locked_chain.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "name: unit_locked_chain",
                "output_dir: models/unit_locked_chain",
                "sparsity:",
                "  enabled: true",
                "  include_mini_ipip: false",
                "hyperparameters:",
                "  locked_params: artifacts/tuned_params.json",
                "training:",
                "  cv_folds: 0",
                "  random_state: 42",
                "validation:",
                "  min_pearson_r: 0.0",
                "  min_coverage_90: 0.0",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(train, "_train_domain_models", lambda *_args, **_kwargs: _dummy_domain_models())
    monkeypatch.setattr(train, "_validate_model_outputs", lambda *_args, **_kwargs: {"ok": {"passed": True}})
    monkeypatch.setattr(train, "_evaluate_domain_models", lambda *_args, **_kwargs: _make_eval_metrics(r=0.92, coverage=0.9))
    monkeypatch.setattr(
        train,
        "_compute_calibration_params",
        lambda *_args, **_kwargs: {
            domain: {"observed_coverage": 0.9, "scale_factor": 1.0}
            for domain in DOMAINS
        },
    )
    monkeypatch.setattr(train.joblib, "dump", lambda *_args, **_kwargs: None)

    monkeypatch.setattr(sys, "argv", ["07_train.py", "--config", str(cfg_path)])
    rc = train.main()
    assert rc == 0

    report_path = tmp_path / "models" / "unit_locked_chain" / "training_report.json"
    with open(report_path) as f:
        report = json.load(f)

    source = report["config"]["hyperparameters_source"]
    assert source["mode"] == "config_locked_params"
    assert source["path"] == str(tuned_path)
    assert source["file_sha256"] == file_sha256(tuned_path)
    assert source["payload_provenance"]["script"] == "06_tune.py"
    assert source["hyperparameters_sha256"] == train._stable_json_sha256(
        tuned_payload["hyperparameters"]
    )
    assert report["data"]["hyperparameters_source_sha256"] == file_sha256(tuned_path)


def test_train_fails_closed_when_locked_params_edited_after_tune(tmp_path, monkeypatch) -> None:
    """A5.1: editing tuned_params.json without re-running `make tune` (so the
    .original.json sidecar disagrees) fails the strict_data_hash lock."""
    train = _load_pipeline_module("07_train.py")
    monkeypatch.setattr(train, "PACKAGE_ROOT", tmp_path)

    data_dir = tmp_path / "data" / "processed"
    artifacts_dir = tmp_path / "artifacts"
    data_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)

    frame = _make_dataset()
    frame.to_parquet(data_dir / "train.parquet", index=False)
    frame.to_parquet(data_dir / "val.parquet", index=False)
    _write_item_info(data_dir / "item_info.json")

    prov = {
        "script": "06_tune.py",
        "git_hash": "feedbeef",
        "train_sha256": file_sha256(data_dir / "train.parquet"),
        "val_sha256": file_sha256(data_dir / "val.parquet"),
        "item_info_sha256": file_sha256(data_dir / "item_info.json"),
    }
    # Loaded params (edited) vs the tune-time witness (original) disagree.
    tuned_payload = {"hyperparameters": {"n_estimators": 321, "max_depth": 7}, "provenance": prov}
    with open(artifacts_dir / "tuned_params.json", "w") as f:
        json.dump(tuned_payload, f, indent=2)
    original_payload = {"hyperparameters": {"n_estimators": 999, "max_depth": 3}, "provenance": prov}
    with open(artifacts_dir / "tuned_params.original.json", "w") as f:
        json.dump(original_payload, f, indent=2)

    cfg_path = tmp_path / "cfg_edited.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "name: unit_edited",
                "output_dir: models/unit_edited",
                "sparsity:",
                "  enabled: false",
                "hyperparameters:",
                "  locked_params: artifacts/tuned_params.json",
                "  lock_policy: strict_data_hash",
                "training:",
                "  cv_folds: 0",
                "  random_state: 42",
                "validation:",
                "  min_pearson_r: 0.0",
                "  min_coverage_90: 0.0",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["07_train.py", "--config", str(cfg_path)])
    assert train.main() == 1


def test_train_fails_closed_when_test_parquet_missing_in_publication_mode(tmp_path, monkeypatch) -> None:
    """A5.6: require_test_split makes a missing test.parquet a hard error."""
    train = _load_pipeline_module("07_train.py")
    monkeypatch.setattr(train, "PACKAGE_ROOT", tmp_path)
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    frame = _make_dataset(n_rows=24)
    frame.iloc[:14].to_parquet(data_dir / "train.parquet", index=False)
    frame.iloc[14:].to_parquet(data_dir / "val.parquet", index=False)
    _write_item_info(data_dir / "item_info.json")

    cfg_path = tmp_path / "cfg_pub_no_test.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "name: unit_pub_no_test",
                "output_dir: models/unit_pub_no_test",
                "require_test_split: true",
                "sparsity:",
                "  enabled: false",
                "training:",
                "  cv_folds: 0",
                "  random_state: 42",
                "validation:",
                "  min_pearson_r: 0.0",
                "  min_coverage_90: 0.0",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["07_train.py", "--config", str(cfg_path)])
    assert train.main() == 1


def test_train_fails_closed_when_split_metadata_missing_in_publication_mode(tmp_path, monkeypatch) -> None:
    """A5.6: require_test_split makes a missing split_metadata.json a hard error."""
    train = _load_pipeline_module("07_train.py")
    monkeypatch.setattr(train, "PACKAGE_ROOT", tmp_path)
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    frame = _make_dataset(n_rows=24)
    frame.iloc[:14].to_parquet(data_dir / "train.parquet", index=False)
    frame.iloc[14:19].to_parquet(data_dir / "val.parquet", index=False)
    frame.iloc[19:].to_parquet(data_dir / "test.parquet", index=False)
    _write_item_info(data_dir / "item_info.json")

    cfg_path = tmp_path / "cfg_pub_no_meta.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "name: unit_pub_no_meta",
                "output_dir: models/unit_pub_no_meta",
                "require_test_split: true",
                "sparsity:",
                "  enabled: false",
                "training:",
                "  cv_folds: 0",
                "  random_state: 42",
                "validation:",
                "  min_pearson_r: 0.0",
                "  min_coverage_90: 0.0",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["07_train.py", "--config", str(cfg_path)])
    assert train.main() == 1


def test_train_report_records_test_rows_from_split_metadata(tmp_path, monkeypatch) -> None:
    """A5.7: stage 07 writes data.test_rows from split_metadata.json."""
    train = _load_pipeline_module("07_train.py")
    monkeypatch.setattr(train, "PACKAGE_ROOT", tmp_path)
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    frame = _make_dataset(n_rows=24)
    frame.iloc[:14].to_parquet(data_dir / "train.parquet", index=False)
    frame.iloc[14:19].to_parquet(data_dir / "val.parquet", index=False)
    frame.iloc[19:].to_parquet(data_dir / "test.parquet", index=False)
    _write_item_info(data_dir / "item_info.json")
    with open(data_dir / "split_metadata.json", "w") as f:
        json.dump(
            {
                "train_sha256": file_sha256(data_dir / "train.parquet"),
                "val_sha256": file_sha256(data_dir / "val.parquet"),
                "test_sha256": file_sha256(data_dir / "test.parquet"),
                "test_rows": 5,
            },
            f,
        )

    cfg_path = tmp_path / "cfg_test_rows.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "name: unit_test_rows",
                "output_dir: models/unit_test_rows",
                "sparsity:",
                "  enabled: false",
                "training:",
                "  cv_folds: 0",
                "  random_state: 42",
                "validation:",
                "  min_pearson_r: 0.0",
                "  min_coverage_90: 0.0",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(train, "_train_domain_models", lambda *_a, **_k: _dummy_domain_models())
    monkeypatch.setattr(train, "_validate_model_outputs", lambda *_a, **_k: {"ok": {"passed": True}})
    monkeypatch.setattr(train, "_evaluate_domain_models", lambda *_a, **_k: _make_eval_metrics(r=0.92, coverage=0.9))
    monkeypatch.setattr(
        train,
        "_compute_calibration_params",
        lambda *_a, **_k: {d: {"observed_coverage": 0.9, "scale_factor": 1.0} for d in DOMAINS},
    )
    monkeypatch.setattr(train.joblib, "dump", lambda *_a, **_k: None)
    monkeypatch.setattr(sys, "argv", ["07_train.py", "--config", str(cfg_path)])
    assert train.main() == 0

    report_path = tmp_path / "models" / "unit_test_rows" / "training_report.json"
    with open(report_path) as f:
        report = json.load(f)
    assert report["data"]["test_rows"] == 5
    # A5.3: tree_method/device recorded in provenance.
    assert report["provenance"]["tree_method"] == "hist"
    assert report["provenance"]["device"] == "cpu"


# ---------------------------------------------------------------------------
# Provenance hardening tests
# ---------------------------------------------------------------------------


def test_export_generates_provenance_json(tmp_path) -> None:
    """Verify generate_provenance_document() returns dict with required top-level keys."""
    export = _load_pipeline_module("11_export_onnx.py")

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Write minimal training report
    _write_training_report(
        models_dir / "training_report.json",
        data_overrides={
            "hyperparameters_sha256": "abc123",
            "item_info_sha256": "def456",
            "split_metadata_sha256": "ghi789",
        },
    )

    # Write dummy output files for checksum computation
    (output_dir / "model.onnx").write_bytes(b"fake-onnx-model")
    (output_dir / "config.json").write_text('{"test": true}', encoding="utf-8")

    provenance_dict = {
        "git_hash": "testcommit123",
        "data_snapshot_id": "norms_sha256:abc",
        "preprocessing_version": "testcommit123",
    }

    doc = export.generate_provenance_document(
        models_dir=models_dir,
        output_dir=output_dir,
        provenance_dict=provenance_dict,
    )

    assert isinstance(doc, dict)
    assert doc["schema_version"] == 1
    assert isinstance(doc.get("export"), dict)
    assert isinstance(doc.get("training"), dict)
    assert isinstance(doc.get("artifacts"), dict)
    assert doc["export"]["script"] == "11_export_onnx.py"
    assert doc["export"]["git_hash"] == "testcommit123"

    # Verify actual SHA-256 values in artifacts block match written files
    expected_model_sha = hashlib.sha256(b"fake-onnx-model").hexdigest()
    expected_config_sha = hashlib.sha256(b'{"test": true}').hexdigest()
    assert doc["artifacts"]["model_onnx_sha256"] == expected_model_sha
    assert doc["artifacts"]["config_json_sha256"] == expected_config_sha


def test_export_provenance_json_preserves_training_sha256s(tmp_path) -> None:
    """Verify training.data contains hyperparameters_sha256, item_info_sha256, split_metadata_sha256."""
    export = _load_pipeline_module("11_export_onnx.py")

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    _write_training_report(
        models_dir / "training_report.json",
        data_overrides={
            "hyperparameters_sha256": "hp_sha_test",
            "item_info_sha256": "ii_sha_test",
            "split_metadata_sha256": "sm_sha_test",
        },
    )
    (output_dir / "model.onnx").write_bytes(b"fake")
    (output_dir / "config.json").write_text("{}", encoding="utf-8")

    doc = export.generate_provenance_document(
        models_dir=models_dir,
        output_dir=output_dir,
        provenance_dict={"git_hash": "abc", "data_snapshot_id": "test", "preprocessing_version": "abc"},
    )

    training_data = doc["training"]["data"]
    assert training_data["hyperparameters_sha256"] == "hp_sha_test"
    assert training_data["item_info_sha256"] == "ii_sha_test"
    assert training_data["split_metadata_sha256"] == "sm_sha_test"


def test_upload_bundle_requires_provenance_json(tmp_path) -> None:
    """Verify _validate_output_bundle() fails when provenance.json is missing."""
    upload = _load_pipeline_module("13_upload_hf.py")
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True)

    outputs = [
        f"{domain}_{q}"
        for domain in DOMAINS
        for q in ("q05", "q50", "q95")
    ]
    with open(output_dir / "config.json", "w") as f:
        json.dump(
            {
                "model_file": "model.onnx",
                "outputs": outputs,
                "scores_output": "scores",
                "provenance": {
                    "git_hash": "deadbeef",
                    "data_snapshot_id": "snapshot-a",
                    "training_script": "07_train.py",
                },
            },
            f,
        )
    (output_dir / "README.md").write_text("ok", encoding="utf-8")
    (output_dir / "model.onnx").write_bytes(b"onnx")

    # No provenance.json — should fail
    with pytest.raises(FileNotFoundError) as exc_info:
        upload._validate_output_bundle(output_dir)
    assert "provenance.json" in str(exc_info.value)


def test_figures_writes_manifest_json(tmp_path) -> None:
    """Verify manifest.json schema by building a synthetic manifest matching the pipeline shape."""
    from lib.provenance import build_provenance, relative_to_root, file_sha256

    fig_dir = tmp_path / "figures"
    fig_dir.mkdir(parents=True)

    # Write synthetic source artifacts
    source_a = tmp_path / "source_a.json"
    source_a.write_text('{"data": 1}', encoding="utf-8")
    source_b = tmp_path / "source_b.csv"
    source_b.write_text("col1,col2\n1,2\n", encoding="utf-8")

    source_artifacts = {}
    for label, path in [("source_a", source_a), ("source_b", source_b)]:
        source_artifacts[label] = {
            "path": relative_to_root(path),
            "sha256": file_sha256(path),
        }

    manifest = {
        "schema_version": 1,
        "provenance": build_provenance("12_generate_figures.py"),
        "model_dir": "models/reference",
        "source_artifacts": source_artifacts,
        "figures": [
            {
                "filename": "fig1_test",
                "formats": ["png", "pdf"],
                "source_artifacts": ["source_a"],
            },
        ],
    }

    manifest_path = fig_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Re-load and validate
    with open(manifest_path) as f:
        loaded = json.load(f)

    assert loaded["schema_version"] == 1
    assert isinstance(loaded["provenance"], dict)
    assert loaded["provenance"]["script"] == "12_generate_figures.py"
    assert "git_hash" in loaded["provenance"]
    assert isinstance(loaded["source_artifacts"], dict)
    for label, info in loaded["source_artifacts"].items():
        assert "path" in info
        assert "sha256" in info
        assert len(info["sha256"]) == 64
    assert isinstance(loaded["figures"], list)
    assert len(loaded["figures"]) == 1
    assert loaded["figures"][0]["filename"] == "fig1_test"

    # Verify figures module has the required imports
    figures = _load_pipeline_module("12_generate_figures.py")
    assert hasattr(figures, "build_provenance")
    assert hasattr(figures, "relative_to_root")
    assert hasattr(figures, "file_sha256")


def test_export_provenance_document_no_training_report(tmp_path) -> None:
    """Verify generate_provenance_document() handles missing training report gracefully."""
    export = _load_pipeline_module("11_export_onnx.py")

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # No training_report.json or adaptive_training_report.json written
    (output_dir / "model.onnx").write_bytes(b"fake-model")
    (output_dir / "config.json").write_text('{"ok": true}', encoding="utf-8")

    doc = export.generate_provenance_document(
        models_dir=models_dir,
        output_dir=output_dir,
        provenance_dict={
            "git_hash": "abc123",
            "data_snapshot_id": "test-snap",
            "preprocessing_version": "abc123",
        },
    )

    assert isinstance(doc, dict)
    assert doc["training"] == {}
    assert isinstance(doc["artifacts"], dict)
    assert doc["export"]["git_hash"] == "abc123"


def _make_valid_upload_bundle(output_dir: Path, *, git_hash: str = "deadbeef") -> tuple[str, str]:
    """Helper: write a valid output bundle with consistent provenance + checksums."""
    outputs = [
        f"{domain}_{q}"
        for domain in DOMAINS
        for q in ("q05", "q50", "q95")
    ]
    config_payload = {
        "model_file": "model.onnx",
        "outputs": outputs,
        "scores_output": "scores",
        "provenance": {
            "git_hash": git_hash,
            "data_snapshot_id": "snapshot-a",
            "preprocessing_version": git_hash,
            "model_dir": "models/reference",
            "training_script": "07_train.py",
        },
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_payload, f, indent=2)
    (output_dir / "README.md").write_text("ok", encoding="utf-8")
    (output_dir / "model.onnx").write_bytes(b"onnx")

    from lib.provenance import file_sha256 as _sha256
    config_sha = _sha256(output_dir / "config.json")
    model_sha = _sha256(output_dir / "model.onnx")

    return config_sha, model_sha


def test_upload_bundle_rejects_malformed_provenance_json(tmp_path) -> None:
    """Verify _validate_output_bundle rejects malformed provenance.json."""
    upload = _load_pipeline_module("13_upload_hf.py")
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True)
    _make_valid_upload_bundle(output_dir)

    # Case 1: provenance.json is not a dict
    with open(output_dir / "provenance.json", "w") as f:
        json.dump([1, 2, 3], f)
    with pytest.raises(ValueError, match="JSON object"):
        upload._validate_output_bundle(output_dir)

    # Case 2: missing export block
    with open(output_dir / "provenance.json", "w") as f:
        json.dump({"training": {}}, f)
    with pytest.raises(ValueError, match="export"):
        upload._validate_output_bundle(output_dir)

    # Case 3: missing training block
    with open(output_dir / "provenance.json", "w") as f:
        json.dump({"export": {"script": "11_export_onnx.py"}}, f)
    with pytest.raises(ValueError, match="training"):
        upload._validate_output_bundle(output_dir)


def test_upload_bundle_rejects_mismatched_provenance_config(tmp_path) -> None:
    """Verify cross-check catches git_hash mismatch between provenance.json and config.json."""
    upload = _load_pipeline_module("13_upload_hf.py")
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True)
    config_sha, model_sha = _make_valid_upload_bundle(output_dir, git_hash="deadbeef")

    # provenance.json with different git_hash than config.json
    with open(output_dir / "provenance.json", "w") as f:
        json.dump(
            {
                "schema_version": 1,
                "export": {
                    "script": "11_export_onnx.py",
                    "git_hash": "MISMATCH_HASH",
                    "data_snapshot_id": "snapshot-a",
                    "preprocessing_version": "deadbeef",
                    "model_dir": "models/reference",
                },
                "training": {"provenance": {"script": "07_train.py"}},
                "artifacts": {
                    "config_json_sha256": config_sha,
                    "model_onnx_sha256": model_sha,
                },
            },
            f,
        )

    with pytest.raises(ValueError, match="git_hash"):
        upload._validate_output_bundle(output_dir)


def _make_complete_upload_bundle(
    output_dir: Path,
    *,
    git_hash: str = "deadbeef",
    model_dir: str = "models/reference",
) -> None:
    """Write a complete valid output bundle including provenance.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config_sha, model_sha = _make_valid_upload_bundle(output_dir, git_hash=git_hash)
    # Patch model_dir into config.json
    with open(output_dir / "config.json") as f:
        config = json.load(f)
    config["provenance"]["model_dir"] = model_dir
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    from lib.provenance import file_sha256 as _sha256
    config_sha = _sha256(output_dir / "config.json")

    with open(output_dir / "provenance.json", "w") as f:
        json.dump(
            {
                "schema_version": 1,
                "export": {
                    "script": "11_export_onnx.py",
                    "git_hash": git_hash,
                    "data_snapshot_id": "snapshot-a",
                    "preprocessing_version": git_hash,
                    "model_dir": model_dir,
                },
                "training": {"provenance": {"script": "07_train.py"}},
                "artifacts": {
                    "config_json_sha256": config_sha,
                    "model_onnx_sha256": model_sha,
                },
            },
            f,
        )


def test_upload_discovers_variant_subdirectories(tmp_path) -> None:
    """Verify _discover_variants finds subdirectories with config.json."""
    upload = _load_pipeline_module("13_upload_hf.py")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create two variant subdirectories
    for name in ("reference", "ablation_none"):
        variant_dir = output_dir / name
        _make_complete_upload_bundle(variant_dir, model_dir=f"models/{name}")

    variants = upload._discover_variants(output_dir)
    names = [name for name, _ in variants]
    assert "reference" in names
    assert "ablation_none" in names
    assert len(variants) == 2
    # Each variant path should contain config.json
    for _, vpath in variants:
        assert (vpath / "config.json").exists()


def test_upload_backward_compat_flat_bundle(tmp_path) -> None:
    """Verify _discover_variants returns flat layout when config.json is at root."""
    upload = _load_pipeline_module("13_upload_hf.py")
    output_dir = tmp_path / "output"
    _make_complete_upload_bundle(output_dir)

    variants = upload._discover_variants(output_dir)
    assert len(variants) == 1
    name, vpath = variants[0]
    assert name == ""
    assert vpath == output_dir


def test_upload_generates_repo_readme(tmp_path) -> None:
    """Verify generate_repo_readme includes all variant names."""
    export = _load_pipeline_module("11_export_onnx.py")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    variant_names = ["reference", "ablation_none", "ablation_focused"]
    variants = []
    for name in variant_names:
        variant_dir = output_dir / name
        _make_complete_upload_bundle(variant_dir, model_dir=f"models/{name}")
        variants.append((name, variant_dir))

    readme = export.generate_repo_readme(variants)
    assert "reference" in readme
    assert "ablation_none" in readme
    assert "ablation_focused" in readme
    assert "Primary published model" in readme
    assert "pipeline_tag: tabular-regression" in readme


def test_export_variant_name_derived_from_model_dir(tmp_path) -> None:
    """Verify generate_config includes variant when variant_name is provided."""
    export = _load_pipeline_module("11_export_onnx.py")
    models_dir = tmp_path / "models" / "ablation_none"
    models_dir.mkdir(parents=True)
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)

    norms_map = {d: {"mean": 3.0, "sd": 0.8} for d in DOMAINS}
    provenance_dict = {
        "git_hash": "abc123",
        "data_snapshot_id": "snap",
        "preprocessing_version": "abc123",
    }

    config = export.generate_config(
        models_dir, artifacts_dir, provenance_dict, norms_map,
        variant_name="ablation_none",
    )
    assert config["variant"] == "ablation_none"

    # Without variant_name, no variant key
    config_no_variant = export.generate_config(
        models_dir, artifacts_dir, provenance_dict, norms_map,
    )
    assert "variant" not in config_no_variant


def test_export_provenance_document_includes_variant(tmp_path) -> None:
    """Verify generate_provenance_document includes variant in export block."""
    export = _load_pipeline_module("11_export_onnx.py")
    models_dir = tmp_path / "models" / "reference"
    models_dir.mkdir(parents=True)
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True)
    # Write minimal model.onnx and config.json for checksum computation
    (output_dir / "model.onnx").write_bytes(b"onnx")
    (output_dir / "config.json").write_text("{}", encoding="utf-8")

    provenance_dict = {
        "git_hash": "abc123",
        "data_snapshot_id": "snap",
        "preprocessing_version": "abc123",
    }

    prov_doc = export.generate_provenance_document(
        models_dir=models_dir,
        output_dir=output_dir,
        provenance_dict=provenance_dict,
        variant_name="reference",
    )
    assert prov_doc["export"]["variant"] == "reference"

    # Without variant_name, no variant key in export
    prov_doc_no = export.generate_provenance_document(
        models_dir=models_dir,
        output_dir=output_dir,
        provenance_dict=provenance_dict,
    )
    assert "variant" not in prov_doc_no["export"]


def test_export_readme_includes_variant_tag(tmp_path) -> None:
    """Verify generate_readme adds variant tag and note for ablation variants."""
    export = _load_pipeline_module("11_export_onnx.py")
    models_dir = tmp_path / "models" / "ablation_focused"
    models_dir.mkdir(parents=True)
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)

    # Build a minimal config dict with required provenance fields
    config = {
        "provenance": {
            "git_hash": "abc123",
            "data_snapshot_id": "snap",
            "preprocessing_version": "abc123",
            "model_dir": str(models_dir),
            "n_train_original": 1000,
            "n_train_augmented": 5000,
        },
        "hyperparameters": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
        "norms": {d: {"mean": 3.0, "sd": 0.8} for d in DOMAINS},
    }

    # Create required artifact files with matching provenance
    for fname in ("validation_results.json", "baseline_comparison_results.json", "ml_vs_averaging_comparison.json"):
        artifact = {
            "provenance": {
                "git_hash": "abc123",
                "data_snapshot_id": "snap",
                "preprocessing_version": "abc123",
                "model_dir": str(models_dir),
            },
        }
        if fname == "validation_results.json":
            artifact["metrics"] = {
                "overall": {
                    "pearson_r": 0.95, "mae": 0.1, "coverage_90": 0.91, "n": 500,
                },
            }
            artifact["sparse_20"] = {
                "metrics": {
                    "overall": {
                        "pearson_r": 0.90, "mae": 0.15, "coverage_90": 0.88, "n": 500,
                    },
                },
            }
        elif fname == "baseline_comparison_results.json":
            artifact["overall"] = {
                "20": {
                    "full_50": {"pearson_r": 0.95},
                    "domain_balanced": {"pearson_r": 0.90, "coverage_90": 0.895},
                    "mini_ipip": {"pearson_r": 0.85},
                    "adaptive_topk": {"pearson_r": 0.92},
                },
                "50": {
                    "full_50": {"pearson_r": 0.96},
                },
            }
        elif fname == "ml_vs_averaging_comparison.json":
            artifact["comparisons"] = [
                {"method": "domain_balanced", "n_items": 20, "delta_r": 0.05},
            ]
        with open(artifacts_dir / fname, "w") as f:
            json.dump(artifact, f)

    readme = export.generate_readme(config, artifacts_dir, models_dir, variant_name="ablation_focused")
    assert "variant:ablation_focused" in readme
    assert "research ablation variant" in readme
    assert "primary model is `reference`" in readme


def test_discover_variants_nonexistent_dir(tmp_path) -> None:
    """Verify _discover_variants returns empty list for non-existent directory."""
    upload = _load_pipeline_module("13_upload_hf.py")
    nonexistent = tmp_path / "does_not_exist"
    result = upload._discover_variants(nonexistent)
    assert result == []


def test_discover_variants_hybrid_layout_prefers_subdirs(tmp_path) -> None:
    """Verify _discover_variants prefers subdirectories over stale root config.json."""
    upload = _load_pipeline_module("13_upload_hf.py")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create stale root config.json (leftover from flat layout)
    (output_dir / "config.json").write_text('{"stale": true}', encoding="utf-8")

    # Create variant subdirectories
    for name in ("reference", "ablation_none"):
        variant_dir = output_dir / name
        variant_dir.mkdir()
        (variant_dir / "config.json").write_text('{"ok": true}', encoding="utf-8")

    variants = upload._discover_variants(output_dir)
    names = [n for n, _ in variants]
    assert "reference" in names
    assert "ablation_none" in names
    assert len(variants) == 2
    # Should NOT return flat layout
    assert ("", output_dir) not in variants


def test_export_readme_reference_variant_note(tmp_path) -> None:
    """Verify generate_readme includes 'primary published model' note for reference variant."""
    export = _load_pipeline_module("11_export_onnx.py")
    models_dir = tmp_path / "models" / "reference"
    models_dir.mkdir(parents=True)
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)

    config = {
        "provenance": {
            "git_hash": "abc123",
            "data_snapshot_id": "snap",
            "preprocessing_version": "abc123",
            "model_dir": str(models_dir),
            "n_train_original": 1000,
            "n_train_augmented": 5000,
        },
        "hyperparameters": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
        "norms": {d: {"mean": 3.0, "sd": 0.8} for d in DOMAINS},
    }

    for fname in ("validation_results.json", "baseline_comparison_results.json", "ml_vs_averaging_comparison.json"):
        artifact = {
            "provenance": {
                "git_hash": "abc123",
                "data_snapshot_id": "snap",
                "preprocessing_version": "abc123",
                "model_dir": str(models_dir),
            },
        }
        if fname == "validation_results.json":
            artifact["metrics"] = {
                "overall": {
                    "pearson_r": 0.95, "mae": 0.1, "coverage_90": 0.91, "n": 500,
                },
            }
            artifact["sparse_20"] = {
                "metrics": {
                    "overall": {
                        "pearson_r": 0.90, "mae": 0.15, "coverage_90": 0.88, "n": 500,
                    },
                },
            }
        elif fname == "baseline_comparison_results.json":
            artifact["overall"] = {
                "20": {
                    "full_50": {"pearson_r": 0.95},
                    "domain_balanced": {"pearson_r": 0.90, "coverage_90": 0.895},
                    "mini_ipip": {"pearson_r": 0.85},
                    "adaptive_topk": {"pearson_r": 0.92},
                },
                "50": {
                    "full_50": {"pearson_r": 0.96},
                },
            }
        elif fname == "ml_vs_averaging_comparison.json":
            artifact["comparisons"] = [
                {"method": "domain_balanced", "n_items": 20, "delta_r": 0.05},
            ]
        with open(artifacts_dir / fname, "w") as f:
            json.dump(artifact, f)

    readme = export.generate_readme(config, artifacts_dir, models_dir, variant_name="reference")
    assert "variant:reference" in readme
    assert "primary published model" in readme


def test_repo_readme_no_leading_whitespace(tmp_path) -> None:
    """Verify generate_repo_readme output starts with '---' (no leading whitespace)."""
    export = _load_pipeline_module("11_export_onnx.py")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    variants = []
    for name in ("reference", "ablation_none", "ablation_focused", "ablation_random"):
        variant_dir = output_dir / name
        variant_dir.mkdir()
        variants.append((name, variant_dir))

    readme = export.generate_repo_readme(variants)
    # YAML front-matter must start at column 0
    assert readme.startswith("---\n"), f"README starts with: {readme[:40]!r}"
    # No line should have leading whitespace except tag indentation
    for line in readme.split("\n"):
        if line.startswith("  - "):
            continue  # YAML tag list items
        assert not line.startswith(" "), f"Unexpected leading whitespace: {line!r}"


def test_upload_main_multi_variant_flow(tmp_path, monkeypatch) -> None:
    """E2e: verify multi-variant upload through main() discovers, validates, prefixes, and cleans up."""
    upload = _load_pipeline_module("13_upload_hf.py")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create two valid variant bundles
    for name in ("reference", "ablation_none"):
        _make_complete_upload_bundle(
            output_dir / name,
            git_hash="deadbeef",
            model_dir=f"models/{name}",
        )

    # Mock HfApi to capture upload calls
    uploaded = []

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, **kwargs):
            pass

        def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, revision=None):
            uploaded.append(path_in_repo)

    # Monkeypatch to inject FakeHfApi and skip .env / token checks
    monkeypatch.setattr("sys.argv", [
        "13_upload_hf.py",
        "--repo-id", "test/repo",
        "--output-dir", str(output_dir),
    ])
    monkeypatch.setenv("HF_TOKEN", "fake-token")

    # Inject FakeHfApi via the huggingface_hub import inside main()
    import types
    fake_hf_module = types.ModuleType("huggingface_hub")
    fake_hf_module.HfApi = FakeHfApi
    fake_hf_module.CommitOperationAdd = None
    fake_hf_module.CommitOperationDelete = None
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_module)

    # Create repo-level README (normally generated by `make export-repo-readme`)
    (output_dir / "README.md").write_text("# Repo README\n", encoding="utf-8")

    upload.main()

    # Verify variant-prefixed paths
    assert "reference/config.json" in uploaded
    assert "reference/model.onnx" in uploaded
    assert "reference/README.md" in uploaded
    assert "reference/provenance.json" in uploaded
    assert "ablation_none/config.json" in uploaded
    assert "ablation_none/model.onnx" in uploaded
    # Verify top-level repo README was uploaded
    assert "README.md" in uploaded


def test_upload_main_single_variant_flow(tmp_path, monkeypatch) -> None:
    """E2e: verify --variant uploads files to repo root and deletes stale files."""
    upload = _load_pipeline_module("13_upload_hf.py")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create a valid variant bundle
    _make_complete_upload_bundle(
        output_dir / "reference",
        git_hash="deadbeef",
        model_dir="models/reference",
    )

    # Track operations passed to create_commit
    committed: dict[str, Any] = {}

    class FakeCommitOperationAdd:
        def __init__(self, *, path_in_repo, path_or_fileobj):
            self.path_in_repo = path_in_repo
            self.path_or_fileobj = path_or_fileobj

    class FakeCommitOperationDelete:
        def __init__(self, *, path_in_repo):
            self.path_in_repo = path_in_repo

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, **kwargs):
            pass

        def list_repo_files(self, *, repo_id, revision=None):
            # Simulate stale files already in the repo
            return [
                "config.json",
                "model.onnx",
                "README.md",
                "provenance.json",
                "old_stale_file.txt",
                "ablation_none/config.json",
                ".gitattributes",
            ]

        def create_commit(self, *, repo_id, operations, commit_message, revision=None):
            committed["repo_id"] = repo_id
            committed["operations"] = operations
            committed["commit_message"] = commit_message

    # Monkeypatch to inject fakes
    monkeypatch.setattr("sys.argv", [
        "13_upload_hf.py",
        "--repo-id", "test/repo",
        "--output-dir", str(output_dir),
        "--variant", "reference",
    ])
    monkeypatch.setenv("HF_TOKEN", "fake-token")

    import types
    fake_hf_module = types.ModuleType("huggingface_hub")
    fake_hf_module.HfApi = FakeHfApi
    fake_hf_module.CommitOperationAdd = FakeCommitOperationAdd
    fake_hf_module.CommitOperationDelete = FakeCommitOperationDelete
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_module)

    upload.main()

    # Verify create_commit was called
    assert "operations" in committed
    ops = committed["operations"]

    add_paths = {
        op.path_in_repo for op in ops if isinstance(op, FakeCommitOperationAdd)
    }
    delete_paths = {
        op.path_in_repo for op in ops if isinstance(op, FakeCommitOperationDelete)
    }

    # All 4 bundle files should be uploaded at repo root (no variant prefix)
    assert "config.json" in add_paths
    assert "model.onnx" in add_paths
    assert "README.md" in add_paths
    assert "provenance.json" in add_paths

    # Stale files should be deleted
    assert "old_stale_file.txt" in delete_paths
    assert "ablation_none/config.json" in delete_paths

    # .gitattributes should be preserved (not deleted)
    assert ".gitattributes" not in delete_paths


def test_upload_main_single_variant_missing_dir_exits(tmp_path, monkeypatch) -> None:
    """Verify --variant with a nonexistent variant directory calls sys.exit."""
    upload = _load_pipeline_module("13_upload_hf.py")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Do NOT create the variant directory — it should be missing

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, **kwargs):
            pass

    monkeypatch.setattr("sys.argv", [
        "13_upload_hf.py",
        "--repo-id", "test/repo",
        "--output-dir", str(output_dir),
        "--variant", "nonexistent",
    ])
    monkeypatch.setenv("HF_TOKEN", "fake-token")

    import types
    fake_hf_module = types.ModuleType("huggingface_hub")
    fake_hf_module.HfApi = FakeHfApi
    fake_hf_module.CommitOperationAdd = None
    fake_hf_module.CommitOperationDelete = None
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_module)

    with pytest.raises(SystemExit) as exc_info:
        upload.main()
    assert exc_info.value.code == 1


def test_upload_main_reset_single_variant(tmp_path, monkeypatch) -> None:
    """Verify --reset deletes and recreates repo before single-variant upload."""
    upload = _load_pipeline_module("13_upload_hf.py")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    _make_complete_upload_bundle(
        output_dir / "reference",
        git_hash="deadbeef",
        model_dir="models/reference",
    )

    calls: list[str] = []
    committed: dict[str, Any] = {}

    class FakeCommitOperationAdd:
        def __init__(self, *, path_in_repo, path_or_fileobj):
            self.path_in_repo = path_in_repo
            self.path_or_fileobj = path_or_fileobj

    class FakeCommitOperationDelete:
        def __init__(self, *, path_in_repo):
            self.path_in_repo = path_in_repo

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def delete_repo(self, *, repo_id):
            calls.append(f"delete_repo:{repo_id}")

        def create_repo(self, **kwargs):
            calls.append(f"create_repo:{kwargs.get('repo_id')}")

        def list_repo_files(self, *, repo_id, revision=None):
            # Fresh repo after reset — no stale files
            return [".gitattributes"]

        def create_commit(self, *, repo_id, operations, commit_message, revision=None):
            committed["operations"] = operations

    monkeypatch.setattr("sys.argv", [
        "13_upload_hf.py",
        "--repo-id", "test/repo",
        "--output-dir", str(output_dir),
        "--variant", "reference",
        "--reset",
    ])
    monkeypatch.setenv("HF_TOKEN", "fake-token")

    import types
    fake_hf_module = types.ModuleType("huggingface_hub")
    fake_hf_module.HfApi = FakeHfApi
    fake_hf_module.CommitOperationAdd = FakeCommitOperationAdd
    fake_hf_module.CommitOperationDelete = FakeCommitOperationDelete
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_module)

    upload.main()

    # delete_repo must be called before create_repo
    assert calls == ["delete_repo:test/repo", "create_repo:test/repo"]

    # No delete operations because repo is fresh (only .gitattributes, which is preserved)
    ops = committed["operations"]
    delete_ops = [op for op in ops if isinstance(op, FakeCommitOperationDelete)]
    assert delete_ops == []

    # Upload operations should still be present
    add_paths = {op.path_in_repo for op in ops if isinstance(op, FakeCommitOperationAdd)}
    assert "config.json" in add_paths
    assert "model.onnx" in add_paths


def test_upload_main_reset_multi_variant(tmp_path, monkeypatch) -> None:
    """Verify --reset deletes and recreates repo before multi-variant upload."""
    upload = _load_pipeline_module("13_upload_hf.py")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    for name in ("reference", "ablation_none"):
        _make_complete_upload_bundle(
            output_dir / name,
            git_hash="deadbeef",
            model_dir=f"models/{name}",
        )

    calls: list[str] = []
    uploaded: list[str] = []

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def delete_repo(self, *, repo_id):
            calls.append(f"delete_repo:{repo_id}")

        def create_repo(self, **kwargs):
            calls.append(f"create_repo:{kwargs.get('repo_id')}")

        def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, revision=None):
            uploaded.append(path_in_repo)

    monkeypatch.setattr("sys.argv", [
        "13_upload_hf.py",
        "--repo-id", "test/repo",
        "--output-dir", str(output_dir),
        "--reset",
    ])
    monkeypatch.setenv("HF_TOKEN", "fake-token")

    import types
    fake_hf_module = types.ModuleType("huggingface_hub")
    fake_hf_module.HfApi = FakeHfApi
    fake_hf_module.CommitOperationAdd = None
    fake_hf_module.CommitOperationDelete = None
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_module)

    upload.main()

    # delete_repo must be called before create_repo
    assert calls == ["delete_repo:test/repo", "create_repo:test/repo"]

    # Multi-variant files should still be uploaded with prefixes
    assert "reference/config.json" in uploaded
    assert "ablation_none/config.json" in uploaded


def test_upload_main_reset_delete_repo_failure_logs(tmp_path, monkeypatch, caplog) -> None:
    """Verify --reset logs when delete_repo fails (e.g., repo doesn't exist) and continues."""
    upload = _load_pipeline_module("13_upload_hf.py")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    _make_complete_upload_bundle(
        output_dir / "reference",
        git_hash="deadbeef",
        model_dir="models/reference",
    )

    calls: list[str] = []

    class FakeCommitOperationAdd:
        def __init__(self, *, path_in_repo, path_or_fileobj):
            self.path_in_repo = path_in_repo
            self.path_or_fileobj = path_or_fileobj

    class FakeCommitOperationDelete:
        def __init__(self, *, path_in_repo):
            self.path_in_repo = path_in_repo

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def delete_repo(self, *, repo_id):
            raise RuntimeError("Repository not found")

        def create_repo(self, **kwargs):
            calls.append("create_repo")

        def list_repo_files(self, *, repo_id, revision=None):
            return []

        def create_commit(self, *, repo_id, operations, commit_message, revision=None):
            calls.append("create_commit")

    monkeypatch.setattr("sys.argv", [
        "13_upload_hf.py",
        "--repo-id", "test/repo",
        "--output-dir", str(output_dir),
        "--variant", "reference",
        "--reset",
    ])
    monkeypatch.setenv("HF_TOKEN", "fake-token")

    import types
    fake_hf_module = types.ModuleType("huggingface_hub")
    fake_hf_module.HfApi = FakeHfApi
    fake_hf_module.CommitOperationAdd = FakeCommitOperationAdd
    fake_hf_module.CommitOperationDelete = FakeCommitOperationDelete
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_module)

    import logging
    with caplog.at_level(logging.INFO):
        upload.main()

    # Should have logged the delete_repo failure
    assert any("delete_repo skipped" in msg for msg in caplog.messages), (
        f"Expected 'delete_repo skipped' in log messages, got: {caplog.messages}"
    )

    # Should still continue with create_repo and upload
    assert "create_repo" in calls
    assert "create_commit" in calls
