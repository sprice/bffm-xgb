"""Tests for deterministic XGBoost parallelism resolution helpers."""

from __future__ import annotations

import pytest

from lib.parallelism import coerce_positive_int, resolve_default_xgb_n_jobs


def test_coerce_positive_int_accepts_int_and_numeric_string() -> None:
    assert coerce_positive_int(8, label="x") == 8
    assert coerce_positive_int("12", label="x") == 12


def test_coerce_positive_int_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        coerce_positive_int(0, label="x")
    with pytest.raises(ValueError):
        coerce_positive_int(-3, label="x")
    with pytest.raises(ValueError):
        coerce_positive_int("abc", label="x")
    with pytest.raises(ValueError):
        coerce_positive_int(True, label="x")


def test_resolve_default_xgb_n_jobs_prefers_env(monkeypatch) -> None:
    monkeypatch.setenv("BFFM_XGB_N_JOBS", "14")
    n_jobs, source = resolve_default_xgb_n_jobs()
    assert n_jobs == 14
    assert source == "env:BFFM_XGB_N_JOBS"


def test_resolve_default_xgb_n_jobs_uses_cpu_count(monkeypatch) -> None:
    monkeypatch.delenv("BFFM_XGB_N_JOBS", raising=False)
    monkeypatch.setattr("os.cpu_count", lambda: 10)
    n_jobs, source = resolve_default_xgb_n_jobs()
    assert n_jobs == 10
    assert source == "cpu_count"


def test_resolve_default_xgb_n_jobs_falls_back_to_one(monkeypatch) -> None:
    monkeypatch.delenv("BFFM_XGB_N_JOBS", raising=False)
    monkeypatch.setattr("os.cpu_count", lambda: None)
    n_jobs, source = resolve_default_xgb_n_jobs()
    assert n_jobs == 1
    assert source == "cpu_count_unavailable"

