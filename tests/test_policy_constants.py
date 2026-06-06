"""WS3: training/assessment policy constants are single-sourced.

The deployment-aligned tuning objective (stage 06), the coverage-calibration
scaling policy (stage 07), and the adaptive-stopping policy (stage 10) used to
be hardcoded numeric literals duplicated across the pipeline *and* re-typed in
the docs/web. They now live once in ``lib.constants`` and are consumed by the
pipeline stages and the doc generator. These tests lock the values and assert
each consumer reads them (value-preserving refactor — no retrain).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from lib.constants import (
    ADAPTIVE_STOP,
    CALIBRATION_POLICY,
    TUNING_OBJECTIVE,
    TUNING_OBJECTIVE_FALLBACK,
)

_COUNTER = 0


def _load_module(subdir: str, script_name: str):
    global _COUNTER
    _COUNTER += 1
    module_name = f"test_policy_{subdir}_{script_name.replace('.', '_')}_{_COUNTER}"
    module_path = Path(__file__).resolve().parent.parent / subdir / script_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_constant_values_locked():
    assert TUNING_OBJECTIVE == {
        "sparse20_weight": 0.80,
        "full50_weight": 0.20,
        "sparse20_penalty_weight": 2.0,
        "sparse20_penalty_floor": 0.85,
        "full50_penalty_weight": 1.0,
        "full50_penalty_floor": 0.95,
    }
    assert CALIBRATION_POLICY == {
        "coverage_low": 0.85,
        "coverage_high": 0.95,
        "target_coverage": 0.90,
        "coverage_floor": 0.5,
    }
    assert ADAPTIVE_STOP == {"sem_threshold": 0.45, "min_items_per_domain": 4}
    assert TUNING_OBJECTIVE_FALLBACK == {"penalty_weight": 1.5, "min_r_floor": 0.90}


def test_calibration_scale_uses_policy():
    """stage 07's coverage scaler reproduces the old _scale_for_coverage exactly."""
    train = _load_module("pipeline", "07_train.py")
    f = train.scale_for_coverage
    assert f(0.80) == 0.90 / 0.80          # below low -> target / coverage
    assert f(0.40) == 0.90 / 0.5           # below low, floor clamps the denominator
    assert f(0.97) == 0.90 / 0.97          # above high -> target / coverage
    assert f(0.90) == 1.0                   # in band
    assert f(0.85) == 1.0                   # low boundary is inclusive (not < low)
    assert f(0.95) == 1.0                   # high boundary is inclusive (not > high)


def test_tuning_objective_helper_matches_old_formula():
    """stage 06's deployment-aligned objective reproduces the old arithmetic."""
    tune = _load_module("pipeline", "06_tune.py")
    g = tune.deployment_aligned_objective
    ms, mn, mf = 0.91, 0.83, 0.99
    expected = (
        (0.80 * ms + 0.20 * mf)
        - 2.0 * max(0.0, 0.85 - mn)
        - 1.0 * max(0.0, 0.95 - mf)
    )
    assert g(ms, mn, mf) == expected
    # Both floors cleared -> penalties vanish.
    assert g(0.92, 0.90, 0.99) == 0.80 * 0.92 + 0.20 * 0.99


def test_full50_fallback_objective_matches_old_formula():
    """stage 06's full-50-only fallback reproduces the old 1.5/0.90 arithmetic."""
    tune = _load_module("pipeline", "06_tune.py")
    g = tune.full50_fallback_objective
    mean_full, min_full = 0.97, 0.88
    assert g(mean_full, min_full) == mean_full - 1.5 * max(0.0, 0.90 - min_full)
    # Floor cleared -> no penalty.
    assert g(0.97, 0.95) == 0.97


def test_adaptive_config_defaults_from_constants():
    sim = _load_module("pipeline", "10_simulate.py")
    cfg = sim.AdaptiveConfig()
    assert cfg.sem_threshold == ADAPTIVE_STOP["sem_threshold"]
    assert cfg.min_items_per_domain == ADAPTIVE_STOP["min_items_per_domain"]


def test_repo_facts_emits_policy_constants():
    gdd = _load_module("scripts", "generate_doc_data.py")
    facts = gdd.Facts()
    rf = gdd.build_repo_facts(facts)
    assert rf["tuningObjective"]["sparse20Weight"] == TUNING_OBJECTIVE["sparse20_weight"]
    assert rf["tuningObjective"]["full50Weight"] == TUNING_OBJECTIVE["full50_weight"]
    assert rf["tuningObjective"]["sparse20PenaltyWeight"] == TUNING_OBJECTIVE["sparse20_penalty_weight"]
    assert rf["tuningObjective"]["sparse20PenaltyFloor"] == TUNING_OBJECTIVE["sparse20_penalty_floor"]
    assert rf["tuningObjective"]["full50PenaltyWeight"] == TUNING_OBJECTIVE["full50_penalty_weight"]
    assert rf["tuningObjective"]["full50PenaltyFloor"] == TUNING_OBJECTIVE["full50_penalty_floor"]
    assert rf["calibration"]["coverageLow"] == CALIBRATION_POLICY["coverage_low"]
    assert rf["calibration"]["coverageHigh"] == CALIBRATION_POLICY["coverage_high"]
    assert rf["calibration"]["targetCoverage"] == CALIBRATION_POLICY["target_coverage"]
    assert rf["calibration"]["coverageFloor"] == CALIBRATION_POLICY["coverage_floor"]
    assert rf["adaptiveStop"]["semThreshold"] == ADAPTIVE_STOP["sem_threshold"]
    assert rf["adaptiveStop"]["minItemsPerDomain"] == ADAPTIVE_STOP["min_items_per_domain"]
