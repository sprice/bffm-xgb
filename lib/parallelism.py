"""Utilities for resolving deterministic XGBoost parallelism settings."""

from __future__ import annotations

import os
from typing import Any


XGB_N_JOBS_ENV = "BFFM_XGB_N_JOBS"


def coerce_positive_int(value: Any, *, label: str) -> int:
    """Convert value to positive integer or raise ValueError."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer (got bool).")

    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            raise ValueError(f"{label} must be a positive integer (got empty string).")
        try:
            parsed = int(raw)
        except ValueError as exc:
            raise ValueError(f"{label} must be a positive integer (got {value!r}).") from exc
    else:
        raise ValueError(f"{label} must be a positive integer (got {type(value).__name__}).")

    if parsed <= 0:
        raise ValueError(f"{label} must be >= 1 (got {parsed}).")
    return parsed


def resolve_default_xgb_n_jobs(
    *,
    env_var: str = XGB_N_JOBS_ENV,
) -> tuple[int, str]:
    """Resolve default XGBoost worker count from env var or CPU count."""
    env_value = os.getenv(env_var)
    if isinstance(env_value, str) and env_value.strip():
        return coerce_positive_int(env_value, label=f"${env_var}"), f"env:{env_var}"

    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 1, "cpu_count_unavailable"
    if cpu_count < 1:
        return 1, "cpu_count_invalid"
    return int(cpu_count), "cpu_count"

