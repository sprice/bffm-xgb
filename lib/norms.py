"""Load and validate percentile norms from the committed norms artifact."""

from __future__ import annotations

import json
import os
import math
from functools import lru_cache
from pathlib import Path

from .constants import DOMAINS

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
NORM_ENV_VAR = "IPIP_BFFM_NORMS_PATH"


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PACKAGE_ROOT / candidate


def get_default_norms_path() -> Path:
    """Return configured norms file path (env override, else default artifact path)."""
    override = os.getenv(NORM_ENV_VAR)
    if override:
        return _resolve_path(override)
    return PACKAGE_ROOT / "artifacts" / "ipip_bffm_norms.json"


def _extract_norm_block(
    payload: dict,
    *,
    key: str,
    allow_top_level_fallback: bool,
) -> dict[str, dict[str, float]]:
    raw_norms = payload.get(key, payload) if allow_top_level_fallback else payload.get(key)
    if not isinstance(raw_norms, dict):
        raise ValueError(f"Norm payload must contain a '{key}' object")

    norms: dict[str, dict[str, float]] = {}
    for domain in DOMAINS:
        domain_stats = raw_norms.get(domain)
        if not isinstance(domain_stats, dict):
            raise ValueError(f"Missing {key} stats for domain '{domain}'")

        mean = domain_stats.get("mean")
        sd = domain_stats.get("sd")
        if not isinstance(mean, (int, float)) or not isinstance(sd, (int, float)):
            raise ValueError(f"{key} for domain '{domain}' must include numeric mean/sd")

        mean_f = float(mean)
        sd_f = float(sd)
        if not math.isfinite(mean_f) or not math.isfinite(sd_f) or not (sd_f > 0):
            raise ValueError(
                f"{key} mean/sd must be finite and sd > 0 for domain '{domain}'"
            )

        norms[domain] = {"mean": mean_f, "sd": sd_f}

    return norms


@lru_cache(maxsize=8)
def _load_norm_payload_cached(path_key: str) -> dict:
    path = Path(path_key)
    if not path.exists():
        raise FileNotFoundError(
            f"Norms file not found: {path}. Run stage 03 (make norms) first."
        )
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid norms payload in {path}: expected JSON object")
    return payload


def load_norms(path: str | Path | None = None) -> dict[str, dict[str, float]]:
    """Load norms from JSON and return a defensive copy."""
    resolved = _resolve_path(path) if path is not None else get_default_norms_path()
    payload = _load_norm_payload_cached(str(resolved))
    norms = _extract_norm_block(
        payload,
        key="norms",
        allow_top_level_fallback=True,
    )
    return {domain: {"mean": stats["mean"], "sd": stats["sd"]} for domain, stats in norms.items()}


def clear_norms_cache() -> None:
    """Clear cached norms for tests or explicit reload workflows."""
    _load_norm_payload_cached.cache_clear()


def load_mini_ipip_norms(path: str | Path | None = None) -> dict[str, dict[str, float]]:
    """Load Mini-IPIP domain norms from the stage-03 norms artifact."""
    resolved = _resolve_path(path) if path is not None else get_default_norms_path()
    payload = _load_norm_payload_cached(str(resolved))
    norms = _extract_norm_block(
        payload,
        key="mini_ipip_norms",
        allow_top_level_fallback=False,
    )
    return {domain: {"mean": stats["mean"], "sd": stats["sd"]} for domain, stats in norms.items()}
