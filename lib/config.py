"""Load a pipeline variant config with shared defaults merged underneath.

Variant configs (``configs/<variant>.yaml``) only carry their deltas; the keys
that are byte-identical across every variant live once in ``configs/_base.yaml``
and are deep-merged underneath the variant. The merge is value-equivalent to the
old standalone YAMLs (locked by a test), and config content is not hashed, so
this changes no data/hyperparameter hash and forces no retrain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

BASE_CONFIG_NAME = "_base.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` with ``override`` merged on top (nested dicts key-by-key)."""
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def load_config_with_base(config_path: str | Path) -> dict[str, Any]:
    """Load a variant config, deep-merging ``_base.yaml`` (same dir) underneath.

    The variant always wins on conflicts. If no ``_base.yaml`` is present, the
    variant is returned unchanged.
    """
    config_path = Path(config_path)
    variant = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(variant, dict):
        raise ValueError(f"Expected a YAML mapping at {config_path}")
    base_path = config_path.parent / BASE_CONFIG_NAME
    if not base_path.exists():
        return variant
    base = yaml.safe_load(base_path.read_text()) or {}
    if not isinstance(base, dict):
        raise ValueError(f"Expected a YAML mapping at {base_path}")
    return _deep_merge(base, variant)
