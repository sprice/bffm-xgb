"""
Shared loader for the trained per-domain quantile models.

Stages 08 (validate), 09 (baselines), and 10 (simulate) each used to carry a
near-verbatim copy of this loader. They now import the single implementation
here so the on-disk filename scheme, the quantile set, and the load-error
policy live in exactly one place.

Filenames follow ``f"{MODEL_STEM}_{domain}_{q}.joblib"`` (see lib.constants).
For backward compatibility with bundles produced before the canonical_v1
rename, the loader also reads the legacy ``adaptive_*`` stem, logging a
deprecation warning. Stage 11 keeps its own flat-dict loader (different shape
and a fail-fast policy that a test depends on); it shares only MODEL_STEM.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib

from lib.constants import DOMAINS, LEGACY_MODEL_STEM, MODEL_STEM, QUANTILE_NAME_LIST

log = logging.getLogger(__name__)


def _model_path(models_dir: Path, domain: str, q_name: str) -> Path:
    """Resolve the joblib path for one domain/quantile model.

    Prefers the current ``{MODEL_STEM}_*`` stem; falls back to the legacy
    ``{LEGACY_MODEL_STEM}_*`` name (with a deprecation warning) so stale local
    bundles keep loading until they are re-exported. Returns the (possibly
    non-existent) current-stem path when neither file is present, so the caller
    treats the model as missing.
    """
    current = models_dir / f"{MODEL_STEM}_{domain}_{q_name}.joblib"
    if current.exists():
        return current
    legacy = models_dir / f"{LEGACY_MODEL_STEM}_{domain}_{q_name}.joblib"
    if legacy.exists():
        log.warning(
            "Loading legacy-named model %s; re-export to migrate to %s_*.joblib",
            legacy.name,
            MODEL_STEM,
        )
        return legacy
    return current


def load_domain_models(models_dir: Path) -> dict[str, dict[str, Any]]:
    """Load the trained per-domain quantile models from *models_dir*.

    Returns a nested dict ``{domain: {q_name: model}}``. Only models present on
    disk are loaded; use :func:`missing_models` to detect an incomplete bundle.
    """
    domain_models: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        domain_models[domain] = {}
        for q_name in QUANTILE_NAME_LIST:
            path = _model_path(models_dir, domain, q_name)
            if path.exists():
                domain_models[domain][q_name] = joblib.load(path)
    return domain_models


def missing_models(domain_models: dict[str, dict[str, Any]]) -> list[str]:
    """Return the ``"{domain}_{q_name}"`` keys missing from *domain_models*."""
    missing: list[str] = []
    for domain in DOMAINS:
        models = domain_models.get(domain, {})
        for q_name in QUANTILE_NAME_LIST:
            if q_name not in models:
                missing.append(f"{domain}_{q_name}")
    return missing
