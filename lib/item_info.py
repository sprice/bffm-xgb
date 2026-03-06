"""Strict loading/validation for stage-05 item_info artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .constants import DOMAINS, ITEM_COLUMNS


def _as_float(value: Any) -> float:
    """Convert a numeric value to finite float or raise ValueError."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected numeric value, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("expected finite numeric value")
    return result


def normalize_item_info(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize camelCase/snake_case item_info payloads to snake_case."""
    if not isinstance(raw, dict):
        raise ValueError("item_info payload must be a JSON object")

    if "itemPool" in raw:
        first_item = {
            "id": raw.get("firstItemId"),
            "text": raw.get("firstItemText", ""),
            "domain": raw.get("firstItemDomain", ""),
        }
        item_pool = [
            {
                "id": item.get("id"),
                "home_domain": item.get("homeDomain", ""),
                "own_domain_r": item.get("ownDomainR", 0.0),
                "cross_domain_info": item.get("crossDomainInfo", 0.0),
                "domain_correlations": item.get("domainCorrelations", {}),
                "is_reverse_keyed": item.get("isReverseKeyed", False),
                "rank": item.get("rank"),
            }
            for item in raw.get("itemPool", [])
            if isinstance(item, dict)
        ]
        inter_item_r_bar = raw.get("interItemRBar", raw.get("inter_item_r_bar", {}))
        return {
            "first_item": first_item,
            "item_pool": item_pool,
            "inter_item_r_bar": inter_item_r_bar,
        }

    if "item_pool" in raw:
        first_item_raw = raw.get("first_item", {})
        first_item = first_item_raw if isinstance(first_item_raw, dict) else {}
        item_pool = []
        for item in raw.get("item_pool", []):
            if not isinstance(item, dict):
                continue
            item_id = item.get("id")
            home_domain = item.get("home_domain")
            if not isinstance(home_domain, str) and isinstance(item_id, str):
                home_domain = item_id.rstrip("0123456789")
            item_pool.append(
                {
                    "id": item_id,
                    "home_domain": home_domain or "",
                    "own_domain_r": item.get("own_domain_r", 0.0),
                    "cross_domain_info": item.get("cross_domain_info", 0.0),
                    "domain_correlations": item.get("domain_correlations", {}),
                    "is_reverse_keyed": item.get("is_reverse_keyed", False),
                    "rank": item.get("rank"),
                }
            )
        return {
            "first_item": first_item,
            "item_pool": item_pool,
            "inter_item_r_bar": raw.get("inter_item_r_bar", raw.get("interItemRBar", {})),
        }

    raise ValueError("item_info payload is missing required itemPool/item_pool key")


def validate_item_info(
    item_info: dict[str, Any],
    *,
    require_first_item: bool = False,
    require_inter_item_r_bar: bool = False,
) -> None:
    """Validate normalized item_info content and fail closed on corruption."""
    item_pool = item_info.get("item_pool")
    if not isinstance(item_pool, list) or not item_pool:
        raise ValueError("item_info.item_pool must be a non-empty list")

    item_ids: list[str] = []
    seen: set[str] = set()
    for idx, item in enumerate(item_pool):
        if not isinstance(item, dict):
            raise ValueError(f"item_pool[{idx}] must be an object")
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"item_pool[{idx}] is missing a valid id")
        if item_id in seen:
            raise ValueError(f"item_pool contains duplicate id: {item_id}")
        seen.add(item_id)
        item_ids.append(item_id)

        home_domain = item.get("home_domain")
        if not isinstance(home_domain, str) or home_domain not in DOMAINS:
            raise ValueError(f"item_pool[{idx}] has invalid home_domain for id={item_id}")
        expected_domain = item_id.rstrip("0123456789")
        if home_domain != expected_domain:
            raise ValueError(
                f"item_pool[{idx}] home_domain={home_domain!r} mismatches id={item_id!r}"
            )

        rank = item.get("rank")
        rank_f = _as_float(rank)
        if rank_f < 1:
            raise ValueError(f"item_pool[{idx}] rank must be >= 1")

        _as_float(item.get("own_domain_r", 0.0))
        _as_float(item.get("cross_domain_info", 0.0))

        domain_corrs = item.get("domain_correlations")
        if not isinstance(domain_corrs, dict):
            raise ValueError(f"item_pool[{idx}] domain_correlations must be an object")
        if not domain_corrs:
            raise ValueError(f"item_pool[{idx}] domain_correlations must be non-empty")
        for domain, corr in domain_corrs.items():
            if domain not in DOMAINS:
                raise ValueError(
                    f"item_pool[{idx}] has unknown domain_correlations key: {domain!r}"
                )
            _as_float(corr)

    expected_ids = set(ITEM_COLUMNS)
    actual_ids = set(item_ids)
    missing_ids = sorted(expected_ids - actual_ids)
    extra_ids = sorted(actual_ids - expected_ids)
    if missing_ids or extra_ids:
        details = []
        if missing_ids:
            details.append(f"missing={missing_ids}")
        if extra_ids:
            details.append(f"extra={extra_ids}")
        raise ValueError("item_pool ids do not match expected item set: " + ", ".join(details))

    first_item_raw = item_info.get("first_item", {})
    if require_first_item:
        if not isinstance(first_item_raw, dict):
            raise ValueError("item_info.first_item must be present")
        first_item_id = first_item_raw.get("id")
        if not isinstance(first_item_id, str) or first_item_id not in actual_ids:
            raise ValueError("item_info.first_item.id must reference an item in item_pool")

    inter_item_r_bar = item_info.get("inter_item_r_bar", {})
    if require_inter_item_r_bar:
        if not isinstance(inter_item_r_bar, dict):
            raise ValueError("item_info.inter_item_r_bar must be an object when required")
        for domain in DOMAINS:
            if domain not in inter_item_r_bar:
                raise ValueError(f"item_info.inter_item_r_bar missing domain {domain!r}")
            value = _as_float(inter_item_r_bar[domain])
            if value <= 0:
                raise ValueError(
                    f"item_info.inter_item_r_bar[{domain!r}] must be > 0 for SEM stopping"
                )


def load_item_info_strict(
    path: Path,
    *,
    require_first_item: bool = False,
    require_inter_item_r_bar: bool = False,
    expected_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Load + normalize + validate item_info from a JSON path."""
    if not path.exists():
        raise FileNotFoundError(
            f"Item info file not found: {path}. Run pipeline stage 05 (make correlations)."
        )

    with open(path) as f:
        raw = json.load(f)

    if expected_source_sha256 is not None:
        expected_norm = _normalize_sha256_hex(expected_source_sha256)
        if expected_norm is None:
            raise ValueError(
                "expected_source_sha256 must be a non-empty 64-char SHA-256 hex string."
            )

        source_norm = _extract_item_info_source_sha256(raw)
        if source_norm is None:
            raise ValueError(
                "item_info provenance is missing source_sha256 hash lock. "
                "Re-run stage 05 (make correlations) to regenerate item_info.json."
            )
        if source_norm != expected_norm:
            raise ValueError(
                "item_info provenance mismatch for train split hash: "
                f"expected_source_sha256={expected_norm}, item_info_source_sha256={source_norm}. "
                "Re-run stage 05 (make correlations) for this data split."
            )

    item_info = normalize_item_info(raw)
    validate_item_info(
        item_info,
        require_first_item=require_first_item,
        require_inter_item_r_bar=require_inter_item_r_bar,
    )
    return item_info


def file_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest for a file path."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _find_training_report_path(model_dir: Path) -> Path:
    """Find canonical training report file in a model directory."""
    for filename in ("training_report.json", "adaptive_training_report.json"):
        candidate = model_dir / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Training report not found in model bundle: {model_dir}. "
        "Expected training_report.json."
    )


def _normalize_sha256_hex(value: Any) -> str | None:
    """Normalize a SHA-256 hex string, returning None for empty values."""
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError("Expected 64-char hex SHA-256 value.")
    return normalized


def _extract_item_info_source_sha256(raw_payload: Any) -> str | None:
    """Extract source_sha256 from top-level/provenance item_info payload fields."""
    if not isinstance(raw_payload, dict):
        return None

    candidates: list[Any] = [
        raw_payload.get("source_sha256"),
        raw_payload.get("sourceSha256"),
    ]
    provenance = raw_payload.get("provenance")
    if isinstance(provenance, dict):
        candidates.extend(
            [
                provenance.get("source_sha256"),
                provenance.get("sourceSha256"),
            ]
        )

    for candidate in candidates:
        normalized = _normalize_sha256_hex(candidate)
        if normalized is not None:
            return normalized
    return None


def load_training_report(model_dir: Path) -> tuple[dict[str, Any], Path]:
    """Load canonical training report JSON for a model bundle."""
    report_path = _find_training_report_path(model_dir)
    with open(report_path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Training report must be a JSON object: {report_path}")
    return payload, report_path


def extract_training_data_sha256(report: dict[str, Any], key: str) -> str | None:
    """Extract optional data-level SHA-256 field from training report data block."""
    data = report.get("data", {})
    if not isinstance(data, dict):
        return None
    value = data.get(key)
    return _normalize_sha256_hex(value)


def extract_training_split_signature(report: dict[str, Any]) -> str | None:
    """Extract optional split signature SHA-256 from training report."""
    data = report.get("data", {})
    if not isinstance(data, dict):
        return None
    value = data.get("split_signature")
    return _normalize_sha256_hex(value)


def _extract_item_info_sha256(report: dict[str, Any]) -> str:
    """Extract expected item_info SHA-256 from training report payload."""
    candidate_paths = [
        ("data", "item_info_sha256"),
        ("artifacts", "item_info_sha256"),
    ]
    for section, key in candidate_paths:
        section_payload = report.get(section)
        if isinstance(section_payload, dict):
            value = section_payload.get(key)
            if isinstance(value, str):
                normalized = value.strip().lower()
                if len(normalized) == 64 and all(c in "0123456789abcdef" for c in normalized):
                    return normalized
                raise ValueError(
                    f"Invalid {section}.{key} in training report (expected 64-char hex SHA-256)."
                )

    raise ValueError(
        "Model bundle is missing item_info provenance hash (data.item_info_sha256). "
        "Re-run stage 07 training with strict provenance enabled."
    )


def load_item_info_for_model(
    model_dir: Path,
    data_dir: Path,
    *,
    require_first_item: bool = False,
    require_inter_item_r_bar: bool = False,
) -> tuple[dict[str, Any], Path, str]:
    """Load stage-05 item_info and verify it matches the selected model bundle.

    Verification is hash-based and fail-closed:
    - model_dir/training_report.json must exist and include data.item_info_sha256
    - data_dir/item_info.json must exist and pass strict schema validation
    - current file SHA-256 must exactly match the expected hash from training_report
    """
    report_path = _find_training_report_path(model_dir)
    with open(report_path) as f:
        report = json.load(f)
    if not isinstance(report, dict):
        raise ValueError(f"Training report must be a JSON object: {report_path}")

    expected_sha256 = _extract_item_info_sha256(report)

    item_info_path = data_dir / "item_info.json"
    item_info = load_item_info_strict(
        item_info_path,
        require_first_item=require_first_item,
        require_inter_item_r_bar=require_inter_item_r_bar,
    )
    actual_sha256 = file_sha256(item_info_path).lower()

    if actual_sha256 != expected_sha256:
        raise ValueError(
            "item_info provenance mismatch: stage-05 item_info.json does not match "
            f"model bundle {model_dir}. "
            f"expected_sha256={expected_sha256}, actual_sha256={actual_sha256}. "
            "Re-run stage 05 then stage 07, and regenerate downstream artifacts."
        )

    return item_info, item_info_path, actual_sha256
