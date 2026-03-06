"""Helpers for loading and validating Mini-IPIP mapping artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import DOMAINS, DOMAIN_LABELS, ITEM_COLUMNS

DEFAULT_MAPPING_FILENAME = "mini_ipip_mapping.json"


def _normalize_domain_key(domain_key: str) -> str:
    """Map domain labels/abbr to canonical internal abbreviations."""
    if domain_key in DOMAINS:
        return domain_key
    label_to_abbr = {label: abbr for abbr, label in DOMAIN_LABELS.items()}
    if domain_key in label_to_abbr:
        return label_to_abbr[domain_key]
    raise ValueError(f"Unknown Mini-IPIP domain key: {domain_key!r}")


def _validate_mapping_payload(
    payload: dict[str, Any],
    *,
    expected_items_per_domain: int = 4,
) -> dict[str, list[str]]:
    domains_raw = payload.get("domains")
    if not isinstance(domains_raw, dict):
        raise ValueError("Mini-IPIP mapping payload must include a 'domains' object.")

    normalized: dict[str, list[str]] = {}
    for domain_key, domain_items in domains_raw.items():
        if not isinstance(domain_key, str):
            raise ValueError("Mini-IPIP domain keys must be strings.")
        if not isinstance(domain_items, list):
            raise ValueError(f"Mini-IPIP domain {domain_key!r} must map to a list of item IDs.")
        if not all(isinstance(item_id, str) for item_id in domain_items):
            raise ValueError(f"Mini-IPIP domain {domain_key!r} includes non-string item IDs.")

        domain = _normalize_domain_key(domain_key)
        if domain in normalized:
            raise ValueError(f"Mini-IPIP mapping defines duplicate domain {domain!r}.")
        normalized[domain] = list(domain_items)

    missing_domains = [domain for domain in DOMAINS if domain not in normalized]
    unexpected_domains = sorted(set(normalized.keys()) - set(DOMAINS))
    if missing_domains or unexpected_domains:
        raise ValueError(
            "Mini-IPIP mapping must contain exactly the five Big-5 domains; "
            f"missing={missing_domains}, unexpected={unexpected_domains}"
        )

    all_items: list[str] = []
    for domain in DOMAINS:
        items = normalized[domain]
        if len(items) != expected_items_per_domain:
            raise ValueError(
                f"Mini-IPIP domain {domain!r} must have exactly "
                f"{expected_items_per_domain} items; found {len(items)}."
            )
        all_items.extend(items)

    unique_items = set(all_items)
    if len(unique_items) != len(all_items):
        raise ValueError("Mini-IPIP mapping contains duplicate item IDs across domains.")

    invalid_items = sorted(item_id for item_id in unique_items if item_id not in ITEM_COLUMNS)
    if invalid_items:
        raise ValueError(f"Mini-IPIP mapping contains unknown item IDs: {invalid_items}")

    return {domain: list(normalized[domain]) for domain in DOMAINS}


def load_mini_ipip_mapping(
    path: Path,
    *,
    expected_items_per_domain: int = 4,
) -> dict[str, list[str]]:
    """Load and validate Mini-IPIP mapping, returning internal-domain keyed mapping."""
    if not path.exists():
        raise FileNotFoundError(
            f"Mini-IPIP mapping not found: {path}. "
            "Run with a valid artifacts/mini_ipip_mapping.json."
        )

    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Mini-IPIP mapping must be a JSON object: {path}")

    return _validate_mapping_payload(
        payload,
        expected_items_per_domain=expected_items_per_domain,
    )


def flatten_mini_ipip_items(mapping: dict[str, list[str]]) -> list[str]:
    """Return deterministic flat list of Mini-IPIP item IDs in domain order."""
    missing_domains = [domain for domain in DOMAINS if domain not in mapping]
    if missing_domains:
        raise ValueError(f"Mini-IPIP mapping missing domains: {missing_domains}")
    return [item_id for domain in DOMAINS for item_id in mapping[domain]]
