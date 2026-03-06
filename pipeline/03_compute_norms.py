#!/usr/bin/env python3
"""Compute Big-5 norms from stage-02 SQLite and manage the committed lock file."""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from lib.constants import DOMAINS, DOMAIN_LABELS, ITEM_COLUMNS
from lib.item_info import file_sha256
from lib.mini_ipip import load_mini_ipip_mapping
from lib.provenance import add_provenance_args, build_provenance, relative_to_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

NORM_SCHEMA_VERSION = 2
NORM_DATASET = "IPIP-FFM (openpsychometrics.org)"
NORM_TABLE = "responses"
NORM_SCOPE = "full cleaned dataset from stage 02 SQLite (not split-specific)"


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PACKAGE_ROOT / path


def _default_meta_path(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_suffix(".meta.json")
    return output_path.parent / f"{output_path.name}.meta.json"


def _load_domain_scores_from_sqlite(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(
            f"SQLite database not found: {db_path}. Run stage 02 (make load) first."
        )

    score_cols = [f"{d}_score" for d in DOMAINS]
    query_cols = ITEM_COLUMNS + score_cols
    query = f"SELECT {', '.join(query_cols)} FROM {NORM_TABLE}"
    with sqlite3.connect(str(db_path)) as conn:
        df = pd.read_sql_query(query, conn)

    if df.empty:
        raise ValueError(f"No rows found in {NORM_TABLE} table: {db_path}")

    missing_cols = [c for c in query_cols if c not in df.columns]
    if missing_cols:
        raise ValueError("Missing score columns in responses table: " + ", ".join(missing_cols))

    return df


def _compute_norms(df: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = {}
    for domain in DOMAINS:
        col = f"{domain}_score"
        values = pd.to_numeric(df[col], errors="coerce").dropna().astype(float)
        if values.empty:
            raise ValueError(f"{col} has no valid rows")

        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        n = int(values.shape[0])

        if not np.isfinite(mean) or not np.isfinite(sd) or sd <= 0:
            raise ValueError(
                f"Invalid norm stats for {domain}: mean={mean}, sd={sd}, n={n}"
            )
        stats[domain] = {"mean": mean, "sd": sd, "n": n}

    return stats


def _compute_mini_ipip_norms(
    df: pd.DataFrame,
    mini_ipip_mapping: dict[str, list[str]],
) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = {}
    for domain in DOMAINS:
        items = mini_ipip_mapping[domain]
        missing_items = [item_id for item_id in items if item_id not in df.columns]
        if missing_items:
            raise ValueError(
                f"Mini-IPIP mapping for {domain} includes missing columns: {missing_items}"
            )

        values = pd.to_numeric(df[items].mean(axis=1), errors="coerce").dropna().astype(float)
        if values.empty:
            raise ValueError(f"Mini-IPIP {domain} has no valid rows")

        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        n = int(values.shape[0])
        if not np.isfinite(mean) or not np.isfinite(sd) or sd <= 0:
            raise ValueError(
                f"Invalid Mini-IPIP norm stats for {domain}: mean={mean}, sd={sd}, n={n}"
            )
        stats[domain] = {"mean": mean, "sd": sd, "n": n}

    return stats


def _build_lock_payload(
    computed: dict[str, dict[str, float | int]],
    mini_ipip_computed: dict[str, dict[str, float | int]],
    mini_ipip_mapping_path: Path,
    mini_ipip_mapping_sha256: str,
) -> dict[str, Any]:
    n_total = int(min(int(computed[d]["n"]) for d in DOMAINS))
    return {
        "schema_version": NORM_SCHEMA_VERSION,
        "dataset": NORM_DATASET,
        "table": NORM_TABLE,
        "scope": NORM_SCOPE,
        "n_respondents": n_total,
        "mini_ipip_mapping": {
            "file": mini_ipip_mapping_path.name,
            "sha256": mini_ipip_mapping_sha256,
        },
        "norms": {
            domain: {
                "mean": float(computed[domain]["mean"]),
                "sd": float(computed[domain]["sd"]),
            }
            for domain in DOMAINS
        },
        "mini_ipip_norms": {
            domain: {
                "mean": float(mini_ipip_computed[domain]["mean"]),
                "sd": float(mini_ipip_computed[domain]["sd"]),
            }
            for domain in DOMAINS
        },
    }


def _extract_norms(payload: dict[str, Any], key: str) -> dict[str, dict[str, float]]:
    raw_norms = payload.get(key)
    if not isinstance(raw_norms, dict):
        raise ValueError(f"Norm lock file is missing a '{key}' object")

    extracted: dict[str, dict[str, float]] = {}
    for domain in DOMAINS:
        domain_stats = raw_norms.get(domain)
        if not isinstance(domain_stats, dict):
            raise ValueError(f"Norm lock file missing domain '{domain}' in '{key}'")
        mean = domain_stats.get("mean")
        sd = domain_stats.get("sd")
        if not isinstance(mean, (int, float)) or not isinstance(sd, (int, float)):
            raise ValueError(f"Norm lock file has invalid mean/sd for domain '{domain}' in '{key}'")
        mean_f = float(mean)
        sd_f = float(sd)
        if not np.isfinite(mean_f) or not np.isfinite(sd_f) or sd_f <= 0:
            raise ValueError(f"Norm lock file has non-finite stats for domain '{domain}' in '{key}'")
        extracted[domain] = {"mean": mean_f, "sd": sd_f}

    return extracted


def _load_expected_norms(
    lock_path: Path,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    if not lock_path.exists():
        raise FileNotFoundError(
            f"Norm lock file not found: {lock_path}. Run `make norms` to generate it."
        )

    with open(lock_path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Norm lock file must be a JSON object: {lock_path}")
    return _extract_norms(payload, "norms"), _extract_norms(payload, "mini_ipip_norms")


def _compute_diffs(
    *,
    computed: dict[str, dict[str, float | int]],
    expected: dict[str, dict[str, float]],
) -> tuple[dict[str, dict[str, float]], float]:
    diffs: dict[str, dict[str, float]] = {}
    max_abs_diff = 0.0
    for domain in DOMAINS:
        mean_diff = abs(float(computed[domain]["mean"]) - float(expected[domain]["mean"]))
        sd_diff = abs(float(computed[domain]["sd"]) - float(expected[domain]["sd"]))
        diffs[domain] = {"mean_abs_diff": mean_diff, "sd_abs_diff": sd_diff}
        max_abs_diff = max(max_abs_diff, mean_diff, sd_diff)
    return diffs, max_abs_diff


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute deterministic Big-5 norms and maintain the stage-03 lock file"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/processed/ipip_bffm.db"),
        help="Path to SQLite DB (default: data/processed/ipip_bffm.db)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ipip_bffm_norms.json"),
        help="Norm lock JSON path (default: artifacts/ipip_bffm_norms.json)",
    )
    parser.add_argument(
        "--meta-output",
        type=Path,
        default=None,
        help="Norm metadata JSON path (default: <output>.meta.json)",
    )
    parser.add_argument(
        "--mini-ipip-mapping",
        type=Path,
        default=Path("artifacts/mini_ipip_mapping.json"),
        help=(
            "Mini-IPIP mapping JSON used to compute standalone Mini-IPIP norms "
            "(default: artifacts/mini_ipip_mapping.json)"
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail closed if computed norms drift from the committed lock file",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="Max allowed absolute drift for --check (default: 1e-9)",
    )
    add_provenance_args(parser)
    args = parser.parse_args()

    if args.tolerance < 0:
        log.error("--tolerance must be >= 0")
        return 1

    db_path = _resolve_path(args.db_path)
    output_path = _resolve_path(args.output)
    mini_ipip_mapping_path = _resolve_path(args.mini_ipip_mapping)
    meta_path = (
        _resolve_path(args.meta_output)
        if args.meta_output is not None
        else _default_meta_path(output_path)
    )
    write_meta = (not args.check) or (args.meta_output is not None)

    log.info("=" * 60)
    log.info("IPIP-BFFM Norms (stage 03)")
    log.info("=" * 60)
    log.info("DB path:      %s", db_path)
    log.info("Norm lock:    %s", output_path)
    log.info("Mini-IPIP map:%s", mini_ipip_mapping_path)
    log.info("Meta output:  %s%s", meta_path, "" if write_meta else " (read-only check mode)")
    log.info("Check mode:   %s (tolerance=%g)", bool(args.check), args.tolerance)

    try:
        df = _load_domain_scores_from_sqlite(db_path)
        mini_ipip_mapping = load_mini_ipip_mapping(mini_ipip_mapping_path)
        mini_ipip_mapping_sha256 = file_sha256(mini_ipip_mapping_path)
        computed = _compute_norms(df)
        mini_ipip_computed = _compute_mini_ipip_norms(df, mini_ipip_mapping)
    except (
        FileNotFoundError,
        sqlite3.Error,
        ValueError,
        pd.errors.DatabaseError,
        json.JSONDecodeError,
        OSError,
    ) as e:
        log.error("%s", e)
        return 1

    lock_payload = _build_lock_payload(
        computed,
        mini_ipip_computed,
        mini_ipip_mapping_path=mini_ipip_mapping_path,
        mini_ipip_mapping_sha256=mini_ipip_mapping_sha256,
    )
    expected_norms: dict[str, dict[str, float]]
    expected_mini_ipip_norms: dict[str, dict[str, float]]
    diffs: dict[str, dict[str, float]]
    mini_ipip_diffs: dict[str, dict[str, float]]
    max_abs_diff: float
    check_passed = True

    if args.check:
        try:
            expected_norms, expected_mini_ipip_norms = _load_expected_norms(output_path)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
            log.error("%s", e)
            return 1
        diffs, max_abs_diff = _compute_diffs(computed=computed, expected=expected_norms)
        mini_ipip_diffs, mini_max_abs_diff = _compute_diffs(
            computed=mini_ipip_computed,
            expected=expected_mini_ipip_norms,
        )
        max_abs_diff = max(max_abs_diff, mini_max_abs_diff)
        check_passed = max_abs_diff <= args.tolerance
    else:
        expected_norms = lock_payload["norms"]
        expected_mini_ipip_norms = lock_payload["mini_ipip_norms"]
        diffs, max_abs_diff = _compute_diffs(computed=computed, expected=expected_norms)
        mini_ipip_diffs, mini_max_abs_diff = _compute_diffs(
            computed=mini_ipip_computed,
            expected=expected_mini_ipip_norms,
        )
        max_abs_diff = max(max_abs_diff, mini_max_abs_diff)
        _write_json(output_path, lock_payload)
        log.info("Wrote norm lock file: %s", output_path)

    log.info("Full-50 domain norms:")
    for domain in DOMAINS:
        stats = computed[domain]
        log.info(
            "  %s: mean=%.6f sd=%.6f n=%d (|d_mean|=%.3g |d_sd|=%.3g)",
            DOMAIN_LABELS[domain],
            float(stats["mean"]),
            float(stats["sd"]),
            int(stats["n"]),
            diffs[domain]["mean_abs_diff"],
            diffs[domain]["sd_abs_diff"],
        )
    log.info("Mini-IPIP domain norms:")
    for domain in DOMAINS:
        stats = mini_ipip_computed[domain]
        log.info(
            "  %s: mean=%.6f sd=%.6f n=%d (|d_mean|=%.3g |d_sd|=%.3g)",
            DOMAIN_LABELS[domain],
            float(stats["mean"]),
            float(stats["sd"]),
            int(stats["n"]),
            mini_ipip_diffs[domain]["mean_abs_diff"],
            mini_ipip_diffs[domain]["sd_abs_diff"],
        )

    lock_sha256 = file_sha256(output_path)
    provenance = build_provenance(
        Path(__file__).name,
        args=args,
        extra={
            "db_path": relative_to_root(db_path),
            "table": NORM_TABLE,
            "check_mode": bool(args.check),
            "mini_ipip_mapping": relative_to_root(mini_ipip_mapping_path),
            "norms_lock_path": relative_to_root(output_path),
            "norms_lock_sha256": lock_sha256,
            # Stage 03 defines the canonical data snapshot artifact.
            "data_snapshot_id": f"norms_sha256:{lock_sha256}",
        },
    )
    meta_payload = {
        "provenance": provenance,
        "source": {
            "dataset": NORM_DATASET,
            "table": NORM_TABLE,
            "scope": NORM_SCOPE,
            "db_path": relative_to_root(db_path),
            "n_respondents": lock_payload["n_respondents"],
            "mini_ipip_mapping": {
                "path": relative_to_root(mini_ipip_mapping_path),
                "sha256": mini_ipip_mapping_sha256,
            },
        },
        "computed": {
            domain: {
                "mean": float(computed[domain]["mean"]),
                "sd": float(computed[domain]["sd"]),
                "n": int(computed[domain]["n"]),
            }
            for domain in DOMAINS
        },
        "computed_mini_ipip": {
            domain: {
                "mean": float(mini_ipip_computed[domain]["mean"]),
                "sd": float(mini_ipip_computed[domain]["sd"]),
                "n": int(mini_ipip_computed[domain]["n"]),
            }
            for domain in DOMAINS
        },
        "expected": expected_norms,
        "expected_mini_ipip": expected_mini_ipip_norms,
        "abs_diff": diffs,
        "abs_diff_mini_ipip": mini_ipip_diffs,
        "max_abs_diff": max_abs_diff,
        "tolerance": float(args.tolerance),
        "check_passed": bool(check_passed),
        "lock_file": relative_to_root(output_path),
    }
    if write_meta:
        _write_json(meta_path, meta_payload)
        log.info("Wrote norm metadata: %s", meta_path)

    if args.check and not check_passed:
        log.error(
            "Norm drift detected: max_abs_diff=%.6g exceeds tolerance=%.6g",
            max_abs_diff,
            args.tolerance,
        )
        return 1

    log.info("Norm stage complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
