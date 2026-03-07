#!/usr/bin/env python3
"""Build strict cross-variant research summary from model + artifact bundles."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from lib.constants import VARIANTS
from lib.provenance import build_provenance, file_sha256, relative_to_root, sanitize_paths


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
DEFAULT_ARTIFACTS_VARIANTS_DIR = Path("artifacts/variants")
DEFAULT_OUTPUT_PATH = Path("artifacts/research_summary.json")
DEFAULT_NORMS_PATH = Path("artifacts/ipip_bffm_norms.json")
DEFAULT_TUNED_PARAMS_PATH = Path("artifacts/tuned_params.json")
DEFAULT_MINI_IPIP_MAPPING_PATH = Path("artifacts/mini_ipip_mapping.json")


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PACKAGE_ROOT / p


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML object at {path}")
    return payload


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _normalize_sha256(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if len(v) == 64 and all(c in "0123456789abcdef" for c in v):
        return v
    return None


def _extract_overall(metrics: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(metrics, dict):
        return {}
    overall = metrics.get("overall", {})
    if not isinstance(overall, dict):
        return {}
    result: dict[str, float] = {}
    for key in ("pearson_r", "mae", "coverage_90", "rmse", "within_5_pct", "within_10_pct"):
        value = overall.get(key)
        if isinstance(value, (int, float)):
            result[key] = float(value)
    return result


def _variant_paths(
    *,
    variant: str,
    meta: dict[str, str],
    artifacts_variants_dir: Path,
) -> dict[str, Path]:
    config_path = _resolve(meta["config"])
    model_dir = _resolve(meta["model_dir"])
    artifact_dir = artifacts_variants_dir / variant

    config_payload = _load_yaml(config_path)
    data_dir_raw = config_payload.get("data_dir")
    if not isinstance(data_dir_raw, str) or not data_dir_raw.strip():
        if meta["default_data_regime"] == "ext_est_opn":
            data_dir_raw = "data/processed/ext_est_opn"
        else:
            data_dir_raw = "data/processed/ext_est"
    data_dir = _resolve(data_dir_raw)

    return {
        "config": config_path,
        "model_dir": model_dir,
        "data_dir": data_dir,
        "artifacts_dir": artifact_dir,
        "training_report": model_dir / "training_report.json",
        "validation_results": artifact_dir / "validation_results.json",
        "baseline_comparison_results": artifact_dir / "baseline_comparison_results.json",
        "baseline_comparison_per_domain_csv": artifact_dir / "baseline_comparison_per_domain.csv",
        "ml_vs_averaging_comparison": artifact_dir / "ml_vs_averaging_comparison.json",
        "simulation_results": artifact_dir / "simulation_results.json",
        "split_metadata": data_dir / "split_metadata.json",
        "item_info": data_dir / "item_info.json",
    }


def _collect_variant_summary(
    *,
    variant: str,
    meta: dict[str, str],
    artifacts_variants_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = _variant_paths(
        variant=variant,
        meta=meta,
        artifacts_variants_dir=artifacts_variants_dir,
    )

    required = [
        "training_report",
        "validation_results",
        "baseline_comparison_results",
        "baseline_comparison_per_domain_csv",
        "ml_vs_averaging_comparison",
        "simulation_results",
        "split_metadata",
        "item_info",
    ]
    present = {key: paths[key].exists() for key in required}
    errors: list[str] = [f"missing:{key}" for key, ok in present.items() if not ok]

    loaded: dict[str, Any] = {}
    for key in ("training_report", "validation_results", "baseline_comparison_results", "ml_vs_averaging_comparison", "simulation_results", "split_metadata", "item_info"):
        if present.get(key, False):
            try:
                loaded[key] = _load_json(paths[key])
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                errors.append(f"parse:{key}:{type(exc).__name__}:{exc}")
    if present.get("baseline_comparison_per_domain_csv", False):
        try:
            loaded["baseline_comparison_per_domain_csv"] = _load_csv_rows(
                paths["baseline_comparison_per_domain_csv"]
            )
        except (OSError, csv.Error) as exc:
            errors.append(f"parse:baseline_comparison_per_domain_csv:{type(exc).__name__}:{exc}")

    # Provenance consistency checks (split/test hash must agree across pipeline artifacts).
    # NOTE: split_metadata.json is excluded because it is a prepare-stage build artifact
    # that can become stale when data/ is overwritten by rsync (remote-push).  The
    # authoritative provenance chain is training_report -> validation -> baselines -> simulation.
    split_candidates = [
        _normalize_sha256(loaded.get("training_report", {}).get("data", {}).get("split_signature")),
        _normalize_sha256(loaded.get("validation_results", {}).get("provenance", {}).get("split_signature")),
        _normalize_sha256(loaded.get("baseline_comparison_results", {}).get("provenance", {}).get("split_signature")),
        _normalize_sha256(loaded.get("simulation_results", {}).get("provenance", {}).get("split_signature")),
    ]
    split_values = sorted({v for v in split_candidates if v is not None})
    if len(split_values) > 1:
        errors.append("provenance:split_signature_mismatch")

    test_sha_candidates = [
        _normalize_sha256(loaded.get("training_report", {}).get("data", {}).get("test_sha256")),
        _normalize_sha256(loaded.get("validation_results", {}).get("provenance", {}).get("test_sha256")),
        _normalize_sha256(loaded.get("baseline_comparison_results", {}).get("provenance", {}).get("test_sha256")),
        _normalize_sha256(loaded.get("simulation_results", {}).get("provenance", {}).get("test_sha256")),
    ]
    test_sha_values = sorted({v for v in test_sha_candidates if v is not None})
    if len(test_sha_values) > 1:
        errors.append("provenance:test_sha256_mismatch")

    config_payload = _load_yaml(paths["config"])
    data_regime = "ext_est_opn" if str(paths["data_dir"]).endswith("ext_est_opn") else "ext_est"

    training = loaded.get("training_report", {})
    validation = loaded.get("validation_results", {})
    baselines = loaded.get("baseline_comparison_results", {})
    simulation = loaded.get("simulation_results", {})

    summary = {
        "variant": variant,
        "description": config_payload.get("description"),
        "data_regime": data_regime,
        "paths": {k: relative_to_root(v) for k, v in paths.items()},
        "present": present,
        "training": {
            "provenance": {
                "git_hash": training.get("provenance", {}).get("git_hash"),
                "data_snapshot_id": training.get("provenance", {}).get("data_snapshot_id"),
                "split_signature": training.get("data", {}).get("split_signature"),
                "train_sha256": training.get("data", {}).get("train_sha256"),
                "val_sha256": training.get("data", {}).get("val_sha256"),
                "test_sha256": training.get("data", {}).get("test_sha256"),
                "item_info_sha256": training.get("data", {}).get("item_info_sha256"),
                "hyperparameters_sha256": training.get("data", {}).get("hyperparameters_sha256"),
            },
            "validation_full_50": _extract_overall(training.get("validation_metrics", {})),
            "validation_sparse_20": _extract_overall(training.get("validation_metrics_sparse_20", {})),
        },
        "validation": {
            "provenance": {
                "git_hash": validation.get("provenance", {}).get("git_hash"),
                "split_signature": validation.get("provenance", {}).get("split_signature"),
                "test_sha256": validation.get("provenance", {}).get("test_sha256"),
            },
            "full_50": _extract_overall(validation.get("metrics", {})),
            "sparse_20": _extract_overall(validation.get("sparse_20", {}).get("metrics", {})),
        },
        "baselines": {
            "provenance": {
                "git_hash": baselines.get("provenance", {}).get("git_hash"),
                "split_signature": baselines.get("provenance", {}).get("split_signature"),
                "test_sha256": baselines.get("provenance", {}).get("test_sha256"),
            },
            "k20": {
                "domain_balanced": baselines.get("overall", {}).get("20", {}).get("domain_balanced", {}),
                "domain_constrained_adaptive": baselines.get("overall", {}).get("20", {}).get("domain_constrained_adaptive", {}),
                "mini_ipip": baselines.get("overall", {}).get("20", {}).get("mini_ipip", {}),
                "adaptive_topk": baselines.get("overall", {}).get("20", {}).get("adaptive_topk", {}),
            },
        },
        "simulation": {
            "provenance": {
                "git_hash": simulation.get("provenance", {}).get("git_hash"),
                "split_signature": simulation.get("provenance", {}).get("split_signature"),
                "test_sha256": simulation.get("provenance", {}).get("test_sha256"),
                "calibration_source": simulation.get("provenance", {}).get("calibration_source"),
            },
            "overall": simulation.get("analysis", {}).get("overall_metrics", {}),
            "domain_metrics": simulation.get("analysis", {}).get("domain_metrics", {}),
        },
        # Variant-scoped NOTES inputs so NOTES.md can include full research data
        # across all training runs, not just the reference run.
        "notes_inputs": sanitize_paths({
            "training_report": loaded.get("training_report"),
            "validation_results": loaded.get("validation_results"),
            "baseline_comparison_results": loaded.get("baseline_comparison_results"),
            "baseline_comparison_per_domain_rows": loaded.get("baseline_comparison_per_domain_csv"),
            "ml_vs_averaging_comparison": loaded.get("ml_vs_averaging_comparison"),
            "simulation_results": loaded.get("simulation_results"),
            "split_metadata": loaded.get("split_metadata"),
            "item_info": loaded.get("item_info"),
        }),
        "errors": errors,
    }

    summary["status"] = {
        "complete": all(present.values()) and not errors,
        "split_signature": split_values[0] if len(split_values) == 1 else None,
    }

    return summary, loaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build strict cross-variant research summary from model/artifact bundles."
    )
    parser.add_argument(
        "--artifacts-variants-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_VARIANTS_DIR,
        help="Directory containing per-variant artifact folders (default: artifacts/variants)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON path (default: artifacts/research_summary.json)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any variant is incomplete, has parse errors, or has provenance mismatch.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    artifacts_variants_dir = _resolve(args.artifacts_variants_dir)
    output_path = _resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    variants_summary: dict[str, Any] = {}
    loaded_by_variant: dict[str, dict[str, Any]] = {}
    incomplete: list[str] = []

    for variant, meta in VARIANTS.items():
        summary, loaded = _collect_variant_summary(
            variant=variant,
            meta=meta,
            artifacts_variants_dir=artifacts_variants_dir,
        )
        variants_summary[variant] = summary
        loaded_by_variant[variant] = loaded
        if not summary.get("status", {}).get("complete", False):
            incomplete.append(variant)

    # Canonical reference notes bundle: all NOTES data sections read from here.
    reference_loaded = loaded_by_variant.get("reference", {})
    reference_notes_inputs: dict[str, Any] = sanitize_paths({
        "training_report": reference_loaded.get("training_report"),
        "validation_results": reference_loaded.get("validation_results"),
        "baseline_comparison_results": reference_loaded.get("baseline_comparison_results"),
        "baseline_comparison_per_domain_rows": reference_loaded.get("baseline_comparison_per_domain_csv"),
        "ml_vs_averaging_comparison": reference_loaded.get("ml_vs_averaging_comparison"),
        "simulation_results": reference_loaded.get("simulation_results"),
        "split_metadata": reference_loaded.get("split_metadata"),
        "item_info": reference_loaded.get("item_info"),
    })

    # Global reference artifacts used by NOTES.
    global_paths = {
        "norms": _resolve(DEFAULT_NORMS_PATH),
        "tuned_params": _resolve(DEFAULT_TUNED_PARAMS_PATH),
        "mini_ipip_mapping": _resolve(DEFAULT_MINI_IPIP_MAPPING_PATH),
    }
    for key, path in global_paths.items():
        if not path.exists():
            reference_notes_inputs[key] = {"__error__": f"missing:{path}"}
            continue
        try:
            reference_notes_inputs[key] = _load_json(path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            reference_notes_inputs[key] = {"__error__": f"parse:{key}:{type(exc).__name__}:{exc}"}

    # Build top-level provenance with input artifact checksums
    input_artifacts: dict[str, Any] = {}
    norms_lock_path = _resolve(DEFAULT_NORMS_PATH)
    if norms_lock_path.exists():
        input_artifacts["norms_lock_sha256"] = file_sha256(norms_lock_path)
    mini_ipip_path = _resolve(DEFAULT_MINI_IPIP_MAPPING_PATH)
    if mini_ipip_path.exists():
        input_artifacts["mini_ipip_mapping_sha256"] = file_sha256(mini_ipip_path)

    prov = build_provenance(
        Path(__file__).name,
        extra={
            "input_artifacts": input_artifacts,
            "n_variants": len(variants_summary),
            "variants_included": sorted(variants_summary.keys()),
        },
    )

    payload = {
        "schema_version": 2,
        "provenance": prov,
        "variants": variants_summary,
        "reference_notes_inputs": reference_notes_inputs,
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Wrote research summary: %s", output_path)

    if incomplete:
        log.warning("Incomplete variants in research summary: %s", ", ".join(incomplete))
        if args.strict:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
