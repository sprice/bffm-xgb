"""Reusable provenance/hash verification helpers for pipeline stages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .item_info import (
    extract_training_data_sha256,
    extract_training_split_signature,
    file_sha256,
    load_training_report,
)


def _normalize_sha256(value: Any, *, field: str) -> str:
    """Normalize and validate a SHA-256 hex string."""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string SHA-256 value.")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError(f"{field} must be a 64-char hex SHA-256 value.")
    return normalized


def build_split_signature(
    *,
    train_sha256: str,
    val_sha256: str,
    test_sha256: str,
) -> str:
    """Compute deterministic split signature from train/val/test hashes."""
    train_norm = _normalize_sha256(train_sha256, field="train_sha256")
    val_norm = _normalize_sha256(val_sha256, field="val_sha256")
    test_norm = _normalize_sha256(test_sha256, field="test_sha256")
    payload = (
        f"train={train_norm}\n"
        f"val={val_norm}\n"
        f"test={test_norm}\n"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_split_hashes_from_metadata(
    split_metadata_path: Path,
) -> tuple[dict[str, str], str | None]:
    """Read expected split hashes/signature from stage-04 split metadata."""
    with open(split_metadata_path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Split metadata must be a JSON object: {split_metadata_path}")

    outputs = payload.get("outputs", {})
    if not isinstance(outputs, dict):
        outputs = {}

    def _read_hash(name: str) -> str | None:
        explicit = payload.get(f"{name}_sha256")
        if isinstance(explicit, str) and explicit.strip():
            return _normalize_sha256(explicit, field=f"{name}_sha256")
        output_obj = outputs.get(name, {})
        if isinstance(output_obj, dict):
            value = output_obj.get("sha256")
            if isinstance(value, str) and value.strip():
                return _normalize_sha256(value, field=f"outputs.{name}.sha256")
        return None

    expected = {
        "train_sha256": _read_hash("train"),
        "val_sha256": _read_hash("val"),
        "test_sha256": _read_hash("test"),
    }
    missing = [key for key, value in expected.items() if value is None]
    if missing:
        raise ValueError(
            "Split metadata missing required hashes: "
            + ", ".join(missing)
            + f" ({split_metadata_path})"
        )

    split_signature_raw = payload.get("split_signature")
    split_signature: str | None
    if isinstance(split_signature_raw, str) and split_signature_raw.strip():
        split_signature = _normalize_sha256(
            split_signature_raw,
            field="split_signature",
        )
    else:
        split_signature = None

    return {
        "train_sha256": str(expected["train_sha256"]),
        "val_sha256": str(expected["val_sha256"]),
        "test_sha256": str(expected["test_sha256"]),
    }, split_signature


def verify_split_metadata_hash_lock(
    split_metadata_path: Path,
    *,
    train_sha256: str,
    val_sha256: str,
    test_sha256: str | None,
) -> str:
    """Fail closed when stage-04 split metadata does not match current parquet files."""
    expected_hashes, expected_signature = load_split_hashes_from_metadata(split_metadata_path)
    if test_sha256 is None:
        raise ValueError(
            "split_metadata.json exists but test.parquet hash is unavailable; "
            "cannot verify split identity."
        )

    actual_hashes = {
        "train_sha256": _normalize_sha256(train_sha256, field="train_sha256"),
        "val_sha256": _normalize_sha256(val_sha256, field="val_sha256"),
        "test_sha256": _normalize_sha256(test_sha256, field="test_sha256"),
    }

    for key, actual in actual_hashes.items():
        expected = expected_hashes[key]
        if expected != actual:
            raise ValueError(
                "Split provenance mismatch for "
                f"{key}: expected={expected}, actual={actual}, metadata={split_metadata_path}"
            )

    computed_signature = build_split_signature(
        train_sha256=actual_hashes["train_sha256"],
        val_sha256=actual_hashes["val_sha256"],
        test_sha256=actual_hashes["test_sha256"],
    )
    if expected_signature is not None and expected_signature != computed_signature:
        raise ValueError(
            "Split signature mismatch: "
            f"expected={expected_signature}, actual={computed_signature}, "
            f"metadata={split_metadata_path}"
        )

    return computed_signature


def verify_model_data_split_provenance(
    *,
    model_dir: Path,
    data_dir: Path,
) -> tuple[str | None, str | None]:
    """Fail closed when model-bundle split fingerprints mismatch selected data dir."""
    report, report_path = load_training_report(model_dir)
    expected_test_sha = extract_training_data_sha256(report, "test_sha256")
    expected_split_signature = extract_training_split_signature(report)

    actual_test_sha: str | None = None
    if expected_test_sha is not None:
        test_path = data_dir / "test.parquet"
        if not test_path.exists():
            raise FileNotFoundError(
                "Training report requires test split hash verification but test.parquet "
                f"is missing: {test_path}"
            )
        actual_test_sha = file_sha256(test_path).lower()
        if actual_test_sha != expected_test_sha.lower():
            raise ValueError(
                "Model/data provenance mismatch for test split: "
                f"expected_sha256={expected_test_sha}, actual_sha256={actual_test_sha}, "
                f"report={report_path}, data_dir={data_dir}"
            )

    actual_split_signature: str | None = None
    if expected_split_signature is not None:
        train_path = data_dir / "train.parquet"
        val_path = data_dir / "val.parquet"
        test_path = data_dir / "test.parquet"
        missing = [str(p) for p in (train_path, val_path, test_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Training report requires split signature verification, but split files are missing: "
                + ", ".join(missing)
            )
        actual_split_signature = build_split_signature(
            train_sha256=file_sha256(train_path),
            val_sha256=file_sha256(val_path),
            test_sha256=file_sha256(test_path),
        )
        if actual_split_signature != expected_split_signature.lower():
            raise ValueError(
                "Model/data split signature mismatch: "
                f"expected={expected_split_signature}, actual={actual_split_signature}, "
                f"report={report_path}, data_dir={data_dir}"
            )

    return actual_test_sha, actual_split_signature

