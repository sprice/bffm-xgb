"""Unit tests for scripts/check_provenance.py."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from scripts.check_provenance import (
    ProvenanceChecker,
    check_norms_lock,
    check_norms_meta,
    check_output_bundle,
    check_research_summary,
    main,
)


def test_checker_pass_fail_skip_counts() -> None:
    """Verify ProvenanceChecker.print_summary() returns correct failure count."""
    checker = ProvenanceChecker()
    checker.passed("a", "ok")
    checker.failed("b", "bad")
    checker.skipped("c", "skipped")
    checker.passed("d")

    n_fail = checker.print_summary()
    assert n_fail == 1

    n_pass = sum(1 for s, _, _ in checker.results if s == "PASS")
    n_skip = sum(1 for s, _, _ in checker.results if s == "SKIP")
    assert n_pass == 2
    assert n_skip == 1


def test_check_norms_lock_missing(tmp_path) -> None:
    """Verify failure on missing norms lock file."""
    checker = ProvenanceChecker()
    with patch("scripts.check_provenance.PACKAGE_ROOT", tmp_path):
        (tmp_path / "artifacts").mkdir(parents=True)
        result = check_norms_lock(checker)

    assert result is None
    assert checker.results[0][0] == "FAIL"


def test_check_norms_lock_valid(tmp_path) -> None:
    """Verify pass and correct SHA return on valid norms lock."""
    checker = ProvenanceChecker()
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    norms_path = artifacts_dir / "ipip_bffm_norms.json"
    payload = {"schema_version": 1, "norms": {"ext": {}}, "n_respondents": 100}
    norms_path.write_text(json.dumps(payload), encoding="utf-8")

    expected_sha = hashlib.sha256(norms_path.read_bytes()).hexdigest()

    with patch("scripts.check_provenance.PACKAGE_ROOT", tmp_path):
        result = check_norms_lock(checker)

    assert result == expected_sha
    assert checker.results[0][0] == "PASS"


def test_check_norms_meta_sha_mismatch(tmp_path) -> None:
    """Verify failure on hash mismatch between norms meta and lock."""
    checker = ProvenanceChecker()
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)

    meta_payload = {
        "provenance": {
            "norms_lock_sha256": "wrong_sha_value",
            "data_snapshot_id": "norms_sha256:wrong_sha_value",
        }
    }
    (artifacts_dir / "ipip_bffm_norms.meta.json").write_text(
        json.dumps(meta_payload), encoding="utf-8"
    )

    with patch("scripts.check_provenance.PACKAGE_ROOT", tmp_path):
        check_norms_meta(checker, norms_sha="correct_sha_value")

    assert checker.results[0][0] == "FAIL"
    assert "mismatch" in checker.results[0][2]


def test_check_output_bundle_snapshot_mismatch(tmp_path) -> None:
    """Verify failure when data_snapshot_id doesn't match norms."""
    checker = ProvenanceChecker()
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True)

    # Write config.json so the check doesn't skip
    (output_dir / "config.json").write_text('{"test": true}', encoding="utf-8")

    # Write provenance.json with wrong snapshot
    prov_doc = {
        "export": {
            "script": "11_export_onnx.py",
            "data_snapshot_id": "norms_sha256:wrong_hash",
        },
        "training": {"provenance": {}},
        "artifacts": {},
    }
    with open(output_dir / "provenance.json", "w") as f:
        json.dump(prov_doc, f)

    with patch("scripts.check_provenance.PACKAGE_ROOT", tmp_path):
        check_output_bundle(checker, norms_sha="correct_hash")

    assert checker.results[0][0] == "FAIL"
    assert "mismatch" in checker.results[0][2]


def test_strict_exits_nonzero_on_failure(tmp_path) -> None:
    """Verify main() returns 1 with --strict and a failure."""
    # Create artifacts dir with invalid norms lock
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    # No norms lock file → will FAIL

    with (
        patch("scripts.check_provenance.PACKAGE_ROOT", tmp_path),
        patch("sys.argv", ["check_provenance.py", "--strict"]),
    ):
        result = main()

    assert result == 1


def test_full_exits_nonzero_on_skip(tmp_path) -> None:
    """Verify main() returns 1 with --full and a skip (no failures needed)."""
    # Create valid norms lock so it PASSes, but no other artifacts → SKIPs
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    norms_path = artifacts_dir / "ipip_bffm_norms.json"
    payload = {"schema_version": 1, "norms": {"ext": {}}, "n_respondents": 100}
    norms_path.write_text(json.dumps(payload), encoding="utf-8")

    # Create valid meta sidecar
    norms_sha = hashlib.sha256(norms_path.read_bytes()).hexdigest()
    meta_payload = {
        "provenance": {
            "norms_lock_sha256": norms_sha,
            "data_snapshot_id": f"norms_sha256:{norms_sha}",
        }
    }
    (artifacts_dir / "ipip_bffm_norms.meta.json").write_text(
        json.dumps(meta_payload), encoding="utf-8"
    )

    with (
        patch("scripts.check_provenance.PACKAGE_ROOT", tmp_path),
        patch("sys.argv", ["check_provenance.py", "--full"]),
    ):
        result = main()

    # Should exit 1 because research_summary, output/, figures/ all SKIP
    assert result == 1
