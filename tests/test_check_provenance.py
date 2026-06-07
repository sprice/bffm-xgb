"""Unit tests for scripts/check_provenance.py."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from scripts.check_provenance import (
    ProvenanceChecker,
    _commit_relationship,
    _hyperparameters_drift,
    _model_freshness,
    _release_fresh,
    check_figures_manifest,
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


def _write_variant_bundle(
    tmp_path: Path,
    *,
    variant: str = "reference",
    git_hash: str = "abcdef0",
    config_git_hash: str | None = None,
    data_snapshot_id: str = "x",
    payload_provenance_hash: str | None = None,
    hyperparameters: dict | None = None,
    item_info_sha256: str | None = None,
    item_info_path: str | None = None,
    write_model: bool = False,
) -> Path:
    """Write a valid output/<variant>/ bundle (matching config checksum).

    Pass ``hyperparameters`` to embed a self-describing ``config_locked_params`` lock
    (training.config.hyperparameters + hyperparameters_source) so the model-freshness
    HP re-verification can run end-to-end through check_output_bundle. Pass
    ``item_info_sha256`` + ``item_info_path`` to embed the item_info lock so the
    item_info-lock check runs. Pass ``write_model=True`` to write a model.onnx whose
    SHA matches the recorded lock (otherwise model.onnx is absent, as on a clone).
    """
    vdir = tmp_path / "output" / variant
    vdir.mkdir(parents=True)
    config_path = vdir / "config.json"
    config_doc = {
        "provenance": {"git_hash": config_git_hash or git_hash},
        "model_file": "model.onnx",
    }
    config_path.write_text(json.dumps(config_doc), encoding="utf-8")
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    training: dict = {"provenance": {"git_hash": git_hash}}
    config_block: dict = {}
    if payload_provenance_hash is not None:
        # The nested tune payload hash legitimately differs from the export/train
        # hashes (as in the real committed bundle) and must NOT trip the
        # intra-bundle agreement check.
        config_block["hyperparameters_source"] = {
            "payload_provenance": {"git_hash": payload_provenance_hash}
        }
    if hyperparameters is not None:
        config_block["hyperparameters"] = hyperparameters
        config_block.setdefault("hyperparameters_source", {}).update(
            {"mode": "config_locked_params", "path": "artifacts/tuned_params.json"}
        )
    if config_block:
        training["config"] = config_block
    if item_info_sha256 is not None and item_info_path is not None:
        training["data"] = {
            "item_info_sha256": item_info_sha256,
            "item_info_path": item_info_path,
        }
    model_sha = "deadbeef"  # absent-on-clone default (no model.onnx written)
    if write_model:
        model_bytes = b"onnx-model-bytes"
        (vdir / "model.onnx").write_bytes(model_bytes)
        model_sha = hashlib.sha256(model_bytes).hexdigest()
    prov_doc = {
        "export": {"git_hash": git_hash, "data_snapshot_id": data_snapshot_id},
        "training": training,
        "artifacts": {"config_json_sha256": config_sha, "model_onnx_sha256": model_sha},
    }
    (vdir / "provenance.json").write_text(json.dumps(prov_doc), encoding="utf-8")
    return vdir


def _write_research_summary(tmp_path: Path, *, git_hash: str, norms_sha: str) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "provenance": {
            "git_hash": git_hash,
            "input_artifacts": {"norms_lock_sha256": norms_sha},
        },
        "variants": {},
    }
    (artifacts_dir / "research_summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_check_research_summary_reference_only_ignores_incomplete_ablations(tmp_path) -> None:
    """--reference-only scopes the completeness gate to the reference variant, so an
    incomplete/absent ablation does not fail a single-variant run; without the flag it FAILs."""
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "provenance": {"git_hash": "abc", "input_artifacts": {"norms_lock_sha256": "x"}},
        "variants": {
            "reference": {"status": {"complete": True}},
            "ablation_none": {"status": {"complete": False}},
        },
    }
    (artifacts_dir / "research_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    # Without the flag, the incomplete ablation FAILs the completeness gate.
    # (norms_sha=None skips the norms-mismatch branch so we reach the variants gate.)
    full = ProvenanceChecker()
    with patch("scripts.check_provenance.PACKAGE_ROOT", tmp_path):
        check_research_summary(full, norms_sha=None, head_hash="abc", reference_only=False)
    assert any(r[0] == "FAIL" for r in full.results)

    # With --reference-only, only the (complete) reference variant is required -> PASS.
    ref_only = ProvenanceChecker()
    with patch("scripts.check_provenance.PACKAGE_ROOT", tmp_path):
        check_research_summary(ref_only, norms_sha=None, head_hash="abc", reference_only=True)
    assert all(r[0] != "FAIL" for r in ref_only.results)
    assert any(r[0] == "PASS" for r in ref_only.results)


def test_check_research_summary_reference_only_summary_checked_without_flag_fails(tmp_path) -> None:
    """A summary built --reference-only (provenance.reference_only=True, single variant) must FAIL
    a default provenance-check (absent ablations are not verifiable), and pass only with --reference-only."""
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "provenance": {
            "git_hash": "abc",
            "reference_only": True,
            "input_artifacts": {"norms_lock_sha256": "x"},
        },
        "variants": {"reference": {"status": {"complete": True}}},
    }
    (artifacts_dir / "research_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    # Default check on a reference-only summary -> FAIL (must use --reference-only).
    default = ProvenanceChecker()
    with patch("scripts.check_provenance.PACKAGE_ROOT", tmp_path):
        check_research_summary(default, norms_sha=None, head_hash="abc", reference_only=False)
    assert any(r[0] == "FAIL" for r in default.results)

    # With --reference-only it is accepted.
    ref_only = ProvenanceChecker()
    with patch("scripts.check_provenance.PACKAGE_ROOT", tmp_path):
        check_research_summary(ref_only, norms_sha=None, head_hash="abc", reference_only=True)
    assert all(r[0] != "FAIL" for r in ref_only.results)
    assert any(r[0] == "PASS" for r in ref_only.results)


def test_check_research_summary_norms_mismatch_at_head_fails(tmp_path) -> None:
    """A5.2: a norms mismatch on a summary AT HEAD is a hard FAIL."""
    _write_research_summary(tmp_path, git_hash="headhash", norms_sha="wronghash")
    checker = ProvenanceChecker()
    with patch("scripts.check_provenance.PACKAGE_ROOT", tmp_path):
        check_research_summary(checker, norms_sha="correcthash", head_hash="headhash")
    assert any(r[0] == "FAIL" for r in checker.results)


def test_check_research_summary_norms_mismatch_stale_warns(tmp_path) -> None:
    """A5.2: a norms mismatch on a NON-fresh summary (predates HEAD) is a WARN, not a FAIL.
    (_release_fresh is forced False here to simulate a genuinely stale bundle; in a non-git
    tmp_path it would otherwise degrade to git-unverifiable=fresh.)"""
    _write_research_summary(tmp_path, git_hash="oldhash", norms_sha="wronghash")
    checker = ProvenanceChecker()
    with (
        patch("scripts.check_provenance.PACKAGE_ROOT", tmp_path),
        patch("scripts.check_provenance._release_fresh", return_value=(False, "stale (predates HEAD)")),
    ):
        check_research_summary(checker, norms_sha="correcthash", head_hash="newhead")
    assert any(r[0] == "WARN" for r in checker.results)
    assert all(r[0] != "FAIL" for r in checker.results)


def test_check_output_bundle_cross_variant_warns_then_strict_fails(tmp_path) -> None:
    """A5.2: variants exported from different commits WARN by default, FAIL under strict_head."""
    _write_variant_bundle(tmp_path, variant="reference", git_hash="hashAAAA")
    _write_variant_bundle(tmp_path, variant="ablation_none", git_hash="hashBBBB")
    checker = ProvenanceChecker()
    with patch("scripts.check_provenance.PACKAGE_ROOT", tmp_path):
        check_output_bundle(checker, norms_sha=None, head_hash="hashAAAA", strict_head=False)
    assert any(r[0] == "WARN" and "cross-variant" in r[1] for r in checker.results)
    assert checker.print_summary() == 0

    strict = ProvenanceChecker()
    with patch("scripts.check_provenance.PACKAGE_ROOT", tmp_path):
        check_output_bundle(strict, norms_sha=None, head_hash="hashAAAA", strict_head=True)
    assert any(r[0] == "FAIL" and "cross-variant" in r[1] for r in strict.results)


def test_check_output_bundle_excludes_nested_payload_hash(tmp_path) -> None:
    """A5.2: the nested tune payload_provenance git_hash differing must NOT trip the
    intra-bundle agreement check (it legitimately differs in the real bundle)."""
    _write_variant_bundle(tmp_path, git_hash="exportHASH", payload_provenance_hash="914d82eDIFF")
    checker = ProvenanceChecker()
    with patch("scripts.check_provenance.PACKAGE_ROOT", tmp_path):
        check_output_bundle(checker, norms_sha=None, head_hash="exportHASH", strict_head=True)
    assert not any(r[0] == "FAIL" and "agreement" in r[1] for r in checker.results)


def test_check_figures_manifest_detects_output_tamper(tmp_path) -> None:
    """A5.5: a figure output whose sha256 no longer matches the manifest FAILs;
    a matching one passes (and absent files / no sha256 key degrade to skip)."""
    figs = tmp_path / "figures"
    figs.mkdir(parents=True)
    png = figs / "fig1_test.png"
    png.write_bytes(b"real-figure-bytes")
    good_sha = hashlib.sha256(png.read_bytes()).hexdigest()
    manifest = {
        "provenance": {"git_hash": "h"},
        "source_artifacts": {},
        "figures": [{"filename": "fig1_test", "formats": ["png"], "sha256": {"png": good_sha}}],
    }
    (figs / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    matching = ProvenanceChecker()
    with patch("scripts.check_provenance.PACKAGE_ROOT", tmp_path):
        check_figures_manifest(matching)
    assert all(r[0] != "FAIL" for r in matching.results)

    png.write_bytes(b"tampered-bytes")
    tampered = ProvenanceChecker()
    with patch("scripts.check_provenance.PACKAGE_ROOT", tmp_path):
        check_figures_manifest(tampered)
    assert any(r[0] == "FAIL" and "SHA-256 mismatch" in r[2] for r in tampered.results)


def test_check_output_bundle_snapshot_mismatch(tmp_path) -> None:
    """FAIL when data_snapshot_id mismatches AND the bundle is at HEAD."""
    checker = ProvenanceChecker()
    _write_variant_bundle(
        tmp_path, git_hash="headhash", data_snapshot_id="norms_sha256:wrong_hash"
    )
    with patch("scripts.check_provenance.PACKAGE_ROOT", tmp_path):
        check_output_bundle(
            checker, norms_sha="correct_hash", head_hash="headhash", strict_head=False
        )
    statuses = [r[0] for r in checker.results]
    details = " ".join(r[2] for r in checker.results)
    assert "FAIL" in statuses
    assert "mismatch" in details


def test_check_output_bundle_snapshot_stale_warns_not_fails(tmp_path) -> None:
    """A snapshot mismatch on a bundle that predates HEAD is a WARN, not a FAIL."""
    checker = ProvenanceChecker()
    _write_variant_bundle(
        tmp_path, git_hash="oldhash", data_snapshot_id="norms_sha256:wrong_hash"
    )
    with (
        patch("scripts.check_provenance.PACKAGE_ROOT", tmp_path),
        patch("scripts.check_provenance._release_fresh", return_value=(False, "stale (predates HEAD)")),
    ):
        check_output_bundle(
            checker, norms_sha="correct_hash", head_hash="newhead", strict_head=False
        )
    assert all(r[0] != "FAIL" for r in checker.results)
    assert any(r[0] == "WARN" for r in checker.results)


def test_check_output_bundle_skips_when_no_variants(tmp_path) -> None:
    """Empty output/ dir SKIPs gracefully (no silent pass on the wrong path)."""
    checker = ProvenanceChecker()
    (tmp_path / "output").mkdir(parents=True)
    with patch("scripts.check_provenance.PACKAGE_ROOT", tmp_path):
        check_output_bundle(checker, norms_sha=None, head_hash="h")
    assert checker.results[0][0] == "SKIP"


def test_check_output_bundle_head_stale_warns_then_strict_fails(tmp_path) -> None:
    """A bundle with model-producing code drift is WARN by default and FAIL under
    strict_head. (_model_freshness forced to "code_drift" to simulate the drift; a
    non-git tmp_path would otherwise degrade to git-unverifiable="fresh".)"""
    checker = ProvenanceChecker()
    _write_variant_bundle(tmp_path, git_hash="bundlehash")
    with (
        patch("scripts.check_provenance.PACKAGE_ROOT", tmp_path),
        patch(
            "scripts.check_provenance._model_freshness",
            return_value=("code_drift", "model-producing code changed"),
        ),
    ):
        check_output_bundle(checker, norms_sha=None, head_hash="otherhead", strict_head=False)
    assert any(r[0] == "WARN" for r in checker.results)
    assert checker.print_summary() == 0  # WARN is not a failure

    strict = ProvenanceChecker()
    with (
        patch("scripts.check_provenance.PACKAGE_ROOT", tmp_path),
        patch(
            "scripts.check_provenance._model_freshness",
            return_value=("code_drift", "model-producing code changed"),
        ),
    ):
        check_output_bundle(strict, norms_sha=None, head_hash="otherhead", strict_head=True)
    assert any(r[0] == "FAIL" for r in strict.results)


def test_check_output_bundle_input_drift_fails_even_without_strict_head(tmp_path) -> None:
    """A provable model-INPUT change (locked hyperparameters moved) is a hard FAIL on the
    HEAD-freshness line even under plain (non-strict-head) verification."""
    checker = ProvenanceChecker()
    _write_variant_bundle(tmp_path, git_hash="bundlehash")
    with (
        patch("scripts.check_provenance.PACKAGE_ROOT", tmp_path),
        patch(
            "scripts.check_provenance._model_freshness",
            return_value=("input_drift", "locked hyperparameters changed"),
        ),
    ):
        check_output_bundle(checker, norms_sha=None, head_hash="otherhead", strict_head=False)
    assert any(
        r[0] == "FAIL" and "HEAD freshness" in r[1] and "hyperparameters" in r[2]
        for r in checker.results
    )


def test_check_output_bundle_model_present_reports_model_verified(tmp_path, monkeypatch) -> None:
    """When model.onnx is present and its SHA matches, the bundle line says the model
    checksum was verified (not just config)."""
    monkeypatch.setattr("scripts.check_provenance.PACKAGE_ROOT", tmp_path)
    _write_variant_bundle(tmp_path, git_hash="h", write_model=True)
    checker = ProvenanceChecker()
    check_output_bundle(checker, norms_sha=None, head_hash="h")
    bundle = [r for r in checker.results if r[1].endswith("bundle")]
    assert bundle and bundle[0][0] == "PASS"
    assert "model checksums verified" in bundle[0][2]


def test_check_output_bundle_model_absent_reports_bytes_not_checked(tmp_path, monkeypatch) -> None:
    """When model.onnx is absent (gitignored on a clone), the bundle line is HONEST that
    model bytes were NOT checked -- never a blanket 'checksums verified' (PROV-2)."""
    monkeypatch.setattr("scripts.check_provenance.PACKAGE_ROOT", tmp_path)
    _write_variant_bundle(tmp_path, git_hash="h")  # no model.onnx on disk
    checker = ProvenanceChecker()
    check_output_bundle(checker, norms_sha=None, head_hash="h")
    bundle = [r for r in checker.results if r[1].endswith("bundle")]
    assert bundle and bundle[0][0] == "PASS"
    assert "model bytes NOT checked" in bundle[0][2]
    assert "model.onnx absent" in bundle[0][2]


def test_check_output_bundle_item_info_mismatch_fails(tmp_path, monkeypatch) -> None:
    """A present item_info.json whose SHA != the recorded lock is a hard FAIL (the
    working-tree ranking differs from what the model was trained on) -- never excused as a
    'cosmetic re-stamp', and the bundle PASS is skipped for that variant."""
    monkeypatch.setattr("scripts.check_provenance.PACKAGE_ROOT", tmp_path)
    item_info = tmp_path / "data" / "processed" / "reference" / "item_info.json"
    item_info.parent.mkdir(parents=True)
    item_info.write_text(json.dumps({"ranking": [3, 1, 2]}), encoding="utf-8")
    _write_variant_bundle(
        tmp_path,
        git_hash="h",
        item_info_sha256="deadbeef" * 8,  # 64 hex; will not match the on-disk file
        item_info_path="data/processed/reference/item_info.json",
    )
    checker = ProvenanceChecker()
    check_output_bundle(checker, norms_sha=None, head_hash="h", strict_head=False)
    details = " ".join(r[2] for r in checker.results)
    assert any(r[0] == "FAIL" and "item_info lock" in r[1] for r in checker.results)
    assert "trained against" in details
    assert "stamp" not in details.lower()  # output must not mention re-stamps
    assert not any(r[0] == "PASS" and r[1].endswith("bundle") for r in checker.results)


def test_check_output_bundle_item_info_match_passes(tmp_path, monkeypatch) -> None:
    """A present item_info.json whose SHA equals the recorded lock PASSes."""
    monkeypatch.setattr("scripts.check_provenance.PACKAGE_ROOT", tmp_path)
    item_info = tmp_path / "data" / "processed" / "reference" / "item_info.json"
    item_info.parent.mkdir(parents=True)
    item_info.write_text(json.dumps({"ranking": [3, 1, 2]}), encoding="utf-8")
    real_sha = hashlib.sha256(item_info.read_bytes()).hexdigest()
    _write_variant_bundle(
        tmp_path,
        git_hash="h",
        item_info_sha256=real_sha,
        item_info_path="data/processed/reference/item_info.json",
    )
    checker = ProvenanceChecker()
    check_output_bundle(checker, norms_sha=None, head_hash="h", strict_head=False)
    assert any(r[0] == "PASS" and "item_info lock" in r[1] for r in checker.results)
    assert all(r[0] != "FAIL" for r in checker.results)


def test_check_output_bundle_item_info_absent_skips(tmp_path, monkeypatch) -> None:
    """On a clone the gitignored item_info.json is absent -> SKIP, not FAIL."""
    monkeypatch.setattr("scripts.check_provenance.PACKAGE_ROOT", tmp_path)
    _write_variant_bundle(
        tmp_path,
        git_hash="h",
        item_info_sha256="deadbeef" * 8,
        item_info_path="data/processed/reference/item_info.json",  # not created on disk
    )
    checker = ProvenanceChecker()
    check_output_bundle(checker, norms_sha=None, head_hash="h", strict_head=False)
    assert any(r[0] == "SKIP" and "item_info lock" in r[1] for r in checker.results)
    assert all(r[0] != "FAIL" for r in checker.results)


def test_check_output_bundle_intra_hash_disagreement_fails(tmp_path) -> None:
    """export vs config.provenance git_hash disagreement always FAILS (any mode)."""
    checker = ProvenanceChecker()
    _write_variant_bundle(tmp_path, git_hash="exportHASH", config_git_hash="configHASH")
    with patch("scripts.check_provenance.PACKAGE_ROOT", tmp_path):
        check_output_bundle(checker, norms_sha=None, head_hash="exportHASH", strict_head=False)
    assert any(r[0] == "FAIL" and "agreement" in r[1] for r in checker.results)


def test_warned_status_not_counted_as_failure() -> None:
    """checker.warned() surfaces an advisory without incrementing the failure count."""
    checker = ProvenanceChecker()
    checker.passed("a")
    checker.warned("b", "stale")
    assert checker.print_summary() == 0
    assert any(s == "WARN" for s, _, _ in checker.results)


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


# ---------------------------------------------------------------------------
# P3: norms-meta sidecar absent -> SKIP (not FAIL), so a fresh clone passes.
# ---------------------------------------------------------------------------


def test_check_norms_meta_skips_when_absent(tmp_path) -> None:
    """A fresh clone (sidecar gitignored/not pulled) must SKIP, not FAIL."""
    (tmp_path / "artifacts").mkdir(parents=True)
    checker = ProvenanceChecker()
    with patch("scripts.check_provenance.PACKAGE_ROOT", tmp_path):
        check_norms_meta(checker, norms_sha="abc")
    assert any(r[0] == "SKIP" and "Norms meta" in r[1] for r in checker.results)
    assert all(r[0] != "FAIL" for r in checker.results)


# ---------------------------------------------------------------------------
# P4: _release_fresh — ancestor + generation-scoped-diff freshness (merge-robust,
# git-degrading). Uses a real throwaway git repo so the merge-base / diff logic
# is exercised, not mocked.
# ---------------------------------------------------------------------------


def _init_git_repo(root: Path):
    """Init a throwaway repo with generation dirs + a docs file; return (commit fn, rev fn)."""
    def git(*a: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *a], check=True, capture_output=True, text=True
        ).stdout.strip()

    root.mkdir(parents=True, exist_ok=True)
    git("init", "-q")
    git("config", "user.email", "t@t.test")
    git("config", "user.name", "t")
    git("config", "commit.gpgsign", "false")
    (root / "pipeline").mkdir()
    (root / "pipeline" / "stage.py").write_text("v1\n")
    (root / "docs").mkdir()
    (root / "docs" / "notes.md").write_text("d1\n")

    def commit(msg: str) -> str:
        git("add", "-A")
        git("commit", "-q", "-m", msg)
        return git("rev-parse", "HEAD")

    return root, commit, git


def test_release_fresh_exact_match() -> None:
    assert _release_fresh("abc123", "abc123", strict_head=True)[0] is True
    assert _release_fresh(None, "abc123", strict_head=True)[0] is True  # unknown -> degrade


def test_release_fresh_ancestor_clean_gen_diff_is_fresh(tmp_path, monkeypatch) -> None:
    """A release commit on top of the generation commit that touches only NON-generation
    paths (docs) is fresh even under strict_head."""
    repo, commit, _ = _init_git_repo(tmp_path / "r")
    monkeypatch.setattr("scripts.check_provenance.PACKAGE_ROOT", repo)
    base = commit("base (generation commit)")
    (repo / "docs" / "notes.md").write_text("d2 — release docs\n")
    head = commit("release: docs only (no generation drift)")
    fresh, detail = _release_fresh(base, head, strict_head=True)
    assert fresh is True, detail
    assert "ancestor" in detail


def test_release_fresh_generation_drift_is_stale(tmp_path, monkeypatch) -> None:
    """A commit changing pipeline/ since the bundle commit -> NOT fresh (gen drift)."""
    repo, commit, _ = _init_git_repo(tmp_path / "r")
    monkeypatch.setattr("scripts.check_provenance.PACKAGE_ROOT", repo)
    base = commit("base")
    (repo / "pipeline" / "stage.py").write_text("v2 — generator changed\n")
    head = commit("changed a generator without re-running")
    fresh, detail = _release_fresh(base, head, strict_head=True)
    assert fresh is False
    assert "generation paths changed" in detail


def test_release_fresh_publish_stage_edit_is_fresh(tmp_path, monkeypatch) -> None:
    """Editing pipeline/13_upload_hf.py (the publish/transport stage) does NOT count as
    generation drift -- it ships the already-built bundle, it doesn't produce it -- so the
    bundle stays fresh even under strict_head."""
    repo, commit, _ = _init_git_repo(tmp_path / "r")
    monkeypatch.setattr("scripts.check_provenance.PACKAGE_ROOT", repo)
    (repo / "pipeline" / "13_upload_hf.py").write_text("uploader v1\n")
    base = commit("base (generation commit)")
    (repo / "pipeline" / "13_upload_hf.py").write_text("uploader v2 — add branch support\n")
    head = commit("tooling: add --revision branch upload")
    fresh, detail = _release_fresh(base, head, strict_head=True)
    assert fresh is True, detail
    assert "ancestor" in detail


def test_release_fresh_merge_commit_is_fresh(tmp_path, monkeypatch) -> None:
    """Merge-to-main robustness: the bundle commit stays fresh through a merge commit
    that carried no generation-path changes."""
    repo, commit, git = _init_git_repo(tmp_path / "r")
    monkeypatch.setattr("scripts.check_provenance.PACKAGE_ROOT", repo)
    base = commit("base (generation commit)")
    git("checkout", "-q", "-b", "feature")
    (repo / "docs" / "notes.md").write_text("feature docs\n")
    commit("feature: docs only")
    git("checkout", "-q", "-")  # back to the default branch
    git("merge", "--no-ff", "-q", "-m", "merge feature", "feature")  # merge (result unused)
    fresh, detail = _release_fresh(base, git("rev-parse", "HEAD"), strict_head=True)
    assert fresh is True, detail


def test_release_fresh_non_ancestor_is_stale(tmp_path, monkeypatch) -> None:
    """A bundle hash that is NOT an ancestor of HEAD -> NOT fresh."""
    repo, commit, git = _init_git_repo(tmp_path / "r")
    monkeypatch.setattr("scripts.check_provenance.PACKAGE_ROOT", repo)
    base = commit("base")
    git("checkout", "-q", "-b", "other")
    (repo / "docs" / "notes.md").write_text("divergent\n")
    other = commit("divergent commit (not an ancestor of base)")
    fresh, detail = _release_fresh(other, base, strict_head=True)
    assert fresh is False
    assert "not an ancestor" in detail


def test_release_fresh_git_unavailable_degrades_to_fresh(monkeypatch) -> None:
    """No git / shallow clone (object absent) -> degrade to fresh (rely on sha chain)."""
    monkeypatch.setattr("scripts.check_provenance._git", lambda *a: None)
    fresh, detail = _release_fresh("aaaaaa", "bbbbbb", strict_head=True)
    assert fresh is True
    assert "git-unverifiable" in detail or "git unavailable" in detail


# ---------------------------------------------------------------------------
# Option (a): model-scoped freshness — distinguishes model-producing code drift,
# provable hyperparameter-input drift, and changes that cannot affect model.onnx.
# ---------------------------------------------------------------------------


def _prov_doc_with_hp(
    hp: dict,
    *,
    mode: str = "config_locked_params",
    path: str = "artifacts/tuned_params.json",
) -> dict:
    """Minimal provenance doc carrying a self-describing hyperparameter lock."""
    return {
        "training": {
            "config": {
                "hyperparameters": hp,
                "hyperparameters_source": {"mode": mode, "path": path},
            }
        }
    }


def _write_tuned_params(repo: Path, hp: dict) -> None:
    (repo / "artifacts").mkdir(parents=True, exist_ok=True)
    (repo / "artifacts" / "tuned_params.json").write_text(
        json.dumps({"hyperparameters": hp}), encoding="utf-8"
    )


def test_commit_relationship_kinds() -> None:
    assert _commit_relationship("abc", "abc")[0] == "at_head"
    assert _commit_relationship(None, "abc")[0] == "unverifiable"
    assert _commit_relationship("abc", "unknown")[0] == "unverifiable"


def test_model_freshness_at_head_is_fresh() -> None:
    verdict, _ = _model_freshness("abc123", "abc123", prov_doc=None)
    assert verdict == "fresh"


def test_model_freshness_git_unavailable_is_fresh(monkeypatch) -> None:
    """No git + no provable input drift -> degrade the code signal to fresh (sha chain)."""
    monkeypatch.setattr("scripts.check_provenance._git", lambda *a: None)
    verdict, detail = _model_freshness("aaaaaa", "bbbbbb", prov_doc=None)
    assert verdict == "fresh"
    assert "git-unverifiable" in detail or "git unavailable" in detail


def test_model_freshness_post_hoc_stage_change_is_fresh(tmp_path, monkeypatch) -> None:
    """A change confined to a stage that consumes the model (10_simulate) cannot affect
    model.onnx -> fresh."""
    repo, commit, _ = _init_git_repo(tmp_path / "r")
    monkeypatch.setattr("scripts.check_provenance.PACKAGE_ROOT", repo)
    (repo / "pipeline" / "10_simulate.py").write_text("sim v1\n")
    _write_tuned_params(repo, {"n_estimators": 100})
    base = commit("base (generation commit)")
    (repo / "pipeline" / "10_simulate.py").write_text("sim v2 — adaptive-form tweak\n")
    head = commit("eval: simulation tweak (post-hoc)")
    verdict, detail = _model_freshness(base, head, prov_doc=_prov_doc_with_hp({"n_estimators": 100}))
    assert verdict == "fresh", detail
    assert "cannot affect model.onnx" in detail


def test_model_freshness_training_code_change_is_code_drift(tmp_path, monkeypatch) -> None:
    """A change to a model-producing stage (07_train) with the hyperparameter lock intact
    is code_drift -- named precisely, with the inputs confirmed unchanged."""
    repo, commit, _ = _init_git_repo(tmp_path / "r")
    monkeypatch.setattr("scripts.check_provenance.PACKAGE_ROOT", repo)
    (repo / "pipeline" / "07_train.py").write_text("train v1\n")
    _write_tuned_params(repo, {"n_estimators": 100})
    base = commit("base")
    (repo / "pipeline" / "07_train.py").write_text("train v2 — refactor\n")
    head = commit("refactor trainer")
    verdict, detail = _model_freshness(base, head, prov_doc=_prov_doc_with_hp({"n_estimators": 100}))
    assert verdict == "code_drift", detail
    assert "pipeline/07_train.py" in detail
    assert "hyperparameters verified unchanged" in detail


def test_model_freshness_config_change_is_code_drift(tmp_path, monkeypatch) -> None:
    """A configs/ change (e.g. sparsity / training params) is model-relevant -> code_drift."""
    repo, commit, _ = _init_git_repo(tmp_path / "r")
    monkeypatch.setattr("scripts.check_provenance.PACKAGE_ROOT", repo)
    (repo / "configs").mkdir(exist_ok=True)
    (repo / "configs" / "reference.yaml").write_text("v: 1\n")
    _write_tuned_params(repo, {"n_estimators": 100})
    base = commit("base")
    (repo / "configs" / "reference.yaml").write_text("v: 2  # sparsity tweak\n")
    head = commit("config change")
    verdict, detail = _model_freshness(base, head, prov_doc=_prov_doc_with_hp({"n_estimators": 100}))
    assert verdict == "code_drift", detail
    assert "configs/reference.yaml" in detail


def test_model_freshness_lib_change_is_code_drift(tmp_path, monkeypatch) -> None:
    """A lib/ change (shared training code) -> code_drift (conservative; no hash proves no effect)."""
    repo, commit, _ = _init_git_repo(tmp_path / "r")
    monkeypatch.setattr("scripts.check_provenance.PACKAGE_ROOT", repo)
    (repo / "lib").mkdir(exist_ok=True)
    (repo / "lib" / "sparsity.py").write_text("v = 1\n")
    _write_tuned_params(repo, {"n_estimators": 100})
    base = commit("base")
    (repo / "lib" / "sparsity.py").write_text("v = 2\n")
    head = commit("lib change")
    verdict, detail = _model_freshness(base, head, prov_doc=_prov_doc_with_hp({"n_estimators": 100}))
    assert verdict == "code_drift", detail
    assert "lib/sparsity.py" in detail


def test_model_freshness_data_stage_change_is_code_drift(tmp_path, monkeypatch) -> None:
    """A data-prep stage (05_compute_correlations) is NOT in MODEL_IRRELEVANT_STAGES, so it
    stays code_drift -- on a clone the item_info lock SKIPs, making this the only defense."""
    repo, commit, _ = _init_git_repo(tmp_path / "r")
    monkeypatch.setattr("scripts.check_provenance.PACKAGE_ROOT", repo)
    (repo / "pipeline" / "05_compute_correlations.py").write_text("v1\n")
    _write_tuned_params(repo, {"n_estimators": 100})
    base = commit("base")
    (repo / "pipeline" / "05_compute_correlations.py").write_text("v2 — ranking tweak\n")
    head = commit("stage 05 change")
    verdict, detail = _model_freshness(base, head, prov_doc=_prov_doc_with_hp({"n_estimators": 100}))
    assert verdict == "code_drift", detail
    assert "pipeline/05_compute_correlations.py" in detail


def test_model_freshness_dependency_lockfile_change_is_code_drift(tmp_path, monkeypatch) -> None:
    """A uv.lock change (different resolved library versions) -> code_drift: a retrain at HEAD
    would run against a different environment, so model.onnx could differ (COV-1)."""
    repo, commit, _ = _init_git_repo(tmp_path / "r")
    monkeypatch.setattr("scripts.check_provenance.PACKAGE_ROOT", repo)
    (repo / "uv.lock").write_text("xgboost==3.1.3\n")
    _write_tuned_params(repo, {"n_estimators": 100})
    base = commit("base")
    (repo / "uv.lock").write_text("xgboost==3.2.0\n")
    head = commit("deps: bump xgboost")
    verdict, detail = _model_freshness(base, head, prov_doc=_prov_doc_with_hp({"n_estimators": 100}))
    assert verdict == "code_drift", detail
    assert "uv.lock" in detail


def test_model_freshness_tuner_change_resolved_when_hp_unchanged(tmp_path, monkeypatch) -> None:
    """A change confined to the tuner (06_tune) is resolved to fresh when the locked
    hyperparameters are byte-identical -- the tuner only produces the (unchanged) HP."""
    repo, commit, _ = _init_git_repo(tmp_path / "r")
    monkeypatch.setattr("scripts.check_provenance.PACKAGE_ROOT", repo)
    (repo / "pipeline" / "06_tune.py").write_text("tune v1\n")
    _write_tuned_params(repo, {"n_estimators": 100})
    base = commit("base")
    (repo / "pipeline" / "06_tune.py").write_text("tune v2 — search-space comment\n")
    head = commit("tooling: tuner comment")
    verdict, detail = _model_freshness(base, head, prov_doc=_prov_doc_with_hp({"n_estimators": 100}))
    assert verdict == "fresh", detail
    assert "cannot affect model.onnx" in detail


def test_model_freshness_hyperparameter_drift_is_input_drift(tmp_path, monkeypatch) -> None:
    """When the on-disk locked hyperparameters differ from the bundle's recorded set,
    model.onnx would differ -> input_drift (a hard FAIL)."""
    repo, commit, _ = _init_git_repo(tmp_path / "r")
    monkeypatch.setattr("scripts.check_provenance.PACKAGE_ROOT", repo)
    (repo / "pipeline" / "06_tune.py").write_text("tune v1\n")
    _write_tuned_params(repo, {"n_estimators": 100})
    base = commit("base")
    (repo / "pipeline" / "06_tune.py").write_text("tune v2 — new search space\n")
    _write_tuned_params(repo, {"n_estimators": 200})  # locked HP actually moved
    head = commit("retune: different hyperparameters")
    verdict, detail = _model_freshness(base, head, prov_doc=_prov_doc_with_hp({"n_estimators": 100}))
    assert verdict == "input_drift", detail
    assert "WOULD differ" in detail


def test_model_freshness_input_drift_when_git_unavailable(tmp_path, monkeypatch) -> None:
    """P0 regression: a moved locked hyperparameter is input_drift even when git is
    unverifiable (shallow clone / no git). The HP check is content-only and runs FIRST, so
    it must not be gated behind the git relationship (the original gap)."""
    monkeypatch.setattr("scripts.check_provenance.PACKAGE_ROOT", tmp_path)
    monkeypatch.setattr("scripts.check_provenance._git", lambda *a: None)
    _write_tuned_params(tmp_path, {"n_estimators": 999})
    verdict, detail = _model_freshness(
        "aaaaaa", "bbbbbb", prov_doc=_prov_doc_with_hp({"n_estimators": 100})
    )
    assert verdict == "input_drift", detail


def test_model_freshness_input_drift_at_head(tmp_path, monkeypatch) -> None:
    """P0 regression: a dirty/edited tuned_params.json with the bundle at HEAD still surfaces
    as input_drift (the at_head short-circuit must not skip the HP check)."""
    monkeypatch.setattr("scripts.check_provenance.PACKAGE_ROOT", tmp_path)
    _write_tuned_params(tmp_path, {"n_estimators": 999})
    verdict, _ = _model_freshness("samehash", "samehash", prov_doc=_prov_doc_with_hp({"n_estimators": 100}))
    assert verdict == "input_drift"


def test_model_freshness_input_drift_on_divergent_history(tmp_path, monkeypatch) -> None:
    """A moved locked hyperparameter on rewritten/divergent history (not_ancestor) is
    input_drift (a hard FAIL), not a downgraded code_drift WARN."""
    repo, commit, git = _init_git_repo(tmp_path / "r")
    monkeypatch.setattr("scripts.check_provenance.PACKAGE_ROOT", repo)
    _write_tuned_params(repo, {"n_estimators": 999})
    base = commit("base")
    git("checkout", "-q", "-b", "other")
    (repo / "docs" / "notes.md").write_text("divergent\n")
    other = commit("divergent commit (not an ancestor of base)")
    # bundle=other is NOT an ancestor of base; with drifted on-disk HP this returns via the
    # HP-first check (before the not_ancestor branch), proving input_drift wins on any history.
    verdict, _ = _model_freshness(other, base, prov_doc=_prov_doc_with_hp({"n_estimators": 100}))
    assert verdict == "input_drift"


def test_model_freshness_not_ancestor_with_matching_hp_is_code_drift(tmp_path, monkeypatch) -> None:
    """Divergent/rewritten history (bundle NOT an ancestor of HEAD) with the hyperparameter
    lock intact -> code_drift (conservative; the diff cannot be reasoned about). Exercises the
    not_ancestor branch of _model_freshness directly (HP matches, so the HP-first check passes)."""
    repo, commit, git = _init_git_repo(tmp_path / "r")
    monkeypatch.setattr("scripts.check_provenance.PACKAGE_ROOT", repo)
    _write_tuned_params(repo, {"n_estimators": 100})
    base = commit("base")
    git("checkout", "-q", "-b", "other")
    (repo / "docs" / "notes.md").write_text("divergent\n")
    other = commit("divergent commit (not an ancestor of base)")
    verdict, detail = _model_freshness(other, base, prov_doc=_prov_doc_with_hp({"n_estimators": 100}))
    assert verdict == "code_drift", detail
    assert "not an ancestor" in detail


def test_model_freshness_code_drift_when_hp_source_unverifiable(tmp_path, monkeypatch) -> None:
    """A model-producing code change with a non-re-readable HP source (a cli_params_override
    lock, not backed by a file) stays code_drift and says so -- never silently fresh."""
    repo, commit, _ = _init_git_repo(tmp_path / "r")
    monkeypatch.setattr("scripts.check_provenance.PACKAGE_ROOT", repo)
    (repo / "pipeline" / "11_export_onnx.py").write_text("export v1\n")
    base = commit("base")
    (repo / "pipeline" / "11_export_onnx.py").write_text("export v2 — opset bump\n")
    head = commit("export change")
    verdict, detail = _model_freshness(
        base, head, prov_doc=_prov_doc_with_hp({"n_estimators": 100}, mode="cli_params_override")
    )
    assert verdict == "code_drift", detail
    assert "pipeline/11_export_onnx.py" in detail
    assert "unverifiable" in detail


def test_check_output_bundle_input_drift_end_to_end(tmp_path, monkeypatch) -> None:
    """End-to-end (NOT mocked): a bundle whose recorded locked hyperparameters differ from the
    on-disk artifacts/tuned_params.json FAILs the HEAD-freshness line via the real
    _model_freshness path -- even in a non-git tree, since the HP check is git-independent."""
    monkeypatch.setattr("scripts.check_provenance.PACKAGE_ROOT", tmp_path)
    _write_variant_bundle(tmp_path, git_hash="bundlehash", hyperparameters={"n_estimators": 100})
    _write_tuned_params(tmp_path, {"n_estimators": 999})  # on-disk locked HP moved
    checker = ProvenanceChecker()
    check_output_bundle(checker, norms_sha=None, head_hash="otherhead", strict_head=False)
    assert any(
        r[0] == "FAIL" and "HEAD freshness" in r[1] and "WOULD differ" in r[2]
        for r in checker.results
    )


def test_hyperparameters_drift_match_and_drift(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("scripts.check_provenance.PACKAGE_ROOT", tmp_path)
    _write_tuned_params(tmp_path, {"n_estimators": 100, "max_depth": 5})
    assert (
        _hyperparameters_drift(_prov_doc_with_hp({"n_estimators": 100, "max_depth": 5}))[0]
        == "match"
    )
    assert (
        _hyperparameters_drift(_prov_doc_with_hp({"n_estimators": 100, "max_depth": 6}))[0]
        == "drift"
    )


def test_hyperparameters_drift_unverifiable(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("scripts.check_provenance.PACKAGE_ROOT", tmp_path)
    # No tuned_params.json on disk -> source absent.
    assert _hyperparameters_drift(_prov_doc_with_hp({"n_estimators": 100}))[0] == "unverifiable"
    # A source mode that is not a re-readable file (cli_params_override) -> unverifiable.
    assert (
        _hyperparameters_drift(_prov_doc_with_hp({"n_estimators": 100}, mode="cli_params_override"))[0]
        == "unverifiable"
    )
    # No provenance at all -> unverifiable.
    assert _hyperparameters_drift(None)[0] == "unverifiable"


def test_hyperparameters_drift_pins_to_canonical_source(tmp_path, monkeypatch) -> None:
    """F4: the comparison is pinned to the canonical artifacts/tuned_params.json. A bundle that
    points its source at a different (e.g. attacker-crafted) path is 'unverifiable', not 'match'
    -- even if a matching file exists there -- so a forged source path cannot defeat the check."""
    monkeypatch.setattr("scripts.check_provenance.PACKAGE_ROOT", tmp_path)
    _write_tuned_params(tmp_path, {"n_estimators": 100})  # canonical = the REAL locked HP
    (tmp_path / "artifacts" / "evil.json").write_text(
        json.dumps({"hyperparameters": {"n_estimators": 999}}), encoding="utf-8"
    )
    # A forged recorded set (999) plus a source path pointing at a matching crafted file.
    status, detail = _hyperparameters_drift(
        _prov_doc_with_hp({"n_estimators": 999}, path="artifacts/evil.json")
    )
    assert status == "unverifiable", detail
    assert "canonical" in detail
    # The same forged set compared against the canonical source is correctly real drift.
    assert _hyperparameters_drift(_prov_doc_with_hp({"n_estimators": 999}))[0] == "drift"
