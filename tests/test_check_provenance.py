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
) -> Path:
    """Write a valid output/<variant>/ bundle (matching config checksum)."""
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
    if payload_provenance_hash is not None:
        # The nested tune payload hash legitimately differs from the export/train
        # hashes (as in the real committed bundle) and must NOT trip the
        # intra-bundle agreement check.
        training["config"] = {
            "hyperparameters_source": {
                "payload_provenance": {"git_hash": payload_provenance_hash}
            }
        }
    prov_doc = {
        "export": {"git_hash": git_hash, "data_snapshot_id": data_snapshot_id},
        "training": training,
        "artifacts": {"config_json_sha256": config_sha, "model_onnx_sha256": "deadbeef"},
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
    """A genuinely-stale (non-fresh) bundle is WARN by default and FAIL under strict_head.
    (_release_fresh forced False to simulate gen-path drift / non-ancestor; a non-git
    tmp_path would otherwise degrade to git-unverifiable=fresh.)"""
    checker = ProvenanceChecker()
    _write_variant_bundle(tmp_path, git_hash="bundlehash")
    with (
        patch("scripts.check_provenance.PACKAGE_ROOT", tmp_path),
        patch("scripts.check_provenance._release_fresh", return_value=(False, "generation paths changed")),
    ):
        check_output_bundle(checker, norms_sha=None, head_hash="otherhead", strict_head=False)
    assert any(r[0] == "WARN" for r in checker.results)
    assert checker.print_summary() == 0  # WARN is not a failure

    strict = ProvenanceChecker()
    with (
        patch("scripts.check_provenance.PACKAGE_ROOT", tmp_path),
        patch("scripts.check_provenance._release_fresh", return_value=(False, "generation paths changed")),
    ):
        check_output_bundle(strict, norms_sha=None, head_hash="otherhead", strict_head=True)
    assert any(r[0] == "FAIL" for r in strict.results)


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
    merge_head = git("merge", "--no-ff", "-q", "-m", "merge feature", "feature") or git("rev-parse", "HEAD")
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
