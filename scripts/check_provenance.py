#!/usr/bin/env python3
"""Validate the entire committed artifact provenance tree.

Checks that all provenance sidecars, manifests, and cross-references
are internally consistent. Designed for CI and pre-submission verification.

Usage:
    uv run python scripts/check_provenance.py              # advisory mode
    uv run python scripts/check_provenance.py --strict     # exit 1 on any FAIL (CI)
    uv run python scripts/check_provenance.py --strict-head # also FAIL on git-staleness
    make verify-release                                    # clone-side, no DB / no retrain
    make verify-release STRICT_HEAD=1                      # enforce git-freshness

Status legend:
    PASS  internally consistent.
    WARN  a non-fatal advisory -- the bundle is internally consistent but predates
          HEAD (an ancestor with no model-affecting drift). WARN does NOT fail
          --strict; it DOES fail under --strict-head where noted.
    FAIL  a genuine inconsistency (checksum mismatch, intra-bundle git_hash
          disagreement, substantive generation-code drift, a provable model-input
          change such as the locked hyperparameters moving, or an on-disk item_info
          ranking that no longer matches the bundle's recorded lock).
    SKIP  an artifact is absent (e.g. gitignored on a fresh clone).

Model freshness, specifically (see _model_freshness): the per-variant "HEAD freshness"
check asks the narrower question "could output/<variant>/model.onnx be stale?" rather
than "did any generation file change?". A provable model-INPUT change -- the locked
hyperparameters in artifacts/tuned_params.json differing from the bundle's recorded set --
is ALWAYS a FAIL, checked first on every path (the comparison is content-only and
git-independent, so it is not gated behind the git relationship). For the code signal it
ignores pipeline stages that cannot produce the model (post-hoc eval, figures, upload) and
drops the tuning stage when the hyperparameter lock is unchanged, so the residual WARN is
genuine training/export/config/dependency drift (07_train / 11_export_onnx / lib / configs
/ uv.lock), where no content hash can prove the trained model is unchanged. On a verifiable
(ancestor) path that code signal is never downgraded to PASS; git-unverifiable states
(shallow clone / no git / diff failure) degrade the code signal to "fresh" and rely on the
content-sha chain, while the hyperparameter-input FAIL still applies. The resolved XGBoost
thread count (xgb_n_jobs) and the cross-thread non-determinism of tree_method=hist are
documented out-of-band determinants (docs/pipeline.md), not covered by this cheap check.

Provenance, how it works: a provenance git_hash records the commit a stage was run at.
    The workflow is to run the stage (or the whole pipeline), let it record the current
    commit, then commit the changed files. A provenance hash changes ONLY by genuinely
    re-running the stage that produces the artifact -- artifacts are never edited to point
    at a different commit. This checker is read-only: it reports inconsistencies; it never
    regenerates, retrains, or rewrites a provenance hash.

Release freshness: a bundle is "fresh" when its export git_hash is HEAD, OR an ancestor of
    HEAD with no model-affecting drift under pipeline/ + lib/ + configs/ (release/refresh
    commits and a later merge into main legitimately sit on top of the generation commit).
    Model-affecting drift -> WARN by default, FAIL under --strict-head, with the changed
    paths named. A new model.onnx only ever comes from `make all` / the remote pipeline.

item_info lock: the bundle records the SHA-256 of the item ranking the model was trained
    against (training.data.item_info_sha256 in output/<variant>/provenance.json, originally
    written by stage 07). On a fresh clone the on-disk data/processed/<variant>/item_info.json
    is gitignored and absent -> SKIP. When it is present, its SHA must equal the recorded
    lock; any difference means the working-tree ranking no longer matches what the model was
    trained on -> FAIL. See check_output_bundle.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from lib.constants import REFERENCE_VARIANT, VARIANTS
from lib.provenance import _detect_git_hash, file_sha256


def _load_json(path: Path) -> dict | None:
    """Load a JSON file, returning None on any error."""
    try:
        with open(path) as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return None


# Paths whose contents determine the generated artifacts. A release commit may
# legitimately sit ABOVE the bundle's generation commit (it adds the artifacts,
# docs, tooling) as long as NONE of these changed in between -- that proves the
# generation code at HEAD is byte-identical to the code that produced the bundle.
GENERATION_PATHS = ("pipeline", "lib", "configs")
# ...minus the publish/upload stage: pipeline/13_upload_hf.py only TRANSPORTS the
# already-built bundle to HuggingFace; it does not produce the verified artifacts,
# so editing it (e.g. to add branch support) must not flag the bundle as stale.
GENERATION_PATH_EXCLUDES = (":(exclude)pipeline/13_upload_hf.py",)

# Pipeline stages whose edits cannot change the trained model.onnx: they consume the
# finished model (validation, baselines, the adaptive-form simulation), render figures,
# or only transport the bundle. The model-freshness check (see _model_freshness) ignores
# these so a change confined to post-hoc / publishing code does not flag the model stale.
MODEL_IRRELEVANT_STAGES = (
    "pipeline/08_validate.py",
    "pipeline/09_baselines.py",
    "pipeline/10_simulate.py",
    "pipeline/12_generate_figures.py",
    "pipeline/13_upload_hf.py",
)
# The tuning stage produces only the hyperparameters, which are content-hashed and
# re-verified against the bundle (see _hyperparameters_drift). A tuner edit that leaves
# the locked hyperparameters identical therefore cannot change the model, so it is
# dropped from the model-freshness signal once the lock is confirmed unchanged.
HP_PRODUCING_STAGE = "pipeline/06_tune.py"

# Canonical, git-tracked source of the locked hyperparameters. _hyperparameters_drift
# pins its comparison to this file rather than trusting the path a (potentially forged)
# bundle names, so a tampered provenance.json cannot redirect the check to a matching
# crafted file. All variants record this as their config_locked_params source.
CANONICAL_HP_SOURCE_REL = "artifacts/tuned_params.json"

# Dependency-resolution lockfile. A change here means a retrain at HEAD would run against
# different library versions, so the model could differ; it is therefore treated as
# model-relevant drift (>= WARN). uv.lock is the resolved source of truth -- pyproject.toml
# tool-config edits that do not touch dependency resolution are intentionally not watched,
# and the exact resolved versions / XGBoost thread count are documented out-of-band
# determinants (see docs/pipeline.md), not re-verified by this cheap check.
ENVIRONMENT_PATHS = ("uv.lock",)


def _git(*args: str) -> subprocess.CompletedProcess[str] | None:
    """Run a git command rooted at PACKAGE_ROOT. Returns None if git is
    unavailable (no binary / timeout / OS error) so callers can degrade gracefully."""
    try:
        return subprocess.run(
            ["git", *args],
            cwd=str(PACKAGE_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _commit_present(commit: str) -> bool:
    """True iff `commit` resolves to a commit object here (False on a shallow
    clone / tarball where the object is absent)."""
    cp = _git("rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}")
    return cp is not None and cp.returncode == 0


def _commit_relationship(bundle_hash: str | None, head_hash: str | None) -> tuple[str, str]:
    """Classify how ``bundle_hash`` relates to ``head_hash`` in git history.

    Returns (kind, detail) where kind is one of:
      "unverifiable" -- hashes unknown, no git, or objects absent (shallow clone /
                        tarball). Callers degrade gracefully (rely on the sha chain).
      "at_head"      -- bundle_hash == head_hash.
      "ancestor"     -- bundle_hash is a strict ancestor of head_hash.
      "not_ancestor" -- a real, differing commit that is NOT an ancestor of HEAD.

    Shared by _release_fresh (generic bundle freshness) and _model_freshness
    (model-scoped freshness) so the git-degradation rules live in exactly one place."""
    if not bundle_hash or not head_hash or head_hash == "unknown":
        return "unverifiable", "head/bundle hash unknown (git-unverifiable)"
    if bundle_hash == head_hash:
        return "at_head", "at HEAD"
    if _git("rev-parse", "--git-dir") is None:
        return "unverifiable", "git unavailable (git-unverifiable)"
    if not _commit_present(bundle_hash) or not _commit_present(head_hash):
        return "unverifiable", "bundle/HEAD object absent (shallow clone; git-unverifiable)"
    anc = _git("merge-base", "--is-ancestor", bundle_hash, head_hash)
    if anc is None:
        return "unverifiable", "git unavailable (git-unverifiable)"
    if anc.returncode != 0:
        return (
            "not_ancestor",
            f"bundle {bundle_hash[:12]}... is not an ancestor of HEAD {head_hash[:12]}...",
        )
    return "ancestor", f"bundle {bundle_hash[:12]}... is an ancestor of HEAD"


def _release_fresh(
    bundle_hash: str | None, head_hash: str | None, *, strict_head: bool
) -> tuple[bool, str]:
    """Decide whether a bundle recorded at ``bundle_hash`` is fresh w.r.t. ``head_hash``.

    Fresh when:
      (a) bundle_hash == head_hash (exact match -- the original rule), OR
      (b) bundle_hash is an ANCESTOR of head_hash AND no GENERATION_PATHS file
          changed between them. This lets the release/refresh commits (and a later
          merge into main) sit on top of the generation commit without going stale,
          as long as the pipeline/lib/configs code is unchanged.

    Graceful degradation: if git is unavailable or the bundle/HEAD object is
    absent (shallow clone / tarball), the relationship can't be evaluated -> return
    fresh=True with a 'git-unverifiable' note, so the content-sha chain is relied
    upon rather than failing. ``strict_head`` only changes the CALLER's
    PASS/WARN/FAIL mapping, not this predicate. Returns (fresh, detail)."""
    kind, detail = _commit_relationship(bundle_hash, head_hash)
    if kind in ("unverifiable", "at_head"):
        return True, detail
    if kind == "not_ancestor":
        return False, detail
    # kind == "ancestor": _commit_relationship guarantees both are real commit hashes.
    assert bundle_hash is not None and head_hash is not None
    diff = _git(
        "diff", "--name-only", bundle_hash, head_hash, "--",
        *GENERATION_PATHS, *GENERATION_PATH_EXCLUDES,
    )
    if diff is None or diff.returncode != 0:
        return True, "gen-path diff unavailable (git-unverifiable)"
    changed = [ln for ln in diff.stdout.splitlines() if ln.strip()]
    if changed:
        shown = ", ".join(changed[:5]) + (" …" if len(changed) > 5 else "")
        return (
            False,
            f"generation paths changed since bundle commit {bundle_hash[:12]}...: "
            f"{shown} -- regenerate the bundle by re-running the pipeline and commit",
        )
    return True, f"bundle {bundle_hash[:12]}... is an ancestor of HEAD; no generation-path drift"


def _hp_note(hp_status: str) -> str:
    """Human-readable note about the hyperparameter-lock re-verification result."""
    if hp_status == "match":
        return "locked hyperparameters verified unchanged"
    return "locked hyperparameters source unverifiable on this checkout"


def _hyperparameters_drift(prov_doc: dict[str, Any] | None) -> tuple[str, str]:
    """Re-verify the bundle's locked hyperparameters against their on-disk source.

    The bundle is self-describing: training.config.hyperparameters_source records the
    mode and path the locked params came from. In ``config_locked_params`` mode the
    source is the git-tracked canonical artifact (CANONICAL_HP_SOURCE_REL), carrying a
    ``hyperparameters`` block re-readable even on a fresh clone. We compare that block to
    the bundle's recorded training.config.hyperparameters.

    The comparison is pinned to the CANONICAL committed source, NOT to whatever path the
    bundle names: a forged provenance.json could otherwise point ``path`` at a file
    crafted to match its own forged recorded hyperparameters. A bundle that declares a
    non-canonical source path is treated conservatively as "unverifiable" -- the checker
    never reads an arbitrary bundle-named path and never raises a spurious FAIL.

    Returns (status, detail) where status is:
      "match"        -- canonical-source hyperparameters equal the bundle's locked set.
      "drift"        -- they differ -> a retrained model WOULD differ (caller FAILs).
      "unverifiable" -- no recorded params, an absent/unreadable canonical source, a
                        non-canonical source path, or a source mode that is not a
                        re-readable file here (default_params / cli_params_override) ->
                        caller stays conservative (no clearing).
    """
    training = prov_doc.get("training") if isinstance(prov_doc, dict) else None
    config = training.get("config") if isinstance(training, dict) else None
    if not isinstance(config, dict):
        return "unverifiable", "no training.config block in provenance"
    recorded = config.get("hyperparameters")
    source = config.get("hyperparameters_source")
    if not isinstance(recorded, dict) or not isinstance(source, dict):
        return "unverifiable", "no recorded hyperparameters / source in provenance"
    mode = source.get("mode")
    path = source.get("path")
    if mode != "config_locked_params" or not isinstance(path, str) or not path:
        return "unverifiable", f"hyperparameters source mode={mode!r} not re-readable here"
    # Pin to the canonical committed artifact; a bundle naming any other path is not
    # trusted to choose the comparison target (see docstring).
    canonical = PACKAGE_ROOT / CANONICAL_HP_SOURCE_REL
    declared = Path(path)
    if not declared.is_absolute():
        declared = PACKAGE_ROOT / declared
    if declared.resolve() != canonical.resolve():
        return "unverifiable", (
            f"hyperparameters source {path!r} is not the canonical {CANONICAL_HP_SOURCE_REL}"
        )
    if not canonical.exists():
        return "unverifiable", f"{CANONICAL_HP_SOURCE_REL} absent (gitignored / not pulled)"
    src_doc = _load_json(canonical)
    current = src_doc.get("hyperparameters") if isinstance(src_doc, dict) else None
    if not isinstance(current, dict):
        return "unverifiable", f"{CANONICAL_HP_SOURCE_REL} has no hyperparameters block"
    if current == recorded:
        return "match", f"hyperparameters match {CANONICAL_HP_SOURCE_REL}"
    return "drift", f"hyperparameters in {CANONICAL_HP_SOURCE_REL} differ from the bundle's locked set"


def _model_freshness(
    bundle_hash: str | None,
    head_hash: str | None,
    *,
    prov_doc: dict[str, Any] | None,
) -> tuple[str, str]:
    """Decide whether output/<variant>/model.onnx could be stale w.r.t. HEAD.

    Scope: this answers "could model.onnx itself be stale?" -- it does NOT police the
    other bundle members. config.json integrity is covered by its own checksum (verified
    by the caller) and the human-readable README by check-docs.

    Returns (verdict, detail) where verdict is one of:
      "input_drift" -- a content-hashed model INPUT (the locked hyperparameters) provably
                       changed vs the bundle's recorded set, so model.onnx WOULD differ
                       (ALWAYS FAIL; checked first on every path -- see below).
      "fresh"       -- the model-producing code, dependencies, AND the re-verifiable
                       inputs are unchanged between the bundle commit and HEAD (PASS).
      "code_drift"  -- model-producing CODE (07_train / 11_export_onnx / lib), a config,
                       or the dependency lockfile changed and no content hash can prove
                       the trained result is identical (WARN by default, FAIL under
                       --strict-head).

    The hyperparameter-input check runs FIRST and unconditionally: it is content-only and
    git-independent (artifacts/tuned_params.json is git-tracked, so it is present even on a
    shallow/tarball clone), so it must NOT be gated behind the git-relationship verdict --
    otherwise a moved hyperparameter could slip through as "fresh" on a clone or at HEAD.

    For the code/dependency signal this ignores pipeline stages that cannot affect
    model.onnx (MODEL_IRRELEVANT_STAGES) and drops the tuning stage when the hyperparameter
    lock is confirmed unchanged, leaving only genuine training/export/config/dependency
    drift to warn about. On a verifiable (ancestor) path it never downgrades such a change
    to "fresh"; git-unverifiable states (shallow clone / no git / diff failure) degrade the
    *code* signal to "fresh" and rely on the content-sha chain, but the hyperparameter-input
    FAIL above still applies. Determinants no file diff can capture -- the resolved XGBoost
    thread count (xgb_n_jobs) and the cross-thread non-determinism of tree_method=hist --
    are documented out-of-band (docs/pipeline.md) and are not covered here.
    """
    # A provable model-INPUT change is ALWAYS a FAIL, checked before any git short-circuit.
    hp_status, hp_detail = _hyperparameters_drift(prov_doc)
    if hp_status == "drift":
        return "input_drift", (
            f"locked hyperparameters changed vs the bundle's recorded set: {hp_detail} -- "
            "model.onnx WOULD differ; a retrain is required (not a value-preserving refactor)"
        )

    kind, detail = _commit_relationship(bundle_hash, head_hash)
    if kind in ("unverifiable", "at_head"):
        return "fresh", detail
    if kind == "not_ancestor":
        # Divergent / rewritten history -- we cannot reason about the diff; stay
        # conservative and treat it as code drift (WARN, or FAIL under --strict-head).
        return "code_drift", detail

    # kind == "ancestor": _commit_relationship guarantees both are real commit hashes;
    # inspect exactly what changed under the generation + dependency paths.
    assert bundle_hash is not None and head_hash is not None
    diff = _git(
        "diff", "--name-only", bundle_hash, head_hash, "--",
        *GENERATION_PATHS, *ENVIRONMENT_PATHS,
    )
    if diff is None or diff.returncode != 0:
        return "fresh", "gen-path diff unavailable (git-unverifiable)"
    changed = [ln for ln in diff.stdout.splitlines() if ln.strip()]
    relevant = [f for f in changed if f not in MODEL_IRRELEVANT_STAGES]
    if hp_status == "match":
        relevant = [f for f in relevant if f != HP_PRODUCING_STAGE]

    bh = bundle_hash[:12]
    if not relevant:
        if changed:
            return "fresh", (
                f"bundle {bh}... is an ancestor of HEAD; model-producing code unchanged "
                "(changed generation files cannot affect model.onnx -- post-hoc/eval/figure "
                f"stages and/or the tuner; {_hp_note(hp_status)})"
            )
        return "fresh", f"bundle {bh}... is an ancestor of HEAD; no generation-path drift"

    shown = ", ".join(relevant[:5]) + (" …" if len(relevant) > 5 else "")
    return "code_drift", (
        f"model-producing code or dependencies changed since bundle commit {bh}...: {shown}; "
        f"{_hp_note(hp_status)} -- model.onnx is affected ONLY if this change altered the "
        "trained trees or ONNX bytes (a value-preserving refactor needs no retrain). To "
        "confirm, retrain at the recorded xgb_n_jobs and compare model.onnx."
    )


class ProvenanceChecker:
    """Accumulates check results and prints a summary."""

    def __init__(self) -> None:
        self.results: list[tuple[str, str, str]] = []  # (status, label, detail)

    def passed(self, label: str, detail: str = "") -> None:
        self.results.append(("PASS", label, detail))

    def failed(self, label: str, detail: str = "") -> None:
        self.results.append(("FAIL", label, detail))

    def skipped(self, label: str, detail: str = "") -> None:
        self.results.append(("SKIP", label, detail))

    def warned(self, label: str, detail: str = "") -> None:
        """A non-fatal advisory (e.g. a stale-but-internally-consistent bundle)."""
        self.results.append(("WARN", label, detail))

    def print_summary(self) -> int:
        """Print results and return count of failures (WARN is not a failure)."""
        print("\nProvenance verification")
        print("=" * 55)
        for status, label, detail in self.results:
            suffix = f" ({detail})" if detail else ""
            print(f"[{status}] {label}{suffix}")
        print("=" * 55)
        n_pass = sum(1 for s, _, _ in self.results if s == "PASS")
        n_fail = sum(1 for s, _, _ in self.results if s == "FAIL")
        n_warn = sum(1 for s, _, _ in self.results if s == "WARN")
        n_skip = sum(1 for s, _, _ in self.results if s == "SKIP")
        total = len(self.results)
        print(
            f"{total} checks: {n_pass} passed, {n_fail} failed, "
            f"{n_warn} warned, {n_skip} skipped"
        )
        return n_fail


def check_norms_lock(checker: ProvenanceChecker) -> str | None:
    """Check A: Norms lock file exists and is valid JSON."""
    path = PACKAGE_ROOT / "artifacts" / "ipip_bffm_norms.json"
    if not path.exists():
        checker.failed("Norms lock", "file not found")
        return None

    payload = _load_json(path)
    if payload is None:
        checker.failed("Norms lock", "invalid JSON")
        return None

    if "schema_version" not in payload:
        checker.failed("Norms lock", "missing schema_version")
        return None
    if "norms" not in payload:
        checker.failed("Norms lock", "missing norms key")
        return None
    if "n_respondents" not in payload:
        checker.failed("Norms lock", "missing n_respondents")
        return None

    sha = file_sha256(path)
    checker.passed("Norms lock: artifacts/ipip_bffm_norms.json", f"sha256: {sha[:12]}...")
    return sha


def check_norms_meta(checker: ProvenanceChecker, norms_sha: str | None) -> None:
    """Check B: Norms meta sidecar consistency with lock."""
    path = PACKAGE_ROOT / "artifacts" / "ipip_bffm_norms.meta.json"
    if not path.exists():
        checker.skipped("Norms meta sidecar", "not populated (sidecar is gitignored / not pulled on a clone)")
        return

    payload = _load_json(path)
    if payload is None:
        checker.failed("Norms meta sidecar", "invalid JSON")
        return

    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        checker.failed("Norms meta sidecar", "missing provenance block")
        return

    if norms_sha is None:
        checker.skipped("Norms meta sidecar", "norms lock not available for comparison")
        return

    meta_lock_sha = provenance.get("norms_lock_sha256", "")
    if meta_lock_sha.lower() != norms_sha.lower():
        checker.failed(
            "Norms meta sidecar",
            f"norms_lock_sha256 mismatch: meta={meta_lock_sha[:12]}... vs lock={norms_sha[:12]}...",
        )
        return

    expected_snapshot = f"norms_sha256:{norms_sha}"
    actual_snapshot = provenance.get("data_snapshot_id", "")
    if actual_snapshot != expected_snapshot:
        checker.failed(
            "Norms meta sidecar",
            f"data_snapshot_id mismatch: {actual_snapshot} vs {expected_snapshot}",
        )
        return

    checker.passed("Norms meta sidecar", "consistent with lock")


def check_research_summary(
    checker: ProvenanceChecker,
    norms_sha: str | None,
    *,
    head_hash: str | None = None,
    reference_only: bool = False,
) -> None:
    """Check C: research_summary.json has top-level provenance.

    A norms-reference mismatch is a hard FAIL only when the summary is at HEAD;
    a summary that predates HEAD legitimately references older norms -> WARN
    (so a stale-but-committed summary does not break ``make provenance-check``).

    When ``reference_only`` is set, the all-variants completeness gate is scoped to
    the reference variant only (a single-variant pipeline run produces no ablation
    bundles), so a leftover incomplete ablation key cannot fail the check.
    """
    path = PACKAGE_ROOT / "artifacts" / "research_summary.json"
    if not path.exists():
        checker.skipped("research_summary.json", "not populated (run `make research-summary`)")
        return

    payload = _load_json(path)
    if payload is None:
        checker.failed("research_summary.json", "invalid JSON")
        return

    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        checker.failed("research_summary.json", "missing top-level provenance key")
        return

    # A summary built --reference-only contains only the reference variant, so the
    # all-variants completeness gate below cannot detect the absent ablations. Fail
    # loudly if such a summary is checked WITHOUT --reference-only (older summaries
    # predate this provenance key -> bool(None) is False -> no effect on them).
    if bool(provenance.get("reference_only")) and not reference_only:
        checker.failed(
            "research_summary.json",
            "summary was built --reference-only (single variant); re-run the check "
            "with --reference-only, or rebuild the full-variant summary",
        )
        return

    summary_git_hash = provenance.get("git_hash")
    summary_git_hash = summary_git_hash if isinstance(summary_git_hash, str) and summary_git_hash else None
    # Fresh = at HEAD, OR an ancestor of HEAD with no generation-path drift (so a
    # release commit on top of the generation commit is still "current").
    fresh, _fresh_detail = _release_fresh(summary_git_hash, head_hash, strict_head=False)

    if norms_sha is not None:
        input_artifacts = provenance.get("input_artifacts", {})
        if isinstance(input_artifacts, dict):
            summary_norms_sha = input_artifacts.get("norms_lock_sha256", "")
            if not summary_norms_sha:
                checker.failed(
                    "research_summary.json",
                    "missing norms_lock_sha256 in provenance.input_artifacts",
                )
                return
            if summary_norms_sha.lower() != norms_sha.lower():
                detail = (
                    f"norms_lock_sha256 mismatch: {summary_norms_sha[:12]}... "
                    f"vs {norms_sha[:12]}..."
                )
                if fresh:
                    checker.failed("research_summary.json", detail)
                else:
                    checker.warned(
                        "research_summary.json",
                        detail + " (summary predates HEAD; regenerate in the next run)",
                    )
                return

    # Check all variants complete
    variants = payload.get("variants", {})
    if isinstance(variants, dict):
        incomplete = [
            v for v, data in variants.items()
            if (not reference_only or v == REFERENCE_VARIANT)
            and isinstance(data, dict) and not data.get("status", {}).get("complete", False)
        ]
        if incomplete:
            checker.failed(
                "research_summary.json",
                f"incomplete variants: {', '.join(incomplete)}",
            )
            return

    checker.passed("research_summary.json", "top-level provenance, norms reference valid")


def check_output_bundle(
    checker: ProvenanceChecker,
    norms_sha: str | None,
    *,
    head_hash: str | None = None,
    strict_head: bool = False,
) -> None:
    """Check D: per-variant output/<variant>/ provenance bundles.

    Iterates the canonical variant registry (lib.constants.VARIANTS) -- NOT a
    directory scan -- so a stale leftover dir (e.g. a removed ablation) is never
    validated. HEAD-staleness and cross-variant git_hash disagreement are
    WARNINGS by default and only FAIL under ``strict_head`` (so this passes on the
    currently-committed bundle, which predates HEAD). Checksum mismatches and
    intra-bundle git_hash disagreement always FAIL.
    """
    if head_hash is None:
        head_hash = _detect_git_hash()
    head_known = bool(head_hash) and head_hash != "unknown"

    output_root = PACKAGE_ROOT / "output"
    if not output_root.is_dir():
        checker.skipped("output/", "not populated (run `make export-all`)")
        return

    variant_dirs = [
        (name, output_root / name)
        for name in VARIANTS
        if (output_root / name / "config.json").is_file()
    ]
    if not variant_dirs:
        checker.skipped("output/", "no variant bundles (run `make export-all`)")
        return

    seen_bundle_hashes: set[str] = set()

    for variant, vdir in variant_dirs:
        label = f"output/{variant}"
        config_path = vdir / "config.json"
        prov_path = vdir / "provenance.json"
        if not prov_path.exists():
            checker.failed(label, "missing provenance.json")
            continue
        prov_doc = _load_json(prov_path)
        config_doc = _load_json(config_path)
        if prov_doc is None or config_doc is None:
            checker.failed(label, "invalid JSON in config.json/provenance.json")
            continue

        export = prov_doc.get("export")
        training = prov_doc.get("training")
        if not isinstance(export, dict) or not isinstance(training, dict):
            checker.failed(label, "provenance.json missing export/training block")
            continue

        bundle_hash = export.get("git_hash")
        bundle_hash_str = bundle_hash if isinstance(bundle_hash, str) and bundle_hash else None
        # Generic freshness (any generation-path drift) gates the norms-snapshot check
        # below: a stale bundle legitimately predates the current norms. The narrower
        # model-freshness verdict (does model.onnx itself risk staleness?) is computed
        # separately for the HEAD-freshness line further down.
        fresh, _fresh_detail = _release_fresh(bundle_hash_str, head_hash, strict_head=strict_head)

        # Norms snapshot: hard FAIL only when the bundle is at HEAD; otherwise a
        # stale bundle legitimately predates the current norms -> WARN.
        if norms_sha is not None:
            snapshot_id = export.get("data_snapshot_id", "")
            expected_snapshot = f"norms_sha256:{norms_sha}"
            if snapshot_id != expected_snapshot:
                if fresh:
                    checker.failed(
                        label,
                        f"data_snapshot_id mismatch: {snapshot_id!r} vs {expected_snapshot!r}",
                    )
                    continue
                checker.warned(label, "data_snapshot_id stale (bundle predates HEAD)")

        # Checksum verification (always hard; passes on the committed bundle). model.onnx
        # is gitignored, so on a fresh clone it is absent and its bytes cannot be checked --
        # track that so the bundle line reports honestly (config verified; model bytes not
        # checked) rather than claiming a blanket "checksums verified".
        model_verified = False
        artifacts = prov_doc.get("artifacts", {})
        if isinstance(artifacts, dict):
            config_sha = artifacts.get("config_json_sha256")
            if not isinstance(config_sha, str) or not config_sha:
                checker.failed(label, "artifacts block missing config_json_sha256")
                continue
            if file_sha256(config_path).lower() != config_sha.lower():
                checker.failed(label, "config_json_sha256 does not match config.json")
                continue
            model_name = config_doc.get("model_file", "model.onnx")
            model_path = vdir / (model_name if isinstance(model_name, str) else "model.onnx")
            model_sha = artifacts.get("model_onnx_sha256")
            if not isinstance(model_sha, str) or not model_sha:
                checker.failed(label, "artifacts block missing model_onnx_sha256")
                continue
            if model_path.exists():
                if file_sha256(model_path).lower() != model_sha.lower():
                    checker.failed(label, "model_onnx_sha256 does not match model.onnx")
                    continue
                model_verified = True

        # item_info lock: the bundle records the SHA-256 of the item ranking the model
        # was trained against (training.data.item_info_sha256). The on-disk file is
        # gitignored, so on a fresh clone it is absent -> SKIP. When present, its SHA
        # must equal the recorded lock; any difference means the working-tree ranking no
        # longer matches what the model was trained on -> FAIL.
        data_block = training.get("data") if isinstance(training, dict) else None
        if isinstance(data_block, dict):
            locked_item_info_sha = data_block.get("item_info_sha256")
            item_info_rel = data_block.get("item_info_path")
            if (
                isinstance(locked_item_info_sha, str)
                and locked_item_info_sha
                and isinstance(item_info_rel, str)
                and item_info_rel
            ):
                item_info_path = Path(item_info_rel)
                if not item_info_path.is_absolute():
                    item_info_path = PACKAGE_ROOT / item_info_path
                if not item_info_path.exists():
                    checker.skipped(
                        f"{label} item_info lock",
                        "item_info.json absent (gitignored / not pulled on a clone)",
                    )
                else:
                    disk_item_info_sha = file_sha256(item_info_path)
                    if disk_item_info_sha.lower() == locked_item_info_sha.lower():
                        checker.passed(
                            f"{label} item_info lock",
                            f"on-disk matches lock ({locked_item_info_sha[:12]}...)",
                        )
                    else:
                        checker.failed(
                            f"{label} item_info lock",
                            f"on-disk {disk_item_info_sha[:12]}... != recorded lock "
                            f"{locked_item_info_sha[:12]}...: the working-tree item ranking "
                            "differs from the one the model was trained against (re-run the "
                            "pipeline and commit, or restore the recorded item_info.json)",
                        )
                        continue

        # Intra-bundle git_hash agreement: export / training.provenance /
        # config.provenance must agree regardless of staleness (always FAIL).
        # (The nested tune payload_provenance hash legitimately differs and is
        # deliberately excluded.)
        intra: list[tuple[str, str]] = []
        if bundle_hash_str:
            intra.append(("export", bundle_hash_str))
        training_prov = training.get("provenance")
        if isinstance(training_prov, dict):
            th = training_prov.get("git_hash")
            if isinstance(th, str) and th:
                intra.append(("training.provenance", th))
        config_prov = config_doc.get("provenance")
        if isinstance(config_prov, dict):
            ch = config_prov.get("git_hash")
            if isinstance(ch, str) and ch:
                intra.append(("config.provenance", ch))
        if len({h for _, h in intra}) > 1:
            detail = ", ".join(f"{src}={h[:12]}..." for src, h in intra)
            checker.failed(f"{label} git_hash agreement", f"intra-bundle disagreement: {detail}")
            continue

        # HEAD freshness, scoped to the trained model (see _model_freshness): a bundle
        # at HEAD -- or an ancestor of HEAD whose model-producing code AND hash-locked
        # inputs are unchanged -- is FRESH and PASSes even under --strict-head. Genuine
        # training/export code drift is WARN by default, FAIL under --strict-head; a
        # provable model-INPUT change (locked hyperparameters) always FAILs. Changes
        # confined to post-hoc/eval/figure/upload stages, or to the tuner when the
        # hyperparameter lock is unchanged, do NOT flag the model stale. Git-unverifiable
        # (shallow clone / no git) degrades to PASS.
        if bundle_hash_str and head_known:
            verdict, model_detail = _model_freshness(
                bundle_hash_str, head_hash, prov_doc=prov_doc
            )
            if verdict == "fresh":
                checker.passed(f"{label} HEAD freshness", model_detail)
            elif verdict == "input_drift" or strict_head:
                checker.failed(f"{label} HEAD freshness", model_detail)
            else:
                checker.warned(f"{label} HEAD freshness", model_detail)

        if bundle_hash_str:
            seen_bundle_hashes.add(bundle_hash_str)
        if model_verified:
            checker.passed(
                f"{label} bundle", "provenance.json valid, config + model checksums verified"
            )
        else:
            checker.passed(
                f"{label} bundle",
                "provenance.json valid, config checksum verified; model.onnx absent "
                "(gitignored / not pulled) -- model bytes NOT checked",
            )

    # Cross-variant agreement: all variants should be exported from one commit.
    if len(seen_bundle_hashes) > 1:
        msg = "variants exported from different commits: " + ", ".join(
            sorted(h[:12] + "..." for h in seen_bundle_hashes)
        )
        if strict_head:
            checker.failed("output/ cross-variant git_hash", msg)
        else:
            checker.warned("output/ cross-variant git_hash", msg)


def check_figures_manifest(checker: ProvenanceChecker) -> None:
    """Check E: figures/manifest.json."""
    path = PACKAGE_ROOT / "figures" / "manifest.json"
    if not path.exists():
        checker.skipped("figures/manifest.json", "not populated (run `make figures`)")
        return

    payload = _load_json(path)
    if payload is None:
        checker.failed("figures/manifest.json", "invalid JSON")
        return

    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        checker.failed("figures/manifest.json", "missing provenance key")
        return

    # Verify source artifact checksums (soft fail if files are gitignored)
    source_artifacts = payload.get("source_artifacts", {})
    if isinstance(source_artifacts, dict):
        for label, info in source_artifacts.items():
            if not isinstance(info, dict):
                continue
            artifact_path_str = info.get("path")
            expected_sha = info.get("sha256")
            if not isinstance(artifact_path_str, str) or not isinstance(expected_sha, str):
                continue
            artifact_path = Path(artifact_path_str)
            if not artifact_path.is_absolute():
                artifact_path = PACKAGE_ROOT / artifact_path
            if not artifact_path.exists():
                # Soft fail: source artifacts may be gitignored
                continue
            actual_sha = file_sha256(artifact_path)
            if actual_sha.lower() != expected_sha.lower():
                checker.failed(
                    "figures/manifest.json",
                    f"source artifact {label} SHA-256 mismatch",
                )
                return

    # Verify figure OUTPUT checksums (A5.5) for any rendered files present.
    # Degrades gracefully: figure PNG/PDFs are gitignored (absent in CI) and
    # older manifests predate the per-figure sha256 key -> skip, never fail.
    figures = payload.get("figures", [])
    if isinstance(figures, list):
        for entry in figures:
            if not isinstance(entry, dict):
                continue
            filename = entry.get("filename")
            sha_map = entry.get("sha256")
            if not isinstance(filename, str) or not isinstance(sha_map, dict):
                continue
            for fmt, expected in sha_map.items():
                if not isinstance(fmt, str) or not isinstance(expected, str):
                    continue
                fig_path = PACKAGE_ROOT / "figures" / f"{filename}.{fmt}"
                if not fig_path.exists():
                    continue
                if file_sha256(fig_path).lower() != expected.lower():
                    checker.failed(
                        "figures/manifest.json",
                        f"figure {filename}.{fmt} SHA-256 mismatch",
                    )
                    return

    checker.passed("figures/manifest.json", "provenance valid, source checksums verified")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the committed artifact provenance tree."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 on any failure (for CI).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Treat SKIP as failure (require all artifacts populated).",
    )
    parser.add_argument(
        "--strict-head",
        action="store_true",
        help=(
            "Treat an artifact git_hash != current HEAD (and cross-variant "
            "git_hash disagreement) as a failure instead of a warning. Use once "
            "the published bundle has been regenerated at HEAD."
        ),
    )
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help=(
            "Scope the research_summary completeness gate to the reference variant "
            "only (for a single-variant pipeline run that produced no ablation bundles)."
        ),
    )
    args = parser.parse_args()

    checker = ProvenanceChecker()
    head_hash = _detect_git_hash()

    # Auto-detect reference-only from the published summary so a fresh clone need
    # not pass the flag (the bundle self-describes how it was built).
    reference_only = args.reference_only
    if not reference_only:
        rs = _load_json(PACKAGE_ROOT / "artifacts" / "research_summary.json")
        reference_only = bool(
            isinstance(rs, dict) and rs.get("provenance", {}).get("reference_only")
        )

    norms_sha = check_norms_lock(checker)
    check_norms_meta(checker, norms_sha)
    check_research_summary(checker, norms_sha, head_hash=head_hash, reference_only=reference_only)
    check_output_bundle(checker, norms_sha, head_hash=head_hash, strict_head=args.strict_head)
    check_figures_manifest(checker)

    n_fail = checker.print_summary()
    n_skip = sum(1 for s, _, _ in checker.results if s == "SKIP")

    if (args.strict or args.strict_head) and n_fail > 0:
        return 1
    if args.full and (n_fail > 0 or n_skip > 0):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
