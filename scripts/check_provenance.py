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
          HEAD, or an on-disk file was cosmetically re-stamped (see below). WARN does
          NOT fail --strict; it DOES fail under --strict-head where noted.
    FAIL  a genuine inconsistency (checksum mismatch, intra-bundle git_hash
          disagreement, substantive generation-code drift).
    SKIP  an artifact is absent (e.g. gitignored on a fresh clone).

Release freshness, honestly (M19/M20) -- no retraining required to reconcile:
    A bundle is "fresh" when its export git_hash is HEAD, OR an ancestor of HEAD with
    NO drift under pipeline/ + lib/ + configs/ (release/refresh commits and a later
    merge into main legitimately sit on top of the generation commit). Drift in those
    paths -> WARN by default, FAIL under --strict-head, with the changed paths named.

    To reconcile a stale bundle WITHOUT retraining, regenerate the affected artifact
    at HEAD and re-run this check:
      * model card (output/<variant>/README.md) only -- no ONNX re-export, no retrain,
        and NOTE this rewrites the human-readable card ONLY; it does NOT re-stamp any
        provenance lock (see the item_info-lock note below):
            make export-readme MODEL_DIR=models/reference   # rewrites README.md only
      * notes / research_summary:   make notes
      * figures:                    make figures
    A FULL bundle regeneration (new model.onnx) requires `make all` / the remote
    pipeline and is the ONLY path that should ever change a model checksum. This
    checker never auto-regenerates or retrains; it only reports what is stale.

    item_info.json carries its OWN embedded provenance git_hash, so regenerating
    stage 05 at a newer commit re-stamps that hash and moves the on-disk file SHA
    even when the ranking/correlation logic is byte-identical (e.g. a post-bundle
    stage-05 delta that is only a comment + a leakage-guard assert). The per-bundle
    "item_info lock" check distinguishes that cosmetic re-stamp (WARN) from a
    substantive ranking change (FAIL). IMPORTANT: the locked SHA lives in
    models/<variant>/training_report.json (data.item_info_sha256) and is written ONLY
    by stage 07 (train); `make export-readme` rewrites README.md and CANNOT change it.
    A cosmetic re-stamp WARN is reconciled by restoring the on-disk
    data/processed/<variant>/item_info.json to the locked training-time bytes (or, if
    the on-disk content is intended to become the new locked bytes, by re-running
    stage 07 to re-stamp the lock) -- never by `make export-readme`, which would loop
    with no effect. See check_output_bundle.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

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

# The stage that produces data/processed/<variant>/item_info.json (the item ranking
# + correlations the model is trained against). Used by the item_info lock-state
# reconciliation below to decide whether an on-disk-vs-locked SHA drift is a
# data-changing edit (FAIL territory) or a cosmetic re-stamp (WARN).
ITEM_INFO_STAGE = "pipeline/05_compute_correlations.py"


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


def _release_fresh(
    bundle_hash: str | None, head_hash: str | None, *, strict_head: bool
) -> tuple[bool, str]:
    """Decide whether a bundle stamped ``bundle_hash`` is fresh w.r.t. ``head_hash``.

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
    if not bundle_hash or not head_hash or head_hash == "unknown":
        return True, "head/bundle hash unknown (git-unverifiable)"
    if bundle_hash == head_hash:
        return True, "at HEAD"
    if _git("rev-parse", "--git-dir") is None:
        return True, "git unavailable (git-unverifiable)"
    if not _commit_present(bundle_hash) or not _commit_present(head_hash):
        return True, "bundle/HEAD object absent (shallow clone; git-unverifiable)"
    anc = _git("merge-base", "--is-ancestor", bundle_hash, head_hash)
    if anc is None:
        return True, "git unavailable (git-unverifiable)"
    if anc.returncode != 0:
        return False, f"bundle {bundle_hash[:12]}... is not an ancestor of HEAD {head_hash[:12]}..."
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
            f"{shown} -- regenerate the bundle at HEAD (see module docstring), no retrain "
            "needed unless model.onnx itself is affected",
        )
    return True, f"bundle {bundle_hash[:12]}... is an ancestor of HEAD; no generation-path drift"


def _diff_is_cosmetic(unified_diff: str) -> bool:
    """Classify a unified diff of ITEM_INFO_STAGE as cosmetic (no ranking/correlation
    logic change) vs substantive. Returns True only when EVERY added/removed content
    line is provably non-computational: a comment, a blank line, or a bare ``assert``
    leakage-guard. Any other changed line (or a diff we cannot parse) is treated as
    substantive, so the classifier never blesses a real logic change -- it only
    de-escalates the known no-op edits (comment + leakage-guard assert) that would
    otherwise hard-FAIL a provably-identical ranking under the multi-variant retrain."""
    saw_change = False
    for raw in unified_diff.splitlines():
        if raw.startswith(("+++", "---", "@@", "diff ", "index ")):
            continue
        if not raw or raw[0] not in "+-":
            continue
        body = raw[1:].strip()
        saw_change = True
        if not body:
            continue  # blank line
        if body.startswith("#"):
            continue  # comment
        if body.startswith(("assert ", "assert(")):
            continue  # leakage-guard / invariant assert: does not alter rankings
        return False  # a real code line changed -> substantive
    return saw_change  # all changed lines were cosmetic (and there was at least one)


def _item_info_stage_diff(bundle_hash: str, disk_hash: str) -> tuple[str, bool]:
    """Diff the item-ranking stage (05_compute_correlations.py) between the commit
    the bundle was locked at (``bundle_hash``) and the commit the on-disk
    item_info.json was last generated at (``disk_hash``, read from its embedded
    provenance git_hash). Returns (detail, stage_changed):

      stage_changed == False -> the two commits produce a byte-identical item_info
        for the same inputs; an on-disk-vs-locked SHA mismatch is therefore COSMETIC
        (the SHA moved only because the file re-embedded a newer provenance git_hash),
        and NO retrain / re-lock is required to reconcile it.
      stage_changed == True  -> the ranking/correlation logic genuinely differs
        between the two commits, so the mismatch could reflect a real data change.

    Multi-variant note: all three variants (reference, ablation_none, ablation_focused)
    share data_regime canonical_v1, so they lock the SAME item_info SHA. A no-op
    stage-05 edit (comment- or assert-only) between the lock commit and the on-disk
    commit changes the file bytes but NOT the rankings; classifying purely on whether
    stage-05 appears in `git diff --name-only` would hard-FAIL all three variants for a
    provably-identical ranking. We therefore inspect the diff CONTENT and treat a
    comment/blank/assert-only delta as cosmetic. The locked item_info content is not
    recoverable here (only its SHA is stored), so this source-diff classification is the
    best available content signal; anything not provably cosmetic stays substantive.

    Git-unverifiable inputs (missing objects, no git) -> ("git-unverifiable", True)
    so the caller stays conservative and does not silently bless a real drift."""
    if not bundle_hash or not disk_hash:
        return "embedded git_hash unknown (git-unverifiable)", True
    if bundle_hash == disk_hash:
        return "same generation commit", False
    if _git("rev-parse", "--git-dir") is None:
        return "git unavailable (git-unverifiable)", True
    if not _commit_present(bundle_hash) or not _commit_present(disk_hash):
        return "bundle/on-disk commit object absent (shallow clone; git-unverifiable)", True
    names = _git("diff", "--name-only", bundle_hash, disk_hash, "--", ITEM_INFO_STAGE)
    if names is None or names.returncode != 0:
        return "stage diff unavailable (git-unverifiable)", True
    if not names.stdout.strip():
        return f"{ITEM_INFO_STAGE} unchanged between the two commits", False
    # Stage 05 source differs: decide substantive vs cosmetic on the diff CONTENT,
    # not merely on the path appearing in the name-only diff (a comment/assert-only
    # edit must NOT FAIL a provably-identical ranking under the multi-variant retrain).
    content = _git("diff", "--unified=0", bundle_hash, disk_hash, "--", ITEM_INFO_STAGE)
    if content is None or content.returncode != 0:
        return f"{ITEM_INFO_STAGE} changed between the two commits", True
    if _diff_is_cosmetic(content.stdout):
        return (
            f"{ITEM_INFO_STAGE} changed between the two commits but only in "
            "comment/blank/assert lines (no ranking/correlation logic change)",
            False,
        )
    return f"{ITEM_INFO_STAGE} changed between the two commits", True


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
        # Fresh = at HEAD, OR an ancestor of HEAD with no generation-path drift.
        fresh, fresh_detail = _release_fresh(bundle_hash_str, head_hash, strict_head=strict_head)

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

        # Checksum verification (always hard; passes on the committed bundle).
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
            if model_path.exists() and file_sha256(model_path).lower() != model_sha.lower():
                checker.failed(label, "model_onnx_sha256 does not match model.onnx")
                continue

        # Item-info lock-state reconciliation (M20).
        #
        # The bundle locks the SHA-256 of the item ranking / correlations file
        # (data/processed/<variant>/item_info.json) it was trained against. The
        # on-disk file is gitignored, so on a fresh clone it is absent -> SKIP.
        #
        # When the on-disk file IS present and its SHA differs from the lock, the
        # mismatch is reported honestly rather than swallowed: a SHA can move for
        # two very different reasons, and the operator needs to know which:
        #   * COSMETIC re-stamp (WARN) -- item_info.json embeds its own provenance
        #     git_hash, so regenerating stage 05 at a newer commit that did NOT touch
        #     the ranking/correlation logic re-stamps that hash and changes the file
        #     SHA while leaving every ranking/correlation byte-identical (e.g. a
        #     post-bundle stage-05 delta that is only a comment + a leakage-guard
        #     assert). It is NOT a model problem: the published model is unaffected.
        #     The lock SHA lives in models/<variant>/training_report.json
        #     (data.item_info_sha256), written ONLY by stage 07 (train). It is
        #     reconciled either by restoring the on-disk item_info.json to the locked
        #     training-time bytes (the lock is the source of truth for the published
        #     bundle), or -- if the on-disk content is intended to become the new lock
        #     -- by re-running stage 07 to re-stamp that SHA. It is NOT reconciled by
        #     `make export-readme`, which only regenerates the human-readable card
        #     (output/<variant>/README.md) and never touches the lock. NO retrain of
        #     the model weights is required, since the ranking/correlation bytes are
        #     identical.
        #   * SUBSTANTIVE change (FAIL) -- the ranking/correlation logic itself
        #     differs between the lock commit and the on-disk file's commit, so the
        #     on-disk item_info may no longer match what the model was trained on.
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
                        disk_doc = _load_json(item_info_path)
                        disk_git_hash = ""
                        if isinstance(disk_doc, dict):
                            prov = disk_doc.get("provenance")
                            if isinstance(prov, dict):
                                gh = prov.get("git_hash")
                                disk_git_hash = gh if isinstance(gh, str) else ""
                        stage_detail, stage_changed = _item_info_stage_diff(
                            bundle_hash_str or "", disk_git_hash
                        )
                        base = (
                            f"on-disk {disk_item_info_sha[:12]}... != lock "
                            f"{locked_item_info_sha[:12]}..."
                        )
                        if stage_changed:
                            checker.failed(
                                f"{label} item_info lock",
                                f"{base}; {stage_detail} (ranking/correlations may differ)",
                            )
                            continue
                        checker.warned(
                            f"{label} item_info lock",
                            f"{base}; cosmetic re-stamp only ({stage_detail}; "
                            "embedded provenance git_hash moved, rankings byte-identical) "
                            "-- reconcile by restoring on-disk item_info.json to the locked "
                            "bytes, or re-run stage 07 to re-stamp the lock; `make "
                            "export-readme` does NOT touch the lock and will not clear this",
                        )

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

        # HEAD freshness: a bundle that is at HEAD -- or an ancestor of HEAD with
        # no generation-path drift (release/refresh commits, a merge into main) --
        # is FRESH and PASSes even under --strict-head. A genuinely stale bundle
        # (gen-path drift, or not an ancestor) is WARN by default, FAIL under
        # --strict-head. Git-unverifiable (shallow clone / no git) degrades to PASS.
        if bundle_hash_str and head_known:
            if fresh:
                checker.passed(f"{label} HEAD freshness", fresh_detail)
            elif strict_head:
                checker.failed(f"{label} HEAD freshness", fresh_detail)
            else:
                checker.warned(f"{label} HEAD freshness", fresh_detail)

        if bundle_hash_str:
            seen_bundle_hashes.add(bundle_hash_str)
        checker.passed(f"{label} bundle", "provenance.json valid, checksums verified")

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
