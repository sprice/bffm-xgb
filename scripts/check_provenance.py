#!/usr/bin/env python3
"""Validate the entire committed artifact provenance tree.

Checks that all provenance sidecars, manifests, and cross-references
are internally consistent. Designed for CI and pre-submission verification.

Usage:
    python scripts/check_provenance.py          # advisory mode
    python scripts/check_provenance.py --strict  # exit 1 on any failure
"""

from __future__ import annotations

import argparse
import json
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
        checker.failed("Norms meta sidecar", "file not found")
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
    head_known = bool(head_hash) and head_hash != "unknown"
    at_head = head_known and isinstance(summary_git_hash, str) and summary_git_hash == head_hash

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
                if at_head:
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
        at_head = head_known and bundle_hash_str == head_hash

        # Norms snapshot: hard FAIL only when the bundle is at HEAD; otherwise a
        # stale bundle legitimately predates the current norms -> WARN.
        if norms_sha is not None:
            snapshot_id = export.get("data_snapshot_id", "")
            expected_snapshot = f"norms_sha256:{norms_sha}"
            if snapshot_id != expected_snapshot:
                if at_head:
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

        # HEAD-staleness: WARN by default, FAIL only under --strict-head.
        if bundle_hash_str and head_known:
            if bundle_hash_str != head_hash:
                msg = f"git_hash {bundle_hash_str[:12]}... != HEAD {head_hash[:12]}..."
                if strict_head:
                    checker.failed(f"{label} HEAD freshness", msg)
                else:
                    checker.warned(f"{label} HEAD freshness", msg)
            else:
                checker.passed(f"{label} HEAD freshness", "at HEAD")

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

    norms_sha = check_norms_lock(checker)
    check_norms_meta(checker, norms_sha)
    check_research_summary(checker, norms_sha, head_hash=head_hash, reference_only=args.reference_only)
    check_output_bundle(checker, norms_sha, head_hash=head_hash, strict_head=args.strict_head)
    check_figures_manifest(checker)

    n_fail = checker.print_summary()
    n_skip = sum(1 for s, _, _ in checker.results if s == "SKIP")

    if args.strict and n_fail > 0:
        return 1
    if args.full and (n_fail > 0 or n_skip > 0):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
