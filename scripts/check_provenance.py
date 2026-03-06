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

from lib.provenance import file_sha256


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

    def print_summary(self) -> int:
        """Print results and return count of failures."""
        print("\nProvenance verification")
        print("=" * 55)
        for status, label, detail in self.results:
            suffix = f" ({detail})" if detail else ""
            print(f"[{status}] {label}{suffix}")
        print("=" * 55)
        n_pass = sum(1 for s, _, _ in self.results if s == "PASS")
        n_fail = sum(1 for s, _, _ in self.results if s == "FAIL")
        n_skip = sum(1 for s, _, _ in self.results if s == "SKIP")
        total = len(self.results)
        print(f"{total} checks: {n_pass} passed, {n_fail} failed, {n_skip} skipped")
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


def check_research_summary(checker: ProvenanceChecker, norms_sha: str | None) -> None:
    """Check C: research_summary.json has top-level provenance."""
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
                checker.failed(
                    "research_summary.json",
                    f"norms_lock_sha256 mismatch: {summary_norms_sha[:12]}... vs {norms_sha[:12]}...",
                )
                return

    # Check all variants complete
    variants = payload.get("variants", {})
    if isinstance(variants, dict):
        incomplete = [
            v for v, data in variants.items()
            if isinstance(data, dict) and not data.get("status", {}).get("complete", False)
        ]
        if incomplete:
            checker.failed(
                "research_summary.json",
                f"incomplete variants: {', '.join(incomplete)}",
            )
            return

    checker.passed("research_summary.json", "top-level provenance, norms reference valid")


def check_output_bundle(checker: ProvenanceChecker, norms_sha: str | None) -> None:
    """Check D: output/ bundle provenance.json."""
    config_path = PACKAGE_ROOT / "output" / "config.json"
    if not config_path.exists():
        checker.skipped("output/", "not populated (run `make export`)")
        return

    prov_path = PACKAGE_ROOT / "output" / "provenance.json"
    if not prov_path.exists():
        checker.failed("output/", "missing provenance.json")
        return

    prov_doc = _load_json(prov_path)
    if prov_doc is None:
        checker.failed("output/provenance.json", "invalid JSON")
        return

    export = prov_doc.get("export")
    if not isinstance(export, dict):
        checker.failed("output/provenance.json", "missing export block")
        return

    training = prov_doc.get("training")
    if not isinstance(training, dict):
        checker.failed("output/provenance.json", "missing training block")
        return

    # Check norms reference
    if norms_sha is not None:
        snapshot_id = export.get("data_snapshot_id", "")
        expected_snapshot = f"norms_sha256:{norms_sha}"
        if not snapshot_id or snapshot_id != expected_snapshot:
            checker.failed(
                "output/provenance.json",
                f"data_snapshot_id mismatch: {snapshot_id!r} vs {expected_snapshot!r}",
            )
            return

    # Verify artifact checksums
    artifacts = prov_doc.get("artifacts", {})
    if isinstance(artifacts, dict):
        config_sha = artifacts.get("config_json_sha256")
        if not isinstance(config_sha, str) or not config_sha:
            checker.failed(
                "output/provenance.json",
                "artifacts block missing config_json_sha256",
            )
            return
        actual_config_sha = file_sha256(config_path)
        if actual_config_sha.lower() != config_sha.lower():
            checker.failed(
                "output/provenance.json",
                "config_json_sha256 does not match actual config.json",
            )
            return

        model_path = PACKAGE_ROOT / "output" / "model.onnx"
        model_sha = artifacts.get("model_onnx_sha256")
        if not isinstance(model_sha, str) or not model_sha:
            checker.failed(
                "output/provenance.json",
                "artifacts block missing model_onnx_sha256",
            )
            return
        if model_path.exists():
            actual_model_sha = file_sha256(model_path)
            if actual_model_sha.lower() != model_sha.lower():
                checker.failed(
                    "output/provenance.json",
                    "model_onnx_sha256 does not match actual model.onnx",
                )
                return

    checker.passed("output/ bundle", "provenance.json valid, checksums verified")


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
    args = parser.parse_args()

    checker = ProvenanceChecker()

    norms_sha = check_norms_lock(checker)
    check_norms_meta(checker, norms_sha)
    check_research_summary(checker, norms_sha)
    check_output_bundle(checker, norms_sha)
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
