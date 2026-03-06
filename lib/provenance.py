#!/usr/bin/env python3
"""
Shared provenance metadata utilities for IPIP-BFFM pipeline scripts.

Provides helper functions to:
- Auto-detect the current git commit hash
- Build a standardised provenance dict for JSON artifacts
- Add common CLI flags (--preprocess-tag, --bootstrap-b,
  --bootstrap-seed) to any argparse parser

Every major JSON artifact produced by the pipeline should include a
`provenance` key whose value is the dict returned by build_provenance().
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
NORMS_PATH_ENV = "IPIP_BFFM_NORMS_PATH"


def relative_to_root(path: str | Path) -> str:
    """Convert a path to be relative to the package root.

    If the path is under PACKAGE_ROOT, returns the relative portion.
    Otherwise returns the original string unchanged.
    """
    try:
        return str(Path(path).resolve().relative_to(PACKAGE_ROOT))
    except ValueError:
        return str(path)


def sanitize_paths(obj: Any) -> Any:
    """Recursively convert absolute paths under PACKAGE_ROOT to relative paths.

    Walks a nested dict/list structure. Any string value that looks like an
    absolute path under PACKAGE_ROOT is converted to a relative path via
    :func:`relative_to_root`. Non-path strings, numbers, booleans, and None
    values are passed through unchanged.

    This is useful for sanitizing verbatim JSON blocks (e.g. training reports)
    that may contain absolute paths from earlier pipeline runs.
    """
    if isinstance(obj, dict):
        return {k: sanitize_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_paths(item) for item in obj]
    if isinstance(obj, str):
        # Only attempt conversion for strings that look like absolute paths
        # under PACKAGE_ROOT (avoid mangling hashes, URLs, etc.)
        pkg_root_str = str(PACKAGE_ROOT)
        if obj.startswith(pkg_root_str + os.sep) or obj == pkg_root_str:
            return relative_to_root(obj)
        return obj
    return obj


DATA_SNAPSHOT_ID_ENV = "IPIP_BFFM_DATA_SNAPSHOT_ID"


def _detect_git_hash() -> str:
    """Return the current HEAD commit hash, or 'unknown' if not in a git repo.

    Falls back to reading .git-hash (written by remote-push for headless envs).
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    git_hash_file = PACKAGE_ROOT / ".git-hash"
    if git_hash_file.exists():
        content = git_hash_file.read_text().strip()
        if content:
            return content

    return "unknown"


def file_sha256(path: Path) -> str:
    """Compute SHA-256 digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_norms_lock_path() -> Path:
    """Resolve norms lock path (supports IPIP_BFFM_NORMS_PATH override)."""
    override = str(os.environ.get(NORMS_PATH_ENV, "")).strip()
    if override:
        candidate = Path(override)
        if candidate.is_absolute():
            return candidate
        return PACKAGE_ROOT / candidate
    return PACKAGE_ROOT / "artifacts" / "ipip_bffm_norms.json"


def _resolve_norms_meta_path() -> Path:
    """Resolve norms metadata sidecar path from resolved norms lock path."""
    lock_path = _resolve_norms_lock_path()
    if lock_path.suffix:
        return lock_path.with_suffix(".meta.json")
    return lock_path.parent / f"{lock_path.name}.meta.json"


def _resolve_data_snapshot_id(
    *,
    args: argparse.Namespace | None,
    git_hash: str,
) -> str:
    """Resolve stable data snapshot identifier (not date-coupled)."""
    if args is not None and hasattr(args, "data_snapshot_id") and args.data_snapshot_id is not None:
        value = str(args.data_snapshot_id).strip()
        if value:
            return value

    env_value = str(os.environ.get(DATA_SNAPSHOT_ID_ENV, "")).strip()
    if env_value:
        return env_value

    norms_lock_path = _resolve_norms_lock_path()
    if norms_lock_path.exists():
        try:
            return f"norms_sha256:{file_sha256(norms_lock_path)}"
        except OSError:
            pass

    return f"git:{git_hash}"


def add_provenance_args(parser: argparse.ArgumentParser) -> None:
    """Append the standard provenance CLI flags to *parser*.

    Flags added
    -----------
    --preprocess-tag TAG         Override preprocessing version tag (default: git hash)
    --bootstrap-b N              Number of bootstrap resamples (scripts may
                                 also expose this under their own flag name)
    --bootstrap-seed N           Fixed RNG seed for bootstrap resampling
    """
    group = parser.add_argument_group("provenance", "Reproducibility / provenance metadata")
    group.add_argument(
        "--data-snapshot-id",
        type=str,
        default=None,
        metavar="ID",
        help=(
            "Stable snapshot identifier. If omitted, build_provenance derives one "
            "from artifacts/ipip_bffm_norms.json SHA-256 (or git hash fallback)."
        ),
    )
    group.add_argument(
        "--preprocess-tag",
        type=str,
        default=None,
        metavar="TAG",
        help="Override preprocessing version tag (default: auto-detect git hash)",
    )
    group.add_argument(
        "--bootstrap-b",
        type=int,
        default=None,
        metavar="N",
        help="Number of bootstrap resamples (overrides script-specific default)",
    )
    group.add_argument(
        "--bootstrap-seed",
        type=int,
        default=None,
        metavar="N",
        help="Fixed RNG seed for bootstrap resampling",
    )


def build_provenance(
    script: str,
    *,
    args: argparse.Namespace | None = None,
    bootstrap: dict[str, Any] | None = None,
    rng_seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a provenance metadata dict suitable for embedding in JSON artifacts.

    Parameters
    ----------
    script:
        The filename of the calling script (e.g. ``Path(__file__).name``).
    args:
        Parsed argparse Namespace.  If it contains ``data_snapshot`` or
        ``preprocess_tag`` attributes (from :func:`add_provenance_args`),
        those values are used; otherwise sensible defaults are applied.
    bootstrap:
        Optional bootstrap configuration dict.  When provided, it is stored
        verbatim under the ``bootstrap`` key.  Typical keys:
        ``B``, ``seed``, ``type``, ``paired``, ``stratified``, ``strata_def``.
    rng_seed:
        The primary RNG seed used by the script (if applicable).
    extra:
        Arbitrary additional key-value pairs merged into the provenance dict.

    Returns
    -------
    dict
        A provenance dict ready for ``json.dump``.
    """
    git_hash = _detect_git_hash()

    # Resolve stable snapshot fields
    data_snapshot_id = _resolve_data_snapshot_id(args=args, git_hash=git_hash)

    # Resolve preprocessing_version
    preprocessing_version: str
    if args is not None and hasattr(args, "preprocess_tag") and args.preprocess_tag is not None:
        preprocessing_version = args.preprocess_tag
    else:
        preprocessing_version = git_hash

    prov: dict[str, Any] = {
        "data_snapshot_id": data_snapshot_id,
        "preprocessing_version": preprocessing_version,
        "script": script,
        "git_hash": git_hash,
    }

    if rng_seed is not None:
        prov["rng_seed"] = rng_seed

    if bootstrap is not None:
        prov["bootstrap"] = bootstrap

    if extra:
        prov.update(extra)

    return prov
