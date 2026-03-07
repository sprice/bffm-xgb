#!/usr/bin/env python3
"""Upload model artifacts to HuggingFace Hub."""

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

import argparse
import json
import logging

from lib.constants import DOMAINS
from lib.provenance import file_sha256

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

QUANTILE_NAMES = ("q05", "q50", "q95")


def _validate_output_bundle(output_dir: Path) -> list[Path]:
    """Fail closed unless output bundle matches expected merged ONNX export shape."""
    if not output_dir.exists() or not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    required_static = ["config.json", "README.md", "provenance.json"]
    for filename in required_static:
        path = output_dir / filename
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Missing required export file: {path}")

    # Validate provenance.json structure
    prov_path = output_dir / "provenance.json"
    with open(prov_path) as f:
        prov_doc = json.load(f)
    if not isinstance(prov_doc, dict):
        raise ValueError("output/provenance.json must be a JSON object.")
    if not isinstance(prov_doc.get("export"), dict):
        raise ValueError("output/provenance.json missing required 'export' block.")
    if not isinstance(prov_doc.get("training"), dict):
        raise ValueError("output/provenance.json missing required 'training' block.")

    config_path = output_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError("output/config.json must be a JSON object.")

    # Cross-check provenance.json export fields against config.json provenance
    export_block = prov_doc.get("export", {})
    config_prov = config.get("provenance", {})
    for key in ("git_hash", "data_snapshot_id", "preprocessing_version", "model_dir"):
        export_val = export_block.get(key)
        config_val = config_prov.get(key)
        if not isinstance(export_val, str) or not export_val.strip():
            raise ValueError(
                f"provenance.json export.{key} missing or empty."
            )
        if not isinstance(config_val, str) or not config_val.strip():
            raise ValueError(
                f"config.json provenance.{key} missing or empty."
            )
        if export_val != config_val:
            raise ValueError(
                f"provenance.json export.{key} ({export_val!r}) does not match "
                f"config.json provenance.{key} ({config_val!r})."
            )

    # Verify artifact SHA-256s against actual files
    artifacts_block = prov_doc.get("artifacts", {})
    if isinstance(artifacts_block, dict):
        model_file_name = config.get("model_file", "model.onnx")
        for field, filename in [
            ("config_json_sha256", "config.json"),
            ("model_onnx_sha256", model_file_name),
        ]:
            expected_sha = artifacts_block.get(field)
            if not isinstance(expected_sha, str) or not expected_sha.strip():
                raise ValueError(f"provenance.json missing artifacts.{field}")
            actual_sha = file_sha256(output_dir / filename)
            if actual_sha.lower() != expected_sha.lower():
                raise ValueError(
                    f"provenance.json artifacts.{field} does not match actual {filename}"
                )

    provenance = config.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("output/config.json missing provenance block.")
    for key in ("git_hash", "data_snapshot_id", "training_script"):
        value = provenance.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"output/config.json missing provenance.{key}.")

    model_file = config.get("model_file")
    if not isinstance(model_file, str) or not model_file.strip():
        raise ValueError("output/config.json missing model_file.")

    model_path = output_dir / model_file
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    outputs = config.get("outputs")
    expected_outputs = [
        f"{domain}_{q}" for domain in DOMAINS for q in QUANTILE_NAMES
    ]
    if not isinstance(outputs, list) or outputs != expected_outputs:
        raise ValueError(
            "output/config.json outputs does not match expected 15 output names."
        )

    allowed_files = set(required_static) | {model_file}
    regular_files = {p.name for p in output_dir.iterdir() if p.is_file()}
    unexpected = sorted(regular_files - allowed_files)
    if unexpected:
        raise ValueError(
            "Unexpected files in output bundle (refusing upload): " + ", ".join(unexpected)
        )

    upload_order = required_static + [model_file]
    return [output_dir / name for name in upload_order]


def _discover_variants(output_dir: Path) -> list[tuple[str, Path]]:
    """Scan output_dir for variant subdirectories containing config.json.

    Returns list of (variant_name, variant_path) tuples.
    Backward-compat: if output_dir itself has config.json (flat layout)
    AND no variant subdirectories exist, return [("", output_dir)].
    """
    if not output_dir.is_dir():
        return []

    # Scan for variant subdirectories first
    variants = []
    for child in sorted(output_dir.iterdir()):
        if child.is_dir() and (child / "config.json").is_file():
            variants.append((child.name, child))

    if variants:
        # Warn if stale root config.json coexists with variant subdirs
        if (output_dir / "config.json").is_file():
            stale = [p.name for p in output_dir.iterdir() if p.is_file()]
            log.warning(
                "Ignoring stale root files %s in favour of %d variant subdirectories",
                stale,
                len(variants),
            )
        return variants

    # Fall back to flat (legacy) layout
    if (output_dir / "config.json").is_file():
        return [("", output_dir)]

    return []


def _resolve_output_dir(path: Path) -> Path:
    """Resolve relative output directories against PACKAGE_ROOT."""
    if path.is_absolute():
        return path
    return PACKAGE_ROOT / path


def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="HuggingFace repo ID (default: HF_REPO_ID env var)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "output",
        help="Directory containing model files to upload",
    )
    parser.add_argument("--private", action="store_true", help="Make repo private")
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Upload a single variant with files at repo root (e.g. --variant reference)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the repo to clear commit history before uploading",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
    except ImportError:
        log.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    import os

    # Load .env file if present
    env_path = PACKAGE_ROOT / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())

    token = os.environ.get("HF_TOKEN")
    if not token:
        log.error("HF_TOKEN not set. Create a .env file (see .env.example) or set HF_TOKEN env var")
        sys.exit(1)

    repo_id = args.repo_id or os.environ.get("HF_REPO_ID")
    if not repo_id:
        log.error("No repo ID. Pass --repo-id or set HF_REPO_ID in .env")
        sys.exit(1)
    args.repo_id = repo_id

    api = HfApi(token=token)

    # Create repo (optionally reset to clear commit history)
    if args.reset:
        log.info("Resetting repo: %s (deleting and recreating)", args.repo_id)
        try:
            api.delete_repo(repo_id=args.repo_id)
        except Exception as exc:
            log.info("delete_repo skipped (repo may not exist yet): %s", exc)
    log.info("Creating/updating repo: %s", args.repo_id)
    api.create_repo(repo_id=args.repo_id, exist_ok=True, private=args.private)

    output_dir = _resolve_output_dir(args.output_dir)

    upload_manifest: list[tuple[Path, str]]
    if args.variant:
        # Single-variant mode: upload one variant's files to repo root
        variant_path = output_dir / args.variant
        if not variant_path.is_dir():
            log.error("Variant directory not found: %s", variant_path)
            sys.exit(1)

        try:
            files = _validate_output_bundle(variant_path)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
            log.error("Validation failed for variant %r: %s", args.variant, e)
            sys.exit(1)

        upload_manifest = [
            (file_path, file_path.name) for file_path in files
        ]
        log.info("Uploading %d files from variant %r (to repo root)", len(upload_manifest), args.variant)
    else:
        # Multi-variant mode: upload all variants in subdirectories
        variants = _discover_variants(output_dir)
        if not variants:
            log.error("No variant bundles found in %s", output_dir)
            sys.exit(1)

        upload_manifest = []
        for variant_name, variant_path in variants:
            try:
                files = _validate_output_bundle(variant_path)
            except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
                log.error("Validation failed for variant %r: %s", variant_name or "(flat)", e)
                sys.exit(1)

            for file_path in files:
                if variant_name:
                    path_in_repo = f"{variant_name}/{file_path.name}"
                else:
                    path_in_repo = file_path.name
                upload_manifest.append((file_path, path_in_repo))

        # Include repo-level README if it exists (generated by `make export-repo-readme`)
        repo_readme_path = output_dir / "README.md"
        if repo_readme_path.is_file():
            upload_manifest.append((repo_readme_path, "README.md"))

        log.info("Uploading %d files from %d variant(s)", len(upload_manifest), len(variants))
    if args.variant:
        # Single-variant mode: atomic commit with stale-file cleanup
        operations: list[CommitOperationAdd | CommitOperationDelete] = []
        for local_path, path_in_repo in sorted(upload_manifest, key=lambda x: x[1]):
            log.info("  %s (%s bytes)", path_in_repo, f"{local_path.stat().st_size:,}")
            operations.append(
                CommitOperationAdd(
                    path_in_repo=path_in_repo,
                    path_or_fileobj=str(local_path),
                )
            )

        # Delete stale files that are not in the upload manifest
        new_files = {op.path_in_repo for op in operations if isinstance(op, CommitOperationAdd)}
        preserve = {".gitattributes"}
        try:
            existing = set(api.list_repo_files(repo_id=args.repo_id))
            stale = existing - new_files - preserve
            if stale:
                log.info("Deleting %d stale file(s) from repo", len(stale))
                for path in sorted(stale):
                    log.info("  (delete) %s", path)
                    operations.append(CommitOperationDelete(path_in_repo=path))
        except Exception as exc:
            log.warning("Could not list existing repo files, skipping cleanup: %s", exc)

        api.create_commit(
            repo_id=args.repo_id,
            operations=operations,
            commit_message=f"Upload variant {args.variant}",
        )
    else:
        # Multi-variant mode: individual file uploads
        for local_path, path_in_repo in sorted(upload_manifest, key=lambda x: x[1]):
            log.info("  %s (%s bytes)", path_in_repo, f"{local_path.stat().st_size:,}")
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=args.repo_id,
            )

    log.info("Upload complete: https://huggingface.co/%s", args.repo_id)


if __name__ == "__main__":
    main()
