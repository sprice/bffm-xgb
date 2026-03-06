#!/usr/bin/env python3
"""Deploy the BFFM-XGB web assessment app to a HuggingFace Docker Space."""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import (
        CommitOperationAdd,
        CommitOperationDelete,
        HfApi,
        create_repo,
    )
    from huggingface_hub.errors import HfHubHTTPError
except ImportError:
    print("Install huggingface_hub: pip install huggingface_hub", file=sys.stderr)
    sys.exit(1)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PACKAGE_ROOT / "web"


def _load_dotenv() -> None:
    """Load variables from PACKAGE_ROOT/.env if the file exists.

    Uses python-dotenv when available, falling back to a minimal manual
    parser (same approach used by pipeline/13_upload_hf.py).  Variables
    already present in the environment are never overwritten.
    """
    env_path = PACKAGE_ROOT / ".env"
    if not env_path.exists():
        return

    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except ImportError:
        # Fallback: manual parser identical to 13_upload_hf.py
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())


def main():
    _load_dotenv()

    token = os.environ.get("HF_TOKEN")
    space_id = os.environ.get("HF_SPACE_ID")
    repo_id = os.environ.get("HF_REPO_ID")

    if not token:
        print("ERROR: HF_TOKEN not set", file=sys.stderr)
        sys.exit(1)
    if not space_id:
        print("ERROR: HF_SPACE_ID not set", file=sys.stderr)
        sys.exit(1)
    if not repo_id:
        print("ERROR: HF_REPO_ID not set (needed so the container can download the model)", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)

    print(f"Creating/updating Space: {space_id}")
    create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
        token=token,
    )

    # Set Space secrets so the container can fetch and verify the model.
    print(f"  Setting HF_REPO_ID secret to: {repo_id}")
    api.add_space_secret(repo_id=space_id, key="HF_REPO_ID", value=repo_id)

    for key in ("HF_REVISION", "HF_SHA256_CONFIG", "HF_SHA256_MODEL"):
        value = os.environ.get(key)
        if value:
            print(f"  Setting {key} secret")
            api.add_space_secret(repo_id=space_id, key=key, value=value)
        else:
            try:
                api.delete_space_secret(repo_id=space_id, key=key)
                print(f"  Cleared {key} secret (not set locally)")
            except HfHubHTTPError as exc:
                if exc.response.status_code != 404:
                    print(f"  Warning: failed to clear {key} secret: {exc}")
            except Exception as exc:
                print(f"  Warning: failed to clear {key} secret: {exc}")

    # Verify required build artifacts exist
    dist_dir = WEB_DIR / "dist"
    if not dist_dir.exists():
        print("ERROR: dist/ not found. Run 'npm run build' first.", file=sys.stderr)
        sys.exit(1)

    # Build a single atomic commit with all files.
    operations: list[CommitOperationAdd | CommitOperationDelete] = []

    # Add individual root files
    for fname in ["package.json", "package-lock.json", "Dockerfile"]:
        fpath = WEB_DIR / fname
        if fpath.exists():
            print(f"  Staging {fname}")
            operations.append(
                CommitOperationAdd(
                    path_in_repo=fname,
                    path_or_fileobj=str(fpath),
                )
            )

    # Add dist/ files
    print("  Staging dist/")
    for fpath in sorted(dist_dir.rglob("*")):
        if fpath.is_file():
            rel = fpath.relative_to(WEB_DIR)
            operations.append(
                CommitOperationAdd(
                    path_in_repo=rel.as_posix(),
                    path_or_fileobj=str(fpath),
                )
            )

    # Remove stale files that are no longer part of the build output.
    new_files = {op.path_in_repo for op in operations if isinstance(op, CommitOperationAdd)}
    preserve = {".gitattributes", "README.md"}
    try:
        existing = set(api.list_repo_files(repo_id=space_id, repo_type="space"))
        stale = existing - new_files - preserve
        if stale:
            print(f"  Deleting {len(stale)} stale file(s)")
            for path in sorted(stale):
                operations.append(CommitOperationDelete(path_in_repo=path))
    except Exception as exc:
        print(f"  Warning: could not list existing files, skipping cleanup: {exc}")

    print(f"  Committing {len(operations)} operations in a single atomic commit...")
    api.create_commit(
        repo_id=space_id,
        repo_type="space",
        operations=operations,
        commit_message="Deploy web app",
    )

    print(f"Done! Space: https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    main()
