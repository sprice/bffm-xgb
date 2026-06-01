#!/usr/bin/env python
"""Pull the published reference ``model.onnx`` from Hugging Face into
``output/reference/``, sha256-verified against the ``.env`` pin.

A fresh clone has the small release bundle in git (config.json, provenance.json,
research_summary.json, figures/manifest.json, the norms meta) but NOT the ~248 MB
``model.onnx`` (gitignored). This fetches it from the pinned, immutable HF revision
and verifies its sha256 against ``HF_SHA256_MODEL`` so ``make verify-release`` can
check the model bytes too. Fail-closed: a missing pin or a sha mismatch refuses to
install (mirrors the web runtime's integrity contract in
``web/src/server/predictor.ts``). Idempotent: an already-present, verified model is
left untouched.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    """Populate os.environ from the repo .env (only keys not already set)."""
    env_path = PACKAGE_ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    _load_env()
    repo_id = os.environ.get("HF_REPO_ID")
    revision = os.environ.get("HF_REVISION")
    sha_model = os.environ.get("HF_SHA256_MODEL")
    variant = os.environ.get("HF_VARIANT", "reference")
    token = os.environ.get("HF_TOKEN") or None

    missing = [
        name
        for name, value in (
            ("HF_REPO_ID", repo_id),
            ("HF_REVISION", revision),
            ("HF_SHA256_MODEL", sha_model),
        )
        if not value
    ]
    if missing:
        print(
            f"ERROR: missing required HF pin(s) in .env: {', '.join(missing)}. "
            "A pinned, content-verified model is required (no unpinned 'main' fallback).",
            file=sys.stderr,
        )
        return 1
    assert repo_id is not None and revision is not None and sha_model is not None  # narrow for type-checkers

    dest = PACKAGE_ROOT / "output" / variant / "model.onnx"
    if dest.exists() and _sha256(dest).lower() == sha_model.lower():
        print(f"model.onnx already present and verified ({sha_model[:12]}...); skipping download")
        return 0

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub not installed (run `uv sync`)", file=sys.stderr)
        return 1

    print(f"Downloading {variant}/model.onnx from {repo_id}@{revision[:12]}... (sha-verified)")
    try:
        local = hf_hub_download(
            repo_id=repo_id,
            filename=f"{variant}/model.onnx",
            revision=revision,
            token=token,
        )
    except Exception as exc:  # noqa: BLE001 - surface any HF/network failure clearly
        print(f"ERROR: HF download failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    actual = _sha256(Path(local))
    if actual.lower() != sha_model.lower():
        print(
            f"ERROR: sha256 mismatch -- downloaded {actual[:12]}... but .env pins "
            f"{sha_model[:12]}...; refusing to install (fail-closed).",
            file=sys.stderr,
        )
        return 1

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local, dest)
    print(f"Verified + installed {dest.relative_to(PACKAGE_ROOT)} ({actual[:12]}...)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
