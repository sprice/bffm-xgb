#!/usr/bin/env python3
"""Guard and optionally clean local outputs that conflict with remote-reference."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

CONFLICT_PATHS = [
    "artifacts/research_summary.json",
    "artifacts/variants/ablation_none",
    "artifacts/variants/ablation_focused",
    "artifacts/variants/ablation_stratified",
    "notes/NOTES.md",
    "output/ablation_none",
    "output/ablation_focused",
    "output/ablation_stratified",
    "logs/train-ablation-none.log",
    "logs/train-ablation-focused.log",
    "logs/train-ablation-stratified.log",
    "logs/eval-ablation-none.log",
    "logs/eval-ablation-focused.log",
    "logs/eval-ablation-stratified.log",
]


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fail closed when local non-reference artifacts would make "
            "`make remote-reference` ambiguous, or remove them with --force."
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove conflicting generated outputs instead of failing.",
    )
    args = parser.parse_args()

    conflicts = [rel for rel in CONFLICT_PATHS if (PACKAGE_ROOT / rel).exists()]
    if not conflicts:
        print("Workspace is clean for remote-reference.")
        return 0

    if not args.force:
        print("ERROR: remote-reference refuses to run with existing non-reference outputs:")
        for rel in conflicts:
            print(f"  - {rel}")
        print("Run `make clean`, remove these paths manually, or rerun with `FORCE=1`.")
        return 1

    print("Clearing stale local non-reference outputs for remote-reference:")
    for rel in conflicts:
        print(f"  - {rel}")
        _remove_path(PACKAGE_ROOT / rel)
    print("Workspace cleaned for remote-reference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
