#!/usr/bin/env python3
"""Backup-aware clean/restore helper for generated pipeline outputs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
BACKUP_ROOT = PACKAGE_ROOT / ".backup"
MANIFEST_PATH = BACKUP_ROOT / "manifest.json"

EXACT_CLEAN_TARGETS = [
    "data/raw",
    "data/processed",
    "artifacts/ipip_bffm_norms.json",
    "artifacts/ipip_bffm_norms.meta.json",
    "artifacts/tuned_params.json",
    "artifacts/tuned_params.original.json",
    "artifacts/research_summary.json",
    "artifacts/variants",
    "logs",
    "notes/NOTES.md",
    "output/README.md",
    "figures/manifest.json",
    ".git-hash",
    "pipeline.log",
    "pipeline-timing.log",
    ".pipeline-exit-code",
]

DIR_GLOBS = [
    "models/*",
    "output/*",
]

FILE_GLOBS = [
    "figures/*.png",
    "figures/*.pdf",
]


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _relative_targets() -> list[Path]:
    targets: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        rel = path.relative_to(PACKAGE_ROOT)
        key = rel.as_posix()
        if key not in seen:
            seen.add(key)
            targets.append(rel)

    for rel_str in EXACT_CLEAN_TARGETS:
        path = PACKAGE_ROOT / rel_str
        if _path_exists(path):
            add(path)

    for pattern in DIR_GLOBS:
        for path in sorted(PACKAGE_ROOT.glob(pattern)):
            if path.is_dir():
                add(path)

    for pattern in FILE_GLOBS:
        for path in sorted(PACKAGE_ROOT.glob(pattern)):
            if path.is_file():
                add(path)

    return sorted(targets, key=lambda p: p.as_posix())


def _confirm_or_abort(*, force: bool) -> None:
    if force:
        return
    if not sys.stdin.isatty():
        raise RuntimeError("No TTY attached. Use FORCE=1 to skip confirmation.")
    answer = input("Proceed? [y/N] ").strip()
    if answer.lower() != "y":
        raise RuntimeError("Aborted.")


def _write_manifest(paths: list[Path]) -> None:
    payload = {
        "version": 1,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "paths": [path.as_posix() for path in paths],
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _copy_path(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dest, copy_function=shutil.copy2)
    else:
        shutil.copy2(src, dest)


def clean(*, force: bool) -> int:
    targets = _relative_targets()
    if not targets:
        if BACKUP_ROOT.exists():
            print("Nothing to clean. Existing .backup preserved.")
        else:
            print("Nothing to clean.")
        return 0

    print("")
    print("WARNING: The following generated outputs will be moved into .backup/:")
    print("")
    for rel in targets:
        print(f"  {rel.as_posix()}")
    print("")
    if BACKUP_ROOT.exists():
        print("WARNING: Existing .backup/ will be deleted and replaced by this new backup.")
        print("")
    print("The cleaned locations will end up empty again, but the moved outputs will remain")
    print("available under .backup/ for `make restore`.")
    print("")

    try:
        _confirm_or_abort(force=force)
    except RuntimeError as exc:
        print(str(exc))
        return 1

    if BACKUP_ROOT.exists():
        shutil.rmtree(BACKUP_ROOT)
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)

    for rel in targets:
        src = PACKAGE_ROOT / rel
        dest = BACKUP_ROOT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))

    _write_manifest(targets)
    print(f"Clean complete. Backed up {len(targets)} path(s) to .backup/.")
    return 0


def restore(*, force: bool) -> int:
    if not BACKUP_ROOT.exists():
        print("ERROR: .backup/ does not exist. Nothing to restore.")
        return 1
    if not MANIFEST_PATH.exists():
        print("ERROR: .backup/manifest.json is missing. Cannot determine what to restore.")
        return 1

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        payload = json.load(f)

    raw_paths = payload.get("paths")
    if not isinstance(raw_paths, list) or not all(isinstance(p, str) for p in raw_paths):
        print("ERROR: Invalid .backup/manifest.json; expected a string path list.")
        return 1

    paths = [Path(p) for p in raw_paths]

    missing_sources: list[Path] = []
    conflicts: list[Path] = []
    for rel in paths:
        src = BACKUP_ROOT / rel
        dest = PACKAGE_ROOT / rel
        if not _path_exists(src):
            missing_sources.append(rel)
        if _path_exists(dest):
            conflicts.append(rel)

    if missing_sources:
        for rel in missing_sources:
            print(f"ERROR: Backup source missing: {rel.as_posix()}")
        return 1

    if conflicts and not force:
        for rel in conflicts:
            print(f"ERROR: Restore conflict at {rel.as_posix()}")
        print("Restore aborted. Use FORCE=1 to overwrite conflicting destinations.")
        return 1

    if conflicts and force:
        for rel in conflicts:
            _remove_path(PACKAGE_ROOT / rel)

    for rel in paths:
        _copy_path(BACKUP_ROOT / rel, PACKAGE_ROOT / rel)

    print(f"Restore complete. Restored {len(paths)} path(s) from .backup/.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Backup-aware clean/restore helper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    clean_parser = subparsers.add_parser("clean", help="Move generated outputs into .backup/.")
    clean_parser.add_argument("--force", action="store_true", help="Skip confirmation prompt.")

    restore_parser = subparsers.add_parser("restore", help="Copy backed-up outputs back into place.")
    restore_parser.add_argument("--force", action="store_true", help="Overwrite conflicting destinations.")

    args = parser.parse_args()

    if args.command == "clean":
        return clean(force=args.force)
    if args.command == "restore":
        return restore(force=args.force)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
