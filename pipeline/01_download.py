#!/usr/bin/env python3
"""Download the IPIP-FFM dataset ZIP from openpsychometrics.org."""

import sys
import os
import hashlib
import logging
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# Add package root to path for lib imports
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

URL = "https://openpsychometrics.org/_rawdata/IPIP-FFM-data-8Nov2018.zip"
EXPECTED_ZIP_SHA256 = "d19ca933d974c371a48896c7dce61c005780953c21fe88bb9a95382d8ef22904"
RAW_DIR = PACKAGE_ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "IPIP-FFM-data-8Nov2018.zip"
EXTRACT_DIR = RAW_DIR / "IPIP-FFM-data-8Nov2018"
CSV_PATH = EXTRACT_DIR / "data-final.csv"


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Report download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        mb_done = downloaded / 1_048_576
        mb_total = total_size / 1_048_576
        print(f"\r  {pct:5.1f}%  ({mb_done:.1f} / {mb_total:.1f} MB)", end="", flush=True)
    else:
        mb_done = downloaded / 1_048_576
        print(f"\r  {mb_done:.1f} MB downloaded", end="", flush=True)


def _file_sha256(path: Path) -> str:
    """Compute SHA-256 digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_zip_integrity(path: Path) -> None:
    """Fail closed when downloaded archive hash does not match expected value."""
    actual = _file_sha256(path).lower()
    expected = EXPECTED_ZIP_SHA256.lower()
    if actual != expected:
        raise ValueError(
            "Dataset ZIP SHA-256 mismatch. "
            f"expected={expected}, actual={actual}, path={path}"
        )


def _resolve_csv_member_name(zf: zipfile.ZipFile) -> str:
    """Resolve the unique data-final.csv member path from the dataset ZIP."""
    members = [
        info.filename
        for info in zf.infolist()
        if (not info.is_dir()) and Path(info.filename).name == "data-final.csv"
    ]
    if len(members) != 1:
        raise ValueError(
            "Dataset ZIP must contain exactly one data-final.csv member; "
            f"found {len(members)}."
        )
    return members[0]


def _verify_extracted_csv_matches_zip(zip_path: Path, csv_path: Path) -> None:
    """Fail closed when extracted CSV bytes diverge from ZIP member bytes."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Extracted CSV not found: {csv_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        member_name = _resolve_csv_member_name(zf)
        expected_hasher = hashlib.sha256()
        with zf.open(member_name, "r") as member:
            for chunk in iter(lambda: member.read(1 << 20), b""):
                expected_hasher.update(chunk)

    expected = expected_hasher.hexdigest().lower()
    actual = _file_sha256(csv_path).lower()
    if actual != expected:
        raise ValueError(
            "Extracted CSV SHA-256 mismatch vs ZIP member. "
            f"expected={expected}, actual={actual}, csv_path={csv_path}"
        )


def _safe_extract_all(zip_path: Path, dest_dir: Path) -> None:
    """Safely extract ZIP contents without allowing path traversal."""
    dest_root = dest_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = (dest_dir / member.filename).resolve()
            if os.path.commonpath([str(dest_root), str(member_path)]) != str(dest_root):
                raise ValueError(f"Unsafe ZIP member path detected: {member.filename!r}")
        zf.extractall(dest_dir)


def main() -> int:
    log.info("IPIP-FFM Dataset Download")
    log.info("URL: %s", URL)

    # Check if already extracted, but still verify the source ZIP hash.
    if CSV_PATH.exists():
        size_mb = CSV_PATH.stat().st_size / 1_048_576
        if not ZIP_PATH.exists():
            log.error(
                "Extracted CSV exists but source ZIP is missing: %s. "
                "Cannot verify dataset integrity on rerun.",
                ZIP_PATH,
            )
            log.error("Re-run stage 01 download to restore verified source artifacts.")
            return 1
        try:
            _verify_zip_integrity(ZIP_PATH)
            _verify_extracted_csv_matches_zip(ZIP_PATH, CSV_PATH)
        except (OSError, ValueError, zipfile.BadZipFile) as e:
            log.error("ZIP integrity check failed: %s", e)
            return 1
        zip_size_mb = ZIP_PATH.stat().st_size / 1_048_576
        log.info(
            "Already extracted: %s (%.1f MB); verified source ZIP and CSV payload: %s (%.1f MB) -- skipping",
            CSV_PATH,
            size_mb,
            ZIP_PATH,
            zip_size_mb,
        )
        return 0

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Download ZIP if not present
    if not ZIP_PATH.exists():
        log.info("Downloading to %s ...", ZIP_PATH)
        urlretrieve(URL, ZIP_PATH, reporthook=_progress_hook)
        print()  # newline after progress
        size_mb = ZIP_PATH.stat().st_size / 1_048_576
        log.info("Downloaded %.1f MB", size_mb)
    else:
        size_mb = ZIP_PATH.stat().st_size / 1_048_576
        log.info("ZIP already exists: %s (%.1f MB)", ZIP_PATH, size_mb)

    try:
        _verify_zip_integrity(ZIP_PATH)
    except (OSError, ValueError) as e:
        log.error("ZIP integrity check failed: %s", e)
        return 1

    # Extract
    log.info("Extracting to %s ...", EXTRACT_DIR)
    try:
        _safe_extract_all(ZIP_PATH, RAW_DIR)
    except (OSError, zipfile.BadZipFile, ValueError) as e:
        log.error("Extraction failed: %s", e)
        return 1

    if not CSV_PATH.exists():
        log.error("Expected CSV not found after extraction: %s", CSV_PATH)
        return 1

    try:
        _verify_extracted_csv_matches_zip(ZIP_PATH, CSV_PATH)
    except (OSError, ValueError, zipfile.BadZipFile) as e:
        log.error("Post-extract CSV integrity check failed: %s", e)
        return 1

    csv_size_mb = CSV_PATH.stat().st_size / 1_048_576
    log.info("Extracted: %s (%.1f MB)", CSV_PATH, csv_size_mb)
    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
