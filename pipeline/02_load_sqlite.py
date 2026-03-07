#!/usr/bin/env python3
"""Load IPIP-FFM CSV into a local SQLite database with IPC filtering, reverse scoring, and domain scores."""

import sys
import json
import hashlib
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# Add package root to path for lib imports
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from lib.constants import (
    DOMAINS,
    DOMAIN_CSV_TO_INTERNAL,
    ITEMS_PER_DOMAIN,
    REVERSE_KEYED,
)
from lib.provenance import build_provenance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Paths
CSV_PATH = PACKAGE_ROOT / "data" / "raw" / "IPIP-FFM-data-8Nov2018" / "data-final.csv"
ZIP_PATH = PACKAGE_ROOT / "data" / "raw" / "IPIP-FFM-data-8Nov2018.zip"
DB_DIR = PACKAGE_ROOT / "data" / "processed"
DB_PATH = DB_DIR / "ipip_bffm.db"
LOAD_METADATA_PATH = DB_DIR / "load_metadata.json"


def _file_sha256(path: Path) -> str:
    """Compute SHA-256 digest for a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_csv_item_columns() -> list[str]:
    """Get uppercase CSV column names for all 50 items (EXT1 .. OPN10)."""
    cols = []
    for csv_domain in REVERSE_KEYED:
        for i in range(1, ITEMS_PER_DOMAIN + 1):
            cols.append(f"{csv_domain}{i}")
    return cols


def load_csv(path: Path) -> pd.DataFrame:
    """Load the tab-separated IPIP-FFM CSV."""
    log.info("Loading CSV from %s", path)
    df = pd.read_csv(path, sep="\t", low_memory=False)
    log.info("  Loaded %s rows, %d columns", f"{len(df):,}", len(df.columns))
    return df


def apply_reverse_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """Reverse-score negatively-keyed items: value -> 6 - value."""
    df = df.copy()
    n_reversed = 0
    for csv_domain, item_nums in REVERSE_KEYED.items():
        for num in item_nums:
            col = f"{csv_domain}{num}"
            if col in df.columns:
                mask = df[col].notna()
                df.loc[mask, col] = 6 - df.loc[mask, col]
                n_reversed += 1
    log.info("  Reverse-scored %d item columns", n_reversed)
    return df


def filter_valid_responses(df: pd.DataFrame, item_cols: list[str]) -> pd.DataFrame:
    """Keep only rows where ALL 50 items are valid integers 1-5."""
    before = len(df)
    valid_mask = df[item_cols].isin([1, 2, 3, 4, 5]).all(axis=1)
    df_valid = df[valid_mask].copy()
    after = len(df_valid)
    dropped = before - after
    log.info(
        "  Filtered: %s -> %s rows (dropped %s, %.1f%%)",
        f"{before:,}",
        f"{after:,}",
        f"{dropped:,}",
        dropped / before * 100 if before else 0,
    )
    return df_valid


def filter_unique_ip(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where the IP appears exactly once in the full raw dataset (IPC == 1).

    Per codebook: "For max cleanliness, only use records where [IPC] is 1."
    IPC is pre-computed over the entire raw dataset, so this is stricter than
    post-cleaning deduplication — it also removes rows whose IP had other
    (possibly invalid) submissions. This removes both true repeat submissions
    and false positives from shared networks (universities, VPNs).
    """
    before = len(df)
    if "IPC" not in df.columns:
        raise ValueError("IPC column not found in data; cannot apply IP-uniqueness filter")
    df_unique = df[df["IPC"] == 1].copy()
    after = len(df_unique)
    dropped = before - after
    log.info(
        "  IP-unique filter: %s -> %s rows (dropped %s, %.1f%%)",
        f"{before:,}",
        f"{after:,}",
        f"{dropped:,}",
        dropped / before * 100 if before else 0,
    )
    return df_unique


def compute_domain_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean domain scores from (reverse-scored) items."""
    df = df.copy()
    for csv_domain, internal in DOMAIN_CSV_TO_INTERNAL.items():
        item_cols = [f"{csv_domain}{i}" for i in range(1, ITEMS_PER_DOMAIN + 1)]
        available = [c for c in item_cols if c in df.columns]
        if available:
            df[f"{internal}_score"] = df[available].mean(axis=1)
    return df


def rename_columns_to_lowercase(df: pd.DataFrame) -> pd.DataFrame:
    """Rename uppercase item columns (EXT1) to lowercase (ext1)."""
    rename_map = {}
    for csv_domain, internal in DOMAIN_CSV_TO_INTERNAL.items():
        for i in range(1, ITEMS_PER_DOMAIN + 1):
            csv_col = f"{csv_domain}{i}"
            if csv_col in df.columns:
                rename_map[csv_col] = f"{internal}{i}"
    df = df.rename(columns=rename_map)
    return df


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select and order only the columns we want in SQLite."""
    item_cols = [f"{d}{i}" for d in DOMAINS for i in range(1, ITEMS_PER_DOMAIN + 1)]
    score_cols = [f"{d}_score" for d in DOMAINS]

    # country column (may or may not exist)
    extra_cols = []
    if "country" in df.columns:
        extra_cols.append("country")

    selected = item_cols + score_cols + extra_cols
    available = [c for c in selected if c in df.columns]
    return df[available].copy()


def write_to_sqlite(df: pd.DataFrame, db_path: Path) -> None:
    """Write the processed DataFrame to a SQLite database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing DB to start fresh
    if db_path.exists():
        db_path.unlink()
        log.info("  Removed existing database")

    log.info("  Writing %s rows to %s", f"{len(df):,}", db_path)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Build CREATE TABLE statement
    item_cols = [f"{d}{i}" for d in DOMAINS for i in range(1, ITEMS_PER_DOMAIN + 1)]
    score_cols = [f"{d}_score" for d in DOMAINS]

    col_defs = ["respondent_id INTEGER PRIMARY KEY AUTOINCREMENT"]
    for col in item_cols:
        col_defs.append(f"{col} INTEGER NOT NULL")
    for col in score_cols:
        col_defs.append(f"{col} REAL NOT NULL")
    col_defs.append("country TEXT")

    create_sql = "CREATE TABLE responses (\n  " + ",\n  ".join(col_defs) + "\n)"
    cursor.execute(create_sql)

    # Insert data in batches
    insert_cols = item_cols + score_cols
    if "country" in df.columns:
        insert_cols.append("country")

    placeholders = ", ".join(["?"] * len(insert_cols))
    insert_sql = f"INSERT INTO responses ({', '.join(insert_cols)}) VALUES ({placeholders})"

    batch_size = 10_000
    total = len(df)
    inserted = 0

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = df.iloc[start:end]
        rows = batch[insert_cols].values.tolist()
        cursor.executemany(insert_sql, rows)
        inserted += len(rows)
        if inserted % 100_000 == 0 or inserted == total:
            log.info("    Inserted %s / %s rows", f"{inserted:,}", f"{total:,}")

    conn.commit()

    # Verify row count
    cursor.execute("SELECT COUNT(*) FROM responses")
    count = cursor.fetchone()[0]
    log.info("  Verified: %s rows in responses table", f"{count:,}")

    # Log DB file size
    conn.close()
    size_mb = db_path.stat().st_size / 1_048_576
    log.info("  Database size: %.1f MB", size_mb)


def log_domain_stats(df: pd.DataFrame) -> None:
    """Log summary statistics for domain scores."""
    log.info("  Domain score statistics:")
    for domain in DOMAINS:
        col = f"{domain}_score"
        if col in df.columns:
            vals = df[col]
            log.info(
                "    %s: mean=%.3f, std=%.3f, min=%.2f, max=%.2f",
                domain.upper(),
                vals.mean(),
                vals.std(),
                vals.min(),
                vals.max(),
            )


def write_load_metadata(
    *,
    csv_path: Path,
    zip_path: Path,
    db_path: Path,
    n_raw: int,
    n_valid_items: int,
    n_valid: int,
) -> None:
    """Write fail-closed stage-02 provenance metadata."""
    payload = {
        "provenance": build_provenance(
            Path(__file__).name,
            extra={
                "csv_path": str(csv_path),
                "zip_path": str(zip_path),
                "db_path": str(db_path),
            },
        ),
        "inputs": {
            "zip_path": str(zip_path),
            "zip_sha256": _file_sha256(zip_path),
            "csv_path": str(csv_path),
            "csv_sha256": _file_sha256(csv_path),
        },
        "outputs": {
            "db_path": str(db_path),
            "db_sha256": _file_sha256(db_path),
            "table": "responses",
        },
        "row_counts": {
            "n_raw": int(n_raw),
            "n_valid_items": int(n_valid_items),
            "n_valid": int(n_valid),
            "n_dropped_invalid_items": int(n_raw - n_valid_items),
            "n_dropped_ipc_filter": int(n_valid_items - n_valid),
            "n_dropped": int(n_raw - n_valid),
        },
    }

    with open(LOAD_METADATA_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("  Wrote load provenance metadata: %s", LOAD_METADATA_PATH)


def main() -> int:
    log.info("=" * 60)
    log.info("IPIP-BFFM: Load CSV into SQLite")
    log.info("=" * 60)

    if not CSV_PATH.exists():
        log.error("CSV not found: %s", CSV_PATH)
        log.error("Run 01_download.py first.")
        return 1
    if not ZIP_PATH.exists():
        log.error("Source ZIP not found: %s", ZIP_PATH)
        log.error("Run 01_download.py first to restore verified source archive.")
        return 1

    # Step 1: Load CSV
    log.info("Step 1: Loading CSV...")
    df = load_csv(CSV_PATH)
    n_raw = len(df)

    # Step 2: Check item columns exist
    csv_item_cols = get_csv_item_columns()
    available = [c for c in csv_item_cols if c in df.columns]
    log.info("Step 2: Found %d / %d item columns", len(available), len(csv_item_cols))
    if len(available) < len(csv_item_cols):
        missing = sorted(set(csv_item_cols) - set(available))
        raise ValueError(f"Missing required item columns: {missing}")

    # Step 3: Filter valid responses (before reverse scoring, on raw values)
    log.info("Step 3: Filtering valid responses (all 50 items must be 1-5)...")
    df = filter_valid_responses(df, csv_item_cols)
    n_valid_items = len(df)

    # Step 4: Filter to unique IP respondents (IPC == 1)
    log.info("Step 4: Filtering to unique IP respondents (IPC == 1)...")
    df = filter_unique_ip(df)
    n_valid = len(df)

    # Step 5: Apply reverse scoring
    log.info("Step 5: Applying reverse scoring...")
    df = apply_reverse_scoring(df)

    # Step 6: Compute domain scores
    log.info("Step 6: Computing domain scores...")
    df = compute_domain_scores(df)
    log_domain_stats(df)

    # Step 7: Rename columns to lowercase
    log.info("Step 7: Renaming columns to lowercase...")
    df = rename_columns_to_lowercase(df)

    # Step 8: Select output columns
    log.info("Step 8: Selecting output columns...")
    df = select_output_columns(df)
    log.info("  Output columns (%d): %s", len(df.columns), list(df.columns))

    # Step 9: Write to SQLite
    log.info("Step 9: Writing to SQLite...")
    write_to_sqlite(df, DB_PATH)

    # Step 10: Write provenance metadata
    log.info("Step 10: Writing load metadata...")
    try:
        write_load_metadata(
            csv_path=CSV_PATH,
            zip_path=ZIP_PATH,
            db_path=DB_PATH,
            n_raw=n_raw,
            n_valid_items=n_valid_items,
            n_valid=n_valid,
        )
    except (FileNotFoundError, OSError, ValueError, TypeError) as e:
        log.error("Failed to write load metadata: %s", e)
        return 1

    log.info("=" * 60)
    log.info("Done. Database: %s", DB_PATH)
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
