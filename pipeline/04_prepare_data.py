#!/usr/bin/env python3
"""Prepare train/val/test splits with percentiles from SQLite.

The split is a single plain random 70/15/15 partition with a fixed seed (the
canonical_v1 split). Percentile targets are computed from train-only norms
(stage 03), so no validation/test information leaks into the targets.
"""

import argparse
import hashlib
import json
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Add package root to path for lib imports
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from lib.constants import DOMAINS, ITEMS_PER_DOMAIN
from lib.item_info import file_sha256
from lib.norms import load_norms
from lib.provenance import add_provenance_args, build_provenance, relative_to_root
from lib.scoring import raw_score_to_percentile
from lib.splits import (
    CANONICAL_SEED,
    CANONICAL_SPLIT_ID,
    CANONICAL_TEST_SIZE,
    CANONICAL_VAL_SIZE,
    SPLIT_SCHEME,
    assign_splits,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/processed/ipip_bffm.db")
DEFAULT_OUTPUT_DIR = Path(f"data/processed/{CANONICAL_SPLIT_ID}")
DEFAULT_NORMS_PATH = Path("artifacts/ipip_bffm_norms.json")

ITEM_COLUMNS = [f"{d}{i}" for d in DOMAINS for i in range(1, ITEMS_PER_DOMAIN + 1)]
SCORE_COLUMNS = [f"{d}_score" for d in DOMAINS]
PERCENTILE_COLUMNS = [f"{d}_percentile" for d in DOMAINS]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare IPIP-BFFM train/val/test splits with percentiles. "
            "Single plain random 70/15/15 split (canonical_v1); percentiles use "
            "train-only norms."
        )
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=CANONICAL_TEST_SIZE,
        help=f"Fraction for test set (default: {CANONICAL_TEST_SIZE})",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=CANONICAL_VAL_SIZE,
        help=f"Fraction for validation set (default: {CANONICAL_VAL_SIZE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=CANONICAL_SEED,
        help=f"Random seed for reproducibility (default: {CANONICAL_SEED})",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only use first N rows (for quick development / smoke runs)",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to stage-02 SQLite DB (default: data/processed/ipip_bffm.db)",
    )
    parser.add_argument(
        "--norms",
        type=Path,
        default=DEFAULT_NORMS_PATH,
        help=(
            "Path to the stage-03 train-only norms artifact used to compute "
            "percentile targets (default: artifacts/ipip_bffm_norms.json)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for split parquet + metadata (default: data/processed/{CANONICAL_SPLIT_ID})",
    )
    add_provenance_args(parser)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_from_sqlite(db_path: Path, sample: int | None = None) -> pd.DataFrame:
    """Load all responses from the SQLite database (ordered by respondent_id)."""
    log.info("Loading from %s", db_path)
    conn = sqlite3.connect(str(db_path))
    if sample is not None:
        df = pd.read_sql_query(
            f"SELECT * FROM responses ORDER BY respondent_id LIMIT {sample}",
            conn,
        )
    else:
        df = pd.read_sql_query("SELECT * FROM responses ORDER BY respondent_id", conn)
    conn.close()
    log.info("  Loaded %s rows, %d columns", f"{len(df):,}", len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Percentile computation
# ---------------------------------------------------------------------------

def add_percentile_columns(df: pd.DataFrame, norms: dict) -> pd.DataFrame:
    """Add percentile columns for each domain using z-score against train norms."""
    df = df.copy()
    for domain in DOMAINS:
        score_col = f"{domain}_score"
        pct_col = f"{domain}_percentile"
        if score_col in df.columns:
            df[pct_col] = raw_score_to_percentile(df[score_col].values, domain, norms=norms)
    return df


def assert_norms_match_split(norms_path: Path, *, seed: float, test_size: float, val_size: float) -> None:
    """Fail closed unless the norms artifact was fit on this exact train split.

    Percentile targets are leakage-free only if they come from norms fit on
    precisely the rows stage 04 labels ``train``. Stage 03 records the split it
    fit on in a ``split`` block (id/scheme/seed/test_size/val_size, fit_on). We
    refuse to proceed if that block is missing (e.g. a stale full-dataset norms
    file) or if it does not match the split this run is producing, rather than
    silently applying mismatched norms.
    """
    with open(norms_path) as f:
        payload = json.load(f)
    split = payload.get("split") if isinstance(payload, dict) else None
    if not isinstance(split, dict):
        raise ValueError(
            f"Norms artifact {norms_path} has no 'split' block; it was not fit on "
            f"the {CANONICAL_SPLIT_ID} train split (stale/full-dataset norms re-introduce "
            "val/test leakage). Re-run stage 03 (make norms)."
        )
    if split.get("fit_on") != "train":
        raise ValueError(
            f"Norms artifact {norms_path} was not fit on the train split "
            f"(split.fit_on={split.get('fit_on')!r}); refusing to compute leaky percentiles."
        )
    if split.get("id") != CANONICAL_SPLIT_ID:
        raise ValueError(
            f"Norms artifact {norms_path} split id {split.get('id')!r} != {CANONICAL_SPLIT_ID!r}."
        )
    mismatches = []
    if split.get("seed") != seed:
        mismatches.append(f"seed (norms={split.get('seed')}, run={seed})")
    if split.get("test_size") != test_size:
        mismatches.append(f"test_size (norms={split.get('test_size')}, run={test_size})")
    if split.get("val_size") != val_size:
        mismatches.append(f"val_size (norms={split.get('val_size')}, run={val_size})")
    if mismatches:
        raise ValueError(
            f"Split parameters differ from the norms artifact {norms_path}: "
            + "; ".join(mismatches)
            + ". The split this run produces would not match the rows the norms were "
            "fit on. Re-run stage 03 with matching parameters or use the canonical defaults."
        )


def validate_percentile_computation() -> bool:
    """Validate z-score percentile computation against known values."""
    test_cases = [
        (0.0, 50.0),
        (1.0, 84.1345),
        (-1.0, 15.8655),
        (2.0, 97.7250),
        (-2.0, 2.2750),
    ]
    import scipy.special

    all_ok = True
    for z, expected in test_cases:
        actual = 0.5 * (1.0 + scipy.special.erf(z / np.sqrt(2.0))) * 100
        if abs(actual - expected) > 0.01:
            log.error("  FAIL: z=%.1f expected %.4f, got %.4f", z, expected, actual)
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Splitting (plain random, seed-locked)
# ---------------------------------------------------------------------------

def random_split(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Plain random train/val/test split keyed on respondent_id (deterministic).

    Uses the shared :func:`lib.splits.assign_splits` so that the ``train`` rows
    here are exactly the rows stage 03 fit its norms on.
    """
    if "respondent_id" not in df.columns:
        raise ValueError("respondent_id column required for the canonical split")

    labels = assign_splits(
        df["respondent_id"].to_numpy(),
        seed=seed,
        test_size=test_size,
        val_size=val_size,
    )
    train_df = df.loc[labels == "train"]
    val_df = df.loc[labels == "val"]
    test_df = df.loc[labels == "test"]
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """Validate that splits have similar distributions using KS tests."""
    validation = {}

    for domain in DOMAINS:
        score_col = f"{domain}_score"
        if score_col not in train_df.columns:
            continue

        train_vals = train_df[score_col]
        val_vals = val_df[score_col]
        test_vals = test_df[score_col]

        train_mean = float(train_vals.mean())
        val_mean = float(val_vals.mean())
        test_mean = float(test_vals.mean())

        max_diff = max(
            abs(train_mean - val_mean),
            abs(train_mean - test_mean),
            abs(val_mean - test_mean),
        )

        ks_stat, ks_pval = stats.ks_2samp(train_vals, test_vals)

        validation[domain] = {
            "train_mean": train_mean,
            "val_mean": val_mean,
            "test_mean": test_mean,
            "max_mean_diff": float(max_diff),
            "train_std": float(train_vals.std()),
            "val_std": float(val_vals.std()),
            "test_std": float(test_vals.std()),
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pval),
        }

    return validation


def log_validation(validation: dict) -> None:
    """Log validation results."""
    all_ok = True
    for domain, v in validation.items():
        ok = v["max_mean_diff"] < 0.05 and v["ks_pvalue"] > 0.01
        status = "OK" if ok else "WARN"
        if not ok:
            all_ok = False
        log.info(
            "    %s: max_diff=%.4f, KS_stat=%.4f, KS_p=%.3f  [%s]",
            domain.upper(),
            v["max_mean_diff"],
            v["ks_statistic"],
            v["ks_pvalue"],
            status,
        )
    if not all_ok:
        log.warning("  Some domain distributions differ significantly across splits.")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def select_parquet_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select and cast columns for the output parquet files."""
    cols = []

    # Item columns
    for col in ITEM_COLUMNS:
        if col in df.columns:
            cols.append(col)

    # Domain raw score columns
    for col in SCORE_COLUMNS:
        if col in df.columns:
            cols.append(col)

    # Domain percentile columns
    for col in PERCENTILE_COLUMNS:
        if col in df.columns:
            cols.append(col)

    out = df[cols].copy()

    # Cast item columns to float32
    for col in ITEM_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype(np.float32)

    return out


def write_metadata(
    output_dir: Path,
    df_all: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    validation: dict,
    *,
    seed: int,
    test_size: float,
    val_size: float,
    db_path: Path,
    norms_path: Path,
    norms_sha256: str,
    train_path: Path,
    val_path: Path,
    test_path: Path,
    args: argparse.Namespace,
) -> None:
    """Write split metadata JSON."""
    db_sha256 = file_sha256(db_path)
    train_sha256 = file_sha256(train_path)
    val_sha256 = file_sha256(val_path)
    test_sha256 = file_sha256(test_path)
    split_signature = hashlib.sha256(
        (
            f"train={train_sha256}\n"
            f"val={val_sha256}\n"
            f"test={test_sha256}\n"
        ).encode()
    ).hexdigest()

    meta_path = output_dir / "split_metadata.json"
    provenance = build_provenance(
        Path(__file__).name,
        args=args,
        rng_seed=seed,
        extra={
            "db_path": relative_to_root(db_path),
            "output_dir": relative_to_root(output_dir),
            "split_id": CANONICAL_SPLIT_ID,
            "split_scheme": SPLIT_SCHEME,
            "split_signature": split_signature,
        },
    )

    metadata = {
        "provenance": provenance,
        "split_id": CANONICAL_SPLIT_ID,
        "split_scheme": SPLIT_SCHEME,
        "seed": seed,
        "test_size": test_size,
        "val_size": val_size,
        "total_valid": len(df_all),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_frac": round(len(train_df) / len(df_all), 4),
        "val_frac": round(len(val_df) / len(df_all), 4),
        "test_frac": round(len(test_df) / len(df_all), 4),
        "item_columns": ITEM_COLUMNS,
        "score_columns": SCORE_COLUMNS,
        "percentile_columns": PERCENTILE_COLUMNS,
        # Percentile targets are derived from these train-only norms.
        "norms_path": relative_to_root(norms_path),
        "norms_sha256": norms_sha256,
        "validation": validation,
        "split_signature": split_signature,
        "inputs": {
            "sqlite_db": {
                "path": str(db_path),
                "sha256": db_sha256,
            },
            "norms": {
                "path": relative_to_root(norms_path),
                "sha256": norms_sha256,
            },
        },
        "outputs": {
            "train": {
                "path": str(train_path),
                "sha256": train_sha256,
                "rows": int(len(train_df)),
            },
            "val": {
                "path": str(val_path),
                "sha256": val_sha256,
                "rows": int(len(val_df)),
            },
            "test": {
                "path": str(test_path),
                "sha256": test_sha256,
                "rows": int(len(test_df)),
            },
            "split_metadata": {
                "path": str(meta_path),
            },
        },
        # Convenience mirrors used by strict model/data provenance checks.
        "train_sha256": train_sha256,
        "val_sha256": val_sha256,
        "test_sha256": test_sha256,
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("  Saved %s", meta_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    test_size = args.test_size
    val_size = args.val_size
    seed = args.seed
    sample = args.sample
    db_path = args.db_path if args.db_path.is_absolute() else PACKAGE_ROOT / args.db_path
    norms_path = args.norms if args.norms.is_absolute() else PACKAGE_ROOT / args.norms
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else PACKAGE_ROOT / args.output_dir
    )

    log.info("=" * 60)
    log.info("IPIP-BFFM: Prepare Train/Val/Test Splits (canonical_v1, random)")
    log.info("=" * 60)
    log.info(
        "  Config: test_size=%.2f, val_size=%.2f, seed=%d%s",
        test_size,
        val_size,
        seed,
        f", sample={sample}" if sample else "",
    )
    log.info("  DB path:    %s", db_path)
    log.info("  Norms:      %s", norms_path)
    log.info("  Output dir: %s", output_dir)

    if not db_path.exists():
        log.error("Database not found: %s", db_path)
        log.error("Run 02_load_sqlite.py first.")
        return 1
    if not norms_path.exists():
        log.error("Norms artifact not found: %s", norms_path)
        log.error("Run 03_compute_norms.py (make norms) first.")
        return 1

    # --sample loads only the first N respondent ids, so assign_splits partitions
    # a different id set than stage 03 (which always uses the full population).
    # The resulting 'train' rows are NOT the rows the norms were fit on, silently
    # breaking leakage-freeness. Refuse rather than produce an incoherent dataset.
    if sample is not None:
        log.error(
            "--sample changes the respondent id set, so the split no longer matches the "
            "stage-03 norms (which are fit on the full-population train split). The "
            "resulting percentile targets would be leaky/incoherent. Re-run stage 03 on the "
            "same sample, or run without --sample."
        )
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load from SQLite
    log.info("Step 1: Loading data from SQLite...")
    df = load_from_sqlite(db_path, sample=sample)

    # Step 2: Validate percentile computation
    log.info("Step 2: Validating percentile computation...")
    if validate_percentile_computation():
        log.info("  Z-score percentile computation validated.")
    else:
        log.error("  Percentile computation validation FAILED.")
        return 1

    # Step 3: Add percentile columns using train-only norms
    log.info("Step 3: Adding percentile columns (train-only norms)...")
    try:
        assert_norms_match_split(
            norms_path, seed=seed, test_size=test_size, val_size=val_size
        )
        norms = load_norms(norms_path)
    except (ValueError, OSError, json.JSONDecodeError) as e:
        log.error("%s", e)
        return 1
    norms_sha256 = file_sha256(norms_path)
    df = add_percentile_columns(df, norms)
    for domain in DOMAINS:
        pct_col = f"{domain}_percentile"
        if pct_col in df.columns:
            log.info(
                "    %s_percentile: mean=%.1f, std=%.1f, min=%.1f, max=%.1f",
                domain,
                df[pct_col].mean(),
                df[pct_col].std(),
                df[pct_col].min(),
                df[pct_col].max(),
            )

    # Step 4: Plain random split (seed-locked, keyed on respondent_id)
    log.info(
        "Step 4: Random split (train=%.0f%%, val=%.0f%%, test=%.0f%%, seed=%d)...",
        (1 - test_size - val_size) * 100,
        val_size * 100,
        test_size * 100,
        seed,
    )
    try:
        train_df, val_df, test_df = random_split(
            df, test_size=test_size, val_size=val_size, seed=seed
        )
    except ValueError as e:
        log.error("%s", e)
        return 1

    log.info("  Train: %s (%.1f%%)", f"{len(train_df):,}", len(train_df) / len(df) * 100)
    log.info("  Val:   %s (%.1f%%)", f"{len(val_df):,}", len(val_df) / len(df) * 100)
    log.info("  Test:  %s (%.1f%%)", f"{len(test_df):,}", len(test_df) / len(df) * 100)

    # Step 5: Validate splits
    log.info("Step 5: Validating split distributions (KS tests)...")
    validation = validate_splits(train_df, val_df, test_df)
    log_validation(validation)

    # Step 6: Log domain score stats per split
    log.info("Step 6: Domain score statistics per split...")
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        log.info("  %s:", split_name)
        for domain in DOMAINS:
            col = f"{domain}_score"
            if col in split_df.columns:
                log.info(
                    "    %s: mean=%.3f, std=%.3f",
                    domain.upper(),
                    split_df[col].mean(),
                    split_df[col].std(),
                )

    # Step 7: Select columns and write parquet
    log.info("Step 7: Writing parquet files...")

    train_out = select_parquet_columns(train_df)
    val_out = select_parquet_columns(val_df)
    test_out = select_parquet_columns(test_df)

    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"

    train_out.to_parquet(train_path, index=False)
    val_out.to_parquet(val_path, index=False)
    test_out.to_parquet(test_path, index=False)

    log.info("  Saved %s (%s rows)", train_path, f"{len(train_out):,}")
    log.info("  Saved %s (%s rows)", val_path, f"{len(val_out):,}")
    log.info("  Saved %s (%s rows)", test_path, f"{len(test_out):,}")

    # Log file sizes
    for p in [train_path, val_path, test_path]:
        size_mb = p.stat().st_size / 1_048_576
        log.info("    %s: %.1f MB", p.name, size_mb)

    # Step 8: Write metadata
    log.info("Step 8: Writing metadata...")
    write_metadata(
        output_dir,
        df,
        train_df,
        val_df,
        test_df,
        validation,
        seed=seed,
        test_size=test_size,
        val_size=val_size,
        db_path=db_path,
        norms_path=norms_path,
        norms_sha256=norms_sha256,
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        args=args,
    )

    log.info("=" * 60)
    log.info("Data preparation complete.")
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
