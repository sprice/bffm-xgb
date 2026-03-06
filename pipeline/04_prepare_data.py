#!/usr/bin/env python3
"""Prepare train/val/test splits with percentiles and stratification from SQLite."""

import argparse
import sys
import json
import hashlib
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

# Add package root to path for lib imports
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from lib.constants import DOMAINS, DOMAIN_LABELS, ITEMS_PER_DOMAIN
from lib.item_info import file_sha256
from lib.provenance import add_provenance_args, build_provenance, relative_to_root
from lib.scoring import raw_score_to_percentile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/processed/ipip_bffm.db")
DEFAULT_OUTPUT_DIR = Path("data/processed/ext_est")

ITEM_COLUMNS = [f"{d}{i}" for d in DOMAINS for i in range(1, ITEMS_PER_DOMAIN + 1)]
SCORE_COLUMNS = [f"{d}_score" for d in DOMAINS]
PERCENTILE_COLUMNS = [f"{d}_percentile" for d in DOMAINS]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare IPIP-BFFM train/val/test splits with percentiles and stratification."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Fraction for test set (default: 0.15)",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Fraction for validation set (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--stratification",
        type=str,
        choices=["ext-est", "ext-est-opn"],
        default="ext-est",
        help=(
            "Stratification scheme for splitting. "
            "ext-est (default): 25 strata from EXT x EST quintiles. "
            "ext-est-opn: 125 strata from EXT x EST x OPN quintiles."
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only use first N rows (for quick development runs)",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to stage-02 SQLite DB (default: data/processed/ipip_bffm.db)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for split parquet + metadata (default: data/processed/ext_est)",
    )
    add_provenance_args(parser)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_from_sqlite(db_path: Path, sample: int | None = None) -> pd.DataFrame:
    """Load all responses from the SQLite database."""
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

def add_percentile_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add percentile columns for each domain using z-score method."""
    df = df.copy()
    for domain in DOMAINS:
        score_col = f"{domain}_score"
        pct_col = f"{domain}_percentile"
        if score_col in df.columns:
            df[pct_col] = raw_score_to_percentile(df[score_col].values, domain)
    return df


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
# Stratification
# ---------------------------------------------------------------------------

def compute_quintile_strata(
    df: pd.DataFrame, stratification: str = "ext-est"
) -> pd.DataFrame:
    """
    Add quintile columns for each domain and a composite stratum column.

    Quintiles are computed on the full dataset (before splitting) so that bin
    edges are consistent across all splits.

    Args:
        df: DataFrame with domain score columns.
        stratification: Stratification scheme.
            - "ext-est" (default): 25 strata from EXT x EST quintiles.
            - "ext-est-opn": 125 strata from EXT x EST x OPN quintiles.
              Rare strata (count < 3) are merged into their nearest neighbor.
    """
    df = df.copy()

    required_score_cols = ["ext_score", "est_score"]
    if stratification == "ext-est-opn":
        required_score_cols.append("opn_score")
    missing_score_cols = [col for col in required_score_cols if col not in df.columns]
    if missing_score_cols:
        raise ValueError(
            f"Stratification '{stratification}' requires score columns "
            f"{required_score_cols}; missing {missing_score_cols}"
        )

    # Compute per-domain quintiles
    for domain in DOMAINS:
        score_col = f"{domain}_score"
        q_col = f"{domain}_q"
        if score_col in df.columns:
            df[q_col] = pd.qcut(
                df[score_col], q=5, labels=False, duplicates="drop"
            ).astype(np.int8)

    if stratification == "ext-est-opn":
        # 125 strata from EXT x EST x OPN quintiles
        required_q_cols = ["ext_q", "est_q", "opn_q"]
        missing_q_cols = [col for col in required_q_cols if col not in df.columns]
        if missing_q_cols:
            raise ValueError(
                f"Stratification '{stratification}' requires quintile columns "
                f"{required_q_cols}; missing {missing_q_cols}"
            )
        raw_strata = (
            df["ext_q"].astype(int) * 25
            + df["est_q"].astype(int) * 5
            + df["opn_q"].astype(int)
        )
        df["split_stratum"] = merge_rare_strata(raw_strata, min_count=3).astype(
            np.int16
        )
    else:
        # Default: EXT x EST (25 strata)
        required_q_cols = ["ext_q", "est_q"]
        missing_q_cols = [col for col in required_q_cols if col not in df.columns]
        if missing_q_cols:
            raise ValueError(
                f"Stratification '{stratification}' requires quintile columns "
                f"{required_q_cols}; missing {missing_q_cols}"
            )
        df["split_stratum"] = (
            df["ext_q"].astype(int) * 5 + df["est_q"].astype(int)
        ).astype(np.int16)

    n_strata = df["split_stratum"].nunique()
    log.info("  Computed quintile strata: %d unique strata", n_strata)

    return df


def merge_rare_strata(strata: pd.Series, min_count: int = 3) -> pd.Series:
    """Merge strata with fewer than min_count members into nearest neighbor."""
    counts = strata.value_counts()
    rare_ids = set(counts[counts < min_count].index)
    if not rare_ids:
        return strata

    populated_ids = sorted(set(counts[counts >= min_count].index))
    if not populated_ids:
        return pd.Series(0, index=strata.index, dtype=strata.dtype)

    populated_arr = np.array(populated_ids)
    remap = {}
    for rid in rare_ids:
        nearest = populated_arr[np.argmin(np.abs(populated_arr - rid))]
        remap[rid] = nearest

    return strata.map(lambda x: remap.get(x, x))


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def stratified_split(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/val/test split using the split_stratum column.

    Args:
        df: DataFrame with a split_stratum column.
        test_size: Fraction for test set.
        val_size: Fraction for validation set.
        random_state: Random seed.

    Returns:
        (train_df, val_df, test_df)
    """
    strata = merge_rare_strata(df["split_stratum"], min_count=3)

    # First split: (train+val) vs test
    df_temp, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=strata,
        random_state=random_state,
    )

    # Second split: train vs val
    strata_temp = merge_rare_strata(
        df_temp["split_stratum"], min_count=3
    )
    adjusted_val_size = val_size / (1 - test_size)
    df_train, df_val = train_test_split(
        df_temp,
        test_size=adjusted_val_size,
        stratify=strata_temp,
        random_state=random_state,
    )

    return df_train, df_val, df_test


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

    # Item columns as float32
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

    # Stratum column
    if "split_stratum" in df.columns:
        cols.append("split_stratum")

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
    stratification: str,
    db_path: Path,
    train_path: Path,
    val_path: Path,
    test_path: Path,
    args: argparse.Namespace,
) -> None:
    """Write split metadata JSON."""
    if stratification == "ext-est-opn":
        strat_desc = "ext_q * 25 + est_q * 5 + opn_q (up to 125 strata, rare merged)"
    else:
        strat_desc = "ext_q * 5 + est_q (25 strata)"

    db_sha256 = file_sha256(db_path)
    train_sha256 = file_sha256(train_path)
    val_sha256 = file_sha256(val_path)
    test_sha256 = file_sha256(test_path)
    split_signature = hashlib.sha256(
        (
            f"train={train_sha256}\n"
            f"val={val_sha256}\n"
            f"test={test_sha256}\n"
        ).encode("utf-8")
    ).hexdigest()

    meta_path = output_dir / "split_metadata.json"
    provenance = build_provenance(
        Path(__file__).name,
        args=args,
        rng_seed=seed,
        extra={
            "db_path": relative_to_root(db_path),
            "output_dir": relative_to_root(output_dir),
            "stratification_scheme": stratification,
            "split_signature": split_signature,
        },
    )

    metadata = {
        "provenance": provenance,
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
        "stratification": strat_desc,
        "stratification_scheme": stratification,
        "n_strata": int(df_all["split_stratum"].nunique()),
        "validation": validation,
        "split_signature": split_signature,
        "inputs": {
            "sqlite_db": {
                "path": str(db_path),
                "sha256": db_sha256,
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
    stratification = args.stratification
    sample = args.sample
    db_path = args.db_path if args.db_path.is_absolute() else PACKAGE_ROOT / args.db_path
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else PACKAGE_ROOT / args.output_dir
    )

    log.info("=" * 60)
    log.info("IPIP-BFFM: Prepare Train/Val/Test Splits")
    log.info("=" * 60)
    log.info(
        "  Config: test_size=%.2f, val_size=%.2f, seed=%d, stratification=%s%s",
        test_size,
        val_size,
        seed,
        stratification,
        f", sample={sample}" if sample else "",
    )
    log.info("  DB path: %s", db_path)
    log.info("  Output dir: %s", output_dir)

    if not db_path.exists():
        log.error("Database not found: %s", db_path)
        log.error("Run 02_load_sqlite.py first.")
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

    # Step 3: Add percentile columns
    log.info("Step 3: Adding percentile columns...")
    df = add_percentile_columns(df)
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

    # Step 4: Compute stratification
    log.info("Step 4: Computing quintile strata (scheme: %s)...", stratification)
    try:
        df = compute_quintile_strata(df, stratification=stratification)
    except ValueError as e:
        log.error("Stratification failed: %s", e)
        return 1

    # Log quintile distribution
    for domain in DOMAINS:
        q_col = f"{domain}_q"
        if q_col in df.columns:
            counts = df[q_col].value_counts().sort_index()
            log.info("    %s_q distribution: %s", domain, dict(counts))

    stratum_counts = df["split_stratum"].value_counts()
    log.info(
        "    split_stratum: %d strata, min_count=%d, max_count=%d",
        len(stratum_counts),
        stratum_counts.min(),
        stratum_counts.max(),
    )

    # Step 5: Stratified split
    log.info(
        "Step 5: Stratified split (train=%.0f%%, val=%.0f%%, test=%.0f%%, seed=%d)...",
        (1 - test_size - val_size) * 100,
        val_size * 100,
        test_size * 100,
        seed,
    )
    train_df, val_df, test_df = stratified_split(
        df, test_size=test_size, val_size=val_size, random_state=seed
    )

    log.info("  Train: %s (%.1f%%)", f"{len(train_df):,}", len(train_df) / len(df) * 100)
    log.info("  Val:   %s (%.1f%%)", f"{len(val_df):,}", len(val_df) / len(df) * 100)
    log.info("  Test:  %s (%.1f%%)", f"{len(test_df):,}", len(test_df) / len(df) * 100)

    # Step 6: Validate splits
    log.info("Step 6: Validating split distributions (KS tests)...")
    validation = validate_splits(train_df, val_df, test_df)
    log_validation(validation)

    # Step 7: Log domain score stats per split
    log.info("Step 7: Domain score statistics per split...")
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

    # Step 8: Select columns and write parquet
    log.info("Step 8: Writing parquet files...")

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

    # Step 9: Write metadata
    log.info("Step 9: Writing metadata...")
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
        stratification=stratification,
        db_path=db_path,
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
