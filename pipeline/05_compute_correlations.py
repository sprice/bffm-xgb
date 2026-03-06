#!/usr/bin/env python3
"""Compute item-domain correlations, cross-domain info, and select universal first item."""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# Add package root to path for lib imports
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from lib.constants import DOMAINS, DOMAIN_LABELS, ITEMS_PER_DOMAIN, REVERSE_KEYED
from lib.item_info import file_sha256
from lib.provenance import add_provenance_args, build_provenance, relative_to_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data/processed/ext_est")

ITEM_COLUMNS = [f"{d}{i}" for d in DOMAINS for i in range(1, ITEMS_PER_DOMAIN + 1)]
SCORE_COLUMNS = [f"{d}_score" for d in DOMAINS]

REVERSE_KEYED_LOWER = {domain.lower(): items for domain, items in REVERSE_KEYED.items()}

# Item text mapping (from IPIP-BFFM questions)
ITEM_TEXTS = {
    "ext1": "I am the life of the party.",
    "ext2": "I don't talk a lot.",
    "ext3": "I feel comfortable around people.",
    "ext4": "I keep in the background.",
    "ext5": "I start conversations.",
    "ext6": "I have little to say.",
    "ext7": "I talk to a lot of different people at parties.",
    "ext8": "I don't like to draw attention to myself.",
    "ext9": "I don't mind being the center of attention.",
    "ext10": "I am quiet around strangers.",
    "agr1": "I feel little concern for others.",
    "agr2": "I am interested in people.",
    "agr3": "I insult people.",
    "agr4": "I sympathize with others' feelings.",
    "agr5": "I am not interested in other people's problems.",
    "agr6": "I have a soft heart.",
    "agr7": "I am not really interested in others.",
    "agr8": "I take time out for others.",
    "agr9": "I feel others' emotions.",
    "agr10": "I make people feel at ease.",
    "csn1": "I am always prepared.",
    "csn2": "I leave my belongings around.",
    "csn3": "I pay attention to details.",
    "csn4": "I make a mess of things.",
    "csn5": "I get chores done right away.",
    "csn6": "I often forget to put things back in their proper place.",
    "csn7": "I like order.",
    "csn8": "I shirk my duties.",
    "csn9": "I follow a schedule.",
    "csn10": "I am exacting in my work.",
    "est1": "I get stressed out easily.",
    "est2": "I am relaxed most of the time.",
    "est3": "I worry about things.",
    "est4": "I seldom feel blue.",
    "est5": "I am easily disturbed.",
    "est6": "I get upset easily.",
    "est7": "I change my mood a lot.",
    "est8": "I have frequent mood swings.",
    "est9": "I get irritated easily.",
    "est10": "I often feel blue.",
    "opn1": "I have a rich vocabulary.",
    "opn2": "I have difficulty understanding abstract ideas.",
    "opn3": "I have a vivid imagination.",
    "opn4": "I am not interested in abstract ideas.",
    "opn5": "I have excellent ideas.",
    "opn6": "I do not have a good imagination.",
    "opn7": "I am quick to understand things.",
    "opn8": "I use difficult words.",
    "opn9": "I spend time reflecting on things.",
    "opn10": "I am full of ideas.",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute item correlations and item ranking metadata from training split."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory with train.parquet; outputs written here (default: data/processed/ext_est)",
    )
    add_provenance_args(parser)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Correlation computation
# ---------------------------------------------------------------------------

def compute_item_domain_correlations(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Compute Pearson correlation between each item and each domain score.

    For own-domain correlations, uses a corrected score that excludes the item
    itself (avoiding part-whole contamination).

    Returns:
        Dict mapping item_id -> {domain: r_value}
    """
    # Pre-compute domain arrays for corrected own-domain scores
    domain_vals_cache: dict[str, np.ndarray] = {}
    domain_sum_cache: dict[str, np.ndarray] = {}
    domain_count_cache: dict[str, np.ndarray] = {}
    domain_cols_cache: dict[str, list[str]] = {}

    for domain in DOMAINS:
        domain_items = [f"{domain}{i}" for i in range(1, ITEMS_PER_DOMAIN + 1)]
        available = [c for c in domain_items if c in df.columns]
        if available:
            vals = df[available].values.astype(float)
            domain_vals_cache[domain] = vals
            domain_sum_cache[domain] = np.nansum(vals, axis=1)
            domain_count_cache[domain] = np.sum(~np.isnan(vals), axis=1)
            domain_cols_cache[domain] = available

    correlations: dict[str, dict[str, float]] = {}

    item_cols = [c for c in ITEM_COLUMNS if c in df.columns]

    for item_col in tqdm(item_cols, desc="Computing correlations"):
        item_values = df[item_col].values.astype(float)
        item_domain = item_col.rstrip("0123456789")
        item_corrs: dict[str, float] = {}

        for domain in DOMAINS:
            score_col = f"{domain}_score"
            if score_col not in df.columns:
                continue

            if domain == item_domain:
                # Corrected own-domain: exclude this item to avoid part-whole contamination
                if domain not in domain_cols_cache:
                    continue
                cols = domain_cols_cache[domain]
                if item_col not in cols:
                    continue
                j = cols.index(item_col)
                item_in_domain = domain_vals_cache[domain][:, j]
                valid_item = ~np.isnan(item_in_domain)
                corrected_sum = domain_sum_cache[domain] - np.where(valid_item, item_in_domain, 0.0)
                corrected_count = domain_count_cache[domain] - valid_item.astype(int)
                score_values = np.where(corrected_count > 0, corrected_sum / corrected_count, np.nan)
            else:
                # Cross-domain: use full domain score
                score_values = df[score_col].values.astype(float)

            # Remove NaN pairs
            mask = ~np.isnan(item_values) & ~np.isnan(score_values)
            if mask.sum() < 100:
                continue

            r, _ = stats.pearsonr(item_values[mask], score_values[mask])
            item_corrs[domain] = float(r)

        correlations[item_col] = item_corrs

    return correlations


def compute_cross_domain_info(correlations: dict[str, dict[str, float]]) -> dict[str, float]:
    """
    Compute cross-domain information score for each item.

    Cross-domain info = sum of |r| across ALL domains (including own domain).
    Higher values indicate items that provide information about multiple domains.
    """
    cross_info = {}
    for item_id, domain_corrs in correlations.items():
        cross_info[item_id] = sum(abs(r) for r in domain_corrs.values())
    return cross_info


def compute_response_distributions(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute response distribution stats (mean, std, skew, entropy) per item."""
    distributions = {}
    item_cols = [c for c in ITEM_COLUMNS if c in df.columns]

    for item_col in item_cols:
        values = df[item_col].dropna()
        if len(values) < 100:
            continue

        value_counts = values.value_counts(normalize=True).sort_index()
        probs = value_counts.values
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))

        distributions[item_col] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "skew": float(stats.skew(values)),
            "kurtosis": float(stats.kurtosis(values)),
            "entropy": entropy,
        }

    return distributions


def compute_inter_item_correlations(df: pd.DataFrame) -> dict[str, float]:
    """
    Compute mean inter-item correlation (r_bar) for each domain.

    Used in the Spearman-Brown prophecy formula for Cronbach SEM.
    """
    r_bars = {}
    for domain in DOMAINS:
        domain_items = [f"{domain}{i}" for i in range(1, ITEMS_PER_DOMAIN + 1)]
        available = [c for c in domain_items if c in df.columns]

        if len(available) < 2:
            r_bars[domain] = 0.0
            continue

        corr_matrix = df[available].corr(method="pearson").values
        n = corr_matrix.shape[0]
        upper_tri = corr_matrix[np.triu_indices(n, k=1)]
        upper_tri = upper_tri[~np.isnan(upper_tri)]

        r_bars[domain] = float(np.mean(upper_tri)) if len(upper_tri) > 0 else 0.0

    return r_bars


# ---------------------------------------------------------------------------
# Ranking and first-item selection
# ---------------------------------------------------------------------------

def rank_items(
    correlations: dict[str, dict[str, float]],
    cross_info: dict[str, float],
    distributions: dict[str, dict[str, float]],
) -> list[dict]:
    """
    Rank all items by composite score for adaptive selection.

    Composite = cross_domain_info + 0.2 * entropy - 0.1 * |skew|
    """
    pool = []

    for item_id, domain_corrs in correlations.items():
        if item_id not in distributions:
            continue

        dist = distributions[item_id]
        home_domain = item_id.rstrip("0123456789")
        item_num = int(item_id[len(home_domain):])

        info_score = cross_info.get(item_id, 0.0)
        skew_penalty = abs(dist["skew"]) * 0.1
        composite = info_score + dist["entropy"] * 0.2 - skew_penalty

        pool.append({
            "id": item_id,
            "home_domain": home_domain,
            "item_number": item_num,
            "text": ITEM_TEXTS.get(item_id, ""),
            "is_reverse_keyed": item_num in REVERSE_KEYED_LOWER.get(home_domain, []),
            "cross_domain_info": info_score,
            "own_domain_r": domain_corrs.get(home_domain, 0.0),
            "domain_correlations": domain_corrs,
            "entropy": dist["entropy"],
            "skew": dist["skew"],
            "composite_score": composite,
        })

    pool.sort(key=lambda x: x["composite_score"], reverse=True)

    for i, item in enumerate(pool):
        item["rank"] = i + 1

    return pool


def select_first_item(item_pool: list[dict]) -> dict:
    """
    Select the universal first item.

    Criteria:
    1. |skew| < 1.0
    2. entropy > 1.3
    3. Among qualifying items, pick the one with the highest cross_domain_info.
    """
    candidates = [
        item for item in item_pool
        if abs(item["skew"]) < 1.0 and item["entropy"] > 1.3
    ]

    if not candidates:
        log.warning("No items meet first-item criteria; falling back to top composite.")
        return item_pool[0]

    candidates.sort(key=lambda x: x["cross_domain_info"], reverse=True)
    return candidates[0]


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_item_correlations(
    path: Path,
    correlations: dict[str, dict[str, float]],
    source_path: Path,
    n_samples: int,
    provenance: dict[str, Any] | None = None,
) -> None:
    """Write full correlation matrix JSON."""
    provenance_payload = dict(provenance) if provenance is not None else build_provenance(Path(__file__).name)
    provenance_payload.setdefault("source", relative_to_root(source_path))
    provenance_payload.setdefault("source_sha256", file_sha256(source_path))

    output = {
        "source": relative_to_root(source_path),
        "source_sha256": file_sha256(source_path),
        "n_samples": n_samples,
        "n_items": len(correlations),
        "domains": DOMAINS,
        "correlations": correlations,
        "provenance": provenance_payload,
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("  Saved %s", path)


def write_item_info(
    path: Path,
    item_pool: list[dict],
    first_item: dict,
    inter_item_r_bars: dict[str, float] | None = None,
    source_path: Path | None = None,
    source_sha256: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> None:
    """
    Write item_info.json in the same format as output/item_info.json.

    Format:
    {
      "firstItemId": "...",
      "firstItemText": "...",
      "firstItemDomain": "...",
      "itemPool": [
        {
          "id": "...",
          "rank": 1,
          "homeDomain": "...",
          "crossDomainInfo": ...,
          "ownDomainR": ...,
          "domainCorrelations": { ... },
          "isReverseKeyed": true/false
        },
        ...
      ]
    }
    """
    provenance_payload = dict(provenance) if provenance is not None else build_provenance(Path(__file__).name)
    if source_path is not None:
        provenance_payload.setdefault("source", relative_to_root(source_path))
        provenance_payload.setdefault("source_sha256", source_sha256 or file_sha256(source_path))

    output = {
        "firstItemId": first_item["id"],
        "firstItemText": first_item["text"],
        "firstItemDomain": first_item["home_domain"],
        "itemPool": [
            {
                "id": item["id"],
                "rank": item["rank"],
                "homeDomain": item["home_domain"],
                "crossDomainInfo": item["cross_domain_info"],
                "ownDomainR": item["own_domain_r"],
                "domainCorrelations": item["domain_correlations"],
                "isReverseKeyed": item["is_reverse_keyed"],
            }
            for item in item_pool
        ],
        "interItemRBar": inter_item_r_bars,
        "provenance": provenance_payload,
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("  Saved %s", path)


def write_first_item(
    path: Path,
    first_item: dict,
    all_candidates: list[dict],
) -> None:
    """Write first_item.json with the selected first item and top candidates."""
    output = {
        "selected_item": {
            "id": first_item["id"],
            "text": first_item["text"],
            "home_domain": first_item["home_domain"],
            "home_domain_label": DOMAIN_LABELS[first_item["home_domain"]],
            "is_reverse_keyed": first_item["is_reverse_keyed"],
            "cross_domain_info": first_item["cross_domain_info"],
            "own_domain_r": first_item["own_domain_r"],
            "domain_correlations": first_item["domain_correlations"],
            "composite_score": first_item["composite_score"],
        },
        "top_candidates": [
            {
                "id": c["id"],
                "text": c["text"],
                "home_domain": c["home_domain"],
                "cross_domain_info": c["cross_domain_info"],
                "own_domain_r": c["own_domain_r"],
                "composite_score": c["composite_score"],
            }
            for c in all_candidates[:10]
        ],
        "selection_criteria": {
            "primary": "Cross-domain information score (sum |r| across all domains)",
            "filters": ["|skew| < 1.0", "entropy > 1.3"],
        },
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("  Saved %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    data_dir = args.data_dir if args.data_dir.is_absolute() else PACKAGE_ROOT / args.data_dir
    train_path = data_dir / "train.parquet"
    output_dir = data_dir

    log.info("=" * 60)
    log.info("IPIP-BFFM: Compute Correlations & Select First Item")
    log.info("=" * 60)
    log.info("Data dir: %s", data_dir)

    if not train_path.exists():
        log.error("Training data not found: %s", train_path)
        log.error("Run 04_prepare_data.py first.")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load training data
    log.info("Step 1: Loading training data...")
    df = pd.read_parquet(train_path)
    log.info("  Loaded %s rows, %d columns", f"{len(df):,}", len(df.columns))

    # Step 2: Compute item-domain correlations
    log.info("Step 2: Computing item-domain correlations (50 items x 5 domains)...")
    correlations = compute_item_domain_correlations(df)
    log.info("  Computed correlations for %d items", len(correlations))

    # Step 3: Compute cross-domain information scores
    log.info("Step 3: Computing cross-domain information scores...")
    cross_info = compute_cross_domain_info(correlations)

    sorted_items = sorted(cross_info.items(), key=lambda x: x[1], reverse=True)
    log.info("  Top 10 items by cross-domain info:")
    for item_id, score in sorted_items[:10]:
        log.info("    %s: %.3f", item_id, score)

    # Step 4: Compute inter-item correlations (r_bar)
    log.info("Step 4: Computing inter-item correlations (r_bar)...")
    inter_item_r_bars = compute_inter_item_correlations(df)
    for domain in DOMAINS:
        log.info("    %s: r_bar = %.3f", DOMAIN_LABELS[domain], inter_item_r_bars[domain])

    # Step 5: Compute response distributions
    log.info("Step 5: Computing response distributions...")
    distributions = compute_response_distributions(df)
    log.info("  Computed distributions for %d items", len(distributions))

    # Step 6: Rank items for adaptive selection
    log.info("Step 6: Ranking items by composite score...")
    item_pool = rank_items(correlations, cross_info, distributions)

    log.info("  Top 10 items by composite score:")
    for item in item_pool[:10]:
        log.info(
            "    %2d. %s: info=%.3f, own_r=%.3f, entropy=%.2f",
            item["rank"],
            item["id"],
            item["cross_domain_info"],
            item["own_domain_r"],
            item["entropy"],
        )

    # Step 7: Select universal first item
    log.info("Step 7: Selecting universal first item...")
    first_item = select_first_item(item_pool)
    log.info("  Selected: %s", first_item["id"])
    log.info("  Text: \"%s\"", first_item["text"])
    log.info("  Domain: %s", DOMAIN_LABELS[first_item["home_domain"]])
    log.info("  Cross-domain info: %.3f", first_item["cross_domain_info"])
    log.info("  Own domain r: %.3f", first_item["own_domain_r"])

    # Step 8: Domain correlation summary
    log.info("Step 8: Domain correlation summary...")
    for domain in DOMAINS:
        domain_items = [item for item in item_pool if item["home_domain"] == domain]
        own_corrs = [item["own_domain_r"] for item in domain_items]
        if own_corrs:
            log.info(
                "    %s: mean own-domain r = %.3f (range: %.3f - %.3f)",
                DOMAIN_LABELS[domain],
                np.mean(own_corrs),
                min(own_corrs),
                max(own_corrs),
            )

    # Step 9: Write outputs
    log.info("Step 9: Writing output files...")

    source_sha256 = file_sha256(train_path)
    artifact_provenance = build_provenance(
        Path(__file__).name,
        args=args,
        extra={
            "data_dir": relative_to_root(data_dir),
            "source": relative_to_root(train_path),
            "source_sha256": source_sha256,
        },
    )

    corr_path = output_dir / "item_correlations.json"
    write_item_correlations(
        corr_path,
        correlations,
        train_path,
        len(df),
        provenance=artifact_provenance,
    )

    info_path = output_dir / "item_info.json"
    write_item_info(
        info_path,
        item_pool,
        first_item,
        inter_item_r_bars=inter_item_r_bars,
        source_path=train_path,
        source_sha256=source_sha256,
        provenance=artifact_provenance,
    )

    first_path = output_dir / "first_item.json"
    write_first_item(first_path, first_item, item_pool)

    log.info("=" * 60)
    log.info("Correlation analysis complete.")
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
