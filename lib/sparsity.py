"""
Sparsity augmentation functions for IPIP-BFFM adaptive assessment training.

Provides multiple strategies for masking item responses during training to
simulate partial response patterns encountered during adaptive assessment.

Key functions:
- apply_adaptive_sparsity: Weighted random sparsity using item information scores
- apply_adaptive_sparsity_balanced: Guaranteed domain coverage with weighted fill
- apply_imbalanced_sparsity: Allows 0-item domains (greedy, skewed, extreme patterns)
- apply_focused_sparsity: Phase 11a bucket distribution with configurable sub-strategies
- apply_multipass_sparsity: Multi-pass augmentation for training data expansion
- apply_sparsity_single: Dispatch wrapper routing to the appropriate strategy
"""

from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

from .constants import DOMAINS


def apply_adaptive_sparsity(
    X: pd.DataFrame,
    item_info: dict,
    min_items: int = 8,
    max_items: int = 40,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Apply sparsity that mimics adaptive item selection patterns.

    Items with higher cross-domain information are more likely to be selected,
    matching the actual adaptive selection algorithm.

    Uses fully vectorized Gumbel-max trick for fast weighted sampling.

    Args:
        X: Feature matrix with item columns
        item_info: Dict containing item_pool with cross-domain info scores
        min_items: Minimum items to keep per sample (default: 8)
        max_items: Maximum items to keep per sample (default: 40)
        rng: Random number generator (default: creates new with seed 42)

    Returns:
        DataFrame with NaN for unselected items
    """
    # Convert to numpy for faster operations
    X_arr = X.values.astype(np.float64)
    n_rows, n_cols = X_arr.shape
    cols = list(X.columns)

    # Get item pool with cross-domain info scores
    item_pool = item_info.get("item_pool", [])
    item_ranks = {}
    for item in item_pool:
        item_id = item.get("id", "")
        rank = item.get("rank", 50)
        item_ranks[item_id] = rank

    # Build probability weights based on rank (higher rank = lower selection prob)
    ranks = np.array([item_ranks.get(col, 50) for col in cols])
    weights = 1.0 / (ranks + 1)
    weights = weights / weights.sum()
    log_weights = np.log(weights + 1e-10)

    if rng is None:
        rng = np.random.default_rng(42)

    # Pre-generate all random values at once (fully vectorized)
    max_items = min(max_items, n_cols)
    n_items_all = rng.integers(min_items, max_items + 1, size=n_rows)

    # Gumbel-max trick: generate all noise at once
    gumbel_noise = rng.gumbel(size=(n_rows, n_cols))
    perturbed_scores = log_weights + gumbel_noise  # Broadcasting

    # Get ranking of each item per row (argsort of argsort gives ranks)
    # Higher perturbed score = higher rank = more likely to be selected
    item_ranks_per_row = np.argsort(np.argsort(-perturbed_scores, axis=1), axis=1)

    # Create mask: keep items where rank < n_items for that row
    # item_ranks_per_row[i, j] < n_items_all[i] means item j is selected in row i
    keep_mask = item_ranks_per_row < n_items_all[:, np.newaxis]

    # Apply NaN to unselected items
    X_arr[~keep_mask] = np.nan

    # Return as DataFrame with float64 dtype (numpy NaN for XGBoost compatibility)
    return pd.DataFrame(X_arr, columns=cols, index=X.index)


def apply_adaptive_sparsity_balanced(
    X: pd.DataFrame,
    item_info: dict,
    min_items_per_domain: int = 4,
    min_total_items: int = 20,
    max_total_items: int = 40,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Apply sparsity with guaranteed domain coverage.

    Phase 1: Randomly select min_items_per_domain from EACH domain
    Phase 2: Fill remaining slots with cross-domain-weighted selection

    This addresses the training bias where Intellect items were 25x less likely
    to appear in training samples than Extraversion items, causing the opn_q95
    model to collapse.

    Args:
        X: Feature matrix with item columns
        item_info: Dict containing item_pool with cross-domain info scores
        min_items_per_domain: Minimum items to keep from each domain (default: 4)
        min_total_items: Minimum total items to keep (default: 20)
        max_total_items: Maximum total items to keep (default: 40)
        rng: Random number generator (default: creates new with seed 42)

    Returns:
        DataFrame with NaN for unselected items
    """
    if rng is None:
        rng = np.random.default_rng(42)

    X_arr = X.values.astype(np.float64)
    n_rows, n_cols = X_arr.shape
    cols = list(X.columns)

    # Group columns by domain
    domain_cols = {domain: [] for domain in DOMAINS}
    for i, col in enumerate(cols):
        for domain in DOMAINS:
            if col.startswith(domain):
                domain_cols[domain].append(i)
                break

    # Get item pool with cross-domain info scores for Phase 2 weighted selection
    item_pool = item_info.get("item_pool", [])
    item_ranks = {}
    for item in item_pool:
        item_id = item.get("id", "")
        rank = item.get("rank", 50)
        item_ranks[item_id] = rank

    # Build probability weights for Phase 2 (remaining slots)
    ranks = np.array([item_ranks.get(col, 50) for col in cols])
    weights = 1.0 / (ranks + 1)
    weights = weights / weights.sum()
    log_weights = np.log(weights + 1e-10)

    # Pre-generate total items per row
    n_items_all = rng.integers(min_total_items, max_total_items + 1, size=n_rows)

    # For each row, we need to:
    # 1. Select min_items_per_domain from each domain (Phase 1)
    # 2. Fill remaining slots with weighted selection (Phase 2)

    # Phase 1: Guaranteed domain coverage
    # Generate random selections for each domain
    phase1_selections = np.zeros((n_rows, n_cols), dtype=bool)

    for domain, col_indices in domain_cols.items():
        if len(col_indices) == 0:
            continue

        n_domain_items = len(col_indices)
        items_to_select = min(min_items_per_domain, n_domain_items)

        # For each row, randomly select items_to_select items from this domain
        # Use Gumbel-max trick for fast sampling
        domain_noise = rng.gumbel(size=(n_rows, n_domain_items))
        domain_ranks = np.argsort(np.argsort(-domain_noise, axis=1), axis=1)

        # Mark items where rank < items_to_select as selected
        for local_idx, global_idx in enumerate(col_indices):
            phase1_selections[:, global_idx] = domain_ranks[:, local_idx] < items_to_select

    # Count Phase 1 selections per row
    phase1_counts = phase1_selections.sum(axis=1)

    # Phase 2: Fill remaining slots with weighted selection
    # For items NOT already selected in Phase 1, use weighted sampling
    remaining_slots = np.maximum(0, n_items_all - phase1_counts)

    # Create mask for Phase 2 eligible items (not selected in Phase 1)
    phase2_eligible = ~phase1_selections

    # Use Gumbel-max trick with weights for Phase 2
    gumbel_noise = rng.gumbel(size=(n_rows, n_cols))
    # Set score to -inf for already selected items (Phase 1)
    perturbed_scores = np.where(
        phase2_eligible,
        log_weights + gumbel_noise,
        -np.inf
    )

    # Get ranking of each item per row for Phase 2
    item_ranks_per_row = np.argsort(np.argsort(-perturbed_scores, axis=1), axis=1)

    # Phase 2 selection: select top remaining_slots items from eligible pool
    phase2_selections = (item_ranks_per_row < remaining_slots[:, np.newaxis]) & phase2_eligible

    # Final selection: Phase 1 OR Phase 2
    keep_mask = phase1_selections | phase2_selections

    # Apply NaN to unselected items
    X_arr[~keep_mask] = np.nan

    return pd.DataFrame(X_arr, columns=cols, index=X.index)


def apply_imbalanced_sparsity(
    X: pd.DataFrame,
    item_info: dict,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Apply imbalanced sparsity patterns that allow 0-item domains.

    Generates three sub-pattern types:
      - 50% greedy-mimicking: select top-K items from ranked item pool (replicates
        what adaptive_topk produces, including 0-item domains at low K)
      - 30% random-skewed: randomly zero out 1-2 domains, distribute items across rest
      - 20% extreme-skewed: heavily load 1-2 domains (6-8 items), 0-1 for others

    This addresses the train/eval distribution mismatch where apply_focused_sparsity()
    guarantees min 2 items per domain but evaluation strategies like adaptive_topk
    produce 0-item domains (e.g., 0 Intellect items at K<=20).

    Args:
        X: Feature matrix with item columns
        item_info: Dict containing item_pool with cross-domain info scores
        rng: Random number generator

    Returns:
        DataFrame with NaN for unselected items
    """
    if rng is None:
        rng = np.random.default_rng(42)

    X_arr = X.values.astype(np.float64)
    n_rows, n_cols = X_arr.shape
    cols = list(X.columns)
    col_to_idx = {c: i for i, c in enumerate(cols)}

    # Build domain membership lookup
    domain_col_indices = {domain: [] for domain in DOMAINS}
    for i, col in enumerate(cols):
        for domain in DOMAINS:
            if col.startswith(domain):
                domain_col_indices[domain].append(i)
                break

    # Get ranked item pool (sorted by composite score, same as adaptive_topk)
    item_pool = item_info.get("item_pool", [])
    ranked_item_ids = [item["id"] for item in item_pool if item["id"] in col_to_idx]

    # Assign each row to a sub-pattern
    sub_pattern = rng.choice(3, size=n_rows, p=[0.50, 0.30, 0.20])

    # --- Sub-pattern 0: Greedy-mimicking (50%) ---
    # Select top-K items from ranked pool with K ~ Uniform(5, 25)
    # Vectorized: group rows by K value (only 21 unique values: 5-25)
    mask_greedy = sub_pattern == 0
    n_greedy = int(mask_greedy.sum())
    if n_greedy > 0:
        k_values = rng.integers(5, 26, size=n_greedy)
        greedy_row_idx = np.where(mask_greedy)[0]
        # Pre-compute column indices for ranked items
        ranked_col_indices = np.array([col_to_idx[item_id] for item_id in ranked_item_ids])
        # Group rows by K value and batch-apply NaN masks
        for k in range(5, 26):
            k_mask = k_values == k
            if not k_mask.any():
                continue
            rows = greedy_row_idx[k_mask]
            # Columns to NaN = all columns NOT in the top-K ranked items
            selected_cols = set(ranked_col_indices[:k].tolist())
            nan_cols = np.array([j for j in range(n_cols) if j not in selected_cols])
            if len(nan_cols) > 0:
                X_arr[np.ix_(rows, nan_cols)] = np.nan

    # --- Sub-pattern 1: Random-skewed (30%) ---
    # Zero out 1-2 domains entirely, distribute K items across remaining domains
    # Vectorized: group rows by drop configuration (C(5,1)+C(5,2)=15 configs)
    mask_skewed = sub_pattern == 1
    n_skewed = int(mask_skewed.sum())
    if n_skewed > 0:
        skewed_row_idx = np.where(mask_skewed)[0]
        n_domains_to_drop = rng.choice([1, 2], size=n_skewed, p=[0.6, 0.4])
        k_values = rng.integers(5, 21, size=n_skewed)

        # Pre-generate domain drop choices for each row
        # Group by (frozenset of dropped domains) for batch processing
        drop_config_rows = defaultdict(list)
        drop_config_indices = defaultdict(list)
        for i in range(n_skewed):
            domains_to_drop = rng.choice(len(DOMAINS), size=n_domains_to_drop[i], replace=False)
            drop_key = tuple(sorted(domains_to_drop.tolist()))
            drop_config_rows[drop_key].append(i)
            drop_config_indices[drop_key].append(skewed_row_idx[i])

        for drop_key, local_indices in drop_config_rows.items():
            dropped_domain_set = set(DOMAINS[d] for d in drop_key)
            # Collect eligible column indices (not in dropped domains)
            eligible_indices = []
            for domain in DOMAINS:
                if domain not in dropped_domain_set:
                    eligible_indices.extend(domain_col_indices[domain])
            eligible_arr = np.array(eligible_indices)

            rows = np.array(drop_config_indices[drop_key])
            local_k_values = k_values[np.array(local_indices)]

            # For each row in this config, select K items from eligible columns
            for idx_in_batch, (row_idx, k) in enumerate(zip(rows, local_k_values)):
                k = min(int(k), len(eligible_arr))
                if k > 0:
                    selected = set(rng.choice(eligible_arr, size=k, replace=False).tolist())
                else:
                    selected = set()
                nan_cols = np.array([j for j in range(n_cols) if j not in selected])
                if len(nan_cols) > 0:
                    X_arr[row_idx, nan_cols] = np.nan

    # --- Sub-pattern 2: Extreme-skewed (20%) ---
    # Heavily load 1-2 domains (6-8 items each), 0-1 items for others
    # Vectorized: group rows by heavy domain configuration (C(5,1)+C(5,2)=15 configs)
    mask_extreme = sub_pattern == 2
    n_extreme = int(mask_extreme.sum())
    if n_extreme > 0:
        extreme_row_idx = np.where(mask_extreme)[0]
        n_heavy_domains = rng.choice([1, 2], size=n_extreme, p=[0.4, 0.6])

        # Pre-generate heavy domain choices and group by configuration
        heavy_config_rows = defaultdict(list)
        heavy_config_indices = defaultdict(list)
        for i in range(n_extreme):
            heavy_domain_idx = rng.choice(len(DOMAINS), size=n_heavy_domains[i], replace=False)
            heavy_key = tuple(sorted(heavy_domain_idx.tolist()))
            heavy_config_rows[heavy_key].append(i)
            heavy_config_indices[heavy_key].append(extreme_row_idx[i])

        for heavy_key, local_indices in heavy_config_rows.items():
            heavy_domains = set(DOMAINS[d] for d in heavy_key)
            rows = np.array(heavy_config_indices[heavy_key])

            for row_idx in rows:
                selected = set()

                # Heavy domains: 6-8 items each
                for domain in DOMAINS:
                    if domain in heavy_domains:
                        avail = domain_col_indices[domain]
                        n_pick = min(rng.integers(6, 9), len(avail))
                        picked = rng.choice(avail, size=n_pick, replace=False)
                        selected.update(picked)
                    else:
                        # Light domains: 0-1 items
                        avail = domain_col_indices[domain]
                        n_pick = rng.integers(0, 2)  # 0 or 1
                        if n_pick > 0 and len(avail) > 0:
                            picked = rng.choice(avail, size=n_pick, replace=False)
                            selected.update(picked)

                nan_cols = np.array([j for j in range(n_cols) if j not in selected])
                if len(nan_cols) > 0:
                    X_arr[row_idx, nan_cols] = np.nan

    return pd.DataFrame(X_arr, columns=cols, index=X.index)


def apply_focused_sparsity(
    X: pd.DataFrame,
    item_info: dict,
    mini_ipip_items: Optional[dict[str, list[str]]] = None,
    include_mini_ipip: bool = True,
    include_imbalanced: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Apply Phase 11a focused sparsity distribution.

    Default distribution (sums to 100%):
      - 50% of samples: 10-20 items (target assessment range), min 2 per domain
      - 10% of samples: Exact Mini-IPIP pattern (20 items, 4 per domain)
      - 25% of samples: 21-35 items (transition range), min 4 per domain
      - 15% of samples: 36-50 items (full information), min 4 per domain

    With include_imbalanced=True (A0.1 distribution):
      - 40% of samples: 10-20 items (target assessment range), min 2 per domain
      - 10% of samples: Exact Mini-IPIP pattern (20 items, 4 per domain)
      - 20% of samples: 21-35 items (transition range), min 4 per domain
      - 15% of samples: 36-50 items (full information), min 4 per domain
      - 15% of samples: 5-25 items, imbalanced (0-item domains allowed)

    When include_mini_ipip is False, the 10% is redistributed proportionally.

    Args:
        X: Feature matrix with item columns
        item_info: Dict containing item_pool with cross-domain info scores
        mini_ipip_items: Dict mapping domain abbr -> list of Mini-IPIP item IDs
        include_mini_ipip: Whether to inject Mini-IPIP patterns (default: True)
        include_imbalanced: Whether to include imbalanced patterns (default: False)
        rng: Random number generator

    Returns:
        DataFrame with NaN for unselected items
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_rows = len(X)
    # NOTE: This function modifies X in-place. Callers must pass a copy if
    # the original DataFrame needs to be preserved.
    X_result = X

    # Define bucket probabilities (5 buckets when imbalanced, 4 otherwise)
    has_mini_ipip = include_mini_ipip and mini_ipip_items is not None

    if include_imbalanced:
        # A0.1 distribution: balanced majority + 15% imbalanced
        if has_mini_ipip:
            # [balanced 10-20, mini-ipip, balanced 21-35, balanced 36-50, imbalanced]
            bucket_probs = np.array([0.40, 0.10, 0.20, 0.15, 0.15])
        else:
            # No Mini-IPIP: redistribute its 10% proportionally across buckets 0,2,3
            # Base: [0.40, 0, 0.20, 0.15, 0.15], then add 10% to 0/2/3 by ratio
            base = np.array([0.40, 0.0, 0.20, 0.15])
            base = base / base.sum() * 0.85  # Scale to fill 85% (leaving 15% for imbalanced)
            bucket_probs = np.array([base[0], 0.0, base[2], base[3], 0.15])
        n_buckets = 5
    else:
        if has_mini_ipip:
            bucket_probs = np.array([0.50, 0.10, 0.25, 0.15])
        else:
            bucket_probs = np.array([50.0 / 90, 0.0, 25.0 / 90, 15.0 / 90])
        n_buckets = 4

    # Assign each row to a bucket
    bucket_assignments = rng.choice(n_buckets, size=n_rows, p=bucket_probs)

    # Bucket 0: 10-20 items, min 2 per domain (target assessment range)
    mask_b0 = bucket_assignments == 0
    if mask_b0.any():
        child_rng = np.random.default_rng(rng.integers(0, 2**31))
        X_b0 = apply_adaptive_sparsity_balanced(
            X_result.loc[mask_b0].copy(), item_info,
            min_items_per_domain=2, min_total_items=10, max_total_items=20,
            rng=child_rng,
        )
        X_result.loc[mask_b0] = X_b0.values

    # Bucket 1: Exact Mini-IPIP pattern (20 items, 4 per domain)
    mask_b1 = bucket_assignments == 1
    if mask_b1.any() and mini_ipip_items is not None:
        mini_ipip_cols = set()
        for domain_items in mini_ipip_items.values():
            mini_ipip_cols.update(domain_items)
        for col in X_result.columns:
            if col not in mini_ipip_cols:
                X_result.loc[mask_b1, col] = np.nan

    # Bucket 2: 21-35 items, min 4 per domain (transition range)
    mask_b2 = bucket_assignments == 2
    if mask_b2.any():
        child_rng = np.random.default_rng(rng.integers(0, 2**31))
        X_b2 = apply_adaptive_sparsity_balanced(
            X_result.loc[mask_b2].copy(), item_info,
            min_items_per_domain=4, min_total_items=21, max_total_items=35,
            rng=child_rng,
        )
        X_result.loc[mask_b2] = X_b2.values

    # Bucket 3: 36-50 items, min 4 per domain (full information)
    mask_b3 = bucket_assignments == 3
    if mask_b3.any():
        child_rng = np.random.default_rng(rng.integers(0, 2**31))
        X_b3 = apply_adaptive_sparsity_balanced(
            X_result.loc[mask_b3].copy(), item_info,
            min_items_per_domain=4, min_total_items=36, max_total_items=50,
            rng=child_rng,
        )
        X_result.loc[mask_b3] = X_b3.values

    # Bucket 4: Imbalanced patterns (0-item domains allowed)
    if include_imbalanced and n_buckets == 5:
        mask_b4 = bucket_assignments == 4
        if mask_b4.any():
            child_rng = np.random.default_rng(rng.integers(0, 2**31))
            X_b4 = apply_imbalanced_sparsity(
                X_result.loc[mask_b4].copy(), item_info,
                rng=child_rng,
            )
            X_result.loc[mask_b4] = X_b4.values

    return X_result


def apply_multipass_sparsity(
    X: pd.DataFrame,
    y: pd.DataFrame,
    item_info: dict,
    n_passes: int = 3,
    balanced: bool = True,
    base_seed: int = 42,
    min_items_per_domain: int = 4,
    min_total_items: int = 20,
    max_total_items: int = 40,
    focused: bool = False,
    mini_ipip_items: Optional[dict[str, list[str]]] = None,
    include_mini_ipip: bool = True,
    include_imbalanced: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply sparsity augmentation with multiple passes for more diverse training patterns.

    Each pass uses a different random seed, producing different sparsity masks.
    All passes are concatenated, effectively multiplying the training data.

    Args:
        X: Feature matrix
        y: Target matrix
        item_info: Item information dict
        n_passes: Number of augmentation passes (default: 3)
        balanced: Use domain-balanced sparsity
        base_seed: Base random seed (each pass uses base_seed + pass_idx)
        min_items_per_domain: Min items per domain for balanced sparsity
        min_total_items: Min total items for balanced sparsity
        max_total_items: Max total items for balanced sparsity
        focused: Use Phase 11a focused sparsity distribution
        mini_ipip_items: Mini-IPIP item mapping for focused sparsity
        include_mini_ipip: Whether to inject Mini-IPIP patterns
        include_imbalanced: Whether to include imbalanced patterns (0-item domains)

    Returns:
        (X_augmented, y_augmented) with n_passes * len(X) rows
    """
    X_parts = []

    for pass_idx in range(n_passes):
        rng = np.random.default_rng(base_seed + pass_idx)

        if focused:
            X_aug = apply_focused_sparsity(
                X.copy(), item_info,
                mini_ipip_items=mini_ipip_items,
                include_mini_ipip=include_mini_ipip,
                include_imbalanced=include_imbalanced,
                rng=rng,
            )
        elif balanced:
            X_aug = apply_adaptive_sparsity_balanced(
                X.copy(), item_info,
                min_items_per_domain=min_items_per_domain,
                min_total_items=min_total_items,
                max_total_items=max_total_items,
                rng=rng,
            )
        else:
            X_aug = apply_adaptive_sparsity(
                X.copy(), item_info,
                rng=rng,
            )

        X_parts.append(X_aug)

    # Concatenate all passes; tile y instead of copying per pass
    X_combined = pd.concat(X_parts, ignore_index=True)
    y_combined = pd.concat([y] * n_passes, ignore_index=True)

    # Shuffle
    shuffle_idx = np.random.default_rng(base_seed).permutation(len(X_combined))
    X_combined = X_combined.iloc[shuffle_idx].reset_index(drop=True)
    y_combined = y_combined.iloc[shuffle_idx].reset_index(drop=True)

    return X_combined, y_combined


def apply_sparsity_single(
    X: pd.DataFrame,
    item_info: dict,
    balanced: bool = True,
    focused: bool = False,
    mini_ipip_items: Optional[dict[str, list[str]]] = None,
    include_mini_ipip: bool = True,
    include_imbalanced: bool = False,
    min_items_per_domain: int = 4,
    min_total_items: int = 20,
    max_total_items: int = 40,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Dispatch wrapper: apply the appropriate sparsity method to X.

    Routes to apply_focused_sparsity(), apply_adaptive_sparsity_balanced(),
    or apply_adaptive_sparsity() based on flags.

    Args:
        X: Feature matrix with item columns
        item_info: Dict containing item_pool with cross-domain info scores
        balanced: Use domain-balanced sparsity (default: True)
        focused: Use Phase 11a focused sparsity distribution (default: False)
        mini_ipip_items: Mini-IPIP item mapping for focused sparsity
        include_mini_ipip: Whether to inject Mini-IPIP patterns
        include_imbalanced: Whether to include imbalanced patterns
        min_items_per_domain: Min items per domain for balanced sparsity
        min_total_items: Min total items for balanced sparsity
        max_total_items: Max total items for balanced sparsity
        rng: Random number generator

    Returns:
        DataFrame with NaN for unselected items
    """
    if focused:
        return apply_focused_sparsity(
            X, item_info,
            mini_ipip_items=mini_ipip_items,
            include_mini_ipip=include_mini_ipip,
            include_imbalanced=include_imbalanced,
            rng=rng,
        )
    elif balanced:
        return apply_adaptive_sparsity_balanced(
            X, item_info,
            min_items_per_domain=min_items_per_domain,
            min_total_items=min_total_items,
            max_total_items=max_total_items,
            rng=rng,
        )
    else:
        return apply_adaptive_sparsity(X, item_info, rng=rng)
