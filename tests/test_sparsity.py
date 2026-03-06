"""Tests for lib/sparsity.py — masking functions."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from lib.constants import DOMAINS, ITEM_COLUMNS, ITEMS_PER_DOMAIN
from lib.sparsity import (
    apply_adaptive_sparsity,
    apply_adaptive_sparsity_balanced,
    apply_focused_sparsity,
    apply_imbalanced_sparsity,
    apply_multipass_sparsity,
    apply_sparsity_single,
)


def _count_non_nan_per_domain(row: pd.Series) -> dict[str, int]:
    """Count non-NaN items per domain in a single row."""
    counts = {}
    for domain in DOMAINS:
        cols = [f"{domain}{i}" for i in range(1, ITEMS_PER_DOMAIN + 1)]
        counts[domain] = int(row[cols].notna().sum())
    return counts


class TestAdaptiveSparsity:
    def test_shape_preserved(self, synthetic_X, item_info):
        result = apply_adaptive_sparsity(synthetic_X.copy(), item_info, rng=np.random.default_rng(0))
        assert result.shape == synthetic_X.shape

    def test_nan_count_in_range(self, synthetic_X, item_info):
        result = apply_adaptive_sparsity(
            synthetic_X.copy(), item_info, min_items=8, max_items=40,
            rng=np.random.default_rng(0),
        )
        non_nan_counts = result.notna().sum(axis=1)
        assert (non_nan_counts >= 8).all()
        assert (non_nan_counts <= 40).all()

    def test_reproducible_with_seed(self, synthetic_X, item_info):
        a = apply_adaptive_sparsity(synthetic_X.copy(), item_info, rng=np.random.default_rng(99))
        b = apply_adaptive_sparsity(synthetic_X.copy(), item_info, rng=np.random.default_rng(99))
        pd.testing.assert_frame_equal(a, b)

    def test_original_values_preserved(self, synthetic_X, item_info):
        """Non-NaN positions should retain their original values."""
        result = apply_adaptive_sparsity(synthetic_X.copy(), item_info, rng=np.random.default_rng(0))
        mask = result.notna()
        np.testing.assert_array_equal(
            result.values[mask.values],
            synthetic_X.values[mask.values],
        )

    def test_min_items_equals_n_cols_no_masking(self, synthetic_X, item_info):
        """When min_items == max_items == n_cols, no NaN values should appear."""
        n_cols = synthetic_X.shape[1]  # 50
        result = apply_adaptive_sparsity(
            synthetic_X.copy(), item_info,
            min_items=n_cols, max_items=n_cols,
            rng=np.random.default_rng(0),
        )
        assert not result.isna().any().any(), (
            "Expected no NaN values when min_items == max_items == n_cols"
        )

    def test_weighted_selection_bias(self, big_synthetic_X, item_info):
        """Low-rank items should be selected more often than high-rank items."""
        result = apply_adaptive_sparsity(
            big_synthetic_X.copy(), item_info, rng=np.random.default_rng(0),
        )
        # Count selection frequency per column
        selection_freq = result.notna().sum(axis=0).values
        # Get ranks in column order
        rank_by_col = {}
        for item in item_info["item_pool"]:
            rank_by_col[item["id"]] = item["rank"]
        ranks = np.array([rank_by_col[col] for col in result.columns])
        # Lower rank number = more informative = should be selected more often
        # Expect negative Spearman correlation (lower rank -> higher frequency)
        rho, _ = stats.spearmanr(ranks, selection_freq)
        assert rho < -0.5, f"Expected strong negative correlation, got rho={rho:.3f}"


class TestAdaptiveSparsityBalanced:
    def test_domain_coverage_guarantee(self, synthetic_X, item_info):
        """Every domain should have >= min_items_per_domain non-NaN items per row."""
        min_per = 4
        result = apply_adaptive_sparsity_balanced(
            synthetic_X.copy(), item_info,
            min_items_per_domain=min_per, rng=np.random.default_rng(0),
        )
        for idx in range(len(result)):
            counts = _count_non_nan_per_domain(result.iloc[idx])
            for domain, count in counts.items():
                assert count >= min_per, (
                    f"Row {idx}, {domain}: {count} items < min {min_per}"
                )

    def test_total_items_in_range(self, synthetic_X, item_info):
        """Phase 2 should fill to min_total_items when Phase 1 alone is insufficient."""
        result = apply_adaptive_sparsity_balanced(
            synthetic_X.copy(), item_info,
            min_items_per_domain=2, min_total_items=25, max_total_items=40,
            rng=np.random.default_rng(0),
        )
        non_nan_counts = result.notna().sum(axis=1)
        assert (non_nan_counts >= 25).all(), (
            f"Min non-NaN count {non_nan_counts.min()} < 25"
        )
        assert (non_nan_counts <= 40).all()

    def test_phase2_weighted_fill(self, big_synthetic_X, item_info):
        """Phase 2 items (beyond per-domain minimums) should be biased by rank.

        After Phase 1 selects min_items_per_domain from each domain (2*5=10),
        Phase 2 fills the remaining slots using cross-domain-weighted selection.
        The Phase 2 items should show a negative rank-frequency correlation,
        proving that the weighting scheme is active.
        """
        result = apply_adaptive_sparsity_balanced(
            big_synthetic_X.copy(), item_info,
            min_items_per_domain=2, min_total_items=25, max_total_items=40,
            rng=np.random.default_rng(0),
        )
        # To isolate Phase 2, re-run Phase 1 separately and subtract
        # Phase 1 gives exactly min_items_per_domain per domain — we can't
        # directly separate them, but we can check that overall selection
        # frequency correlates with rank (Phase 2 dominates the signal since
        # Phase 1 is uniform within domains but Phase 2 is rank-weighted).
        selection_freq = result.notna().sum(axis=0).values
        rank_by_col = {}
        for item in item_info["item_pool"]:
            rank_by_col[item["id"]] = item["rank"]
        ranks = np.array([rank_by_col[col] for col in result.columns])
        rho, _ = stats.spearmanr(ranks, selection_freq)
        # Phase 2 weighted fill should create a negative rank-frequency correlation
        assert rho < -0.3, (
            f"Expected Phase 2 weighted fill to produce negative rank-frequency "
            f"correlation, got rho={rho:.3f}"
        )


    def test_original_values_preserved(self, synthetic_X, item_info):
        """Non-NaN positions should retain their original values (balanced variant)."""
        result = apply_adaptive_sparsity_balanced(
            synthetic_X.copy(), item_info,
            min_items_per_domain=2, min_total_items=20, max_total_items=40,
            rng=np.random.default_rng(0),
        )
        mask = result.notna()
        np.testing.assert_array_equal(
            result.values[mask.values],
            synthetic_X.values[mask.values],
        )


class TestImbalancedSparsity:
    def test_allows_zero_item_domains(self, synthetic_X, item_info):
        """With 200 rows, a meaningful fraction should have at least one 0-item domain.

        Extreme-skewed (20%) generates 0-item domains frequently, and random-skewed
        (30%) drops entire domains. Expect at least 10 rows with a 0-item domain.
        """
        big_X = pd.concat([synthetic_X] * 2, ignore_index=True)  # 200 rows
        result = apply_imbalanced_sparsity(big_X.copy(), item_info, rng=np.random.default_rng(0))
        zero_domain_count = 0
        for idx in range(len(result)):
            counts = _count_non_nan_per_domain(result.iloc[idx])
            if any(c == 0 for c in counts.values()):
                zero_domain_count += 1
        assert zero_domain_count >= 10, (
            f"Expected >= 10 rows with a 0-item domain, got {zero_domain_count}"
        )

    def test_shape_preserved(self, synthetic_X, item_info):
        result = apply_imbalanced_sparsity(synthetic_X.copy(), item_info, rng=np.random.default_rng(0))
        assert result.shape == synthetic_X.shape

    def test_reproducible_with_seed(self, synthetic_X, item_info):
        a = apply_imbalanced_sparsity(synthetic_X.copy(), item_info, rng=np.random.default_rng(7))
        b = apply_imbalanced_sparsity(synthetic_X.copy(), item_info, rng=np.random.default_rng(7))
        pd.testing.assert_frame_equal(a, b)

    def test_greedy_subpattern_selects_top_k(self, big_synthetic_X, item_info):
        """Greedy sub-pattern (50%) should select exactly the top-K items from ranked pool."""
        result = apply_imbalanced_sparsity(
            big_synthetic_X.copy(), item_info, rng=np.random.default_rng(0),
        )
        # Ranked pool order = item_pool order = ITEM_COLUMNS order
        ranked_ids = [item["id"] for item in item_info["item_pool"]]
        ranked_set_by_k = {k: set(ranked_ids[:k]) for k in range(5, 26)}

        greedy_count = 0
        for idx in range(len(result)):
            row = result.iloc[idx]
            non_nan_cols = set(row.index[row.notna()])
            n_items = len(non_nan_cols)
            # Greedy rows have K in [5, 25] and non-NaN cols == top-K of ranked pool
            if 5 <= n_items <= 25 and non_nan_cols == ranked_set_by_k.get(n_items, set()):
                greedy_count += 1

        # With 50% greedy and 500 rows, expect ~250 (std ≈ 11.2)
        # Threshold of 200 is ~4.5 sigma below the mean — safe against flaking
        assert greedy_count >= 200, (
            f"Expected >= 200 greedy-pattern rows, got {greedy_count}"
        )


class TestFocusedSparsity:
    def test_shape_preserved(self, synthetic_X, item_info):
        result = apply_focused_sparsity(
            synthetic_X.copy(), item_info, rng=np.random.default_rng(0),
        )
        assert result.shape == synthetic_X.shape

    def test_mini_ipip_bucket_exact_items(self, big_synthetic_X, item_info, mini_ipip_items):
        """All Mini-IPIP bucket rows should have exactly the 20 correct columns."""
        rng = np.random.default_rng(42)
        result = apply_focused_sparsity(
            big_synthetic_X.copy(), item_info,
            mini_ipip_items=mini_ipip_items,
            include_mini_ipip=True,
            rng=rng,
        )
        mini_cols = set()
        for items in mini_ipip_items.values():
            mini_cols.update(items)

        mini_ipip_rows = []
        for idx in range(len(result)):
            row = result.iloc[idx]
            non_nan_cols = set(row.index[row.notna()])
            if non_nan_cols == mini_cols:
                mini_ipip_rows.append(idx)

        # With 10% bucket and 500 rows, expect ~50
        assert len(mini_ipip_rows) >= 20, (
            f"Expected >= 20 Mini-IPIP rows, got {len(mini_ipip_rows)}"
        )
        # Every matching row must have exactly 20 items
        for idx in mini_ipip_rows:
            row = result.iloc[idx]
            assert row.notna().sum() == 20, (
                f"Row {idx}: expected 20 non-NaN items, got {row.notna().sum()}"
            )

    def test_in_place_mutation(self, synthetic_X, item_info):
        """apply_focused_sparsity modifies X in-place; verify mutation."""
        original = synthetic_X.copy()
        target = synthetic_X.copy()
        apply_focused_sparsity(target, item_info, rng=np.random.default_rng(0))
        assert target.isna().any().any(), "Expected in-place NaN modification"
        assert not original.isna().any().any(), "Original should be untouched"

    def test_imbalanced_bucket_included(self, big_synthetic_X, item_info):
        """With include_imbalanced=True, some rows should have 0-item domains."""
        result = apply_focused_sparsity(
            big_synthetic_X.copy(), item_info,
            include_imbalanced=True,
            rng=np.random.default_rng(0),
        )
        found_zero = False
        for idx in range(len(result)):
            counts = _count_non_nan_per_domain(result.iloc[idx])
            if any(c == 0 for c in counts.values()):
                found_zero = True
                break
        assert found_zero, "No row with a 0-item domain found (imbalanced bucket)"

    def test_bucket_distribution(self, big_synthetic_X, item_info, mini_ipip_items):
        """Verify approximate bucket proportions (50/10/25/15%) within ±10pp.

        Categorizes each row by its item count and domain coverage into buckets:
        - Bucket 0 (50%): 10-20 items, min 2 per domain
        - Bucket 1 (10%): exactly 20 items matching Mini-IPIP pattern
        - Bucket 2 (25%): 21-35 items, min 4 per domain
        - Bucket 3 (15%): 36-50 items, min 4 per domain
        """
        rng = np.random.default_rng(42)
        result = apply_focused_sparsity(
            big_synthetic_X.copy(), item_info,
            mini_ipip_items=mini_ipip_items,
            include_mini_ipip=True,
            rng=rng,
        )
        mini_cols = set()
        for items in mini_ipip_items.values():
            mini_cols.update(items)

        n_rows = len(result)
        bucket_counts = [0, 0, 0, 0]

        for idx in range(n_rows):
            row = result.iloc[idx]
            non_nan_cols = set(row.index[row.notna()])
            n_items = len(non_nan_cols)
            counts = _count_non_nan_per_domain(row)

            # Check buckets in priority order (bucket 1 first since it's exact)
            if non_nan_cols == mini_cols and n_items == 20:
                bucket_counts[1] += 1
            elif 36 <= n_items <= 50 and all(c >= 4 for c in counts.values()):
                bucket_counts[3] += 1
            elif 21 <= n_items <= 35 and all(c >= 4 for c in counts.values()):
                bucket_counts[2] += 1
            elif 10 <= n_items <= 20 and all(c >= 2 for c in counts.values()):
                bucket_counts[0] += 1
            # else: unclassified (edge cases, allow some)

        proportions = [c / n_rows * 100 for c in bucket_counts]
        # Expected: 50%, 10%, 25%, 15% — allow ±10 percentage points
        expected = [50.0, 10.0, 25.0, 15.0]
        for i, (actual, exp) in enumerate(zip(proportions, expected)):
            assert abs(actual - exp) < 10.0, (
                f"Bucket {i}: {actual:.1f}% vs expected {exp:.1f}% "
                f"(counts: {bucket_counts})"
            )

    def test_include_mini_ipip_false(self, big_synthetic_X, item_info, mini_ipip_items):
        """With include_mini_ipip=False, no rows should match Mini-IPIP pattern."""
        result = apply_focused_sparsity(
            big_synthetic_X.copy(), item_info,
            mini_ipip_items=mini_ipip_items,
            include_mini_ipip=False,
            rng=np.random.default_rng(42),
        )
        mini_cols = set()
        for items in mini_ipip_items.values():
            mini_cols.update(items)

        for idx in range(len(result)):
            row = result.iloc[idx]
            non_nan_cols = set(row.index[row.notna()])
            assert non_nan_cols != mini_cols, (
                f"Row {idx} matches Mini-IPIP pattern despite include_mini_ipip=False"
            )

        # Sparsity should still be applied
        assert result.isna().any().any(), "Expected NaN values with sparsity"


class TestSparsitySingle:
    def test_balanced_dispatch_domain_guarantees(self, synthetic_X, item_info):
        """balanced=True with min_items_per_domain=2 should guarantee 2-item coverage."""
        result = apply_sparsity_single(
            synthetic_X.copy(), item_info,
            balanced=True, focused=False,
            min_items_per_domain=2,
            rng=np.random.default_rng(0),
        )
        for idx in range(len(result)):
            counts = _count_non_nan_per_domain(result.iloc[idx])
            for domain, count in counts.items():
                assert count >= 2, (
                    f"Row {idx}, {domain}: {count} items < 2"
                )

    def test_focused_dispatch_matches_direct(self, synthetic_X, item_info):
        """focused=True should route to apply_focused_sparsity with identical output."""
        dispatched = apply_sparsity_single(
            synthetic_X.copy(), item_info,
            balanced=False, focused=True,
            rng=np.random.default_rng(0),
        )
        direct = apply_focused_sparsity(
            synthetic_X.copy(), item_info,
            rng=np.random.default_rng(0),
        )
        pd.testing.assert_frame_equal(dispatched, direct)

    def test_unbalanced_dispatch_matches_direct(self, synthetic_X, item_info):
        """balanced=False, focused=False should route to apply_adaptive_sparsity."""
        dispatched = apply_sparsity_single(
            synthetic_X.copy(), item_info,
            balanced=False, focused=False,
            rng=np.random.default_rng(0),
        )
        direct = apply_adaptive_sparsity(
            synthetic_X.copy(), item_info,
            rng=np.random.default_rng(0),
        )
        pd.testing.assert_frame_equal(dispatched, direct)


class TestMultipassSparsity:
    def test_output_row_count(self, synthetic_X, synthetic_y, item_info):
        n_passes = 3
        X_aug, y_aug = apply_multipass_sparsity(
            synthetic_X, synthetic_y, item_info,
            n_passes=n_passes, balanced=True,
        )
        assert len(X_aug) == n_passes * len(synthetic_X)
        assert len(y_aug) == n_passes * len(synthetic_y)

    def test_columns_preserved(self, synthetic_X, synthetic_y, item_info):
        X_aug, y_aug = apply_multipass_sparsity(
            synthetic_X, synthetic_y, item_info,
            n_passes=2, balanced=True,
        )
        assert list(X_aug.columns) == list(synthetic_X.columns)
        assert list(y_aug.columns) == list(synthetic_y.columns)

    def test_reproducible(self, synthetic_X, synthetic_y, item_info):
        a_X, a_y = apply_multipass_sparsity(
            synthetic_X, synthetic_y, item_info,
            n_passes=2, balanced=True, base_seed=42,
        )
        b_X, b_y = apply_multipass_sparsity(
            synthetic_X, synthetic_y, item_info,
            n_passes=2, balanced=True, base_seed=42,
        )
        pd.testing.assert_frame_equal(a_X, b_X)
        pd.testing.assert_frame_equal(a_y, b_y)

    def test_focused_path(self, synthetic_X, synthetic_y, item_info):
        """Multipass with focused=True should apply focused sparsity distribution.

        Verifies that focused routing actually happened by checking:
        1. Shape and NaN presence (basic)
        2. Some rows have domain-balanced 10-20 item patterns (bucket 0)
        3. Some rows have heavier 36-50 item patterns (bucket 3)
        This proves focused bucket distribution is active, not just simple sparsity.
        """
        X_aug, y_aug = apply_multipass_sparsity(
            synthetic_X, synthetic_y, item_info,
            n_passes=2, focused=True, base_seed=42,
        )
        assert len(X_aug) == 2 * len(synthetic_X)
        assert list(X_aug.columns) == list(synthetic_X.columns)
        assert X_aug.isna().any().any()

        # Check focused bucket distribution properties
        non_nan_counts = X_aug.notna().sum(axis=1)

        # Bucket 0 (50%): 10-20 items with min 2 per domain
        bucket0_rows = X_aug[(non_nan_counts >= 10) & (non_nan_counts <= 20)]
        bucket0_domain_balanced = 0
        for idx in range(len(bucket0_rows)):
            counts = _count_non_nan_per_domain(bucket0_rows.iloc[idx])
            if all(c >= 2 for c in counts.values()):
                bucket0_domain_balanced += 1
        assert bucket0_domain_balanced > 0, (
            "No rows found with domain-balanced 10-20 item pattern (bucket 0)"
        )

        # Bucket 3 (15%): 36-50 items with min 4 per domain
        bucket3_rows = X_aug[(non_nan_counts >= 36) & (non_nan_counts <= 50)]
        bucket3_heavy = 0
        for idx in range(len(bucket3_rows)):
            counts = _count_non_nan_per_domain(bucket3_rows.iloc[idx])
            if all(c >= 4 for c in counts.values()):
                bucket3_heavy += 1
        assert bucket3_heavy > 0, (
            "No rows found with heavy 36-50 item pattern (bucket 3)"
        )

    def test_xy_alignment_after_shuffle(self, item_info):
        """X/y rows must still correspond after the internal shuffle."""
        rng = np.random.default_rng(0)
        n = 50
        X = pd.DataFrame(
            rng.integers(1, 6, size=(n, len(ITEM_COLUMNS))).astype(np.float64),
            columns=ITEM_COLUMNS,
        )
        # Tag each row with its index via y (all domain columns get the same value)
        y = pd.DataFrame({d: np.arange(n, dtype=np.float64) for d in DOMAINS})

        X_aug, y_aug = apply_multipass_sparsity(
            X, y, item_info, n_passes=3, balanced=True, base_seed=42,
        )

        # For each augmented row, y tells us the original X row index.
        # Non-NaN values in X_aug should match the original X at those columns.
        for i in range(len(X_aug)):
            orig_idx = int(y_aug.iloc[i]["ext"])
            x_row = X_aug.iloc[i]
            non_nan_mask = x_row.notna()
            np.testing.assert_array_equal(
                x_row[non_nan_mask].values,
                X.iloc[orig_idx][non_nan_mask].values,
                err_msg=f"Row {i} (orig {orig_idx}): values don't match original",
            )
