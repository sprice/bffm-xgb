"""Deterministic, leakage-free train/val/test assignment shared by stages 03 and 04.

The canonical split is a plain random 70/15/15 partition with a fixed seed. Both
the norms stage (03) and the prepare stage (04) must agree on which respondents
are ``train`` so that norms are fit on training rows only; they therefore both
call :func:`assign_splits` with the same respondent ids and seed.

Assignment is keyed on a *stable sort of the respondent ids*, so the result is
identical regardless of the order in which each stage happens to load its rows.
There is no stratification: at this dataset's scale a plain random split is
already balanced on every domain, and stratifying on the target scores (or
computing bin edges over the full dataset before splitting) is avoided.
"""

from __future__ import annotations

import numpy as np

# Canonical split identity (the single split that governs every headline claim).
CANONICAL_SPLIT_ID = "canonical_v1"
SPLIT_SCHEME = "random"
CANONICAL_SEED = 42
CANONICAL_TEST_SIZE = 0.15
CANONICAL_VAL_SIZE = 0.15

SPLIT_LABELS = ("train", "val", "test")


def assign_splits(
    respondent_ids,
    *,
    seed: int = CANONICAL_SEED,
    test_size: float = CANONICAL_TEST_SIZE,
    val_size: float = CANONICAL_VAL_SIZE,
) -> np.ndarray:
    """Return an array of ``"train"``/``"val"``/``"test"`` labels, aligned to the
    order of ``respondent_ids``.

    The assignment is a pure, deterministic function of the *set* of respondent
    ids and the seed: it is computed in a canonical (stable-sorted) id order, so
    two callers that pass the same ids in any order get the same per-respondent
    label. This is what lets stage 03 fit norms on exactly the rows stage 04
    later writes as ``train``.
    """
    ids = np.asarray(respondent_ids)
    # The canonical order comes from a sort of the raw ids, which is dtype
    # sensitive (integers sort numerically, strings lexicographically). Both
    # stages must derive the same train set, so require integer ids: this turns
    # any accidental dtype drift (or NaN-bearing float ids) into a loud failure
    # instead of a silently divergent split.
    if ids.dtype.kind not in ("i", "u"):
        raise ValueError(
            "respondent ids must be integer-typed for a stable, dtype-independent "
            f"split key (got dtype {ids.dtype!r})"
        )
    n = ids.shape[0]
    if n == 0:
        raise ValueError("assign_splits requires at least one respondent id")
    if not (0.0 < test_size < 1.0) or not (0.0 < val_size < 1.0) or test_size + val_size >= 1.0:
        raise ValueError(f"invalid split sizes: test={test_size}, val={val_size}")
    if np.unique(ids).shape[0] != n:
        raise ValueError("respondent ids must be unique for a deterministic split")

    # Canonical order: stable sort by respondent id so independent callers agree.
    order = np.argsort(ids, kind="stable")

    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    if n_test == 0 or n_val == 0 or (n_test + n_val) >= n:
        raise ValueError(f"split sizes leave an empty partition for n={n}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)  # permute positions within the canonical order

    labels_canonical = np.empty(n, dtype=object)
    labels_canonical[perm[:n_test]] = "test"
    labels_canonical[perm[n_test : n_test + n_val]] = "val"
    labels_canonical[perm[n_test + n_val :]] = "train"

    # Map canonical-order labels back to the caller's input order.
    labels = np.empty(n, dtype=object)
    labels[order] = labels_canonical
    return labels
