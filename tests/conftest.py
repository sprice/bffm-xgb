"""Shared fixtures for lib/ unit tests."""

import numpy as np
import pandas as pd
import pytest

from lib.constants import DOMAINS, ITEM_COLUMNS, ITEMS_PER_DOMAIN


@pytest.fixture
def synthetic_X():
    """100 x 50 DataFrame of random integers 1-5, seeded."""
    rng = np.random.default_rng(0)
    data = rng.integers(1, 6, size=(100, len(ITEM_COLUMNS))).astype(np.float64)
    return pd.DataFrame(data, columns=ITEM_COLUMNS)


@pytest.fixture
def big_synthetic_X():
    """500 x 50 DataFrame of random integers 1-5 for statistical tests."""
    rng = np.random.default_rng(2)
    data = rng.integers(1, 6, size=(500, len(ITEM_COLUMNS))).astype(np.float64)
    return pd.DataFrame(data, columns=ITEM_COLUMNS)


@pytest.fixture
def synthetic_y():
    """100 x 5 DataFrame of random floats 1-5, seeded."""
    rng = np.random.default_rng(1)
    data = rng.uniform(1.0, 5.0, size=(100, len(DOMAINS)))
    return pd.DataFrame(data, columns=DOMAINS)


@pytest.fixture
def item_info():
    """Minimal item_info dict with item_pool list (50 items), shuffled ranks."""
    rng = np.random.default_rng(12345)
    ranks = rng.permutation(50) + 1  # shuffled 1-50
    pool = []
    for i, col in enumerate(ITEM_COLUMNS):
        domain = col[:3]
        pool.append({"id": col, "rank": int(ranks[i]), "homeDomain": domain})
    return {"item_pool": pool}


@pytest.fixture
def mini_ipip_items():
    """Dict mapping domain abbr -> list of 4 item IDs (non-trivial selection)."""
    return {
        domain: [f"{domain}{i}" for i in [2, 5, 7, 9]]
        for domain in DOMAINS
    }
