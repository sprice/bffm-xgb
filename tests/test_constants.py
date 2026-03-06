"""Tests for lib/constants.py — validate structural invariants."""

from lib.constants import (
    DEFAULT_PARAMS,
    DOMAIN_CSV_TO_INTERNAL,
    DOMAIN_LABELS,
    DOMAINS,
    ITEM_COLUMNS,
    ITEMS_PER_DOMAIN,
    QUANTILE_NAMES,
    QUANTILES,
    REVERSE_KEYED,
)


class TestDomainStructure:
    def test_five_domains(self):
        assert len(DOMAINS) == 5

    def test_items_per_domain_is_10(self):
        assert ITEMS_PER_DOMAIN == 10

    def test_total_item_columns(self):
        assert len(ITEM_COLUMNS) == 50

    def test_column_naming_convention(self):
        """Each column matches {domain}{1-10}."""
        for domain in DOMAINS:
            for i in range(1, ITEMS_PER_DOMAIN + 1):
                assert f"{domain}{i}" in ITEM_COLUMNS

    def test_domain_labels_cover_all_domains(self):
        assert set(DOMAIN_LABELS.keys()) == set(DOMAINS)


class TestQuantileConfig:
    def test_quantile_names_maps_all_quantiles(self):
        assert set(QUANTILE_NAMES.keys()) == set(QUANTILES)

    def test_quantiles_sorted_ascending(self):
        assert QUANTILES == sorted(QUANTILES)


class TestReverseKeyed:
    def test_indices_valid_range(self):
        """All reverse-keyed indices are between 1 and ITEMS_PER_DOMAIN."""
        for domain_key, indices in REVERSE_KEYED.items():
            for idx in indices:
                assert 1 <= idx <= ITEMS_PER_DOMAIN, (
                    f"{domain_key} index {idx} out of range"
                )

    def test_keys_map_to_domains(self):
        """All REVERSE_KEYED keys correspond to a domain (via CSV mapping)."""
        for key in REVERSE_KEYED:
            assert key in DOMAIN_CSV_TO_INTERNAL, f"{key} not in CSV mapping"


class TestDefaultParams:
    def test_expected_xgboost_keys(self):
        expected = {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "reg_alpha",
            "reg_lambda",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
        }
        assert expected.issubset(set(DEFAULT_PARAMS.keys()))
