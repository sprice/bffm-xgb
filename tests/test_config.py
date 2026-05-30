"""Tests for lib/config.py — the variant + _base.yaml merge.

The A8.3 dedup moved the byte-identical keys into configs/_base.yaml. These
tests lock the merged result of each variant to the EXACT pre-refactor config
dict, so the deduplication is provably value-equivalent (no data/hyperparameter
hash change, no retrain).
"""

from pathlib import Path

from lib.config import _deep_merge, load_config_with_base

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

# Frozen snapshots of the original standalone configs (pre-A8.3). The merged
# variant + _base.yaml MUST reproduce these exactly.
EXPECTED = {
    "reference": {
        "name": "reference",
        "description": "Published model: focused + Mini-IPIP + imbalanced sparsity",
        "output_dir": "models/reference",
        "data_dir": "data/processed/canonical_v1",
        "artifacts_dir": "artifacts",
        "require_test_split": True,
        "sparsity": {
            "enabled": True,
            "focused": True,
            "include_mini_ipip": True,
            "include_imbalanced": True,
            "n_augmentation_passes": 3,
        },
        "hyperparameters": {
            "locked_params": "artifacts/tuned_params.json",
            "lock_policy": "strict_data_hash",
        },
        "training": {"random_state": 42, "n_jobs": 16},
        "validation": {
            "min_pearson_r": 0.90,
            "min_coverage_90": 0.88,
            "per_domain": {"min_pearson_r": 0.90, "min_coverage_90": 0.88},
            "sparse_20": {
                "enabled": True,
                "min_pearson_r": 0.90,
                "min_coverage_90": 0.88,
                "per_domain": {"min_pearson_r": 0.85, "min_coverage_90": 0.84},
            },
        },
    },
    "ablation_none": {
        "name": "ablation_none",
        "description": "Ablation: no sparsity augmentation (baseline)",
        "output_dir": "models/ablation_none",
        "data_dir": "data/processed/canonical_v1",
        "artifacts_dir": "artifacts",
        "require_test_split": True,
        "sparsity": {"enabled": False},
        "hyperparameters": {
            "locked_params": "artifacts/tuned_params.json",
            "lock_policy": "reference_model_hash",
            "reference_model_dir": "models/reference",
        },
        "training": {"random_state": 42, "n_jobs": 16},
        "validation": {
            "min_pearson_r": 0.80,
            "min_coverage_90": 0.80,
            "per_domain": {"min_pearson_r": 0.74, "min_coverage_90": 0.72},
            "sparse_20": {"enabled": False},
        },
    },
    "ablation_focused": {
        "name": "ablation_focused",
        "description": "Ablation: focused sparsity only (no imbalanced patterns)",
        "output_dir": "models/ablation_focused",
        "data_dir": "data/processed/canonical_v1",
        "artifacts_dir": "artifacts",
        "require_test_split": True,
        "sparsity": {
            "enabled": True,
            "focused": True,
            "include_mini_ipip": True,
            "include_imbalanced": False,
            "n_augmentation_passes": 3,
        },
        "hyperparameters": {
            "locked_params": "artifacts/tuned_params.json",
            "lock_policy": "reference_model_hash",
            "reference_model_dir": "models/reference",
        },
        "training": {"random_state": 42, "n_jobs": 16},
        "validation": {
            "min_pearson_r": 0.88,
            "min_coverage_90": 0.85,
            "per_domain": {"min_pearson_r": 0.85, "min_coverage_90": 0.82},
            "sparse_20": {
                "enabled": True,
                "min_pearson_r": 0.87,
                "min_coverage_90": 0.84,
                "per_domain": {"min_pearson_r": 0.80, "min_coverage_90": 0.78},
            },
        },
    },
}


class TestDeepMerge:
    def test_override_wins_on_scalars(self):
        assert _deep_merge({"a": 1, "b": 2}, {"b": 3}) == {"a": 1, "b": 3}

    def test_nested_dicts_merge_key_by_key(self):
        merged = _deep_merge({"x": {"a": 1, "b": 2}}, {"x": {"b": 3, "c": 4}})
        assert merged == {"x": {"a": 1, "b": 3, "c": 4}}

    def test_does_not_mutate_inputs(self):
        base = {"x": {"a": 1}}
        _deep_merge(base, {"x": {"b": 2}})
        assert base == {"x": {"a": 1}}


class TestConfigEquivalence:
    def test_each_variant_matches_pre_refactor_config(self):
        for variant, expected in EXPECTED.items():
            merged = load_config_with_base(CONFIGS_DIR / f"{variant}.yaml")
            assert merged == expected, f"{variant} merged config drifted from the original"

    def test_shared_keys_live_only_in_base(self):
        # The dedup is real: variant files must NOT re-declare the shared keys.
        import yaml

        for variant in EXPECTED:
            raw = yaml.safe_load((CONFIGS_DIR / f"{variant}.yaml").read_text())
            assert "data_dir" not in raw
            assert "artifacts_dir" not in raw
            assert "require_test_split" not in raw
            assert "random_state" not in raw.get("training", {})
            assert "locked_params" not in raw.get("hyperparameters", {})
