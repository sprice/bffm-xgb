"""
Single-source-of-truth constants for the IPIP-BFFM sparse quantile model pipeline.

Contains domain configuration, quantile settings, default hyperparameters,
reverse-keyed item definitions, and column name mappings.
All other modules in this package import constants from here.
"""

# Domain configuration
DOMAINS = ["ext", "agr", "csn", "est", "opn"]
# Compact internal labels (no spaces) — used as artifact/CSV keys and code labels.
DOMAIN_LABELS = {
    "ext": "Extraversion",
    "agr": "Agreeableness",
    "csn": "Conscientiousness",
    "est": "EmotionalStability",
    "opn": "Intellect",
}
# Human-facing display labels for docs/model cards/notes. Single source of truth
# so the doc generator, the notes generator, and the model-card template
# (pipeline/11) cannot drift apart (they previously each hardcoded these).
DOMAIN_DISPLAY_LABELS = {
    "ext": "Extraversion",
    "agr": "Agreeableness",
    "csn": "Conscientiousness",
    "est": "Emotional Stability",
    "opn": "Intellect/Imagination",
}
ITEMS_PER_DOMAIN = 10
ITEM_COLUMNS = [f"{d}{i}" for d in DOMAINS for i in range(1, ITEMS_PER_DOMAIN + 1)]

# Quantile configuration
QUANTILES = [0.05, 0.5, 0.95]
QUANTILE_NAMES = {0.05: "q05", 0.5: "q50", 0.95: "q95"}
# Ordered quantile-name list (single source of truth for the ["q05","q50","q95"]
# loops that the pipeline stages and the shared model loader iterate over).
QUANTILE_NAME_LIST = [QUANTILE_NAMES[q] for q in QUANTILES]

# On-disk trained-model filename stem: f"{MODEL_STEM}_{domain}_{q}.joblib".
# The model is a sparse quantile regressor, not an adaptive selector (adaptive
# selection is the documented negative result), so the neutral stem is "quantile".
# LEGACY_MODEL_STEM is the pre-canonical_v1 name kept only as a read fallback.
MODEL_STEM = "quantile"
LEGACY_MODEL_STEM = "adaptive"

# Default hyperparameters
DEFAULT_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 6,
    "learning_rate": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
}

DEFAULT_EARLY_STOPPING_ROUNDS = 25

# --------------------------------------------------------------------------- #
# Hand-set training / assessment policy constants (NOT tuned).
#
# These are deliberate design choices of the training and adaptive-assessment
# logic. They were previously hardcoded as numeric literals in pipeline/06,
# pipeline/07 and pipeline/10 *and* re-typed in the docs/web prose, which made
# them silent drift risks. They now live here once; the pipeline stages and the
# doc generator (scripts/generate_doc_data.py -> repoFacts) both read them, so a
# change is made in exactly one place. Config content is not hashed, so editing
# these forces no retrain.
# --------------------------------------------------------------------------- #

# Stage 06 (pipeline/06_tune.py) deployment-aligned Optuna objective:
#   composite = sparse20_weight*mean_r_sparse20 + full50_weight*mean_r_full
#               - sparse20_penalty_weight*max(0, sparse20_penalty_floor - min_r_sparse20)
#               - full50_penalty_weight*max(0, full50_penalty_floor - mean_r_full)
TUNING_OBJECTIVE = {
    "sparse20_weight": 0.80,
    "full50_weight": 0.20,
    "sparse20_penalty_weight": 2.0,
    "sparse20_penalty_floor": 0.85,
    "full50_penalty_weight": 1.0,
    "full50_penalty_floor": 0.95,
}

# Conservative full-50-only fallback objective used by stage 06 when sparse-20
# evaluation is unavailable: composite = mean_r_full - penalty_weight*max(0,
# min_r_floor - min_r_full). Kept separate from TUNING_OBJECTIVE because it is a
# distinct branch and is not surfaced in the docs/web.
TUNING_OBJECTIVE_FALLBACK = {
    "penalty_weight": 1.5,
    "min_r_floor": 0.90,
}

# Stage 07 (pipeline/07_train.py) prediction-interval coverage calibration. The
# 90% PI is scaled toward the nominal target when observed coverage falls below
# coverage_low or rises above coverage_high; coverage_floor clamps the maximum
# upward scale (avoids dividing by a tiny observed coverage).
CALIBRATION_POLICY = {
    "coverage_low": 0.85,
    "coverage_high": 0.95,
    "target_coverage": 0.90,
    "coverage_floor": 0.5,
}

# Stage 10 (pipeline/10_simulate.py) adaptive-stopping policy. The SEM check is
# gated on min_items_per_domain (a hard floor of 4*5 = 20 items), so in practice
# every respondent stops at exactly 20 items.
ADAPTIVE_STOP = {
    "sem_threshold": 0.45,
    "min_items_per_domain": 4,
}

# Pipeline execution defaults
DEFAULT_STAGE07_CV_FOLDS = 3
DEFAULT_RESEARCH_EVAL_PARALLEL = 4
DEFAULT_LOCAL_CV_PARALLEL_FOLDS = 1
DEFAULT_REMOTE_CV_PARALLEL_FOLDS = 2

# Reverse-keyed items (1-indexed item numbers within each domain)
# Source: https://ipip.ori.org/newBigFive5broadKey.htm
REVERSE_KEYED = {
    "EXT": [2, 4, 6, 8, 10],
    "EST": [1, 3, 5, 6, 7, 8, 9, 10],
    "AGR": [1, 3, 5, 7],
    "CSN": [2, 4, 6, 8],
    "OPN": [2, 4, 6],
}

# Column name mappings (uppercase CSV -> lowercase internal)
DOMAIN_CSV_TO_INTERNAL = {
    "EXT": "ext",
    "EST": "est",
    "AGR": "agr",
    "CSN": "csn",
    "OPN": "opn",
}

# Model variant registry (reference + ablations)
VARIANTS = {
    "reference": {
        "config": "configs/reference.yaml",
        "model_dir": "models/reference",
        "default_data_regime": "canonical_v1",
    },
    "ablation_none": {
        "config": "configs/ablation_none.yaml",
        "model_dir": "models/ablation_none",
        "default_data_regime": "canonical_v1",
    },
    "ablation_focused": {
        "config": "configs/ablation_focused.yaml",
        "model_dir": "models/ablation_focused",
        "default_data_regime": "canonical_v1",
    },
}

# The single variant produced by a reference-only pipeline run (the rest are
# ablations). The --reference-only paths in build_research_summary /
# generate_notes_data / check_provenance use this as the one source of truth for
# "which variant survives reference-only" instead of hardcoding the literal.
REFERENCE_VARIANT = "reference"


def reference_only_variants() -> dict[str, dict[str, str]]:
    """VARIANTS filtered to just the reference variant.

    Lets a single-variant run scope its cross-variant logic without being held to
    the all-variants completeness contract. Returns the same value type as VARIANTS
    (a slice), so callers that iterate it stay type-identical to the full loop.
    """
    return {REFERENCE_VARIANT: VARIANTS[REFERENCE_VARIANT]}
