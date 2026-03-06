"""
Single-source-of-truth constants for the IPIP-BFFM adaptive assessment pipeline.

Contains domain configuration, quantile settings, default hyperparameters,
reverse-keyed item definitions, and column name mappings.
All other modules in this package import constants from here.
"""

# Domain configuration
DOMAINS = ["ext", "agr", "csn", "est", "opn"]
DOMAIN_LABELS = {
    "ext": "Extraversion",
    "agr": "Agreeableness",
    "csn": "Conscientiousness",
    "est": "EmotionalStability",
    "opn": "Intellect",
}
ITEMS_PER_DOMAIN = 10
ITEM_COLUMNS = [f"{d}{i}" for d in DOMAINS for i in range(1, ITEMS_PER_DOMAIN + 1)]

# Quantile configuration
QUANTILES = [0.05, 0.5, 0.95]
QUANTILE_NAMES = {0.05: "q05", 0.5: "q50", 0.95: "q95"}

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
        "default_data_regime": "ext_est",
    },
    "ablation_none": {
        "config": "configs/ablation_none.yaml",
        "model_dir": "models/ablation_none",
        "default_data_regime": "ext_est",
    },
    "ablation_focused": {
        "config": "configs/ablation_focused.yaml",
        "model_dir": "models/ablation_focused",
        "default_data_regime": "ext_est",
    },
    "ablation_stratified": {
        "config": "configs/ablation_stratified.yaml",
        "model_dir": "models/ablation_stratified",
        "default_data_regime": "ext_est_opn",
    },
}
