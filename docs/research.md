# Research Notes

Technical details on model architecture, sparsity augmentation, norms, data, and limitations.

## Model Architecture

- **Algorithm:** XGBoost quantile regression with pinball loss
- **Models:** 15 total (5 domains x 3 quantiles: q05, q50, q95)
- **Input:** 50 features (float32 at inference; float64 during training), one per IPIP-BFFM item; NaN for unanswered items
- **Output:** Raw domain score (1--5 scale), converted to percentile via z-score norms
- **Cross-validation:** 3-fold cross-validation robustness analysis with evaluation split before augmentation
- **Hyperparameters:** Tuned via Optuna TPE search (stage 06); stored in `artifacts/tuned_params.json`

## Sparsity Augmentation

The key idea is **sparsity augmentation**: each training respondent (who answered all 50 items) is augmented 3 times (`n_augmentation_passes=3`), each time with a different random mask that sets a subset of items to NaN. The model only trains on masked data and learns to predict accurately regardless of which items are present.

The reference model uses **focused sparsity** with the A0.1 distribution, which assigns each augmented row to one of five masking buckets:

| Bucket | Proportion | Items Kept | Description                                       |
| ------ | ---------- | ---------- | ------------------------------------------------- |
| 0      | 40%        | 10--20     | Domain-balanced masking (min 2 items per domain)  |
| 1      | 10%        | 20         | Mini-IPIP subset (fixed 4 items x 5 domains)      |
| 2      | 20%        | 21--35     | Moderate sparsity (min 4 items per domain)        |
| 3      | 15%        | 36--50     | Light sparsity (min 4 items per domain)           |
| 4      | 15%        | varies     | Imbalanced patterns (some domains get zero items) |

Within buckets 0--3, item selection is weighted by cross-domain information scores (from step 05), so more informative items are retained more frequently.

Bucket 4 (imbalanced patterns) simulates real-world adaptive behavior where some domains receive many items while others receive none:

- **Greedy-mimicking** (50%): selects exactly the top-K items from the ranked item pool
- **Random-skewed** (30%): drops 1--2 random domains entirely
- **Extreme-skewed** (20%): concentrates items in 1--2 domains, 0--1 items in others

This teaches the model to handle arbitrary missing-item patterns, enabling accurate predictions from as few as 20 items.

## Norms

Raw-score to percentile conversion uses z-score transformation with norms derived from the full cleaned stage-02 SQLite response table (`responses`, OSPP dataset). The single source of truth is `artifacts/ipip_bffm_norms.json` and includes both `norms` (full-50 scoring) and `mini_ipip_norms` (standalone Mini-IPIP scoring); regenerate with `make norms` and validate with `make norms-check`.

## Data

Training data comes from the [Open-Source Psychometrics Project](https://openpsychometrics.org/) (OSPP) dataset:

- **Split:** Stratified train/val/test split (default 70/15/15) using EXT x EST quintile strata
- **Augmentation:** Training set is augmented via 3 sparsity passes (see [Sparsity Augmentation](#sparsity-augmentation))
- **Split before augmentation:** Train/val/test split is performed before augmentation to prevent data leakage
- **RNG seed:** 42

## Limitations

- Norms are derived from self-selected online respondents (OSPP); they may not represent the general population
- Models are trained on English-language IPIP items only
- Accuracy degrades with fewer items; 20 items is the recommended minimum for reliable scoring
- Not intended for clinical diagnosis
