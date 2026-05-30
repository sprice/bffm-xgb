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

Raw-score to percentile conversion uses z-score transformation with norms derived from the **training split** of the cleaned stage-02 SQLite response table (`responses`, OSPP dataset), so held-out validation/test rows do not leak into the percentile targets. The single source of truth is `artifacts/ipip_bffm_norms.json` and includes both `norms` (full-50 scoring) and `mini_ipip_norms` (standalone Mini-IPIP scoring); regenerate with `make norms` and validate with `make norms-check`.

## Data

Training data comes from the [Open-Source Psychometrics Project](https://openpsychometrics.org/) (OSPP) dataset:

- **Split:** Single plain random train/val/test split (70/15/15, seed-locked) — the canonical `canonical_v1` partition used for every headline claim. At this dataset's scale a random split is already balanced on every domain, so no target stratification is applied.
- **Augmentation:** Training set is augmented via 3 sparsity passes (see [Sparsity Augmentation](#sparsity-augmentation))
- **Split before augmentation:** Train/val/test split is performed before augmentation to prevent data leakage
- **RNG seed:** 42

## Evaluation

All accuracy metrics are computed on the held-out `canonical_v1` test split (*N* = 90,499). "Overall *r*" is a **respondent-pooled** Pearson correlation: the five domains' predicted and true percentile vectors are stacked into a single length-5*N* vector before correlating (numerically ≈ the mean of the per-domain *r*, because every domain is on a common 0--100 percentile scale). Prediction intervals are the raw q05/q95 quantile spreads (no post-hoc width adjustment; every fitted `scale_factor` is 1.0), validated to ~90% empirical coverage at the 20-item operating point.

Headline 20-item numbers refer to the **fixed, pre-specified domain-balanced form** (top-4 items per domain — the deployed web form), not a post-hoc best-of-grid selection. The model's *general* partial-response accuracy under random balanced 20-item masking is lower (*r* ≈ .909). The full-50 self-recovery *r* ≈ 1 reflects score recovery against a target computed from the same 50 items, not external validity.

**Internal-consistency reliability.** Stage 05 computes Cronbach's alpha (raw + standardized), the mean inter-item correlation, and McDonald's omega for each domain across three forms — the full 10-item domains, the deployed domain-balanced 20-item form, and the Mini-IPIP 4-item form — on the training split, written to `data/processed/canonical_v1/reliability.json` and surfaced in [NOTES.md](../notes/NOTES.md). Reliability bounds how high score-recovery *r* can plausibly go.

## Limitations

- Norms are derived from self-selected online respondents (OSPP); they may not represent the general population
- Models are trained on English-language IPIP items only
- The deployed 20-item domain-balanced Emotional Stability subscale (est1, est6, est7, est8) is composed entirely of reverse-keyed items. This maximizes within-domain discrimination but makes the EST short-form score vulnerable to acquiescence (yea-saying) response bias; the other four domains mix keyed directions, and the full 50-item assessment is unaffected. The selection rule (`_select_domain_balanced`) ranks purely by |own-domain *r*|, which is why this domain is single-keyed.
- Accuracy degrades with fewer items; 20 items is the recommended minimum for reliable scoring
- Not intended for clinical diagnosis
