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

The key idea is **sparsity augmentation**: each fit respondent (who answered all 50 items) is augmented several times (`n_augmentation_passes`), each time with a different random mask that sets a subset of items to NaN. The model only trains on masked data and learns to predict accurately regardless of which items are present. Augmentation is applied only to the fit rows: a 15% early-stopping hold-out is carved out of the training rows *before* augmentation, so just the remaining fit respondents are tripled (`n_train_augmented` in the reference config):

<!-- BEGIN GENERATED: augmentation-counts -->
- **Training rows (canonical split):** 422,326
- **Early-stopping hold-out (split off *before* augmentation):** 63,349 rows
- **Fit respondents (the rows that get augmented):** 358,977
- **Augmentation passes:** 3 (each fit respondent re-masked 3×)
- **Augmented training rows:** 358,977 × 3 = 1,076,931
<!-- END GENERATED: augmentation-counts -->

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

The full-50 per-domain reference statistics (training split) used for the z-score transform:

<!-- BEGIN GENERATED: norms-table -->
| Domain                | Mean   | SD     |
| --------------------- | ------ | ------ |
| Extraversion          | 2.9146 | 0.9112 |
| Agreeableness         | 3.7590 | 0.7366 |
| Conscientiousness     | 3.3426 | 0.7390 |
| Emotional Stability   | 2.9190 | 0.8611 |
| Intellect/Imagination | 3.9387 | 0.6183 |
<!-- END GENERATED: norms-table -->

Worked example:

<!-- BEGIN GENERATED: norms-worked-example -->
An Extraversion raw score of 3.80 maps to *z* = (3.80 − 2.9146) / 0.9112 ≈ 0.972, i.e. about the 83.4th percentile of the reference sample.
<!-- END GENERATED: norms-worked-example -->

(The percentile transform uses the locked mean/SD and the normal CDF — a stable runtime convention, distinct from an empirical percentile-rank lookup on the raw sample.)

## Data

Training data comes from the [Open-Source Psychometrics Project](https://openpsychometrics.org/) (OSPP) dataset. The canonical `canonical_v1` partition is a single plain random train/val/test split used for every headline claim; at this dataset's scale a random split is already balanced on every domain, so no target stratification is applied:

<!-- BEGIN GENERATED: data-counts -->
- **Valid respondents:** 603,322 (from 1,015,341 raw rows after cleaning)
- **Split (`canonical_v1`, random, seed 42):** 70/15/15 = 422,326 train / 90,498 val / 90,498 test
<!-- END GENERATED: data-counts -->

- **Augmentation:** The training fit rows are augmented via sparsity passes (see [Sparsity Augmentation](#sparsity-augmentation) for the row accounting). The early-stopping hold-out is split from the train rows *before* augmentation, so only the remaining fit respondents are tripled.
- **Split before augmentation:** The canonical train/val/test split is performed before augmentation to prevent leakage; the within-train early-stopping hold-out above is likewise carved out before any masking, distinct from that canonical split.

## Evaluation

All accuracy metrics are computed on the held-out `canonical_v1` test split (the `test` rows in [Data](#data)). "Overall *r*" is a **respondent-pooled** Pearson correlation: the five domains' predicted and true percentile vectors are stacked into a single length-5*N* vector before correlating (numerically ≈ the mean of the per-domain *r*, because every domain is on a common 0--100 percentile scale). Prediction intervals are the raw q05/q95 quantile spreads (no post-hoc width adjustment; every fitted `scale_factor` is 1.0). Empirical coverage and the paired recovery *r* at each operating point are:

<!-- BEGIN GENERATED: coverage-by-regime -->
- **Deployed 20-item form (SEM-stopped adaptive sim):** 89.5% empirical coverage (*r* = 0.93) — slightly under the nominal 90% by design.
- **Random balanced 20-item masking:** 89.8% empirical coverage (*r* = 0.911).
- **Full 50 items (self-recovery):** 92.6% empirical coverage (*r* = 0.9997).
<!-- END GENERATED: coverage-by-regime -->

Because the intervals are raw spreads (scale stays 1.0), the deployed-form intervals slightly under-cover the nominal 90% by design.

Headline 20-item numbers refer to the **fixed, pre-specified domain-balanced form** (top-4 items per domain — the deployed web form), not a post-hoc best-of-grid selection. The model's *general* partial-response accuracy under random balanced 20-item masking is lower (the random-balanced row above). The near-perfect full-50 self-recovery (the full-50 row above) reflects score recovery against a target computed from the same 50 items, not external validity.

Sparse-20 headline metrics are computed on a single fixed balanced mask (RNG seed 42); the reported bootstrap CI reflects respondent-sampling variance only and does not include mask-selection variance (stage-07 training averages over multiple masks).

**Internal-consistency reliability.** Stage 05 computes Cronbach's alpha (raw + standardized), the mean inter-item correlation, and McDonald's omega for each domain across three forms — the full 10-item domains, the deployed domain-balanced 20-item form, and the Mini-IPIP 4-item form — on the training split, written to `data/processed/canonical_v1/reliability.json` and surfaced in [NOTES.md](../notes/NOTES.md). Reliability bounds how high score-recovery *r* can plausibly go.

## Validity Boundary

What this project does and does not establish. The headline metrics are **score recovery**, not external-trait validity — be explicit about the boundary when citing them.

| Validity dimension | Status | Evidence / notes |
| --- | --- | --- |
| Score recovery | **Established** | Recovers the full 50-item domain score from a subset on a held-out OSPP split: *r* ≈ .93 at the deployed 20-item form, ≈ 1.0 at 50 items (a near-tautological ceiling — same items) |
| Internal-consistency reliability | **Reported** | Cronbach α / McDonald ω per domain and form (stage 05 → `reliability.json`) |
| Construct validity | Not established | No factor-structure or convergent/discriminant analysis against external measures |
| External / out-of-distribution validity | **Not established** | All evaluation is a held-out split of the *same* self-selected OSPP sample; no external dataset, so generalization beyond OSPP-like respondents is unverified |
| Measurement invariance / subgroup | **Not established** | No gender/age/region breakdown of recovery accuracy |
| Intended use | Not validated for clinical / high-stakes use | Not validated for clinical diagnosis or high-stakes selection |

## Limitations

- Norms are derived from self-selected online respondents (OSPP); they may not represent the general population
- Models are trained on English-language IPIP items only
- **No demographic-subgroup or measurement-invariance analysis** has been performed; accuracy may vary by gender, age, or region
- **No external / out-of-distribution validation:** every reported number is on a held-out split of the *same* OSPP dataset — these are score-recovery, not external-trait, metrics (see the Validity Boundary table above)
- The deployed 20-item domain-balanced Emotional Stability subscale (est1, est6, est7, est8) is composed entirely of reverse-keyed items. This maximizes within-domain discrimination but makes the EST short-form score vulnerable to acquiescence (yea-saying) response bias; the other four domains mix keyed directions, and the full 50-item assessment is unaffected. The selection rule (`_select_domain_balanced`) ranks purely by |own-domain *r*|, which is why this domain is single-keyed.
- Accuracy degrades with fewer items; 20 items is the recommended minimum for reliable scoring
- Not intended for clinical diagnosis
