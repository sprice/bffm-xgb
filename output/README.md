---
license: cc0-1.0
language: en
tags:
  - personality
  - psychometrics
  - big-five
  - ipip
  - xgboost
  - onnx
  - quantile-regression
library_name: onnxruntime
pipeline_tag: tabular-regression
---

# IPIP-BFFM Sparse Quantile Models

XGBoost quantile regression models for the 50-item [IPIP Big-Five Factor Markers](https://ipip.ori.org/newBigFive5broadKey.htm) (BFFM) personality assessment, exported as ONNX for cross-platform inference.

## What These Models Do

Each model takes up to 50 item responses (Likert 1--5) and predicts Big Five domain scores (Extraversion, Agreeableness, Conscientiousness, Emotional Stability, Intellect). The exported calibration regimes are fit for full 50-item completion and the primary domain-balanced 20-item sparse regime.

**Key capability: sparse input.** The models produce accurate predictions even when most items are unanswered (NaN). This allows fixed short-form assessments (as few as 20 items) without retraining or switching models.

## How It Works

- **15 models in one graph** -- 5 domains x 3 quantiles (q05, q50, q95), merged into a single ONNX file
- **Structured sparsity augmentation** -- training responses are masked into *structured* partial-response patterns (not uniform random dropout): focused buckets spanning 10-50 retained items (a 10-20-item target-assessment range, a 21-35-item transition range, and a 36-50-item near-complete range, each keeping a minimum number of items per domain), explicit injection of the Mini-IPIP 4-per-domain 20-item pattern, and roughly 15% imbalanced patterns that allow 0-item domains so the model also sees skewed coverage. Within each bucket the retained items are filled by information-rank-weighted sampling (items with higher cross-domain information are more likely to be kept). The deployed operating point is the domain-balanced 4-per-domain 20-item form
- **Quantile regression** -- pinball loss at tau = 0.05, 0.50, 0.95 provides median predictions with empirical 90% prediction intervals whose coverage is validated for the full_50 and sparse_20_balanced runtime regimes (raw quantile spreads; no post-hoc width adjustment is applied). Empirical 90% prediction-interval coverage is approximately 89.5% for the deployed domain-balanced 20-item form (held-out baseline evaluation), approximately 89.8% for the sparse_20_balanced runtime regime (validation under random balanced 20-item masking), and approximately 92.6% for the full_50 regime (validation)
- **Norms-based percentiles** -- raw predictions are converted to population percentiles using z-score norms fit on the training split only (n = 422,326); validation and test rows are held out so the norms do not leak into the percentile targets

## Variants

| Variant     | Description             |
|-------------|-------------------------|
| `reference` | Primary published model |

The primary model is **`reference`**. Other variants are research ablations that isolate the contribution of each sparsity augmentation strategy.

Each variant directory contains:
- `model.onnx` -- merged ONNX model (5 domains x 3 quantiles)
- `config.json` -- runtime configuration, feature names, and norms
- `README.md` -- variant-specific model card with performance tables
- `provenance.json` -- full audit trail (git hash, data snapshot, training config)

## Source Code

Training pipeline, evaluation scripts, and inference packages (Python + TypeScript): [github.com/sprice/bffm-xgb](https://github.com/sprice/bffm-xgb)

## License

CC0 1.0 Universal -- Public Domain Dedication
