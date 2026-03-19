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

**Key capability: sparse input.** The models produce accurate predictions even when most items are unanswered (NaN). This allows adaptive and short-form assessments (as few as 20 items) without retraining or switching models.

## How It Works

- **15 models in one graph** -- 5 domains x 3 quantiles (q05, q50, q95), merged into a single ONNX file
- **Sparsity augmentation** -- during training, complete responses are randomly masked to simulate missing items, teaching the model to handle arbitrary missing-item patterns
- **Quantile regression** -- pinball loss at tau = 0.05, 0.50, 0.95 provides median predictions with uncertainty bounds that are explicitly calibrated for full_50 and sparse_20_balanced runtime regimes
- **Norms-based percentiles** -- raw predictions are converted to population percentiles using z-score norms derived from ~603k respondents

## Variants

| Variant               | Description               |
|-----------------------|---------------------------|
| `ablation_focused`    | Research ablation variant |
| `ablation_none`       | Research ablation variant |
| `ablation_stratified` | Research ablation variant |
| `reference`           | Primary published model   |

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
