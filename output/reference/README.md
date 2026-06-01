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
  - variant:reference
library_name: onnxruntime
pipeline_tag: tabular-regression
---

# IPIP-BFFM Sparse Quantile Model

> This is the primary published model.

Sparse-input XGBoost quantile regression models for the 50-item IPIP Big-Five Factor Markers (BFFM) personality assessment, exported to ONNX format.

## Model Description

This package contains a **single merged ONNX model** with 15 outputs (5 personality domains × 3 quantiles) that predicts Big Five personality scores from item responses. The model performs **sparse-input scoring** — it produces accurate predictions even when many items are unanswered (NaN), enabling fixed short-form assessments such as the primary domain-balanced 20-item form. (Adaptive item *selection* was tested and underperforms the fixed balanced form; see the performance table.)

| Domain                | Code  | Items      |
|-----------------------|-------|------------|
| Extraversion          | `ext` | ext1-ext10 |
| Agreeableness         | `agr` | agr1-agr10 |
| Conscientiousness     | `csn` | csn1-csn10 |
| Emotional Stability   | `est` | est1-est10 |
| Intellect/Imagination | `opn` | opn1-opn10 |

Each domain has three quantile models:
- **q05** -- 5th percentile (lower bound of 90% prediction interval, PI)
- **q50** -- median (point estimate)
- **q95** -- 95th percentile (upper bound of 90% prediction interval, PI)

## Input Specification

- **Shape:** `[batch_size, 50]` -- one column per IPIP-BFFM item
- **Dtype:** `float32`
- **Values:** `1.0` to `5.0` (Likert scale), or `NaN` for unanswered items
- **Feature order:** ext1, ext2, ..., ext10, agr1, ..., agr10, csn1, ..., csn10, est1, ..., est10, opn1, ..., opn10

## Output Specification

- **Shape:** `[batch_size, 1]` per quantile output; the merged `scores` tensor is `[batch_size, 15]` (5 domains × 3 quantiles, in `config.outputs` order).
- **Scale:** Raw domain score on the **per-item-mean 1-5 scale** — the mean of the 10 item responses for the domain, **not** the 10-50 summed scale a 10-item sum would give. A domain mean of 3.0 is neutral; see the Norms table for population means/SDs.
- **Nominal range:** `[1, 5]`. Because these are gradient-boosted regressors (not bounded transforms), the raw `q05`/`q50`/`q95` predictions can fall **outside** `[1, 5]` — typically near the extremes of a domain's score range, and more so for low-information sparse inputs. They are **not** clamped: treat `[1, 5]` as the nominal/target range, not a hard guarantee, if you consume the raw `scores` tensor.
- **Percentile conversion:** Use the provided norms (z-score → CDF). The transform is monotonic and saturates near 0/100, so out-of-range raw values shift the reported percentile by well under one percentile point; the reference inference packages report percentiles, not raw scores.

## Quick Start (Python)

Requires `onnxruntime`, `numpy`, `scipy`.

```python
import json, numpy as np, onnxruntime as ort
from scipy.stats import norm

# Load model and config
sess = ort.InferenceSession("model.onnx")
with open("config.json") as f:
    config = json.load(f)

# Build input array (NaN = unanswered)
# Reverse-keyed items must already be transformed via `6 - raw_value`.
responses = {
    "ext3": 4.0, "ext5": 5.0, "agr1": 3.0, "agr7": 4.0,
    "csn1": 5.0, "csn4": 3.0, "est9": 4.0, "est10": 3.0,
    "opn5": 3.0, "opn10": 4.0,
}
features = config["input"]["feature_names"]
arr = np.full((1, len(features)), np.nan, dtype=np.float32)
for item, val in responses.items():
    arr[0, features.index(item)] = val

# Run inference (single call for all 15 outputs)
outputs = sess.run(config["outputs"], {"input": arr})
scores = dict(zip(config["outputs"], outputs))

# Convert to percentiles
for domain in config["domains"]:
    raw = float(scores[f"{domain}_q50"].flatten()[0])
    n = config["norms"][domain]
    pct = norm.cdf((raw - n["mean"]) / n["sd"]) * 100
    print(f"{domain}: {pct:.1f}th percentile (raw={raw:.3f})")
```

## Quick Start (TypeScript)

Requires `onnxruntime-node`.

```typescript
import { readFileSync } from "node:fs";
import * as ort from "onnxruntime-node";

// Load model and config
const config = JSON.parse(readFileSync("config.json", "utf-8"));
const session = await ort.InferenceSession.create("model.onnx");

// Build input array (NaN = unanswered)
// Reverse-keyed items must already be transformed via `6 - rawValue`.
const responses: Record<string, number> = {
  ext3: 4.0, ext5: 5.0, agr1: 3.0, agr7: 4.0,
  csn1: 5.0, csn4: 3.0, est9: 4.0, est10: 3.0,
  opn5: 3.0, opn10: 4.0,
};
const features: string[] = config.input.feature_names;
const arr = new Float32Array(features.length).fill(NaN);
for (const [item, val] of Object.entries(responses)) {
  arr[features.indexOf(item)] = val;
}

// Run inference (single call for all 15 outputs)
const output = await session.run({
  input: new ort.Tensor("float32", arr, [1, features.length]),
});

for (const domain of config.domains) {
  const raw = (output[`${domain}_q50`].data as Float32Array)[0];
  console.log(`${domain}: raw=${raw.toFixed(3)}`);
}

session.release();
```

## Training Details

- **Algorithm:** XGBoost quantile regression with pinball loss
- **Training data:** 422,326 respondents from the Open-Source Psychometrics Project (OSPP), augmented to 1,076,931 via sparsity augmentation
- **Sparsity augmentation:** Training samples are randomly masked to simulate adaptive (partial) responses, teaching the model to handle missing items
- **Hyperparameters:** n_estimators=9056, max_depth=5, learning_rate=0.0107
- **Cross-validation:** 3-fold cross-validation robustness analysis with evaluation split before augmentation

## Performance

Evaluated on held-out test respondents:

| Strategy          | Items (K) | Correlation (r) |
|-------------------|-----------|-----------------|
| Full assessment   | 50        | 0.9997          |
| Domain-balanced   | 20        | 0.928           |
| Mini-IPIP mapping | 20        | 0.907           |
| Greedy top-K      | 20        | 0.823           |

> The domain-balanced 20-item form is the pre-specified primary operating point and the deployed web form (not a post-hoc best-of-grid selection). The full-50 row recovers a target computed from the same 50 items, so *r* ≈ 1 reflects score recovery, not external validity.

90% prediction-interval coverage: 89.5% (deployed domain-balanced 20-item form), 92.6% (full 50-item). Under *random* balanced 20-item masking the model's general coverage is 89.8%.

ML advantage over simple averaging: +0.006 r (domain-balanced K=20).

## Norms

Population norms for raw-score -> percentile conversion (from OSPP dataset):

| Domain                | Mean  | SD    |
|-----------------------|-------|-------|
| Extraversion          | 2.915 | 0.911 |
| Agreeableness         | 3.759 | 0.737 |
| Conscientiousness     | 3.343 | 0.739 |
| Emotional Stability   | 2.919 | 0.861 |
| Intellect/Imagination | 3.939 | 0.618 |

## Limitations

- Norms are derived from self-selected online respondents (OSPP); they may not represent the general population
- Models are trained on English-language IPIP items only
- No demographic-subgroup or measurement-invariance analysis has been performed; accuracy may vary by gender, age, or region
- No external / out-of-distribution validation: every reported number is on a held-out split of the *same* OSPP dataset, so these are score-recovery (recovering the full-scale score from a subset of its own items), not external-trait, metrics
- Standalone Python/TypeScript inference expects reverse-keyed items to be preprocessed before scoring; the web app applies that transform server-side
- Exported calibration regimes are `full_50` and `sparse_20_balanced`; arbitrary sub-50 response patterns use the sparse regime as a fallback rather than a separately fit calibration curve
- The deployed 20-item domain-balanced Emotional Stability subscale (est1, est6, est7, est8) is composed entirely of reverse-keyed items, so the short-form EST score is vulnerable to acquiescence (yea-saying) response bias; the other four domains mix keyed directions, and the full 50-item assessment is unaffected
- Accuracy degrades with fewer items; 20 items is the recommended minimum for reliable scoring
- Not intended for clinical diagnosis or high-stakes selection decisions

## Item Source

The IPIP-BFFM items are from the [International Personality Item Pool](https://ipip.ori.org/) and are in the **public domain**.

## License

CC0 1.0 Universal -- Public Domain Dedication
