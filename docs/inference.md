# Inference Guide

## Overview

The inference packages provide standalone Big Five personality prediction from IPIP-BFFM item responses. Each package loads pre-trained ONNX models exported by the pipeline and returns percentile scores with calibrated 90% prediction intervals.

Models and configuration are in [`output/reference/`](../output/reference/) (the published reference variant). Each variant directory contains:
- `model.onnx` — XGBoost quantile regression models (5 domains × 3 quantiles)
- `config.json` — feature names, norms, calibration factors, and all metadata needed for inference
- `provenance.json` — build provenance (git hash, data snapshot ID, etc.)

## Python

**Directory:** [`python/`](../python/)

```bash
pip install onnxruntime numpy scipy pytest
```

```python
from inference import IPIPBFFMPredictor

predictor = IPIPBFFMPredictor()

# Values below assume reverse-keyed items have already been transformed via
# `6 - raw_value` before inference.
result = predictor.predict({
    "ext3": 4.0, "ext5": 5.0, "agr1": 3.0, "agr7": 4.0,
    "csn1": 5.0, "csn4": 3.0, "est9": 4.0, "est10": 3.0,
    "opn5": 3.0, "opn10": 4.0,
})

for domain in ["ext", "agr", "csn", "est", "opn"]:
    r = result[domain]
    print(f"{domain}: {r['percentile']['q50']}th pct "
          f"(90% CI: {r['percentile']['q05']}--{r['percentile']['q95']})")
```

Run tests: `python -m pytest -v`

## TypeScript

**Directory:** [`typescript/`](../typescript/)

```bash
npm ci
```

```typescript
import { IPIPBFFMPredictor } from "./inference.js";

const predictor = await IPIPBFFMPredictor.create();

// Values below assume reverse-keyed items have already been transformed via
// `6 - rawValue` before inference.
const result = await predictor.predict({
  ext3: 4.0, ext5: 5.0, agr1: 3.0, agr7: 4.0,
  csn1: 5.0, csn4: 3.0, est9: 4.0, est10: 3.0,
  opn5: 3.0, opn10: 4.0,
});

for (const domain of ["ext", "agr", "csn", "est", "opn"] as const) {
  const r = result[domain];
  console.log(`${domain}: ${r.percentile.q50}th pct `
    + `(90% CI: ${r.percentile.q05}--${r.percentile.q95})`);
}

predictor.dispose();
```

Run tests: `npm test`

## Reverse-Scoring

The Python and TypeScript inference packages expect inputs to already match training-time preprocessing. You must reverse-score the 24 negatively keyed IPIP-BFFM items yourself before calling `predict()`. The [web app](../web/) does this automatically on the server; the standalone packages do not.

The 24 reverse-keyed items are defined in `lib/constants.py` (`REVERSE_KEYED_ITEMS`). To reverse-score: `6 - raw_value`.

## Calibration Note

Exported inference dispatches between two calibrated regimes by answered-item count:

| Regime | Items Answered | Description |
|--------|---------------|-------------|
| `full_50` | 50 | All items answered |
| `sparse_20_balanced` | ≤49 | Primary 20-item domain-balanced operating point |

Predictions remain available for arbitrary partial-response patterns, but the strongest calibration claim is for the primary 20-item domain-balanced operating point rather than every possible sub-50 response pattern.

## Raw ONNX Usage

For direct ONNX session usage without the inference wrappers, see [`output/reference/README.md`](../output/reference/README.md) for model card details including input/output tensor specifications.
