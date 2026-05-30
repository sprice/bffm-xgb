# Inference Guide

## Overview

The inference packages provide standalone Big Five personality prediction from IPIP-BFFM item responses. Each package loads pre-trained ONNX models exported by the pipeline and returns percentile scores with empirical 90% prediction intervals (validated to ~90% coverage at the 20-item operating point; no post-hoc width adjustment is applied).

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
          f"(90% PI: {r['percentile']['q05']}--{r['percentile']['q95']})")
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
    + `(90% PI: ${r.percentile.q05}--${r.percentile.q95})`);
}

predictor.dispose();
```

Run tests: `npm test`

## Reverse-Scoring

The Python and TypeScript inference packages expect inputs to already match training-time preprocessing. You must reverse-score the 24 negatively keyed IPIP-BFFM items yourself before calling `predict()`. The [web app](../web/) does this automatically on the server; the standalone packages do not.

The 24 reverse-keyed items are defined in `lib/constants.py` (`REVERSE_KEYED_ITEMS`). To reverse-score: `6 - raw_value`.

## Calibration Note

Exported inference dispatches between two coverage-validated regimes by answered-item count:

| Regime | Items Answered | Description |
|--------|---------------|-------------|
| `full_50` | 50 | All items answered |
| `sparse_20_balanced` | ≤49 | Primary 20-item domain-balanced operating point |

Predictions remain available for arbitrary partial-response patterns, but the intervals are raw quantile spreads (no post-hoc width adjustment), and their coverage is validated only at the primary 20-item domain-balanced operating point — not at every possible sub-50 response pattern.

## Determinism

All three runtimes (Python, TypeScript, web) load the ONNX model in a
**single-threaded, sequential** session (`intra_op_num_threads = inter_op_num_threads = 1`,
sequential execution mode, CPU execution provider). Multi-threaded ONNX Runtime
sums partial results in a host-dependent order, which can perturb raw scores
enough to flip a percentile that rounds to one decimal place on machines with
different core counts. Pinning the session to one thread makes the reported
percentile a deterministic function of the input on any host, at no real cost
(inference is a single row at a time).

The percentile transform uses the **exact** standard-normal CDF in every
runtime: Python via `scipy.stats.norm.cdf`, TypeScript and web via an
`erf`-based implementation (`erf.ts`, a port of the fdlibm error function that
matches `scipy.special.erf` to within ~1 ULP). This replaced an earlier
Abramowitz–Stegun polynomial approximation so the deployed percentiles use the
same CDF that produced the reported metrics. Cross-runtime numerical parity is
locked by a committed golden-vector test.

## Raw ONNX Usage

For direct ONNX session usage without the inference wrappers, see [`output/reference/README.md`](../output/reference/README.md) for model card details including input/output tensor specifications.
