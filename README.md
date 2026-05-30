# BFFM-XGB: Big Five From 20 Questions

Open-source pipeline for training XGBoost quantile regression models that predict Big Five personality scores from partial questionnaire responses. Trained on [~603k respondents](https://openpsychometrics.org/_rawdata) from the [IPIP-BFFM](https://ipip.ori.org/newBigFive5broadKey.htm) dataset with sparsity augmentation, the 15 exported ONNX models (5 domains x 3 quantiles) produce percentile scores with empirical 90% prediction intervals from as few as 20 items (validated to ~90% coverage at the 20-item operating point; no post-hoc width adjustment is applied).

**What's here:**
- **Models** — Pre-trained ONNX models, public domain, on [HuggingFace](https://huggingface.co/shawnprice/bffm-xgb)
- **Pipeline** — Reproducible end-to-end training: data download through ONNX export and publication figures
- **Inference packages** — Python and TypeScript libraries for running predictions
- **Live demo** — [big5.shawnprice.com](https://big5.shawnprice.com/)

### Comparison with the Mini-IPIP

The [Mini-IPIP](https://ipip.ori.org/MiniIPIPTable.htm) is the standard short personality test in psychology research (Donnellan et al., 2006). Both approaches use 20 items (4 per domain) to recover the full 50-item IPIP-BFFM scale scores. All *r* values are Pearson correlations with the full 50-item scale on a held-out test set (*N* = 90,499). "Overall *r*" is a respondent-pooled correlation across all five domains' stacked percentile vectors (numerically ≈ the mean of the per-domain *r*).

|                              | Mini-IPIP              | BFFM-XGB-20                 |
| ---------------------------- | ---------------------- | --------------------------- |
| **Items**                    | 20 (4 per domain)      | 20 (4 per domain)           |
| **Item selection**           | Expert-curated brevity | Top-4 by within-domain *r*  |
| **Scoring**                  | Simple scale averaging | XGBoost quantile regression |
| **Overall *r***              | .906                   | **.927**                    |
| **MAE (percentile pts)**     | 9.2                    | **8.2**                     |
| **90% prediction intervals** | —                      | ✓ (89.5% coverage)          |

The BFFM-XGB-20 figures above (*r* = .927, 89.5% coverage) are the **fixed, pre-specified domain-balanced 20-item form** — the top-4 items per domain — which is the deployed web form, not a post-hoc best-of-grid selection. Evaluated instead under *random* balanced 20-item masking, the model's general partial-response accuracy is *r* ≈ .909 (≈90% coverage).

**Per-domain accuracy at *K* = 20:**

| Domain                | Mini-IPIP *α* | Mini-IPIP *r* | BFFM-XGB-20 *r* |
| --------------------- | ------------- | ------------- | --------------- |
| Extraversion          | .77           | .939          | **.947**        |
| Agreeableness         | .70           | .911          | **.920**        |
| Conscientiousness     | .69           | .909          | **.919**        |
| Emotional Stability   | .68           | .929          | **.937**        |
| Intellect/Imagination | .65           | .842          | **.910**        |

With 15 items and XGBoost scoring (3 per domain), BFFM-XGB reaches *r* = .908, matching the 20-item simple-averaging Mini-IPIP (*r* = .906) — fewer items, though the comparison also differs in scoring method and item set.

The 20-item gain combines two levers: the **scoring method** (XGBoost vs simple averaging) and the **item set** (top-4-by-*r* vs the expert-curated Mini-IPIP items). Holding the item set fixed, XGBoost scoring alone adds roughly +0.01 *r* per domain over averaging; the remainder is item selection. The split is not uniform — for Emotional Stability the Mini-IPIP items scored with XGBoost slightly *beat* the top-4-by-*r* selection, so that domain's improvement is scoring, not selection. See the per-domain decomposition in [NOTES.md](notes/NOTES.md) and `artifacts/variants/reference/ml_vs_averaging_comparison.json`.

## Quick Start

| Language   | Directory                    | Install                                      | Docs                                          |
| ---------- | ---------------------------- | -------------------------------------------- | --------------------------------------------- |
| Python     | [`python/`](python/)         | `pip install onnxruntime numpy scipy pytest` | [Inference guide](docs/inference.md)          |
| TypeScript | [`typescript/`](typescript/) | `npm ci`                                     | [Inference guide](docs/inference.md)          |

Give it answers (1–5 scale, reverse-scored), get percentiles with 90% prediction intervals. See [docs/inference.md](docs/inference.md) for full code examples.

## Reproduce

```bash
make setup     # Python (uv sync), TypeScript, and web dependencies
make all       # Full pipeline: download through figures
make test      # All tests: lib, inference, and web
make lint      # Lint (ruff)
make typecheck # Static type check (basedpyright, standard mode)
```

See [docs/pipeline.md](docs/pipeline.md) for pipeline stages, training variants, hyperparameter tuning, and research evaluation.

## Directory Structure

```
bffm-xgb/
├── artifacts/          Pipeline artifacts and static reference data
├── configs/            YAML training configurations (reference + ablation variants)
├── data/               Downloaded and processed data (gitignored)
├── docs/               Documentation (inference, pipeline, research, infrastructure)
├── figures/            Generated publication figures (gitignored, regenerable)
├── infra/              Terraform configs for AWS CPU/GPU instances
├── lib/                Shared Python library
├── models/             Trained model checkpoints (gitignored)
├── notes/              Research notes (auto-generated data sections)
├── output/             Exported ONNX models by variant (reference/, ablation_*/)
├── pipeline/           Numbered pipeline scripts (01 through 13)
├── python/             Python inference package with tests
├── scripts/            Pipeline utilities (research summary, notes, provenance, backup, deployment)
├── templates/          Jinja2 templates for generated outputs
├── tests/              Unit tests for lib/ modules (pytest)
├── typescript/         TypeScript inference package with tests (vitest)
├── web/                React + Hono web assessment app (deployed to HuggingFace Spaces)
├── .env.example        Template for HF_TOKEN (required for `make upload-hf`)
├── .gitignore
├── LICENSE.md          MIT License
├── Makefile            Orchestrates the full pipeline
├── NOTICES.md          Third-party attributions (CC0, IPIP, OSPP)
├── pyproject.toml      Project metadata, dependencies, and tool config (ruff, basedpyright, pytest)
├── uv.lock             Locked, hashed dependency versions (uv)
├── .python-version     Pinned Python (uv)
└── README.md           This file
```

## Documentation

- [**Inference Guide**](docs/inference.md) — Python/TypeScript usage, code examples, reverse-scoring
- [**Pipeline Guide**](docs/pipeline.md) — Full reproduction, pipeline stages, training, research evaluation
- [**Research Notes**](docs/research.md) — Model architecture, sparsity augmentation, norms, data, limitations
- [**Infrastructure Guide**](docs/infrastructure.md) — AWS remote training (CPU/GPU, spot/on-demand)
- [**Web App**](web/README.md) — React + Hono assessment app
- [**Model Cards**](output/README.md) — ONNX model details and provenance
- [**NOTES.md**](notes/NOTES.md) — Auto-generated research notes with cross-variant evaluation data

## Limitations

- Norms are derived from self-selected online respondents (OSPP); they may not represent the general population
- Models are trained on English-language IPIP items only
- Accuracy degrades with fewer items; 20 items is the recommended minimum for reliable scoring
- Intended for educational use only

## License

[MIT](LICENSE.md). See [NOTICES.md](NOTICES.md) for third-party attributions.
