# Research Notes: Multivariate Scoring for Big Five Personality Assessment

> Companion document for the IPIP-BFFM sparse quantile model research.
> Contains the complete research narrative, all pipeline-derived metrics,
> and auto-updatable data sections.

## How This Document Is Maintained

This document has two types of content:

1. **Narrative sections** (static content): the research journey, architectural
   decisions, and interpretive commentary. These are not auto-generated and should
   be updated manually when the research evolves.

2. **Data sections** (auto-generated): tables and metrics derived from pipeline
   artifacts. These live between `<!-- BEGIN:section_name -->` and
   `<!-- END:section_name -->` markers and can be refreshed by running:

   ```bash
   make notes
   ```

   The notes generator reads from `artifacts/research_summary.json` (built from
   per-variant model/evaluation artifacts) and updates only the marked sections,
   preserving all surrounding commentary.

## Canonical Reporting Specification (Source of Truth)

To prevent drift across manuscripts, this project uses one canonical reporting
spec for metrics and aggregation. Unless explicitly labeled as a robustness
check, reported values should follow this definition set.

### Targets (what is being predicted)

- **Primary target:** full 50-item IPIP-BFFM **scale-score percentile** (0--100),
  computed via the production transform:
  raw domain score $\rightarrow z=(x-\mu)/\sigma \rightarrow \Phi(z)\times 100$.
- **Secondary target:** raw domain score on the 1--5 scale (used internally for
  quantile modeling and diagnostics).

### Primary operating point

- **Primary operating point:** *K* = 20 items, **domain-balanced** (4 items per
  domain), XGBoost cross-domain scoring, 90% prediction intervals.
- Other *K* values (5--50) are reported for scaling behavior.

### Canonical aggregation across domains

- **Overall Pearson *r* (canonical):** concatenate predicted and true percentiles
  across all five domains (5×*N* pairs) and compute one Pearson correlation.
- **Per-domain *r* (required companion):** whenever overall *r* is reported at
  *K* = 20, include per-domain Pearson *r*.
- **Robustness (optional):** mean Fisher-$z$ across domains; if used, label it as
  robustness and do not mix it with canonical overall *r*.

### Canonical error and uncertainty metrics

- **MAE (canonical):** mean absolute error in percentile points over the same
  concatenated 5×*N* pairs as overall *r* (unless explicitly per-domain).
- **Within-5 / Within-10:** proportion within 5 or 10 percentile points over
  concatenated 5×*N* pairs.
- **90% PI coverage (canonical):** proportion of true values within q05--q95
  bounds (target 90%), reported overall and per-domain where relevant.
- **Mean PI width (diagnostic):** calibration/behavior diagnostic, not a headline
  metric.

### Bootstrap and confidence intervals

- **Bootstrap unit:** respondent-level resampling.
- **Default resamples:** 1,000.
- **Paired comparisons:** bootstrap paired differences (Δ*r*, ΔMAE).

### Precision Convention (NOTES → PAPER)

NOTES.md stores values at full pipeline precision (4 decimal places for *r*, exact
CI bounds). When transferring to papers, use:

- **Abstract/Introduction prose:** 3 decimals; CIs to 3 decimals.
- **Results tables:** 4 decimals for headline *K* = 20 comparisons where needed;
  otherwise 3 decimals.
- **Results prose:** 4 decimals only for tiny decomposition deltas; otherwise 3.
- **Discussion/Conclusion:** 3 decimals unless introducing a new statistical claim.
- **Within a paragraph/table:** keep decimal precision consistent.

---

## Research Journey

### Motivation and Starting Point

This project began after reading Glöckner, Michels, and Giersch (2020), who
explored ML-based scoring for personality assessment on a modest sample.

> Glöckner, A., Michels, M., & Giersch, D. (2020). *Predicting personality
> test scores with machine learning methodology: Investigation of a new
> approach to psychological assessment* [Unpublished preprint]. PsyArXiv.
> https://doi.org/10.31234/osf.io/ysd3f

The practical question was straightforward: could we train an adaptive model
that delivers a Big Five assessment in as few questions as possible, choosing
which item to ask next based on responses so far and stopping early once
precision is sufficient?

We scaled the test on a large sample from the Open-Source Psychometrics
Project (see the dataset table below for exact counts; orders of magnitude
larger than Glöckner et al.) and trained XGBoost quantile-regression models that
predict full-scale percentiles from partial item patterns.

### Finding 1: Adaptive Item Selection Does Not Work

The central finding was unexpected: unconstrained adaptive item selection fails
in the short-form range. The idea of picking the next most informative item
across all five domains sounds optimal, but it produces severe domain starvation.

At *K* = 20, adaptive top-*K* by cross-domain utility performed far worse
than simple domain-balanced selection (4 items per domain, chosen by
within-domain correlation). The gap is large and statistically clear; even
random item selection outperformed greedy. The headline results and baseline
curves tables below have the exact values and confidence intervals.

The mechanism is domain starvation. Greedy utility ranking concentrates
items heavily on Extraversion and Emotional Stability while starving
Conscientiousness and Intellect/Openness entirely (the domain starvation
table below shows the exact allocation). Extraversion items dominate because
they correlate moderately with other domains, which inflates their composite
utility scores; but those cross-domain correlations are too weak to actually
predict the other domains. The best-served domain achieves near-perfect
recovery while the worst-served domain falls to chance-level prediction.

This is not a cold-start problem. A constrained greedy start (one item per
domain first, then greedy fill) still fell far below domain-balanced and
random. Imbalanced sparsity retraining improved adaptive top-*K* only
marginally. The core issue is persistent re-concentration, not initial
allocation.

### Finding 2: The Adaptive Dream Collapses to the Static Strategy

We built a full adaptive assessment simulation with SEM-based stopping
(threshold 0.45, minimum 4 items per domain, held-out respondents from the
test split). The result was definitive: every respondent converged to exactly
20 items in a 4-4-4-4-4 allocation. The adaptive strategy collapsed to the
static domain-balanced strategy.

Simulation performance closely matches the static baseline on all metrics;
the simulation results and headline results tables below show the comparison.
Correlation-ranked adaptive selection is, for all practical purposes,
equivalent to the optimal static form.

The best practical configuration turned out to be the simplest: pick the top
4 items per domain by within-domain correlation, score them with XGBoost, and
skip the adaptive machinery entirely.

### Finding 3: Scoring Method Is a Real Lever

While the adaptive *selection* dream died, the ML *scoring* side of the
original motivation held up. Holding items fixed, XGBoost cross-domain scoring
beats simple averaging at every tested budget. The ML vs averaging table below
has the exact deltas for domain-balanced and Mini-IPIP item sets across
*K* = 10 through *K* = 25.

The scoring gain is largest when item budgets are smallest, where cross-domain
information sharing matters most. Applying XGBoost to existing Mini-IPIP
responses improves recovery with no item changes: a drop-in upgrade.

### Training and Pipeline Lessons

Sparse-input training is required. Without sparsity augmentation, performance
at the same operating point drops sharply; the cross-variant overview table
shows the no-sparsity ablation falling well below the reference on both
sparse-20 validation and the *K* = 20 baseline.

Two historical pipeline fixes were also important:

- **CV leakage fix:** split before augmentation for early stopping.
- **Quantile crossing fix:** enforce quantile ordering after transform.

---

## Data Reference

All tables below are derived from pipeline artifacts and can be refreshed by
running `python scripts/generate_notes_data.py`.

### Headline Results (K=20 Operating Point)

The key numbers for the abstract and introduction.

<!-- BEGIN:headline_k20 -->
| Metric                                    | Value                   |
|-------------------------------------------|-------------------------|
| Domain-balanced r                         | 0.9262 [0.9258, 0.9266] |
| Domain-balanced MAE                       | 8.21 pp                 |
| Domain-balanced 90% coverage              | 90.1%                   |
| Mini-IPIP r                               | 0.9056                  |
| Adaptive top-K r (greedy)                 | 0.8196 [0.8187, 0.8205] |
| ML vs averaging delta r (domain-balanced) | +0.0159                 |
| ML vs averaging delta r (Mini-IPIP)       | +0.0112                 |
<!-- END:headline_k20 -->

### Cross-Variant Overview (Reference + Ablations)

Auto-generated from `artifacts/research_summary.json`, which aggregates
`models/*/training_report.json` and per-variant evaluation artifacts.

<!-- BEGIN:ablation_overview -->
| Variant                    | Data Regime | Train Val r (full-50) | Validate r (full-50) | Validate r (sparse-20) | Baselines K20 r (domain-balanced) | Simulation r | Complete |
|----------------------------|-------------|-----------------------|----------------------|------------------------|-----------------------------------|--------------|----------|
| Reference                  | ext_est     | 0.9994                | 0.9994               | 0.9086                 | 0.9262                            | 0.9249       | yes      |
| Ablation: No Sparsity      | ext_est     | 0.9999                | 0.9999               | 0.7507                 | 0.7899                            | 0.7855       | yes      |
| Ablation: Focused Only     | ext_est     | 0.9994                | 0.9994               | 0.9086                 | 0.9257                            | 0.9247       | yes      |
| Ablation: Stratified Split | ext_est_opn | 0.9994                | 0.9994               | 0.9087                 | 0.9256                            | 0.9247       | yes      |
<!-- END:ablation_overview -->

### Cross-Variant Provenance Locks

Short-hash provenance view used to confirm all reported numbers are tied to
their exact split and hyperparameter locks.

<!-- BEGIN:ablation_provenance -->
| Variant                    | Split Signature | Train SHA256 | Hyperparams SHA256 | Git Hash     | Errors |
|----------------------------|-----------------|--------------|--------------------|--------------|--------|
| Reference                  | 274c9fa6bc18    | fe50332715d9 | cfd57f4de038       | 3e63b7f0a66b | none   |
| Ablation: No Sparsity      | 274c9fa6bc18    | fe50332715d9 | cfd57f4de038       | 3e63b7f0a66b | none   |
| Ablation: Focused Only     | 274c9fa6bc18    | fe50332715d9 | cfd57f4de038       | 3e63b7f0a66b | none   |
| Ablation: Stratified Split | 5e2d53aba0c4    | 50e0ac18bca0 | cfd57f4de038       | 3e63b7f0a66b | none   |
<!-- END:ablation_provenance -->

### Cross-Variant Detailed Validation (All Runs)

Full-50 and sparse-20 validation tables for each trained variant.

<!-- BEGIN:ablation_validation_details -->
#### Reference


**Full-50 validation:**


| Domain              | r          | MAE      | RMSE     | Within-5  | 90% Coverage | Raw Crossing Rate |
|---------------------|------------|----------|----------|-----------|--------------|-------------------|
| Extraversion        | 0.9997     | 0.48     | 0.71     | 100.0%    | 94.0%        | 32.5%             |
| Agreeableness       | 0.9994     | 0.72     | 1.01     | 99.9%     | 93.4%        | 24.3%             |
| Conscientiousness   | 0.9993     | 0.80     | 1.10     | 99.9%     | 91.0%        | 21.9%             |
| Emotional Stability | 0.9995     | 0.64     | 0.90     | 100.0%    | 93.8%        | 25.8%             |
| Intellect/Openness  | 0.9990     | 0.98     | 1.34     | 99.7%     | 94.6%        | 20.5%             |
| **Overall**         | **0.9994** | **0.72** | **1.03** | **99.9%** | ---          | ---               |


**Sparse-20 validation:**


| Domain              | r          | MAE      | RMSE      | Within-5  | 90% Coverage |
|---------------------|------------|----------|-----------|-----------|--------------|
| Extraversion        | 0.9326     | 7.86     | 10.87     | 46.6%     | 90.8%        |
| Agreeableness       | 0.9058     | 9.02     | 12.33     | 41.0%     | 90.2%        |
| Conscientiousness   | 0.8921     | 9.93     | 13.48     | 38.0%     | 90.4%        |
| Emotional Stability | 0.9225     | 8.43     | 11.58     | 43.5%     | 90.7%        |
| Intellect/Openness  | 0.8884     | 10.02    | 13.68     | 38.0%     | 89.9%        |
| **Overall**         | **0.9086** | **9.05** | **12.44** | **41.4%** | **90.4%**    |

#### Ablation: No Sparsity


**Full-50 validation:**


| Domain              | r          | MAE      | RMSE     | Within-5   | 90% Coverage | Raw Crossing Rate |
|---------------------|------------|----------|----------|------------|--------------|-------------------|
| Extraversion        | 0.9999     | 0.22     | 0.32     | 100.0%     | 95.1%        | 34.2%             |
| Agreeableness       | 0.9999     | 0.24     | 0.36     | 100.0%     | 89.7%        | 32.2%             |
| Conscientiousness   | 0.9999     | 0.28     | 0.40     | 100.0%     | 95.0%        | 37.2%             |
| Emotional Stability | 0.9999     | 0.23     | 0.33     | 100.0%     | 88.7%        | 34.3%             |
| Intellect/Openness  | 0.9999     | 0.24     | 0.37     | 100.0%     | 89.9%        | 31.7%             |
| **Overall**         | **0.9999** | **0.24** | **0.36** | **100.0%** | ---          | ---               |


**Sparse-20 validation:**


| Domain              | r          | MAE       | RMSE      | Within-5  | 90% Coverage |
|---------------------|------------|-----------|-----------|-----------|--------------|
| Extraversion        | 0.8696     | 39.16     | 45.68     | 9.4%      | 1.7%         |
| Agreeableness       | 0.8343     | 30.38     | 36.23     | 12.4%     | 4.0%         |
| Conscientiousness   | 0.7982     | 38.33     | 44.74     | 9.0%      | 1.6%         |
| Emotional Stability | 0.8499     | 40.83     | 47.40     | 8.2%      | 1.6%         |
| Intellect/Openness  | 0.8078     | 31.81     | 38.04     | 12.9%     | 4.8%         |
| **Overall**         | **0.7507** | **36.10** | **42.65** | **10.4%** | **2.7%**     |

#### Ablation: Focused Only


**Full-50 validation:**


| Domain              | r          | MAE      | RMSE     | Within-5  | 90% Coverage | Raw Crossing Rate |
|---------------------|------------|----------|----------|-----------|--------------|-------------------|
| Extraversion        | 0.9997     | 0.50     | 0.73     | 100.0%    | 93.7%        | 30.1%             |
| Agreeableness       | 0.9994     | 0.74     | 1.05     | 99.9%     | 93.9%        | 20.7%             |
| Conscientiousness   | 0.9993     | 0.82     | 1.13     | 99.9%     | 91.0%        | 19.8%             |
| Emotional Stability | 0.9995     | 0.66     | 0.93     | 100.0%    | 94.0%        | 26.0%             |
| Intellect/Openness  | 0.9990     | 0.99     | 1.36     | 99.7%     | 94.7%        | 20.2%             |
| **Overall**         | **0.9994** | **0.74** | **1.06** | **99.9%** | ---          | ---               |


**Sparse-20 validation:**


| Domain              | r          | MAE      | RMSE      | Within-5  | 90% Coverage |
|---------------------|------------|----------|-----------|-----------|--------------|
| Extraversion        | 0.9325     | 7.86     | 10.89     | 46.6%     | 90.8%        |
| Agreeableness       | 0.9058     | 9.02     | 12.36     | 41.1%     | 90.8%        |
| Conscientiousness   | 0.8921     | 9.93     | 13.51     | 38.1%     | 90.5%        |
| Emotional Stability | 0.9225     | 8.43     | 11.59     | 43.6%     | 90.9%        |
| Intellect/Openness  | 0.8885     | 10.04    | 13.73     | 38.2%     | 90.3%        |
| **Overall**         | **0.9086** | **9.06** | **12.46** | **41.5%** | **90.7%**    |

#### Ablation: Stratified Split


**Full-50 validation:**


| Domain              | r          | MAE      | RMSE     | Within-5  | 90% Coverage | Raw Crossing Rate |
|---------------------|------------|----------|----------|-----------|--------------|-------------------|
| Extraversion        | 0.9997     | 0.49     | 0.72     | 100.0%    | 94.0%        | 29.4%             |
| Agreeableness       | 0.9994     | 0.74     | 1.04     | 99.9%     | 93.8%        | 22.0%             |
| Conscientiousness   | 0.9993     | 0.83     | 1.14     | 99.9%     | 94.8%        | 19.4%             |
| Emotional Stability | 0.9995     | 0.66     | 0.93     | 100.0%    | 93.7%        | 24.8%             |
| Intellect/Openness  | 0.9990     | 0.99     | 1.36     | 99.7%     | 94.7%        | 19.0%             |
| **Overall**         | **0.9994** | **0.74** | **1.06** | **99.9%** | ---          | ---               |


**Sparse-20 validation:**


| Domain              | r          | MAE      | RMSE      | Within-5  | 90% Coverage |
|---------------------|------------|----------|-----------|-----------|--------------|
| Extraversion        | 0.9331     | 7.83     | 10.83     | 46.4%     | 91.0%        |
| Agreeableness       | 0.9060     | 9.03     | 12.34     | 41.0%     | 90.8%        |
| Conscientiousness   | 0.8926     | 9.92     | 13.48     | 38.0%     | 90.6%        |
| Emotional Stability | 0.9222     | 8.46     | 11.62     | 43.5%     | 90.8%        |
| Intellect/Openness  | 0.8879     | 10.04    | 13.74     | 38.3%     | 90.2%        |
| **Overall**         | **0.9087** | **9.06** | **12.45** | **41.4%** | **90.7%**    |
<!-- END:ablation_validation_details -->

### Cross-Variant Baseline Curves (All Runs)

Item-selection baseline curves (K=5..50) for each trained variant.

<!-- BEGIN:ablation_baselines_details -->
#### Reference


| K  | Domain-Balanced          | Mini-IPIP | First-N              | Random               | Adaptive Top-K       | Greedy-Balanced      | Worst-K              |
|----|--------------------------|-----------|----------------------|----------------------|----------------------|----------------------|----------------------|
| 5  | 0.742 [0.741, 0.743]     | ---       | 0.694 [0.692, 0.695] | 0.598 [0.598, 0.599] | 0.570 [0.568, 0.571] | 0.655 [0.653, 0.657] | 0.473 [0.471, 0.474] |
| 10 | 0.852 [0.851, 0.853]     | ---       | 0.818 [0.817, 0.819] | 0.761 [0.761, 0.762] | 0.702 [0.700, 0.703] | 0.753 [0.752, 0.754] | 0.667 [0.665, 0.668] |
| 15 | 0.907 [0.907, 0.908]     | ---       | 0.871 [0.870, 0.872] | 0.843 [0.843, 0.843] | 0.793 [0.792, 0.794] | 0.818 [0.817, 0.819] | 0.754 [0.753, 0.755] |
| 20 | **0.926 [0.926, 0.927]** | **0.906** | 0.908 [0.908, 0.909] | 0.893 [0.892, 0.893] | 0.820 [0.819, 0.821] | 0.838 [0.837, 0.839] | 0.819 [0.818, 0.820] |
| 25 | 0.944 [0.944, 0.944]     | ---       | 0.937 [0.937, 0.938] | 0.932 [0.931, 0.932] | 0.898 [0.897, 0.898] | 0.898 [0.897, 0.898] | 0.883 [0.882, 0.883] |
| 30 | 0.960 [0.959, 0.960]     | ---       | 0.955 [0.955, 0.956] | 0.951 [0.950, 0.951] | 0.927 [0.927, 0.928] | 0.927 [0.927, 0.928] | 0.926 [0.926, 0.926] |
| 40 | 0.983 [0.982, 0.983]     | ---       | 0.984 [0.984, 0.984] | 0.981 [0.981, 0.981] | 0.975 [0.974, 0.975] | 0.975 [0.974, 0.975] | 0.977 [0.977, 0.977] |
| 50 | 0.999 [0.999, 0.999]     | ---       | 0.999 [0.999, 0.999] | 0.999 [0.999, 0.999] | 0.999 [0.999, 0.999] | 0.999 [0.999, 0.999] | 0.999 [0.999, 0.999] |

#### Ablation: No Sparsity


| K  | Domain-Balanced          | Mini-IPIP | First-N              | Random               | Adaptive Top-K       | Greedy-Balanced      | Worst-K              |
|----|--------------------------|-----------|----------------------|----------------------|----------------------|----------------------|----------------------|
| 5  | 0.358 [0.356, 0.360]     | ---       | 0.301 [0.299, 0.303] | 0.297 [0.296, 0.298] | 0.308 [0.306, 0.309] | 0.255 [0.253, 0.257] | 0.213 [0.210, 0.215] |
| 10 | 0.561 [0.560, 0.563]     | ---       | 0.534 [0.533, 0.536] | 0.433 [0.432, 0.434] | 0.376 [0.374, 0.377] | 0.426 [0.425, 0.427] | 0.351 [0.350, 0.353] |
| 15 | 0.723 [0.723, 0.724]     | ---       | 0.664 [0.662, 0.665] | 0.528 [0.527, 0.529] | 0.548 [0.547, 0.549] | 0.533 [0.532, 0.534] | 0.404 [0.402, 0.405] |
| 20 | **0.790 [0.789, 0.791]** | **0.906** | 0.767 [0.766, 0.768] | 0.650 [0.649, 0.651] | 0.506 [0.505, 0.507] | 0.506 [0.505, 0.507] | 0.458 [0.457, 0.460] |
| 25 | 0.858 [0.858, 0.859]     | ---       | 0.843 [0.843, 0.844] | 0.706 [0.705, 0.706] | 0.645 [0.644, 0.646] | 0.645 [0.644, 0.646] | 0.508 [0.506, 0.509] |
| 30 | 0.897 [0.897, 0.897]     | ---       | 0.884 [0.884, 0.884] | 0.797 [0.796, 0.797] | 0.745 [0.744, 0.746] | 0.745 [0.744, 0.746] | 0.591 [0.590, 0.592] |
| 40 | 0.967 [0.967, 0.967]     | ---       | 0.965 [0.965, 0.965] | 0.914 [0.914, 0.914] | 0.899 [0.898, 0.899] | 0.899 [0.898, 0.899] | 0.767 [0.766, 0.768] |
| 50 | 1.000 [1.000, 1.000]     | ---       | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |

#### Ablation: Focused Only


| K  | Domain-Balanced          | Mini-IPIP | First-N              | Random               | Adaptive Top-K       | Greedy-Balanced      | Worst-K              |
|----|--------------------------|-----------|----------------------|----------------------|----------------------|----------------------|----------------------|
| 5  | 0.741 [0.740, 0.742]     | ---       | 0.693 [0.691, 0.694] | 0.594 [0.593, 0.595] | 0.566 [0.564, 0.567] | 0.654 [0.653, 0.656] | 0.468 [0.466, 0.470] |
| 10 | 0.852 [0.851, 0.852]     | ---       | 0.818 [0.817, 0.819] | 0.759 [0.759, 0.760] | 0.696 [0.695, 0.697] | 0.751 [0.750, 0.752] | 0.660 [0.659, 0.662] |
| 15 | 0.907 [0.907, 0.908]     | ---       | 0.871 [0.870, 0.872] | 0.842 [0.842, 0.842] | 0.788 [0.787, 0.789] | 0.815 [0.814, 0.816] | 0.748 [0.747, 0.749] |
| 20 | **0.926 [0.925, 0.926]** | **0.906** | 0.909 [0.908, 0.909] | 0.892 [0.892, 0.893] | 0.815 [0.814, 0.816] | 0.834 [0.833, 0.835] | 0.812 [0.811, 0.813] |
| 25 | 0.944 [0.944, 0.944]     | ---       | 0.937 [0.937, 0.938] | 0.931 [0.931, 0.932] | 0.896 [0.895, 0.896] | 0.896 [0.895, 0.896] | 0.879 [0.878, 0.880] |
| 30 | 0.959 [0.959, 0.960]     | ---       | 0.955 [0.955, 0.956] | 0.951 [0.950, 0.951] | 0.927 [0.926, 0.927] | 0.927 [0.926, 0.927] | 0.924 [0.924, 0.925] |
| 40 | 0.983 [0.983, 0.983]     | ---       | 0.984 [0.984, 0.984] | 0.981 [0.981, 0.981] | 0.975 [0.974, 0.975] | 0.975 [0.974, 0.975] | 0.977 [0.977, 0.977] |
| 50 | 0.999 [0.999, 0.999]     | ---       | 0.999 [0.999, 0.999] | 0.999 [0.999, 0.999] | 0.999 [0.999, 0.999] | 0.999 [0.999, 0.999] | 0.999 [0.999, 0.999] |

#### Ablation: Stratified Split


| K  | Domain-Balanced          | Mini-IPIP | First-N              | Random               | Adaptive Top-K       | Greedy-Balanced      | Worst-K              |
|----|--------------------------|-----------|----------------------|----------------------|----------------------|----------------------|----------------------|
| 5  | 0.741 [0.739, 0.742]     | ---       | 0.693 [0.692, 0.695] | 0.595 [0.594, 0.595] | 0.566 [0.565, 0.568] | 0.656 [0.654, 0.658] | 0.469 [0.467, 0.471] |
| 10 | 0.851 [0.850, 0.852]     | ---       | 0.818 [0.817, 0.819] | 0.759 [0.759, 0.760] | 0.697 [0.696, 0.698] | 0.751 [0.750, 0.752] | 0.660 [0.659, 0.662] |
| 15 | 0.907 [0.906, 0.908]     | ---       | 0.871 [0.871, 0.872] | 0.842 [0.842, 0.843] | 0.787 [0.786, 0.788] | 0.815 [0.814, 0.815] | 0.748 [0.747, 0.749] |
| 20 | **0.926 [0.925, 0.926]** | **0.905** | 0.909 [0.908, 0.909] | 0.892 [0.892, 0.892] | 0.815 [0.814, 0.816] | 0.834 [0.833, 0.835] | 0.813 [0.812, 0.813] |
| 25 | 0.944 [0.943, 0.944]     | ---       | 0.937 [0.937, 0.938] | 0.931 [0.931, 0.932] | 0.896 [0.896, 0.897] | 0.896 [0.896, 0.897] | 0.878 [0.877, 0.879] |
| 30 | 0.959 [0.959, 0.960]     | ---       | 0.956 [0.955, 0.956] | 0.951 [0.951, 0.951] | 0.927 [0.927, 0.927] | 0.927 [0.927, 0.927] | 0.924 [0.923, 0.924] |
| 40 | 0.983 [0.983, 0.983]     | ---       | 0.984 [0.984, 0.984] | 0.981 [0.981, 0.981] | 0.967 [0.967, 0.967] | 0.967 [0.967, 0.967] | 0.977 [0.977, 0.977] |
| 50 | 0.999 [0.999, 0.999]     | ---       | 0.999 [0.999, 0.999] | 0.999 [0.999, 0.999] | 0.999 [0.999, 0.999] | 0.999 [0.999, 0.999] | 0.999 [0.999, 0.999] |
<!-- END:ablation_baselines_details -->

### Cross-Variant Per-Domain K=20 (All Runs)

Per-domain K=20 breakdown (domain-balanced, Mini-IPIP, first-N) for each variant.

<!-- BEGIN:ablation_per_domain_k20_details -->
#### Reference


**Domain-Balanced (4 items per domain):**

| Domain              | r      | Items | CI Lower | CI Upper |
|---------------------|--------|-------|----------|----------|
| Extraversion        | 0.9459 | 4     | 0.9453   | 0.9466   |
| Agreeableness       | 0.9180 | 4     | 0.9171   | 0.9190   |
| Conscientiousness   | 0.9193 | 4     | 0.9184   | 0.9203   |
| Emotional Stability | 0.9364 | 4     | 0.9356   | 0.9371   |
| Intellect/Openness  | 0.9099 | 4     | 0.9088   | 0.9111   |

**Mini-IPIP (4 items per domain):**

| Domain              | r      | Items | CI Lower | CI Upper |
|---------------------|--------|-------|----------|----------|
| Extraversion        | 0.9369 | 4     | ---      | ---      |
| Agreeableness       | 0.9101 | 4     | ---      | ---      |
| Conscientiousness   | 0.9097 | 4     | ---      | ---      |
| Emotional Stability | 0.9254 | 4     | ---      | ---      |
| Intellect/Openness  | 0.8437 | 4     | ---      | ---      |

**First-N (4 items per domain):**

| Domain              | r      | Items | CI Lower | CI Upper |
|---------------------|--------|-------|----------|----------|
| Extraversion        | 0.9384 | 4     | 0.9376   | 0.9392   |
| Agreeableness       | 0.9229 | 4     | 0.9220   | 0.9239   |
| Conscientiousness   | 0.8946 | 4     | 0.8935   | 0.8958   |
| Emotional Stability | 0.8730 | 4     | 0.8715   | 0.8744   |
| Intellect/Openness  | 0.9128 | 4     | 0.9117   | 0.9138   |

#### Ablation: No Sparsity


**Domain-Balanced (4 items per domain):**

| Domain              | r      | Items | CI Lower | CI Upper |
|---------------------|--------|-------|----------|----------|
| Extraversion        | 0.8956 | 4     | 0.8947   | 0.8964   |
| Agreeableness       | 0.8475 | 4     | 0.8464   | 0.8487   |
| Conscientiousness   | 0.8582 | 4     | 0.8571   | 0.8594   |
| Emotional Stability | 0.8761 | 4     | 0.8752   | 0.8771   |
| Intellect/Openness  | 0.8468 | 4     | 0.8455   | 0.8482   |

**Mini-IPIP (4 items per domain):**

| Domain              | r      | Items | CI Lower | CI Upper |
|---------------------|--------|-------|----------|----------|
| Extraversion        | 0.9369 | 4     | ---      | ---      |
| Agreeableness       | 0.9101 | 4     | ---      | ---      |
| Conscientiousness   | 0.9097 | 4     | ---      | ---      |
| Emotional Stability | 0.9254 | 4     | ---      | ---      |
| Intellect/Openness  | 0.8437 | 4     | ---      | ---      |

**First-N (4 items per domain):**

| Domain              | r      | Items | CI Lower | CI Upper |
|---------------------|--------|-------|----------|----------|
| Extraversion        | 0.8849 | 4     | 0.8839   | 0.8858   |
| Agreeableness       | 0.8404 | 4     | 0.8391   | 0.8417   |
| Conscientiousness   | 0.8208 | 4     | 0.8194   | 0.8223   |
| Emotional Stability | 0.8142 | 4     | 0.8125   | 0.8161   |
| Intellect/Openness  | 0.8354 | 4     | 0.8340   | 0.8368   |

#### Ablation: Focused Only


**Domain-Balanced (4 items per domain):**

| Domain              | r      | Items | CI Lower | CI Upper |
|---------------------|--------|-------|----------|----------|
| Extraversion        | 0.9457 | 4     | 0.9450   | 0.9464   |
| Agreeableness       | 0.9182 | 4     | 0.9173   | 0.9191   |
| Conscientiousness   | 0.9187 | 4     | 0.9178   | 0.9198   |
| Emotional Stability | 0.9355 | 4     | 0.9347   | 0.9363   |
| Intellect/Openness  | 0.9094 | 4     | 0.9083   | 0.9105   |

**Mini-IPIP (4 items per domain):**

| Domain              | r      | Items | CI Lower | CI Upper |
|---------------------|--------|-------|----------|----------|
| Extraversion        | 0.9369 | 4     | ---      | ---      |
| Agreeableness       | 0.9101 | 4     | ---      | ---      |
| Conscientiousness   | 0.9097 | 4     | ---      | ---      |
| Emotional Stability | 0.9254 | 4     | ---      | ---      |
| Intellect/Openness  | 0.8437 | 4     | ---      | ---      |

**First-N (4 items per domain):**

| Domain              | r      | Items | CI Lower | CI Upper |
|---------------------|--------|-------|----------|----------|
| Extraversion        | 0.9385 | 4     | 0.9377   | 0.9393   |
| Agreeableness       | 0.9229 | 4     | 0.9220   | 0.9239   |
| Conscientiousness   | 0.8951 | 4     | 0.8940   | 0.8963   |
| Emotional Stability | 0.8732 | 4     | 0.8717   | 0.8746   |
| Intellect/Openness  | 0.9125 | 4     | 0.9114   | 0.9135   |

#### Ablation: Stratified Split


**Domain-Balanced (4 items per domain):**

| Domain              | r      | Items | CI Lower | CI Upper |
|---------------------|--------|-------|----------|----------|
| Extraversion        | 0.9454 | 4     | 0.9447   | 0.9461   |
| Agreeableness       | 0.9175 | 4     | 0.9165   | 0.9184   |
| Conscientiousness   | 0.9188 | 4     | 0.9178   | 0.9198   |
| Emotional Stability | 0.9357 | 4     | 0.9348   | 0.9364   |
| Intellect/Openness  | 0.9095 | 4     | 0.9084   | 0.9106   |

**Mini-IPIP (4 items per domain):**

| Domain              | r      | Items | CI Lower | CI Upper |
|---------------------|--------|-------|----------|----------|
| Extraversion        | 0.9370 | 4     | ---      | ---      |
| Agreeableness       | 0.9094 | 4     | ---      | ---      |
| Conscientiousness   | 0.9092 | 4     | ---      | ---      |
| Emotional Stability | 0.9249 | 4     | ---      | ---      |
| Intellect/Openness  | 0.8426 | 4     | ---      | ---      |

**First-N (4 items per domain):**

| Domain              | r      | Items | CI Lower | CI Upper |
|---------------------|--------|-------|----------|----------|
| Extraversion        | 0.9384 | 4     | 0.9376   | 0.9392   |
| Agreeableness       | 0.9220 | 4     | 0.9211   | 0.9231   |
| Conscientiousness   | 0.8955 | 4     | 0.8943   | 0.8966   |
| Emotional Stability | 0.8744 | 4     | 0.8729   | 0.8757   |
| Intellect/Openness  | 0.9128 | 4     | 0.9118   | 0.9139   |
<!-- END:ablation_per_domain_k20_details -->

### Cross-Variant Domain Starvation at K=20 (All Runs)

Adaptive top-K domain allocation and resulting per-domain accuracy for each variant.

<!-- BEGIN:ablation_domain_starvation_details -->
#### Reference


| Domain              | Items | Share | r      | CI Lower | CI Upper |
|---------------------|-------|-------|--------|----------|----------|
| Extraversion        | 9     | 45%   | 0.9935 | 0.9935   | 0.9936   |
| Emotional Stability | 6     | 30%   | 0.9692 | 0.9688   | 0.9696   |
| Agreeableness       | 3     | 15%   | 0.8214 | 0.8194   | 0.8233   |
| Conscientiousness   | 2     | 10%   | 0.7577 | 0.7552   | 0.7601   |
| Intellect/Openness  | 0     | 0%    | 0.4107 | 0.4059   | 0.4156   |

#### Ablation: No Sparsity


| Domain              | Items | Share | r       | CI Lower | CI Upper |
|---------------------|-------|-------|---------|----------|----------|
| Extraversion        | 9     | 45%   | 0.9883  | 0.9883   | 0.9884   |
| Emotional Stability | 6     | 30%   | 0.9127  | 0.9121   | 0.9133   |
| Agreeableness       | 3     | 15%   | 0.7516  | 0.7493   | 0.7537   |
| Conscientiousness   | 2     | 10%   | 0.6983  | 0.6958   | 0.7012   |
| Intellect/Openness  | 0     | 0%    | -0.1318 | -0.1383  | -0.1258  |

#### Ablation: Focused Only


| Domain              | Items | Share | r      | CI Lower | CI Upper |
|---------------------|-------|-------|--------|----------|----------|
| Extraversion        | 9     | 45%   | 0.9935 | 0.9934   | 0.9936   |
| Emotional Stability | 6     | 30%   | 0.9683 | 0.9679   | 0.9687   |
| Agreeableness       | 3     | 15%   | 0.8156 | 0.8135   | 0.8175   |
| Conscientiousness   | 2     | 10%   | 0.7545 | 0.7519   | 0.7570   |
| Intellect/Openness  | 0     | 0%    | 0.4020 | 0.3972   | 0.4071   |

#### Ablation: Stratified Split


| Domain              | Items | Share | r      | CI Lower | CI Upper |
|---------------------|-------|-------|--------|----------|----------|
| Extraversion        | 9     | 45%   | 0.9936 | 0.9935   | 0.9937   |
| Emotional Stability | 6     | 30%   | 0.9684 | 0.9680   | 0.9688   |
| Agreeableness       | 3     | 15%   | 0.8160 | 0.8139   | 0.8181   |
| Conscientiousness   | 2     | 10%   | 0.7525 | 0.7500   | 0.7550   |
| Intellect/Openness  | 0     | 0%    | 0.4040 | 0.3997   | 0.4086   |
<!-- END:ablation_domain_starvation_details -->

### Cross-Variant ML vs Averaging (All Runs)

ML-vs-averaging deltas for matched item sets across each variant.

<!-- BEGIN:ablation_ml_vs_averaging_details -->
#### Reference


| Strategy        | K  | ML r   | Avg r  | Delta r | ML MAE | Avg MAE | Delta MAE |
|-----------------|----|--------|--------|---------|--------|---------|-----------|
| Domain-balanced | 10 | 0.8520 | 0.8285 | +0.0234 | 11.77  | 14.14   | -2.37     |
| Domain-balanced | 15 | 0.9073 | 0.8858 | +0.0215 | 9.20   | 11.29   | -2.10     |
| Domain-balanced | 20 | 0.9262 | 0.9102 | +0.0159 | 8.21   | 9.86    | -1.65     |
| Mini-IPIP       | 20 | 0.9168 | 0.9056 | +0.0112 | 8.63   | 9.23    | -0.60     |
| Domain-balanced | 25 | 0.9441 | 0.9308 | +0.0133 | 7.19   | 8.50    | -1.31     |

#### Ablation: No Sparsity


| Strategy        | K  | ML r   | Avg r  | Delta r | ML MAE | Avg MAE | Delta MAE |
|-----------------|----|--------|--------|---------|--------|---------|-----------|
| Domain-balanced | 10 | 0.5614 | 0.8285 | -0.2672 | 43.34  | 14.14   | +29.20    |
| Domain-balanced | 15 | 0.7235 | 0.8858 | -0.1623 | 39.84  | 11.29   | +28.54    |
| Domain-balanced | 20 | 0.7899 | 0.9102 | -0.1204 | 35.46  | 9.86    | +25.60    |
| Mini-IPIP       | 20 | 0.7636 | 0.9056 | -0.1419 | 36.13  | 9.23    | +26.90    |
| Domain-balanced | 25 | 0.8582 | 0.9308 | -0.0726 | 30.09  | 8.50    | +21.60    |

#### Ablation: Focused Only


| Strategy        | K  | ML r   | Avg r  | Delta r | ML MAE | Avg MAE | Delta MAE |
|-----------------|----|--------|--------|---------|--------|---------|-----------|
| Domain-balanced | 10 | 0.8515 | 0.8285 | +0.0230 | 11.75  | 14.14   | -2.39     |
| Domain-balanced | 15 | 0.9071 | 0.8858 | +0.0213 | 9.20   | 11.29   | -2.09     |
| Domain-balanced | 20 | 0.9257 | 0.9102 | +0.0155 | 8.24   | 9.86    | -1.62     |
| Mini-IPIP       | 20 | 0.9168 | 0.9056 | +0.0113 | 8.63   | 9.23    | -0.60     |
| Domain-balanced | 25 | 0.9439 | 0.9308 | +0.0131 | 7.21   | 8.50    | -1.28     |

#### Ablation: Stratified Split


| Strategy        | K  | ML r   | Avg r  | Delta r | ML MAE | Avg MAE | Delta MAE |
|-----------------|----|--------|--------|---------|--------|---------|-----------|
| Domain-balanced | 10 | 0.8512 | 0.8277 | +0.0235 | 11.77  | 14.19   | -2.42     |
| Domain-balanced | 15 | 0.9070 | 0.8855 | +0.0215 | 9.21   | 11.30   | -2.09     |
| Domain-balanced | 20 | 0.9256 | 0.9097 | +0.0159 | 8.24   | 9.88    | -1.64     |
| Mini-IPIP       | 20 | 0.9166 | 0.9051 | +0.0116 | 8.63   | 9.25    | -0.61     |
| Domain-balanced | 25 | 0.9436 | 0.9303 | +0.0133 | 7.22   | 8.52    | -1.29     |
<!-- END:ablation_ml_vs_averaging_details -->

### Cross-Variant Simulation Results (All Runs)

Adaptive simulation outcomes for each variant at the operating point.

<!-- BEGIN:ablation_simulation_details -->
#### Reference


| Domain              | r          | MAE      | RMSE      | Within-5  | 90% Coverage |
|---------------------|------------|----------|-----------|-----------|--------------|
| Extraversion        | 0.9362     | 7.77     | 10.60     | 46.0%     | 91.6%        |
| Agreeableness       | 0.9161     | 8.57     | 11.77     | 43.2%     | 87.6%        |
| Conscientiousness   | 0.9215     | 8.62     | 11.65     | 41.6%     | 89.1%        |
| Emotional Stability | 0.9405     | 7.54     | 10.27     | 47.0%     | 90.3%        |
| Intellect/Openness  | 0.9095     | 9.00     | 12.49     | 42.6%     | 90.3%        |
| **Overall**         | **0.9249** | **8.30** | **11.39** | **44.1%** | **89.8%**    |

#### Ablation: No Sparsity


| Domain              | r          | MAE       | RMSE      | Within-5 | 90% Coverage |
|---------------------|------------|-----------|-----------|----------|--------------|
| Extraversion        | 0.8823     | 39.70     | 46.08     | 9.1%     | 1.4%         |
| Agreeableness       | 0.8471     | 30.57     | 36.02     | 10.7%    | 2.8%         |
| Conscientiousness   | 0.8604     | 35.40     | 41.11     | 9.1%     | 1.7%         |
| Emotional Stability | 0.8774     | 41.07     | 47.19     | 7.4%     | 1.1%         |
| Intellect/Openness  | 0.8474     | 31.00     | 36.71     | 12.1%    | 4.1%         |
| **Overall**         | **0.7855** | **35.55** | **41.68** | **9.7%** | **2.2%**     |

#### Ablation: Focused Only


| Domain              | r          | MAE      | RMSE      | Within-5  | 90% Coverage |
|---------------------|------------|----------|-----------|-----------|--------------|
| Extraversion        | 0.9360     | 7.80     | 10.62     | 45.9%     | 91.5%        |
| Agreeableness       | 0.9165     | 8.57     | 11.75     | 43.2%     | 88.5%        |
| Conscientiousness   | 0.9210     | 8.67     | 11.70     | 41.8%     | 89.7%        |
| Emotional Stability | 0.9402     | 7.53     | 10.27     | 47.5%     | 89.8%        |
| Intellect/Openness  | 0.9092     | 9.05     | 12.58     | 42.6%     | 90.1%        |
| **Overall**         | **0.9247** | **8.32** | **11.42** | **44.2%** | **89.9%**    |

#### Ablation: Stratified Split


| Domain              | r          | MAE      | RMSE      | Within-5  | 90% Coverage |
|---------------------|------------|----------|-----------|-----------|--------------|
| Extraversion        | 0.9363     | 7.67     | 10.53     | 46.7%     | 92.2%        |
| Agreeableness       | 0.9178     | 8.62     | 11.73     | 41.9%     | 88.5%        |
| Conscientiousness   | 0.9181     | 8.66     | 11.73     | 41.6%     | 89.5%        |
| Emotional Stability | 0.9373     | 7.68     | 10.52     | 46.5%     | 89.6%        |
| Intellect/Openness  | 0.9136     | 8.88     | 12.29     | 43.2%     | 90.0%        |
| **Overall**         | **0.9247** | **8.30** | **11.38** | **44.0%** | **90.0%**    |
<!-- END:ablation_simulation_details -->

### Dataset

Valid respondents from the Open-Source Psychometrics Project, stratified
split into train/validation/test.

<!-- BEGIN:data_splits -->
| Split       | Respondents | Fraction |
|-------------|-------------|----------|
| Total valid | 874,434     | 100%     |
| Train       | 612,103     | 70.0%    |
| Validation  | 131,165     | 15.0%    |
| Test        | 131,166     | 15.0%    |

Stratification: ext_q * 5 + est_q (25 strata) (25 strata, seed=42).

| Domain              | Max Mean Diff | KS Statistic | KS p-value |
|---------------------|---------------|--------------|------------|
| Extraversion        | 0.0006        | 0.0019       | 0.823      |
| Agreeableness       | 0.0005        | 0.0009       | 1.000      |
| Conscientiousness   | 0.0032        | 0.0027       | 0.396      |
| Emotional Stability | 0.0006        | 0.0011       | 1.000      |
| Intellect/Openness  | 0.0032        | 0.0013       | 0.995      |
<!-- END:data_splits -->

### Training Configuration

Sparsity augmentation and training settings from `configs/reference.yaml`.

<!-- BEGIN:training_config -->
| Setting                     | Value     |
|-----------------------------|-----------|
| Config name                 | reference |
| Sparsity enabled            | True      |
| Focused bucketing           | True      |
| Include Mini-IPIP patterns  | True      |
| Include imbalanced patterns | True      |
| Augmentation passes         | 3         |
| CV folds                    | 5         |
| Random state                | 42        |
| Min Pearson r gate          | 0.90      |
| Min 90% coverage gate       | 0.88      |
<!-- END:training_config -->

### Model Configuration

<!-- BEGIN:model_config -->
| Parameter        | Value                                       |
|------------------|---------------------------------------------|
| Algorithm        | XGBoost quantile regression (pinball loss)  |
| Models           | 15 (5 domains x 3 quantiles: q05, q50, q95) |
| n_estimators     | 1,995                                       |
| max_depth        | 5                                           |
| learning_rate    | 0.0230                                      |
| min_child_weight | 10                                          |
| subsample        | 0.644                                       |
| colsample_bytree | 0.544                                       |
| reg_lambda       | 1.817                                       |
| reg_alpha        | 2.145                                       |
| RNG seed         | 42                                          |
<!-- END:model_config -->

### Hyperparameter Overrides

If any hyperparameters were manually adjusted after Optuna tuning (e.g., to
control model size for deployment), the changes are logged here.

<!-- BEGIN:hyperparameter_overrides -->
*No manual overrides. All hyperparameters as selected by Optuna.*
<!-- END:hyperparameter_overrides -->

### Population Norms

Derived from the training-split OSPP respondents (see dataset table for count).
Used for raw-score to percentile conversion via
`percentile = phi((raw - mu) / sigma) * 100`.

<!-- BEGIN:norms -->
| Domain              | Mean   | SD     |
|---------------------|--------|--------|
| Extraversion        | 2.9595 | 0.9104 |
| Agreeableness       | 3.7736 | 0.7293 |
| Conscientiousness   | 3.3661 | 0.7373 |
| Emotional Stability | 2.9334 | 0.8586 |
| Intellect/Openness  | 3.8989 | 0.6302 |
<!-- END:norms -->

### Ceiling Check: Full-Model Accuracy (50 Items, 90K Test)

Sanity check with all 50 items present. Near-perfect reconstruction is
expected; these numbers confirm the model and inference pipeline work
correctly, not the operating-point accuracy. See the K=20 tables below for
the primary evaluation.

<!-- BEGIN:validation -->
| Domain              | r          | MAE      | RMSE     | Within-5  | 90% Coverage | Raw Crossing Rate |
|---------------------|------------|----------|----------|-----------|--------------|-------------------|
| Extraversion        | 0.9997     | 0.48     | 0.71     | 100.0%    | 94.0%        | 32.5%             |
| Agreeableness       | 0.9994     | 0.72     | 1.01     | 99.9%     | 93.4%        | 24.3%             |
| Conscientiousness   | 0.9993     | 0.80     | 1.10     | 99.9%     | 91.0%        | 21.9%             |
| Emotional Stability | 0.9995     | 0.64     | 0.90     | 100.0%    | 93.8%        | 25.8%             |
| Intellect/Openness  | 0.9990     | 0.98     | 1.34     | 99.7%     | 94.6%        | 20.5%             |
| **Overall**         | **0.9994** | **0.72** | **1.03** | **99.9%** | ---          | ---               |
<!-- END:validation -->

Coverage: all domains exceed the 90% nominal target, as the table shows. Raw
crossing rates confirm that the nonlinear raw-to-percentile CDF transform
frequently inverts quantile ordering. The post-transform sort operation catches
and corrects these inversions.

### Ceiling Check by Quintile (50 Items, 90K Test)

Per-domain performance varies systematically by score quintile (best in tails,
worst near the center), and interval widths expand near the middle of the
distribution.

<!-- BEGIN:validation_quintiles -->
| Domain                             | Quintile | n      | MAE      | 90% Coverage | Mean PI Width |
|------------------------------------|----------|--------|----------|--------------|---------------|
| Extraversion                       | Q1       | 27,302 | 0.23     | 91.4%        | 1.19          |
| Extraversion                       | Q2       | 27,898 | 0.64     | 94.9%        | 3.28          |
| Extraversion                       | Q3       | 25,724 | 0.76     | 95.6%        | 4.12          |
| Extraversion                       | Q4       | 26,075 | 0.56     | 95.3%        | 3.09          |
| Extraversion                       | Q5       | 24,167 | 0.20     | 92.8%        | 1.08          |
| Agreeableness                      | Q1       | 28,722 | 0.60     | 90.9%        | 2.77          |
| Agreeableness                      | Q2       | 28,650 | 1.01     | 96.6%        | 5.81          |
| Agreeableness                      | Q3       | 22,368 | 0.94     | 97.6%        | 5.71          |
| Agreeableness                      | Q4       | 27,227 | 0.71     | 96.9%        | 4.06          |
| Agreeableness                      | Q5       | 24,199 | 0.30     | 84.6%        | 1.37          |
| Conscientiousness                  | Q1       | 27,351 | 0.52     | 86.3%        | 2.26          |
| Conscientiousness                  | Q2       | 29,921 | 1.03     | 92.6%        | 5.25          |
| Conscientiousness                  | Q3       | 26,248 | 1.09     | 95.6%        | 6.33          |
| Conscientiousness                  | Q4       | 22,572 | 0.89     | 93.9%        | 4.97          |
| Conscientiousness                  | Q5       | 25,074 | 0.43     | 86.8%        | 1.91          |
| Emotional Stability                | Q1       | 30,611 | 0.33     | 91.1%        | 1.77          |
| Emotional Stability                | Q2       | 25,691 | 0.81     | 96.0%        | 4.50          |
| Emotional Stability                | Q3       | 27,356 | 0.95     | 96.8%        | 5.41          |
| Emotional Stability                | Q4       | 23,047 | 0.78     | 94.4%        | 4.04          |
| Emotional Stability                | Q5       | 24,461 | 0.34     | 91.1%        | 1.61          |
| Intellect/Openness                 | Q1       | 31,150 | 0.84     | 92.1%        | 3.58          |
| Intellect/Openness                 | Q2       | 27,149 | 1.33     | 97.7%        | 7.54          |
| Intellect/Openness                 | Q3       | 23,510 | 1.26     | 99.3%        | 8.33          |
| Intellect/Openness                 | Q4       | 27,745 | 0.99     | 98.6%        | 5.95          |
| Intellect/Openness                 | Q5       | 21,612 | 0.43     | 84.1%        | 1.76          |
| **Avg Tails (Q1/Q5, all domains)** | ---      | ---    | **0.42** | **89.1%**    | **1.93**      |
| **Avg Center (Q3, all domains)**   | ---      | ---    | **1.00** | **97.0%**    | **5.98**      |
<!-- END:validation_quintiles -->

### Item Selection Strategies (5-50 Items, 90K Test)

Pearson r with full 50-item scores. Bootstrap 95% CIs from 1,000 resamples.

<!-- BEGIN:baselines -->
| K  | Domain-Balanced          | Mini-IPIP | First-N              | Random               | Adaptive Top-K       | Greedy-Balanced      | Worst-K              |
|----|--------------------------|-----------|----------------------|----------------------|----------------------|----------------------|----------------------|
| 5  | 0.742 [0.741, 0.743]     | ---       | 0.694 [0.692, 0.695] | 0.598 [0.598, 0.599] | 0.570 [0.568, 0.571] | 0.655 [0.653, 0.657] | 0.473 [0.471, 0.474] |
| 10 | 0.852 [0.851, 0.853]     | ---       | 0.818 [0.817, 0.819] | 0.761 [0.761, 0.762] | 0.702 [0.700, 0.703] | 0.753 [0.752, 0.754] | 0.667 [0.665, 0.668] |
| 15 | 0.907 [0.907, 0.908]     | ---       | 0.871 [0.870, 0.872] | 0.843 [0.843, 0.843] | 0.793 [0.792, 0.794] | 0.818 [0.817, 0.819] | 0.754 [0.753, 0.755] |
| 20 | **0.926 [0.926, 0.927]** | **0.906** | 0.908 [0.908, 0.909] | 0.893 [0.892, 0.893] | 0.820 [0.819, 0.821] | 0.838 [0.837, 0.839] | 0.819 [0.818, 0.820] |
| 25 | 0.944 [0.944, 0.944]     | ---       | 0.937 [0.937, 0.938] | 0.932 [0.931, 0.932] | 0.898 [0.897, 0.898] | 0.898 [0.897, 0.898] | 0.883 [0.882, 0.883] |
| 30 | 0.960 [0.959, 0.960]     | ---       | 0.955 [0.955, 0.956] | 0.951 [0.950, 0.951] | 0.927 [0.927, 0.928] | 0.927 [0.927, 0.928] | 0.926 [0.926, 0.926] |
| 40 | 0.983 [0.982, 0.983]     | ---       | 0.984 [0.984, 0.984] | 0.981 [0.981, 0.981] | 0.975 [0.974, 0.975] | 0.975 [0.974, 0.975] | 0.977 [0.977, 0.977] |
| 50 | 0.999 [0.999, 0.999]     | ---       | 0.999 [0.999, 0.999] | 0.999 [0.999, 0.999] | 0.999 [0.999, 0.999] | 0.999 [0.999, 0.999] | 0.999 [0.999, 0.999] |
<!-- END:baselines -->

At K>=25, greedy-balanced and adaptive top-K converge (identical item sets once the
greedy tail dominates). At K=40, greedy approaches nearly match the full-scale
ceiling because most items are included regardless of selection strategy. The
paradox is about the operating range of interest (10-25 items), not item selection
in general.

### Per-Domain Breakdown at K=20

<!-- BEGIN:per_domain_k20 -->
**Domain-Balanced (4 items per domain):**

| Domain              | r      | Items | CI Lower | CI Upper |
|---------------------|--------|-------|----------|----------|
| Extraversion        | 0.9459 | 4     | 0.9453   | 0.9466   |
| Agreeableness       | 0.9180 | 4     | 0.9171   | 0.9190   |
| Conscientiousness   | 0.9193 | 4     | 0.9184   | 0.9203   |
| Emotional Stability | 0.9364 | 4     | 0.9356   | 0.9371   |
| Intellect/Openness  | 0.9099 | 4     | 0.9088   | 0.9111   |

**Mini-IPIP (4 items per domain):**

| Domain              | r      | Items | CI Lower | CI Upper |
|---------------------|--------|-------|----------|----------|
| Extraversion        | 0.9369 | 4     | ---      | ---      |
| Agreeableness       | 0.9101 | 4     | ---      | ---      |
| Conscientiousness   | 0.9097 | 4     | ---      | ---      |
| Emotional Stability | 0.9254 | 4     | ---      | ---      |
| Intellect/Openness  | 0.8437 | 4     | ---      | ---      |

**First-N (4 items per domain):**

| Domain              | r      | Items | CI Lower | CI Upper |
|---------------------|--------|-------|----------|----------|
| Extraversion        | 0.9384 | 4     | 0.9376   | 0.9392   |
| Agreeableness       | 0.9229 | 4     | 0.9220   | 0.9239   |
| Conscientiousness   | 0.8946 | 4     | 0.8935   | 0.8958   |
| Emotional Stability | 0.8730 | 4     | 0.8715   | 0.8744   |
| Intellect/Openness  | 0.9128 | 4     | 0.9117   | 0.9138   |
<!-- END:per_domain_k20 -->

Domain-balanced achieves the most uniform per-domain performance. Mini-IPIP shows
a clear Intellect/Openness gap relative to domain-balanced (visible in the table
above) because its Intellect items were selected for brevity, not discrimination.

### Domain Starvation (Greedy Selection at K=20)

<!-- BEGIN:domain_starvation -->
| Domain              | Items | Share | r      | CI Lower | CI Upper |
|---------------------|-------|-------|--------|----------|----------|
| Extraversion        | 9     | 45%   | 0.9935 | 0.9935   | 0.9936   |
| Emotional Stability | 6     | 30%   | 0.9692 | 0.9688   | 0.9696   |
| Agreeableness       | 3     | 15%   | 0.8214 | 0.8194   | 0.8233   |
| Conscientiousness   | 2     | 10%   | 0.7577 | 0.7552   | 0.7601   |
| Intellect/Openness  | 0     | 0%    | 0.4107 | 0.4059   | 0.4156   |
<!-- END:domain_starvation -->

Intellect first appears just outside the top 20 in the greedy ranking (see the
greedy item ranking table below). The table above shows the resulting allocation:
Extraversion dominates because its items correlate moderately with all other
domains, while Intellect items rank last because they are the most
psychometrically independent factor.

### ML vs Simple Averaging (Same Items)

<!-- BEGIN:ml_vs_averaging -->
| Strategy        | K  | ML r   | Avg r  | Delta r | ML MAE | Avg MAE | Delta MAE |
|-----------------|----|--------|--------|---------|--------|---------|-----------|
| Domain-balanced | 10 | 0.8520 | 0.8285 | +0.0234 | 11.77  | 14.14   | -2.37     |
| Domain-balanced | 15 | 0.9073 | 0.8858 | +0.0215 | 9.20   | 11.29   | -2.10     |
| Domain-balanced | 20 | 0.9262 | 0.9102 | +0.0159 | 8.21   | 9.86    | -1.65     |
| Mini-IPIP       | 20 | 0.9168 | 0.9056 | +0.0112 | 8.63   | 9.23    | -0.60     |
| Domain-balanced | 25 | 0.9441 | 0.9308 | +0.0133 | 7.19   | 8.50    | -1.31     |
<!-- END:ml_vs_averaging -->

The ML advantage holds across all tested item counts, as the table shows, and is
largest at fewer items where cross-domain information sharing matters most. Both
correlation and MAE improvements are consistent across budgets.

### Simulation Results (20-Item Operating Point)

Held-out respondents from the test split, correlation-ranked selection,
SEM threshold 0.45, min 4 items per domain. All respondents converge to
exactly 20 items (4-4-4-4-4).

<!-- BEGIN:simulation -->
| Domain              | r          | MAE      | RMSE      | Within-5  | 90% Coverage |
|---------------------|------------|----------|-----------|-----------|--------------|
| Extraversion        | 0.9362     | 7.77     | 10.60     | 46.0%     | 91.6%        |
| Agreeableness       | 0.9161     | 8.57     | 11.77     | 43.2%     | 87.6%        |
| Conscientiousness   | 0.9215     | 8.62     | 11.65     | 41.6%     | 89.1%        |
| Emotional Stability | 0.9405     | 7.54     | 10.27     | 47.0%     | 90.3%        |
| Intellect/Openness  | 0.9095     | 9.00     | 12.49     | 42.6%     | 90.3%        |
| **Overall**         | **0.9249** | **8.30** | **11.39** | **44.1%** | **89.8%**    |
<!-- END:simulation -->

Simulation results closely match the static baseline evaluation (compare the
overall *r* here to the headline results table). Correlation-ranked selection is
equivalent to the optimal static strategy; the agreement between the two also
confirms the pipeline end-to-end.

### Calibration

<!-- BEGIN:calibration -->
Calibration regime: `sparse_20_balanced` (domain-balanced, 20 items, 4 per domain).

| Domain              | Observed Coverage | Scale Factor |
|---------------------|-------------------|--------------|
| Extraversion        | 91.0%             | 1.0          |
| Agreeableness       | 90.1%             | 1.0          |
| Conscientiousness   | 90.2%             | 1.0          |
| Emotional Stability | 90.6%             | 1.0          |
| Intellect/Openness  | 90.0%             | 1.0          |
<!-- END:calibration -->

All domains achieve near-nominal 90% coverage without requiring any scaling
adjustment. This is a direct result of the sparse calibration fix (B5.2) which
aligned the calibration procedure with the production operating point.

### Calibration Policy

When each calibration regime is applied, based on item count.

<!-- BEGIN:calibration_policy -->
| Condition                                | Calibration Regime   |
|------------------------------------------|----------------------|
| 50+ items (full scale)                   | `full_50`            |
| Below 50 items                           | `sparse_20_balanced` |
| Fallback (if sparse calibration missing) | `none`               |
<!-- END:calibration_policy -->

### Domain-Balanced 20-Item Set

Top 4 items per domain by within-domain correlation. These are the items selected
by the `domain_balanced` strategy.

<!-- BEGIN:domain_balanced_items -->
| Rank | Item  | Domain              | Own-Domain r | Reverse-Keyed |
|------|-------|---------------------|--------------|---------------|
| 1    | ext4  | Extraversion        | 0.713        | yes           |
| 2    | ext5  | Extraversion        | 0.705        | no            |
| 3    | ext7  | Extraversion        | 0.688        | no            |
| 4    | ext2  | Extraversion        | 0.674        | yes           |
| 5    | agr4  | Agreeableness       | 0.710        | no            |
| 6    | agr9  | Agreeableness       | 0.635        | no            |
| 7    | agr7  | Agreeableness       | 0.617        | yes           |
| 8    | agr5  | Agreeableness       | 0.617        | yes           |
| 9    | csn6  | Conscientiousness   | 0.585        | yes           |
| 10   | csn1  | Conscientiousness   | 0.580        | no            |
| 11   | csn5  | Conscientiousness   | 0.568        | no            |
| 12   | csn4  | Conscientiousness   | 0.558        | yes           |
| 13   | est8  | Emotional Stability | 0.691        | yes           |
| 14   | est6  | Emotional Stability | 0.684        | yes           |
| 15   | est1  | Emotional Stability | 0.668        | yes           |
| 16   | est7  | Emotional Stability | 0.664        | yes           |
| 17   | opn10 | Intellect/Openness  | 0.598        | no            |
| 18   | opn1  | Intellect/Openness  | 0.533        | no            |
| 19   | opn2  | Intellect/Openness  | 0.529        | yes           |
| 20   | opn5  | Intellect/Openness  | 0.515        | no            |
<!-- END:domain_balanced_items -->

This 20-item set differs from the Mini-IPIP: the domain-balanced set selects by
maximum within-domain correlation, while Mini-IPIP was designed for brevity and
broad coverage.

### Greedy Item Ranking (Cross-Domain Info Score)

Top 20 items by cross-domain information score. The domain distribution is heavily
skewed, which drives the domain starvation mechanism described above (see the
domain starvation table for counts).

<!-- BEGIN:greedy_ranking -->
| Rank | Item  | Domain | Own-Domain r | Info Score | Ext   | Agr   | Csn   | Est   | Opn   |
|------|-------|--------|--------------|------------|-------|-------|-------|-------|-------|
| 1    | ext3  | ext    | 0.637        | 1.524      | 0.637 | 0.357 | 0.148 | 0.327 | 0.054 |
| 2    | ext5  | ext    | 0.705        | 1.446      | 0.705 | 0.325 | 0.110 | 0.166 | 0.140 |
| 3    | ext7  | ext    | 0.688        | 1.272      | 0.688 | 0.268 | 0.058 | 0.169 | 0.090 |
| 4    | est10 | est    | 0.611        | 1.269      | 0.294 | 0.086 | 0.255 | 0.611 | 0.022 |
| 5    | ext4  | ext    | 0.713        | 1.223      | 0.713 | 0.186 | 0.063 | 0.187 | 0.075 |
| 6    | ext6  | ext    | 0.548        | 1.264      | 0.548 | 0.241 | 0.062 | 0.123 | 0.291 |
| 7    | est9  | est    | 0.624        | 1.184      | 0.139 | 0.185 | 0.151 | 0.624 | 0.085 |
| 8    | agr10 | agr    | 0.417        | 1.246      | 0.366 | 0.417 | 0.154 | 0.174 | 0.135 |
| 9    | agr7  | agr    | 0.617        | 1.270      | 0.374 | 0.617 | 0.079 | 0.092 | 0.108 |
| 10   | ext10 | ext    | 0.661        | 1.223      | 0.661 | 0.186 | 0.070 | 0.221 | 0.084 |
| 11   | csn4  | csn    | 0.558        | 1.173      | 0.100 | 0.101 | 0.558 | 0.373 | 0.041 |
| 12   | est8  | est    | 0.691        | 1.159      | 0.105 | 0.057 | 0.250 | 0.691 | 0.055 |
| 13   | agr2  | agr    | 0.540        | 1.217      | 0.401 | 0.540 | 0.053 | 0.081 | 0.142 |
| 14   | est6  | est    | 0.684        | 1.105      | 0.129 | 0.005 | 0.174 | 0.684 | 0.113 |
| 15   | est1  | est    | 0.668        | 1.119      | 0.178 | 0.053 | 0.116 | 0.668 | 0.105 |
| 16   | est7  | est    | 0.664        | 1.080      | 0.085 | 0.043 | 0.239 | 0.664 | 0.049 |
| 17   | ext2  | ext    | 0.674        | 1.082      | 0.674 | 0.242 | 0.001 | 0.070 | 0.095 |
| 18   | ext1  | ext    | 0.651        | 1.061      | 0.651 | 0.191 | 0.015 | 0.127 | 0.078 |
| 19   | ext9  | ext    | 0.599        | 1.011      | 0.599 | 0.096 | 0.018 | 0.132 | 0.166 |
| 20   | csn8  | csn    | 0.463        | 1.058      | 0.097 | 0.177 | 0.463 | 0.241 | 0.080 |
<!-- END:greedy_ranking -->

The first Intellect item falls just outside the top 20. Extraversion dominates
because its items show moderate cross-domain correlations (visible in the info
score columns above). These cross-loadings inflate composite info scores but are
too weak to reliably predict other domains.

### Mini-IPIP Reference

Donnellan et al. (2006) fixed 20-item mapping to IPIP-BFFM items.

<!-- BEGIN:mini_ipip -->
| Domain              | Items                  | Reported Alpha |
|---------------------|------------------------|----------------|
| Extraversion        | ext1, ext7, ext2, ext4 | 0.77           |
| Agreeableness       | agr4, agr9, agr7, agr5 | 0.70           |
| Conscientiousness   | csn5, csn7, csn6, csn4 | 0.69           |
| Emotional Stability | est8, est6, est2, est4 | 0.68           |
| Intellect/Openness  | opn3, opn2, opn4, opn6 | 0.65           |
<!-- END:mini_ipip -->

The key differences from the domain-balanced set: Mini-IPIP includes weaker
discriminators in Conscientiousness and Emotional Stability (csn7, est2) where
domain-balanced selects higher-correlation items (csn6, est1). Mini-IPIP's
Intellect items (opn3, opn2, opn4, opn6) all rank low by cross-domain info
score; this reflects the psychometric independence of Intellect/Openness from
the other four factors. The greedy ranking table above provides the full
item-level detail.

---

## Extensions

Placeholder sections for analyses under consideration.

### Sparsity Augmentation Ablation

Preliminary results from ablation configs (reference vs focused-only vs
stratified vs no-augmentation). Formalize when ablation artifacts are standardized.

### Per-Quintile Fairness Analysis

Validation data includes quintile-stratified metrics. Verify that accuracy is
uniform across the score distribution (no regression-to-mean bias, no tail
effects).

### Item-Level Diagnostics

Per-item prediction contribution analysis. Which items contribute most to
cross-domain prediction? Does item importance match psychometric discrimination?

### Comparison with IRT-CAT Approaches

Literature comparison with Nieto et al. (2017, 2018) bifactor CAT for Big Five.
How does XGBoost sparse quantile compare to multidimensional IRT-CAT on matched
item pools?

### Cross-Validation Diagnostics

3-fold cross-validation robustness metrics and the leakage fix impact. Document the coverage
improvement from splitting eval before augmentation.
