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
Conscientiousness and Intellect/Imagination entirely (the domain starvation
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
| Domain-balanced r                         | 0.9267 [0.9262, 0.9272] |
| Domain-balanced MAE                       | 8.18 pp                 |
| Domain-balanced 90% coverage              | 89.5%                   |
| Mini-IPIP r                               | 0.9064                  |
| Constrained-adaptive r                    | 0.9052 [0.9046, 0.9058] |
| Adaptive top-K r (greedy)                 | 0.8200 [0.8191, 0.8211] |
| ML vs averaging delta r (domain-balanced) | +0.0165                 |
| ML vs averaging delta r (Mini-IPIP)       | +0.0108                 |
<!-- END:headline_k20 -->

### Cross-Variant Overview (Reference + Ablations)

Auto-generated from `artifacts/research_summary.json`, which aggregates
`models/*/training_report.json` and per-variant evaluation artifacts.

<!-- BEGIN:ablation_overview -->
| Variant                    | Data Regime | Train Val r (full-50) | Validate r (full-50) | Validate r (sparse-20) | Baselines K20 r (domain-balanced) | Simulation r | Complete |
|----------------------------|-------------|-----------------------|----------------------|------------------------|-----------------------------------|--------------|----------|
| Reference                  | ext_est     | 0.9995                | 0.9995               | 0.9095                 | 0.9267                            | 0.9236       | yes      |
| Ablation: No Sparsity      | ext_est     | 1.0000                | 1.0000               | 0.7417                 | 0.7788                            | 0.7718       | yes      |
| Ablation: Focused Only     | ext_est     | 0.9995                | 0.9995               | 0.9094                 | 0.9265                            | 0.9234       | yes      |
| Ablation: Stratified Split | ext_est_opn | 0.9995                | 0.9995               | 0.9099                 | 0.9268                            | 0.9245       | yes      |
<!-- END:ablation_overview -->

### Cross-Variant Provenance Locks

Short-hash provenance view used to confirm all reported numbers are tied to
their exact split and hyperparameter locks.

<!-- BEGIN:ablation_provenance -->
| Variant                    | Split Signature | Train SHA256 | Hyperparams SHA256 | Git Hash     | Errors |
|----------------------------|-----------------|--------------|--------------------|--------------|--------|
| Reference                  | 58fd3f36d435    | cbb2105feeae | 0149fad500ae       | 9575ef602239 | none   |
| Ablation: No Sparsity      | 58fd3f36d435    | cbb2105feeae | 0149fad500ae       | 9575ef602239 | none   |
| Ablation: Focused Only     | 58fd3f36d435    | cbb2105feeae | 0149fad500ae       | 9575ef602239 | none   |
| Ablation: Stratified Split | 3e01ea12dbae    | 961c5de88ad1 | 0149fad500ae       | 9575ef602239 | none   |
<!-- END:ablation_provenance -->

### Cross-Variant Detailed Validation (All Runs)

Full-50 and sparse-20 validation tables for each trained variant.

<!-- BEGIN:ablation_validation_details -->
#### Reference


**Full-50 validation:**


| Domain                | r          | MAE      | RMSE     | Within-5   | 90% Coverage | Raw Crossing Rate |
|-----------------------|------------|----------|----------|------------|--------------|-------------------|
| Extraversion          | 0.9998     | 0.40     | 0.58     | 100.0%     | 92.8%        | 28.8%             |
| Agreeableness         | 0.9996     | 0.60     | 0.85     | 100.0%     | 93.0%        | 21.1%             |
| Conscientiousness     | 0.9995     | 0.67     | 0.92     | 100.0%     | 94.4%        | 21.7%             |
| Emotional Stability   | 0.9997     | 0.54     | 0.76     | 100.0%     | 92.7%        | 26.0%             |
| Intellect/Imagination | 0.9992     | 0.88     | 1.22     | 99.9%      | 93.8%        | 19.7%             |
| **Overall**           | **0.9995** | **0.62** | **0.89** | **100.0%** | ---          | ---               |


**Sparse-20 validation:**


| Domain                | r          | MAE      | RMSE      | Within-5  | 90% Coverage |
|-----------------------|------------|----------|-----------|-----------|--------------|
| Extraversion          | 0.9347     | 7.77     | 10.73     | 46.5%     | 91.0%        |
| Agreeableness         | 0.9071     | 8.94     | 12.25     | 41.3%     | 90.0%        |
| Conscientiousness     | 0.8927     | 9.90     | 13.46     | 38.1%     | 89.8%        |
| Emotional Stability   | 0.9229     | 8.41     | 11.59     | 44.0%     | 90.2%        |
| Intellect/Imagination | 0.8879     | 9.98     | 13.69     | 38.2%     | 89.9%        |
| **Overall**           | **0.9095** | **9.00** | **12.40** | **41.6%** | **90.2%**    |

#### Ablation: No Sparsity


**Full-50 validation:**


| Domain                | r          | MAE      | RMSE     | Within-5   | 90% Coverage | Raw Crossing Rate |
|-----------------------|------------|----------|----------|------------|--------------|-------------------|
| Extraversion          | 1.0000     | 0.14     | 0.20     | 100.0%     | 95.1%        | 38.8%             |
| Agreeableness         | 1.0000     | 0.17     | 0.25     | 100.0%     | 94.9%        | 36.8%             |
| Conscientiousness     | 1.0000     | 0.18     | 0.26     | 100.0%     | 94.9%        | 42.4%             |
| Emotional Stability   | 1.0000     | 0.16     | 0.23     | 100.0%     | 94.8%        | 36.4%             |
| Intellect/Imagination | 1.0000     | 0.17     | 0.26     | 100.0%     | 94.8%        | 38.2%             |
| **Overall**           | **1.0000** | **0.16** | **0.24** | **100.0%** | ---          | ---               |


**Sparse-20 validation:**


| Domain                | r          | MAE       | RMSE      | Within-5  | 90% Coverage |
|-----------------------|------------|-----------|-----------|-----------|--------------|
| Extraversion          | 0.8692     | 39.83     | 46.38     | 9.1%      | 1.0%         |
| Agreeableness         | 0.8360     | 30.56     | 36.41     | 11.9%     | 3.9%         |
| Conscientiousness     | 0.7974     | 38.82     | 45.22     | 8.5%      | 1.5%         |
| Emotional Stability   | 0.8485     | 41.05     | 47.65     | 8.4%      | 1.0%         |
| Intellect/Imagination | 0.8076     | 31.42     | 37.63     | 13.6%     | 5.0%         |
| **Overall**           | **0.7417** | **36.34** | **42.91** | **10.3%** | **2.5%**     |

#### Ablation: Focused Only


**Full-50 validation:**


| Domain                | r          | MAE      | RMSE     | Within-5   | 90% Coverage | Raw Crossing Rate |
|-----------------------|------------|----------|----------|------------|--------------|-------------------|
| Extraversion          | 0.9998     | 0.41     | 0.59     | 100.0%     | 92.6%        | 28.8%             |
| Agreeableness         | 0.9996     | 0.61     | 0.86     | 100.0%     | 93.0%        | 20.8%             |
| Conscientiousness     | 0.9995     | 0.68     | 0.94     | 100.0%     | 94.3%        | 19.9%             |
| Emotional Stability   | 0.9997     | 0.55     | 0.77     | 100.0%     | 92.4%        | 26.0%             |
| Intellect/Imagination | 0.9991     | 0.90     | 1.24     | 99.8%      | 93.5%        | 19.1%             |
| **Overall**           | **0.9995** | **0.63** | **0.91** | **100.0%** | ---          | ---               |


**Sparse-20 validation:**


| Domain                | r          | MAE      | RMSE      | Within-5  | 90% Coverage |
|-----------------------|------------|----------|-----------|-----------|--------------|
| Extraversion          | 0.9348     | 7.76     | 10.71     | 46.6%     | 91.0%        |
| Agreeableness         | 0.9071     | 8.96     | 12.30     | 41.5%     | 90.5%        |
| Conscientiousness     | 0.8927     | 9.91     | 13.50     | 38.3%     | 90.1%        |
| Emotional Stability   | 0.9229     | 8.42     | 11.60     | 43.9%     | 90.4%        |
| Intellect/Imagination | 0.8878     | 10.01    | 13.76     | 38.5%     | 90.2%        |
| **Overall**           | **0.9094** | **9.01** | **12.43** | **41.8%** | **90.4%**    |

#### Ablation: Stratified Split


**Full-50 validation:**


| Domain                | r          | MAE      | RMSE     | Within-5  | 90% Coverage | Raw Crossing Rate |
|-----------------------|------------|----------|----------|-----------|--------------|-------------------|
| Extraversion          | 0.9998     | 0.42     | 0.61     | 100.0%    | 92.2%        | 29.3%             |
| Agreeableness         | 0.9995     | 0.64     | 0.90     | 100.0%    | 93.1%        | 19.8%             |
| Conscientiousness     | 0.9995     | 0.69     | 0.94     | 100.0%    | 94.1%        | 19.6%             |
| Emotional Stability   | 0.9996     | 0.57     | 0.80     | 100.0%    | 92.4%        | 26.2%             |
| Intellect/Imagination | 0.9991     | 0.90     | 1.24     | 99.8%     | 93.7%        | 18.7%             |
| **Overall**           | **0.9995** | **0.64** | **0.92** | **99.9%** | ---          | ---               |


**Sparse-20 validation:**


| Domain                | r          | MAE      | RMSE      | Within-5  | 90% Coverage |
|-----------------------|------------|----------|-----------|-----------|--------------|
| Extraversion          | 0.9342     | 7.80     | 10.76     | 46.5%     | 90.9%        |
| Agreeableness         | 0.9086     | 8.91     | 12.22     | 41.6%     | 90.5%        |
| Conscientiousness     | 0.8932     | 9.90     | 13.49     | 38.2%     | 90.4%        |
| Emotional Stability   | 0.9231     | 8.39     | 11.59     | 44.1%     | 90.7%        |
| Intellect/Imagination | 0.8887     | 10.02    | 13.71     | 38.3%     | 90.2%        |
| **Overall**           | **0.9099** | **9.00** | **12.40** | **41.7%** | **90.5%**    |
<!-- END:ablation_validation_details -->

### Cross-Variant Baseline Curves (All Runs)

Item-selection baseline curves (K=5..50) for each trained variant.

<!-- BEGIN:ablation_baselines_details -->
#### Reference


| K  | Domain-Balanced          | Constrained-Adaptive | Mini-IPIP | First-N              | Random               | Adaptive Top-K       | Greedy-Balanced      | Worst-K              |
|----|--------------------------|----------------------|-----------|----------------------|----------------------|----------------------|----------------------|----------------------|
| 5  | 0.744 [0.743, 0.746]     | 0.686 [0.685, 0.688] | ---       | 0.692 [0.691, 0.694] | 0.599 [0.598, 0.600] | 0.576 [0.574, 0.577] | 0.656 [0.654, 0.658] | 0.461 [0.459, 0.463] |
| 10 | 0.852 [0.851, 0.853]     | 0.791 [0.790, 0.793] | ---       | 0.819 [0.818, 0.820] | 0.762 [0.761, 0.763] | 0.703 [0.702, 0.705] | 0.761 [0.760, 0.763] | 0.640 [0.638, 0.641] |
| 15 | 0.908 [0.907, 0.908]     | 0.845 [0.844, 0.846] | ---       | 0.872 [0.871, 0.873] | 0.844 [0.843, 0.845] | 0.790 [0.789, 0.791] | 0.821 [0.820, 0.823] | 0.756 [0.755, 0.757] |
| 20 | **0.927 [0.926, 0.927]** | 0.905 [0.905, 0.906] | **0.906** | 0.910 [0.909, 0.910] | 0.893 [0.893, 0.894] | 0.820 [0.819, 0.821] | 0.848 [0.847, 0.849] | 0.804 [0.803, 0.805] |
| 25 | 0.945 [0.944, 0.945]     | 0.939 [0.938, 0.939] | ---       | 0.939 [0.938, 0.939] | 0.932 [0.932, 0.933] | 0.899 [0.898, 0.899] | 0.899 [0.898, 0.899] | 0.885 [0.884, 0.886] |
| 30 | 0.960 [0.960, 0.961]     | 0.952 [0.951, 0.952] | ---       | 0.956 [0.956, 0.957] | 0.951 [0.951, 0.952] | 0.928 [0.928, 0.929] | 0.928 [0.928, 0.929] | 0.928 [0.927, 0.929] |
| 40 | 0.983 [0.983, 0.983]     | 0.980 [0.980, 0.980] | ---       | 0.984 [0.984, 0.984] | 0.981 [0.981, 0.981] | 0.968 [0.967, 0.968] | 0.968 [0.967, 0.968] | 0.977 [0.977, 0.978] |
| 50 | 1.000 [1.000, 1.000]     | 1.000 [1.000, 1.000] | ---       | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |

#### Ablation: No Sparsity


| K  | Domain-Balanced          | Constrained-Adaptive | Mini-IPIP | First-N              | Random               | Adaptive Top-K       | Greedy-Balanced      | Worst-K              |
|----|--------------------------|----------------------|-----------|----------------------|----------------------|----------------------|----------------------|----------------------|
| 5  | 0.333 [0.331, 0.335]     | 0.271 [0.268, 0.273] | ---       | 0.285 [0.283, 0.288] | 0.286 [0.284, 0.287] | 0.305 [0.303, 0.307] | 0.237 [0.235, 0.239] | 0.209 [0.206, 0.212] |
| 10 | 0.539 [0.538, 0.541]     | 0.462 [0.460, 0.464] | ---       | 0.516 [0.514, 0.518] | 0.425 [0.424, 0.426] | 0.376 [0.374, 0.378] | 0.425 [0.423, 0.427] | 0.301 [0.299, 0.303] |
| 15 | 0.703 [0.702, 0.704]     | 0.613 [0.612, 0.615] | ---       | 0.647 [0.645, 0.648] | 0.521 [0.520, 0.522] | 0.494 [0.492, 0.495] | 0.553 [0.552, 0.555] | 0.396 [0.394, 0.398] |
| 20 | **0.779 [0.778, 0.780]** | 0.737 [0.736, 0.738] | **0.906** | 0.759 [0.758, 0.760] | 0.646 [0.645, 0.646] | 0.505 [0.503, 0.506] | 0.550 [0.548, 0.552] | 0.413 [0.411, 0.415] |
| 25 | 0.853 [0.852, 0.854]     | 0.826 [0.825, 0.826] | ---       | 0.840 [0.839, 0.841] | 0.700 [0.699, 0.700] | 0.645 [0.643, 0.646] | 0.645 [0.643, 0.646] | 0.503 [0.502, 0.505] |
| 30 | 0.900 [0.899, 0.900]     | 0.881 [0.881, 0.882] | ---       | 0.883 [0.883, 0.884] | 0.795 [0.794, 0.795] | 0.747 [0.745, 0.748] | 0.747 [0.745, 0.748] | 0.588 [0.586, 0.589] |
| 40 | 0.967 [0.967, 0.967]     | 0.961 [0.960, 0.961] | ---       | 0.965 [0.964, 0.965] | 0.912 [0.912, 0.913] | 0.853 [0.852, 0.854] | 0.853 [0.852, 0.854] | 0.762 [0.761, 0.763] |
| 50 | 1.000 [1.000, 1.000]     | 1.000 [1.000, 1.000] | ---       | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |

#### Ablation: Focused Only


| K  | Domain-Balanced          | Constrained-Adaptive | Mini-IPIP | First-N              | Random               | Adaptive Top-K       | Greedy-Balanced      | Worst-K              |
|----|--------------------------|----------------------|-----------|----------------------|----------------------|----------------------|----------------------|----------------------|
| 5  | 0.745 [0.743, 0.746]     | 0.687 [0.685, 0.689] | ---       | 0.693 [0.691, 0.695] | 0.596 [0.595, 0.597] | 0.572 [0.570, 0.573] | 0.656 [0.654, 0.658] | 0.458 [0.456, 0.460] |
| 10 | 0.852 [0.851, 0.853]     | 0.792 [0.790, 0.793] | ---       | 0.819 [0.818, 0.820] | 0.760 [0.760, 0.761] | 0.698 [0.697, 0.700] | 0.760 [0.758, 0.761] | 0.634 [0.632, 0.635] |
| 15 | 0.908 [0.907, 0.909]     | 0.846 [0.845, 0.847] | ---       | 0.872 [0.871, 0.873] | 0.843 [0.842, 0.844] | 0.785 [0.784, 0.786] | 0.819 [0.817, 0.820] | 0.751 [0.750, 0.752] |
| 20 | **0.927 [0.926, 0.927]** | 0.906 [0.905, 0.906] | **0.906** | 0.909 [0.909, 0.910] | 0.893 [0.892, 0.893] | 0.816 [0.815, 0.817] | 0.846 [0.845, 0.847] | 0.798 [0.797, 0.799] |
| 25 | 0.944 [0.944, 0.945]     | 0.939 [0.938, 0.939] | ---       | 0.938 [0.938, 0.939] | 0.932 [0.932, 0.932] | 0.897 [0.897, 0.898] | 0.897 [0.897, 0.898] | 0.881 [0.880, 0.881] |
| 30 | 0.960 [0.960, 0.960]     | 0.952 [0.951, 0.952] | ---       | 0.956 [0.956, 0.957] | 0.951 [0.951, 0.952] | 0.928 [0.928, 0.929] | 0.928 [0.928, 0.929] | 0.926 [0.925, 0.926] |
| 40 | 0.983 [0.983, 0.983]     | 0.980 [0.980, 0.980] | ---       | 0.984 [0.984, 0.984] | 0.981 [0.981, 0.981] | 0.968 [0.967, 0.968] | 0.968 [0.967, 0.968] | 0.977 [0.977, 0.978] |
| 50 | 1.000 [1.000, 1.000]     | 1.000 [1.000, 1.000] | ---       | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |

#### Ablation: Stratified Split


| K  | Domain-Balanced          | Constrained-Adaptive | Mini-IPIP | First-N              | Random               | Adaptive Top-K       | Greedy-Balanced      | Worst-K              |
|----|--------------------------|----------------------|-----------|----------------------|----------------------|----------------------|----------------------|----------------------|
| 5  | 0.744 [0.743, 0.746]     | 0.690 [0.688, 0.692] | ---       | 0.694 [0.692, 0.695] | 0.596 [0.595, 0.597] | 0.572 [0.570, 0.573] | 0.659 [0.658, 0.661] | 0.460 [0.458, 0.462] |
| 10 | 0.852 [0.851, 0.853]     | 0.793 [0.791, 0.794] | ---       | 0.819 [0.818, 0.820] | 0.761 [0.760, 0.762] | 0.700 [0.698, 0.701] | 0.727 [0.725, 0.728] | 0.636 [0.635, 0.638] |
| 15 | 0.908 [0.908, 0.909]     | 0.846 [0.845, 0.847] | ---       | 0.873 [0.872, 0.873] | 0.844 [0.843, 0.844] | 0.786 [0.785, 0.787] | 0.814 [0.813, 0.815] | 0.753 [0.752, 0.754] |
| 20 | **0.927 [0.926, 0.927]** | 0.906 [0.905, 0.906] | **0.907** | 0.910 [0.909, 0.910] | 0.893 [0.893, 0.894] | 0.817 [0.816, 0.818] | 0.847 [0.846, 0.848] | 0.800 [0.799, 0.801] |
| 25 | 0.944 [0.944, 0.945]     | 0.939 [0.938, 0.939] | ---       | 0.938 [0.938, 0.939] | 0.932 [0.932, 0.933] | 0.897 [0.897, 0.898] | 0.897 [0.897, 0.898] | 0.881 [0.880, 0.882] |
| 30 | 0.960 [0.960, 0.960]     | 0.952 [0.951, 0.952] | ---       | 0.956 [0.956, 0.957] | 0.951 [0.951, 0.952] | 0.928 [0.928, 0.929] | 0.928 [0.928, 0.929] | 0.925 [0.925, 0.926] |
| 40 | 0.983 [0.983, 0.983]     | 0.981 [0.981, 0.981] | ---       | 0.984 [0.984, 0.984] | 0.981 [0.981, 0.982] | 0.968 [0.967, 0.968] | 0.968 [0.967, 0.968] | 0.978 [0.977, 0.978] |
| 50 | 1.000 [1.000, 1.000]     | 1.000 [1.000, 1.000] | ---       | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
<!-- END:ablation_baselines_details -->

### Cross-Variant Per-Domain K=20 (All Runs)

Per-domain K=20 breakdown (domain-balanced, Mini-IPIP, first-N) for each variant.

<!-- BEGIN:ablation_per_domain_k20_details -->
#### Reference


**Domain-Balanced (4 items per domain):**

| Domain                | r      | Items | CI Lower | CI Upper |
|-----------------------|--------|-------|----------|----------|
| Extraversion          | 0.9465 | 4     | 0.9457   | 0.9473   |
| Agreeableness         | 0.9196 | 4     | 0.9184   | 0.9207   |
| Conscientiousness     | 0.9192 | 4     | 0.9181   | 0.9203   |
| Emotional Stability   | 0.9366 | 4     | 0.9357   | 0.9375   |
| Intellect/Imagination | 0.9100 | 4     | 0.9087   | 0.9114   |

**Mini-IPIP (4 items per domain):**

| Domain                | r      | Items | CI Lower | CI Upper |
|-----------------------|--------|-------|----------|----------|
| Extraversion          | 0.9385 | 4     | ---      | ---      |
| Agreeableness         | 0.9113 | 4     | ---      | ---      |
| Conscientiousness     | 0.9086 | 4     | ---      | ---      |
| Emotional Stability   | 0.9292 | 4     | ---      | ---      |
| Intellect/Imagination | 0.8419 | 4     | ---      | ---      |

**First-N (4 items per domain):**

| Domain                | r      | Items | CI Lower | CI Upper |
|-----------------------|--------|-------|----------|----------|
| Extraversion          | 0.9392 | 4     | 0.9383   | 0.9401   |
| Agreeableness         | 0.9264 | 4     | 0.9253   | 0.9275   |
| Conscientiousness     | 0.8943 | 4     | 0.8929   | 0.8958   |
| Emotional Stability   | 0.8768 | 4     | 0.8752   | 0.8786   |
| Intellect/Imagination | 0.9115 | 4     | 0.9102   | 0.9126   |

#### Ablation: No Sparsity


**Domain-Balanced (4 items per domain):**

| Domain                | r      | Items | CI Lower | CI Upper |
|-----------------------|--------|-------|----------|----------|
| Extraversion          | 0.8921 | 4     | 0.8911   | 0.8932   |
| Agreeableness         | 0.8462 | 4     | 0.8448   | 0.8477   |
| Conscientiousness     | 0.8614 | 4     | 0.8601   | 0.8627   |
| Emotional Stability   | 0.8616 | 4     | 0.8604   | 0.8628   |
| Intellect/Imagination | 0.8444 | 4     | 0.8427   | 0.8460   |

**Mini-IPIP (4 items per domain):**

| Domain                | r      | Items | CI Lower | CI Upper |
|-----------------------|--------|-------|----------|----------|
| Extraversion          | 0.9385 | 4     | ---      | ---      |
| Agreeableness         | 0.9113 | 4     | ---      | ---      |
| Conscientiousness     | 0.9086 | 4     | ---      | ---      |
| Emotional Stability   | 0.9292 | 4     | ---      | ---      |
| Intellect/Imagination | 0.8419 | 4     | ---      | ---      |

**First-N (4 items per domain):**

| Domain                | r      | Items | CI Lower | CI Upper |
|-----------------------|--------|-------|----------|----------|
| Extraversion          | 0.8867 | 4     | 0.8857   | 0.8878   |
| Agreeableness         | 0.8406 | 4     | 0.8391   | 0.8423   |
| Conscientiousness     | 0.8168 | 4     | 0.8150   | 0.8186   |
| Emotional Stability   | 0.8190 | 4     | 0.8170   | 0.8211   |
| Intellect/Imagination | 0.8331 | 4     | 0.8314   | 0.8347   |

#### Ablation: Focused Only


**Domain-Balanced (4 items per domain):**

| Domain                | r      | Items | CI Lower | CI Upper |
|-----------------------|--------|-------|----------|----------|
| Extraversion          | 0.9464 | 4     | 0.9456   | 0.9472   |
| Agreeableness         | 0.9196 | 4     | 0.9185   | 0.9208   |
| Conscientiousness     | 0.9188 | 4     | 0.9177   | 0.9199   |
| Emotional Stability   | 0.9363 | 4     | 0.9354   | 0.9372   |
| Intellect/Imagination | 0.9101 | 4     | 0.9088   | 0.9115   |

**Mini-IPIP (4 items per domain):**

| Domain                | r      | Items | CI Lower | CI Upper |
|-----------------------|--------|-------|----------|----------|
| Extraversion          | 0.9385 | 4     | ---      | ---      |
| Agreeableness         | 0.9113 | 4     | ---      | ---      |
| Conscientiousness     | 0.9086 | 4     | ---      | ---      |
| Emotional Stability   | 0.9292 | 4     | ---      | ---      |
| Intellect/Imagination | 0.8419 | 4     | ---      | ---      |

**First-N (4 items per domain):**

| Domain                | r      | Items | CI Lower | CI Upper |
|-----------------------|--------|-------|----------|----------|
| Extraversion          | 0.9393 | 4     | 0.9383   | 0.9402   |
| Agreeableness         | 0.9258 | 4     | 0.9248   | 0.9270   |
| Conscientiousness     | 0.8944 | 4     | 0.8930   | 0.8959   |
| Emotional Stability   | 0.8764 | 4     | 0.8747   | 0.8782   |
| Intellect/Imagination | 0.9106 | 4     | 0.9094   | 0.9118   |

#### Ablation: Stratified Split


**Domain-Balanced (4 items per domain):**

| Domain                | r      | Items | CI Lower | CI Upper |
|-----------------------|--------|-------|----------|----------|
| Extraversion          | 0.9463 | 4     | 0.9455   | 0.9471   |
| Agreeableness         | 0.9206 | 4     | 0.9195   | 0.9218   |
| Conscientiousness     | 0.9197 | 4     | 0.9185   | 0.9208   |
| Emotional Stability   | 0.9367 | 4     | 0.9358   | 0.9376   |
| Intellect/Imagination | 0.9094 | 4     | 0.9080   | 0.9107   |

**Mini-IPIP (4 items per domain):**

| Domain                | r      | Items | CI Lower | CI Upper |
|-----------------------|--------|-------|----------|----------|
| Extraversion          | 0.9377 | 4     | ---      | ---      |
| Agreeableness         | 0.9126 | 4     | ---      | ---      |
| Conscientiousness     | 0.9092 | 4     | ---      | ---      |
| Emotional Stability   | 0.9294 | 4     | ---      | ---      |
| Intellect/Imagination | 0.8418 | 4     | ---      | ---      |

**First-N (4 items per domain):**

| Domain                | r      | Items | CI Lower | CI Upper |
|-----------------------|--------|-------|----------|----------|
| Extraversion          | 0.9398 | 4     | 0.9389   | 0.9408   |
| Agreeableness         | 0.9254 | 4     | 0.9243   | 0.9265   |
| Conscientiousness     | 0.8956 | 4     | 0.8940   | 0.8971   |
| Emotional Stability   | 0.8763 | 4     | 0.8747   | 0.8780   |
| Intellect/Imagination | 0.9110 | 4     | 0.9097   | 0.9122   |
<!-- END:ablation_per_domain_k20_details -->

### Cross-Variant Domain Starvation at K=20 (All Runs)

Adaptive top-K domain allocation and resulting per-domain accuracy for each variant.

<!-- BEGIN:ablation_domain_starvation_details -->
#### Reference


| Domain                | Items | Share | r      | CI Lower | CI Upper |
|-----------------------|-------|-------|--------|----------|----------|
| Extraversion          | 9     | 45%   | 0.9939 | 0.9938   | 0.9940   |
| Emotional Stability   | 6     | 30%   | 0.9703 | 0.9699   | 0.9707   |
| Agreeableness         | 3     | 15%   | 0.8218 | 0.8194   | 0.8243   |
| Conscientiousness     | 2     | 10%   | 0.7562 | 0.7529   | 0.7592   |
| Intellect/Imagination | 0     | 0%    | 0.4109 | 0.4054   | 0.4165   |

#### Ablation: No Sparsity


| Domain                | Items | Share | r       | CI Lower | CI Upper |
|-----------------------|-------|-------|---------|----------|----------|
| Extraversion          | 9     | 45%   | 0.9886  | 0.9885   | 0.9887   |
| Emotional Stability   | 6     | 30%   | 0.9095  | 0.9088   | 0.9103   |
| Agreeableness         | 3     | 15%   | 0.7552  | 0.7528   | 0.7578   |
| Conscientiousness     | 2     | 10%   | 0.6980  | 0.6947   | 0.7013   |
| Intellect/Imagination | 0     | 0%    | -0.0434 | -0.0498  | -0.0366  |

#### Ablation: Focused Only


| Domain                | Items | Share | r      | CI Lower | CI Upper |
|-----------------------|-------|-------|--------|----------|----------|
| Extraversion          | 9     | 45%   | 0.9938 | 0.9937   | 0.9939   |
| Emotional Stability   | 6     | 30%   | 0.9695 | 0.9691   | 0.9699   |
| Agreeableness         | 3     | 15%   | 0.8185 | 0.8161   | 0.8209   |
| Conscientiousness     | 2     | 10%   | 0.7543 | 0.7511   | 0.7573   |
| Intellect/Imagination | 0     | 0%    | 0.4023 | 0.3967   | 0.4077   |

#### Ablation: Stratified Split


| Domain                | Items | Share | r      | CI Lower | CI Upper |
|-----------------------|-------|-------|--------|----------|----------|
| Extraversion          | 9     | 45%   | 0.9938 | 0.9937   | 0.9939   |
| Emotional Stability   | 6     | 30%   | 0.9693 | 0.9688   | 0.9697   |
| Agreeableness         | 3     | 15%   | 0.8215 | 0.8192   | 0.8239   |
| Conscientiousness     | 2     | 10%   | 0.7578 | 0.7548   | 0.7611   |
| Intellect/Imagination | 0     | 0%    | 0.4081 | 0.4023   | 0.4137   |
<!-- END:ablation_domain_starvation_details -->

### Cross-Variant ML vs Averaging (All Runs)

ML-vs-averaging deltas for matched item sets across each variant.

<!-- BEGIN:ablation_ml_vs_averaging_details -->
#### Reference


| Strategy        | K  | ML r   | Avg r  | Delta r | ML MAE | Avg MAE | Delta MAE |
|-----------------|----|--------|--------|---------|--------|---------|-----------|
| Domain-balanced | 10 | 0.8518 | 0.8293 | +0.0225 | 11.74  | 14.27   | -2.53     |
| Domain-balanced | 15 | 0.9079 | 0.8862 | +0.0217 | 9.16   | 11.27   | -2.10     |
| Domain-balanced | 20 | 0.9267 | 0.9102 | +0.0165 | 8.18   | 9.85    | -1.67     |
| Mini-IPIP       | 20 | 0.9172 | 0.9064 | +0.0108 | 8.60   | 9.17    | -0.57     |
| Domain-balanced | 25 | 0.9445 | 0.9302 | +0.0144 | 7.16   | 8.54    | -1.38     |

#### Ablation: No Sparsity


| Strategy        | K  | ML r   | Avg r  | Delta r | ML MAE | Avg MAE | Delta MAE |
|-----------------|----|--------|--------|---------|--------|---------|-----------|
| Domain-balanced | 10 | 0.5390 | 0.8293 | -0.2903 | 43.69  | 14.27   | +29.42    |
| Domain-balanced | 15 | 0.7033 | 0.8862 | -0.1829 | 40.11  | 11.27   | +28.84    |
| Domain-balanced | 20 | 0.7788 | 0.9102 | -0.1314 | 35.70  | 9.85    | +25.85    |
| Mini-IPIP       | 20 | 0.7556 | 0.9064 | -0.1508 | 36.40  | 9.17    | +27.23    |
| Domain-balanced | 25 | 0.8531 | 0.9302 | -0.0771 | 30.23  | 8.54    | +21.69    |

#### Ablation: Focused Only


| Strategy        | K  | ML r   | Avg r  | Delta r | ML MAE | Avg MAE | Delta MAE |
|-----------------|----|--------|--------|---------|--------|---------|-----------|
| Domain-balanced | 10 | 0.8518 | 0.8293 | +0.0225 | 11.76  | 14.27   | -2.51     |
| Domain-balanced | 15 | 0.9080 | 0.8862 | +0.0218 | 9.16   | 11.27   | -2.11     |
| Domain-balanced | 20 | 0.9265 | 0.9102 | +0.0163 | 8.19   | 9.85    | -1.66     |
| Mini-IPIP       | 20 | 0.9172 | 0.9064 | +0.0108 | 8.60   | 9.17    | -0.57     |
| Domain-balanced | 25 | 0.9443 | 0.9302 | +0.0141 | 7.19   | 8.54    | -1.35     |

#### Ablation: Stratified Split


| Strategy        | K  | ML r   | Avg r  | Delta r | ML MAE | Avg MAE | Delta MAE |
|-----------------|----|--------|--------|---------|--------|---------|-----------|
| Domain-balanced | 10 | 0.8521 | 0.8295 | +0.0226 | 11.74  | 14.27   | -2.52     |
| Domain-balanced | 15 | 0.9082 | 0.8863 | +0.0219 | 9.16   | 11.28   | -2.12     |
| Domain-balanced | 20 | 0.9268 | 0.9103 | +0.0164 | 8.19   | 9.85    | -1.66     |
| Mini-IPIP       | 20 | 0.9175 | 0.9067 | +0.0108 | 8.60   | 9.17    | -0.57     |
| Domain-balanced | 25 | 0.9444 | 0.9301 | +0.0142 | 7.20   | 8.55    | -1.36     |
<!-- END:ablation_ml_vs_averaging_details -->

### Cross-Variant Simulation Results (All Runs)

Adaptive simulation outcomes for each variant at the operating point.

<!-- BEGIN:ablation_simulation_details -->
#### Reference


| Domain                | r          | MAE      | RMSE      | Within-5  | 90% Coverage |
|-----------------------|------------|----------|-----------|-----------|--------------|
| Extraversion          | 0.9347     | 7.78     | 10.67     | 46.2%     | 91.1%        |
| Agreeableness         | 0.9176     | 8.41     | 11.53     | 43.4%     | 88.6%        |
| Conscientiousness     | 0.9169     | 8.78     | 11.85     | 41.5%     | 89.6%        |
| Emotional Stability   | 0.9379     | 7.72     | 10.45     | 45.3%     | 88.9%        |
| Intellect/Imagination | 0.9099     | 8.97     | 12.43     | 42.8%     | 90.1%        |
| **Overall**           | **0.9236** | **8.33** | **11.41** | **43.8%** | **89.7%**    |

#### Ablation: No Sparsity


| Domain                | r          | MAE       | RMSE      | Within-5 | 90% Coverage |
|-----------------------|------------|-----------|-----------|----------|--------------|
| Extraversion          | 0.8773     | 40.16     | 46.66     | 8.7%     | 0.7%         |
| Agreeableness         | 0.8460     | 30.84     | 36.17     | 10.0%    | 2.8%         |
| Conscientiousness     | 0.8597     | 36.61     | 42.19     | 8.3%     | 1.5%         |
| Emotional Stability   | 0.8618     | 41.46     | 47.53     | 6.8%     | 0.5%         |
| Intellect/Imagination | 0.8422     | 30.78     | 36.50     | 12.7%    | 3.9%         |
| **Overall**           | **0.7718** | **35.97** | **42.09** | **9.3%** | **1.9%**     |

#### Ablation: Focused Only


| Domain                | r          | MAE      | RMSE      | Within-5  | 90% Coverage |
|-----------------------|------------|----------|-----------|-----------|--------------|
| Extraversion          | 0.9346     | 7.79     | 10.67     | 46.1%     | 91.6%        |
| Agreeableness         | 0.9176     | 8.42     | 11.50     | 42.9%     | 89.1%        |
| Conscientiousness     | 0.9167     | 8.79     | 11.87     | 41.5%     | 89.3%        |
| Emotional Stability   | 0.9373     | 7.75     | 10.49     | 45.6%     | 89.9%        |
| Intellect/Imagination | 0.9102     | 8.97     | 12.45     | 42.5%     | 90.9%        |
| **Overall**           | **0.9234** | **8.34** | **11.42** | **43.7%** | **90.2%**    |

#### Ablation: Stratified Split


| Domain                | r          | MAE      | RMSE      | Within-5  | 90% Coverage |
|-----------------------|------------|----------|-----------|-----------|--------------|
| Extraversion          | 0.9364     | 7.68     | 10.53     | 46.4%     | 91.7%        |
| Agreeableness         | 0.9176     | 8.51     | 11.60     | 42.3%     | 88.6%        |
| Conscientiousness     | 0.9211     | 8.57     | 11.69     | 42.7%     | 89.5%        |
| Emotional Stability   | 0.9373     | 7.72     | 10.53     | 47.0%     | 89.7%        |
| Intellect/Imagination | 0.9095     | 9.06     | 12.38     | 41.2%     | 90.2%        |
| **Overall**           | **0.9245** | **8.31** | **11.37** | **43.9%** | **90.0%**    |
<!-- END:ablation_simulation_details -->

### Dataset

Valid respondents from the Open-Source Psychometrics Project, stratified
split into train/validation/test.

<!-- BEGIN:data_splits -->
| Split       | Respondents | Fraction |
|-------------|-------------|----------|
| Total valid | 603,322     | 100%     |
| Train       | 422,324     | 70.0%    |
| Validation  | 90,499      | 15.0%    |
| Test        | 90,499      | 15.0%    |

Stratification: ext_q * 5 + est_q (25 strata) (25 strata, seed=42).

| Domain                | Max Mean Diff | KS Statistic | KS p-value |
|-----------------------|---------------|--------------|------------|
| Extraversion          | 0.0031        | 0.0015       | 0.995      |
| Agreeableness         | 0.0010        | 0.0021       | 0.889      |
| Conscientiousness     | 0.0047        | 0.0024       | 0.802      |
| Emotional Stability   | 0.0011        | 0.0021       | 0.897      |
| Intellect/Imagination | 0.0020        | 0.0019       | 0.950      |
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
| CV folds                    | 3         |
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
| n_estimators     | 9,092                                       |
| max_depth        | 4                                           |
| learning_rate    | 0.0118                                      |
| min_child_weight | 7                                           |
| subsample        | 0.753                                       |
| colsample_bytree | 0.548                                       |
| reg_lambda       | 2.124                                       |
| reg_alpha        | 4.717                                       |
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
| Domain                | Mean   | SD     |
|-----------------------|--------|--------|
| Extraversion          | 2.9138 | 0.9112 |
| Agreeableness         | 3.7589 | 0.7363 |
| Conscientiousness     | 3.3426 | 0.7391 |
| Emotional Stability   | 2.9187 | 0.8607 |
| Intellect/Imagination | 3.9389 | 0.6183 |
<!-- END:norms -->

### Ceiling Check: Full-Model Accuracy (50 Items, 90K Test)

Sanity check with all 50 items present. Near-perfect reconstruction is
expected; these numbers confirm the model and inference pipeline work
correctly, not the operating-point accuracy. See the K=20 tables below for
the primary evaluation.

<!-- BEGIN:validation -->
| Domain                | r          | MAE      | RMSE     | Within-5   | 90% Coverage | Raw Crossing Rate |
|-----------------------|------------|----------|----------|------------|--------------|-------------------|
| Extraversion          | 0.9998     | 0.40     | 0.58     | 100.0%     | 92.8%        | 28.8%             |
| Agreeableness         | 0.9996     | 0.60     | 0.85     | 100.0%     | 93.0%        | 21.1%             |
| Conscientiousness     | 0.9995     | 0.67     | 0.92     | 100.0%     | 94.4%        | 21.7%             |
| Emotional Stability   | 0.9997     | 0.54     | 0.76     | 100.0%     | 92.7%        | 26.0%             |
| Intellect/Imagination | 0.9992     | 0.88     | 1.22     | 99.9%      | 93.8%        | 19.7%             |
| **Overall**           | **0.9995** | **0.62** | **0.89** | **100.0%** | ---          | ---               |
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
| Extraversion                       | Q1       | 20,313 | 0.21     | 89.6%        | 1.05          |
| Extraversion                       | Q2       | 16,357 | 0.52     | 94.3%        | 2.81          |
| Extraversion                       | Q3       | 20,808 | 0.61     | 95.6%        | 3.53          |
| Extraversion                       | Q4       | 17,385 | 0.46     | 93.4%        | 2.53          |
| Extraversion                       | Q5       | 15,636 | 0.18     | 91.2%        | 0.91          |
| Agreeableness                      | Q1       | 20,433 | 0.52     | 90.7%        | 2.55          |
| Agreeableness                      | Q2       | 19,851 | 0.85     | 96.5%        | 5.60          |
| Agreeableness                      | Q3       | 15,271 | 0.76     | 96.3%        | 5.57          |
| Agreeableness                      | Q4       | 18,806 | 0.59     | 95.1%        | 4.07          |
| Agreeableness                      | Q5       | 16,138 | 0.27     | 86.0%        | 1.53          |
| Conscientiousness                  | Q1       | 20,072 | 0.46     | 90.3%        | 2.34          |
| Conscientiousness                  | Q2       | 16,336 | 0.85     | 96.6%        | 5.14          |
| Conscientiousness                  | Q3       | 18,112 | 0.94     | 98.4%        | 6.22          |
| Conscientiousness                  | Q4       | 19,617 | 0.77     | 96.9%        | 4.99          |
| Conscientiousness                  | Q5       | 16,362 | 0.35     | 89.9%        | 1.87          |
| Emotional Stability                | Q1       | 18,485 | 0.28     | 88.6%        | 1.32          |
| Emotional Stability                | Q2       | 21,064 | 0.65     | 94.6%        | 3.62          |
| Emotional Stability                | Q3       | 15,043 | 0.80     | 96.1%        | 4.68          |
| Emotional Stability                | Q4       | 19,338 | 0.67     | 94.5%        | 3.68          |
| Emotional Stability                | Q5       | 16,569 | 0.29     | 89.6%        | 1.41          |
| Intellect/Imagination              | Q1       | 19,317 | 0.71     | 89.5%        | 3.05          |
| Intellect/Imagination              | Q2       | 18,472 | 1.25     | 96.5%        | 6.78          |
| Intellect/Imagination              | Q3       | 16,569 | 1.15     | 98.8%        | 7.76          |
| Intellect/Imagination              | Q4       | 20,307 | 0.89     | 98.5%        | 5.88          |
| Intellect/Imagination              | Q5       | 15,834 | 0.38     | 84.4%        | 1.84          |
| **Avg Tails (Q1/Q5, all domains)** | ---      | ---    | **0.36** | **89.0%**    | **1.79**      |
| **Avg Center (Q3, all domains)**   | ---      | ---    | **0.86** | **97.0%**    | **5.55**      |
<!-- END:validation_quintiles -->

### Item Selection Strategies (5-50 Items, 90K Test)

Pearson r with full 50-item scores. Bootstrap 95% CIs from 1,000 resamples.

<!-- BEGIN:baselines -->
| K  | Domain-Balanced          | Constrained-Adaptive | Mini-IPIP | First-N              | Random               | Adaptive Top-K       | Greedy-Balanced      | Worst-K              |
|----|--------------------------|----------------------|-----------|----------------------|----------------------|----------------------|----------------------|----------------------|
| 5  | 0.744 [0.743, 0.746]     | 0.686 [0.685, 0.688] | ---       | 0.692 [0.691, 0.694] | 0.599 [0.598, 0.600] | 0.576 [0.574, 0.577] | 0.656 [0.654, 0.658] | 0.461 [0.459, 0.463] |
| 10 | 0.852 [0.851, 0.853]     | 0.791 [0.790, 0.793] | ---       | 0.819 [0.818, 0.820] | 0.762 [0.761, 0.763] | 0.703 [0.702, 0.705] | 0.761 [0.760, 0.763] | 0.640 [0.638, 0.641] |
| 15 | 0.908 [0.907, 0.908]     | 0.845 [0.844, 0.846] | ---       | 0.872 [0.871, 0.873] | 0.844 [0.843, 0.845] | 0.790 [0.789, 0.791] | 0.821 [0.820, 0.823] | 0.756 [0.755, 0.757] |
| 20 | **0.927 [0.926, 0.927]** | 0.905 [0.905, 0.906] | **0.906** | 0.910 [0.909, 0.910] | 0.893 [0.893, 0.894] | 0.820 [0.819, 0.821] | 0.848 [0.847, 0.849] | 0.804 [0.803, 0.805] |
| 25 | 0.945 [0.944, 0.945]     | 0.939 [0.938, 0.939] | ---       | 0.939 [0.938, 0.939] | 0.932 [0.932, 0.933] | 0.899 [0.898, 0.899] | 0.899 [0.898, 0.899] | 0.885 [0.884, 0.886] |
| 30 | 0.960 [0.960, 0.961]     | 0.952 [0.951, 0.952] | ---       | 0.956 [0.956, 0.957] | 0.951 [0.951, 0.952] | 0.928 [0.928, 0.929] | 0.928 [0.928, 0.929] | 0.928 [0.927, 0.929] |
| 40 | 0.983 [0.983, 0.983]     | 0.980 [0.980, 0.980] | ---       | 0.984 [0.984, 0.984] | 0.981 [0.981, 0.981] | 0.968 [0.967, 0.968] | 0.968 [0.967, 0.968] | 0.977 [0.977, 0.978] |
| 50 | 1.000 [1.000, 1.000]     | 1.000 [1.000, 1.000] | ---       | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
<!-- END:baselines -->

At K>=25, greedy-balanced and adaptive top-K converge (identical item sets once the
greedy tail dominates). At K=40, greedy approaches nearly match the full-scale
ceiling because most items are included regardless of selection strategy. The
paradox is about the operating range of interest (10-25 items), not item selection
in general.

### Per-Domain Breakdown at K=20

<!-- BEGIN:per_domain_k20 -->
**Domain-Balanced (4 items per domain):**

| Domain                | r      | Items | CI Lower | CI Upper |
|-----------------------|--------|-------|----------|----------|
| Extraversion          | 0.9465 | 4     | 0.9457   | 0.9473   |
| Agreeableness         | 0.9196 | 4     | 0.9184   | 0.9207   |
| Conscientiousness     | 0.9192 | 4     | 0.9181   | 0.9203   |
| Emotional Stability   | 0.9366 | 4     | 0.9357   | 0.9375   |
| Intellect/Imagination | 0.9100 | 4     | 0.9087   | 0.9114   |

**Mini-IPIP (4 items per domain):**

| Domain                | r      | Items | CI Lower | CI Upper |
|-----------------------|--------|-------|----------|----------|
| Extraversion          | 0.9385 | 4     | ---      | ---      |
| Agreeableness         | 0.9113 | 4     | ---      | ---      |
| Conscientiousness     | 0.9086 | 4     | ---      | ---      |
| Emotional Stability   | 0.9292 | 4     | ---      | ---      |
| Intellect/Imagination | 0.8419 | 4     | ---      | ---      |

**First-N (4 items per domain):**

| Domain                | r      | Items | CI Lower | CI Upper |
|-----------------------|--------|-------|----------|----------|
| Extraversion          | 0.9392 | 4     | 0.9383   | 0.9401   |
| Agreeableness         | 0.9264 | 4     | 0.9253   | 0.9275   |
| Conscientiousness     | 0.8943 | 4     | 0.8929   | 0.8958   |
| Emotional Stability   | 0.8768 | 4     | 0.8752   | 0.8786   |
| Intellect/Imagination | 0.9115 | 4     | 0.9102   | 0.9126   |
<!-- END:per_domain_k20 -->

Domain-balanced achieves the most uniform per-domain performance. Mini-IPIP shows
a clear Intellect/Imagination gap relative to domain-balanced (visible in the table
above) because its Intellect items were selected for brevity, not discrimination.

### Domain Starvation (Greedy Selection at K=20)

<!-- BEGIN:domain_starvation -->
| Domain                | Items | Share | r      | CI Lower | CI Upper |
|-----------------------|-------|-------|--------|----------|----------|
| Extraversion          | 9     | 45%   | 0.9939 | 0.9938   | 0.9940   |
| Emotional Stability   | 6     | 30%   | 0.9703 | 0.9699   | 0.9707   |
| Agreeableness         | 3     | 15%   | 0.8218 | 0.8194   | 0.8243   |
| Conscientiousness     | 2     | 10%   | 0.7562 | 0.7529   | 0.7592   |
| Intellect/Imagination | 0     | 0%    | 0.4109 | 0.4054   | 0.4165   |
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
| Domain-balanced | 10 | 0.8518 | 0.8293 | +0.0225 | 11.74  | 14.27   | -2.53     |
| Domain-balanced | 15 | 0.9079 | 0.8862 | +0.0217 | 9.16   | 11.27   | -2.10     |
| Domain-balanced | 20 | 0.9267 | 0.9102 | +0.0165 | 8.18   | 9.85    | -1.67     |
| Mini-IPIP       | 20 | 0.9172 | 0.9064 | +0.0108 | 8.60   | 9.17    | -0.57     |
| Domain-balanced | 25 | 0.9445 | 0.9302 | +0.0144 | 7.16   | 8.54    | -1.38     |
<!-- END:ml_vs_averaging -->

The ML advantage holds across all tested item counts, as the table shows, and is
largest at fewer items where cross-domain information sharing matters most. Both
correlation and MAE improvements are consistent across budgets.

### Simulation Results (20-Item Operating Point)

Held-out respondents from the test split, correlation-ranked selection,
SEM threshold 0.45, min 4 items per domain. All respondents converge to
exactly 20 items (4-4-4-4-4).

<!-- BEGIN:simulation -->
| Domain                | r          | MAE      | RMSE      | Within-5  | 90% Coverage |
|-----------------------|------------|----------|-----------|-----------|--------------|
| Extraversion          | 0.9347     | 7.78     | 10.67     | 46.2%     | 91.1%        |
| Agreeableness         | 0.9176     | 8.41     | 11.53     | 43.4%     | 88.6%        |
| Conscientiousness     | 0.9169     | 8.78     | 11.85     | 41.5%     | 89.6%        |
| Emotional Stability   | 0.9379     | 7.72     | 10.45     | 45.3%     | 88.9%        |
| Intellect/Imagination | 0.9099     | 8.97     | 12.43     | 42.8%     | 90.1%        |
| **Overall**           | **0.9236** | **8.33** | **11.41** | **43.8%** | **89.7%**    |
<!-- END:simulation -->

Simulation results closely match the static baseline evaluation (compare the
overall *r* here to the headline results table). Correlation-ranked selection is
equivalent to the optimal static strategy; the agreement between the two also
confirms the pipeline end-to-end.

### Calibration

<!-- BEGIN:calibration -->
Calibration regime: `sparse_20_balanced` (domain-balanced, 20 items, 4 per domain).

| Domain                | Observed Coverage | Scale Factor |
|-----------------------|-------------------|--------------|
| Extraversion          | 91.0%             | 1.0          |
| Agreeableness         | 90.0%             | 1.0          |
| Conscientiousness     | 89.9%             | 1.0          |
| Emotional Stability   | 90.3%             | 1.0          |
| Intellect/Imagination | 89.7%             | 1.0          |
<!-- END:calibration -->

All domains achieve near-nominal 90% coverage without requiring any scaling
adjustment at the explicitly supported calibration regimes. In this project,
the strongest calibration claim is for full 50-item completion and the primary
20-item domain-balanced operating point.

### Calibration Policy

When each calibration regime is applied, based on item count. Current exported
runtime uses `full_50` for complete responses and falls back to
`sparse_20_balanced` for sub-50 response patterns; that fallback supports point
prediction broadly, but the strongest calibration claim remains the primary
20-item domain-balanced regime.

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
| Rank | Item  | Domain                | Own-Domain r | Reverse-Keyed |
|------|-------|-----------------------|--------------|---------------|
| 1    | ext4  | Extraversion          | 0.718        | yes           |
| 2    | ext5  | Extraversion          | 0.707        | no            |
| 3    | ext7  | Extraversion          | 0.689        | no            |
| 4    | ext2  | Extraversion          | 0.675        | yes           |
| 5    | agr4  | Agreeableness         | 0.717        | no            |
| 6    | agr9  | Agreeableness         | 0.639        | no            |
| 7    | agr7  | Agreeableness         | 0.627        | yes           |
| 8    | agr5  | Agreeableness         | 0.626        | yes           |
| 9    | csn6  | Conscientiousness     | 0.590        | yes           |
| 10   | csn1  | Conscientiousness     | 0.576        | no            |
| 11   | csn5  | Conscientiousness     | 0.570        | no            |
| 12   | csn4  | Conscientiousness     | 0.561        | yes           |
| 13   | est8  | Emotional Stability   | 0.692        | yes           |
| 14   | est6  | Emotional Stability   | 0.685        | yes           |
| 15   | est1  | Emotional Stability   | 0.669        | yes           |
| 16   | est7  | Emotional Stability   | 0.662        | yes           |
| 17   | opn10 | Intellect/Imagination | 0.600        | no            |
| 18   | opn2  | Intellect/Imagination | 0.527        | yes           |
| 19   | opn1  | Intellect/Imagination | 0.521        | no            |
| 20   | opn5  | Intellect/Imagination | 0.513        | no            |
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
| 1    | ext3  | ext    | 0.638        | 1.548      | 0.638 | 0.361 | 0.150 | 0.335 | 0.063 |
| 2    | ext5  | ext    | 0.707        | 1.466      | 0.707 | 0.331 | 0.112 | 0.169 | 0.146 |
| 3    | ext7  | ext    | 0.689        | 1.303      | 0.689 | 0.274 | 0.062 | 0.173 | 0.104 |
| 4    | est10 | est    | 0.620        | 1.275      | 0.294 | 0.088 | 0.260 | 0.620 | 0.014 |
| 5    | ext4  | ext    | 0.718        | 1.255      | 0.718 | 0.191 | 0.068 | 0.187 | 0.091 |
| 6    | ext6  | ext    | 0.550        | 1.290      | 0.550 | 0.242 | 0.069 | 0.126 | 0.304 |
| 7    | agr10 | agr    | 0.417        | 1.252      | 0.378 | 0.417 | 0.153 | 0.177 | 0.127 |
| 8    | agr7  | agr    | 0.627        | 1.275      | 0.378 | 0.627 | 0.073 | 0.090 | 0.108 |
| 9    | ext10 | ext    | 0.666        | 1.238      | 0.666 | 0.191 | 0.073 | 0.219 | 0.088 |
| 10   | est9  | est    | 0.622        | 1.179      | 0.143 | 0.185 | 0.149 | 0.622 | 0.080 |
| 11   | csn4  | csn    | 0.561        | 1.182      | 0.108 | 0.094 | 0.561 | 0.378 | 0.041 |
| 12   | est8  | est    | 0.692        | 1.158      | 0.107 | 0.056 | 0.256 | 0.692 | 0.048 |
| 13   | est6  | est    | 0.685        | 1.109      | 0.131 | 0.003 | 0.176 | 0.685 | 0.114 |
| 14   | est1  | est    | 0.669        | 1.128      | 0.184 | 0.051 | 0.120 | 0.669 | 0.105 |
| 15   | ext2  | ext    | 0.675        | 1.105      | 0.675 | 0.248 | 0.005 | 0.070 | 0.107 |
| 16   | agr2  | agr    | 0.545        | 1.208      | 0.405 | 0.545 | 0.046 | 0.079 | 0.132 |
| 17   | est7  | est    | 0.662        | 1.074      | 0.085 | 0.040 | 0.243 | 0.662 | 0.043 |
| 18   | ext1  | ext    | 0.655        | 1.100      | 0.655 | 0.199 | 0.018 | 0.132 | 0.096 |
| 19   | csn8  | csn    | 0.472        | 1.075      | 0.106 | 0.180 | 0.472 | 0.244 | 0.072 |
| 20   | ext9  | ext    | 0.600        | 1.019      | 0.600 | 0.096 | 0.017 | 0.136 | 0.171 |
<!-- END:greedy_ranking -->

The first Intellect item falls just outside the top 20. Extraversion dominates
because its items show moderate cross-domain correlations (visible in the info
score columns above). These cross-loadings inflate composite info scores but are
too weak to reliably predict other domains.

### Mini-IPIP Reference

Donnellan et al. (2006) fixed 20-item mapping to IPIP-BFFM items.

<!-- BEGIN:mini_ipip -->
| Domain                | Items                  | Reported Alpha |
|-----------------------|------------------------|----------------|
| Extraversion          | ext1, ext7, ext2, ext4 | 0.77           |
| Agreeableness         | agr4, agr9, agr7, agr5 | 0.70           |
| Conscientiousness     | csn5, csn7, csn6, csn4 | 0.69           |
| Emotional Stability   | est8, est6, est2, est4 | 0.68           |
| Intellect/Imagination | opn3, opn2, opn4, opn6 | 0.65           |
<!-- END:mini_ipip -->

The key differences from the domain-balanced set: Mini-IPIP includes weaker
discriminators in Conscientiousness and Emotional Stability (csn7, est2) where
domain-balanced selects higher-correlation items (csn6, est1). Mini-IPIP's
Intellect items (opn3, opn2, opn4, opn6) all rank low by cross-domain info
score; this reflects the psychometric independence of Intellect/Imagination from
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
