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
recovery while the worst-served domain falls to weak/moderate prediction
(*r* ≈ 0.43, driven only by cross-domain inference).

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
at the same operating point drops sharply: the no-sparsity ablation falls well
below the reference on both sparse-20 validation and the *K* = 20 baseline.
(The cross-variant tables below report this comparison only when the ablation
suite is run; a reference-only build shows the reference variant alone.)

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
| Domain-balanced r                         | 0.9277 [0.9273, 0.9282] |
| Domain-balanced MAE                       | 8.13 pp                 |
| Domain-balanced 90% coverage              | 89.5%                   |
| Mini-IPIP r                               | 0.9068 [0.9061, 0.9074] |
| Constrained-adaptive r                    | 0.9066 [0.9060, 0.9072] |
| Adaptive top-K r (greedy)                 | 0.8225 [0.8215, 0.8236] |
| ML vs averaging delta r (domain-balanced) | +0.0063                 |
| ML vs averaging delta r (Mini-IPIP)       | +0.0113                 |
<!-- END:headline_k20 -->

### Cross-Variant Overview (Reference + Ablations)

Auto-generated from `artifacts/research_summary.json`, which aggregates
`models/*/training_report.json` and per-variant evaluation artifacts.

<!-- BEGIN:ablation_overview -->
> Ablation variants (no-sparsity, focused-only) were not run in this reference-only build; only the reference run is shown below.
| Variant   | Data Regime  | Train Val r (full-50) | Validate r (full-50) | Validate r (sparse-20) | Baselines K20 r (domain-balanced) | Simulation r | Complete |
|-----------|--------------|-----------------------|----------------------|------------------------|-----------------------------------|--------------|----------|
| Reference | canonical_v1 | 0.9997                | 0.9997               | 0.9107                 | 0.9277                            | 0.9273       | yes      |
<!-- END:ablation_overview -->

### Cross-Variant Provenance Locks

Short-hash provenance view used to confirm all reported numbers are tied to
their exact split and hyperparameter locks.

<!-- BEGIN:ablation_provenance -->
> Ablation variants (no-sparsity, focused-only) were not run in this reference-only build; only the reference run is shown below.
| Variant   | Split Signature | Train SHA256 | Hyperparams SHA256 | Git Hash     | Errors |
|-----------|-----------------|--------------|--------------------|--------------|--------|
| Reference | f688d5f439e8    | 858fd9123edd | eb1fc341bc5c       | d8b79fd23cd0 | none   |
<!-- END:ablation_provenance -->

### Cross-Variant Detailed Validation (All Runs)

Full-50 and sparse-20 validation tables for each trained variant.

<!-- BEGIN:ablation_validation_details -->
> Ablation variants (no-sparsity, focused-only) were not run in this reference-only build; only the reference run is shown below.


#### Reference


**Full-50 validation:**


| Domain                | r          | MAE      | RMSE     | Within-5   | 90% Coverage | Central Cov (20-80) | Tail Cov (<20,>80) | Raw Crossing Rate |
|-----------------------|------------|----------|----------|------------|--------------|---------------------|--------------------|-------------------|
| Extraversion          | 0.9999     | 0.34     | 0.49     | 100.0%     | 92.9%        | 94.0%               | 91.5%              | 27.5%             |
| Agreeableness         | 0.9997     | 0.51     | 0.72     | 100.0%     | 91.7%        | 94.6%               | 87.8%              | 26.5%             |
| Conscientiousness     | 0.9997     | 0.56     | 0.78     | 100.0%     | 93.1%        | 95.9%               | 89.5%              | 23.2%             |
| Emotional Stability   | 0.9998     | 0.46     | 0.65     | 100.0%     | 91.8%        | 93.8%               | 89.2%              | 26.6%             |
| Intellect/Imagination | 0.9995     | 0.69     | 0.96     | 100.0%     | 93.5%        | 96.6%               | 89.5%              | 23.0%             |
| **Overall**           | **0.9997** | **0.51** | **0.74** | **100.0%** | **92.6%**    | **95.0%**           | **89.5%**          | **25.4%**         |


**Sparse-20 validation:**


| Domain                | r          | MAE      | RMSE      | Within-5  | 90% Coverage |
|-----------------------|------------|----------|-----------|-----------|--------------|
| Extraversion          | 0.9358     | 7.72     | 10.68     | 46.9%     | 90.5%        |
| Agreeableness         | 0.9089     | 8.88     | 12.19     | 41.7%     | 89.8%        |
| Conscientiousness     | 0.8943     | 9.84     | 13.37     | 38.3%     | 89.4%        |
| Emotional Stability   | 0.9244     | 8.33     | 11.50     | 44.3%     | 90.0%        |
| Intellect/Imagination | 0.8879     | 10.01    | 13.73     | 38.4%     | 89.2%        |
| **Overall**           | **0.9107** | **8.96** | **12.35** | **41.9%** | **89.8%**    |
<!-- END:ablation_validation_details -->

### Cross-Variant Baseline Curves (All Runs)

Item-selection baseline curves (K=5..50) for each trained variant.

<!-- BEGIN:ablation_baselines_details -->
> Ablation variants (no-sparsity, focused-only) were not run in this reference-only build; only the reference run is shown below.


#### Reference


| K  | Domain-Balanced          | Constrained-Adaptive | Mini-IPIP                | First-N              | Random               | Adaptive Top-K       | Greedy-Balanced      | Worst-K              |
|----|--------------------------|----------------------|--------------------------|----------------------|----------------------|----------------------|----------------------|----------------------|
| 5  | 0.748 [0.746, 0.749]     | 0.691 [0.689, 0.693] | ---                      | 0.698 [0.696, 0.699] | 0.604 [0.603, 0.604] | 0.578 [0.576, 0.579] | 0.660 [0.658, 0.662] | 0.467 [0.465, 0.469] |
| 10 | 0.853 [0.852, 0.854]     | 0.795 [0.794, 0.796] | ---                      | 0.821 [0.820, 0.822] | 0.766 [0.765, 0.766] | 0.707 [0.705, 0.708] | 0.764 [0.763, 0.765] | 0.649 [0.647, 0.651] |
| 15 | 0.909 [0.908, 0.910]     | 0.849 [0.848, 0.850] | ---                      | 0.873 [0.873, 0.874] | 0.847 [0.846, 0.847] | 0.793 [0.792, 0.794] | 0.824 [0.823, 0.825] | 0.763 [0.762, 0.765] |
| 20 | **0.928 [0.927, 0.928]** | 0.907 [0.906, 0.907] | **0.907 [0.906, 0.907]** | 0.911 [0.910, 0.911] | 0.895 [0.895, 0.896] | 0.823 [0.822, 0.824] | 0.850 [0.849, 0.851] | 0.810 [0.809, 0.811] |
| 25 | 0.945 [0.945, 0.946]     | 0.939 [0.939, 0.940] | ---                      | 0.939 [0.939, 0.939] | 0.934 [0.933, 0.934] | 0.900 [0.899, 0.900] | 0.900 [0.899, 0.900] | 0.889 [0.888, 0.889] |
| 30 | 0.961 [0.961, 0.961]     | 0.952 [0.952, 0.953] | ---                      | 0.957 [0.957, 0.957] | 0.952 [0.952, 0.952] | 0.929 [0.928, 0.929] | 0.929 [0.928, 0.929] | 0.929 [0.929, 0.930] |
| 40 | 0.983 [0.983, 0.983]     | 0.980 [0.980, 0.980] | ---                      | 0.984 [0.984, 0.984] | 0.982 [0.982, 0.982] | 0.968 [0.968, 0.968] | 0.968 [0.968, 0.968] | 0.978 [0.977, 0.978] |
| 50 | 1.000 [1.000, 1.000]     | 1.000 [1.000, 1.000] | ---                      | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
<!-- END:ablation_baselines_details -->

### Cross-Variant Per-Domain K=20 (All Runs)

Per-domain K=20 breakdown (domain-balanced, Mini-IPIP, first-N) for each variant.

<!-- BEGIN:ablation_per_domain_k20_details -->
> Ablation variants (no-sparsity, focused-only) were not run in this reference-only build; only the reference run is shown below.


#### Reference


**Domain-Balanced (4 items per domain):**

| Domain                | r      | Items | CI Lower | CI Upper |
|-----------------------|--------|-------|----------|----------|
| Extraversion          | 0.9470 | 4     | 0.9462   | 0.9477   |
| Agreeableness         | 0.9202 | 4     | 0.9190   | 0.9213   |
| Conscientiousness     | 0.9206 | 4     | 0.9194   | 0.9218   |
| Emotional Stability   | 0.9376 | 4     | 0.9367   | 0.9385   |
| Intellect/Imagination | 0.9120 | 4     | 0.9108   | 0.9133   |

**Mini-IPIP (4 items per domain):**

| Domain                | r      | Items | CI Lower | CI Upper |
|-----------------------|--------|-------|----------|----------|
| Extraversion          | 0.9376 | 4     | 0.9367   | 0.9385   |
| Agreeableness         | 0.9118 | 4     | 0.9105   | 0.9131   |
| Conscientiousness     | 0.9099 | 4     | 0.9086   | 0.9113   |
| Emotional Stability   | 0.9298 | 4     | 0.9288   | 0.9308   |
| Intellect/Imagination | 0.8422 | 4     | 0.8400   | 0.8442   |

**First-N (4 items per domain):**

| Domain                | r      | Items | CI Lower | CI Upper |
|-----------------------|--------|-------|----------|----------|
| Extraversion          | 0.9397 | 4     | 0.9388   | 0.9406   |
| Agreeableness         | 0.9270 | 4     | 0.9259   | 0.9282   |
| Conscientiousness     | 0.8963 | 4     | 0.8948   | 0.8976   |
| Emotional Stability   | 0.8781 | 4     | 0.8764   | 0.8798   |
| Intellect/Imagination | 0.9121 | 4     | 0.9109   | 0.9134   |
<!-- END:ablation_per_domain_k20_details -->

### Cross-Variant Domain Starvation at K=20 (All Runs)

Adaptive top-K domain allocation and resulting per-domain accuracy for each variant.

<!-- BEGIN:ablation_domain_starvation_details -->
> Ablation variants (no-sparsity, focused-only) were not run in this reference-only build; only the reference run is shown below.


#### Reference


| Domain                | Items | Share | r      | CI Lower | CI Upper |
|-----------------------|-------|-------|--------|----------|----------|
| Extraversion          | 9     | 45%   | 0.9940 | 0.9939   | 0.9941   |
| Emotional Stability   | 6     | 30%   | 0.9703 | 0.9699   | 0.9708   |
| Agreeableness         | 3     | 15%   | 0.8250 | 0.8226   | 0.8274   |
| Conscientiousness     | 2     | 10%   | 0.7613 | 0.7582   | 0.7641   |
| Intellect/Imagination | 0     | 0%    | 0.4258 | 0.4206   | 0.4312   |
<!-- END:ablation_domain_starvation_details -->

### Cross-Variant ML vs Averaging (All Runs)

ML-vs-averaging deltas for matched item sets across each variant.

<!-- BEGIN:ablation_ml_vs_averaging_details -->
> Ablation variants (no-sparsity, focused-only) were not run in this reference-only build; only the reference run is shown below.


#### Reference


| Strategy        | K  | ML r   | Avg r  | Delta r | ML MAE | Avg MAE | Delta MAE |
|-----------------|----|--------|--------|---------|--------|---------|-----------|
| Domain-balanced | 10 | 0.8533 | 0.8428 | +0.0105 | 11.64  | 12.09   | -0.44     |
| Domain-balanced | 15 | 0.9090 | 0.9026 | +0.0065 | 9.10   | 9.49    | -0.39     |
| Domain-balanced | 20 | 0.9277 | 0.9214 | +0.0063 | 8.13   | 8.52    | -0.39     |
| Mini-IPIP       | 20 | 0.9180 | 0.9068 | +0.0113 | 8.57   | 9.17    | -0.60     |
| Domain-balanced | 25 | 0.9452 | 0.9395 | +0.0057 | 7.10   | 7.47    | -0.37     |
<!-- END:ablation_ml_vs_averaging_details -->

### Cross-Variant Simulation Results (All Runs)

Adaptive simulation outcomes for each variant at the operating point.

<!-- BEGIN:ablation_simulation_details -->
> Ablation variants (no-sparsity, focused-only) were not run in this reference-only build; only the reference run is shown below.


#### Reference


Simulated on a random 5,000-respondent subsample of the held-out test split (the baseline and validation tables use the full *N* = 90,498), so these estimates carry wider confidence intervals and are not co-powered with the headline numbers.

| Domain                | r          | MAE      | RMSE      | Within-5  | 90% Coverage |
|-----------------------|------------|----------|-----------|-----------|--------------|
| Extraversion          | 0.9384     | 7.55     | 10.41     | 47.0%     | 91.8%        |
| Agreeableness         | 0.9214     | 8.33     | 11.44     | 43.3%     | 88.0%        |
| Conscientiousness     | 0.9235     | 8.42     | 11.52     | 43.7%     | 89.6%        |
| Emotional Stability   | 0.9390     | 7.63     | 10.37     | 46.2%     | 88.7%        |
| Intellect/Imagination | 0.9135     | 8.84     | 12.18     | 41.7%     | 89.2%        |
| **Overall**           | **0.9273** | **8.16** | **11.20** | **44.4%** | **89.5%**    |
<!-- END:ablation_simulation_details -->

### Dataset

Valid respondents from the Open-Source Psychometrics Project, split at
random into train/validation/test (70/15/15, seed-locked).

<!-- BEGIN:data_splits -->
| Split       | Respondents | Fraction |
|-------------|-------------|----------|
| Total valid | 603,322     | 100%     |
| Train       | 422,326     | 70.0%    |
| Validation  | 90,498      | 15.0%    |
| Test        | 90,498      | 15.0%    |

Split: canonical_v1 — plain random partition (70/15/15, seed=42).

| Domain                | Max Mean Diff | KS Statistic | KS p-value |
|-----------------------|---------------|--------------|------------|
| Extraversion          | 0.0081        | 0.0038       | 0.230      |
| Agreeableness         | 0.0040        | 0.0028       | 0.583      |
| Conscientiousness     | 0.0012        | 0.0036       | 0.286      |
| Emotional Stability   | 0.0096        | 0.0034       | 0.352      |
| Intellect/Imagination | 0.0037        | 0.0025       | 0.741      |
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
| n_estimators     | 9,056                                       |
| max_depth        | 5                                           |
| learning_rate    | 0.0107                                      |
| min_child_weight | 4                                           |
| subsample        | 0.651                                       |
| colsample_bytree | 0.529                                       |
| reg_lambda       | 4.564                                       |
| reg_alpha        | 4.612                                       |
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
| Extraversion          | 2.9146 | 0.9112 |
| Agreeableness         | 3.7590 | 0.7366 |
| Conscientiousness     | 3.3426 | 0.7390 |
| Emotional Stability   | 2.9190 | 0.8611 |
| Intellect/Imagination | 3.9387 | 0.6183 |
<!-- END:norms -->

### Ceiling Check: Full-Model Accuracy (50 Items, 90K Test)

Sanity check with all 50 items present. Near-perfect reconstruction is
expected; these numbers confirm the model and inference pipeline work
correctly, not the operating-point accuracy. See the K=20 tables below for
the primary evaluation.

<!-- BEGIN:validation -->
| Domain                | r          | MAE      | RMSE     | Within-5   | 90% Coverage | Central Cov (20-80) | Tail Cov (<20,>80) | Raw Crossing Rate |
|-----------------------|------------|----------|----------|------------|--------------|---------------------|--------------------|-------------------|
| Extraversion          | 0.9999     | 0.34     | 0.49     | 100.0%     | 92.9%        | 94.0%               | 91.5%              | 27.5%             |
| Agreeableness         | 0.9997     | 0.51     | 0.72     | 100.0%     | 91.7%        | 94.6%               | 87.8%              | 26.5%             |
| Conscientiousness     | 0.9997     | 0.56     | 0.78     | 100.0%     | 93.1%        | 95.9%               | 89.5%              | 23.2%             |
| Emotional Stability   | 0.9998     | 0.46     | 0.65     | 100.0%     | 91.8%        | 93.8%               | 89.2%              | 26.6%             |
| Intellect/Imagination | 0.9995     | 0.69     | 0.96     | 100.0%     | 93.5%        | 96.6%               | 89.5%              | 23.0%             |
| **Overall**           | **0.9997** | **0.51** | **0.74** | **100.0%** | **92.6%**    | **95.0%**           | **89.5%**          | **25.4%**         |
<!-- END:validation -->

Coverage is **not uniform across the score range**: aggregate (90%) coverage
meets or exceeds the nominal target in the central band (20-80 percentile), but
the intervals **under-cover at the score extremes** (below the 20th / above the
80th percentile) — compare the "Central Cov" and "Tail Cov" columns. The quintile
table below shows the same pattern more finely. The "Raw Crossing Rate" column is
the **pre-sort** rate: the three independently-trained q05/q50/q95 models are not
jointly monotone and disagree on ordering for a substantial fraction of
full-information predictions. The reported intervals are forced monotonic with a
sort before use, so the post-sort crossing rate is zero **by construction** — it
is therefore not reported as if it were a measured quantity.

### Ceiling Check by Quintile (50 Items, 90K Test)

Per-domain performance varies systematically by score quintile (best in tails,
worst near the center), and interval widths expand near the middle of the
distribution.

<!-- BEGIN:validation_quintiles -->
| Domain                             | Quintile | n      | MAE      | 90% Coverage | Mean PI Width |
|------------------------------------|----------|--------|----------|--------------|---------------|
| Extraversion                       | Q1       | 20,538 | 0.18     | 91.2%        | 0.98          |
| Extraversion                       | Q2       | 16,387 | 0.44     | 93.2%        | 2.46          |
| Extraversion                       | Q3       | 17,383 | 0.53     | 95.2%        | 3.25          |
| Extraversion                       | Q4       | 20,600 | 0.41     | 93.2%        | 2.43          |
| Extraversion                       | Q5       | 15,590 | 0.14     | 91.6%        | 0.79          |
| Agreeableness                      | Q1       | 20,686 | 0.40     | 90.2%        | 2.05          |
| Agreeableness                      | Q2       | 19,622 | 0.72     | 94.5%        | 4.37          |
| Agreeableness                      | Q3       | 15,425 | 0.70     | 94.9%        | 4.21          |
| Agreeableness                      | Q4       | 18,516 | 0.51     | 94.0%        | 3.00          |
| Agreeableness                      | Q5       | 16,249 | 0.22     | 84.7%        | 1.05          |
| Conscientiousness                  | Q1       | 19,760 | 0.35     | 89.8%        | 1.86          |
| Conscientiousness                  | Q2       | 16,603 | 0.73     | 94.8%        | 4.18          |
| Conscientiousness                  | Q3       | 18,315 | 0.82     | 96.8%        | 5.10          |
| Conscientiousness                  | Q4       | 19,272 | 0.64     | 95.4%        | 3.93          |
| Conscientiousness                  | Q5       | 16,548 | 0.26     | 88.2%        | 1.42          |
| Emotional Stability                | Q1       | 18,277 | 0.23     | 89.2%        | 1.22          |
| Emotional Stability                | Q2       | 21,114 | 0.57     | 93.5%        | 3.21          |
| Emotional Stability                | Q3       | 15,186 | 0.71     | 94.5%        | 4.02          |
| Emotional Stability                | Q4       | 19,213 | 0.56     | 92.9%        | 3.11          |
| Emotional Stability                | Q5       | 16,708 | 0.23     | 89.0%        | 1.18          |
| Intellect/Imagination              | Q1       | 19,321 | 0.48     | 91.3%        | 2.50          |
| Intellect/Imagination              | Q2       | 18,314 | 0.93     | 95.9%        | 5.40          |
| Intellect/Imagination              | Q3       | 22,038 | 0.94     | 97.3%        | 6.12          |
| Intellect/Imagination              | Q4       | 14,731 | 0.71     | 96.1%        | 4.29          |
| Intellect/Imagination              | Q5       | 16,094 | 0.28     | 85.9%        | 1.50          |
| **Avg Tails (Q1/Q5, all domains)** | ---      | ---    | **0.28** | **89.1%**    | **1.46**      |
| **Avg Center (Q3, all domains)**   | ---      | ---    | **0.74** | **95.7%**    | **4.54**      |
<!-- END:validation_quintiles -->

### Item Selection Strategies (5-50 Items, 90K Test)

Pearson r with full 50-item scores. Bootstrap 95% CIs from 1,000 resamples.

<!-- BEGIN:baselines -->
| K  | Domain-Balanced          | Constrained-Adaptive | Mini-IPIP                | First-N              | Random               | Adaptive Top-K       | Greedy-Balanced      | Worst-K              |
|----|--------------------------|----------------------|--------------------------|----------------------|----------------------|----------------------|----------------------|----------------------|
| 5  | 0.748 [0.746, 0.749]     | 0.691 [0.689, 0.693] | ---                      | 0.698 [0.696, 0.699] | 0.604 [0.603, 0.604] | 0.578 [0.576, 0.579] | 0.660 [0.658, 0.662] | 0.467 [0.465, 0.469] |
| 10 | 0.853 [0.852, 0.854]     | 0.795 [0.794, 0.796] | ---                      | 0.821 [0.820, 0.822] | 0.766 [0.765, 0.766] | 0.707 [0.705, 0.708] | 0.764 [0.763, 0.765] | 0.649 [0.647, 0.651] |
| 15 | 0.909 [0.908, 0.910]     | 0.849 [0.848, 0.850] | ---                      | 0.873 [0.873, 0.874] | 0.847 [0.846, 0.847] | 0.793 [0.792, 0.794] | 0.824 [0.823, 0.825] | 0.763 [0.762, 0.765] |
| 20 | **0.928 [0.927, 0.928]** | 0.907 [0.906, 0.907] | **0.907 [0.906, 0.907]** | 0.911 [0.910, 0.911] | 0.895 [0.895, 0.896] | 0.823 [0.822, 0.824] | 0.850 [0.849, 0.851] | 0.810 [0.809, 0.811] |
| 25 | 0.945 [0.945, 0.946]     | 0.939 [0.939, 0.940] | ---                      | 0.939 [0.939, 0.939] | 0.934 [0.933, 0.934] | 0.900 [0.899, 0.900] | 0.900 [0.899, 0.900] | 0.889 [0.888, 0.889] |
| 30 | 0.961 [0.961, 0.961]     | 0.952 [0.952, 0.953] | ---                      | 0.957 [0.957, 0.957] | 0.952 [0.952, 0.952] | 0.929 [0.928, 0.929] | 0.929 [0.928, 0.929] | 0.929 [0.929, 0.930] |
| 40 | 0.983 [0.983, 0.983]     | 0.980 [0.980, 0.980] | ---                      | 0.984 [0.984, 0.984] | 0.982 [0.982, 0.982] | 0.968 [0.968, 0.968] | 0.968 [0.968, 0.968] | 0.978 [0.977, 0.978] |
| 50 | 1.000 [1.000, 1.000]     | 1.000 [1.000, 1.000] | ---                      | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
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
| Extraversion          | 0.9470 | 4     | 0.9462   | 0.9477   |
| Agreeableness         | 0.9202 | 4     | 0.9190   | 0.9213   |
| Conscientiousness     | 0.9206 | 4     | 0.9194   | 0.9218   |
| Emotional Stability   | 0.9376 | 4     | 0.9367   | 0.9385   |
| Intellect/Imagination | 0.9120 | 4     | 0.9108   | 0.9133   |

**Mini-IPIP (4 items per domain):**

| Domain                | r      | Items | CI Lower | CI Upper |
|-----------------------|--------|-------|----------|----------|
| Extraversion          | 0.9376 | 4     | 0.9367   | 0.9385   |
| Agreeableness         | 0.9118 | 4     | 0.9105   | 0.9131   |
| Conscientiousness     | 0.9099 | 4     | 0.9086   | 0.9113   |
| Emotional Stability   | 0.9298 | 4     | 0.9288   | 0.9308   |
| Intellect/Imagination | 0.8422 | 4     | 0.8400   | 0.8442   |

**First-N (4 items per domain):**

| Domain                | r      | Items | CI Lower | CI Upper |
|-----------------------|--------|-------|----------|----------|
| Extraversion          | 0.9397 | 4     | 0.9388   | 0.9406   |
| Agreeableness         | 0.9270 | 4     | 0.9259   | 0.9282   |
| Conscientiousness     | 0.8963 | 4     | 0.8948   | 0.8976   |
| Emotional Stability   | 0.8781 | 4     | 0.8764   | 0.8798   |
| Intellect/Imagination | 0.9121 | 4     | 0.9109   | 0.9134   |
<!-- END:per_domain_k20 -->

Domain-balanced achieves the most uniform per-domain performance. Mini-IPIP shows
a clear Intellect/Imagination gap relative to domain-balanced (visible in the table
above) because its Intellect items were selected for brevity, not discrimination.

### Domain Starvation (Greedy Selection at K=20)

<!-- BEGIN:domain_starvation -->
| Domain                | Items | Share | r      | CI Lower | CI Upper |
|-----------------------|-------|-------|--------|----------|----------|
| Extraversion          | 9     | 45%   | 0.9940 | 0.9939   | 0.9941   |
| Emotional Stability   | 6     | 30%   | 0.9703 | 0.9699   | 0.9708   |
| Agreeableness         | 3     | 15%   | 0.8250 | 0.8226   | 0.8274   |
| Conscientiousness     | 2     | 10%   | 0.7613 | 0.7582   | 0.7641   |
| Intellect/Imagination | 0     | 0%    | 0.4258 | 0.4206   | 0.4312   |
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
| Domain-balanced | 10 | 0.8533 | 0.8428 | +0.0105 | 11.64  | 12.09   | -0.44     |
| Domain-balanced | 15 | 0.9090 | 0.9026 | +0.0065 | 9.10   | 9.49    | -0.39     |
| Domain-balanced | 20 | 0.9277 | 0.9214 | +0.0063 | 8.13   | 8.52    | -0.39     |
| Mini-IPIP       | 20 | 0.9180 | 0.9068 | +0.0113 | 8.57   | 9.17    | -0.60     |
| Domain-balanced | 25 | 0.9452 | 0.9395 | +0.0057 | 7.10   | 7.47    | -0.37     |
<!-- END:ml_vs_averaging -->

The ML advantage holds across all tested item counts, as the table shows, and is
largest at fewer items where cross-domain information sharing matters most. Both
correlation and MAE improvements are consistent across budgets.

#### Decomposing the Headline: Scoring vs Item Selection (K = 20)

The headline BFFM-XGB-20 vs Mini-IPIP gap mixes two distinct levers — the
**scoring method** (XGBoost vs simple averaging) and the **item set**
(top-4-by-*r* vs the expert-curated Mini-IPIP items). The table below holds the
item set fixed and reports the *scoring* gain (ML − averaging) separately for each
set, so the two contributions are not conflated.

<!-- BEGIN:ml_vs_averaging_per_domain -->
| Domain                | DB items: ML r | DB items: Avg r | DB scoring Δr | Mini-IPIP items: ML r | Mini-IPIP items: Avg r | Mini-IPIP scoring Δr |
|-----------------------|----------------|-----------------|---------------|-----------------------|------------------------|----------------------|
| Extraversion          | 0.9470         | 0.9428          | +0.0042       | 0.9444                | 0.9376                 | +0.0068              |
| Agreeableness         | 0.9202         | 0.9118          | +0.0083       | 0.9203                | 0.9118                 | +0.0085              |
| Conscientiousness     | 0.9206         | 0.9157          | +0.0049       | 0.9156                | 0.9099                 | +0.0058              |
| Emotional Stability   | 0.9376         | 0.9300          | +0.0076       | 0.9439                | 0.9298                 | +0.0142              |
| Intellect/Imagination | 0.9120         | 0.9049          | +0.0071       | 0.8621                | 0.8422                 | +0.0200              |
<!-- END:ml_vs_averaging_per_domain -->

Reading the table: the "scoring Δr" columns are the pure ML-over-averaging gain on
identical items. For **Emotional Stability** the two 20-item sets recover almost
equally well under ML scoring while averaging trails both — so the headline EST gain
is driven by the **scoring method, not the item selection** (compare the small ML-*r*
gap between the two EST rows against their much larger "scoring Δr"). The full per-row
decomposition is in `artifacts/variants/reference/ml_vs_averaging_comparison.json`.

The Mini-IPIP comparator now carries the same respondent-level bootstrap 95% CIs as
every XGBoost method (see the item-selection table above), and the headline
domain-balanced-ML vs Mini-IPIP-averaging gap is reported with a *paired*
(same-respondent) bootstrap CI in the `xgb_vs_mini_ipip_paired` block of that
artifact — so the overall-r contrast (see the table above) can be judged for
significance.

### Simulation Results (20-Item Operating Point)

Held-out respondents from the test split, correlation-ranked selection,
SEM threshold 0.45, min 4 items per domain. All respondents converge to
exactly 20 items (4-4-4-4-4).

<!-- BEGIN:simulation -->
Simulated on a random 5,000-respondent subsample of the held-out test split (the baseline and validation tables use the full *N* = 90,498), so these estimates carry wider confidence intervals and are not co-powered with the headline numbers.

| Domain                | r          | MAE      | RMSE      | Within-5  | 90% Coverage |
|-----------------------|------------|----------|-----------|-----------|--------------|
| Extraversion          | 0.9384     | 7.55     | 10.41     | 47.0%     | 91.8%        |
| Agreeableness         | 0.9214     | 8.33     | 11.44     | 43.3%     | 88.0%        |
| Conscientiousness     | 0.9235     | 8.42     | 11.52     | 43.7%     | 89.6%        |
| Emotional Stability   | 0.9390     | 7.63     | 10.37     | 46.2%     | 88.7%        |
| Intellect/Imagination | 0.9135     | 8.84     | 12.18     | 41.7%     | 89.2%        |
| **Overall**           | **0.9273** | **8.16** | **11.20** | **44.4%** | **89.5%**    |
<!-- END:simulation -->

Simulation results closely match the static baseline evaluation (compare the
overall *r* here to the headline results table). Note the simulation is run on a
random subsample of the test split (see the table caption for the count), not the
full held-out test split used for the baseline and validation tables, so its
estimates are less precise and should not be read as co-powered with the headline
numbers. Correlation-ranked selection is equivalent to the optimal static strategy;
the agreement between the two also confirms the pipeline end-to-end.

### Calibration

<!-- BEGIN:calibration -->
Calibration regime: `sparse_20_balanced` (domain-balanced, 20 items, 4 per domain).

| Domain                | Observed Coverage | Scale Factor |
|-----------------------|-------------------|--------------|
| Extraversion          | 90.4%             | 1.0          |
| Agreeableness         | 89.7%             | 1.0          |
| Conscientiousness     | 89.3%             | 1.0          |
| Emotional Stability   | 89.8%             | 1.0          |
| Intellect/Imagination | 89.4%             | 1.0          |
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
| 4    | ext2  | Extraversion          | 0.676        | yes           |
| 5    | agr4  | Agreeableness         | 0.716        | no            |
| 6    | agr9  | Agreeableness         | 0.639        | no            |
| 7    | agr7  | Agreeableness         | 0.627        | yes           |
| 8    | agr5  | Agreeableness         | 0.625        | yes           |
| 9    | csn6  | Conscientiousness     | 0.590        | yes           |
| 10   | csn1  | Conscientiousness     | 0.576        | no            |
| 11   | csn5  | Conscientiousness     | 0.569        | no            |
| 12   | csn4  | Conscientiousness     | 0.560        | yes           |
| 13   | est8  | Emotional Stability   | 0.692        | yes           |
| 14   | est6  | Emotional Stability   | 0.685        | yes           |
| 15   | est1  | Emotional Stability   | 0.669        | yes           |
| 16   | est7  | Emotional Stability   | 0.662        | yes           |
| 17   | opn10 | Intellect/Imagination | 0.600        | no            |
| 18   | opn2  | Intellect/Imagination | 0.528        | yes           |
| 19   | opn1  | Intellect/Imagination | 0.520        | no            |
| 20   | opn5  | Intellect/Imagination | 0.513        | no            |
<!-- END:domain_balanced_items -->

This 20-item set differs from the Mini-IPIP: the domain-balanced set selects by
maximum within-domain correlation, while Mini-IPIP was designed for brevity and
broad coverage.

Note the deployed Emotional Stability four-item subset (est1, est6, est7, est8) is
entirely **reverse-keyed** — a consequence of ranking purely by within-domain
correlation. This maximizes discrimination but makes the EST short-form score
vulnerable to acquiescence (yea-saying) bias; the other four domains mix keyed
directions, and the full 50-item assessment is unaffected.

### Internal-Consistency Reliability

Cronbach's alpha for each domain across three forms — the full 10-item domains, the
deployed domain-balanced 20-item form (4 items/domain), and the Mini-IPIP 4-item
form — computed on the training split. (Reliability bounds how high score-recovery
*r* can plausibly go; standardized alpha, mean inter-item *r*, and McDonald's omega
are in `reliability.json`.)

<!-- BEGIN:reliability -->
Cronbach's alpha by domain, computed on the **training split**. Standardized alpha, mean inter-item *r*, and McDonald's omega are in `reliability.json`.

| Domain                | Full 50-item | Domain-balanced 20 | Mini-IPIP 20 |
|-----------------------|--------------|--------------------|--------------|
| Extraversion          | 0.898        | 0.825              | 0.816        |
| Agreeableness         | 0.843        | 0.808              | 0.808        |
| Conscientiousness     | 0.823        | 0.728              | 0.700        |
| Emotional Stability   | 0.874        | 0.826              | 0.694        |
| Intellect/Imagination | 0.800        | 0.683              | 0.695        |
<!-- END:reliability -->

### Greedy Item Ranking (Cross-Domain Info Score)

Top 20 items by cross-domain information score. The domain distribution is heavily
skewed, which drives the domain starvation mechanism described above (see the
domain starvation table for counts).

<!-- BEGIN:greedy_ranking -->
| Rank | Item  | Domain | Own-Domain r | Info Score | Ext   | Agr   | Csn   | Est   | Opn   |
|------|-------|--------|--------------|------------|-------|-------|-------|-------|-------|
| 1    | ext3  | ext    | 0.637        | 1.546      | 0.637 | 0.362 | 0.150 | 0.335 | 0.062 |
| 2    | ext5  | ext    | 0.707        | 1.465      | 0.707 | 0.330 | 0.112 | 0.169 | 0.146 |
| 3    | ext7  | ext    | 0.689        | 1.304      | 0.689 | 0.274 | 0.062 | 0.173 | 0.105 |
| 4    | est10 | est    | 0.620        | 1.276      | 0.293 | 0.090 | 0.259 | 0.620 | 0.015 |
| 5    | ext4  | ext    | 0.718        | 1.253      | 0.718 | 0.190 | 0.067 | 0.186 | 0.091 |
| 6    | ext6  | ext    | 0.549        | 1.286      | 0.549 | 0.242 | 0.067 | 0.125 | 0.302 |
| 7    | agr10 | agr    | 0.416        | 1.251      | 0.377 | 0.416 | 0.153 | 0.177 | 0.129 |
| 8    | agr7  | agr    | 0.627        | 1.275      | 0.377 | 0.627 | 0.074 | 0.091 | 0.107 |
| 9    | ext10 | ext    | 0.666        | 1.238      | 0.666 | 0.192 | 0.074 | 0.219 | 0.088 |
| 10   | est9  | est    | 0.622        | 1.175      | 0.142 | 0.185 | 0.147 | 0.622 | 0.079 |
| 11   | csn4  | csn    | 0.560        | 1.183      | 0.108 | 0.097 | 0.560 | 0.377 | 0.041 |
| 12   | est8  | est    | 0.692        | 1.158      | 0.106 | 0.058 | 0.254 | 0.692 | 0.048 |
| 13   | est6  | est    | 0.685        | 1.108      | 0.131 | 0.004 | 0.175 | 0.685 | 0.114 |
| 14   | est1  | est    | 0.669        | 1.124      | 0.184 | 0.050 | 0.118 | 0.669 | 0.103 |
| 15   | ext2  | ext    | 0.676        | 1.104      | 0.676 | 0.247 | 0.004 | 0.070 | 0.106 |
| 16   | agr2  | agr    | 0.544        | 1.206      | 0.404 | 0.544 | 0.047 | 0.079 | 0.132 |
| 17   | ext1  | ext    | 0.656        | 1.101      | 0.656 | 0.200 | 0.018 | 0.131 | 0.095 |
| 18   | est7  | est    | 0.662        | 1.071      | 0.084 | 0.041 | 0.241 | 0.662 | 0.042 |
| 19   | csn8  | csn    | 0.471        | 1.070      | 0.105 | 0.180 | 0.471 | 0.243 | 0.070 |
| 20   | ext9  | ext    | 0.599        | 1.015      | 0.599 | 0.094 | 0.017 | 0.135 | 0.169 |
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
no-augmentation). Formalize when ablation artifacts are standardized.

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
