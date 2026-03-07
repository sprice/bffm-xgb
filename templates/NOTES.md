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
*Data will be populated after training run.*
<!-- END:headline_k20 -->

### Cross-Variant Overview (Reference + Ablations)

Auto-generated from `artifacts/research_summary.json`, which aggregates
`models/*/training_report.json` and per-variant evaluation artifacts.

<!-- BEGIN:ablation_overview -->
*Data will be populated after training run.*
<!-- END:ablation_overview -->

### Cross-Variant Provenance Locks

Short-hash provenance view used to confirm all reported numbers are tied to
their exact split and hyperparameter locks.

<!-- BEGIN:ablation_provenance -->
*Data will be populated after training run.*
<!-- END:ablation_provenance -->

### Cross-Variant Detailed Validation (All Runs)

Full-50 and sparse-20 validation tables for each trained variant.

<!-- BEGIN:ablation_validation_details -->
*Data will be populated after training run.*
<!-- END:ablation_validation_details -->

### Cross-Variant Baseline Curves (All Runs)

Item-selection baseline curves (K=5..50) for each trained variant.

<!-- BEGIN:ablation_baselines_details -->
*Data will be populated after training run.*
<!-- END:ablation_baselines_details -->

### Cross-Variant Per-Domain K=20 (All Runs)

Per-domain K=20 breakdown (domain-balanced, Mini-IPIP, first-N) for each variant.

<!-- BEGIN:ablation_per_domain_k20_details -->
*Data will be populated after training run.*
<!-- END:ablation_per_domain_k20_details -->

### Cross-Variant Domain Starvation at K=20 (All Runs)

Adaptive top-K domain allocation and resulting per-domain accuracy for each variant.

<!-- BEGIN:ablation_domain_starvation_details -->
*Data will be populated after training run.*
<!-- END:ablation_domain_starvation_details -->

### Cross-Variant ML vs Averaging (All Runs)

ML-vs-averaging deltas for matched item sets across each variant.

<!-- BEGIN:ablation_ml_vs_averaging_details -->
*Data will be populated after training run.*
<!-- END:ablation_ml_vs_averaging_details -->

### Cross-Variant Simulation Results (All Runs)

Adaptive simulation outcomes for each variant at the operating point.

<!-- BEGIN:ablation_simulation_details -->
*Data will be populated after training run.*
<!-- END:ablation_simulation_details -->

### Dataset

Valid respondents from the Open-Source Psychometrics Project, stratified
split into train/validation/test.

<!-- BEGIN:data_splits -->
*Data will be populated after training run.*
<!-- END:data_splits -->

### Training Configuration

Sparsity augmentation and training settings from `configs/reference.yaml`.

<!-- BEGIN:training_config -->
*Data will be populated after training run.*
<!-- END:training_config -->

### Model Configuration

<!-- BEGIN:model_config -->
*Data will be populated after training run.*
<!-- END:model_config -->

### Hyperparameter Overrides

If any hyperparameters were manually adjusted after Optuna tuning (e.g., to
control model size for deployment), the changes are logged here.

<!-- BEGIN:hyperparameter_overrides -->
*Data will be populated after training run.*
<!-- END:hyperparameter_overrides -->

### Population Norms

Derived from the training-split OSPP respondents (see dataset table for count).
Used for raw-score to percentile conversion via
`percentile = phi((raw - mu) / sigma) * 100`.

<!-- BEGIN:norms -->
*Data will be populated after training run.*
<!-- END:norms -->

### Ceiling Check: Full-Model Accuracy (50 Items, 90K Test)

Sanity check with all 50 items present. Near-perfect reconstruction is
expected; these numbers confirm the model and inference pipeline work
correctly, not the operating-point accuracy. See the K=20 tables below for
the primary evaluation.

<!-- BEGIN:validation -->
*Data will be populated after training run.*
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
*Data will be populated after training run.*
<!-- END:validation_quintiles -->

### Item Selection Strategies (5-50 Items, 90K Test)

Pearson r with full 50-item scores. Bootstrap 95% CIs from 1,000 resamples.

<!-- BEGIN:baselines -->
*Data will be populated after training run.*
<!-- END:baselines -->

At K>=25, greedy-balanced and adaptive top-K converge (identical item sets once the
greedy tail dominates). At K=40, greedy approaches nearly match the full-scale
ceiling because most items are included regardless of selection strategy. The
paradox is about the operating range of interest (10-25 items), not item selection
in general.

### Per-Domain Breakdown at K=20

<!-- BEGIN:per_domain_k20 -->
*Data will be populated after training run.*
<!-- END:per_domain_k20 -->

Domain-balanced achieves the most uniform per-domain performance. Mini-IPIP shows
a clear Intellect/Openness gap relative to domain-balanced (visible in the table
above) because its Intellect items were selected for brevity, not discrimination.

### Domain Starvation (Greedy Selection at K=20)

<!-- BEGIN:domain_starvation -->
*Data will be populated after training run.*
<!-- END:domain_starvation -->

Intellect first appears just outside the top 20 in the greedy ranking (see the
greedy item ranking table below). The table above shows the resulting allocation:
Extraversion dominates because its items correlate moderately with all other
domains, while Intellect items rank last because they are the most
psychometrically independent factor.

### ML vs Simple Averaging (Same Items)

<!-- BEGIN:ml_vs_averaging -->
*Data will be populated after training run.*
<!-- END:ml_vs_averaging -->

The ML advantage holds across all tested item counts, as the table shows, and is
largest at fewer items where cross-domain information sharing matters most. Both
correlation and MAE improvements are consistent across budgets.

### Simulation Results (20-Item Operating Point)

Held-out respondents from the test split, correlation-ranked selection,
SEM threshold 0.45, min 4 items per domain. All respondents converge to
exactly 20 items (4-4-4-4-4).

<!-- BEGIN:simulation -->
*Data will be populated after training run.*
<!-- END:simulation -->

Simulation results closely match the static baseline evaluation (compare the
overall *r* here to the headline results table). Correlation-ranked selection is
equivalent to the optimal static strategy; the agreement between the two also
confirms the pipeline end-to-end.

### Calibration

<!-- BEGIN:calibration -->
*Data will be populated after training run.*
<!-- END:calibration -->

All domains achieve near-nominal 90% coverage without requiring any scaling
adjustment. This is a direct result of the sparse calibration fix (B5.2) which
aligned the calibration procedure with the production operating point.

### Calibration Policy

When each calibration regime is applied, based on item count.

<!-- BEGIN:calibration_policy -->
*Data will be populated after training run.*
<!-- END:calibration_policy -->

### Domain-Balanced 20-Item Set

Top 4 items per domain by within-domain correlation. These are the items selected
by the `domain_balanced` strategy.

<!-- BEGIN:domain_balanced_items -->
*Data will be populated after training run.*
<!-- END:domain_balanced_items -->

This 20-item set differs from the Mini-IPIP: the domain-balanced set selects by
maximum within-domain correlation, while Mini-IPIP was designed for brevity and
broad coverage.

### Greedy Item Ranking (Cross-Domain Info Score)

Top 20 items by cross-domain information score. The domain distribution is heavily
skewed, which drives the domain starvation mechanism described above (see the
domain starvation table for counts).

<!-- BEGIN:greedy_ranking -->
*Data will be populated after training run.*
<!-- END:greedy_ranking -->

The first Intellect item falls just outside the top 20. Extraversion dominates
because its items show moderate cross-domain correlations (visible in the info
score columns above). These cross-loadings inflate composite info scores but are
too weak to reliably predict other domains.

### Mini-IPIP Reference

Donnellan et al. (2006) fixed 20-item mapping to IPIP-BFFM items.

<!-- BEGIN:mini_ipip -->
*Data will be populated after training run.*
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
