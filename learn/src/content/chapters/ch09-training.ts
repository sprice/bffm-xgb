import { officialDocs, repoFacts } from "../data";
import {
  abbr,
  callout,
  codeBlock,
  internalFiles,
  lead,
  list,
  paragraph,
  resourceList,
  section,
  table,
} from "../helpers";
import type { Chapter } from "../../types";

export const chapter09Training: Chapter = {
  slug: "09-stage-07-training",
  order: 9,
  title: "Stage 07 Training",
  kicker: "Sparsity augmentation, calibration, quality gates, and provenance locks",
  summary:
    "Learn how the 15 models are trained, why sparsity augmentation is essential, how calibration factors are derived, and how the repo prevents stale hyperparameter or split reuse.",
  content: `
    ${section(
      "What Stage 07 Produces",
      `
        ${lead(
          `Stage 07 trains the full ${abbr("model family", "The complete set of related models used together at runtime, here one per domain and quantile.")}: five personality domains times three quantiles, for 15 XGBoost models total.`,
        )}
        ${paragraph(
          `It also computes validation metrics, sparse-20 gate metrics, calibration parameters, ${abbr("cross-validation robustness", "Performance estimates from repeated train/validation splits used to check whether results are stable rather than lucky.")} estimates, and a detailed <code>training_report.json</code> that becomes a provenance anchor for later stages.`,
        )}
        ${internalFiles([
          "pipeline/07_train.py",
          "models/reference/training_report.json",
          "models/reference/calibration_params_sparse_20_balanced.json",
          "lib/sparsity.py",
          "lib/provenance_checks.py",
        ])}
      `,
    )}
    ${section(
      "The 15-Model Bundle",
      `
        ${table(
          ["Domain", "Quantiles", "Purpose"],
          [
            ["ext", "q05, q50, q95", "Lower, median, and upper Extraversion estimates"],
            ["agr", "q05, q50, q95", "Agreeableness interval + point estimate"],
            ["csn", "q05, q50, q95", "Conscientiousness interval + point estimate"],
            ["est", "q05, q50, q95", "Emotional Stability interval + point estimate"],
            ["opn", "q05, q50, q95", "Intellect / Openness interval + point estimate"],
          ],
        )}
        ${paragraph(
          `The median model is the ${abbr("point prediction", "The single central predicted value, as opposed to an uncertainty interval.")}. Lower and upper quantile models together define a ${abbr("nominal 90% interval", "An interval intended to contain the true value about 90% of the time before checking empirical calibration.")}. Later code converts these raw-score quantiles into percentile quantiles and optionally rescales interval width for calibration.`,
        )}
      `,
    )}
    ${section(
      "Sparsity Augmentation Is The Center Of Gravity",
      `
        ${paragraph(
          `The most important training idea in this repo is ${abbr("sparsity augmentation", "Creating artificially masked training examples so the model learns to operate when many inputs are missing.")}. Original respondents answered all 50 items, but deployment often uses 20 or fewer. So stage 07 creates masked copies of training rows where many items are hidden as <code>NaN</code>.`,
        )}
        ${table(
          ["Bucket", "Share", "Pattern"],
          [
            ["Balanced sparse", "40%", "Keep 10-20 items with at least some domain coverage"],
            ["Mini-IPIP injection", "10%", "Exactly the Mini-IPIP subset"],
            ["Transition range", "20%", "Keep 21-35 items"],
            ["Light sparsity", "15%", "Keep 36-50 items"],
            ["Imbalanced patterns", "15%", "Allow 0-item domains and skewed domain allocation"],
          ],
        )}
        ${paragraph(
          `The imbalanced bucket matters because it exposes the model to the ugly patterns greedy selection can create. Even though greedy selection isn't the final recommendation, the model learns not to panic when it sees badly unbalanced inputs; it's seen worse during training.`,
        )}
        ${callout(
          "why",
          "Why no-sparsity ablations matter so much",
          paragraph(
            `The ${abbr("ablation", "A controlled variant used to test what happens when one important design choice is removed or changed.")} without sparsity augmentation makes this clear: sparse training is the main condition under which the model can outperform simple averaging on partial responses.`,
          ),
        )}
      `,
    )}
    ${section(
      "Split Before Augmentation",
      `
        ${paragraph(
          `Here's a detail that's easy to miss: the code splits an ${abbr("early-stopping evaluation slice", "A held-out subset used during training to decide when to stop adding trees before the model starts overfitting.")} before augmentation when using multi-pass sparsity. Splitting first prevents augmented copies of the same respondent from leaking across fit/eval boundaries.`,
        )}
        ${paragraph(
          "Why? If one raw respondent is duplicated into multiple masked variants and those variants leak into both sides of a validation boundary, reported performance becomes overly optimistic.",
        )}
      `,
    )}
    ${section(
      "Calibration Factors",
      `
        ${paragraph(
          `After fitting q05, q50, and q95 models, stage 07 checks ${abbr("empirical interval coverage", "The proportion of true values that actually fall inside the predicted interval on held-out data.")}. If the nominal 90% interval is too narrow or too wide, it derives a simple scale factor to adjust reported percentile interval width later.`,
        )}
        ${codeBlock(
          `if coverage < 0.85: scale = 0.90 / max(coverage, 0.5)
elif coverage > 0.95: scale = 0.90 / coverage
else: scale = 1.0`,
          "text",
        )}
        ${paragraph(
          `Notice the modesty here. Rather than a full ${abbr("post-hoc probabilistic recalibration", "A more elaborate procedure that adjusts predicted uncertainty after training so reported probabilities or intervals better match reality.")}, this is a conservative scaling rule; it just nudges runtime intervals closer to nominal behavior.`,
        )}
      `,
    )}
    ${section(
      "Quality Gates",
      `
        ${paragraph(
          `Reference config encodes minimum acceptable performance thresholds for both full-50 and sparse-20 validation. Training can abort before saving if those ${abbr("quality gates", "Executable acceptance thresholds that a training run must satisfy before it is considered valid.")} fail.`,
        )}
        ${table(
          ["Gate", "Reference threshold"],
          [
            ["Overall full-50 r", "≥ 0.90"],
            ["Overall full-50 90% coverage", "≥ 0.88"],
            ["Per-domain full-50 r", "≥ 0.90"],
            ["Per-domain full-50 90% coverage", "≥ 0.88"],
            ["Overall sparse-20 r", "≥ 0.90"],
            ["Overall sparse-20 90% coverage", "≥ 0.88"],
            ["Per-domain sparse-20 r", "≥ 0.85"],
            ["Per-domain sparse-20 90% coverage", "≥ 0.84"],
          ],
        )}
        ${paragraph(
          "A script finishing doesn't mean the run is good. Quality gates make acceptance criteria executable, so a bad run can't quietly ship.",
        )}
      `,
    )}
    ${section(
      "Provenance Locks",
      `
        ${paragraph(
          `Stage 07 enforces a strict ${abbr("hyperparameter lock policy", "A rule for deciding when previously tuned model settings are still allowed to be reused.")}. Under <code>strict_data_hash</code>, it checks that the locked params file's provenance matches the current train hash, validation hash, split signature, and item_info hash. Any mismatch and it fails closed.`,
        )}
        ${paragraph(
          "So nobody can casually retune on one split and silently retrain on another while pretending the params are still valid. Any such change has to be explicit.",
        )}
        ${callout(
          "repo",
          "Reproducibility means identity, not just code",
          paragraph(
            "A training run is code plus exact data identity, plus exact item-ranking identity, plus exact hyperparameter identity. This repo makes that fact concrete.",
          ),
        )}
      `,
    )}
    ${section(
      "Current Reference Outcomes",
      `
        ${table(
          ["Metric", "Current value"],
          [
            ["Full-50 validation r", repoFacts.validation.full50R.toFixed(4)],
            ["Full-50 validation MAE", repoFacts.validation.full50Mae.toFixed(2)],
            ["Full-50 coverage", `${(repoFacts.validation.full50Coverage * 100).toFixed(1)}%`],
            ["Sparse-20 validation r", repoFacts.validation.sparse20R.toFixed(4)],
            ["Sparse-20 validation MAE", repoFacts.validation.sparse20Mae.toFixed(2)],
            ["Sparse-20 coverage", `${(repoFacts.validation.sparse20Coverage * 100).toFixed(1)}%`],
          ],
        )}
      `,
    )}
    ${section(
      "Further Reading",
      resourceList([
        {
          label: "XGBoost docs",
          href: officialDocs.xgboost,
          note: "Official docs for the underlying model family",
        },
        {
          label: "Optuna docs",
          href: officialDocs.optuna,
          note: "Stage 07 consumes the lock artifact stage 06 produced",
        },
      ]),
    )}`,
};
