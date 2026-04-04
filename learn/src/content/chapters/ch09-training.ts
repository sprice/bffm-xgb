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
          `The median model is the ${abbr("point prediction", "The single central predicted value, as opposed to an uncertainty interval.")}. The lower and upper quantile models define a ${abbr("nominal 90% interval", "An interval intended to contain the true value about 90% of the time before checking empirical calibration.")}. Later code converts these raw-score quantiles into percentile quantiles and then optionally rescales interval width for calibration.`,
        )}
      `,
    )}
    ${section(
      "Sparsity Augmentation Is The Center Of Gravity",
      `
        ${paragraph(
          `The most important training idea in this repo is ${abbr("sparsity augmentation", "Creating artificially masked training examples so the model learns to operate when many inputs are missing.")}. The original respondents answered all 50 items. Deployment often uses 20 items or fewer. So stage 07 creates masked copies of training rows where many items are hidden as <code>NaN</code>.`,
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
          `The imbalanced bucket matters because it exposes the model to the ugly patterns that greedy selection can create. That is a subtle but excellent engineering move: even though greedy selection is not the final recommendation, the model is taught not to panic when it sees badly unbalanced inputs.`,
        )}
        ${callout(
          "why",
          "Why no-sparsity ablations matter so much",
          paragraph(
            `The ${abbr("ablation", "A controlled variant used to test what happens when one important design choice is removed or changed.")} without sparsity augmentation shows that sparse training is not an optimization flourish. It is the main condition under which the model can outperform simple averaging on partial responses.`,
          ),
        )}
      `,
    )}
    ${section(
      "Split Before Augmentation",
      `
        ${paragraph(
          `One subtle but important implementation detail: the code splits an ${abbr("early-stopping evaluation slice", "A held-out subset used during training to decide when to stop adding trees before the model starts overfitting.")} before augmentation when using multi-pass sparsity. That prevents augmented copies of the same respondent from leaking across fit/eval boundaries.`,
        )}
        ${paragraph(
          "This matters because if one raw respondent is duplicated into multiple masked variants and those variants leak into both sides of a validation boundary, the reported performance becomes overly optimistic.",
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
          `Notice the modesty of this calibration scheme. It is not a giant ${abbr("post-hoc probabilistic recalibration", "A more elaborate procedure that adjusts predicted uncertainty after training so reported probabilities or intervals better match reality.")}. It is a conservative scaling rule meant to bring runtime intervals closer to nominal behavior.`,
        )}
      `,
    )}
    ${section(
      "Quality Gates",
      `
        ${paragraph(
          `The reference config encodes minimum acceptable performance thresholds for both full-50 validation and sparse-20 validation. Training can abort before saving if those ${abbr("quality gates", "Executable acceptance thresholds that a training run must satisfy before it is considered valid.")} fail.`,
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
          "This is another excellent engineering pattern: do not quietly ship a new training run just because the script finished. Make the acceptance criteria executable.",
        )}
      `,
    )}
    ${section(
      "Provenance Locks",
      `
        ${paragraph(
          `Stage 07 has a strict ${abbr("hyperparameter lock policy", "A rule for deciding when previously tuned model settings are still allowed to be reused.")}. Under <code>strict_data_hash</code>, it checks that the locked params file's provenance matches the current train hash, validation hash, split signature, and item_info hash. If not, it fails closed.`,
        )}
        ${paragraph(
          "That means no one can casually retune on one split and then silently retrain on another while pretending the params are still valid. The repo forces any such change to be explicit.",
        )}
        ${callout(
          "repo",
          "This is the main reproducibility lesson",
          paragraph(
            "A training run is not just code plus random seed. It is code plus exact data identity plus exact item-ranking identity plus exact hyperparameter identity. This repo makes that fact concrete.",
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
