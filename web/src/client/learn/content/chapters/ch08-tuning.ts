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
import { mountQuantileLossWidget } from "../../widgets";

export const chapter08Tuning: Chapter = {
  slug: "08-stage-06-tuning",
  order: 8,
  title: "Stage 06 Tuning",
  kicker: "How Optuna searches for XGBoost hyperparameters",
  summary:
    "Understand the search space, the deployment-aligned objective, why stage 06 tunes q50 first, and how the locked hyperparameter artifact becomes part of training provenance.",
  content: `
    ${section(
      "What Stage 06 Is Doing",
      `
        ${lead(
          `Stage 06 uses ${abbr("Optuna", "A hyperparameter-optimization framework used to search over model settings efficiently.")} to search over XGBoost hyperparameters. Its goal isn't abstract "best fit"—it's explicitly trying to find settings that perform well on the ${abbr("sparse operating regime", "The real deployment condition where many questionnaire items are missing and only a short form is available.")} the repo cares about.`,
        )}
        ${paragraph(
          `The tuning stage ${abbr("subsamples", "Uses only a subset of the available training rows to make tuning faster while preserving the overall data pattern.")} the training data for speed, precomputes sparsity masks so trials don't waste time remasking from scratch, trains median models at <code>q50</code> for each domain, and scores them on both sparse-20 and full-50 validation behavior.`,
        )}
        ${internalFiles([
          "pipeline/06_tune.py",
          "artifacts/tuned_params.json",
          "configs/reference.yaml",
          "lib/sparsity.py",
        ])}
      `,
    )}
    ${section(
      "The Search Space",
      `
        ${table(
          ["Hyperparameter", "What it controls", "Current checked-in tuned_params value"],
          [
            ["n_estimators", "How many boosting rounds", String(repoFacts.currentLockedParams.n_estimators)],
            ["max_depth", "How complex each tree can become", String(repoFacts.currentLockedParams.max_depth)],
            ["learning_rate", "How much each tree contributes", String(repoFacts.currentLockedParams.learning_rate)],
            ["reg_alpha", "L1 regularization", String(repoFacts.currentLockedParams.reg_alpha)],
            ["reg_lambda", "L2 regularization", String(repoFacts.currentLockedParams.reg_lambda)],
            ["subsample", "Fraction of rows used per tree", String(repoFacts.currentLockedParams.subsample)],
            ["colsample_bytree", "Fraction of columns used per tree", String(repoFacts.currentLockedParams.colsample_bytree)],
            ["min_child_weight", "Minimum effective weight before splitting", String(repoFacts.currentLockedParams.min_child_weight)],
          ],
        )}
        ${paragraph(
          `These values live in <code>artifacts/tuned_params.json</code>. Later stages consume that file as a ${abbr("lock artifact", "A saved artifact treated as an authoritative dependency that later stages must match exactly.")}; it's the authoritative source downstream training must match exactly.`,
        )}
      `,
    )}
    ${section(
      "The Objective Function",
      `
        ${paragraph(
          `The tuning objective is ${abbr("deployment-aligned", "Designed to reward performance in the real use case the model will face after shipping.")}. It primarily rewards sparse-20 performance while still penalizing bad full-50 behavior.`,
        )}
        ${codeBlock(
          `objective = 0.80 * mean_r_sparse20
         + 0.20 * mean_r_full
         - 2.0 * max(0, 0.85 - min_r_sparse20)
         - 1.0 * max(0, 0.95 - mean_r_full)`,
          "text",
        )}
        ${paragraph(
          "Read this slowly. Most of the reward comes from sparse-20 average correlation, with a smaller share from full-50 performance. Penalties fire if sparse-20 minimum domain performance drops too low or if full-50 average performance is too weak.",
        )}
        ${callout(
          "why",
          "Why this is better than a generic tuning objective",
          paragraph(
            `Tuning only on full-50 accuracy lets the search drift toward models that look wonderful when every item is present but fail the actual short-form use case. The explicit sparse objective forces the search to care about the ${abbr("deployment regime", "The actual pattern of inputs the model will see in real use, especially short partial questionnaires.").replace("deployment regime", "deployment regime")}.`,
          ),
        )}
      `,
    )}
    ${section(
      "Why Tune q50 First?",
      `
        ${paragraph(
          `Stage 06 tunes median models first rather than fitting a separate Optuna study for each quantile. The median prediction is the ${abbr("central point estimate", "The main single predicted value, as opposed to lower and upper uncertainty bounds.")} and already captures most of the structural difficulty of the mapping; one well-tuned configuration covers the dominant complexity.`,
        )}
        ${paragraph(
          "A full triple tuning process (q05, q50, q95) would multiply cost and complexity. Instead, the repo tunes a strong general XGBoost configuration and reuses it across quantiles in stage 07, changing only the quantile target parameter.",
        )}
        ${callout(
          "note",
          "Tradeoff",
          paragraph(
            "The same hyperparameters aren't guaranteed to be globally optimal for every quantile. The repo values a strong, reproducible, tractable training workflow over exhaustive per-quantile searches.",
          ),
        )}
      `,
    )}
    ${section(
      "Optuna And TPE",
      `
        ${paragraph(
          `Optuna is the optimization framework. Here it uses a ${abbr("TPE sampler", "Tree-structured Parzen Estimator: a search method that models promising and less-promising regions of the hyperparameter space.")}, which models promising and less-promising regions of the search space and chooses future trials accordingly—smarter than blind ${abbr("random search", "Trying randomly chosen hyperparameter settings without modeling which regions are promising.")} and cheaper than naive ${abbr("grid search", "Exhaustively trying combinations from a predefined grid of hyperparameter values.")}.`,
        )}
        ${paragraph(
          "It also handles parallel trials by dividing the available XGBoost thread budget across them. Uncontrolled parallelism can make heavy ML jobs unstable or inefficient, so the stage manages this explicitly.",
        )}
      `,
    )}
    ${section(
      "Pinball Loss Intuition",
      `
        ${paragraph(
          `${abbr("Quantile regression", "Regression that predicts selected quantiles such as q05, q50, and q95 instead of only a mean prediction.")} uses ${abbr("pinball loss", "The asymmetric loss function used to train quantile predictions.")}. A high quantile like q95 is punished heavily when it predicts too low; a low quantile like q05 is punished heavily when it predicts too high. That asymmetry is what makes quantile estimates behave like quantiles rather than ordinary mean predictions.`,
        )}
        <div id="quantile-loss-widget"></div>
      `,
    )}
    ${section(
      "What Leaves Stage 06",
      `
        ${list([
          "A locked hyperparameter JSON artifact",
          `Provenance fields tying the tuned params to train/val/test hashes, split signature, and ${abbr("item_info", "The stage-05 item-ranking metadata artifact used by later sparse-training and evaluation steps.")}`,
          `A ${abbr("fail-closed", "Designed to stop with an error instead of continuing silently when artifacts don't match expected conditions.").replace("fail-closed", "fail-closed")} dependency for later training`,
        ])}
        ${paragraph(
          "Provenance here is load-bearing. Stage 07 checks it to decide whether the current data and item metadata are allowed to reuse the tuned parameters. If the hashes don't match under strict lock policy, training stops.",
        )}
      `,
    )}
    ${section(
      "Further Reading",
      resourceList([
        {
          label: "Optuna docs",
          href: officialDocs.optuna,
          note: "Official tuning framework docs",
        },
        {
          label: "Optuna paper",
          href: officialDocs.optunaPaper,
          note: "Original paper on the framework",
        },
        {
          label: "Quantile regression paper",
          href: officialDocs.quantileRegressionPaper,
          note: "Classic regression quantiles paper",
        },
      ]),
    )}`,
  afterRender: () => {
    mountQuantileLossWidget();
  },
};
