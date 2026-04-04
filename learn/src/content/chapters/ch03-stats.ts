import { officialDocs, repoFacts } from "../data";
import {
  abbr,
  callout,
  codeBlock,
  lead,
  list,
  mathBlock,
  ordered,
  paragraph,
  resourceList,
  section,
  table,
} from "../helpers";
import type { Chapter } from "../../types";
import { mountZScoreWidget } from "../../widgets";

export const chapter03Stats: Chapter = {
  slug: "03-statistics",
  order: 3,
  title: "Statistics Foundations",
  kicker: "The minimum stats needed to read every pipeline stage comfortably",
  summary:
    "Build the stats vocabulary of the repo from zero: means, SDs, z-scores, percentiles, Pearson r, train/validation/test splits, overfitting, and bootstrap confidence intervals.",
  content: `
    ${section(
      "Means, Variation, And Why SD Matters",
      `
        ${lead(
          `Most pipeline artifacts are summaries of ${abbr("distributions", "How values are spread across a variable, including their center, spread, and shape.")}. If you cannot reason about center and spread, the rest of the repo stays opaque.`,
        )}
        ${paragraph(
          `A mean gives the center of a variable. A ${abbr("standard deviation", "A measure of spread showing how far values typically sit from the mean.")} tells you how spread out the values are around that mean. In this project, each domain's norm table stores both because percentile conversion depends on both.`,
        )}
        ${mathBlock(
          "<mrow><mi>z</mi><mo>=</mo><mfrac><mrow><mi>x</mi><mo>-</mo><mi>&mu;</mi></mrow><mi>&sigma;</mi></mfrac></mrow>",
          "z = (x - mean) / sd",
        )}
        ${paragraph(
          `If an Extraversion raw score is above the norm mean, its z-score is positive. If it is below, the z-score is negative. Percentiles are then computed from the ${abbr("standard normal cumulative distribution", "The curve used to convert a z-score into the proportion of the population expected to fall below it.")}.`,
        )}
        <div id="z-score-widget"></div>
      `,
    )}
    ${section(
      "Worked Example: Raw Score To Percentile",
      `
        ${paragraph(
          `Using the current Extraversion norms, mean ≈ ${repoFacts.norms.ext.mean.toFixed(4)} and SD ≈ ${repoFacts.norms.ext.sd.toFixed(4)}. Suppose a respondent's predicted raw score is 3.80.`,
        )}
        ${codeBlock(
          `z = (3.80 - 2.9138) / 0.9112 ≈ 0.972\npercentile ≈ 83.5`,
          "text",
        )}
        ${paragraph(
          "That means the respondent scored higher than roughly 83.5% of the reference sample on Extraversion. This is exactly the style of conversion used by <code>lib/scoring.py</code> and the inference packages.",
        )}
        ${callout(
          "note",
          "Important percentile nuance",
          paragraph(
            `In this repo, percentile conversion is a ${abbr("normal-CDF transform", "A conversion that maps z-scores to percentiles using the normal cumulative distribution function.")} using the locked mean and SD for each domain. That is not the same thing as an empirical percentile rank lookup from the raw sample. It is a modeling convenience and a stable runtime convention.`,
          ),
        )}
      `,
    )}
    ${section(
      "Pearson Correlation",
      `
        ${paragraph(
          `${abbr("Pearson correlation", "A statistic that measures linear association between two variables, usually written as r.")} measures linear association between two variables. Values near +1 mean strong positive alignment. Values near 0 mean weak linear relationship. Values near -1 mean strong inverse relationship.`,
        )}
        ${paragraph(
          "The repo uses Pearson r everywhere: item-domain correlations in stage 05, overall performance metrics in validation, baseline comparisons, and simulation outputs.",
        )}
        ${table(
          ["r value", "Very rough interpretation"],
          [
            ["0.00", "No linear relationship"],
            ["0.20", "Weak positive relationship"],
            ["0.50", "Moderate relationship"],
            ["0.80", "Strong relationship"],
            ["0.95+", "Near-perfect reconstruction"],
          ],
        )}
        ${callout(
          "note",
          "Important nuance",
          paragraph(
            `In this repo, high Pearson r means the short-form predictions track the full-scale criterion well. It does not automatically mean every domain is equally good or that absolute error is small in every case. That is why the repo also tracks ${abbr("MAE", "Mean absolute error: the average size of prediction errors without regard to direction.")}, ${abbr("RMSE", "Root mean squared error: an error metric that penalizes larger mistakes more strongly.")}, within-5, and coverage.`,
          ),
        )}
      `,
    )}
    ${section(
      "Train, Validation, And Test Splits",
      `
        ${paragraph(
          `A training set fits the model. A validation set helps choose ${abbr("hyperparameters", "Settings chosen before or around training, such as tree depth or learning rate, that control how the model learns.")} and assess whether the model generalizes while development is still happening. A test set is held out for later evaluation so you do not silently tune against the final scoreboard.`,
        )}
        ${table(
          ["Split", "Current row count", "What it is used for here"],
          [
            ["Train", repoFacts.trainRows.toLocaleString(), "Tuning and training, including sparsity augmentation"],
            ["Validation", repoFacts.valRows.toLocaleString(), "Hyperparameter search, quality gates, calibration estimation"],
            ["Test", repoFacts.testRows.toLocaleString(), "Post-training evaluation, baselines, simulation"],
          ],
        )}
        ${paragraph(
          `The repo goes one step further: it hash-locks these files and computes a <code>split_signature</code>. Later stages verify that the data they are evaluating matches the split that training actually used.`,
        )}
      `,
    )}
    ${section(
      "Overfitting And Distribution Shift",
      `
        ${paragraph(
          `${abbr("Overfitting", "When a model learns patterns that work on the training data but do not generalize well to new data.")} happens when a model learns peculiarities of the training data that do not generalize. ${abbr("Distribution shift", "A mismatch between the kinds of inputs seen during training and the kinds seen later during deployment.")} happens when the inputs seen during deployment differ from the inputs seen during training.`,
        )}
        ${paragraph(
          "This project is especially concerned with distribution shift. Training starts from complete 50-item response vectors. Deployment often uses 20-item partial vectors. If the model never trains on sparse patterns, it can look excellent on full data and still fail badly on sparse input.",
        )}
        ${callout(
          "warning",
          "One of the repo's strongest lessons",
          paragraph(
            `${abbr("Sparse-input augmentation", "Artificially creating training examples with many missing items so the model learns the deployment regime.")} is not a cosmetic trick. It is the main defense against train/deploy mismatch. Without it, the ML model can underperform simple averaging on the very sparse regime it was built to improve.`,
          ),
        )}
      `,
    )}
    ${section(
      "Bootstrap Confidence Intervals",
      `
        ${paragraph(
          `A ${abbr("bootstrap interval", "An uncertainty interval built by repeatedly resampling the observed data and recalculating the statistic of interest.")} estimates uncertainty by resampling the observed data many times. In this repo, respondent-level bootstrap resamples are used to quantify uncertainty around performance metrics like Pearson r and MAE.`,
        )}
        ${ordered([
          "Take the held-out respondents.",
          `Sample respondents ${abbr("with replacement", "Allowing the same respondent to be drawn multiple times in one bootstrap sample.").replace("with replacement", "with replacement")} to create a new bootstrap sample.`,
          "Recompute the metric on that sample.",
          "Repeat many times.",
          "Use the empirical distribution of those metric values to form a confidence interval.",
        ])}
        ${paragraph(
          `The key advantage is that bootstrap intervals are flexible. They work well for complicated metrics where deriving a neat ${abbr("closed-form standard error", "A directly derived mathematical formula for uncertainty, rather than one estimated by simulation or resampling.")} is awkward or undesirable.`,
        )}
      `,
    )}
    ${section(
      "Key Statistical Concepts For What Follows",
      `
        ${list([
          "A raw score is just a location on the questionnaire's 1-5 scale.",
          "A percentile is a norm-relative interpretation of that raw score.",
          "Pearson r tells you how well predictions track the criterion.",
          `Train / validation / test split protects honest ${abbr("model selection", "Choosing models or hyperparameters based on validation evidence without contaminating the final test evaluation.").replace("model selection", "model selection")}.`,
          "Bootstrap intervals tell you how stable reported metrics are.",
          "Distribution shift is the main statistical danger in sparse scoring.",
        ])}
      `,
    )}
    ${section(
      "Further Reading",
      resourceList([
        {
          label: "Python tutorial",
          href: officialDocs.python,
          note: "Official Python tutorial",
        },
        {
          label: "NumPy quickstart",
          href: officialDocs.numpy,
          note: "Official array-programming quickstart",
        },
        {
          label: "SciPy stats reference",
          href: officialDocs.scipy,
          note: "Official stats API reference",
        },
        {
          label: "scikit-learn train_test_split",
          href: officialDocs.sklearn,
          note: "Official split utility docs",
        },
      ]),
    )}`,
  afterRender: () => {
    mountZScoreWidget();
  },
};
