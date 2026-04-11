import { officialDocs } from "../data";
import {
  abbr,
  callout,
  codeBlock,
  lead,
  list,
  paragraph,
  resourceList,
  section,
  table,
} from "../helpers";
import type { Chapter } from "../../types";

export const chapter07Xgboost: Chapter = {
  slug: "07-xgboost",
  order: 7,
  title: "XGBoost Fundamentals",
  kicker: "Why tree boosting is the ML core of this repo",
  summary:
    "Build an engineer-friendly mental model of trees, boosting, missing-value handling, quantile prediction, and the main alternatives that could have been used instead.",
  content: `
    ${section(
      "From Linear Models To Boosted Trees",
      `
        ${lead(
          `${abbr("XGBoost", "A high-performance implementation of gradient-boosted decision trees, widely used for tabular machine-learning problems.")} is a ${abbr("gradient-boosted decision tree", "A model built by adding many small decision trees in sequence, each correcting errors left by the earlier ones.")} library. It builds many small trees sequentially, where each new tree tries to fix the mistakes left behind by the earlier ones.`,
        )}
        ${paragraph(
          `A ${abbr("decision tree", "A model that makes predictions by routing each example through if/then splits on feature values.")} splits the data with questions like "is ext3 > 3.5?" and assigns predictions based on the path a row takes through those splits. When a feature is missing, XGBoost doesn't ask a separate literal "is this missing?" question; each split learns a default branch for missing values.`,
        )}
        ${paragraph(
          `Why does that fit this repo well? Because the inputs are ${abbr("tabular", "Organized as rows and columns, like a spreadsheet or database table.")}, moderately sized in feature count, full of ${abbr("nonlinear interactions", "Relationships where the combined effect of features is not just a simple straight-line sum.")}, and frequently sparse at ${abbr("inference time", "The moment when a trained model is used to make predictions on new data.")}. That's a very tree-friendly problem shape.`,
        )}
        ${codeBlock(
          `Prediction = tree_1(x) + learning_rate * tree_2(x) + learning_rate * tree_3(x) + ...`,
          "text",
        )}
      `,
    )}
    ${section(
      "Why XGBoost Fits This Problem",
      `
        ${table(
          ["Property of the problem", "Why XGBoost likes it"],
          [
            ["Tabular input (50 item columns)", "Tree ensembles are strongest on structured tabular data"],
            ["Nonlinear interactions", `Trees can model branch-dependent effects without manual ${abbr("feature engineering", "Hand-crafting extra input features or transformations before training.").replace("feature engineering", "feature engineering")}`],
            ["Missing items at inference", `XGBoost handles missing values natively via learned split directions`],
            ["Large training set", "XGBoost scales well and is mature operationally"],
            ["Need for quantiles", `Modern XGBoost supports ${abbr("quantile loss", "A loss function used to train lower, median, and upper quantile predictions instead of only mean predictions.")} directly`],
          ],
        )}
        ${callout(
          "why",
          "The single most important practical reason",
          paragraph(
            `This repo needs a model family that can treat unanswered items as missing rather than forcing crude ${abbr("imputation", "Filling in missing values with estimated replacements before training or inference.")}. XGBoost's native missing-value behavior is a major fit advantage here.`,
          ),
        )}
      `,
    )}
    ${section(
      "How Boosting Feels Intuitively",
      `
        ${paragraph(
          "Imagine starting with a rough predictor of Extraversion. It gets many rows about right but misses some patterns, especially when certain items are missing. A new tree is fit to reduce those remaining errors. Another corrects what's still left. Repeat enough times and you get a flexible but structured ensemble.",
        )}
        ${paragraph(
          `The ${abbr("bias-variance tradeoff", "The balance between a model being too simple to fit real patterns and too flexible to generalize well.")} is controlled through learning rate, tree depth, subsampling, column sampling, and ${abbr("regularization", "Penalties or constraints that keep a model from becoming too complex or unstable.")}. You'll see those exact knobs in stage 06 tuning.`,
        )}
      `,
    )}
    ${section(
      "Native Missing Values",
      `
        ${paragraph(
          `Most ML introductions hide missing data behind ${abbr("preprocessing", "Data-cleaning or transformation work done before model training or inference.")}. This repo doesn't. During training-time sparsity augmentation, unasked items are explicitly set to <code>NaN</code>, and XGBoost learns a default branch for missing values at each split.`,
        )}
        ${paragraph(
          `So the model can use patterns like "if ext3 is missing, follow the learned default branch; if ext3 is present and high, take the threshold branch." Better aligned with the actual deployment problem than pretending unanswered items are ordinary zeros or fully imputing them away.`,
        )}
        ${callout(
          "warning",
          "One historical cleanup",
          paragraph(
            `The predecessor repo at times discussed ${abbr("zero-encoding", "Representing missing answers as zeros instead of as true missing values.")} unanswered items. Now the path is cleaner: unanswered items are treated as missing via <code>NaN</code>, and the inference packages preserve that convention.`,
          ),
        )}
      `,
    )}
    ${section(
      "Why Not A Neural Network?",
      `
        ${paragraph(
          `A ${abbr("neural model", "A model based on layers of learned weights and nonlinear activations, such as a neural network.")} was an obvious alternative; the predecessor paper notes earlier work in that direction. But for this repo's specific problem, tree boosting has several advantages.`,
        )}
        ${table(
          ["Model family", "Pros here", "Cons here"],
          [
            ["XGBoost", "Excellent on tabular data, strong with missing values, operationally mature, good with moderate feature counts", "Less elegant for fully sequential adaptive policies than a custom neural policy network"],
            ["Ridge / linear regression", "Simple, interpretable, cheap", "Misses nonlinear interactions and complex missingness structure"],
            ["Within-domain-only models", "Psychometrically conservative", "Throws away cross-domain signal, which is one of the main gains in this repo"],
            ["Neural networks", "Flexible and fashionable, could model complex interactions", "Often unnecessary or inferior on tabular data, more tuning burden, less convenient ONNX path for this exact problem shape"],
            [`${abbr("IRT / CAT", "Item Response Theory and Computerized Adaptive Testing: psychometric frameworks for latent-trait modeling and adaptive question selection.")}`, "Psychometrically principled for adaptive testing", "Requires stronger modeling assumptions and a different item-response framework than the one this repo actually uses"],
          ],
        )}
        ${callout(
          "note",
          "Most honest summary",
          paragraph(
            "XGBoost is a strong pragmatic match for this tabular, sparse-input, score-recovery problem. It outperforms simpler baselines without the engineering burden of more ambitious alternatives.",
          ),
        )}
      `,
    )}
    ${section(
      "What To Watch For In The Code",
      `
        ${list([
          "<code>objective='reg:quantileerror'</code> in the training and tuning scripts",
          "<code>quantile_alpha</code> set to 0.05, 0.50, or 0.95",
          "Hyperparameters like <code>max_depth</code>, <code>learning_rate</code>, <code>subsample</code>, and <code>colsample_bytree</code>",
          `Explicit ${abbr("GPU", "Graphics Processing Unit: hardware that can accelerate some machine-learning training workloads.")} support in tune/train, but CPU-first evaluation later`,
          "NaN-masked feature matrices passed straight into the models",
        ])}
      `,
    )}
    ${section(
      "Further Reading",
      resourceList([
        {
          label: "XGBoost docs",
          href: officialDocs.xgboost,
          note: "Official documentation",
        },
        {
          label: "XGBoost paper",
          href: officialDocs.xgboostPaper,
          note: "Original system paper",
        },
      ]),
    )}`,
};
