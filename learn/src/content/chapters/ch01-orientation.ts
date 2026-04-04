import { officialDocs, repoFacts } from "../data";
import {
  abbr,
  callout,
  exampleCard,
  internalFiles,
  lead,
  list,
  paragraph,
  resourceList,
  section,
  table,
} from "../helpers";
import type { Chapter } from "../../types";

export const chapter01Orientation: Chapter = {
  slug: "01-orientation",
  order: 1,
  title: "Orientation",
  kicker: "What this repo is actually trying to prove",
  summary:
    "Start with the research question, the current outputs, the historical pivot from adaptive item selection to better scoring, and the shape of the full learning path.",
  content: `
    ${section(
      "The Big Picture",
      `
        ${lead(
          `This repository is not a generic ${abbr("machine-learning", "A field of computing where models learn patterns from data to make predictions.")} demo. It is a ${abbr("research pipeline", "A staged, reproducible workflow for turning data into results and artifacts.")} and ${abbr("runtime product", "The packaged system that is actually used after training, rather than the training code itself.")} for predicting ${abbr("Big Five personality scores", "Scores for the five broad personality domains: Extraversion, Agreeableness, Conscientiousness, Emotional Stability, and Intellect or Openness.")} from partial answers to the 50-item ${abbr("IPIP-BFFM inventory", "The IPIP Big Five Factor Markers questionnaire: a public-domain 50-item Big Five personality measure.")}.`,
        )}
        ${paragraph(
          `At the highest level, the code asks a practical question: if a person answers only part of a long personality questionnaire, can a model recover the score they would have received from all 50 items? The current answer in this repo is yes, but only under a specific design: ${abbr("domain-balanced short forms", "Short questionnaires that keep coverage spread across all five Big Five domains instead of over-sampling one or two domains.")}, ${abbr("sparse-input training", "Training the model on examples where many items are deliberately hidden or missing so it learns to handle partial responses.")}, ${abbr("quantile regression", "A regression approach that predicts multiple quantiles, such as lower, median, and upper estimates, instead of only one average prediction.")}, and strong ${abbr("provenance checks", "Hash and metadata checks that verify where an artifact came from and whether it still matches the data and code that produced it.")}.`,
        )}
        ${paragraph(
          `A second important point: the codebase changed direction over time. The predecessor repo began with an ${abbr("adaptive-testing", "A testing approach where later questions depend on earlier answers.")} idea where the next item was chosen greedily by ${abbr("cross-domain utility", "A heuristic score for how useful one item seems for predicting scores across multiple personality domains.")}. That idea was tested, then invalidated. The current repo keeps the useful pieces of that work, especially ${abbr("sparse-input scoring", "Using a model to score a person even when many questionnaire items were never answered.")} and ${abbr("uncertainty estimation", "Producing intervals or ranges that show how uncertain a prediction is, not just a single point estimate.")}, while moving away from greedy adaptive item selection.`,
        )}
        <div class="figure-strip">
          <div class="mini-stat"><strong>${repoFacts.totalValidRespondents.toLocaleString()}</strong><span>valid respondents in the current cleaned dataset</span></div>
          <div class="mini-stat"><strong>15</strong><span>XGBoost models: 5 domains × 3 quantiles</span></div>
          <div class="mini-stat"><strong>${repoFacts.validation.full50R.toFixed(4)}</strong><span>full-50 validation Pearson r</span></div>
          <div class="mini-stat"><strong>${repoFacts.validation.sparse20R.toFixed(4)}</strong><span>sparse-20 validation Pearson r</span></div>
          <div class="mini-stat"><strong>${repoFacts.baselineK20.domainBalancedR.toFixed(4)}</strong><span>domain-balanced 20-item baseline r</span></div>
        </div>
        ${callout(
          "why",
          "The central claim",
          paragraph(
            `The novel claim is not “machine learning can do personality assessment.” The sharper claim is “for ${abbr("fixed short forms", "A predetermined shorter questionnaire where everyone receives the same subset of items.")}, better scoring can recover full-scale scores more accurately than ${abbr("simple averaging", "Scoring by averaging the answered items directly, without a learned predictive model.")}, as long as training matches sparse deployment.”`,
          ),
        )}
      `,
    )}
    ${section(
      "What You Will Learn In Order",
      `
        ${table(
          ["Module", "What it teaches", "Why it matters here"],
          [
            [
              `Psychology + ${abbr("psychometrics", "The field concerned with how psychological traits are measured, scored, and evaluated for reliability and validity.")}`,
              `What the Big Five are, how items become scale scores, and why reliability and ${abbr("norms", "Reference-population statistics used to interpret a raw score, often as a percentile.")} matter.`,
              "Without this, the model is just predicting mysterious numbers.",
            ],
            [
              "Statistics",
              `Means, ${abbr("SDs", "Standard deviations: measures of how spread out values are around their mean.")}, z-scores, percentiles, correlation, ${abbr("bootstrap", "A resampling method used to estimate uncertainty by repeatedly sampling with replacement.")}, ${abbr("train/val/test logic", "The practice of separating data into training, validation, and test sets so model selection and final evaluation stay honest.")}.`,
              "These concepts appear in nearly every stage of the pipeline.",
            ],
            [
              "Codebase map",
              `How the numbered scripts, ${abbr("artifacts", "Named output files produced by pipeline stages, such as norms, splits, models, and reports.")}, configs, runtime packages, and web app fit together.`,
              `You need the ${abbr("artifact graph", "The chain of which artifacts are produced by each stage and consumed by later stages.")} before you can safely change anything.`,
            ],
            [
              `${abbr("XGBoost", "A high-performance gradient-boosted decision tree library that works especially well on tabular data.")} + ${abbr("quantile regression", "Regression that predicts specific quantiles such as q05, q50, and q95 rather than only a mean.")}`,
              `How tree boosting works, why it handles sparse ${abbr("tabular inputs", "Data organized as rows and columns, like a spreadsheet or database table.")} well, and how ${abbr("prediction intervals", "Ranges intended to contain the true value with a chosen level of confidence, such as 90%.")} are produced.`,
              "This is the ML core of the project.",
            ],
            [
              "Evaluation + simulation",
              `How the repo validates claims, where the adaptive idea failed, and why ${abbr("domain-balanced selection", "Choosing items so each Big Five domain keeps enough representation instead of letting one domain dominate.")} won.`,
              "This is where design decisions are justified empirically.",
            ],
          ],
        )}
        <div class="comparison-grid">
          ${exampleCard(
            "Question the repo is built around",
            "<p>If a person answers only some items, can we reconstruct their full-scale Big Five percentile scores accurately enough to be useful?</p>",
          )}
          ${exampleCard(
            "Question the repo is not trying to answer",
            `<p>Does this system discover the true ${abbr("latent structure", "The underlying unobserved traits or factors assumed to generate observed questionnaire responses.")} of personality better than classical psychometrics? That would require a different validation program.</p>`,
          )}
        </div>
      `,
    )}
    ${section(
      "The Historical Pivot You Should Keep In Mind",
      `
        ${paragraph(
          `The old adaptive framing was: choose a universal first item, then repeatedly choose the next most globally informative item. That feels elegant. It also failed in practice. Greedy ${abbr("cross-domain utility", "A whole-profile item-ranking heuristic based on how strongly an item correlates across the five domains.")} concentrated too many items in Extraversion and Emotional Stability and starved Intellect/Openness.`,
        )}
        ${paragraph(
          `The current codebase still carries adaptive-assessment DNA. Stage 05 still computes a universal first item. Stage 10 still simulates ${abbr("SEM-based stopping", "Stopping rules based on the standard error of measurement: a classical-test-theory estimate of score precision.")}. The difference is that the final recommended scoring path is not greedy adaptive selection. It is strong cross-domain scoring from ${abbr("domain-balanced partial responses", "Incomplete answer sets that still preserve enough coverage across all five domains to recover the full-scale score well.")}.`,
        )}
        ${callout(
          "warning",
          "Do not confuse model quality with selection quality",
          paragraph(
            `One of the most instructive findings in this project is that the model can be excellent while the ${abbr("item-selection policy", "The rule that decides which question to ask next or which subset of items to keep.")} is bad. All strategies converge near full information. The failure happens in which items are chosen early, not in whether ${abbr("XGBoost", "A gradient-boosted tree model used here as the main predictor.")} can fit the mapping at all.`,
          ),
        )}
      `,
    )}
    ${section(
      "Codebase Anchor Points",
      `
        ${internalFiles([
          "pipeline/01_download.py through pipeline/13_upload_hf.py",
          "lib/sparsity.py",
          "lib/provenance.py",
          "lib/provenance_checks.py",
          "python/inference.py",
          "typescript/inference.ts",
          "web/src/server/predictor.ts",
          "configs/reference.yaml",
          "artifacts/tuned_params.json",
          "artifacts/research_summary.json",
        ])}
        ${paragraph(
          `If you only remember one structural idea from this chapter, remember this: the numbered scripts produce named ${abbr("artifacts", "Saved outputs like split files, item metadata, tuned parameters, model bundles, and reports.")}, and later scripts consume those artifacts under ${abbr("hash locks", "Checks that compare file fingerprints so later stages can verify they are using the exact expected inputs.")}. This repo is intentionally built so that stale or mismatched data should ${abbr("fail closed", "Stop with an error instead of continuing silently with invalid assumptions.")}.`,
        )}
      `,
    )}
    ${section(
      "Recommended Reading Strategy",
      `
        ${list([
          "Read linearly the first time. The chapters are ordered to front-load psychology and statistics before ML details.",
          "When a chapter references a script, open that file in parallel and scan the top-level flow rather than reading every line immediately.",
          `Treat every stage as an ${abbr("artifact transformer", "A step that reads one set of saved files, transforms them, and writes a new set of saved outputs.").replace("artifact transformer", "artifact transformer")}: ask what it reads, what it writes, and what assumptions it locks in for the next stage.`,
          "Pay attention to historical invalidations. Those are some of the most valuable lessons in the codebase.",
        ])}
      `,
    )}
    ${section(
      "Further Reading",
      resourceList([
        {
          label: "IPIP Big Five Factor Markers",
          href: officialDocs.ipip,
          note: "Official IPIP item key and construct background",
        },
        {
          label: "Open Psychometrics raw datasets",
          href: officialDocs.openPsychometricsData,
          note: "Current source data used by the pipeline",
        },
        {
          label: "Current repo README",
          href: "https://github.com/sprice/bffm-xgb",
          note: "Repository-level overview and install/run guide",
        },
      ]),
    )}`,
};
