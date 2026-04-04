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
import { mountSemWidget } from "../../widgets";

export const chapter10Evaluation: Chapter = {
  slug: "10-stages-08-to-10",
  order: 10,
  title: "Stages 08 To 10",
  kicker: "Validation, baselines, simulation, and the failure of greedy adaptive selection",
  summary:
    "See how the repo tests its claims: held-out validation, baseline strategy comparisons, SEM stopping simulation, and the empirical reason the original adaptive idea was rejected.",
  content: `
    ${section(
      "Stage 08: Validate The Trained Models",
      `
        ${lead(
          `Stage 08 answers the first sober question after training: do the trained models actually generalize on ${abbr("held-out respondents", "People kept out of training so performance can be evaluated on genuinely unseen data.")}?`,
        )}
        ${paragraph(
          "Validation reports both full-50 and sparse-20 behavior. Full-50 tells you whether the model family can reconstruct nearly complete information. Sparse-20 tells you whether the deployment regime still looks good after missingness enters the picture.",
        )}
        ${table(
          ["Regime", "Current r", "Current MAE", "Current coverage"],
          [
            ["Full 50", repoFacts.validation.full50R.toFixed(4), repoFacts.validation.full50Mae.toFixed(2), `${(repoFacts.validation.full50Coverage * 100).toFixed(1)}%`],
            ["Sparse 20", repoFacts.validation.sparse20R.toFixed(4), repoFacts.validation.sparse20Mae.toFixed(2), `${(repoFacts.validation.sparse20Coverage * 100).toFixed(1)}%`],
          ],
        )}
      `,
    )}
    ${section(
      "Stage 09: Baselines Matter More Than People Think",
      `
        ${paragraph(
          `Stage 09 compares multiple ${abbr("item-selection strategies", "Rules for choosing which questionnaire items to ask or keep in a short form.")} and multiple ${abbr("scoring baselines", "Simpler comparison methods used to judge whether the main model is actually adding value.")}; each baseline is strong enough to be a credible alternative.`,
        )}
        ${table(
          ["20-item strategy", "Current Pearson r", "Interpretation"],
          [
            ["Domain-balanced", repoFacts.baselineK20.domainBalancedR.toFixed(4), "Best current practical strategy"],
            ["Domain-constrained adaptive", repoFacts.baselineK20.constrainedAdaptiveR.toFixed(4), "Adaptive but forced to respect domain coverage early"],
            ["Mini-IPIP standalone", repoFacts.baselineK20.miniIpipStandaloneR.toFixed(4), "Simple averaging with Mini-IPIP-specific norms"],
            ["Adaptive top-k", repoFacts.baselineK20.adaptiveTopKR.toFixed(4), "Historically important failure case"],
          ],
        )}
        ${paragraph(
          "Central empirical finding: unconstrained greedy adaptive selection underperforms simpler balanced approaches.",
        )}
        ${callout(
          "note",
          "Important result-family distinction",
          paragraph(
            `In the current repo, the <code>mini_ipip</code> row in stage 09 is a standalone Mini-IPIP scoring baseline—not Mini-IPIP scored by the XGBoost model. The ML-scored Mini-IPIP result appears in the separate ${abbr("ML-vs-averaging comparison", "An artifact that compares learned model scoring against simple averaging on the same selected item sets.")}; there the checked-in numbers are roughly <code>r = 0.9172</code> for ML versus <code>r = 0.9064</code> for averaging.`,
          ),
        )}
      `,
    )}
    ${section(
      "Why Greedy Adaptive Selection Failed",
      `
        ${paragraph(
          `The old intuition was attractive: rank items by global predictive utility, ask the best ones first. But the actual result was ${abbr("domain starvation", "A failure mode where some personality domains get too few items because the selection rule keeps favoring other domains.")}. Greedy selection kept choosing highly cross-correlated items (especially from Extraversion and Emotional Stability) and delayed or omitted items from more psychometrically distinct domains like Intellect/Openness.`,
        )}
        <div class="bar-list">
          <div class="bar-row"><span>Extraversion</span><div class="bar-track"><div class="bar-fill" style="width:45%"></div></div><strong>9 items</strong></div>
          <div class="bar-row"><span>Emotional Stability</span><div class="bar-track"><div class="bar-fill" style="width:30%"></div></div><strong>6 items</strong></div>
          <div class="bar-row"><span>Agreeableness</span><div class="bar-track"><div class="bar-fill" style="width:15%"></div></div><strong>3 items</strong></div>
          <div class="bar-row"><span>Conscientiousness</span><div class="bar-track"><div class="bar-fill" style="width:10%"></div></div><strong>2 items</strong></div>
          <div class="bar-row"><span>Intellect / Openness</span><div class="bar-track"><div class="bar-fill" style="width:0%"></div></div><strong>0 items</strong></div>
        </div>
        ${callout(
          "warning",
          "Core mechanism",
          paragraph(
            `The greedy criterion optimized total cross-domain utility rather than adequate coverage of each domain. In a ${abbr("multidimensional instrument", "A questionnaire meant to measure multiple distinct traits rather than only one overall trait.")}, adequate per-domain coverage is usually what you need for usable score recovery.`,
          ),
        )}
      `,
    )}
    ${section(
      "Stage 10: Simulation And SEM-Based Stopping",
      `
        ${paragraph(
          `Stage 10 simulates assessment behavior on held-out respondents. Stopping logic aligns with ${abbr("classical test theory", "A traditional psychometric framework that analyzes observed scores in terms of true score plus measurement error.")}-style ${abbr("SEM", "Standard error of measurement: an estimate of how precise a score is.")} reasoning rather than pure greedy information gain.`,
        )}
        ${codeBlock(
          `alpha_k = (k * r_bar) / (1 + (k - 1) * r_bar)
SEM = SD_domain * sqrt(1 - alpha_k)`,
          "text",
        )}
        ${paragraph(
          "Earlier work experimented with SEM thresholds like 0.42 and 0.38 during exploration. Currently the canonical operating point uses a 0.45 threshold plus a minimum of 4 items per domain; in practice that produces a fixed 20-item pattern in the reference simulation.",
        )}
        ${callout(
          "note",
          "Scope of this SEM formula",
          paragraph(
            `A heuristic built from mean inter-item correlation and a ${abbr("Cronbach-alpha", "A classical internal-consistency reliability estimate based on how strongly items in a scale relate to each other.")}-style reliability approximation. Useful for this repo's stopping logic; other definitions of SEM exist, and ${abbr("IRT", "Item Response Theory: a psychometric framework that models item responses via latent traits and item parameters.")} information calculations work differently.`,
          ),
        )}
        <div id="sem-widget"></div>
        ${paragraph(
          `The current simulation still uses <code>selection_strategy = correlation_ranked</code>, so it isn't literally "adaptive stopping only." But under the checked-in operating point, domain coverage and the SEM target constrain the policy so heavily that the resulting assessment behaves much more like a fixed balanced 20-item path than like the original free-form greedy adaptive vision.`,
        )}
      `,
    )}
    ${section(
      "Worked Example: Why 4 Extraversion Items Clear The Threshold",
      `
        ${paragraph(
          `Using current Extraversion values, r̄ ≈ ${repoFacts.interItemRBar.ext.toFixed(4)} and SD ≈ ${repoFacts.norms.ext.sd.toFixed(4)}.`,
        )}
        ${codeBlock(
          `k = 3  -> alpha ≈ 0.726, SEM ≈ 0.477
k = 4  -> alpha ≈ 0.779, SEM ≈ 0.428`,
          "text",
        )}
        ${paragraph(
          `So moving from 3 to 4 Extraversion items pushes SEM below a 0.45 target. Stopping is framed in terms of ${abbr("measurement precision", "How narrowly and reliably a score estimates the trait rather than fluctuating because of measurement error.")}; the item count follows from the precision requirement.`,
        )}
      `,
    )}
    ${section(
      "Current Simulation Outcome",
      `
        ${table(
          ["Metric", "Current value"],
          [
            ["Simulation Pearson r", repoFacts.simulation.overallR.toFixed(4)],
            ["Simulation MAE", repoFacts.simulation.overallMae.toFixed(2)],
            ["Simulation 90% coverage", `${(repoFacts.simulation.overallCoverage * 100).toFixed(1)}%`],
            ["Mean items per domain", String(repoFacts.simulation.meanItemsPerDomain)],
            ["Mean total items", String(repoFacts.simulation.meanItemsTotal)],
          ],
        )}
        ${paragraph(
          `Adaptivity in the current repo survives mainly in stopping logic and ${abbr("runtime flexibility", "The ability of the shipped inference system to score different partial-response patterns at prediction time.")}. Greedy next-item selection is gone.`,
        )}
      `,
    )}
    ${section(
      "The Most Important Lesson From Evaluation",
      `
        ${list([
          "Held-out validation is the real test of generalization.",
          "Strong baselines keep ML claims honest.",
          "Scoring quality and item-selection quality are separate questions.",
          "Adaptive assessment claims need psychometric stopping logic to remain principled.",
          "Invalidated ideas (like greedy adaptive selection) are part of the repo's knowledge.",
        ])}
        ${internalFiles([
          "pipeline/08_validate.py",
          "pipeline/09_baselines.py",
          "pipeline/10_simulate.py",
          "artifacts/research_summary.json",
        ])}
      `,
    )}
    ${section(
      "Further Reading",
      resourceList([
        {
          label: "Open Psychometrics data",
          href: officialDocs.openPsychometricsData,
          note: "Primary data source behind the held-out evaluations",
        },
        {
          label: "Mini-IPIP paper",
          href: officialDocs.miniIpipPaper,
          note: "Published short-form baseline the repo tries to beat",
        },
      ]),
    )}`,
  afterRender: () => {
    mountSemWidget();
  },
};
