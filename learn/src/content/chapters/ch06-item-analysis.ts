import { officialDocs, repoFacts } from "../data";
import {
  abbr,
  callout,
  codeBlock,
  internalFiles,
  lead,
  list,
  mathBlock,
  paragraph,
  resourceList,
  section,
  table,
} from "../helpers";
import type { Chapter } from "../../types";

export const chapter06ItemAnalysis: Chapter = {
  slug: "06-stage-05-item-analysis",
  order: 6,
  title: "Stage 05 Item Analysis",
  kicker: "Item-domain correlations, information ranking, and the universal first item",
  summary:
    "Understand how the repo ranks items, why corrected item-total correlation matters, how the first item was chosen, and why that adaptive-looking artifact survived even after greedy selection was invalidated.",
  content: `
    ${section(
      "What Stage 05 Produces",
      `
        ${lead(
          `Stage 05 turns the training split into ${abbr("psychometric metadata", "Measurement-related information about items and scales, such as correlations, score structure, and response behavior.")}: item-domain correlations, cross-domain information scores, inter-item correlation summaries, a ranked item pool, and a selected universal first item.`,
        )}
        ${paragraph(
          `This stage matters for two reasons. First, it creates ranking artifacts used by later training, baseline, and simulation code. Second, it exposes a conceptual bridge between psychometrics and machine learning: each item carries domain membership, ${abbr("response distributions", "The pattern of how often respondents choose each response option for an item.")}, and varying predictive usefulness.`,
        )}
        ${internalFiles([
          "pipeline/05_compute_correlations.py",
          "data/processed/ext_est/item_correlations.json",
          "data/processed/ext_est/item_info.json",
          "data/processed/ext_est/first_item.json",
        ])}
      `,
    )}
    ${section(
      "Corrected Own-Domain Correlation",
      `
        ${paragraph(
          `For each item, the repo computes its correlation with each domain score. But for the item's own domain, it uses a ${abbr("corrected score", "A domain score recalculated after removing the item being evaluated so the item is not correlated with a scale that contains itself.")} that excludes the item itself.`,
        )}
        ${paragraph(
          `Why? Because otherwise the item would be correlated with a scale that literally contains itself. That inflates the relationship through ${abbr("part-whole contamination", "Artificially inflating a correlation because the item being tested is also part of the total score it is compared against.")}.`,
        )}
        ${mathBlock(
          "<mrow><mi>corrected</mi><mo>=</mo><mfrac><mrow><mi>sum of other domain items</mi></mrow><mrow><mi>count of other domain items</mi></mrow></mfrac></mrow>",
          "corrected domain score = mean of the other items in the domain",
        )}
        ${paragraph(
          "Correcting this way makes the own-domain correlation a better proxy for item quality rather than merely item inclusion.",
        )}
      `,
    )}
    ${section(
      "Cross-Domain Information",
      `
        ${paragraph(
          `The repo then computes a cross-domain information score as the sum of the absolute item correlations across all five domains. Despite the name, this includes the item's own domain too. So it is really a ${abbr("whole-profile utility score", "A heuristic ranking score for how useful an item seems for recovering the overall five-domain profile.")}, not a purely out-of-domain score.`,
        )}
        ${codeBlock(
          "cross_domain_info = |r_ext| + |r_agr| + |r_csn| + |r_est| + |r_opn|",
          "text",
        )}
        ${paragraph(
          `Don't confuse this with ${abbr("Fisher information", "A formal information quantity used in statistics and item response theory to describe how informative an observation is about an unknown parameter.")} in the ${abbr("IRT", "Item Response Theory: a psychometric framework that models responses in terms of latent traits and item parameters.")} sense; it's a heuristic ranking based on correlation structure. The distinction matters because the predecessor repo originally used this score as the engine of greedy adaptive item selection, and later evidence showed the heuristic can over-concentrate in highly connected domains.`,
        )}
      `,
    )}
    ${section(
      "Why ext3 Became The Universal First Item",
      `
        ${paragraph(
          `In the current repo, the selected first item is <code>${repoFacts.firstItemId}</code>: "${repoFacts.firstItemText}" with cross-domain information ≈ ${repoFacts.firstItemCrossDomainInfo.toFixed(4)}.`,
        )}
        ${table(
          ["Criterion", "How stage 05 uses it"],
          [
            ["High cross-domain information", "Primary ranking signal"],
            ["Reasonable response distribution", `Filters by ${abbr("entropy", "A measure of how spread out item responses are across answer choices.")} and ${abbr("skew", "A measure of how asymmetrically an item's responses are distributed.")}`],
            ["Own-domain strength", "Recorded so strong home-domain items are visible"],
            ["Reverse-key status", "Tracked for runtime correctness"],
          ],
        )}
        ${paragraph(
          "Historically, this made sense. If everyone starts with the same informative question, the model can condition later choices on an early response that carries signal about multiple domains.",
        )}
        ${callout(
          "warning",
          "What changed later",
          paragraph(
            "A good universal first item doesn't prove that greedy follow-up selection works. The repo eventually shows the opposite: you can choose a sensible opener and still harm the assessment if the later policy keeps reallocating toward the same dominant domains.",
          ),
        )}
      `,
    )}
    ${section(
      "Inter-Item Correlation And SEM Inputs",
      `
        ${paragraph(
          `Stage 05 also computes ${abbr("mean inter-item correlation", "The average correlation among items within the same domain, often used as a rough measure of internal consistency.")} within each domain, usually written as <code>r̄</code>. These values feed the later ${abbr("SEM", "Standard error of measurement: a precision estimate used here in stopping logic.")} calculations used in adaptive stopping logic.`,
        )}
        ${table(
          ["Domain", "Current mean inter-item r̄"],
          [
            ["Extraversion", repoFacts.interItemRBar.ext.toFixed(4)],
            ["Agreeableness", repoFacts.interItemRBar.agr.toFixed(4)],
            ["Conscientiousness", repoFacts.interItemRBar.csn.toFixed(4)],
            ["Emotional Stability", repoFacts.interItemRBar.est.toFixed(4)],
            ["Intellect / Openness", repoFacts.interItemRBar.opn.toFixed(4)],
          ],
        )}
        ${paragraph(
          "A higher mean inter-item correlation means items within that domain hang together more tightly, so the domain is easier to estimate precisely with fewer items. That's part of why some domains accumulate precision faster than others during simulation.",
        )}
      `,
    )}
    ${section(
      "The Deeper Lesson From Stage 05",
      `
        ${list([
          "Item ranking metadata is useful even if the original adaptive-selection dream fails.",
          "Psychometric item quality and ML utility overlap, but they are not identical.",
          `Cross-domain signal is real, but using it greedily for item choice can be harmful in a ${abbr("multidimensional test", "An assessment that tries to measure several distinct traits at once rather than just one trait.")}.`,
          `This stage produces artifacts that later steps rely on very heavily, so ${abbr("provenance locking", "Strict validation that later stages are using the exact expected artifact versions.").replace("provenance locking", "provenance locking")} around item_info is strict.`,
        ])}
        ${callout(
          "repo",
          "Strict dependency",
          paragraph(
            "Stages 06 and 07 refuse to proceed with stale or mismatched <code>item_info.json</code> when sparse objectives or sparsity augmentation depend on it. Item ranking changes can ripple into every later result.",
          ),
        )}
      `,
    )}
    ${section(
      "Further Reading",
      resourceList([
        {
          label: "IPIP item key",
          href: officialDocs.ipip,
          note: "Official item wording and domain key",
        },
        {
          label: "Mini-IPIP key",
          href: officialDocs.miniIpip,
          note: "Useful contrast with the custom domain-balanced ranking used here",
        },
      ]),
    )}`,
};
