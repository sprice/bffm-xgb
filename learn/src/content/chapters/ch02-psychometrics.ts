import { officialDocs, repoFacts } from "../data";
import {
  abbr,
  callout,
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
import { mountReverseScoreWidget } from "../../widgets";

export const chapter02Psychometrics: Chapter = {
  slug: "02-psychometrics",
  order: 2,
  title: "Psychology And Psychometrics",
  kicker: "What the model is predicting, and why those targets mean something",
  summary:
    "Learn the Big Five, IPIP-BFFM item scoring, reverse-keying, norms, reliability, validity, and the measurement tradeoffs that make short-form scoring difficult.",
  content: `
    ${section(
      "The Five Domains",
      `
        ${lead(
          `The ${abbr("IPIP-BFFM inventory", "The IPIP Big Five Factor Markers questionnaire: a public-domain 50-item Big Five personality measure.")} measures five broad personality domains: Extraversion, Agreeableness, Conscientiousness, Emotional Stability, and Intellect/Openness.`,
        )}
        ${paragraph(
          `In this repo, each domain has 10 items, so the full questionnaire has 50 total. A domain score is the average of its 10 ${abbr("keyed items", "Questionnaire items after accounting for whether they should count positively or negatively toward the trait.")}. The model never predicts a hidden mystical quantity directly; it predicts the score that the full questionnaire would have produced under the inventory's scoring rules.`,
        )}
        ${table(
          ["Domain", "Plain-language intuition", "Typical item flavor"],
          [
            ["Extraversion", "Sociability, assertiveness, energetic engagement with people", '"I feel comfortable around people."'],
            ["Agreeableness", "Compassion, warmth, concern for others", '"I sympathize with others’ feelings."'],
            ["Conscientiousness", "Orderliness, dutifulness, self-discipline", '"I get chores done right away."'],
            ["Emotional Stability", "Lower negative emotional volatility", '"I am relaxed most of the time."'],
            ["Intellect/Openness", "Curiosity, imagination, interest in ideas", '"I have a vivid imagination."'],
          ],
        )}
      `,
    )}
    ${section(
      "Items, Keying, And Scale Scores",
      `
        ${paragraph(
          `Some IPIP items are positively keyed and some are negatively keyed. A positively keyed item points in the same direction as the trait. A negatively keyed item points in the opposite direction and must be ${abbr("reverse-scored", "Converted so higher numbers always mean more of the trait, even for negatively worded items.")} before aggregation.`,
        )}
        ${mathBlock(
          "<mrow><mi>reverse</mi><mo>=</mo><mn>6</mn><mo>-</mo><mi>raw</mi></mrow>",
          "reverse = 6 - raw",
        )}
        ${paragraph(
          `On a 1-5 ${abbr("Likert scale", "An ordered response scale such as strongly disagree to strongly agree, usually encoded numerically.")}, reverse-scoring turns 1 into 5, 2 into 4, 3 into 3, 4 into 2, and 5 into 1. Without this step, the domain mean would mix opposite directions and become ${abbr("psychometrically", "In a way that concerns measurement quality and interpretability.")} incoherent.`,
        )}
        <div id="reverse-score-widget"></div>
        ${callout(
          "repo",
          "Where this lives in the repo",
          list([
            "<code>lib/constants.py</code> defines the reverse-keyed item IDs.",
            "<code>pipeline/02_load_sqlite.py</code> applies reverse-scoring before writing SQLite.",
            "<code>web/src/server/reverse-score.ts</code> mirrors the same rule for runtime input handling.",
          ]),
        )}
      `,
    )}
    ${section(
      "What A Domain Score Is",
      `
        ${paragraph(
          `After reverse-scoring, each domain score is a mean. If a respondent answered the 10 Extraversion items with keyed values that average to 3.8, then <code>ext_score = 3.8</code>. This is the ${abbr("raw-scale target", "The score on the original questionnaire scale before any normalization or percentile conversion.")}. The model is trained on these raw 1-5 domain means and later converts them to ${abbr("percentiles", "Rank-like interpretations that place a score relative to a reference population.")}.`,
        )}
        ${paragraph(
          `That distinction matters. Raw scores live on the measurement scale of the inventory. Percentiles live on the ${abbr("reference-population scale", "A score interpretation relative to the distribution of scores in a reference sample.")}. Raw-score modeling plus later normalization lets the repo keep one consistent scoring definition and one consistent ${abbr("norm table", "A lookup of mean and standard deviation used to interpret scores relative to a population.")}.`,
        )}
        ${callout(
          "why",
          "Why predict raw score instead of percentile directly?",
          paragraph(
            `Percentiles depend on ${abbr("norms", "Reference statistics such as mean and standard deviation used to interpret scores.")}. Raw scores are the immediate psychometric output of the instrument. Predicting raw scores and converting afterward keeps training aligned with the questionnaire itself while letting runtime reuse the same norm transform everywhere.`,
          ),
        )}
      `,
    )}
    ${section(
      "Reliability, Validity, And Why Short Forms Are Hard",
      `
        ${paragraph(
          `${abbr("Reliability", "How consistently a measure captures the same trait rather than fluctuating due to noise.")} asks whether a measure is consistent. ${abbr("Validity", "Whether a score supports the interpretation you want to make from it.")} asks whether the score means what you think it means. These are related but different. A score can be very consistent and still measure the wrong thing.`,
        )}
        ${paragraph(
          "Short forms are hard because each removed item cuts away information. If you go from 10 items per domain to 4 items per domain, reliability usually drops. The question becomes whether smarter item choice and smarter scoring can recover enough of what was lost.",
        )}
        ${table(
          ["Concept", "Simple meaning", "Why the repo cares"],
          [
            ["Internal consistency", "Do items in the same domain move together?", "Used through mean inter-item correlation and SEM reasoning."],
            ["Reliability", "How much score variance is signal rather than noise?", "Short forms lose reliability unless item selection is careful."],
            ["Validity", "Does the score support the intended interpretation?", `A crucial caveat: this repo focuses on ${abbr("score recovery", "Reconstructing the full questionnaire score from partial answers.")}, not full ${abbr("construct-validity", "Evidence that a measure truly reflects the psychological construct it claims to measure.").replace("construct-validity", "construct-validity")} replacement.`],
            ["Norms", "Reference distribution for interpreting a raw score", "Needed to turn a raw predicted score into a percentile."],
          ],
        )}
        ${callout(
          "warning",
          "Important limitation",
          paragraph(
            `This codebase mainly demonstrates reconstruction of full-scale scores. That is not the same thing as proving equal ${abbr("external validity", "Evidence that a score relates appropriately to outside outcomes or behaviors.")}, equal ${abbr("factor structure", "The pattern of underlying latent dimensions inferred from how items relate to each other.")}, or equal clinical interpretability. This guide returns to that distinction throughout.`,
          ),
        )}
      `,
    )}
    ${section(
      "Norms And Percentiles",
      `
        ${paragraph(
          `A raw score becomes interpretable by comparing it to a reference population. The repo computes norms from the full cleaned dataset and uses them to convert a raw score into a ${abbr("z-score", "A standardized score showing how many standard deviations a value is above or below the mean.")} and then a percentile.`,
        )}
        ${table(
          ["Domain", "Current mean", "Current SD"],
          [
            ["Extraversion", repoFacts.norms.ext.mean.toFixed(4), repoFacts.norms.ext.sd.toFixed(4)],
            ["Agreeableness", repoFacts.norms.agr.mean.toFixed(4), repoFacts.norms.agr.sd.toFixed(4)],
            ["Conscientiousness", repoFacts.norms.csn.mean.toFixed(4), repoFacts.norms.csn.sd.toFixed(4)],
            ["Emotional Stability", repoFacts.norms.est.mean.toFixed(4), repoFacts.norms.est.sd.toFixed(4)],
            ["Intellect / Openness", repoFacts.norms.opn.mean.toFixed(4), repoFacts.norms.opn.sd.toFixed(4)],
          ],
        )}
        ${paragraph(
          `The current repo locks these values in <code>artifacts/ipip_bffm_norms.json</code>. Later stages refer to a stable <code>data_snapshot_id</code> derived from the norms file hash. That means the norms are not just descriptive statistics; they are part of the ${abbr("run identity", "The exact artifact fingerprint that distinguishes one reproducible pipeline state from another.")}.`,
        )}
      `,
    )}
    ${section(
      "Key Psychometric Concepts To Watch For",
      `
        ${list([
          "Reverse-keying protects score direction.",
          "Domain means define the raw-score targets.",
          "Norms define the percentile interpretation.",
          `Inter-item correlation and ${abbr("SEM", "Standard error of measurement: an estimate of how much observed scores vary because of measurement error.").replace("SEM", "SEM")} justify assessment length.`,
          `Content balancing matters because the instrument is ${abbr("multidimensional", "Measuring several related but distinct traits rather than only one single trait.")}, not one-dimensional.`,
        ])}
        ${internalFiles([
          "pipeline/02_load_sqlite.py",
          "pipeline/03_compute_norms.py",
          "pipeline/05_compute_correlations.py",
          "lib/scoring.py",
          "artifacts/ipip_bffm_norms.json",
          "artifacts/mini_ipip_mapping.json",
        ])}
      `,
    )}
    ${section(
      "Further Reading",
      resourceList([
        {
          label: "IPIP Big Five 50-item key",
          href: officialDocs.ipip,
          note: "Official item key, domain mapping, and item wording",
        },
        {
          label: "Mini-IPIP item key",
          href: officialDocs.miniIpip,
          note: "Official short-form mapping used for baseline comparison",
        },
        {
          label: "Mini-IPIP paper",
          href: officialDocs.miniIpipPaper,
          note: "Primary source on the 20-item short form",
        },
      ]),
    )}`,
  afterRender: () => {
    mountReverseScoreWidget();
  },
};
