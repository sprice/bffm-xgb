import { officialDocs, repoFacts } from "../data";
import {
  abbr,
  callout,
  internalFiles,
  lead,
  list,
  paragraph,
  resourceList,
  section,
  table,
} from "../helpers";
import type { Chapter } from "../../types";

export const chapter05DataPipeline: Chapter = {
  slug: "05-stages-01-to-04",
  order: 5,
  title: "Stages 01 To 04",
  kicker: "Data download, cleaning, norms, and split creation",
  summary:
    "Trace the front half of the pipeline from raw OSPP download to split parquets, including IPC filtering, reverse-scoring, train-only norm computation, and the single plain random split.",
  content: `
    ${section(
      "Stage 01: Download And Verify",
      `
        ${lead(
          `The pipeline begins by downloading the ${abbr("Open Psychometrics Project", "A public site that hosts openly available psychology questionnaires and raw datasets.")} IPIP-FFM ZIP and verifying it before trusting it.`,
        )}
        ${paragraph(
          `Research pipelines quietly become unreliable when upstream data can change or when downloaded payloads aren't checked. Stage 01 records expected ${abbr("hashes", "File fingerprints used to verify that a file is exactly the expected content.")} for both the ZIP and the extracted CSV, anchoring ${abbr("data-engineering hygiene", "Practical safeguards that keep a data pipeline reliable, reproducible, and resistant to accidental drift.")} at the very first step.`,
        )}
        ${callout(
          "why",
          "Why this matters",
          paragraph(
            "If the raw dataset changes, everything downstream changes: norms, splits, model fit, and published claims. Hash verification prevents accidental drift from masquerading as a normal rerun.",
          ),
        )}
      `,
    )}
    ${section(
      "Stage 02: Load Into SQLite And Clean",
      `
        ${paragraph(
          `Stage 02 loads the raw ${abbr("tab-separated CSV", "A plain-text table file where columns are separated by tabs instead of commas.")} into ${abbr("pandas", "Python's main DataFrame library for loading, transforming, and analyzing tabular data.")}, filters invalid rows, applies reverse-scoring, computes domain means, and writes a local ${abbr("SQLite", "A lightweight embedded relational database stored in a single file.")} database. SQLite here serves as a deterministic, queryable checkpoint between raw text data and split-ready tabular data.`,
        )}
        ${table(
          ["Cleaning step", "What it does", "Why it exists"],
          [
            ["Valid 1-5 item responses", "Drops rows with missing or malformed item values", "All downstream scoring assumes a full keyed response matrix"],
            ["IPC == 1", `Keeps only rows whose ${abbr("IP", "Internet Protocol address: a rough network identifier used here as part of a duplicate-response filter.").replace("IP", "IP")} appears once in the raw data`, "Removes repeated submissions and some low-cleanliness cases"],
            ["Reverse-scoring", "Transforms negatively keyed items with 6 - raw", "Aligns all items to the same trait direction"],
            ["Domain means", "Computes ext_score, agr_score, csn_score, est_score, opn_score", "Creates the training targets used later"],
          ],
        )}
        ${paragraph(
          `The current cleaned dataset ends up with ${repoFacts.totalValidRespondents.toLocaleString()} valid respondents, substantially smaller than the older predecessor repo's sample. Tighter cleaning (especially the IP uniqueness rule) accounts for the difference.`,
        )}
        ${internalFiles([
          "pipeline/01_download.py",
          "pipeline/02_load_sqlite.py",
          "data/processed/ipip_bffm.db",
          "data/processed/load_metadata.json",
        ])}
      `,
    )}
    ${section(
      "Stage 03: Compute Locked Norms",
      `
        ${paragraph(
          `Stage 03 computes the norm tables from the training split of <code>canonical_v1</code> only — the held-out validation and test rows are excluded, so they cannot leak into the percentile targets. Norms serve as a ${abbr("reference distribution", "The population distribution used to interpret scores, such as by turning raw scores into percentiles.")} for score interpretation, fit on the same training rows the model learns from.`,
        )}
        ${paragraph(
          `The stage writes a lock file and ${abbr("sidecar metadata", "A companion metadata file that travels alongside a main artifact and records its provenance or validation details.")} metadata. Later stages use a stable <code>data_snapshot_id</code> derived from the norms file hash, effectively promoting the norm artifact into a ${abbr("run identity anchor", "A fingerprinted artifact that helps define exactly which reproducible pipeline state a later run belongs to.")}.`,
        )}
        ${callout(
          "note",
          "Transfer-learning idea",
          paragraph(
            "Any pipeline that depends on a derived reference artifact benefits from hashing and locking it; you can then reason about reproducibility without depending on wall-clock dates or human memory.",
          ),
        )}
      `,
    )}
    ${section(
      "Stage 04: Prepare Split Data",
      `
        ${paragraph(
          `Stage 04 loads the cleaned SQLite rows, computes percentile columns from ${abbr("train-only norms", "Mean and standard deviation computed on the training split alone, so held-out validation/test rows never influence the percentile targets.")}, splits the rows at random into train/validation/test, and writes the ${abbr("parquets", "Apache Parquet files: compact columnar data files commonly used for analytics and ML pipelines.")} plus <code>split_metadata.json</code>.`,
        )}
        ${table(
          ["Current split", "Rows"],
          [
            ["Train", repoFacts.trainRows.toLocaleString()],
            ["Validation", repoFacts.valRows.toLocaleString()],
            ["Test", repoFacts.testRows.toLocaleString()],
          ],
        )}
        ${paragraph(
          `The split is a single plain random 70/15/15 partition — the canonical <code>canonical_v1</code> split — keyed on respondent id with a fixed seed, so it is fully reproducible. There is no ${abbr("stratification", "Splitting within balanced score bins so each split has a similar score distribution; unnecessary at this dataset's scale.")}: with 600,000+ respondents a random split is already balanced on every domain.`,
        )}
      `,
    )}
    ${section(
      "Why A Plain Random Split?",
      `
        ${paragraph(
          "The predecessor repo used stratified splits — an <code>ext-est</code> scheme (Extraversion × Emotional-Stability quintiles) and a three-domain <code>ext-est-opn</code> variant. Both were dropped in favor of a single plain random split.",
        )}
        ${paragraph(
          `The deciding evidence: an ${abbr("ablation", "A controlled variant that changes one design choice to test whether it really matters.")} comparing the two stratified schemes moved Openness recovery by a negligible amount — indistinguishable from noise. At this dataset's scale, how you stratify the split barely changes the numbers.`,
        )}
        ${callout(
          "why",
          "Why not stratify at all?",
          paragraph(
            "With 600,000+ respondents, the law of large numbers makes a random 70/15/15 split almost perfectly balanced on every domain on its own. Stratification is a small-sample safeguard; here it only adds complexity. It also avoids two subtle smells the stratified approach had: stratifying on the very scores being predicted, and computing the quintile bin edges over the full dataset (including held-out rows) before splitting.",
          ),
        )}
      `,
    )}
    ${section(
      "What Leaves Stage 04",
      `
        ${list([
          "<code>train.parquet</code>, <code>val.parquet</code>, <code>test.parquet</code>",
          "<code>split_metadata.json</code> with file hashes, split signature, and distribution checks",
          "Per-domain percentile columns",
          "Quintile columns such as <code>ext_q</code>, <code>est_q</code>, and optionally <code>opn_q</code>",
        ])}
        ${paragraph(
          `At this stage, raw data engineering becomes ${abbr("model-ready", "Prepared in the exact row/column structure expected by later machine-learning stages.").replace("model-ready", "model-ready")} research data. Everything downstream assumes the split files and metadata are trustworthy; the provenance helpers enforce that assumption.`,
        )}
      `,
    )}
    ${section(
      "Further Reading",
      resourceList([
        {
          label: "pandas getting started",
          href: officialDocs.pandas,
          note: "Official DataFrame guide used heavily in stages 02-05",
        },
        {
          label: "SQLite docs",
          href: officialDocs.sqlite,
          note: "Official SQLite documentation",
        },
        {
          label: "Apache Parquet docs",
          href: officialDocs.parquet,
          note: "Columnar file format used for train/val/test artifacts",
        },
      ]),
    )}`,
};
