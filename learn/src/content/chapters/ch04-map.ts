import {
  abbr,
  callout,
  internalFiles,
  lead,
  list,
  paragraph,
  section,
  table,
} from "../helpers";
import type { Chapter } from "../../types";

export const chapter04Map: Chapter = {
  slug: "04-codebase-map",
  order: 4,
  title: "Codebase Map",
  kicker: "How the repository is organized, and how artifacts move through it",
  summary:
    "Build a mental model of the repo as an artifact pipeline: numbered Python stages, shared libraries, runtime packages, web app, infrastructure, and the variant/evaluation layout.",
  content: `
    ${section(
      "Think In Artifacts, Not Just Scripts",
      `
        ${lead(
          `The fastest way to understand this repo is to stop thinking of it as “a lot of Python files” and start thinking of it as a graph of ${abbr("artifacts", "Named output files such as norms, split files, model bundles, and reports produced by pipeline stages.")} with ${abbr("hash-locked", "Protected by file fingerprints so later stages can verify they are consuming the exact expected inputs.")} handoffs.`,
        )}
        ${paragraph(
          `Every numbered pipeline script reads one set of files, computes a transformation, and writes another set of files. Later scripts trust those outputs only if ${abbr("provenance", "Metadata describing where an artifact came from and what exact inputs and code produced it.")}, file hashes, and ${abbr("split signatures", "Identifiers that encode the exact train/validation/test split identity for a run.")} match. That design is one of the strongest engineering choices in the project.`,
        )}
        ${callout(
          "why",
          "Why this matters for learning",
          paragraph(
            `If you know what each stage consumes and emits, you can understand the system even before reading the internal function details. If you skip the ${abbr("artifact graph", "The chain of which outputs are produced by one stage and consumed by later stages.")}, the later provenance checks feel arbitrary.`,
          ),
        )}
      `,
    )}
    ${section(
      "Top-Level Layout",
      `
        ${table(
          ["Path", "Role"],
          [
            ["pipeline/", "The numbered end-to-end research pipeline"],
            ["lib/", "Shared Python utilities: constants, sparsity, scoring, provenance, bootstrap"],
            ["configs/", "Variant configs for reference and ablation runs"],
            ["data/processed/", "SQLite DB, split parquets, item metadata, first-item outputs"],
            ["models/", "Trained model bundles and training reports"],
            ["artifacts/", "Norms, tuned params, cross-variant summaries, published metadata"],
            ["output/", "Final ONNX export bundles used by inference and Hugging Face"],
            ["python/ and typescript/", "Standalone inference packages"],
            ["web/", "The deployment-facing web app and API server"],
            ["infra/", `${abbr("Terraform", "Infrastructure-as-code tooling for defining and creating cloud resources reproducibly.")} for CPU and GPU ${abbr("AWS", "Amazon Web Services, the cloud platform used for remote compute in this repo.")} instances`],
          ],
        )}
        ${internalFiles([
          "Makefile",
          "scripts/run-pipeline.sh",
          "configs/reference.yaml",
          "docs/pipeline.md",
          "docs/research.md",
          "docs/inference.md",
          "docs/infrastructure.md",
        ])}
      `,
    )}
    ${section(
      "The 13 Pipeline Stages",
      `
        ${table(
          ["Stage", "Script", "Main output"],
          [
            ["01", "download", "raw ZIP + extracted CSV"],
            ["02", "load SQLite", "cleaned SQLite response table"],
            ["03", "compute norms", "locked norm JSON"],
            ["04", "prepare data", "train / val / test parquets + split metadata"],
            ["05", "compute correlations", "item correlations, item_info, first_item"],
            ["06", "tune", "locked XGBoost hyperparameters"],
            ["07", "train", "15 model files + training report + calibration"],
            ["08", "validate", "held-out validation metrics with bootstrap"],
            ["09", "baselines", "comparison of item-selection strategies"],
            ["10", "simulate", `adaptive / ${abbr("SEM", "Standard error of measurement: an estimate of score precision used in stopping logic.").replace("SEM", "SEM")} stopping simulation outputs`],
            ["11", "export ONNX", `merged ${abbr("ONNX", "An open model format used to package trained models for portable inference across languages and runtimes.").replace("ONNX", "ONNX")} model + config + provenance`],
            ["12", "generate figures", "publication charts"],
            ["13", "upload HF", `published model bundle on ${abbr("Hugging Face", "A platform for sharing machine-learning models, datasets, and demos.")}`],
          ],
        )}
        ${paragraph(
          "Stages 01-07 build the model. Stages 08-10 test the research claims. Stages 11-13 package and publish the runtime artifact. The course will follow roughly the same order.",
        )}
      `,
    )}
    ${section(
      "Variants And Why They Exist",
      `
        ${paragraph(
          `The repo does not train one model only. It trains a reference variant and several ${abbr("ablations", "Controlled variants where one important design choice is removed or changed to test whether it really matters.")}. This is a research pattern: you do not merely show that your preferred setup works; you test whether the important result disappears when a key design choice is removed.`,
        )}
        ${table(
          ["Config", "Meaning"],
          [
            ["reference.yaml", "Focused sparsity + Mini-IPIP injection + imbalanced patterns"],
            ["ablation_none.yaml", "No sparsity augmentation"],
            ["ablation_focused.yaml", "Focused sparsity without imbalanced patterns"],
            ["ablation_stratified.yaml", "Alternative split regime using ext-est-opn"],
          ],
        )}
        ${callout(
          "note",
          "Core research logic",
          paragraph(
            `Ablations tell you which design choice is carrying the result. In this project, the major empirical winner is the presence of sparse training at all. Fine details of the ${abbr("augmentation recipe", "The exact mix of artificially masked training examples used to teach the model the sparse deployment regime.")} matter less than simply matching train-time inputs to deploy-time sparsity.`,
          ),
        )}
      `,
    )}
    ${section(
      "Orchestration",
      `
        ${paragraph(
          `The Makefile and <code>scripts/run-pipeline.sh</code> are the ${abbr("orchestration layer", "The commands and scripts that coordinate how the whole pipeline is executed across stages and machines.")}. They define how stages run locally, how reference-only or full multi-variant runs are executed, and how CPU/GPU remote workflows hand off work between machines.`,
        )}
        ${list([
          "Use the numbered scripts to understand the science.",
          "Use the Makefile to understand the operational workflow.",
          "Use the docs folder to understand the intended usage and deployment story.",
        ])}
      `,
    )}
    ${section(
      "The Mental Model To Carry Forward",
      `
        ${list([
          "The numbered pipeline is the spine of the repo.",
          "Shared logic lives in lib/ and is reused across stages.",
          "Configs define variants rather than changing code paths ad hoc.",
          `Inference packages and the web app sit downstream of exported ${abbr("ONNX artifacts", "Portable saved model bundles in ONNX format that the runtime packages load for inference.")}.`,
          "Infrastructure exists to make heavy tuning and training practical, not because the runtime requires cloud complexity.",
        ])}
      `,
    )}`,
};
