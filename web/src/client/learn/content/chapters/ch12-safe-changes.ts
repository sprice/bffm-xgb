import { officialDocs } from "../data";
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

export const chapter12SafeChanges: Chapter = {
  slug: "12-safe-changes",
  order: 12,
  title: "Safe Changes",
  kicker: "How the design prevents invalid results",
  summary:
    "See how the repo's provenance system prevents stale artifacts from contaminating results: what gets invalidated when something changes, what the guards protect against, and where the biggest hidden failure modes live.",
  content: `
    ${section(
      "The Core Rule",
      `
        ${lead(
          `The dangerous mistake in this repo: making a change that silently invalidates earlier ${abbr("artifacts", "Saved outputs such as split files, tuned parameters, models, and reports produced by earlier stages.")} while leaving them in place, so later results look legitimate.`,
        )}
        ${paragraph(
          `The repo's ${abbr("provenance system", "The hashes and metadata checks that verify where artifacts came from and whether they still match the expected data/code state.")} exists to prevent exactly that. Artifact boundaries encoded in the codebase are the first line of defense against self-deception.`,
        )}
      `,
    )}
    ${section(
      "What Gets Invalidated When Something Changes",
      `
        ${table(
          ["If this changes...", "Minimum rerun", "Why"],
          [
            ["Raw-data cleaning or reverse-keying", "Stages 02 onward", "Raw scores, norms, splits, and training targets all change"],
            ["Norm computation", "Stages 03 onward", "Percentile interpretation and data snapshot identity change"],
            ["Split logic", "Stages 04 onward", "Train/val/test identities and split signatures change"],
            ["Item ranking logic", "Stages 05 onward", `Sparse masking, tuning objective, training, baselines, and simulation all depend on ${abbr("item_info", "The stage-05 artifact containing item-ranking metadata used by later sparse-training and evaluation steps.")}`],
            ["Hyperparameter search strategy", "Stages 06 onward", "Locked params artifact changes"],
            ["Sparsity augmentation recipe", "Stage 07 onward, plus evaluation", "Sparse behavior claims must be revalidated"],
            ["Inference/export format", "Stage 11 onward, plus runtime tests", "Serving bundle changes even if training doesn't"],
          ],
        )}
      `,
    )}
    ${section(
      "What The Provenance System Guards Against",
      `
        ${list([
          "Using stale tuned parameters with a new split or new item_info artifact",
          "Changing reverse-key definitions in one runtime path but not another",
          `Evaluating a model bundle against the wrong ${abbr("data regime", "The exact input condition a result assumes, such as full-50 responses, sparse-20 responses, or a different split strategy.")}`,
          "Treating old generated artifacts as if they came from the current code",
          "Making a sparse-training change and then looking only at full-50 validation",
          `Confusing score reconstruction gains with full ${abbr("psychometric validity", "Evidence that a score supports the intended psychological interpretation, not just that it numerically matches another score.")} gains`,
        ])}
        ${callout(
          "warning",
          "A useful mental test",
          paragraph(
            "For any conceptual change, the question is: which hashes should now be different? If the answer isn't clear, the change boundary isn't yet well understood.",
          ),
        )}
      `,
    )}
    ${section(
      "How The Repo Supports Safe Experimentation",
      `
        ${paragraph(
          "Variants and artifacts let you scope experiments tightly without contaminating the reference pipeline.",
        )}
        ${table(
          ["Goal", "How the repo handles it"],
          [
            ["New split strategy", "Duplicate stage-04 output directory, regenerate stage 05 item metadata for that regime, then train/evaluate as a separate variant"],
            ["New sparsity bucket", `Add a new ${abbr("config variant", "A separate configuration file representing one controlled experimental setup.")}, rerun 07-10, compare against reference in research summary`],
            ["Changed ONNX export behavior", "Leave training artifacts alone, rerun 11, then run inference and web tests against the new export"],
            ["New baseline idea", "Work inside stage 09 first; training artifacts stay untouched until the evaluation question is worth pursuing"],
          ],
        )}
      `,
    )}
    ${section(
      "How The Provenance Guards Work Together",
      `
        ${list([
          "Split signatures verify train/val/test identity.",
          "item_info hashes verify item-ranking identity.",
          "Hyperparameter lock policies verify that tuned-params reuse is legitimate.",
          `Model/data pairing checks prevent evaluating the wrong model against the wrong ${abbr("split regime", "The exact train/validation/test split scheme and data preparation setup used for a run.")}.`,
          "Quality gates stop low-quality models before they're quietly treated as valid artifacts.",
        ])}
        ${internalFiles([
          "lib/provenance.py",
          "lib/provenance_checks.py",
          "lib/item_info.py",
          "pipeline/07_train.py",
          "pipeline/08_validate.py",
          "Makefile",
        ])}
      `,
    )}
    ${section(
      "Open Questions And Future Directions",
      `
        ${paragraph(
          `Reasonable future directions include richer ${abbr("calibration", "Adjusting predicted intervals or uncertainty so they better match observed behavior on held-out data.")} methods, more explicit psychometric validation beyond score recovery, comparison against stronger ${abbr("IRT/CAT baselines", "Adaptive-testing baselines built from Item Response Theory and Computerized Adaptive Testing methods.")}, or a more specialized sequential selection policy that enforces domain balance structurally (rather than hoping a generic utility score behaves well).`,
        )}
        ${paragraph(
          "And the discipline is worth preserving regardless: explicit artifacts, strong baselines, and the willingness to invalidate attractive ideas when the held-out evidence says no.",
        )}
      `,
    )}
    ${section(
      "What To Remember",
      `
        ${list([
          "Each numbered stage consumes specific artifacts and produces new ones.",
          "Sparse-input augmentation is the essential ingredient for short-form scoring.",
          "Raw scores and percentiles are different things with different uses.",
          "Greedy adaptive selection failed because it starved some domains.",
          "ONNX is the deployment boundary between research and runtime.",
          "The provenance system enforces that changes propagate correctly through the pipeline.",
        ])}
      `,
    )}
    ${section(
      "Further Reading",
      resourceList([
        {
          label: "Python tutorial",
          href: officialDocs.python,
          note: "Useful if you want to deepen comfort with the implementation language",
        },
        {
          label: "XGBoost docs",
          href: officialDocs.xgboost,
          note: "Most important external tech doc for future experiments",
        },
        {
          label: "ONNX Runtime docs",
          href: officialDocs.onnxruntime,
          note: "Useful for runtime/export work",
        },
      ]),
    )}`,
};
