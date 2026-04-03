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
  kicker: "How to modify the system without fooling yourself",
  summary:
    "Learn the safest change workflows in the repo: what to rerun, what artifacts become stale, what provenance checks protect you from, and where the biggest hidden failure modes live.",
  content: `
    ${section(
      "The Core Rule",
      `
        ${lead(
          `In this repo, the dangerous mistake is not merely a bug. It is making a change that silently invalidates your earlier ${abbr("artifacts", "Saved outputs such as split files, tuned parameters, models, and reports produced by earlier stages.")} while leaving them in place so later results look legitimate.`,
        )}
        ${paragraph(
          `The repo's ${abbr("provenance system", "The hashes and metadata checks that verify where artifacts came from and whether they still match the expected data/code state.")} exists to prevent exactly that. Your job as a maintainer is to respect the artifact boundaries the code already encodes.`,
        )}
      `,
    )}
    ${section(
      "What To Rerun When You Change Something",
      `
        ${table(
          ["If you change...", "Minimum rerun", "Why"],
          [
            ["Raw-data cleaning or reverse-keying", "Stages 02 onward", "Raw scores, norms, splits, and training targets all change"],
            ["Norm computation", "Stages 03 onward", "Percentile interpretation and data snapshot identity change"],
            ["Split logic", "Stages 04 onward", "Train/val/test identities and split signatures change"],
            ["Item ranking logic", "Stages 05 onward", `Sparse masking, tuning objective, training, baselines, and simulation all depend on ${abbr("item_info", "The stage-05 artifact containing item-ranking metadata used by later sparse-training and evaluation steps.")}`],
            ["Hyperparameter search strategy", "Stages 06 onward", "Locked params artifact changes"],
            ["Sparsity augmentation recipe", "Stage 07 onward, plus evaluation", "Sparse behavior claims must be revalidated"],
            ["Inference/export format", "Stage 11 onward, plus runtime tests", "Serving bundle changes even if training does not"],
          ],
        )}
      `,
    )}
    ${section(
      "High-Risk Failure Modes",
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
          "One practical habit",
          paragraph(
            "Whenever you make a conceptual change, ask which hashes should now be different. If you cannot answer that, you probably do not yet understand the change boundary well enough.",
          ),
        )}
      `,
    )}
    ${section(
      "Recommended Change Workflows",
      `
        ${paragraph(
          "If you want to experiment safely, make the scope explicit and keep the artifact chain short.",
        )}
        ${table(
          ["Goal", "Safe workflow"],
          [
            ["Try a new split strategy", "Duplicate stage-04 output directory, regenerate stage 05 item metadata for that regime, then train/evaluate as a separate variant"],
            ["Try a new sparsity bucket", `Add a new ${abbr("config variant", "A separate configuration file representing one controlled experimental setup.")}, rerun 07-10, compare against reference in research summary`],
            ["Change ONNX export behavior", "Leave training artifacts alone, rerun 11, then run inference and web tests against the new export"],
            ["Investigate a baseline idea", "Work inside stage 09 first; do not modify training until you know the evaluation question is worth it"],
          ],
        )}
      `,
    )}
    ${section(
      "How The Existing Guards Help You",
      `
        ${list([
          "Split signatures verify train/val/test identity.",
          "item_info hashes verify item-ranking identity.",
          "Hyperparameter lock policies verify tuned-params reuse is legitimate.",
          `Model/data pairing checks prevent evaluating the wrong model against the wrong ${abbr("split regime", "The exact train/validation/test split scheme and data preparation setup used for a run.")}.`,
          "Quality gates stop low-quality models before they are quietly treated as valid artifacts.",
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
      "If You Rebuilt This Today",
      `
        ${paragraph(
          `A few reasonable future directions would be worth considering: richer ${abbr("calibration", "Adjusting predicted intervals or uncertainty so they better match observed behavior on held-out data.")} methods, more explicit psychometric validation beyond score recovery, comparison against stronger ${abbr("IRT/CAT baselines", "Adaptive-testing baselines built from Item Response Theory and Computerized Adaptive Testing methods.")}, or a more specialized sequential selection policy that enforces domain balance structurally rather than hoping a generic utility score behaves well.`,
        )}
        ${paragraph(
          "But the most important thing is to preserve the current repo's discipline: explicit artifacts, strong baselines, and the willingness to invalidate attractive ideas when the held-out evidence says no.",
        )}
      `,
    )}
    ${section(
      "Course Exit Checklist",
      `
        ${list([
          "You can explain what each numbered stage consumes and produces.",
          "You understand why sparse-input augmentation is essential.",
          "You know the difference between raw scores and percentiles.",
          "You understand why greedy adaptive selection failed here.",
          "You can explain why ONNX is the deployment boundary.",
          "You know which stages must be rerun after each major class of change.",
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
