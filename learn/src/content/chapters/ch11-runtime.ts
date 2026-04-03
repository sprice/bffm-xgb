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

export const chapter11Runtime: Chapter = {
  slug: "11-stages-11-to-13-runtime",
  order: 11,
  title: "Stages 11 To 13 And Runtime",
  kicker: "ONNX export, inference packages, web delivery, Hugging Face, and light infra",
  summary:
    "Understand how trained models become a portable runtime bundle, why ONNX was chosen, how Python and TypeScript wrappers work, and where Terraform/AWS fit into the story.",
  content: `
    ${section(
      "From Trained Models To A Runtime Artifact",
      `
        ${lead(
          `The training code does not ship directly. Stage 11 converts the trained model family into a ${abbr("runtime bundle", "The packaged inference artifact used after training, rather than the training code itself.")} centered on a merged ${abbr("ONNX graph", "A computation graph saved in the ONNX format so it can run portably across languages and runtimes.")} plus a config file and provenance metadata.`,
        )}
        ${paragraph(
          `This is a ${abbr("productization", "The work of turning research code into a portable, reusable artifact suitable for real deployment.")} step. The repo wants one portable inference bundle that can be used from Python, TypeScript, and the web app without requiring the full training stack or Python-serving process in production.`,
        )}
        ${list([
          "<code>model.onnx</code>: one merged graph with 15 outputs",
          "<code>config.json</code>: feature names, norms, output names, calibration",
          "<code>provenance.json</code>: run identity and traceability",
          "Model card README files for publishing and downstream reuse",
        ])}
      `,
    )}
    ${section(
      "Why ONNX?",
      `
        ${paragraph(
          `The user already stated one pragmatic reason: ONNX made it easy to publish on ${abbr("Hugging Face", "A platform for sharing machine-learning models, datasets, and demos.")} and use the same exported model in the web-facing runtime. That is already a sufficient engineering reason.`,
        )}
        ${table(
          ["Choice", "Pros", "Cons"],
          [
            ["ONNX export", "Portable, language-agnostic runtime, easier sharing and deployment, good fit for HF hosting", "Conversion complexity, compatibility edge cases, another artifact layer to validate"],
            ["Serve Python joblib models directly", "Simpler training-to-serving path", "Harder browser/server portability, heavier runtime dependency chain"],
            ["Custom tree-walking JSON runtime", "Maximum control", "More maintenance burden and more room for inference drift"],
          ],
        )}
        ${callout(
          "note",
          "Most honest explanation",
          paragraph(
            `ONNX was chosen because it is a good ${abbr("deployment boundary", "A clean separation between the artifacts used in production and the heavier code used during training and experimentation.")}. It separates research-time training concerns from runtime-time inference concerns and makes cross-language reuse realistic.`,
          ),
        )}
      `,
    )}
    ${section(
      "The Export Details That Matter",
      `
        ${paragraph(
          `The export stage does more than format conversion. It also validates ${abbr("parity", "Agreement between two implementations, here the original model outputs and the exported ONNX outputs.")} between the original model outputs and the ONNX outputs, merges 15 individual graphs into one shared-input graph, and writes the runtime metadata the wrappers depend on.`,
        )}
        ${paragraph(
          "The code even includes a compatibility patch for newer XGBoost serialization quirks. That is the kind of production detail that disappears from research summaries but matters a lot in real systems.",
        )}
        ${internalFiles([
          "pipeline/11_export_onnx.py",
          "output/reference/model.onnx",
          "output/reference/config.json",
          "output/reference/provenance.json",
          "output/reference/README.md",
        ])}
      `,
    )}
    ${section(
      "Inference Packages",
      `
        ${paragraph(
          `The Python and TypeScript ${abbr("inference packages", "Small runtime libraries whose job is to load the exported model and produce predictions from new inputs.")} are thin wrappers around the ONNX session. They validate inputs, create a 50-position vector with <code>NaN</code> for unanswered items, run the ONNX graph once, convert raw outputs to percentiles, sort quantiles into ${abbr("monotonic order", "Forced order where the lower quantile stays below the median and the median stays below the upper quantile.")}, and apply the ${abbr("calibrated interval regime", "The rule for adjusting interval width so predicted uncertainty better matches observed coverage.")}.`,
        )}
        ${codeBlock(
          `responses -> Float32Array with NaN for unanswered items
-> ONNX session.run()
-> raw q05/q50/q95 per domain
-> raw-to-percentile transform
-> optional interval scaling by calibration regime`,
          "text",
        )}
      `,
    )}
    ${section(
      "The Web App, Lightly",
      `
        ${paragraph(
          `The web app is not the scientific heart of the repo, but it is an interesting delivery surface. The server handles reverse-scoring and ONNX inference. The client mainly handles assessment flow, ${abbr("persistence", "Saving local state so progress or results survive reloads or later visits.")}, and result sharing. This is a good separation because it keeps psychometric correctness and model execution centralized.`,
        )}
        ${internalFiles([
          "python/inference.py",
          "typescript/inference.ts",
          "web/src/server/index.ts",
          "web/src/server/predictor.ts",
          "web/src/server/reverse-score.ts",
          "web/src/client/App.tsx",
        ])}
      `,
    )}
    ${section(
      "Stages 12 And 13",
      `
        ${paragraph(
          "Stage 12 generates research figures from evaluation artifacts. Stage 13 uploads the final bundle to Hugging Face. These are publication and distribution steps rather than model-design steps, but they matter because they close the loop between experiment and usable artifact.",
        )}
      `,
    )}
    ${section(
      "Terraform And AWS, Lightly",
      `
        ${paragraph(
          `The infrastructure story is intentionally modest. ${abbr("Terraform", "Infrastructure-as-code tooling used to define and create cloud resources reproducibly.")} exists here because tune/train can be expensive and long-running, and it is useful to stand up reproducible CPU or GPU workers on demand. The runtime product itself is much lighter than the training pipeline.`,
        )}
        ${paragraph(
          `A good mental split is: training infrastructure is heavy because research iteration is heavy; deployed inference is light because ${abbr("ONNX Runtime", "A runtime engine for executing ONNX models outside the original training framework.")} bundles are comparatively simple.`,
        )}
      `,
    )}
    ${section(
      "Further Reading",
      resourceList([
        {
          label: "ONNX docs",
          href: officialDocs.onnx,
          note: "Official ONNX introduction",
        },
        {
          label: "ONNX Runtime docs",
          href: officialDocs.onnxruntime,
          note: "Official runtime docs",
        },
        {
          label: "Hugging Face Hub docs",
          href: officialDocs.huggingfaceHub,
          note: "Official publishing/docs surface",
        },
        {
          label: "Terraform docs",
          href: officialDocs.terraform,
          note: "Official IaC docs",
        },
        {
          label: "AWS EC2 docs",
          href: officialDocs.awsEc2,
          note: "Official cloud compute docs",
        },
        {
          label: "Hono docs",
          href: officialDocs.hono,
          note: "Official web framework docs for the server layer",
        },
      ]),
    )}`,
};
