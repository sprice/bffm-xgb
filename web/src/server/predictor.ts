/**
 * IPIP-BFFM Sparse Quantile Model — TypeScript inference module.
 *
 * Loads a single merged ONNX model and provides prediction from either a
 * Record of item responses or a raw Float32Array.
 *
 * Supports two loading modes:
 *   1. Local: MODEL_DIR env var points to a directory with config.json + model.onnx
 *   2. HuggingFace: HF_REPO_ID env var (+ optional HF_VARIANT, default "reference")
 *      downloads config.json and model.onnx from the HF Hub at startup
 */

import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";
import * as ort from "onnxruntime-node";
import { standardNormalCDF } from "./erf.js";

// Single-threaded, sequential CPU session so the user-facing percentiles are a
// deterministic function of the input. Multi-threaded ONNX Runtime sums partial
// results in a host-dependent order, which can perturb raw scores enough to flip
// a percentile that rounds to one decimal place. Inference here is batch-1, so
// single-threaded is free.
const SESSION_OPTIONS: ort.InferenceSession.SessionOptions = {
  intraOpNumThreads: 1,
  interOpNumThreads: 1,
  executionMode: "sequential",
  executionProviders: ["cpu"],
};

const DOMAINS = ["ext", "agr", "csn", "est", "opn"] as const;
const QUANTILES = ["q05", "q50", "q95"] as const;

type Domain = (typeof DOMAINS)[number];
type Quantile = (typeof QUANTILES)[number];

export interface DomainResult {
  raw: { q05: number; q50: number; q95: number };
  percentile: { q05: number; q50: number; q95: number };
}

export type PredictionResult = Record<Domain, DomainResult>;

interface CalibrationDomain {
  observed_coverage: number;
  scale_factor: number;
}

interface ModelConfig {
  domains: string[];
  model_file: string;
  outputs: string[];
  input: { feature_names: string[] };
  norms: Record<string, { mean: number; sd: number }>;
  calibration?: Record<string, Record<string, CalibrationDomain>>;
}

/**
 * Select the calibration regime by answered-item count: 50+ answered →
 * `full_50`, otherwise `sparse_20_balanced`. NaN entries are unanswered and are
 * not counted. Exported so the K=49/K=50 boundary can be locked by a test; the
 * three runtimes must agree on this dispatch.
 */
export function calibrationRegime(responses: Float32Array): string {
  let nAnswered = 0;
  for (const v of responses) {
    if (!Number.isNaN(v)) nAnswered++;
  }
  return nAnswered >= 50 ? "full_50" : "sparse_20_balanced";
}

export class IPIPBFFMPredictor {
  private session!: ort.InferenceSession;
  private config: ModelConfig;
  private featureIndex: Map<string, number>;
  private outputNames: string[];

  private constructor(config: ModelConfig) {
    this.config = config;
    this.outputNames = config.outputs;
    this.featureIndex = new Map(
      config.input.feature_names.map((name, i) => [name, i])
    );
  }

  static async create(modelDir: string): Promise<IPIPBFFMPredictor> {
    const configPath = join(modelDir, "config.json");
    const config: ModelConfig = JSON.parse(
      readFileSync(configPath, "utf-8")
    );

    const predictor = new IPIPBFFMPredictor(config);
    predictor.session = await ort.InferenceSession.create(
      join(modelDir, config.model_file),
      SESSION_OPTIONS
    );

    return predictor;
  }

  async predict(
    responses: Record<string, number>
  ): Promise<PredictionResult> {
    const unknown = Object.keys(responses)
      .filter((k) => !this.featureIndex.has(k))
      .sort();
    if (unknown.length > 0) {
      const valid = this.config.input.feature_names.join(", ");
      throw new Error(
        `Unrecognized item IDs: ${unknown.join(", ")}. Valid IDs: ${valid}`
      );
    }

    const bad: Record<string, number> = {};
    for (const [k, v] of Object.entries(responses)) {
      if (!Number.isFinite(v) || v < 1 || v > 5) {
        bad[k] = v;
      }
    }
    if (Object.keys(bad).length > 0) {
      throw new Error(
        `Response values must be finite numbers in [1, 5]. Got: ${JSON.stringify(bad)}`
      );
    }

    const n = this.config.input.feature_names.length;
    const arr = new Float32Array(n).fill(NaN);
    for (const [itemId, value] of Object.entries(responses)) {
      const idx = this.featureIndex.get(itemId);
      if (idx !== undefined) {
        arr[idx] = value;
      }
    }
    return this.predictArray(arr);
  }

  async predictArray(responses: Float32Array): Promise<PredictionResult> {
    const n = this.config.input.feature_names.length;
    if (responses.length !== n) {
      throw new Error(
        `Expected Float32Array of length ${n}, got ${responses.length}`
      );
    }

    const tensor = new ort.Tensor("float32", responses, [1, n]);
    const output = await this.session.run({ input: tensor });

    const regime = calibrationRegime(responses);
    const regimeCal = this.config.calibration?.[regime] ?? {};

    const results: Partial<PredictionResult> = {};

    for (const domain of DOMAINS) {
      const rawScores: Record<string, number> = {};
      for (const q of QUANTILES) {
        const key = `${domain}_${q}`;
        rawScores[q] = (output[key].data as Float32Array)[0];
      }

      const norms = this.config.norms[domain];
      const pct: Record<string, number> = {};
      for (const q of QUANTILES) {
        const z = (rawScores[q] - norms.mean) / norms.sd;
        pct[q] = standardNormalCDF(z) * 100;
      }

      const pctVals = [pct.q05, pct.q50, pct.q95].sort((a, b) => a - b);
      pct.q05 = pctVals[0];
      pct.q50 = pctVals[1];
      pct.q95 = pctVals[2];

      const scale = regimeCal[domain]?.scale_factor ?? 1.0;
      if (scale !== 1.0) {
        const halfWidth = 0.5 * (pct.q95 - pct.q05) * scale;
        pct.q05 = Math.min(100, Math.max(0, pct.q50 - halfWidth));
        pct.q95 = Math.min(100, Math.max(0, pct.q50 + halfWidth));
      }

      results[domain] = {
        raw: {
          q05: Math.round(rawScores.q05 * 10000) / 10000,
          q50: Math.round(rawScores.q50 * 10000) / 10000,
          q95: Math.round(rawScores.q95 * 10000) / 10000,
        },
        percentile: {
          q05: Math.round(pct.q05 * 10) / 10,
          q50: Math.round(pct.q50 * 10) / 10,
          q95: Math.round(pct.q95 * 10) / 10,
        },
      };
    }

    return results as PredictionResult;
  }

  dispose(): void {
    this.session?.release();
  }
}

function verifySha256(filePath: string, expectedSha256: string): void {
  const data = readFileSync(filePath);
  const actual = createHash("sha256").update(data).digest("hex");
  if (actual !== expectedSha256) {
    throw new Error(
      `Integrity check failed for ${filePath}: expected sha256=${expectedSha256}, got ${actual}`
    );
  }
}

/**
 * Check whether a file exists in a HuggingFace repo using a HEAD request.
 * Returns false on any network/fetch error so that callers fall through
 * to the actual download, which will surface a clearer error message.
 */
async function hfFileExists(
  repoId: string,
  revision: string,
  filePath: string,
): Promise<boolean> {
  try {
    const url = `https://huggingface.co/${repoId}/resolve/${revision}/${filePath}`;
    const res = await fetch(url, { method: "HEAD", redirect: "follow" });
    return res.ok;
  } catch {
    return false;
  }
}

/**
 * Download a file from the HuggingFace Hub (public repos only, no auth needed).
 * Uses the resolve endpoint which handles CDN redirects.
 */
async function downloadHfFile(
  repoId: string,
  revision: string,
  filePath: string,
  destPath: string,
  expectedSha256?: string
): Promise<void> {
  const url = `https://huggingface.co/${repoId}/resolve/${revision}/${filePath}`;
  console.log(`  Downloading ${url}`);
  const res = await fetch(url, { redirect: "follow" });
  if (!res.ok) {
    throw new Error(`Failed to download ${url}: ${res.status} ${res.statusText}`);
  }
  const buffer = Buffer.from(await res.arrayBuffer());

  if (expectedSha256) {
    const actual = createHash("sha256").update(buffer).digest("hex");
    if (actual !== expectedSha256) {
      throw new Error(
        `Integrity check failed for ${filePath}: expected sha256=${expectedSha256}, got ${actual}`
      );
    }
    console.log(`  Integrity OK (sha256=${actual.slice(0, 12)}…)`);
  }

  writeFileSync(destPath, buffer);
  console.log(`  Saved ${destPath} (${(buffer.length / 1024 / 1024).toFixed(1)} MB)`);
}

/**
 * Resolve the model directory:
 *   1. MODEL_DIR env var → use local path directly
 *   2. HF_REPO_ID env var → download from HuggingFace Hub into a cache dir
 *   3. Fallback → ../output/reference relative to this file
 */
async function resolveModelDir(): Promise<string> {
  // 1. Explicit local path
  if (process.env.MODEL_DIR) {
    return process.env.MODEL_DIR;
  }

  // 2. Download from HuggingFace — fail closed: a pinned commit revision and
  // both file checksums are MANDATORY, so the deployed app can never silently
  // pull from the mutable `main` branch or serve an unverified model.
  const repoId = process.env.HF_REPO_ID;
  if (repoId) {
    const variant = process.env.HF_VARIANT || "reference";
    const revision = process.env.HF_REVISION;
    const configSha = process.env.HF_SHA256_CONFIG;
    const modelSha = process.env.HF_SHA256_MODEL;

    if (!revision || !configSha || !modelSha) {
      const missing: string[] = [];
      if (!revision) missing.push("HF_REVISION (pin a commit sha, not a branch)");
      if (!configSha) missing.push("HF_SHA256_CONFIG");
      if (!modelSha) missing.push("HF_SHA256_MODEL");
      throw new Error(
        `HF_REPO_ID is set but the integrity pins are missing: ${missing.join(", ")}. ` +
          "Refusing to download an unpinned/unverified model. Set these in the " +
          "deploy environment (see .env.example), or use MODEL_DIR for a local model."
      );
    }

    const safeRepoId = repoId.replace(/\//g, "--");
    const cacheDir = join(tmpdir(), "bffm-xgb-model", safeRepoId, revision, variant);

    const configPath = join(cacheDir, "config.json");
    const modelPath = join(cacheDir, "model.onnx");

    if (existsSync(configPath) && existsSync(modelPath)) {
      // Always re-verify the cached files against the pinned checksums.
      verifySha256(configPath, configSha);
      verifySha256(modelPath, modelSha);
      console.log(`Using cached model from ${cacheDir}`);
      return cacheDir;
    }

    // Detect repo layout: files in variant subdirectory or at root
    const variantPrefix = await hfFileExists(repoId, revision, `${variant}/config.json`)
      ? `${variant}/`
      : "";
    if (variantPrefix) {
      console.log(`Downloading model from HF: ${repoId} (variant: ${variant}, revision: ${revision})`);
    } else {
      console.log(`Downloading model from HF: ${repoId} (root layout, revision: ${revision})`);
    }
    mkdirSync(cacheDir, { recursive: true });

    await downloadHfFile(repoId, revision, `${variantPrefix}config.json`, configPath, configSha);
    await downloadHfFile(repoId, revision, `${variantPrefix}model.onnx`, modelPath, modelSha);

    return cacheDir;
  }

  // 3. Fallback to local output directory
  return resolve(import.meta.dirname, "..", "..", "..", "output", "reference");
}

let _predictor: IPIPBFFMPredictor | null = null;
let _loading: Promise<IPIPBFFMPredictor> | null = null;

export function isPredictorReady(): boolean {
  return _predictor !== null;
}

export async function getPredictor(): Promise<IPIPBFFMPredictor> {
  if (_predictor) return _predictor;
  if (!_loading) {
    _loading = resolveModelDir()
      .then((dir) => IPIPBFFMPredictor.create(dir))
      .then((p) => { _predictor = p; return p; });
  }
  return _loading;
}
