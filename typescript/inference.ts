/**
 * IPIP-BFFM Sparse Quantile Model — TypeScript inference module.
 *
 * Loads a single merged ONNX model and provides prediction from either a
 * Record of item responses or a raw Float32Array.
 */

import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import * as ort from "onnxruntime-node";

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
 * Abramowitz & Stegun (1964) approximation of the standard normal CDF.
 * Same implementation as packages/web/lib/utils.ts.
 */
function standardNormalCDF(z: number): number {
  if (!Number.isFinite(z)) return Number.NaN;

  const absZ = Math.abs(z);
  const t = 1 / (1 + 0.2316419 * absZ);
  const d = Math.exp(-0.5 * absZ * absZ) / Math.sqrt(2 * Math.PI);

  const poly =
    t *
    (0.31938153 +
      t *
        (-0.356563782 +
          t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));

  const cnd = 1 - d * poly;
  const p = z >= 0 ? cnd : 1 - cnd;
  return Math.min(1, Math.max(0, p));
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

  /**
   * Create a predictor by loading config and the merged ONNX session.
   *
   * @param modelDir Path to the directory containing model.onnx and config.json.
   *                 Defaults to the parent of this file's directory.
   */
  static async create(modelDir?: string): Promise<IPIPBFFMPredictor> {
    const dir = modelDir ?? resolve(import.meta.dirname, "..", "output", "reference");
    const configPath = join(dir, "config.json");
    const config: ModelConfig = JSON.parse(
      readFileSync(configPath, "utf-8")
    );

    const predictor = new IPIPBFFMPredictor(config);
    predictor.session = await ort.InferenceSession.create(
      join(dir, config.model_file)
    );

    return predictor;
  }

  /**
   * Predict from a Record mapping item IDs to response values (1-5).
   * Missing items are treated as NaN (unanswered).
   *
   * @throws {Error} If any keys in `responses` are not valid item IDs.
   */
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

  private calibrationRegime(responses: Float32Array): string {
    let nAnswered = 0;
    for (const v of responses) {
      if (!Number.isNaN(v)) nAnswered++;
    }
    return nAnswered >= 50 ? "full_50" : "sparse_20_balanced";
  }

  /**
   * Predict from a raw Float32Array of length 50.
   * NaN for unanswered items.
   *
   * @throws {Error} If `responses` does not have the expected length.
   */
  async predictArray(responses: Float32Array): Promise<PredictionResult> {
    const n = this.config.input.feature_names.length;
    if (responses.length !== n) {
      throw new Error(
        `Expected Float32Array of length ${n}, got ${responses.length}`
      );
    }

    const tensor = new ort.Tensor("float32", responses, [1, n]);
    const output = await this.session.run({ input: tensor });

    const regime = this.calibrationRegime(responses);
    const regimeCal = this.config.calibration?.[regime] ?? {};

    const results: Partial<PredictionResult> = {};

    for (const domain of DOMAINS) {
      const rawScores: Record<string, number> = {};
      for (const q of QUANTILES) {
        const key = `${domain}_${q}`;
        rawScores[q] = (output[key].data as Float32Array)[0];
      }

      // Convert raw scores to percentiles
      const norms = this.config.norms[domain];
      const pct: Record<string, number> = {};
      for (const q of QUANTILES) {
        const z = (rawScores[q] - norms.mean) / norms.sd;
        pct[q] = standardNormalCDF(z) * 100;
      }

      // Sort percentiles so q05 <= q50 <= q95
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

  /** Release the ONNX session. */
  dispose(): void {
    this.session?.release();
  }
}
