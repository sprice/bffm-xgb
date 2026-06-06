import { existsSync, readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it, expect } from "vitest";
import {
  DOMAINS,
  QUANTILES,
  FULL_REGIME_MIN_ANSWERED,
  calibrationRegime,
} from "./predictor";

const __dirname = dirname(fileURLToPath(import.meta.url));

// Same model-resolution contract as predictor.test.ts: web/model, or the
// committed golden bundle via MODEL_DIR in CI.
const MODEL_DIR = process.env.MODEL_DIR || resolve(__dirname, "..", "..", "model");
const CONFIG_PATH = resolve(MODEL_DIR, "config.json");
const HAS_MODEL = existsSync(CONFIG_PATH);

if (process.env.REQUIRE_ARTIFACTS === "1" && !HAS_MODEL) {
  throw new Error(
    `REQUIRE_ARTIFACTS=1 but no config.json found at MODEL_DIR=${MODEL_DIR}`,
  );
}

interface Config {
  domains: string[];
  quantiles: number[];
  outputs: string[];
  calibration?: Record<string, unknown>;
}

// The predictor hardcodes DOMAINS / QUANTILES (and the >=50 regime threshold)
// as structural constants rather than reading them from config.json at load
// time. That is deliberate (they are the model's fixed architecture), but it
// means a model whose config disagrees would mis-key the ONNX outputs silently.
// These parity tests are the lock: if the deployed model's architecture ever
// diverges from the constants, CI fails here instead of at a user's request.
describe.skipIf(!HAS_MODEL)("predictor constants ⟷ config.json parity", () => {
  const config: Config = JSON.parse(readFileSync(CONFIG_PATH, "utf-8"));

  it("DOMAINS matches config.domains exactly (order included)", () => {
    expect([...DOMAINS]).toEqual(config.domains);
  });

  it("config.outputs is exactly DOMAINS × QUANTILES in `${domain}_${q}` order", () => {
    // This is the precise contract predict() relies on at `output[`${domain}_${q}`]`.
    const expected = DOMAINS.flatMap((d) => QUANTILES.map((q) => `${d}_${q}`));
    expect(config.outputs).toEqual(expected);
  });

  it("QUANTILES name list lines up with config.quantiles (q05/q50/q95)", () => {
    const expectedNames = config.quantiles.map((q) => `q${String(q).slice(2).padEnd(2, "0")}`);
    expect([...QUANTILES]).toEqual(expectedNames);
  });

  it("calibrationRegime dispatch values are real config.calibration keys", () => {
    const full = new Float32Array(FULL_REGIME_MIN_ANSWERED).fill(3);
    const sparse = new Float32Array(FULL_REGIME_MIN_ANSWERED - 1).fill(3);
    expect(calibrationRegime(full)).toBe("full_50");
    expect(calibrationRegime(sparse)).toBe("sparse_20_balanced");
    const keys = Object.keys(config.calibration ?? {});
    expect(keys).toContain("full_50");
    expect(keys).toContain("sparse_20_balanced");
  });
});

// Pure-logic assertions that do not need a model bundle present.
describe("calibrationRegime boundary (no model needed)", () => {
  it(`switches to full_50 at exactly ${FULL_REGIME_MIN_ANSWERED} answered items`, () => {
    const atThreshold = new Float32Array(FULL_REGIME_MIN_ANSWERED).fill(3);
    const below = new Float32Array(FULL_REGIME_MIN_ANSWERED);
    below.fill(3);
    below[0] = NaN; // one unanswered -> 49 answered
    expect(calibrationRegime(atThreshold)).toBe("full_50");
    expect(calibrationRegime(below)).toBe("sparse_20_balanced");
  });
});
