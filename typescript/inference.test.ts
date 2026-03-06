/**
 * Tests for IPIP-BFFM inference module.
 */

import { existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { IPIPBFFMPredictor } from "./inference.js";

// ── Skip when output artifacts are not built ─────────────────────────────

const __dirname = dirname(fileURLToPath(import.meta.url));
const CONFIG_PATH = resolve(__dirname, "..", "output", "reference", "config.json");
const HAS_CONFIG = existsSync(CONFIG_PATH);

// ── Test vectors ────────────────────────────────────────────────────────

const DOMAINS = ["ext", "agr", "csn", "est", "opn"] as const;
const QUANTILES = ["q05", "q50", "q95"] as const;
const FEATURE_NAMES = DOMAINS.flatMap((d) =>
  Array.from({ length: 10 }, (_, i) => `${d}${i + 1}`)
);

// Input A: Full 50-item response, repeating 1-5 pattern
const INPUT_A_VALUES = Array.from({ length: 50 }, (_, i) => (i % 5) + 1);

// Input B: 20-item sparse response, top-4 per domain
const INPUT_B_ITEMS: Record<string, number> = {
  ext3: 3.0, ext5: 4.0, ext7: 5.0, ext4: 2.0,
  agr10: 3.0, agr7: 4.0, agr2: 5.0, agr9: 2.0,
  csn4: 3.0, csn8: 4.0, csn1: 5.0, csn5: 2.0,
  est10: 3.0, est9: 4.0, est8: 5.0, est6: 2.0,
  opn5: 3.0, opn10: 4.0, opn7: 5.0, opn2: 2.0,
};

// ── Test suite ──────────────────────────────────────────────────────────

let predictor: IPIPBFFMPredictor;

beforeAll(async () => {
  if (!HAS_CONFIG) return;
  predictor = await IPIPBFFMPredictor.create();
}, 30000);

afterAll(() => {
  predictor?.dispose();
});

describe.skipIf(!HAS_CONFIG)("Dict input", () => {
  it("predict() matches predictArray()", async () => {
    const arr = new Float32Array(50).fill(NaN);
    for (const [itemId, val] of Object.entries(INPUT_B_ITEMS)) {
      const idx = FEATURE_NAMES.indexOf(itemId);
      arr[idx] = val;
    }

    const resultArray = await predictor.predictArray(arr);
    const resultDict = await predictor.predict(INPUT_B_ITEMS);

    for (const domain of DOMAINS) {
      for (const q of QUANTILES) {
        expect(resultArray[domain].raw[q]).toBe(resultDict[domain].raw[q]);
        expect(resultArray[domain].percentile[q]).toBe(
          resultDict[domain].percentile[q]
        );
      }
    }
  });
});

describe.skipIf(!HAS_CONFIG)("Quantile ordering", () => {
  it("q05 <= q50 <= q95 in percentile space (full)", async () => {
    const arr = new Float32Array(INPUT_A_VALUES);
    const result = await predictor.predictArray(arr);

    for (const domain of DOMAINS) {
      expect(result[domain].percentile.q05).toBeLessThanOrEqual(
        result[domain].percentile.q50
      );
      expect(result[domain].percentile.q50).toBeLessThanOrEqual(
        result[domain].percentile.q95
      );
    }
  });

  it("q05 <= q50 <= q95 in percentile space (sparse)", async () => {
    const result = await predictor.predict(INPUT_B_ITEMS);

    for (const domain of DOMAINS) {
      expect(result[domain].percentile.q05).toBeLessThanOrEqual(
        result[domain].percentile.q50
      );
      expect(result[domain].percentile.q50).toBeLessThanOrEqual(
        result[domain].percentile.q95
      );
    }
  });
});

describe.skipIf(!HAS_CONFIG)("Percentile range", () => {
  it("all percentiles in [0, 100] (full)", async () => {
    const arr = new Float32Array(INPUT_A_VALUES);
    const result = await predictor.predictArray(arr);

    for (const domain of DOMAINS) {
      for (const q of QUANTILES) {
        expect(result[domain].percentile[q]).toBeGreaterThanOrEqual(0);
        expect(result[domain].percentile[q]).toBeLessThanOrEqual(100);
      }
    }
  });

  it("all percentiles in [0, 100] (sparse)", async () => {
    const result = await predictor.predict(INPUT_B_ITEMS);

    for (const domain of DOMAINS) {
      for (const q of QUANTILES) {
        expect(result[domain].percentile[q]).toBeGreaterThanOrEqual(0);
        expect(result[domain].percentile[q]).toBeLessThanOrEqual(100);
      }
    }
  });
});
