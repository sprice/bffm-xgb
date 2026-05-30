/**
 * Tests for IPIP-BFFM inference module.
 */

import { existsSync, readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { IPIPBFFMPredictor } from "./inference.js";

// ── Run against the committed fixture bundle ─────────────────────────────

const __dirname = dirname(fileURLToPath(import.meta.url));
const FIXTURE_DIR =
  process.env.BFFM_FIXTURE_DIR ??
  resolve(__dirname, "..", "tests", "fixtures", "golden");
const HAS_CONFIG = existsSync(resolve(FIXTURE_DIR, "config.json"));

// REQUIRE_ARTIFACTS=1 (set in CI) turns a missing fixture into a hard failure.
if (process.env.REQUIRE_ARTIFACTS === "1" && !HAS_CONFIG) {
  throw new Error(
    `REQUIRE_ARTIFACTS=1 but the golden fixture config is missing at ${FIXTURE_DIR}`
  );
}

// ── Test vectors ────────────────────────────────────────────────────────

const DOMAINS = ["ext", "agr", "csn", "est", "opn"] as const;
const QUANTILES = ["q05", "q50", "q95"] as const;
const FEATURE_NAMES = DOMAINS.flatMap((d) =>
  Array.from({ length: 10 }, (_, i) => `${d}${i + 1}`)
);

// Input A: Full 50-item response, repeating 1-5 pattern
const INPUT_A_VALUES = Array.from({ length: 50 }, (_, i) => (i % 5) + 1);

// Input B: the canonical deployed balanced-20 set (top-4 per domain), loaded
// from the committed fixture so it cannot drift from the pipeline's selection.
const CANONICAL_20: string[] = HAS_CONFIG
  ? (
      JSON.parse(
        readFileSync(resolve(FIXTURE_DIR, "canonical_items.json"), "utf-8")
      ) as { domain_balanced_20: string[] }
    ).domain_balanced_20
  : [];
const INPUT_B_ITEMS: Record<string, number> = Object.fromEntries(
  CANONICAL_20.map((id, i) => [id, (i % 5) + 1])
);

// ── Test suite ──────────────────────────────────────────────────────────

let predictor: IPIPBFFMPredictor;

beforeAll(async () => {
  if (!HAS_CONFIG) return;
  predictor = await IPIPBFFMPredictor.create(FIXTURE_DIR);
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
