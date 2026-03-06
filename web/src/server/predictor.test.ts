import { existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { IPIPBFFMPredictor } from "./predictor";
import { reverseScore } from "./reverse-score";

const __dirname = dirname(fileURLToPath(import.meta.url));

// Model lives in web/model (symlink or copy) or fallback to output/reference
const MODEL_DIR =
  process.env.MODEL_DIR ||
  resolve(__dirname, "..", "..", "model");
const HAS_MODEL = existsSync(resolve(MODEL_DIR, "config.json"));

const DOMAINS = ["ext", "agr", "csn", "est", "opn"] as const;
const QUANTILES = ["q05", "q50", "q95"] as const;

// 20-item assessment matching web/src/client/items.ts
const ALL_HIGH: Record<string, number> = {
  ext2: 5, ext4: 5, ext5: 5, ext7: 5,
  agr4: 5, agr5: 5, agr7: 5, agr9: 5,
  csn1: 5, csn4: 5, csn5: 5, csn6: 5,
  est1: 5, est6: 5, est7: 5, est8: 5,
  opn1: 5, opn2: 5, opn5: 5, opn10: 5,
};

const ALL_LOW: Record<string, number> = Object.fromEntries(
  Object.keys(ALL_HIGH).map((k) => [k, 1])
);

const MIXED: Record<string, number> = {
  ext2: 4, ext4: 2, ext5: 4, ext7: 5,
  agr4: 5, agr5: 1, agr7: 2, agr9: 4,
  csn1: 3, csn4: 3, csn5: 3, csn6: 3,
  est1: 2, est6: 4, est7: 3, est8: 2,
  opn1: 5, opn2: 1, opn5: 4, opn10: 5,
};

let predictor: IPIPBFFMPredictor;

beforeAll(async () => {
  if (!HAS_MODEL) return;
  predictor = await IPIPBFFMPredictor.create(MODEL_DIR);
}, 30_000);

afterAll(() => {
  predictor?.dispose();
});

describe.skipIf(!HAS_MODEL)("Web scoring pipeline", () => {
  it("all percentiles are in [0, 100]", async () => {
    for (const input of [ALL_HIGH, ALL_LOW, MIXED]) {
      const scored = reverseScore(input);
      const result = await predictor.predict(scored);
      for (const domain of DOMAINS) {
        for (const q of QUANTILES) {
          expect(result[domain].percentile[q]).toBeGreaterThanOrEqual(0);
          expect(result[domain].percentile[q]).toBeLessThanOrEqual(100);
        }
      }
    }
  });

  it("quantile ordering: q05 <= q50 <= q95", async () => {
    for (const input of [ALL_HIGH, ALL_LOW, MIXED]) {
      const scored = reverseScore(input);
      const result = await predictor.predict(scored);
      for (const domain of DOMAINS) {
        expect(result[domain].percentile.q05).toBeLessThanOrEqual(
          result[domain].percentile.q50
        );
        expect(result[domain].percentile.q50).toBeLessThanOrEqual(
          result[domain].percentile.q95
        );
      }
    }
  });

  it("extreme high responses produce high percentiles for non-reverse-dominant domains", async () => {
    // All 5s after reverse scoring: forward items stay 5, reverse items become 1
    // For domains with mostly forward items (opn), median should be high
    const scored = reverseScore(ALL_HIGH);
    const result = await predictor.predict(scored);
    // OPN has 3 forward items (opn1, opn5, opn10) and 1 reverse (opn2)
    // After reverse: opn1=5, opn2=4(6-5=1 wait no, opn2 is reverse so 6-5=1), opn5=5, opn10=5
    // This means mostly high values -> high percentile for opn
    expect(result.opn.percentile.q50).toBeGreaterThan(50);
  });

  it("extreme low responses produce low percentiles for non-reverse-dominant domains", async () => {
    const scored = reverseScore(ALL_LOW);
    const result = await predictor.predict(scored);
    // All 1s: forward items stay 1, reverse items become 5
    // OPN: opn1=1, opn2=5(6-1), opn5=1, opn10=1 -> mostly low -> low percentile
    expect(result.opn.percentile.q50).toBeLessThan(50);
  });

  it("results contain all five domains", async () => {
    const scored = reverseScore(MIXED);
    const result = await predictor.predict(scored);
    for (const domain of DOMAINS) {
      expect(result[domain]).toBeDefined();
      expect(result[domain].raw).toBeDefined();
      expect(result[domain].percentile).toBeDefined();
    }
  });

  it("raw scores are finite numbers", async () => {
    const scored = reverseScore(MIXED);
    const result = await predictor.predict(scored);
    for (const domain of DOMAINS) {
      for (const q of QUANTILES) {
        expect(Number.isFinite(result[domain].raw[q])).toBe(true);
        expect(Number.isFinite(result[domain].percentile[q])).toBe(true);
      }
    }
  });
});
