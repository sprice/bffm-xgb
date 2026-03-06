import { describe, it, expect } from "vitest";
import { encodeResults, decodeResults } from "./results-codec";
import type { PredictionResult } from "../types";

const DOMAINS = ["ext", "agr", "csn", "est", "opn"] as const;

function makeResults(
  values: Record<string, { q05: number; q50: number; q95: number }>
): PredictionResult {
  const result: Partial<PredictionResult> = {};
  for (const domain of DOMAINS) {
    result[domain] = {
      raw: { q05: 0, q50: 0, q95: 0 },
      percentile: values[domain],
    };
  }
  return result as PredictionResult;
}

describe("results-codec", () => {
  describe("round-trip encode/decode", () => {
    it("preserves integer percentiles exactly", () => {
      const input = makeResults({
        ext: { q05: 10, q50: 50, q95: 90 },
        agr: { q05: 20, q50: 60, q95: 80 },
        csn: { q05: 5, q50: 45, q95: 95 },
        est: { q05: 30, q50: 55, q95: 70 },
        opn: { q05: 15, q50: 40, q95: 85 },
      });

      const encoded = encodeResults(input);
      const decoded = decodeResults(encoded)!;

      expect(decoded).not.toBeNull();
      for (const domain of DOMAINS) {
        expect(decoded[domain].percentile.q05).toBe(
          input[domain].percentile.q05
        );
        expect(decoded[domain].percentile.q50).toBe(
          input[domain].percentile.q50
        );
        expect(decoded[domain].percentile.q95).toBe(
          input[domain].percentile.q95
        );
      }
    });

    it("rounds fractional percentiles to nearest integer", () => {
      const input = makeResults({
        ext: { q05: 10.4, q50: 50.6, q95: 90.2 },
        agr: { q05: 20.5, q50: 60.1, q95: 80.9 },
        csn: { q05: 5.3, q50: 45.7, q95: 95.0 },
        est: { q05: 30.0, q50: 55.0, q95: 70.0 },
        opn: { q05: 15.0, q50: 40.0, q95: 85.0 },
      });

      const encoded = encodeResults(input);
      const decoded = decodeResults(encoded)!;

      expect(decoded).not.toBeNull();
      expect(decoded.ext.percentile.q05).toBe(10);
      expect(decoded.ext.percentile.q50).toBe(51);
      expect(decoded.agr.percentile.q05).toBe(21); // 20.5 rounds to 21
      expect(decoded.agr.percentile.q95).toBe(81);
    });

    it("handles boundary values 0 and 100", () => {
      const input = makeResults({
        ext: { q05: 0, q50: 50, q95: 100 },
        agr: { q05: 0, q50: 0, q95: 0 },
        csn: { q05: 100, q50: 100, q95: 100 },
        est: { q05: 0, q50: 100, q95: 100 },
        opn: { q05: 1, q50: 50, q95: 99 },
      });

      const encoded = encodeResults(input);
      const decoded = decodeResults(encoded)!;

      expect(decoded).not.toBeNull();
      for (const domain of DOMAINS) {
        expect(decoded[domain].percentile.q05).toBe(
          Math.round(input[domain].percentile.q05)
        );
        expect(decoded[domain].percentile.q50).toBe(
          Math.round(input[domain].percentile.q50)
        );
        expect(decoded[domain].percentile.q95).toBe(
          Math.round(input[domain].percentile.q95)
        );
      }
    });
  });

  describe("encode", () => {
    it("produces a short base64url string", () => {
      const input = makeResults({
        ext: { q05: 10, q50: 50, q95: 90 },
        agr: { q05: 20, q50: 60, q95: 80 },
        csn: { q05: 5, q50: 45, q95: 95 },
        est: { q05: 30, q50: 55, q95: 70 },
        opn: { q05: 15, q50: 40, q95: 85 },
      });

      const encoded = encodeResults(input);
      expect(encoded.length).toBe(20); // 15 bytes -> 20 base64 chars
      expect(encoded).toMatch(/^[A-Za-z0-9_-]+$/); // base64url chars only
    });

    it("clamps values above 100 to 100", () => {
      const input = makeResults({
        ext: { q05: 0, q50: 50, q95: 150 },
        agr: { q05: 0, q50: 50, q95: 100 },
        csn: { q05: 0, q50: 50, q95: 100 },
        est: { q05: 0, q50: 50, q95: 100 },
        opn: { q05: 0, q50: 50, q95: 100 },
      });

      const decoded = decodeResults(encodeResults(input))!;
      expect(decoded.ext.percentile.q95).toBe(100);
    });

    it("clamps negative values to 0", () => {
      const input = makeResults({
        ext: { q05: -5, q50: 50, q95: 90 },
        agr: { q05: 0, q50: 50, q95: 100 },
        csn: { q05: 0, q50: 50, q95: 100 },
        est: { q05: 0, q50: 50, q95: 100 },
        opn: { q05: 0, q50: 50, q95: 100 },
      });

      const decoded = decodeResults(encodeResults(input))!;
      expect(decoded.ext.percentile.q05).toBe(0);
    });
  });

  describe("decode", () => {
    it("returns null for empty string", () => {
      expect(decodeResults("")).toBeNull();
    });

    it("returns null for wrong-length payload", () => {
      // 10 bytes instead of 15
      expect(decodeResults("AAAAAAAAAA")).toBeNull();
    });

    it("returns null for invalid base64", () => {
      expect(decodeResults("!!!not-base64!!!")).toBeNull();
    });

    it("returns null when a byte exceeds 100", () => {
      // Manually craft 15 bytes where one > 100
      const bytes = new Uint8Array(15).fill(50);
      bytes[0] = 101;
      let binary = "";
      for (const b of bytes) binary += String.fromCharCode(b);
      const encoded = btoa(binary)
        .replace(/\+/g, "-")
        .replace(/\//g, "_")
        .replace(/=+$/, "");
      expect(decodeResults(encoded)).toBeNull();
    });

    it("decoded results have zeroed raw values", () => {
      const input = makeResults({
        ext: { q05: 10, q50: 50, q95: 90 },
        agr: { q05: 20, q50: 60, q95: 80 },
        csn: { q05: 5, q50: 45, q95: 95 },
        est: { q05: 30, q50: 55, q95: 70 },
        opn: { q05: 15, q50: 40, q95: 85 },
      });

      const decoded = decodeResults(encodeResults(input))!;
      for (const domain of DOMAINS) {
        expect(decoded[domain].raw).toEqual({ q05: 0, q50: 0, q95: 0 });
      }
    });
  });

  describe("deterministic encoding", () => {
    it("same input always produces the same hash", () => {
      const input = makeResults({
        ext: { q05: 10, q50: 50, q95: 90 },
        agr: { q05: 20, q50: 60, q95: 80 },
        csn: { q05: 5, q50: 45, q95: 95 },
        est: { q05: 30, q50: 55, q95: 70 },
        opn: { q05: 15, q50: 40, q95: 85 },
      });

      const a = encodeResults(input);
      const b = encodeResults(input);
      expect(a).toBe(b);
    });
  });
});
