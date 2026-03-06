import type { DomainResult, PredictionResult, QuantileValues } from "../types";
import { DOMAIN_ORDER } from "../types";

/**
 * Encodes a PredictionResult into a compact base64url string.
 *
 * Layout: 15 bytes — for each of the 5 domains (ext, agr, csn, est, opn),
 * 3 bytes in order: percentile q05, q50, q95. Each clamped to 0–100.
 *
 * Base64url output is ~20 characters.
 */
export function encodeResults(results: PredictionResult): string {
  const bytes = new Uint8Array(15);
  let i = 0;
  for (const domain of DOMAIN_ORDER) {
    const { q05, q50, q95 } = results[domain].percentile;
    bytes[i++] = clamp(Math.round(q05));
    bytes[i++] = clamp(Math.round(q50));
    bytes[i++] = clamp(Math.round(q95));
  }
  return toBase64Url(bytes);
}

export function decodeResults(hash: string): PredictionResult | null {
  try {
    const bytes = fromBase64Url(hash);
    if (bytes.length !== 15) return null;

    const results: Partial<PredictionResult> = {};
    let i = 0;
    for (const domain of DOMAIN_ORDER) {
      const q05 = bytes[i++];
      const q50 = bytes[i++];
      const q95 = bytes[i++];
      if (q05 > 100 || q50 > 100 || q95 > 100) return null;
      const percentile: QuantileValues = { q05, q50, q95 };
      const raw: QuantileValues = { q05: 0, q50: 0, q95: 0 };
      results[domain] = { raw, percentile } as DomainResult;
    }
    return results as PredictionResult;
  } catch {
    return null;
  }
}

function clamp(n: number): number {
  return Math.max(0, Math.min(100, n));
}

function toBase64Url(bytes: Uint8Array): string {
  let binary = "";
  for (const b of bytes) binary += String.fromCharCode(b);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function fromBase64Url(str: string): Uint8Array {
  const base64 = str.replace(/-/g, "+").replace(/_/g, "/");
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}
