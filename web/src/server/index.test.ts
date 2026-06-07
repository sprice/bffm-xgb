/**
 * End-to-end integration test for the Hono server: drives the real
 * /api/predict and /api/health request→response path against the committed
 * golden fixture (no network, no port bind). CI runs `npm run build` separately
 * to catch TypeScript/build errors; this exercises the deployed surface.
 */

import { existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it, expect, beforeAll } from "vitest";

const __dirname = dirname(fileURLToPath(import.meta.url));
// Point the predictor at the committed fixture BEFORE importing the server.
const FIXTURE_DIR = resolve(__dirname, "..", "..", "..", "tests", "fixtures", "golden");
process.env.MODEL_DIR = process.env.MODEL_DIR ?? FIXTURE_DIR;

const HAS_MODEL = existsSync(resolve(process.env.MODEL_DIR, "config.json"));
if (process.env.REQUIRE_ARTIFACTS === "1" && !HAS_MODEL) {
  throw new Error(
    `REQUIRE_ARTIFACTS=1 but no model.config.json at MODEL_DIR=${process.env.MODEL_DIR}`
  );
}

const DOMAINS = ["ext", "agr", "csn", "est", "opn"] as const;

// Canonical deployed 20-item form (forward-keyed neutral answers are fine here —
// this asserts the request path/shape, not learned model semantics).
const RESPONSES: Record<string, number> = {
  ext2: 3, ext4: 3, ext5: 3, ext7: 3,
  agr4: 3, agr5: 3, agr7: 3, agr9: 3,
  csn1: 3, csn4: 3, csn5: 3, csn6: 3,
  est1: 3, est6: 3, est7: 3, est8: 3,
  opn1: 3, opn2: 3, opn5: 3, opn10: 3,
};

describe.skipIf(!HAS_MODEL)("Hono API integration", () => {
  let app: { request: (input: string, init?: RequestInit) => Response | Promise<Response> };

  beforeAll(async () => {
    const mod = await import("./index.js");
    app = mod.app;
    // Load the model so isPredictorReady() is true and /api/predict scores.
    const predictor = await import("./predictor.js");
    await predictor.getPredictor();
  }, 30_000);

  it("GET /api/health → 200 ok once loaded", async () => {
    const res = await app.request("/api/health");
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ status: "ok" });
  });

  it("POST /api/predict → 200 with five-domain results", async () => {
    const res = await app.request("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ responses: RESPONSES }),
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      results: Record<string, { raw: Record<string, number>; percentile: Record<string, number> }>;
    };
    for (const d of DOMAINS) {
      expect(body.results[d]).toBeDefined();
      const p = body.results[d].percentile;
      expect(p.q05).toBeLessThanOrEqual(p.q50);
      expect(p.q50).toBeLessThanOrEqual(p.q95);
      for (const q of ["q05", "q50", "q95"] as const) {
        expect(p[q]).toBeGreaterThanOrEqual(0);
        expect(p[q]).toBeLessThanOrEqual(100);
      }
    }
  });

  it("POST /api/predict with empty responses → 400", async () => {
    const res = await app.request("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ responses: {} }),
    });
    expect(res.status).toBe(400);
  });

  it("POST /api/predict with an unknown item id → 400", async () => {
    const res = await app.request("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ responses: { not_an_item: 3 } }),
    });
    expect(res.status).toBe(400);
  });

  it("POST /api/predict with malformed JSON → 400", async () => {
    const res = await app.request("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{not json",
    });
    expect(res.status).toBe(400);
  });
});
