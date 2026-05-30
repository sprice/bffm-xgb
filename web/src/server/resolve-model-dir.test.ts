/**
 * Fail-closed contract for the HuggingFace model download: when HF_REPO_ID is
 * set (and no local MODEL_DIR), a pinned commit revision and BOTH file checksums
 * are MANDATORY. The deployed app must never silently pull from the mutable
 * `main` branch or serve an unverified model. MODEL_DIR always wins when set.
 *
 * Each case resets the predictor module (vi.resetModules) so the cached
 * _loading singleton from one case can't leak into the next.
 */

import { existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const __dirname = dirname(fileURLToPath(import.meta.url));
const FIXTURE_DIR = resolve(__dirname, "..", "..", "..", "tests", "fixtures", "golden");
const HAS_FIXTURE = existsSync(resolve(FIXTURE_DIR, "config.json"));

const HF_VARS = ["HF_REPO_ID", "HF_REVISION", "HF_SHA256_CONFIG", "HF_SHA256_MODEL"];

describe("HF download integrity (fail-closed)", () => {
  let savedModelDir: string | undefined;
  const savedHf: Record<string, string | undefined> = {};

  beforeEach(() => {
    vi.resetModules(); // fresh predictor module (fresh _predictor/_loading) per case
    savedModelDir = process.env.MODEL_DIR;
    delete process.env.MODEL_DIR;
    for (const v of HF_VARS) {
      savedHf[v] = process.env[v];
      delete process.env[v];
    }
  });

  afterEach(() => {
    if (savedModelDir !== undefined) process.env.MODEL_DIR = savedModelDir;
    else delete process.env.MODEL_DIR;
    for (const v of HF_VARS) {
      if (savedHf[v] !== undefined) process.env[v] = savedHf[v];
      else delete process.env[v];
    }
  });

  it("rejects when HF_REPO_ID is set without a pinned revision + both checksums", async () => {
    process.env.HF_REPO_ID = "someorg/somerepo";
    const { getPredictor } = await import("./predictor.js");
    await expect(getPredictor()).rejects.toThrow(/integrity pins are missing/);
  });

  it("still rejects (naming the missing checksum) when only the revision is pinned", async () => {
    process.env.HF_REPO_ID = "someorg/somerepo";
    process.env.HF_REVISION = "abc123";
    const { getPredictor } = await import("./predictor.js");
    // A regression that dropped the checksum sub-checks must still fail here.
    await expect(getPredictor()).rejects.toThrow(/HF_SHA256_CONFIG/);
  });

  it("still rejects when a single checksum is missing", async () => {
    process.env.HF_REPO_ID = "someorg/somerepo";
    process.env.HF_REVISION = "abc123";
    process.env.HF_SHA256_CONFIG = "0".repeat(64);
    const { getPredictor } = await import("./predictor.js");
    await expect(getPredictor()).rejects.toThrow(/HF_SHA256_MODEL/);
  });

  it.skipIf(!HAS_FIXTURE)(
    "MODEL_DIR wins over HF_REPO_ID — local model resolves without integrity pins",
    async () => {
      // MODEL_DIR set alongside an unpinned HF_REPO_ID must take the local path
      // (predictor.ts step 1) and never reach the HF fail-closed branch.
      process.env.HF_REPO_ID = "someorg/somerepo";
      process.env.MODEL_DIR = FIXTURE_DIR;
      const { getPredictor } = await import("./predictor.js");
      const predictor = await getPredictor();
      expect(predictor).toBeDefined();
      predictor.dispose();
    }
  );
});
