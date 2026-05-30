import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

// A5.4: the Python, TypeScript, and web runtimes must load the SAME onnxruntime
// version, or cross-runtime numerical parity is only certified for one of them.
// This test fails loudly on any future drift among the four manifests.
const repoRoot = resolve(import.meta.dirname, "..", "..");
const EXPECTED = "1.24.1";

function stripRange(version: string): string {
  return version.replace(/^[\^~>=<\s]+/, "").trim();
}

function readJson(relativePath: string): Record<string, unknown> {
  return JSON.parse(readFileSync(resolve(repoRoot, relativePath), "utf-8"));
}

describe("onnxruntime version parity across runtimes", () => {
  it("web package.json pins the expected onnxruntime-node", () => {
    const pkg = readJson("web/package.json") as {
      dependencies: Record<string, string>;
    };
    expect(stripRange(pkg.dependencies["onnxruntime-node"])).toBe(EXPECTED);
  });

  it("web package-lock resolves the expected onnxruntime-node", () => {
    const lock = readJson("web/package-lock.json") as {
      packages: Record<string, { version?: string }>;
    };
    expect(lock.packages["node_modules/onnxruntime-node"]?.version).toBe(
      EXPECTED,
    );
  });

  it("typescript package.json pins the expected onnxruntime-node", () => {
    const pkg = readJson("typescript/package.json") as {
      dependencies: Record<string, string>;
    };
    expect(stripRange(pkg.dependencies["onnxruntime-node"])).toBe(EXPECTED);
  });

  it("python pyproject pins the same onnxruntime version", () => {
    const text = readFileSync(resolve(repoRoot, "pyproject.toml"), "utf-8");
    const match = text.match(/onnxruntime==([0-9.]+)/);
    expect(match?.[1]).toBe(EXPECTED);
  });
});
