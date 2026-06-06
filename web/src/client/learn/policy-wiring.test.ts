import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { repoFacts } from "./content/data";
import { chapter01Orientation } from "./content/chapters/ch01-orientation";
import { chapter03Stats } from "./content/chapters/ch03-stats";
import { chapter08Tuning } from "./content/chapters/ch08-tuning";
import { chapter09Training } from "./content/chapters/ch09-training";
import { chapter10Evaluation } from "./content/chapters/ch10-eval";

// WS2 + WS3-B: the chapters must render model/policy numbers from repoFacts, not
// hardcoded literals. Two complementary guards: (1) the SOURCE files contain no
// stale literal; (2) the RENDERED content contains the value derived live from
// repoFacts. If a future retrain moves a value, repoFacts updates and the
// rendered prose follows — and these tests fail if anyone re-introduces a literal.

const chaptersDir = resolve(import.meta.dirname, "content", "chapters");
const readChapterSource = (file: string) =>
  readFileSync(resolve(chaptersDir, file), "utf-8");

describe("repoFacts carries the single-sourced training/assessment policy", () => {
  it("exposes the tuning objective weights/penalties", () => {
    expect(repoFacts.tuningObjective).toEqual({
      sparse20Weight: 0.8,
      full50Weight: 0.2,
      sparse20PenaltyWeight: 2.0,
      sparse20PenaltyFloor: 0.85,
      full50PenaltyWeight: 1.0,
      full50PenaltyFloor: 0.95,
    });
  });

  it("exposes the coverage-calibration policy", () => {
    expect(repoFacts.calibration).toEqual({
      coverageLow: 0.85,
      coverageHigh: 0.95,
      targetCoverage: 0.9,
      coverageFloor: 0.5,
    });
  });

  it("exposes the adaptive-stopping policy", () => {
    expect(repoFacts.adaptiveStop).toEqual({
      semThreshold: 0.45,
      minItemsPerDomain: 4,
    });
  });
});

describe("chapter sources contain no hardcoded model/policy literals (WS2/WS3-B)", () => {
  it("the r ≈ 0.93 operating point is interpolated, not typed, in every chapter", () => {
    for (const file of [
      "ch01-orientation.ts",
      "ch03-stats.ts",
      "ch09-training.ts",
      "ch10-eval.ts",
    ]) {
      const src = readChapterSource(file);
      expect(src, `${file} must not hardcode "r ≈ 0.93"`).not.toContain("≈ 0.93");
      expect(src).toContain("repoFacts.baselineK20.domainBalancedR.toFixed(2)");
    }
  });

  it("ch08 tuning objective code block has no literal weights", () => {
    const src = readChapterSource("ch08-tuning.ts");
    expect(src).not.toContain("0.80 * mean_r_sparse20");
    expect(src).toContain("repoFacts.tuningObjective.sparse20Weight");
  });

  it("ch09 calibration code block has no literal thresholds", () => {
    const src = readChapterSource("ch09-training.ts");
    expect(src).not.toContain("if coverage < 0.85");
    expect(src).toContain("repoFacts.calibration.coverageLow");
  });

  it("ch10 SEM worked example has no literal alpha/SEM values", () => {
    const src = readChapterSource("ch10-eval.ts");
    for (const literal of ["0.726", "0.477", "0.779", "0.428"]) {
      expect(src, `ch10 must not hardcode SEM example value ${literal}`).not.toContain(literal);
    }
    expect(src).not.toContain("a 0.45 threshold plus a minimum of 4 items");
    expect(src).toContain("repoFacts.adaptiveStop.semThreshold");
  });
});

describe("rendered chapter content reflects repoFacts values", () => {
  it("renders the operating-point r from repoFacts in ch01/03/09/10", () => {
    const r = repoFacts.baselineK20.domainBalancedR.toFixed(2);
    for (const ch of [chapter01Orientation, chapter03Stats, chapter09Training, chapter10Evaluation]) {
      expect(ch.content).toContain(`r ≈ ${r}`);
    }
  });

  it("renders the tuning objective weights from repoFacts in ch08", () => {
    const o = repoFacts.tuningObjective;
    expect(chapter08Tuning.content).toContain(`${o.sparse20Weight.toFixed(2)} * mean_r_sparse20`);
    expect(chapter08Tuning.content).toContain(`${o.full50PenaltyFloor.toFixed(2)} - mean_r_full`);
  });

  it("renders the calibration thresholds from repoFacts in ch09", () => {
    // codeBlock() HTML-escapes < and >, so assert against the escaped output.
    const c = repoFacts.calibration;
    expect(chapter09Training.content).toContain(`if coverage &lt; ${c.coverageLow.toFixed(2)}`);
    expect(chapter09Training.content).toContain(`elif coverage &gt; ${c.coverageHigh.toFixed(2)}`);
  });

  it("renders the SEM worked example computed from repoFacts in ch10", () => {
    const rBar = repoFacts.interItemRBar.ext;
    const sd = repoFacts.norms.ext.sd;
    const alpha = (k: number) => (k * rBar) / (1 + (k - 1) * rBar);
    const sem = (k: number) => sd * Math.sqrt(1 - alpha(k));
    // "->" is escaped to "-&gt;" inside codeBlock().
    expect(chapter10Evaluation.content).toContain(
      `k = 3  -&gt; alpha ≈ ${alpha(3).toFixed(3)}, SEM ≈ ${sem(3).toFixed(3)}`,
    );
    expect(chapter10Evaluation.content).toContain(
      `k = 4  -&gt; alpha ≈ ${alpha(4).toFixed(3)}, SEM ≈ ${sem(4).toFixed(3)}`,
    );
    expect(chapter10Evaluation.content).toContain(
      `a ${repoFacts.adaptiveStop.semThreshold} threshold plus a minimum of ${repoFacts.adaptiveStop.minItemsPerDomain} items per domain`,
    );
  });
});
