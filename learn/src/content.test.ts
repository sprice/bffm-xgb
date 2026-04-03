import { describe, expect, it } from "vitest";

import { chapters } from "./content";
import { officialDocs } from "./content/data";

describe("learning course structure", () => {
  it("has a linear 12-chapter course with unique slugs and orders", () => {
    expect(chapters).toHaveLength(12);

    const slugs = new Set(chapters.map((chapter) => chapter.slug));
    const orders = new Set(chapters.map((chapter) => chapter.order));

    expect(slugs.size).toBe(chapters.length);
    expect(orders.size).toBe(chapters.length);

    expect(chapters.map((chapter) => chapter.order)).toEqual([
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    ]);
  });

  it("covers the core subject areas the user asked for", () => {
    const joined = chapters.map((chapter) => chapter.content).join(" ");

    expect(joined).toContain("psychometrics");
    expect(joined).toContain("XGBoost");
    expect(joined).toContain("bootstrap");
    expect(joined).toContain("ONNX");
    expect(joined).toContain("sparsity augmentation");
    expect(joined).toContain("adaptive");
  });

  it("gives every chapter meaningful metadata and content", () => {
    for (const chapter of chapters) {
      expect(chapter.title.length).toBeGreaterThan(5);
      expect(chapter.kicker.length).toBeGreaterThan(10);
      expect(chapter.summary.length).toBeGreaterThan(30);
      expect(chapter.content).toContain("lesson-section");
    }
  });

  it("adds glossary-style abbr tooltips to page 1 terms", () => {
    const chapter1 = chapters.find((chapter) => chapter.slug === "01-orientation");
    expect(chapter1).toBeTruthy();
    expect(chapter1!.content).toContain('class="glossary-term"');
    expect(chapter1!.content).toContain("data-tooltip=");
    expect(chapter1!.content).toContain(">IPIP-BFFM inventory<");
    expect(chapter1!.content).toContain(">quantile regression<");
    expect(chapter1!.content).toContain(">XGBoost<");
  });

  it("adds glossary tooltips across the full course", () => {
    for (const chapter of chapters) {
      expect(chapter.content).toContain('class="glossary-term"');
      expect(chapter.content).toContain("data-tooltip=");
    }
  });
});

describe("official docs links", () => {
  it("uses https URLs for all external resources", () => {
    for (const url of Object.values(officialDocs)) {
      expect(url.startsWith("https://")).toBe(true);
    }
  });

  it("points to the expected official domains for core technologies", () => {
    expect(officialDocs.xgboost).toContain("xgboost.readthedocs.io");
    expect(officialDocs.optuna).toContain("optuna.readthedocs.io");
    expect(officialDocs.onnx).toContain("onnx.ai");
    expect(officialDocs.onnxruntime).toContain("onnxruntime.ai");
    expect(officialDocs.python).toContain("docs.python.org");
  });
});
