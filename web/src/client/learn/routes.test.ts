import { describe, expect, it } from "vitest";
import {
  defaultLearnChapterSlug,
  defaultLearnPath,
  getLearnPath,
  isLearnChapterSlug,
} from "./routes";

describe("learn routes", () => {
  it("builds chapter paths from slugs", () => {
    expect(defaultLearnPath).toBe(`/learn/${defaultLearnChapterSlug}`);
    expect(getLearnPath(defaultLearnChapterSlug)).toBe(defaultLearnPath);
  });

  it("validates known chapter slugs", () => {
    expect(isLearnChapterSlug(defaultLearnChapterSlug)).toBe(true);
    expect(isLearnChapterSlug("not-a-real-chapter")).toBe(false);
  });
});
