import { describe, expect, it } from "vitest";
import { isAnalyticsPath, isApiPath, isNavigationRequest } from "./route-guards";

describe("route guards", () => {
  it("treats assessment deep links as navigation requests", () => {
    expect(isNavigationRequest("/assessment/start", "text/html")).toBe(true);
    expect(isNavigationRequest("/assessment/question/3", "text/html")).toBe(true);
    expect(isNavigationRequest("/learn/chapter-1", "text/html")).toBe(true);
  });

  it("treats the analytics endpoint as exact and separate from assessment routes", () => {
    expect(isAnalyticsPath("/a")).toBe(true);
    expect(isAnalyticsPath("/a/health")).toBe(true);
    expect(isAnalyticsPath("/assessment/start")).toBe(false);
    expect(isNavigationRequest("/a", "text/html")).toBe(false);
    expect(isNavigationRequest("/assessment/start", "text/html")).toBe(true);
  });

  it("treats api and asset paths as non-navigation", () => {
    expect(isApiPath("/api")).toBe(true);
    expect(isApiPath("/api/predict")).toBe(true);
    expect(isNavigationRequest("/api/predict", "text/html")).toBe(false);
    expect(isNavigationRequest("/assets/index.css", "text/css")).toBe(false);
    expect(isNavigationRequest("/favicon.svg", "image/svg+xml")).toBe(false);
  });

  it("uses fetch metadata when accept headers are not decisive", () => {
    expect(isNavigationRequest("/assessment/start", "", "navigate")).toBe(true);
    expect(isNavigationRequest("/assessment/start", "", undefined, "document")).toBe(true);
  });
});
