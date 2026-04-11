import { describe, expect, it } from "vitest";
import {
  getToggledTheme,
  resolveTheme,
} from "./theme";

describe("theme helpers", () => {
  it("toggles the resolved theme", () => {
    expect(getToggledTheme("light")).toBe("dark");
    expect(getToggledTheme("dark")).toBe("light");
  });

  it("resolves the active theme correctly", () => {
    expect(resolveTheme("system", "light")).toBe("light");
    expect(resolveTheme("system", "dark")).toBe("dark");
    expect(resolveTheme("light", "dark")).toBe("light");
    expect(resolveTheme("dark", "light")).toBe("dark");
  });
});
