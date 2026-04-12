import { describe, expect, it } from "vitest";
import { getToggledTheme } from "./theme";

describe("theme helpers", () => {
  it("toggles between light and dark", () => {
    expect(getToggledTheme("light")).toBe("dark");
    expect(getToggledTheme("dark")).toBe("light");
  });
});
