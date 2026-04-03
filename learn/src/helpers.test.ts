import { describe, expect, it } from "vitest";

import { splitFormula, splitFormulaThree } from "./content/helpers";

describe("helper formulas", () => {
  it("renders ext-est stratum formula correctly", () => {
    expect(splitFormula(4, 1)).toContain(">21<");
  });

  it("renders ext-est-opn stratum formula correctly", () => {
    expect(splitFormulaThree(4, 1, 3)).toContain(">108<");
  });
});
