import { describe, it, expect } from "vitest";
import { reverseScore } from "./reverse-score";

const REVERSE_KEYED = [
  "ext2", "ext4", "ext6", "ext8", "ext10",
  "agr1", "agr3", "agr5", "agr7",
  "csn2", "csn4", "csn6", "csn8",
  "est1", "est3", "est5", "est6", "est7", "est8", "est9", "est10",
  "opn2", "opn4", "opn6",
];

const FORWARD_KEYED = [
  "ext1", "ext3", "ext5", "ext7", "ext9",
  "agr2", "agr4", "agr6", "agr8", "agr9", "agr10",
  "csn1", "csn3", "csn5", "csn7", "csn9", "csn10",
  "est2", "est4",
  "opn1", "opn3", "opn5", "opn7", "opn8", "opn9", "opn10",
];

describe("reverseScore", () => {
  it("applies 6-x transform to reverse-keyed items", () => {
    for (const id of REVERSE_KEYED) {
      const result = reverseScore({ [id]: 1 });
      expect(result[id]).toBe(5); // 6 - 1
    }
    for (const id of REVERSE_KEYED) {
      const result = reverseScore({ [id]: 4 });
      expect(result[id]).toBe(2); // 6 - 4
    }
  });

  it("leaves forward-keyed items unchanged", () => {
    for (const id of FORWARD_KEYED) {
      const result = reverseScore({ [id]: 3 });
      expect(result[id]).toBe(3);
    }
  });

  it("reverse scoring is symmetric (applying twice returns original)", () => {
    const input: Record<string, number> = {
      ext2: 1, ext5: 4, agr5: 2, agr4: 5, est1: 3, opn1: 3,
    };
    const result = reverseScore(reverseScore(input));
    for (const [key, value] of Object.entries(input)) {
      expect(result[key]).toBe(value);
    }
  });

  it("handles mixed forward and reverse items in one call", () => {
    const input = { ext1: 3, ext2: 3, agr4: 5, agr5: 5 };
    const result = reverseScore(input);
    expect(result.ext1).toBe(3); // forward -> unchanged
    expect(result.ext2).toBe(3); // reverse: 6 - 3 = 3
    expect(result.agr4).toBe(5); // forward -> unchanged
    expect(result.agr5).toBe(1); // reverse: 6 - 5 = 1
  });

  it("maps all Likert values correctly for reverse items", () => {
    const expected: Record<number, number> = { 1: 5, 2: 4, 3: 3, 4: 2, 5: 1 };
    for (const [input, output] of Object.entries(expected)) {
      const result = reverseScore({ est1: Number(input) });
      expect(result.est1).toBe(output);
    }
  });

  it("returns a new object (no mutation)", () => {
    const input = { ext1: 3, ext2: 4 };
    const result = reverseScore(input);
    expect(result).not.toBe(input);
    expect(input.ext2).toBe(4); // original unchanged
  });

  it("handles empty input", () => {
    expect(reverseScore({})).toEqual({});
  });
});
