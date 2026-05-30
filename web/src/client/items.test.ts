/**
 * Locks the deployed 20-item web form (items.ts) to the canonical evaluated
 * item set. canonical_items.json is the committed source of truth (the
 * domain_balanced top-4-per-domain selection + reverse-key flags from
 * lib.constants.REVERSE_KEYED), shared with the Python/TS runtime tests. If the
 * pipeline's item selection ever drifts from the hand-maintained items.ts, this
 * fails.
 */

import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it, expect } from "vitest";
import { ITEMS } from "./items";

const __dirname = dirname(fileURLToPath(import.meta.url));
// web/src/client -> repo root is three levels up.
const canonical = JSON.parse(
  readFileSync(
    resolve(__dirname, "..", "..", "..", "tests", "fixtures", "golden", "canonical_items.json"),
    "utf-8"
  )
) as { domain_balanced_20: string[]; reverse_keyed: Record<string, boolean> };

describe("deployed 20-item form vs canonical set", () => {
  it("has exactly 20 items", () => {
    expect(ITEMS).toHaveLength(20);
    expect(canonical.domain_balanced_20).toHaveLength(20);
  });

  it("item id set equals the canonical domain_balanced-20 set", () => {
    const webIds = new Set(ITEMS.map((i) => i.id));
    const canonicalIds = new Set(canonical.domain_balanced_20);
    expect(webIds).toEqual(canonicalIds);
  });

  it("each item's reverse-key flag matches lib.constants.REVERSE_KEYED", () => {
    for (const item of ITEMS) {
      expect(item.isReverseKeyed).toBe(canonical.reverse_keyed[item.id]);
    }
  });
});
