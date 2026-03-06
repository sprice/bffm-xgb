/**
 * Reverse-scoring for IPIP-BFFM items.
 *
 * The model was trained on data where reverse-keyed items were pre-transformed
 * (6 - raw_value) in pipeline/02_load_sqlite.py. The server must apply the same
 * transform before inference.
 *
 * Source of truth: lib/constants.py REVERSE_KEYED
 */

const REVERSE_KEYED_ITEMS = new Set([
  // EXT: 2, 4, 6, 8, 10
  "ext2", "ext4", "ext6", "ext8", "ext10",
  // AGR: 1, 3, 5, 7
  "agr1", "agr3", "agr5", "agr7",
  // CSN: 2, 4, 6, 8
  "csn2", "csn4", "csn6", "csn8",
  // EST: 1, 3, 5, 6, 7, 8, 9, 10
  "est1", "est3", "est5", "est6", "est7", "est8", "est9", "est10",
  // OPN: 2, 4, 6
  "opn2", "opn4", "opn6",
]);

export function reverseScore(
  responses: Record<string, number>
): Record<string, number> {
  const result: Record<string, number> = {};
  for (const [key, value] of Object.entries(responses)) {
    result[key] = REVERSE_KEYED_ITEMS.has(key) ? 6 - value : value;
  }
  return result;
}
