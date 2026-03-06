import type { Item } from "./types";

/**
 * 20 items selected for the BFFM-XGB-20 sparse assessment.
 * 4 items per domain, balanced by loading direction.
 *
 * Text: pipeline/05_compute_correlations.py ITEM_TEXTS
 * Reverse-keying: lib/constants.py REVERSE_KEYED
 */
export const ITEMS: Item[] = [
  // Extraversion (4 items: 2R, 4R, 5, 7)
  { id: "ext2", text: "I don't talk a lot.", domain: "Extraversion", domainCode: "ext", isReverseKeyed: true },
  { id: "ext4", text: "I keep in the background.", domain: "Extraversion", domainCode: "ext", isReverseKeyed: true },
  { id: "ext5", text: "I start conversations.", domain: "Extraversion", domainCode: "ext", isReverseKeyed: false },
  { id: "ext7", text: "I talk to a lot of different people at parties.", domain: "Extraversion", domainCode: "ext", isReverseKeyed: false },

  // Agreeableness (4 items: 4, 5R, 7R, 9)
  { id: "agr4", text: "I sympathize with others' feelings.", domain: "Agreeableness", domainCode: "agr", isReverseKeyed: false },
  { id: "agr5", text: "I am not interested in other people's problems.", domain: "Agreeableness", domainCode: "agr", isReverseKeyed: true },
  { id: "agr7", text: "I am not really interested in others.", domain: "Agreeableness", domainCode: "agr", isReverseKeyed: true },
  { id: "agr9", text: "I feel others' emotions.", domain: "Agreeableness", domainCode: "agr", isReverseKeyed: false },

  // Conscientiousness (4 items: 1, 4R, 5, 6R)
  { id: "csn1", text: "I am always prepared.", domain: "Conscientiousness", domainCode: "csn", isReverseKeyed: false },
  { id: "csn4", text: "I make a mess of things.", domain: "Conscientiousness", domainCode: "csn", isReverseKeyed: true },
  { id: "csn5", text: "I get chores done right away.", domain: "Conscientiousness", domainCode: "csn", isReverseKeyed: false },
  { id: "csn6", text: "I often forget to put things back in their proper place.", domain: "Conscientiousness", domainCode: "csn", isReverseKeyed: true },

  // Emotional Stability (4 items: 1R, 6R, 7R, 8R)
  { id: "est1", text: "I get stressed out easily.", domain: "Emotional Stability", domainCode: "est", isReverseKeyed: true },
  { id: "est6", text: "I get upset easily.", domain: "Emotional Stability", domainCode: "est", isReverseKeyed: true },
  { id: "est7", text: "I change my mood a lot.", domain: "Emotional Stability", domainCode: "est", isReverseKeyed: true },
  { id: "est8", text: "I have frequent mood swings.", domain: "Emotional Stability", domainCode: "est", isReverseKeyed: true },

  // Intellect/Imagination (4 items: 1, 2R, 5, 10)
  { id: "opn1", text: "I have a rich vocabulary.", domain: "Intellect / Imagination", domainCode: "opn", isReverseKeyed: false },
  { id: "opn2", text: "I have difficulty understanding abstract ideas.", domain: "Intellect / Imagination", domainCode: "opn", isReverseKeyed: true },
  { id: "opn5", text: "I have excellent ideas.", domain: "Intellect / Imagination", domainCode: "opn", isReverseKeyed: false },
  { id: "opn10", text: "I am full of ideas.", domain: "Intellect / Imagination", domainCode: "opn", isReverseKeyed: false },
];
