export const learnChapterSlugs = [
  "01-orientation",
  "02-psychometrics",
  "03-statistics",
  "04-codebase-map",
  "05-stages-01-to-04",
  "06-stage-05-item-analysis",
  "07-xgboost",
  "08-stage-06-tuning",
  "09-stage-07-training",
  "10-stages-08-to-10",
  "11-stages-11-to-13-runtime",
  "12-safe-changes",
] as const;

const learnChapterSlugSet = new Set<string>(learnChapterSlugs);

export const defaultLearnChapterSlug = learnChapterSlugs[0];
export const defaultLearnPath = `/learn/${defaultLearnChapterSlug}`;

export function isLearnChapterSlug(slug: string | undefined): boolean {
  return Boolean(slug && learnChapterSlugSet.has(slug));
}

export function getLearnPath(slug: string): string {
  return `/learn/${slug}`;
}
