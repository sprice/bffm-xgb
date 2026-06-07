import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function ordinalSuffix(n: number): string {
  const mod100 = n % 100;
  if (mod100 >= 11 && mod100 <= 13) return "th";
  switch (n % 10) {
    case 1:
      return "st";
    case 2:
      return "nd";
    case 3:
      return "rd";
    default:
      return "th";
  }
}

export function ordinal(n: number): string {
  return `${n}${ordinalSuffix(n)}`;
}

/**
 * Format a Pearson r (or any 0..1 statistic) for prose display: round to
 * `digits` decimals and strip the leading zero, e.g. 0.9277 -> ".93".
 * Used to source model-derived correlation copy from `repoFacts` instead of
 * hand-typing it, so the displayed value can never drift from the artifacts.
 */
export function formatR(value: number, digits = 2): string {
  return value.toFixed(digits).replace(/^0(?=\.)/, "");
}
