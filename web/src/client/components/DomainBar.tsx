import type { DomainResult } from "../types";
import { ordinal, ordinalSuffix } from "../lib/utils";

interface DomainBarProps {
  label: string;
  description: string;
  result: DomainResult;
  /** Fill color for bar and median dot visuals (3:1 OK for non-text). */
  color: string;
  /** Darker shade for heading label and percentile number (passes AA 4.5:1 as text). */
  textColor: string;
  index: number;
}

export function DomainBar({ label, description, result, color, textColor, index }: DomainBarProps) {
  const { q05, q50, q95 } = result.percentile;
  const roundedQ05 = Math.round(q05);
  const roundedQ50 = Math.round(q50);
  const roundedQ95 = Math.round(q95);

  const baseDelay = 0.25 + index * 0.1;

  return (
    <article
      className="animate-in"
      style={{ animationDelay: `${index * 0.08}s` }}
    >
      <div className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-1 items-baseline">
        <h3
          className="font-semibold text-base col-start-1 row-start-1 m-0"
          style={{ color: textColor }}
        >
          {label}
        </h3>
        <div
          className="text-3xl font-bold col-start-2 row-start-1 text-right font-display"
          style={{ color: textColor }}
        >
          <span aria-hidden="true">
            {roundedQ50}<span className="text-lg">{ordinalSuffix(roundedQ50)}</span>
          </span>
          <span className="sr-only">
            {ordinal(roundedQ50)} percentile. 90 percent confidence interval: {ordinal(roundedQ05)} to {ordinal(roundedQ95)}.
          </span>
        </div>

        <div
          className="col-span-full row-start-2 h-4 bg-border/30 rounded-md relative overflow-visible"
          aria-hidden="true"
        >
          {/* CI range bar – sweeps in from left */}
          <div
            className="absolute top-0 h-full rounded-md opacity-30 animate-scale-x-in"
            style={{
              left: `${q05}%`,
              width: `${Math.max(q95 - q05, 1)}%`,
              backgroundColor: color,
              animationDelay: `${baseDelay}s`,
            }}
          />
          {/* Median dot – pops in after bar fills */}
          <div
            className="absolute top-1/2 w-4 h-4 rounded-full border-2 border-surface shadow-md animate-dot-pop-in"
            style={{
              left: `${q50}%`,
              backgroundColor: color,
              animationDelay: `${baseDelay + 0.25}s`,
            }}
          />
        </div>
      </div>

      <p className="text-sm text-text-muted leading-relaxed mt-2">
        {description}
      </p>
    </article>
  );
}
