import type { DomainResult } from "../types";
import { ordinal, ordinalSuffix } from "../lib/utils";

interface DomainBarProps {
  label: string;
  description: string;
  result: DomainResult;
  color: string;
  index: number;
}

export function DomainBar({ label, description, result, color, index }: DomainBarProps) {
  const { q05, q50, q95 } = result.percentile;
  const roundedQ05 = Math.round(q05);
  const roundedQ50 = Math.round(q50);
  const roundedQ95 = Math.round(q95);

  const baseDelay = 0.25 + index * 0.1;

  return (
    <div
      className="animate-in"
      style={{ animationDelay: `${index * 0.08}s` }}
      role="img"
      aria-label={`${label}: ${ordinal(roundedQ50)} percentile, 90% confidence interval ${ordinal(roundedQ05)} to ${ordinal(roundedQ95)}`}
    >
      <div className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-1 items-baseline" aria-hidden="true">
        <div className="font-semibold text-base col-start-1 row-start-1" style={{ color }}>{label}</div>
        <div
          className="text-3xl font-bold col-start-2 row-start-1 text-right font-display"
          style={{ color }}
        >
          {roundedQ50}<span className="text-lg">{ordinalSuffix(roundedQ50)}</span>
        </div>

        <div className="col-span-full row-start-2 h-4 bg-border/30 rounded-md relative overflow-visible">
          {/* CI range bar – sweeps in from left */}
          <div
            className="absolute top-0 h-full rounded-md opacity-30"
            style={{
              left: `${q05}%`,
              width: `${Math.max(q95 - q05, 1)}%`,
              backgroundColor: color,
              animation: `scale-in-x 0.7s cubic-bezier(0.16, 1, 0.3, 1) both`,
              animationDelay: `${baseDelay}s`,
              transformOrigin: "left",
            }}
          />
          {/* Median dot – pops in after bar fills */}
          <div
            className="absolute top-1/2 w-4 h-4 rounded-full border-2 border-surface shadow-md"
            style={{
              left: `${q50}%`,
              backgroundColor: color,
              animation: `dot-pop 0.45s cubic-bezier(0.16, 1, 0.3, 1) both`,
              animationDelay: `${baseDelay + 0.25}s`,
            }}
          />
        </div>
      </div>

      <p className="text-sm text-text-muted leading-relaxed mt-2" aria-hidden="true">
        {description}
      </p>
    </div>
  );
}
