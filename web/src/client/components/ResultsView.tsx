import type { PredictionResult } from "../types";
import { DOMAIN_LABELS, DOMAIN_ORDER } from "../types";
import { DomainBar } from "./DomainBar";

interface ResultsViewProps {
  results: PredictionResult;
  heading?: string;
  thirdPerson?: boolean;
}

const DOMAIN_COLORS: Record<string, string> = {
  ext: "var(--color-ext)",
  agr: "var(--color-agr)",
  csn: "var(--color-csn)",
  est: "var(--color-est)",
  opn: "var(--color-opn)",
};

const DOMAIN_DESCRIPTIONS: Record<string, string> = {
  ext: "How energized you are by social interaction and external stimulation.",
  agr: "How much you prioritize cooperation, empathy, and getting along with others.",
  csn: "How organized, dependable, and goal-directed you tend to be.",
  est: "How calmly and steadily you handle stress, worry, and negative emotions.",
  opn: "How drawn you are to new ideas, creativity, and intellectual curiosity.",
};

const DOMAIN_DESCRIPTIONS_THIRD_PERSON: Record<string, string> = {
  ext: "How energized one is by social interaction and external stimulation.",
  agr: "How much one prioritizes cooperation, empathy, and getting along with others.",
  csn: "How organized, dependable, and goal-directed one tends to be.",
  est: "How calmly and steadily one handles stress, worry, and negative emotions.",
  opn: "How drawn one is to new ideas, creativity, and intellectual curiosity.",
};

export function ResultsView({ results, heading = "Your Big Five Profile", thirdPerson = false }: ResultsViewProps) {
  return (
    <div className="bg-surface border border-border rounded-lg p-6 sm:p-8 shadow-sm animate-in" role="region" aria-label="Assessment results">
      <h2 className="font-display text-3xl font-semibold text-center mb-1">{heading}</h2>
      <p className="text-center text-text-muted text-base mb-8">
        Percentile scores based on 20 items
      </p>

      <div className="flex flex-col gap-7">
        {DOMAIN_ORDER.map((domain, i) => (
          <DomainBar
            key={domain}
            label={DOMAIN_LABELS[domain]}
            description={(thirdPerson ? DOMAIN_DESCRIPTIONS_THIRD_PERSON : DOMAIN_DESCRIPTIONS)[domain]}
            result={results[domain]}
            color={DOMAIN_COLORS[domain]}
            index={i}
          />
        ))}
      </div>
    </div>
  );
}
