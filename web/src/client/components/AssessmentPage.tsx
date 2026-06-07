import { InfoCards } from "./InfoCards";

interface AssessmentContentProps {
  onStart: () => void;
  primaryActionLabel: string;
  children?: React.ReactNode;
}

export function AssessmentPage({
  onStart,
  primaryActionLabel,
  children,
}: AssessmentContentProps) {
  return (
    <div className="flex flex-col gap-10">
      <div className="pt-2 pb-4">
        <h2 className="font-display text-6xl max-sm:text-4xl font-semibold leading-tight text-text tracking-tight mb-5 animate-in">
          Discover your personality
          <br />
          <span className="text-text-muted">in under 3 minutes</span>
        </h2>

        <p className="text-base text-text-muted tracking-wide uppercase font-medium mb-8 animate-in stagger-1">
          Built on decades of peer-reviewed psychology
        </p>

        <p className="text-lg max-sm:text-base text-text-muted leading-relaxed max-w-[520px] mb-10 animate-in stagger-2">
          Answer 20 questions and see where you stand on each Big Five
          personality trait.
        </p>

        <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 animate-in stagger-3">
          <button
            type="button"
            className="min-h-[52px] px-10 py-4 bg-primary text-white rounded-lg text-lg font-bold shadow-lg hover:bg-primary-hover hover:shadow-xl hover:-translate-y-0.5 active:translate-y-0 active:shadow-md transition-all"
            onClick={onStart}
          >
            {primaryActionLabel}
          </button>

          {children}
        </div>
      </div>

      <div className="flex flex-col gap-8 animate-in stagger-4">
        <InfoCards />
      </div>

      <div className="py-6 animate-in stagger-5">
        <h3 className="font-display text-lg font-semibold text-text mb-3">
          Your privacy
        </h3>

        <p className="text-base text-text-muted leading-relaxed">
          No account needed, and your assessment answers are never stored. When
          you finish, your answers are sent to the server for scoring and immediately discarded;
          nothing is saved on our end. In-progress answers live in your browser
          so you can pick up where you left off. We use cookie-free,
          privacy-friendly analytics (Umami) to count visits; your IP address
          and approximate location are used only to produce aggregate visit and
          geographic statistics, and are not stored by us.
        </p>
      </div>
    </div>
  );
}
