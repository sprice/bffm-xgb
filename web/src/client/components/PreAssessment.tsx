interface PreAssessmentProps {
  onBegin: () => void;
}

export function PreAssessment({ onBegin }: PreAssessmentProps) {
  return (
    <div className="pt-2 animate-in">
      <div className="bg-surface border border-border rounded-lg px-5 sm:px-6 py-8 shadow-sm">
        <h2 className="font-display text-3xl font-semibold text-center mb-8 text-text">
          Before you begin
        </h2>

        <div className="max-w-[520px] mx-auto mb-10 flex flex-col gap-5 text-base text-text-muted leading-relaxed">
          <p>
            <strong className="text-text font-semibold">
              Answer based on how you generally are
            </strong>
            , not how you feel right now or how you'd like to be. Think about
            your typical behavior over the past year or so.
          </p>
          <p>
            <strong className="text-text font-semibold">
              Take this in a calm, neutral state.
            </strong>{" "}
            If you're feeling stressed, excited, or wound up, your results may
            not reflect your usual self. Come back later if now isn't the right
            moment.
          </p>
          <p>
            <strong className="text-text font-semibold">
              Not a clinical or diagnostic tool.
            </strong>{" "}
            A snapshot for personal insight and curiosity, not a verdict.
          </p>
        </div>

        <button
          type="button"
          className="block w-full min-h-[52px] py-3.5 px-6 bg-primary text-white rounded-lg text-lg font-bold hover:bg-primary-hover active:bg-primary-hover transition-colors shadow-sm"
          onClick={onBegin}
        >
          Begin Assessment
        </button>
      </div>
    </div>
  );
}
