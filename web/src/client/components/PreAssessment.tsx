interface PreAssessmentProps {
  onBegin: () => void;
}

export function PreAssessment({ onBegin }: PreAssessmentProps) {
  return (
    <div className="pt-2 animate-in">
      <div className="bg-surface border border-border rounded-lg px-5 sm:px-6 py-8 shadow-sm">
        <h2 className="font-display text-3xl font-semibold text-center mb-8 text-text">Before you begin</h2>

        <ul className="list-none flex flex-col gap-6 mb-10">
          <li className="flex gap-4 items-start text-base text-text leading-relaxed">
            <span className="shrink-0 w-9 h-9 flex items-center justify-center bg-primary-light rounded-full text-lg leading-none text-primary mt-0.5">&#9200;</span>
            <div>
              <strong className="font-semibold">This takes about 2-3 minutes.</strong> There are 20
              questions, each with five response options. No right or wrong
              answers.
            </div>
          </li>

          <li className="flex gap-4 items-start text-base text-text leading-relaxed">
            <span className="shrink-0 w-9 h-9 flex items-center justify-center bg-primary-light rounded-full text-lg leading-none text-primary mt-0.5">&#9829;</span>
            <div>
              <strong className="font-semibold">Answer based on how you generally are</strong>, not how
              you feel right now or how you'd like to be. Think about your
              typical behavior over the past year or so.
            </div>
          </li>

          <li className="flex gap-4 items-start text-base text-text leading-relaxed">
            <span className="shrink-0 w-9 h-9 flex items-center justify-center bg-primary-light rounded-full text-lg leading-none text-primary mt-0.5">&#9978;</span>
            <div>
              <strong className="font-semibold">Try to take this in a calm, neutral state.</strong> If
              you're feeling stressed, excited, or wound up, your
              results may not reflect your usual self. Come back later if now
              isn't the right moment.
            </div>
          </li>

          <li className="flex gap-4 items-start text-base text-text leading-relaxed">
            <span className="shrink-0 w-9 h-9 flex items-center justify-center bg-primary-light rounded-full text-lg leading-none text-primary mt-0.5">&#9432;</span>
            <div>
              <strong className="font-semibold">This is not a clinical or diagnostic tool.</strong> It's
              designed for personal insight and curiosity: a snapshot, not a
              verdict.
            </div>
          </li>
        </ul>

        <button type="button" className="block w-full min-h-[52px] py-3.5 px-6 bg-primary text-white rounded-lg text-lg font-bold hover:bg-primary-hover active:bg-primary-hover transition-colors shadow-sm" onClick={onBegin}>
          Begin Assessment
        </button>
      </div>
    </div>
  );
}
