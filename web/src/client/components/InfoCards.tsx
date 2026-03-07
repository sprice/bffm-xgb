export function InfoCards() {
  return (
    <>
      <div className="bg-surface border border-border rounded-lg p-5 sm:p-6 shadow-sm">
        <h3 className="font-display text-lg font-semibold text-text mb-4">
          About the Big Five
        </h3>

        <blockquote className="border-l-2 border-primary/40 pl-4 mb-4">
          <p className="text-base text-text-muted italic leading-relaxed">
            "[The] five-factor model underlies most contemporary personality
            research, and the model has been described as one of the major
            breakthroughs of quantitative behavioural science. The five-factor
            structure has been confirmed by many subsequent studies across
            cultures and languages, which have replicated the original model and
            reported largely similar factors."
          </p>
          <footer className="mt-2 text-sm text-text-muted">
            —{" "}
            <a
              href="https://en.wikipedia.org/wiki/Big_Five_personality_traits"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:underline"
            >
              Wikipedia: Big Five personality traits
            </a>
          </footer>
        </blockquote>

        <p className="text-base text-text-muted leading-relaxed">
          The questions come from the{" "}
          <a
            href="https://ipip.ori.org/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary font-medium hover:underline"
          >
            International Personality Item Pool
          </a>{" "}
          (IPIP), a public domain collection of personality questions used by
          researchers around the world.
        </p>
      </div>

      <div className="bg-surface border border-border rounded-lg p-5 sm:p-6 shadow-sm">
        <h3 className="font-display text-lg font-semibold text-text mb-4">
          How it works
        </h3>

        <p className="text-base text-text-muted leading-relaxed">
          A custom machine-learning model built for this project,{" "}
          <a
            href="https://github.com/sprice/bffm-xgb"
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary font-medium hover:underline"
          >
            BFFM-XGB
          </a>{" "}
          scores your answers. It was trained on longer personality tests and
          can predict your Big Five scores from just 20 questions. And it's more
          accurate than the standard 20-question Mini Big Five assessment
          (MINI-IPIP).
        </p>

        <p className="text-base text-text-muted leading-relaxed mt-3">
          Everything is open source on{" "}
          <a
            href="https://github.com/sprice/bffm-xgb"
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary font-medium hover:underline"
          >
            GitHub
          </a>
          : the model, the training code, and this website. The model is public
          domain on{" "}
          <a
            href="https://huggingface.co/shawnprice/bffm-xgb"
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary font-medium hover:underline"
          >
            Hugging Face
          </a>
          .
        </p>
      </div>
    </>
  );
}
