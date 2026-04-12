import { Link } from "react-router-dom";
import { defaultLearnPath } from "../learn/routes";

export function HomePage() {
  return (
    <div className="min-h-[70vh] flex flex-col items-center justify-center gap-10 px-4 py-12">
      <div className="max-w-prose space-y-5 text-xl leading-9 text-text-muted text-left">
        <h2 className="text-text text-3xl">History of the Big Five</h2>
        <p>
          The Big Five personality model is the most validated framework in
          personality psychology. Decades of academic research across dozens of
          languages and cultures consistently reproduce the same five
          dimensions: Extraversion, Agreeableness, Conscientiousness, Emotional
          Stability, and Intellect/Imagination. Popular alternatives such as
          Myers-Briggs, Enneagram, DISC, and others lack this empirical and
          academic foundation.
        </p>

        <p>
          In 1884, Sir Francis Galton proposed what became known as the lexical
          hypothesis: if a personality trait matters to people, there will be a
          word for it. He combed through dictionaries looking for
          personality-describing terms. In 1936, Gordon Allport and Henry Odbert
          catalogued roughly 4,500 of these terms. Raymond Cattell narrowed
          these through factor analysis in the 1940s. In 1961, Ernest Tupes and
          Raymond Christal identified five recurring factors that continued to
          emerge. Lewis Goldberg formalized the &ldquo;Big Five&rdquo; label in
          the early 1980s. When researchers repeated the process across
          languages, the same five factors appeared. The structure reflects
          something real about how humans everywhere organize personality.
        </p>

        <p>
          Several open assessments now make Big Five measurement freely
          available. The IPIP (International Personality Item Pool), created by
          Goldberg, provides thousands of public-domain items. Its 50-item BFFM
          (Big Five Factor Markers) is among the most widely used. The BFAS (Big
          Five Aspect Scales), developed by Colin DeYoung, Lena Quilty, and
          Jordan Peterson in 2007, breaks each domain into two finer-grained
          aspects to give a more detailed profile. Shorter instruments like the
          Mini-IPIP compress the assessment to 20 items.
        </p>

        <h2 className="mt-16 text-text text-3xl">BFFM-XGB ML Model</h2>
        <p>
          This project started when I read a{" "}
          <a
            href="https://osf.io/preprints/psyarxiv/ysd3f_v1"
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary hover:underline"
          >
            2020 paper by Gl&ouml;ckner, Michels, and Giersch
          </a>
          . They showed that neural networks could predict full personality
          scores from a short questionnaire. Near the end, they floated the idea
          of adaptive assessments, where each question depends on your previous
          answers. They didn&rsquo;t explore it.
        </p>

        <p>
          I wanted to find out if it would work and built a machine learning
          model to test the idea. I was able to invalidate the adaptive part,
          but the model itself turned out to be good at something else:
          predicting reliable Big Five scores from just 20 fixed questions, with
          a confidence range that shows where the uncertainty actually is.
        </p>

        <p>
          The code is open source on{" "}
          <a
            href="https://github.com/sprice/bffm-xgb"
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary hover:underline"
          >
            GitHub
          </a>
          . The models are public domain on{" "}
          <a
            href="https://huggingface.co/shawnprice/bffm-xgb"
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary hover:underline"
          >
            Hugging Face
          </a>
          .
        </p>
        <p>
          —{" "}
          <a
            href="https://shawnprice.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary hover:underline"
          >
            Shawn Price
          </a>
        </p>
      </div>

      <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
        <Link
          to="/assessment"
          className="min-h-[52px] min-w-[200px] px-6 py-3 bg-primary text-white rounded-lg text-xl font-bold text-center hover:bg-primary-hover active:bg-primary-hover transition-colors"
        >
          Take the assessment
        </Link>
        <Link
          to={defaultLearnPath}
          className="min-h-[52px] min-w-[200px] px-6 py-3 border border-border rounded-lg text-xl font-bold text-text text-center bg-surface hover:bg-primary-lighter transition-colors"
        >
          Learn about the project
        </Link>
      </div>
    </div>
  );
}
