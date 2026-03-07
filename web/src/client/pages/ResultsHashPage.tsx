import { useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { AiChatButtons } from "../components/AiChatButtons";
import { ResultsView } from "../components/ResultsView";
import { ShareCopyButton } from "../components/ShareCopyButton";
import { clearSession, clearResultsHash, saveResultsHash } from "../hooks/use-assessment";
import { decodeResults } from "../lib/results-codec";
import { NotFoundPage } from "./NotFoundPage";

export function ResultsHashPage() {
  const { hash } = useParams<{ hash: string }>();
  const navigate = useNavigate();
  const results = hash ? decodeResults(hash) : null;

  useEffect(() => {
    document.title = "Your Results - Big Five Personality Assessment";
    if (hash && results) {
      saveResultsHash(hash);
    }
  }, [hash]);

  if (!results) {
    return <NotFoundPage />;
  }

  const shareUrl = `${window.location.origin}/shared/${hash}`;

  function handleRetake() {
    clearSession();
    clearResultsHash();
    navigate("/start", { replace: true });
  }

  return (
    <div className="flex flex-col gap-6 animate-in">
      <AiChatButtons results={results} mode="self" />

      <div className="flex flex-wrap justify-center gap-3">
        <ShareCopyButton shareUrl={shareUrl} />
        <button
          type="button"
          className="min-h-[36px] px-4 py-2 border border-border rounded-md bg-surface text-sm text-text transition-colors hover:bg-primary-lighter active:bg-primary-light flex items-center justify-center gap-2"
          onClick={handleRetake}
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M1.5 2.5v4h4" /><path d="M2.3 10a6 6 0 1 0 1.2-6.2L1.5 6.5" /></svg>
          Retake Assessment
        </button>
      </div>

      <div className="max-w-[540px] mx-auto w-full">
        <ResultsView results={results} />
      </div>
    </div>
  );
}
