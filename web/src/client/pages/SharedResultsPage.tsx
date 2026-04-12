import { useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { AiChatButtons } from "../components/AiChatButtons";
import { InfoCards } from "../components/InfoCards";
import { ResultsView } from "../components/ResultsView";
import { ShareCopyButton } from "../components/ShareCopyButton";
import { decodeResults } from "../lib/results-codec";
import { NotFoundPage } from "./NotFoundPage";

export function SharedResultsPage() {
  const { hash } = useParams<{ hash: string }>();
  const navigate = useNavigate();
  const results = hash ? decodeResults(hash) : null;

  useEffect(() => {
    document.title = "Shared Profile - Big Five Personality Assessment";
  }, []);

  if (!results) {
    return <NotFoundPage />;
  }

  const shareUrl = `${window.location.origin}/assessment/shared/${hash}`;

  function handleAction() {
    navigate("/assessment");
  }

  return (
    <div className="max-w-[960px] mx-auto flex flex-col gap-6 animate-in">
      <AiChatButtons results={results} mode="shared" />

      <div className="flex justify-center gap-3">
        <ShareCopyButton shareUrl={shareUrl} />
        <button
          type="button"
          className="min-h-[44px] px-4 py-2 border border-border rounded-md bg-surface text-sm text-text transition-colors hover:bg-primary-lighter active:bg-primary-light flex items-center justify-center gap-2"
          onClick={handleAction}
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M3 2l10 6-10 6V2z" /></svg>
          Take the Assessment
        </button>
      </div>

      <div className="lg:grid lg:grid-cols-[1fr_340px] lg:gap-6 lg:items-start">
        <ResultsView
          results={results}
          heading="Shared Big Five Profile"
          thirdPerson
        />

        <div className="flex flex-col gap-6 mt-6 lg:mt-0">
          <InfoCards />
        </div>
      </div>
    </div>
  );
}
