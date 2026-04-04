import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { LandingPage as LandingContent } from "../components/LandingPage";
import { clearSession, clearResultsHash, hasSession, loadResultsHash } from "../hooks/use-assessment";

const VALID_HASH = /^[A-Za-z0-9_-]{10,30}$/;

export function LandingPage() {
  const navigate = useNavigate();
  const sessionExists = hasSession();
  const raw = loadResultsHash();
  const resultsHash = raw && VALID_HASH.test(raw) ? raw : null;
  const hasSavedState = Boolean(resultsHash || sessionExists);

  useEffect(() => {
    document.title = "Big Five Personality Assessment";
  }, []);

  function handleStart() {
    if (sessionExists) {
      clearSession();
    }
    clearResultsHash();
    navigate("/start");
  }

  return (
    <div className="max-w-[720px] mx-auto">
      <LandingContent
        onStart={handleStart}
        primaryActionLabel={hasSavedState ? "Start Over" : "Take the Assessment"}
      >
        {resultsHash && (
          <button
            type="button"
            className="min-h-[52px] px-10 py-4 border border-border rounded-lg text-lg font-bold bg-surface text-text transition-all hover:bg-primary-lighter hover:-translate-y-0.5 active:translate-y-0 active:bg-primary-light"
            onClick={() => navigate(`/results/${resultsHash}`)}
          >
            View Results
          </button>
        )}
        {!resultsHash && sessionExists && (
          <button
            type="button"
            className="min-h-[52px] px-10 py-4 border border-border rounded-lg text-lg font-bold bg-surface text-text transition-all hover:bg-primary-lighter hover:-translate-y-0.5 active:translate-y-0 active:bg-primary-light"
            onClick={() => navigate("/assessment")}
          >
            Resume Assessment
          </button>
        )}
      </LandingContent>
    </div>
  );
}
