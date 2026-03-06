import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { LandingPage as LandingContent } from "../components/LandingPage";
import { clearSession, hasSession } from "../hooks/use-assessment";

export function LandingPage() {
  const navigate = useNavigate();
  const sessionExists = hasSession();

  useEffect(() => {
    document.title = "Big Five Personality Assessment";
  }, []);

  function handleStart() {
    if (sessionExists) {
      clearSession();
    }
    navigate("/start");
  }

  return (
    <div className="max-w-[720px] mx-auto">
      <LandingContent onStart={handleStart}>
        {sessionExists && (
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
