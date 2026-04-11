import confetti from "canvas-confetti";
import { useEffect, useState } from "react";
import { Navigate, useNavigate } from "react-router-dom";
import { ProgressBar } from "../components/ProgressBar";
import { clearSession, useAssessment } from "../hooks/use-assessment";
import type { PredictionResult } from "../types";

export function AssessmentDonePage() {
  const navigate = useNavigate();
  const { totalItems, answeredCount, allAnswered, rawResponses } =
    useAssessment();

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    document.title = "All Done - Big Five Personality Assessment";
    confetti({
      particleCount: 60,
      spread: 55,
      origin: { y: 0.7 },
      disableForReducedMotion: true,
    });
  }, []);

  if (!allAnswered) {
    return <Navigate to="/assessment" replace />;
  }

  async function handleSubmit() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ responses: rawResponses }),
      });
      const data = (await res.json()) as {
        results?: PredictionResult;
        error?: string;
      };
      if (!res.ok) {
        throw new Error(data.error || "Prediction failed");
      }
      clearSession();
      navigate("/assessment/results", { state: { results: data.results }, replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-[540px] mx-auto">
      <ProgressBar current={answeredCount} total={totalItems} />

      <div className="bg-surface border border-border rounded-lg p-5 sm:p-6 shadow-sm flex flex-col items-center justify-center min-h-[380px] animate-in">
        <h2 className="font-display text-4xl font-semibold mb-2">You're all done!</h2>
        <p className="text-text-muted text-base mb-6">
          All {totalItems} questions answered. Ready to see your results?
        </p>

        <button
          type="button"
          className="min-h-[48px] px-9 py-3.5 bg-primary text-white rounded-lg text-lg font-bold shadow-lg hover:bg-primary-hover hover:shadow-xl hover:-translate-y-0.5 active:translate-y-0 active:shadow-md transition-all disabled:opacity-60 disabled:cursor-not-allowed"
          onClick={handleSubmit}
          disabled={loading}
        >
          {loading ? "Calculating..." : "Get Your Results"}
        </button>

        {error && (
          <div
            className="mt-4 p-3 bg-primary-lighter border border-primary/20 rounded-md text-text text-base"
            role="alert"
          >
            <p>{error}</p>
            <button
              type="button"
              className="mt-2 min-h-[44px] px-4 py-2 text-sm font-medium text-primary hover:underline"
              onClick={handleSubmit}
            >
              Try again
            </button>
          </div>
        )}
      </div>

      <nav className="flex flex-wrap justify-between items-center mt-5 gap-3" aria-label="Question navigation">
        <button
          type="button"
          aria-label="Previous question"
          className="min-h-[44px] min-w-[44px] px-4 py-2.5 border border-border rounded-md bg-surface text-sm text-text transition-colors hover:bg-primary-lighter active:bg-primary-light"
          onClick={() => navigate(`/assessment/question/${totalItems}`)}
        >
          &larr; Previous
        </button>
      </nav>
    </div>
  );
}
