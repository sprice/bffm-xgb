import { lazy, Suspense, useEffect } from "react";
import { Link, Route, Routes, useLocation } from "react-router-dom";
import { trackPageview } from "./lib/analytics";
import { AssessmentPage } from "./pages/AssessmentPage";
import { LandingPage } from "./pages/LandingPage";
import { NotFoundPage } from "./pages/NotFoundPage";
import { ResultsPage } from "./pages/ResultsPage";

const PreAssessmentPage = lazy(() =>
  import("./pages/PreAssessmentPage").then((m) => ({
    default: m.PreAssessmentPage,
  })),
);
const AssessmentQuestionPage = lazy(() =>
  import("./pages/AssessmentQuestionPage").then((m) => ({
    default: m.AssessmentQuestionPage,
  })),
);
const AssessmentDonePage = lazy(() =>
  import("./pages/AssessmentDonePage").then((m) => ({
    default: m.AssessmentDonePage,
  })),
);
const ResultsHashPage = lazy(() =>
  import("./pages/ResultsHashPage").then((m) => ({
    default: m.ResultsHashPage,
  })),
);
const SharedResultsPage = lazy(() =>
  import("./pages/SharedResultsPage").then((m) => ({
    default: m.SharedResultsPage,
  })),
);
const LearnPage = lazy(() =>
  import("./pages/LearnPage").then((m) => ({ default: m.LearnPage })),
);

export function App() {
  const location = useLocation();
  useEffect(() => {
    trackPageview(location.pathname);
  }, [location.pathname]);

  const isLearnRoute = location.pathname === "/learn" || location.pathname.startsWith("/learn/");
  if (isLearnRoute) {
    return (
      <main className="min-h-svh">
        <Suspense>
          <Routes>
            <Route path="/learn/*" element={<LearnPage />} />
            <Route
              path="*"
              element={
                <div className="pt-8 px-4 min-h-svh text-text text-sm">Not found</div>
              }
            />
          </Routes>
        </Suspense>
      </main>
    );
  }

  return (
    <div className="min-h-svh flex flex-col items-center bg-bg">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-50 focus:px-4 focus:py-2 focus:bg-primary focus:text-white focus:rounded-md focus:text-sm focus:font-medium"
      >
        Skip to content
      </a>

      <header className="text-center pt-12 sm:pt-16 pb-8 sm:pb-10 px-4 animate-in">
        <h1 className="font-display text-4xl sm:text-5xl font-semibold tracking-tight text-text leading-none">
          Big Five
        </h1>
        <p className="mt-2 text-xs sm:text-sm text-text-muted tracking-[0.25em] uppercase font-medium">
          Personality Assessment
        </p>
      </header>

      <main
        id="main"
        className="w-full max-w-[960px] px-4 pb-[calc(2rem+env(safe-area-inset-bottom))] flex-1"
        aria-label="Assessment content"
      >
        <Suspense>
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/start" element={<PreAssessmentPage />} />
            <Route path="/assessment" element={<AssessmentPage />} />
            <Route path="/assessment/done" element={<AssessmentDonePage />} />
            <Route
              path="/assessment/:questionNumber"
              element={<AssessmentQuestionPage />}
            />
            <Route path="/results" element={<ResultsPage />} />
            <Route path="/results/:hash" element={<ResultsHashPage />} />
            <Route path="/shared/:hash" element={<SharedResultsPage />} />
            <Route path="*" element={<NotFoundPage />} />
          </Routes>
        </Suspense>
      </main>

      <footer className="text-center py-8 px-4 pb-[calc(2rem+env(safe-area-inset-bottom))] text-text-muted text-sm border-t border-border w-full">
        <p>
          <Link className="text-primary hover:underline" to="/">
            Big 5 Personality Assessment
          </Link>
        </p>
        <p className="mt-1 opacity-60">
          For educational purposes only · Code available on{" "}
          <a
            href="https://github.com/sprice/bffm-xgb"
            target="_blank"
            rel="noopener noreferrer"
            className="underline hover:opacity-80"
          >
            GitHub
          </a>
        </p>
      </footer>
    </div>
  );
}
