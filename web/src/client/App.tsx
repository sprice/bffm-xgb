import { lazy, Suspense, useEffect, useLayoutEffect } from "react";
import { Link, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { trackPageview } from "./lib/analytics";
import { AppHeader } from "./components/AppHeader";
import { defaultLearnPath } from "./learn/routes";
import { useTheme, type ResolvedTheme } from "./theme";
import { HomePage } from "./pages/HomePage";
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

type RouteTheme = {
  light: string;
  dark: string;
};

const BASE_THEME: RouteTheme = {
  light: "#f8f6f3",
  dark: "#1c1917",
};

const LEARN_THEME: RouteTheme = {
  light: "#f4eddc",
  dark: "#252220",
};

function setThemeMetaColors(theme: RouteTheme, resolvedTheme: ResolvedTheme): void {
  const lightMeta = document.querySelector<HTMLMetaElement>('meta[name="theme-color"][media="(prefers-color-scheme: light)"]');
  const darkMeta = document.querySelector<HTMLMetaElement>('meta[name="theme-color"][media="(prefers-color-scheme: dark)"]');
  const activeMeta = document.querySelector<HTMLMetaElement>('meta[name="theme-color"]:not([media])');

  const createMeta = (media?: string): HTMLMetaElement => {
    const created = document.createElement("meta");
    created.setAttribute("name", "theme-color");
    if (media) {
      created.setAttribute("media", media);
    }
    document.head.appendChild(created);
    return created;
  };

  const light = lightMeta ?? createMeta("(prefers-color-scheme: light)");
  const dark = darkMeta ?? createMeta("(prefers-color-scheme: dark)");
  const active = activeMeta ?? createMeta();

  light.setAttribute("content", theme.light);
  dark.setAttribute("content", theme.dark);
  active.setAttribute("content", resolvedTheme === "dark" ? theme.dark : theme.light);
}

function getRouteTheme(pathname: string): RouteTheme {
  const isLearnRoute = pathname === "/learn" || pathname.startsWith("/learn/");
  return isLearnRoute ? LEARN_THEME : BASE_THEME;
}

export function App() {
  const location = useLocation();
  const { resolvedTheme } = useTheme();
  useEffect(() => {
    trackPageview(location.pathname);
  }, [location.pathname]);

  const isLearnRoute = location.pathname === "/learn" || location.pathname.startsWith("/learn/");
  const isHomeRoute = location.pathname === "/";

  useLayoutEffect(() => {
    if (typeof document === "undefined") return;
    setThemeMetaColors(getRouteTheme(location.pathname), resolvedTheme);
  }, [location.pathname, resolvedTheme]);

  if (isLearnRoute) {
    return (
      <div className="h-svh min-h-svh flex flex-col overflow-hidden bg-bg text-text">
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-50 focus:px-4 focus:py-2 focus:bg-primary focus:text-white focus:rounded-md focus:text-sm focus:font-medium"
        >
          Skip to content
        </a>
        <AppHeader />
        <main
          id="main"
          className="w-full flex-1 min-h-0 overflow-hidden"
          aria-label="Learning content"
        >
          <Suspense>
            <Routes>
              <Route path="/learn" element={<Navigate to={defaultLearnPath} replace />} />
              <Route path="/learn/:chapterSlug" element={<LearnPage />} />
              <Route path="*" element={<NotFoundPage />} />
            </Routes>
          </Suspense>
        </main>
      </div>
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
      <AppHeader />

      <main
        id="main"
        className="w-full max-w-[960px] px-4 pb-[calc(2rem+env(safe-area-inset-bottom))] flex-1"
        aria-label="Assessment content"
      >
        <Suspense>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/assessment" element={<LandingPage />} />
            <Route path="/assessment/start" element={<PreAssessmentPage />} />
            <Route path="/assessment/question/:questionNumber" element={<AssessmentQuestionPage />} />
            <Route path="/assessment/done" element={<AssessmentDonePage />} />
            <Route path="/assessment/results" element={<ResultsPage />} />
            <Route path="/assessment/results/:hash" element={<ResultsHashPage />} />
            <Route path="/assessment/shared/:hash" element={<SharedResultsPage />} />
            <Route path="*" element={<NotFoundPage />} />
          </Routes>
        </Suspense>
      </main>

      {!isHomeRoute && (
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
      )}
    </div>
  );
}
