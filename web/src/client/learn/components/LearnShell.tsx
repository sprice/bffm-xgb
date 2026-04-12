import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { cn } from "../../lib/utils";
import { chapters } from "../content";
import { getLearnPath } from "../routes";
import type { Chapter } from "../types";
import { GlossaryTooltip } from "./GlossaryTooltip";
import { LearnContent } from "./LearnContent";
import { LearnPrevNext } from "./LearnPrevNext";
import { LearnSidebar } from "./LearnSidebar";
import { LearnTopbar } from "./LearnTopbar";

const visitedStorageKey = "bffm-xgb-learn-visited";

function loadVisited(): Set<string> {
  if (typeof window === "undefined") return new Set();
  try {
    const raw = window.localStorage.getItem(visitedStorageKey);
    if (!raw) return new Set();
    const parsed = JSON.parse(raw) as string[];
    return new Set(parsed);
  } catch {
    return new Set();
  }
}

function saveVisited(visited: ReadonlySet<string>): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(visitedStorageKey, JSON.stringify([...visited]));
  } catch {
    // Strict/private storage modes: ignore, keep the reader usable.
  }
}

function detectMathMlSupport(): boolean {
  if (typeof document === "undefined") return false;
  const probe = document.createElement("div");
  probe.style.position = "absolute";
  probe.style.visibility = "hidden";
  probe.style.pointerEvents = "none";
  probe.style.inset = "-9999px auto auto -9999px";
  probe.innerHTML =
    '<math xmlns="http://www.w3.org/1998/Math/MathML"><mspace height="23px" width="77px"></mspace></math>';
  document.body.appendChild(probe);
  const rect = probe.getBoundingClientRect();
  probe.remove();
  return Math.abs(rect.width - 77) <= 1 && Math.abs(rect.height - 23) <= 1;
}

function estimatedMinutes(chapter: Chapter): number {
  if (typeof document === "undefined") return 3;
  const temp = document.createElement("div");
  temp.innerHTML = chapter.content;
  const words = (temp.textContent ?? "").trim().split(/\s+/).filter(Boolean).length;
  return Math.max(3, Math.round(words / 180));
}

interface LearnShellProps {
  chapter: Chapter;
}

export function LearnShell({ chapter }: LearnShellProps) {
  const navigate = useNavigate();
  const articleRef = useRef<HTMLElement>(null);
  const [visited, setVisited] = useState<Set<string>>(() => loadVisited());
  const [mathMlMode, setMathMlMode] = useState<"supported" | "fallback" | null>(
    null,
  );

  useEffect(() => {
    setMathMlMode(detectMathMlSupport() ? "supported" : "fallback");
  }, []);

  useEffect(() => {
    setVisited((prev) => {
      if (prev.has(chapter.slug)) return prev;
      const next = new Set(prev);
      next.add(chapter.slug);
      saveVisited(next);
      return next;
    });
  }, [chapter.slug]);

  const handleNavigate = useCallback(
    (slug: string) => navigate(getLearnPath(slug)),
    [navigate],
  );

  useEffect(() => {
    function handleKeydown(e: KeyboardEvent): void {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }
      const currentIndex = chapters.findIndex((c) => c.slug === chapter.slug);
      if (currentIndex < 0) return;
      if (e.key === "ArrowRight" && currentIndex < chapters.length - 1) {
        handleNavigate(chapters[currentIndex + 1].slug);
      }
      if (e.key === "ArrowLeft" && currentIndex > 0) {
        handleNavigate(chapters[currentIndex - 1].slug);
      }
    }
    window.addEventListener("keydown", handleKeydown);
    return () => window.removeEventListener("keydown", handleKeydown);
  }, [chapter.slug, handleNavigate]);

  const { index, readMinutes, prev, next, progress } = useMemo(() => {
    const i = chapters.findIndex((c) => c.slug === chapter.slug);
    return {
      index: i,
      readMinutes: estimatedMinutes(chapter),
      prev: i > 0 ? chapters[i - 1] : null,
      next: i < chapters.length - 1 ? chapters[i + 1] : null,
      progress: (i + 1) / chapters.length,
    };
  }, [chapter]);

  return (
    <div
      className={cn(
        "h-full min-h-0 overflow-hidden",
        mathMlMode === "supported" && "mathml-supported",
        mathMlMode === "fallback" && "mathml-fallback",
      )}
    >
      <div className="grid h-full min-h-0 grid-cols-1 overflow-hidden lg:grid-cols-[minmax(18rem,22rem)_minmax(0,1fr)]">
        <LearnSidebar
          chapters={chapters}
          currentSlug={chapter.slug}
          visited={visited}
        />
        <main
          id="main-content"
          className="min-h-0 overflow-y-auto overscroll-contain px-6 pt-10 pb-24 sm:px-10 lg:px-12"
        >
          <div className="mx-auto max-w-[56rem]">
            <LearnTopbar
              chapter={chapter}
              index={index}
              total={chapters.length}
              readMinutes={readMinutes}
            />
            <div
              className="mt-8 mb-8 h-2 w-full overflow-hidden rounded-full bg-border/40"
              aria-hidden="true"
            >
              <div
                className="h-full w-full origin-left rounded-full bg-primary transition-transform duration-500 ease-out"
                style={{ transform: `scaleX(${progress})` }}
              />
            </div>
            <LearnContent ref={articleRef} chapter={chapter} />
            <LearnPrevNext prev={prev} next={next} />
          </div>
        </main>
      </div>
      <GlossaryTooltip rootRef={articleRef} />
    </div>
  );
}
