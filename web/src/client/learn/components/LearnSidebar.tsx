import { useEffect, useRef } from "react";
import { Link } from "react-router-dom";
import { cn } from "../../lib/utils";
import { getLearnPath } from "../routes";
import type { Chapter } from "../types";

interface LearnSidebarProps {
  chapters: readonly Chapter[];
  currentSlug: string;
  visited: ReadonlySet<string>;
}

export function LearnSidebar({
  chapters,
  currentSlug,
  visited,
}: LearnSidebarProps) {
  const sidebarRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const currentLink = sidebarRef.current?.querySelector<HTMLElement>(
      '[aria-current="page"]',
    );
    currentLink?.scrollIntoView({ block: "nearest" });
  }, [currentSlug]);

  return (
    <nav
      ref={sidebarRef}
      aria-label="Course navigation"
      className="min-h-0 overflow-y-auto overscroll-contain border-b border-border bg-bg px-4 py-8 lg:border-b-0 lg:border-r lg:px-5"
    >
      <div className="px-2 pb-6">
        <div className="font-body text-xs uppercase tracking-widest text-text-muted">
          BFFM-XGB Learning Course
        </div>
        <p className="mt-2 mb-3 font-display text-2xl font-semibold leading-tight text-text">
          Understand the whole pipeline
        </p>
        <p className="text-sm leading-relaxed text-text-muted">
          Linear pages for psychometrics, statistics, XGBoost, training,
          evaluation, export, and safe modification.
        </p>
      </div>
      <ol className="flex list-none flex-col gap-1.5">
        {chapters.map((chapter) => {
          const isCurrent = chapter.slug === currentSlug;
          const isVisited = visited.has(chapter.slug);
          return (
            <li key={chapter.slug}>
              <Link
                to={getLearnPath(chapter.slug)}
                aria-current={isCurrent ? "page" : undefined}
                className={cn(
                  "grid grid-cols-[2.25rem_minmax(0,1fr)] gap-3 rounded-xl border border-transparent px-3 py-3 no-underline transition-all",
                  "hover:border-border hover:bg-primary-lighter",
                  isCurrent &&
                    "border-primary/30 bg-primary-light hover:bg-primary-light",
                )}
              >
                <span
                  className={cn(
                    "font-body text-sm tabular-nums",
                    isCurrent
                      ? "font-semibold text-primary"
                      : isVisited
                        ? "text-text"
                        : "text-text-muted",
                  )}
                >
                  {String(chapter.order).padStart(2, "0")}
                </span>
                <span className="grid gap-1">
                  <strong className="font-display text-base font-semibold leading-snug text-text">
                    {chapter.title}
                  </strong>
                  <span className="text-sm leading-snug text-text-muted">
                    {chapter.kicker}
                  </span>
                </span>
              </Link>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
