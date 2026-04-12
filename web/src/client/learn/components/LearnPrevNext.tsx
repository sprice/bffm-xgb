import { Link } from "react-router-dom";
import { cn } from "../../lib/utils";
import { getLearnPath } from "../routes";
import type { Chapter } from "../types";

interface LearnPrevNextProps {
  prev: Chapter | null;
  next: Chapter | null;
}

export function LearnPrevNext({ prev, next }: LearnPrevNextProps) {
  if (!prev && !next) return null;

  return (
    <div className="mt-10 grid gap-4 sm:grid-cols-2">
      {prev && (
        <NavCard chapter={prev} direction="prev" className="sm:col-start-1" />
      )}
      {next && (
        <NavCard chapter={next} direction="next" className="sm:col-start-2" />
      )}
    </div>
  );
}

function NavCard({
  chapter,
  direction,
  className,
}: {
  chapter: Chapter;
  direction: "prev" | "next";
  className?: string;
}) {
  const isNext = direction === "next";
  return (
    <Link
      to={getLearnPath(chapter.slug)}
      className={cn(
        "grid gap-1 rounded-xl border border-border bg-surface p-5 text-text no-underline shadow-sm transition-colors hover:border-primary hover:bg-primary/5",
        isNext && "sm:text-right",
        className,
      )}
    >
      <span className="font-body text-xs uppercase tracking-widest text-text-muted">
        {isNext ? (
          <>
            Next <span aria-hidden="true">→</span>
          </>
        ) : (
          <>
            <span aria-hidden="true">←</span> Previous
          </>
        )}
      </span>
      <strong className="font-display text-lg font-semibold leading-snug text-text">
        {chapter.title}
      </strong>
    </Link>
  );
}
