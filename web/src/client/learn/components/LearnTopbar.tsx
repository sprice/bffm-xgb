import type { Chapter } from "../types";

interface LearnTopbarProps {
  chapter: Chapter;
  index: number;
  total: number;
  readMinutes: number;
}

export function LearnTopbar({
  chapter,
  index,
  total,
  readMinutes,
}: LearnTopbarProps) {
  return (
    <header className="flex flex-col gap-6 sm:flex-row sm:items-end sm:justify-between">
      <div>
        <div className="font-body text-xs uppercase tracking-widest text-text-muted">
          Chapter {String(chapter.order).padStart(2, "0")}
        </div>
        <h1 className="mt-2 mb-3 font-display text-4xl font-semibold leading-tight text-text lg:text-5xl">
          {chapter.title}
        </h1>
        <p className="max-w-2xl text-base leading-relaxed text-text-muted">
          {chapter.summary}
        </p>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <MetaCard label="Read time" value={`${readMinutes} min`} />
        <MetaCard label="Progress" value={`${index + 1} / ${total}`} />
      </div>
    </header>
  );
}

function MetaCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-border bg-surface p-4 shadow-sm">
      <span className="block font-body text-xs uppercase tracking-widest text-text-muted">
        {label}
      </span>
      <strong className="mt-1 block font-display text-lg font-semibold text-text">
        {value}
      </strong>
    </div>
  );
}
