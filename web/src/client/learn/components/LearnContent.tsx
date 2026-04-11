import { forwardRef, useEffect } from "react";
import { cn } from "../../lib/utils";
import type { Chapter } from "../types";

interface LearnContentProps {
  chapter: Chapter;
}

/**
 * Renders a chapter's authored HTML content and runs its `afterRender` hook
 * so widgets (reverse-score, z-score, quantile-loss, SEM) can mount against
 * the freshly-inserted DOM.
 *
 * Chapter bodies are HTML strings produced by the helpers in
 * `content/helpers.ts` — we render them with `dangerouslySetInnerHTML` rather
 * than converting 2,000+ lines of chapter source to JSX.
 *
 * The article wrapper:
 *   1. Is the visual card (rounded surface + border + shadow + responsive padding)
 *   2. Sets the base type for inheritance — Fraunces, text-base, leading-relaxed
 *   3. Owns the only structural CSS via Tailwind arbitrary variants:
 *      - `[&>section+section]:*` handles section-to-section spacing and divider
 *      - `[&_code]:*` gives inline + block code a consistent monospace base
 *      - `[&_:not(pre)>code]:*` gives inline code its subtle bg/padding pill
 *
 * No `.learn-*` CSS classes are used for styling — everything is Tailwind.
 */
export const LearnContent = forwardRef<HTMLElement, LearnContentProps>(
  function LearnContent({ chapter }, ref) {
    useEffect(() => {
      chapter.afterRender?.();
    }, [chapter]);

    return (
      <article
        ref={ref}
        className={cn(
          "rounded-2xl border border-border bg-surface shadow-sm",
          "p-6 sm:p-8 lg:p-10",
          "font-display text-text text-base leading-relaxed",
          "[&>section+section]:mt-8 [&>section+section]:pt-8",
          "[&>section+section]:border-t [&>section+section]:border-border",
          "[&_code]:font-mono [&_code]:text-[0.9em]",
          "[&_:not(pre)>code]:rounded [&_:not(pre)>code]:bg-text/10",
          "[&_:not(pre)>code]:px-1 [&_:not(pre)>code]:py-0.5",
        )}
        dangerouslySetInnerHTML={{ __html: chapter.content }}
      />
    );
  },
);
