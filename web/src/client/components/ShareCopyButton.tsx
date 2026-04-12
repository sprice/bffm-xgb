import { useCallback, useEffect, useRef, useState } from "react";

interface ShareCopyButtonProps {
  shareUrl: string;
}

export function ShareCopyButton({ shareUrl }: ShareCopyButtonProps) {
  const [copied, setCopied] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current !== null) {
        clearTimeout(timerRef.current);
      }
    };
  }, []);

  const handleCopy = useCallback(() => {
    if (!navigator.clipboard) {
      fallbackCopy(shareUrl);
      return;
    }
    navigator.clipboard.writeText(shareUrl).then(
      () => showCopiedFeedback(),
      () => fallbackCopy(shareUrl)
    );
  }, [shareUrl]);

  function fallbackCopy(text: string) {
    try {
      const textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      document.body.removeChild(textarea);
      showCopiedFeedback();
    } catch {
      // Copy failed silently
    }
  }

  function showCopiedFeedback() {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
    }
    setCopied(true);
    timerRef.current = setTimeout(() => {
      setCopied(false);
      timerRef.current = null;
    }, 2000);
  }

  return (
    <button
      type="button"
      aria-live="polite"
      className="min-h-[44px] px-4 py-2 border border-border rounded-md bg-surface text-sm text-text transition-colors hover:bg-primary-lighter active:bg-primary-light grid"
      onClick={handleCopy}
    >
      <span className={`col-start-1 row-start-1 flex items-center gap-2 ${copied ? "invisible" : ""}`}>
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M2 8v5a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1V8" /><polyline points="5 4 8 1 11 4" /><line x1="8" y1="1" x2="8" y2="10" /></svg>
        Share Results
      </span>
      {copied && (
        <span className="col-start-1 row-start-1 flex items-center justify-center gap-2">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M3 8.5l3.5 3.5L13 4" /></svg>
          Link Copied!
        </span>
      )}
    </button>
  );
}
