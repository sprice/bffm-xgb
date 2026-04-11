import { useEffect, useLayoutEffect, useRef, useState, type RefObject } from "react";

type TooltipState = {
  description: string;
  trigger: HTMLElement;
};

interface GlossaryTooltipProps {
  rootRef: RefObject<HTMLElement | null>;
}

const SPACING = 10;
const VIEWPORT_INSET = 12;

function positionTooltip(tooltip: HTMLElement, trigger: HTMLElement): void {
  const triggerRect = trigger.getBoundingClientRect();
  const tooltipRect = tooltip.getBoundingClientRect();

  let top = triggerRect.top - tooltipRect.height - SPACING;
  if (top < VIEWPORT_INSET) {
    top = triggerRect.bottom + SPACING;
  }

  let left = triggerRect.left + triggerRect.width / 2 - tooltipRect.width / 2;
  left = Math.max(
    VIEWPORT_INSET,
    Math.min(left, window.innerWidth - tooltipRect.width - VIEWPORT_INSET),
  );

  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${top}px`;
}

/**
 * Event-delegated tooltip for `[data-glossary-term]` buttons inside `rootRef`.
 *
 * Chapter content is authored as HTML strings via `dangerouslySetInnerHTML`,
 * so we can't wrap each trigger in a React component. Instead we delegate
 * mouse/focus events from the article root and render the tooltip as a
 * conditional React element. `position: fixed` means its DOM location
 * doesn't affect layout — it's a sibling of the shell inside `LearnShell`.
 *
 * The tooltip is unmounted when closed (not hidden via opacity) so there's
 * no chance of an invisible-but-present element distorting layout or
 * interfering with event propagation.
 */
export function GlossaryTooltip({ rootRef }: GlossaryTooltipProps) {
  const [show, setShow] = useState<TooltipState | null>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);

  // Position the tooltip synchronously after mount, before paint.
  useLayoutEffect(() => {
    if (!show || !tooltipRef.current) return;
    positionTooltip(tooltipRef.current, show.trigger);
  }, [show]);

  useEffect(() => {
    const root: HTMLElement | null = rootRef.current;
    if (!root) return;
    const rootEl = root;

    let activeTrigger: HTMLElement | null = null;

    function open(trigger: HTMLElement): void {
      const description = trigger.dataset.tooltip;
      if (!description) return;
      if (activeTrigger && activeTrigger !== trigger) {
        activeTrigger.removeAttribute("data-tooltip-open");
      }
      activeTrigger = trigger;
      trigger.setAttribute("data-tooltip-open", "true");
      setShow({ description, trigger });
    }

    function close(): void {
      if (activeTrigger) {
        activeTrigger.removeAttribute("data-tooltip-open");
        activeTrigger = null;
      }
      setShow(null);
    }

    function findTrigger(target: EventTarget | null): HTMLElement | null {
      if (!(target instanceof Element)) return null;
      const el = target.closest<HTMLElement>("[data-glossary-term]");
      if (!el || !rootEl.contains(el)) return null;
      return el;
    }

    function onMouseOver(e: MouseEvent): void {
      const trigger = findTrigger(e.target);
      if (trigger) open(trigger);
    }

    function onMouseOut(e: MouseEvent): void {
      const trigger = findTrigger(e.target);
      if (!trigger) return;
      if (document.activeElement === trigger) return;
      close();
    }

    function onFocusIn(e: FocusEvent): void {
      const trigger = findTrigger(e.target);
      if (trigger) open(trigger);
    }

    function onFocusOut(e: FocusEvent): void {
      const trigger = findTrigger(e.target);
      if (trigger) close();
    }

    function onKeyDown(e: KeyboardEvent): void {
      if (e.key === "Escape") close();
    }

    function onReposition(): void {
      if (!activeTrigger || !tooltipRef.current) return;
      positionTooltip(tooltipRef.current, activeTrigger);
    }

    rootEl.addEventListener("mouseover", onMouseOver);
    rootEl.addEventListener("mouseout", onMouseOut);
    rootEl.addEventListener("focusin", onFocusIn);
    rootEl.addEventListener("focusout", onFocusOut);
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("resize", onReposition);
    window.addEventListener("scroll", onReposition, true);

    return () => {
      rootEl.removeEventListener("mouseover", onMouseOver);
      rootEl.removeEventListener("mouseout", onMouseOut);
      rootEl.removeEventListener("focusin", onFocusIn);
      rootEl.removeEventListener("focusout", onFocusOut);
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("resize", onReposition);
      window.removeEventListener("scroll", onReposition, true);
      if (activeTrigger) {
        activeTrigger.removeAttribute("data-tooltip-open");
      }
    };
  }, [rootRef]);

  if (!show) return null;

  return (
    <div
      ref={tooltipRef}
      role="tooltip"
      className="pointer-events-none fixed left-0 top-0 z-[1000] max-w-[min(20rem,calc(100vw-1.5rem))]"
    >
      <div className="rounded-lg border border-white/10 bg-[#1a1816] px-3 py-2.5 font-body text-sm leading-snug text-[#f2f0ed] shadow-xl">
        {show.description}
      </div>
    </div>
  );
}
