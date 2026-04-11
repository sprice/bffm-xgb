import type { Chapter } from "./types";
import { chapters } from "./content";
import { getLearnPath } from "./routes";

const visitedStorageKey = "bffm-xgb-learn-visited";

function detectMathMlSupport(): boolean {
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

function loadVisited(): Set<string> {
  try {
    const raw = window.localStorage.getItem(visitedStorageKey);
    if (!raw) return new Set();
    const parsed = JSON.parse(raw) as string[];
    return new Set(parsed);
  } catch {
    return new Set();
  }
}

function saveVisited(visited: Set<string>): void {
  try {
    window.localStorage.setItem(visitedStorageKey, JSON.stringify([...visited]));
  } catch {
    // Ignore storage failures so the reader experience still works in strict/private environments.
  }
}

function chapterBySlug(slug: string): Chapter {
  return chapters.find((chapter) => chapter.slug === slug) ?? chapters[0];
}

function plainText(html: string): string {
  const temp = document.createElement("div");
  temp.innerHTML = html;
  return temp.textContent ?? "";
}

function estimatedMinutes(chapter: Chapter): number {
  const words = plainText(chapter.content).trim().split(/\s+/).filter(Boolean).length;
  return Math.max(3, Math.round(words / 180));
}

function renderSidebar(current: Chapter, visited: Set<string>): string {
  return `
    <nav class="sidebar" aria-label="Course navigation">
      <div class="sidebar-header">
        <div class="eyebrow">BFFM-XGB Learning Course</div>
        <h1>Understand the whole pipeline</h1>
        <p>Linear pages for psychometrics, statistics, XGBoost, training, evaluation, export, and safe modification.</p>
      </div>
      <ol class="chapter-list">
        ${chapters
          .map((chapter) => {
            const isCurrent = chapter.slug === current.slug;
            const isVisited = visited.has(chapter.slug);
            return `
              <li>
                <a
                  class="chapter-link ${isCurrent ? "is-current" : ""} ${isVisited ? "is-visited" : ""}"
                  href="${getLearnPath(chapter.slug)}"
                  data-chapter-slug="${chapter.slug}"
                  ${isCurrent ? 'aria-current="page"' : ""}
                >
                  <span class="chapter-order">${String(chapter.order).padStart(2, "0")}</span>
                  <span class="chapter-copy">
                    <strong>${chapter.title}</strong>
                    <span>${chapter.kicker}</span>
                  </span>
                </a>
              </li>
            `;
          })
          .join("")}
      </ol>
    </nav>
  `;
}

function renderPrevNext(current: Chapter): string {
  const currentIndex = chapters.findIndex((chapter) => chapter.slug === current.slug);
  const prev = currentIndex > 0 ? chapters[currentIndex - 1] : null;
  const next = currentIndex < chapters.length - 1 ? chapters[currentIndex + 1] : null;

  const items: string[] = [];

  if (prev) {
    items.push(`<a class="nav-card" href="${getLearnPath(prev.slug)}" data-chapter-slug="${prev.slug}">
              <span class="nav-label">Previous</span>
              <strong>${prev.title}</strong>
            </a>`);
  }

  if (next) {
    items.push(`<a class="nav-card" href="${getLearnPath(next.slug)}" data-chapter-slug="${next.slug}">
              <span class="nav-label">Next</span>
              <strong>${next.title}</strong>
            </a>`);
  }

  if (items.length === 0) {
    return "";
  }

  return `
    <div class="prev-next">
      ${items.join("")}
    </div>
  `;
}

type LearnAppOptions = {
  initialSlug: string;
  onNavigate: (slug: string) => void;
};

export type LearnAppHandle = {
  update: (slug: string) => void;
  destroy: () => void;
};

export function mountLearnApp(
  host: HTMLElement,
  { initialSlug, onNavigate }: LearnAppOptions,
): LearnAppHandle {
  const app = host;
  let currentChapterSlug = initialSlug;

  app.classList.add("learn-page");
  const mathMlMode = detectMathMlSupport() ? "mathml-supported" : "mathml-fallback";
  let mathMlModeWasAdded = false;
  if (!app.classList.contains(mathMlMode)) {
    app.classList.add(mathMlMode);
    mathMlModeWasAdded = true;
  }

  let glossaryTooltip: HTMLDivElement | null = null;
  let activeGlossaryTrigger: HTMLElement | null = null;
  let sidebarScrollTop = 0;

  function ensureGlossaryTooltip(): HTMLDivElement {
    if (glossaryTooltip) return glossaryTooltip;

    const tooltip = document.createElement("div");
    tooltip.className = "ui-tooltip";
    tooltip.setAttribute("role", "tooltip");
    tooltip.setAttribute("aria-hidden", "true");
    tooltip.dataset.state = "closed";
    tooltip.dataset.side = "top";
    tooltip.innerHTML = `<div class="ui-tooltip__content"></div>`;
    document.body.appendChild(tooltip);
    glossaryTooltip = tooltip;

    return tooltip;
  }

  function positionGlossaryTooltip(trigger: HTMLElement): void {
    const tooltip = ensureGlossaryTooltip();
    const spacing = 12;
    const viewportInset = 12;
    const rect = trigger.getBoundingClientRect();
    const tooltipRect = tooltip.getBoundingClientRect();

    let side: "top" | "bottom" = "top";
    let top = rect.top - tooltipRect.height - spacing;
    if (top < viewportInset) {
      side = "bottom";
      top = rect.bottom + spacing;
    }

    let left = rect.left + rect.width / 2 - tooltipRect.width / 2;
    left = Math.max(viewportInset, Math.min(left, window.innerWidth - tooltipRect.width - viewportInset));

    tooltip.dataset.side = side;
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
  }

  function showGlossaryTooltip(trigger: HTMLElement): void {
    const description = trigger.dataset.tooltip;
    if (!description) return;

    const tooltip = ensureGlossaryTooltip();
    const content = tooltip.querySelector<HTMLDivElement>(".ui-tooltip__content");
    if (!content) return;

    if (activeGlossaryTrigger && activeGlossaryTrigger !== trigger) {
      activeGlossaryTrigger.removeAttribute("data-tooltip-open");
    }

    activeGlossaryTrigger = trigger;
    trigger.setAttribute("data-tooltip-open", "true");
    content.textContent = description;
    tooltip.dataset.state = "open";
    tooltip.setAttribute("aria-hidden", "false");
    positionGlossaryTooltip(trigger);
  }

  function hideGlossaryTooltip(): void {
    if (activeGlossaryTrigger) {
      activeGlossaryTrigger.removeAttribute("data-tooltip-open");
    }
    activeGlossaryTrigger = null;
    if (!glossaryTooltip) return;
    glossaryTooltip.dataset.state = "closed";
    glossaryTooltip.setAttribute("aria-hidden", "true");
  }

  function wireGlossaryTooltips(scope: ParentNode): void {
    const triggers = scope.querySelectorAll<HTMLElement>(".glossary-term");
    for (const trigger of triggers) {
      if (trigger.dataset.tooltipBound === "true") continue;
      trigger.dataset.tooltipBound = "true";

      trigger.addEventListener("mouseenter", () => {
        showGlossaryTooltip(trigger);
      });

      trigger.addEventListener("mouseleave", () => {
        if (document.activeElement !== trigger) {
          hideGlossaryTooltip();
        }
      });

      trigger.addEventListener("focus", () => {
        showGlossaryTooltip(trigger);
      });

      trigger.addEventListener("blur", () => {
        hideGlossaryTooltip();
      });
    }
  }

  function renderApp(): void {
    hideGlossaryTooltip();

    const previousSidebar = app.querySelector<HTMLElement>(".sidebar");
    if (previousSidebar) {
      sidebarScrollTop = previousSidebar.scrollTop;
    }

    const visited = loadVisited();
    const chapter = chapterBySlug(currentChapterSlug);
    visited.add(chapter.slug);
    saveVisited(visited);

    const index = chapters.findIndex((item) => item.slug === chapter.slug);
    const progress = ((index + 1) / chapters.length) * 100;

    app.innerHTML = `
      <div class="shell">
        ${renderSidebar(chapter, visited)}
        <main class="main" id="main-content">
          <header class="topbar">
            <div>
              <div class="eyebrow">Chapter ${String(chapter.order).padStart(2, "0")}</div>
              <h2>${chapter.title}</h2>
              <p>${chapter.summary}</p>
            </div>
            <div class="chapter-meta">
              <div>
                <span class="meta-label">Read time</span>
                <strong>${estimatedMinutes(chapter)} min</strong>
              </div>
              <div>
                <span class="meta-label">Progress</span>
                <strong>${index + 1} / ${chapters.length}</strong>
              </div>
            </div>
          </header>
          <div class="progress-rail" aria-hidden="true">
            <div class="progress-rail-fill" style="width: ${progress.toFixed(2)}%"></div>
          </div>
          <article class="lesson">
            ${chapter.content}
          </article>
          ${renderPrevNext(chapter)}
        </main>
      </div>
    `;

    const sidebar = app.querySelector<HTMLElement>(".sidebar");
    if (sidebar) {
      sidebar.scrollTop = sidebarScrollTop;
      sidebar
        .querySelector<HTMLElement>('[aria-current="page"]')
        ?.scrollIntoView({ block: "nearest" });
    }

    chapter.afterRender?.();
    wireGlossaryTooltips(app);
  }

  const handleAppClick = (event: MouseEvent): void => {
    const target = event.target;
    if (!(target instanceof Element)) return;

    const link = target.closest<HTMLAnchorElement>("a[data-chapter-slug]");
    if (!link) return;

    if (
      event.defaultPrevented ||
      event.button !== 0 ||
      link.target ||
      link.hasAttribute("download") ||
      event.metaKey ||
      event.altKey ||
      event.ctrlKey ||
      event.shiftKey
    ) {
      return;
    }

    const slug = link.dataset.chapterSlug;
    if (!slug || slug === currentChapterSlug) return;

    event.preventDefault();
    onNavigate(slug);
  };

  const handleKeydown = (event: KeyboardEvent): void => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
      return;
    }
    if (event.key === "Escape") {
      hideGlossaryTooltip();
    }
    const currentIndex = chapters.findIndex((chapter) => chapter.slug === chapterBySlug(currentChapterSlug).slug);
    if (event.key === "ArrowRight" && currentIndex < chapters.length - 1) {
      onNavigate(chapters[currentIndex + 1].slug);
    }
    if (event.key === "ArrowLeft" && currentIndex > 0) {
      onNavigate(chapters[currentIndex - 1].slug);
    }
  };

  const handleResize = (): void => {
    if (activeGlossaryTrigger) {
      positionGlossaryTooltip(activeGlossaryTrigger);
    }
  };

  const handleScroll = (): void => {
    if (activeGlossaryTrigger) {
      positionGlossaryTooltip(activeGlossaryTrigger);
    }
  };

  app.addEventListener("click", handleAppClick);
  window.addEventListener("keydown", handleKeydown);
  window.addEventListener("resize", handleResize);
  window.addEventListener("scroll", handleScroll, true);
  renderApp();

  return {
    update: (slug: string) => {
      if (slug === currentChapterSlug) return;
      currentChapterSlug = slug;
      renderApp();
    },
    destroy: () => {
      app.removeEventListener("click", handleAppClick);
      window.removeEventListener("keydown", handleKeydown);
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("scroll", handleScroll, true);

      hideGlossaryTooltip();
      if (mathMlModeWasAdded) {
        app.classList.remove(mathMlMode);
      }
      if (glossaryTooltip) {
        glossaryTooltip.remove();
      }

      app.classList.remove("learn-page");
      app.innerHTML = "";
    },
  };
}
