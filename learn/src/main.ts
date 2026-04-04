import "./styles.css";

import type { Chapter } from "./types";
import { chapters } from "./content";

const app = document.querySelector<HTMLDivElement>("#app");

if (!app) {
  throw new Error("Missing #app mount node");
}

const visitedStorageKey = "bffm-xgb-learn-visited";
let glossaryTooltip: HTMLDivElement | null = null;
let activeGlossaryTrigger: HTMLElement | null = null;
let sidebarScrollTop = 0;

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

document.documentElement.classList.add(
  detectMathMlSupport() ? "mathml-supported" : "mathml-fallback",
);

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

function currentSlug(): string {
  const hash = window.location.hash.replace(/^#/, "").trim();
  if (!hash) return chapters[0].slug;
  return hash;
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
                  href="#${chapter.slug}"
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

  return `
    <div class="prev-next">
      ${
        prev
          ? `<a class="nav-card" href="#${prev.slug}">
              <span class="nav-label">Previous</span>
              <strong>${prev.title}</strong>
            </a>`
          : `<div class="nav-card nav-card-empty"><span class="nav-label">Previous</span><strong>Start here</strong></div>`
      }
      ${
        next
          ? `<a class="nav-card" href="#${next.slug}">
              <span class="nav-label">Next</span>
              <strong>${next.title}</strong>
            </a>`
          : `<div class="nav-card nav-card-empty"><span class="nav-label">Next</span><strong>Course complete</strong></div>`
      }
    </div>
  `;
}

function renderApp(): void {
  hideGlossaryTooltip();

  const previousSidebar = app.querySelector<HTMLElement>(".sidebar");
  if (previousSidebar) {
    sidebarScrollTop = previousSidebar.scrollTop;
  }

  const visited = loadVisited();
  const chapter = chapterBySlug(currentSlug());
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
    sidebar.querySelector<HTMLElement>('[aria-current="page"]')?.scrollIntoView({ block: "nearest" });
  }

  chapter.afterRender?.();
  wireGlossaryTooltips(app);
}

window.addEventListener("hashchange", renderApp);
window.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
    return;
  }
  if (event.key === "Escape") {
    hideGlossaryTooltip();
  }
  const currentIndex = chapters.findIndex((chapter) => chapter.slug === chapterBySlug(currentSlug()).slug);
  if (event.key === "ArrowRight" && currentIndex < chapters.length - 1) {
    window.location.hash = chapters[currentIndex + 1].slug;
  }
  if (event.key === "ArrowLeft" && currentIndex > 0) {
    window.location.hash = chapters[currentIndex - 1].slug;
  }
});

window.addEventListener("resize", () => {
  if (activeGlossaryTrigger) {
    positionGlossaryTooltip(activeGlossaryTrigger);
  }
});

window.addEventListener("scroll", () => {
  if (activeGlossaryTrigger) {
    positionGlossaryTooltip(activeGlossaryTrigger);
  }
}, true);

renderApp();
