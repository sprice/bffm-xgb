import type { ResourceLink } from "../types";

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

// ─── Glossary term ────────────────────────────────────────────────
// Emits a `data-glossary-term` attribute so GlossaryTooltip can
// delegate events off of it. Styling is 100% Tailwind: dotted
// primary underline, inherits font/color/size from its parent prose,
// flips to primary on hover/focus/open.

export function abbr(label: string, description: string): string {
  const safeDescription = escapeHtml(description);
  const safeAria = escapeHtml(`${label}: ${description}`);
  return `<button type="button" data-glossary-term data-tooltip="${safeDescription}" aria-label="${safeAria}" class="[font:inherit] [color:inherit] bg-transparent border-0 p-0 m-0 cursor-help underline decoration-dotted decoration-primary/60 underline-offset-4 hover:text-primary hover:decoration-primary focus-visible:text-primary focus-visible:decoration-primary data-[tooltip-open=true]:text-primary">${label}</button>`;
}

// ─── Section ──────────────────────────────────────────────────────
// Top-level chapter division with an h2. Uses `space-y-4` so all
// direct children (paragraphs, tables, callouts, lists, cards) share
// a consistent 1rem rhythm. The between-section gap/border is handled
// on the article wrapper in LearnContent via `[&>section+section]:*`.

export function section(title: string, body: string): string {
  return `
    <section class="space-y-4">
      <h2 class="font-display text-2xl font-semibold leading-tight text-text">${title}</h2>
      ${body}
    </section>
  `;
}

// ─── Prose ────────────────────────────────────────────────────────

export function lead(text: string): string {
  return `<p class="text-lg leading-relaxed text-text">${text}</p>`;
}

export function paragraph(text: string): string {
  return `<p>${text}</p>`;
}

export function list(items: string[]): string {
  return `<ul class="list-disc pl-6 space-y-2 marker:text-text-muted">${items
    .map((item) => `<li>${item}</li>`)
    .join("")}</ul>`;
}

export function ordered(items: string[]): string {
  return `<ol class="list-decimal pl-6 space-y-2 marker:text-text-muted">${items
    .map((item) => `<li>${item}</li>`)
    .join("")}</ol>`;
}

// ─── Table ────────────────────────────────────────────────────────

export function table(headers: string[], rows: string[][]): string {
  const head = headers
    .map(
      (header) =>
        `<th class="text-left px-4 py-3 border-b border-border font-body text-xs font-semibold uppercase tracking-widest text-text-muted">${header}</th>`,
    )
    .join("");

  const body = rows
    .map(
      (row) =>
        `<tr class="[&:last-child>td]:border-b-0">${row
          .map(
            (cell) =>
              `<td class="text-left px-4 py-3 border-b border-border align-top">${cell}</td>`,
          )
          .join("")}</tr>`,
    )
    .join("");

  return `
    <div class="my-4 overflow-x-auto rounded-xl border border-border">
      <table class="w-full border-collapse">
        <thead class="bg-primary/5"><tr>${head}</tr></thead>
        <tbody>${body}</tbody>
      </table>
    </div>
  `;
}

// ─── Code block ───────────────────────────────────────────────────
// Always dark, regardless of theme. Inline `<code>` inside prose is
// styled by the `[&_:not(pre)>code]:*` variant on the article wrapper.

export function codeBlock(code: string, language = ""): string {
  return `<pre class="my-4 p-4 rounded-xl bg-[#1a1816] text-[#f2f0ed] overflow-x-auto border border-white/10"><code class="language-${language} font-mono text-sm">${escapeHtml(
    code,
  )}</code></pre>`;
}

// ─── Callouts ─────────────────────────────────────────────────────

type CalloutTone = "note" | "why" | "warning" | "example" | "repo";

const calloutTones: Record<CalloutTone, string> = {
  note: "bg-primary/5 border-primary/20",
  why: "bg-primary/10 border-primary/40",
  warning: "bg-est/10 border-est/30",
  example: "bg-csn/10 border-csn/30",
  repo: "bg-primary/10 border-primary/30",
};

export function callout(tone: CalloutTone, title: string, body: string): string {
  return `
    <aside class="my-4 p-5 rounded-xl border ${calloutTones[tone]}">
      <div class="mb-2 font-body text-xs font-semibold uppercase tracking-widest text-text-muted">${title}</div>
      <div class="space-y-3 [&>*:last-child]:mb-0">${body}</div>
    </aside>
  `;
}

// ─── Math block ───────────────────────────────────────────────────
// `math-rendered` and `math-fallback` class names are preserved so
// the browser-capability gate rules in app.css still work.

export function mathBlock(mathml: string, plainText?: string): string {
  const fallback = plainText
    ? `<div class="math-fallback justify-self-start p-3 rounded-lg bg-surface border border-border font-mono text-sm text-text leading-snug m-0">${escapeHtml(plainText)}</div>`
    : "";
  const label = plainText ? ` aria-label="${escapeHtml(plainText)}"` : "";
  return `
    <div class="my-4 grid gap-4 p-6 rounded-2xl bg-primary/10 border border-primary/20"${label}>
      <math class="math-rendered block m-0 overflow-x-auto text-center text-2xl sm:text-3xl text-text" display="block">${mathml}</math>
      ${fallback}
    </div>
  `;
}

// ─── Cards and grids ──────────────────────────────────────────────

export function exampleCard(title: string, body: string): string {
  return `
    <div class="p-5 rounded-xl border border-border bg-surface">
      <div class="font-display text-lg font-semibold text-text mb-2">${title}</div>
      <div class="space-y-2 [&>*:last-child]:mb-0">${body}</div>
    </div>
  `;
}

export function comparisonGrid(cards: string[]): string {
  return `<div class="my-4 grid gap-4 sm:grid-cols-2">${cards.join("")}</div>`;
}

export function figureStrip(
  stats: Array<{ value: string; label: string }>,
): string {
  return `
    <div class="my-5 grid gap-3 grid-cols-2 sm:grid-cols-5">
      ${stats
        .map(
          (s) => `
        <div class="p-4 rounded-xl border border-border bg-surface">
          <strong class="block font-display text-2xl font-semibold text-text mb-1">${s.value}</strong>
          <span class="block font-body text-xs text-text-muted leading-snug">${s.label}</span>
        </div>
      `,
        )
        .join("")}
    </div>
  `;
}

export function barList(
  rows: Array<{ label: string; percent: number; value: string }>,
): string {
  return `
    <div class="my-4 grid gap-3">
      ${rows
        .map(
          (r) => `
        <div class="grid gap-2 items-center grid-cols-1 sm:grid-cols-[9rem_minmax(0,1fr)_4rem] font-body text-sm">
          <span class="text-text">${r.label}</span>
          <div class="h-2 rounded-full bg-border overflow-hidden">
            <div class="h-full rounded-full bg-gradient-to-r from-primary to-primary-hover" style="width:${r.percent}%"></div>
          </div>
          <strong class="text-text font-semibold">${r.value}</strong>
        </div>
      `,
        )
        .join("")}
    </div>
  `;
}

// ─── Resource list ────────────────────────────────────────────────

export function resourceList(resources: ResourceLink[]): string {
  return `
    <div class="grid gap-3">
      ${resources
        .map(
          (resource) => `
            <a class="grid gap-1 p-4 rounded-xl bg-surface border border-border text-text no-underline transition-colors hover:border-primary hover:bg-primary/5" href="${resource.href}" target="_blank" rel="noreferrer">
              <span class="font-semibold text-text">${resource.label}</span>
              <span class="text-text-muted text-sm">${resource.note ?? "Official docs / source"}</span>
            </a>
          `,
        )
        .join("")}
    </div>
  `;
}

// ─── Internal files (repo-flavored callout) ──────────────────────

export function internalFiles(files: string[]): string {
  return callout(
    "repo",
    "Codebase Map",
    list(files.map((file) => `<code>${file}</code>`)),
  );
}

// ─── Split-formula helpers (inline) ──────────────────────────────

export function splitFormula(extQ: number, estQ: number): string {
  return `${extQ} × 5 + ${estQ} = <strong>${extQ * 5 + estQ}</strong>`;
}

export function splitFormulaThree(
  extQ: number,
  estQ: number,
  opnQ: number,
): string {
  return `${extQ} × 25 + ${estQ} × 5 + ${opnQ} = <strong>${extQ * 25 + estQ * 5 + opnQ}</strong>`;
}
