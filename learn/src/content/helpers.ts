import type { ResourceLink } from "../types";

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

export function abbr(label: string, description: string): string {
  const safeDescription = escapeHtml(description);
  return `
    <button
      type="button"
      class="glossary-term"
      data-tooltip="${safeDescription}"
      aria-label="${escapeHtml(`${label}: ${description}`)}"
    >${label}</button>
  `;
}

export function section(title: string, body: string): string {
  return `
    <section class="lesson-section">
      <h2>${title}</h2>
      ${body}
    </section>
  `;
}

export function lead(text: string): string {
  return `<p class="lead">${text}</p>`;
}

export function paragraph(text: string): string {
  return `<p>${text}</p>`;
}

export function list(items: string[]): string {
  return `<ul>${items.map((item) => `<li>${item}</li>`).join("")}</ul>`;
}

export function ordered(items: string[]): string {
  return `<ol>${items.map((item) => `<li>${item}</li>`).join("")}</ol>`;
}

export function table(headers: string[], rows: string[][]): string {
  return `
    <div class="table-wrap">
      <table>
        <thead>
          <tr>${headers.map((header) => `<th>${header}</th>`).join("")}</tr>
        </thead>
        <tbody>
          ${rows
            .map(
              (row) =>
                `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`,
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

export function codeBlock(code: string, language = ""): string {
  return `
    <pre class="code-block"><code class="language-${language}">${escapeHtml(code)}</code></pre>
  `;
}

export function callout(
  tone: "note" | "why" | "warning" | "example" | "repo",
  title: string,
  body: string,
): string {
  return `
    <aside class="callout callout-${tone}">
      <div class="callout-label">${title}</div>
      <div class="callout-body">${body}</div>
    </aside>
  `;
}

export function mathBlock(mathml: string, plainText?: string): string {
  const fallback = plainText
    ? `<div class="math-fallback">${plainText}</div>`
    : "";
  return `
    <div class="math-block">
      <math display="block">${mathml}</math>
      ${fallback}
    </div>
  `;
}

export function exampleCard(title: string, body: string): string {
  return `
    <div class="example-card">
      <div class="example-card-title">${title}</div>
      <div class="example-card-body">${body}</div>
    </div>
  `;
}

export function resourceList(resources: ResourceLink[]): string {
  return `
    <div class="resource-list">
      ${resources
        .map(
          (resource) => `
            <a class="resource-item" href="${resource.href}" target="_blank" rel="noreferrer">
              <span class="resource-label">${resource.label}</span>
              <span class="resource-note">${resource.note ?? "Official docs / source"}</span>
            </a>
          `,
        )
        .join("")}
    </div>
  `;
}

export function internalFiles(files: string[]): string {
  return callout(
    "repo",
    "Codebase Map",
    list(files.map((file) => `<code>${file}</code>`)),
  );
}

export function splitFormula(extQ: number, estQ: number): string {
  return `${extQ} × 5 + ${estQ} = <strong>${extQ * 5 + estQ}</strong>`;
}

export function splitFormulaThree(extQ: number, estQ: number, opnQ: number): string {
  return `${extQ} × 25 + ${estQ} × 5 + ${opnQ} = <strong>${extQ * 25 + estQ * 5 + opnQ}</strong>`;
}
