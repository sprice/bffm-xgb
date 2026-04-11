function standardNormalCdf(z: number): number {
  const absZ = Math.abs(z);
  const t = 1 / (1 + 0.2316419 * absZ);
  const d = Math.exp((-0.5 * absZ * absZ)) / Math.sqrt(2 * Math.PI);
  const poly =
    t *
    (0.31938153 +
      t *
        (-0.356563782 +
          t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  const cnd = 1 - d * poly;
  return z >= 0 ? cnd : 1 - cnd;
}

function setHtml(id: string, html: string): HTMLElement | null {
  const el = document.getElementById(id);
  if (!el) return null;
  el.innerHTML = html;
  return el;
}

export function mountReverseScoreWidget(id = "reverse-score-widget"): void {
  const container = setHtml(
    id,
    `
      <div class="widget">
        <div class="widget-grid">
          <label>
            Raw response (1-5)
            <input id="${id}-input" type="range" min="1" max="5" step="1" value="2" />
          </label>
          <label>
            Current value
            <input id="${id}-value" type="number" min="1" max="5" step="1" value="2" />
          </label>
        </div>
        <div id="${id}-output" class="widget-output"></div>
      </div>
    `,
  );
  if (!container) return;

  const slider = container.querySelector<HTMLInputElement>(`#${id}-input`);
  const number = container.querySelector<HTMLInputElement>(`#${id}-value`);
  const output = container.querySelector<HTMLDivElement>(`#${id}-output`);
  if (!slider || !number || !output) return;

  const update = (raw: number) => {
    const clipped = Math.max(1, Math.min(5, Math.round(raw)));
    slider.value = String(clipped);
    number.value = String(clipped);
    output.innerHTML = `
      Reverse-scored value: <strong>${6 - clipped}</strong><br />
      Formula: <code>6 - ${clipped} = ${6 - clipped}</code>
    `;
  };

  slider.addEventListener("input", () => update(Number(slider.value)));
  number.addEventListener("input", () => update(Number(number.value)));
  update(2);
}

export function mountZScoreWidget(id = "z-score-widget"): void {
  const container = setHtml(
    id,
    `
      <div class="widget">
        <div class="widget-grid">
          <label>
            Raw score
            <input id="${id}-score" type="number" step="0.01" value="3.8" />
          </label>
          <label>
            Mean
            <input id="${id}-mean" type="number" step="0.01" value="2.9138" />
          </label>
          <label>
            SD
            <input id="${id}-sd" type="number" step="0.01" min="0.01" value="0.9112" />
          </label>
        </div>
        <div id="${id}-output" class="widget-output"></div>
      </div>
    `,
  );
  if (!container) return;

  const scoreInput = container.querySelector<HTMLInputElement>(`#${id}-score`);
  const meanInput = container.querySelector<HTMLInputElement>(`#${id}-mean`);
  const sdInput = container.querySelector<HTMLInputElement>(`#${id}-sd`);
  const output = container.querySelector<HTMLDivElement>(`#${id}-output`);
  if (!scoreInput || !meanInput || !sdInput || !output) return;

  const update = () => {
    const score = Number(scoreInput.value);
    const mean = Number(meanInput.value);
    const sd = Math.max(0.0001, Number(sdInput.value));
    const z = (score - mean) / sd;
    const percentile = standardNormalCdf(z) * 100;
    output.innerHTML = `
      <strong>z</strong> = <code>(${score.toFixed(2)} - ${mean.toFixed(2)}) / ${sd.toFixed(2)} = ${z.toFixed(3)}</code><br />
      <strong>Percentile</strong> ≈ <code>${percentile.toFixed(1)}</code>
    `;
  };

  [scoreInput, meanInput, sdInput].forEach((input) =>
    input.addEventListener("input", update),
  );
  update();
}

export function mountQuantileLossWidget(id = "quantile-loss-widget"): void {
  const container = setHtml(
    id,
    `
      <div class="widget">
        <div class="widget-grid">
          <label>
            True value y
            <input id="${id}-truth" type="number" step="0.1" value="3.8" />
          </label>
          <label>
            Predicted quantile ŷ
            <input id="${id}-pred" type="number" step="0.1" value="3.2" />
          </label>
          <label>
            Quantile τ
            <input id="${id}-tau" type="number" min="0.01" max="0.99" step="0.05" value="0.95" />
          </label>
        </div>
        <div id="${id}-output" class="widget-output"></div>
      </div>
    `,
  );
  if (!container) return;

  const truth = container.querySelector<HTMLInputElement>(`#${id}-truth`);
  const pred = container.querySelector<HTMLInputElement>(`#${id}-pred`);
  const tau = container.querySelector<HTMLInputElement>(`#${id}-tau`);
  const output = container.querySelector<HTMLDivElement>(`#${id}-output`);
  if (!truth || !pred || !tau || !output) return;

  const update = () => {
    const y = Number(truth.value);
    const yHat = Number(pred.value);
    const t = Math.min(0.99, Math.max(0.01, Number(tau.value)));
    const error = y - yHat;
    const loss = error >= 0 ? t * error : (1 - t) * -error;
    const interpretation =
      error >= 0
        ? "Prediction is too low. High-τ quantiles punish that heavily."
        : "Prediction is too high. Low-τ quantiles punish that heavily.";

    output.innerHTML = `
      Error <code>y - ŷ = ${error.toFixed(2)}</code><br />
      Pinball loss = <strong>${loss.toFixed(3)}</strong><br />
      ${interpretation}
    `;
  };

  [truth, pred, tau].forEach((input) => input.addEventListener("input", update));
  update();
}

export function mountSemWidget(id = "sem-widget"): void {
  const container = setHtml(
    id,
    `
      <div class="widget">
        <div class="widget-grid">
          <label>
            Items in domain (k)
            <input id="${id}-k" type="number" min="1" max="10" step="1" value="4" />
          </label>
          <label>
            Mean inter-item correlation (r̄)
            <input id="${id}-rbar" type="number" min="0.01" max="0.99" step="0.01" value="0.47" />
          </label>
          <label>
            Domain SD
            <input id="${id}-sd" type="number" min="0.01" step="0.01" value="0.91" />
          </label>
        </div>
        <div id="${id}-output" class="widget-output"></div>
      </div>
    `,
  );
  if (!container) return;

  const kInput = container.querySelector<HTMLInputElement>(`#${id}-k`);
  const rbarInput = container.querySelector<HTMLInputElement>(`#${id}-rbar`);
  const sdInput = container.querySelector<HTMLInputElement>(`#${id}-sd`);
  const output = container.querySelector<HTMLDivElement>(`#${id}-output`);
  if (!kInput || !rbarInput || !sdInput || !output) return;

  const update = () => {
    const k = Math.max(1, Math.round(Number(kInput.value)));
    const rBar = Math.min(0.99, Math.max(0.01, Number(rbarInput.value)));
    const sd = Math.max(0.01, Number(sdInput.value));
    const alpha = (k * rBar) / (1 + (k - 1) * rBar);
    const sem = sd * Math.sqrt(1 - alpha);
    output.innerHTML = `
      <strong>Projected reliability</strong> α ≈ <code>${alpha.toFixed(3)}</code><br />
      <strong>SEM</strong> ≈ <code>${sem.toFixed(3)}</code><br />
      ${sem <= 0.45 ? "This clears a 0.45 SEM target." : "This does not yet clear a 0.45 SEM target."}
    `;
  };

  [kInput, rbarInput, sdInput].forEach((input) =>
    input.addEventListener("input", update),
  );
  update();
}
