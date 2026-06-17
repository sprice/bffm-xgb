import { onCLS, onFCP, onINP, onLCP, onTTFB, type Metric } from "web-vitals";

const WEBSITE_ID = "4472fc9a-56fc-408a-bfdb-22431094eb10";
const ENDPOINT = "/a";

function payload(extra: Record<string, unknown> = {}) {
  return {
    hostname: location.hostname,
    language: navigator.language,
    referrer: document.referrer,
    screen: `${screen.width}x${screen.height}`,
    title: document.title,
    url: location.pathname,
    website: WEBSITE_ID,
    ...extra,
  };
}

function send(data: Record<string, unknown>) {
  try {
    navigator.sendBeacon(
      ENDPOINT,
      new Blob([JSON.stringify(data)], { type: "application/json" }),
    );
  } catch {
    // silently ignore analytics failures
  }
}

export function trackPageview(url?: string) {
  send({ type: "event", payload: payload(url ? { url } : {}) });
}

export function trackEvent(name: string, data?: Record<string, unknown>) {
  send({ type: "event", payload: payload({ name, ...(data ? { data } : {}) }) });
}

let performanceStarted = false;

// Collect Core Web Vitals and report them once when the page is first hidden
// (the point at which CLS and INP are finalized) as a single Umami "performance"
// event. Idempotent — safe to call on every mount.
export function setupPerformance() {
  if (performanceStarted) return;
  performanceStarted = true;

  const metrics: Record<string, number> = {};
  const collect = (metric: Metric) => {
    metrics[metric.name.toLowerCase()] = metric.value;
  };
  onLCP(collect);
  onINP(collect);
  onCLS(collect);
  onFCP(collect);
  onTTFB(collect);

  let flushed = false;
  const flush = () => {
    if (flushed || Object.keys(metrics).length === 0) return;
    flushed = true;
    send({ type: "performance", payload: payload({ ...metrics }) });
  };
  addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") flush();
  });
  addEventListener("pagehide", flush);
}
