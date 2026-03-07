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
