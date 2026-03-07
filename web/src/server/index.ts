import { serve } from "@hono/node-server";
import { serveStatic } from "@hono/node-server/serve-static";
import { Hono } from "hono";
import { bodyLimit } from "hono/body-limit";
import { resolve } from "node:path";
import { getPredictor, isPredictorReady } from "./predictor.js";
import { reverseScore } from "./reverse-score.js";

const app = new Hono();

app.get("/api/health", (c) =>
  c.json({ status: isPredictorReady() ? "ok" : "loading" })
);

app.post("/api/predict", bodyLimit({ maxSize: 4 * 1024 }), async (c) => {
  if (!isPredictorReady()) {
    return c.json({ error: "Model is still loading, please try again shortly" }, 503);
  }
  let body: { responses: Record<string, number> };
  try {
    body = await c.req.json();
  } catch {
    return c.json({ error: "Invalid JSON in request body" }, 400);
  }

  try {
    if (!body || typeof body !== "object" || Array.isArray(body)) {
      return c.json({ error: "Request body must be a JSON object" }, 400);
    }

    if (!body.responses || typeof body.responses !== "object" || Array.isArray(body.responses)) {
      return c.json({ error: "Missing 'responses' object in request body" }, 400);
    }

    const keys = Object.keys(body.responses);
    if (keys.length === 0) {
      return c.json({ error: "responses must not be empty" }, 400);
    }

    const scored = reverseScore(body.responses);
    const predictor = await getPredictor();
    const results = await predictor.predict(scored);

    return c.json({ results });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    const isValidation =
      message.includes("Unrecognized item IDs") ||
      message.includes("Response values must be") ||
      message.includes("Expected Float32Array");
    if (!isValidation) {
      console.error("Prediction error:", err);
    }
    return c.json(
      { error: isValidation ? message : "Internal server error" },
      isValidation ? 400 : 500
    );
  }
});

// Analytics proxy – forward to Umami Cloud, passing Cloudflare geo headers
app.post("/a", bodyLimit({ maxSize: 4 * 1024 }), async (c) => {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "User-Agent": c.req.header("User-Agent") || "Mozilla/5.0",
  };

  // Forward client IP for geo lookup
  const ip =
    c.req.header("CF-Connecting-IP") ||
    c.req.header("X-Forwarded-For")?.split(",")[0]?.trim();
  if (ip) headers["X-Forwarded-For"] = ip;

  // Forward Cloudflare geo headers so Umami can skip its own geo lookup
  for (const h of ["CF-IPCountry", "CF-RegionCode", "CF-IPCity"]) {
    const v = c.req.header(h);
    if (v) headers[h] = v;
  }

  // Parse and validate the analytics payload
  let payload: unknown;
  try {
    payload = await c.req.json();
  } catch {
    return c.body(null, 400);
  }

  if (
    typeof payload !== "object" ||
    payload === null ||
    Array.isArray(payload)
  ) {
    return c.body(null, 400);
  }

  const p = payload as Record<string, unknown>;

  if (typeof p.type !== "string" || (p.type !== "event" && p.type !== "identify")) {
    return c.body(null, 400);
  }

  if (typeof p.payload !== "object" || p.payload === null || Array.isArray(p.payload)) {
    return c.body(null, 400);
  }

  const inner = p.payload as Record<string, unknown>;

  // Validate expected inner-payload fields
  if (typeof inner.hostname !== "string" || typeof inner.language !== "string" ||
      typeof inner.screen !== "string" || typeof inner.url !== "string") {
    return c.body(null, 400);
  }

  // Overwrite the website ID so callers cannot spoof it
  inner.website = "4472fc9a-56fc-408a-bfdb-22431094eb10";

  try {
    const res = await fetch("https://cloud.umami.is/api/send", {
      method: "POST",
      headers,
      body: JSON.stringify({ type: p.type, payload: inner }),
    });
    if (!res.ok) return c.body(null, 204);
    const text = await res.text();
    try {
      return c.json(JSON.parse(text), res.status as 200);
    } catch {
      return c.body(null, 204);
    }
  } catch {
    return c.body(null, 204);
  }
});

// Serve static files in production
const clientDir = resolve(import.meta.dirname, "..", "client");
app.use("/*", serveStatic({ root: clientDir }));

// SPA fallback: serve index.html only for navigation requests (not asset requests)
app.get("/*", async (c) => {
  const accept = c.req.header("Accept") || "";
  if (accept.includes("text/html")) {
    return serveStatic({ root: clientDir, path: "index.html" })(c, async () => {});
  }
  return c.notFound();
});

const port = Number(process.env.PORT) || 7860;

// Start server immediately so HF Spaces sees the port bound before its startup timeout.
// The model loads in the background; /api/predict returns 503 until ready.
serve({ fetch: app.fetch, port }, (info) => {
  console.log(`Server running on http://localhost:${info.port}`);
});

console.log("Loading ONNX model...");
getPredictor()
  .then(() => console.log("Model loaded successfully."))
  .catch((err) => {
    console.error("Failed to load model:", err);
    process.exit(1);
  });
