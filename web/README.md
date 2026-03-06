# BFFM-XGB Web Assessment

20-item Big Five personality assessment scored by the BFFM-XGB ONNX model. React frontend, Hono API server, deployed as a HuggingFace Docker Space.

## Local Development

Requires Node 22+ and the model files in `output/reference/` (run `make export-reference` from the repo root if missing).

```bash
# Install dependencies
npm install

# Start dev server (Vite on :5173, Hono API on :7860)
npm run dev
```

Vite proxies `/api/*` requests to the Hono server. Hot-reload works for both client and server code.

Or from the repo root:

```bash
make web-setup   # npm ci
make web-dev     # npm run dev
```

## Model Resolution

The server resolves the ONNX model in order:

1. **`MODEL_DIR`** env var — local directory containing `config.json` + `model.onnx`
2. **`HF_REPO_ID`** env var — downloads from the HF model repo at startup (cached in `/tmp`). Uses `HF_VARIANT` to pick the subdirectory (default: `reference`).
3. **Fallback** — `output/reference` relative to the package root

For local dev the fallback resolves to `output/reference/` automatically. In production, `HF_REPO_ID` is set so the Docker image stays small and downloads the model at startup.

## Production Build

```bash
npm run build    # Vite client → dist/client, tsc server → dist/server
npm run start    # node dist/server/index.js on :7860
```

## Docker

```bash
docker build -t bffm-web .
docker run -p 7860:7860 -e HF_REPO_ID=your-org/bffm-xgb bffm-web
```

## Deploy to HuggingFace Space

Set `HF_TOKEN`, `HF_REPO_ID`, and `HF_SPACE_ID` in the repo root `.env` (see `.env.example`), then:

```bash
make deploy-web
```

This builds the app then uploads `Dockerfile`, `package.json`, `package-lock.json`, and `dist/` to the Space (stale files from previous deploys are removed automatically). Dependencies are installed inside the Docker build via `npm ci --omit=dev`. `HF_REPO_ID` is set as a Space secret so the container can download the model at startup — it is not bundled in the image. HF builds the Docker image and exposes it at `https://{user}-{space-name}.hf.space`.

## Architecture

```
Browser → HF proxy → Hono (:7860)
  GET /*            → static files (Vite-built React app)
  POST /api/predict → reverse-score → ONNX inference → percentiles
```

The server reverse-scores the 11 reverse-keyed items (`6 - value`) before inference, matching the preprocessing applied during model training.
