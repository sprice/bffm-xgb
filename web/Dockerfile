FROM node:22-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV NODE_ENV=production
ENV PORT=7860

COPY package.json package-lock.json ./
RUN npm ci --omit=dev
COPY dist ./dist

# Model is downloaded from HF at startup via HF_REPO_ID env var.
# Set HF_REPO_ID and optionally HF_VARIANT (default: "reference").

RUN chown -R node:node /app
USER node

EXPOSE 7860
CMD ["node", "dist/server/index.js"]
