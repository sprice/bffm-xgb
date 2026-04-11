import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

function isNavigationRequest(
  url: string,
  accept?: string,
  secFetchMode?: string,
  secFetchDest?: string,
): boolean {
  if (url === "/") return false;
  if (
    url === "/api" ||
    url.startsWith("/api/") ||
    url === "/a" ||
    url.startsWith("/a/") ||
    url.startsWith("/assets/") ||
    url.startsWith("/@") ||
    url.startsWith("/__vite")
  ) {
    return false;
  }

  if (/\.[^/]+$/.test(url)) {
    return false;
  }

  if (secFetchMode === "navigate" || secFetchDest === "document") {
    return true;
  }

  return Boolean(accept && accept.includes("text/html"));
}

export default defineConfig({
  plugins: [
    tailwindcss(),
    react(),
    {
      name: "spa-fallback",
      apply: "serve",
      configureServer(server) {
        return () => {
          server.middlewares.use(async (req, res, next) => {
            const url = (req.url?.split("?")[0] ?? "");
            const accept = req.headers.accept;
            const secFetchMode = (req.headers["sec-fetch-mode"] as string | undefined) ?? "";
            const secFetchDest = (req.headers["sec-fetch-dest"] as string | undefined) ?? "";

            if (req.method === "GET" && isNavigationRequest(url, accept, secFetchMode, secFetchDest)) {
              try {
                const indexPath = resolve(server.config.root, "index.html");
                const template = await readFile(indexPath, "utf-8");
                const html = await server.transformIndexHtml(url, template);

                res.statusCode = 200;
                res.setHeader("Content-Type", "text/html");
                res.end(html);
                return;
              } catch (error) {
                next(error);
                return;
              }
            }

            next();
          });
        };
      },
    },
  ] as const,
  root: ".",
  appType: "spa",
  build: {
    outDir: "dist/client",
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:7860",
        changeOrigin: true,
      },
      "^/a$": {
        target: "http://localhost:7860",
        changeOrigin: true,
      },
    },
  },
});
