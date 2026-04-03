import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const localViteEntry = resolve(here, "../node_modules/vite/bin/vite.js");
const sharedViteEntry = resolve(here, "../../web/node_modules/vite/bin/vite.js");
const viteEntry = existsSync(localViteEntry) ? localViteEntry : sharedViteEntry;

if (!existsSync(viteEntry)) {
  console.error(
    "Vite binary not found in learn/node_modules or web/node_modules. " +
      "Install learn/ dependencies for standalone builds.",
  );
  process.exit(1);
}

const args = process.argv.slice(2);
if (args[0] === "--") {
  args.shift();
}

const child = spawn(process.execPath, [viteEntry, ...args], {
  cwd: resolve(here, ".."),
  stdio: "inherit",
});

child.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 0);
});
