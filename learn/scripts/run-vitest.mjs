import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const localVitestEntry = resolve(here, "../node_modules/vitest/vitest.mjs");
const sharedVitestEntry = resolve(here, "../../web/node_modules/vitest/vitest.mjs");
const vitestEntry = existsSync(localVitestEntry) ? localVitestEntry : sharedVitestEntry;

if (!existsSync(vitestEntry)) {
  console.error(
    "Vitest binary not found in learn/node_modules or web/node_modules. " +
      "Install learn/ dependencies for standalone builds.",
  );
  process.exit(1);
}

const args = process.argv.slice(2);
if (args[0] === "--") {
  args.shift();
}

const child = spawn(process.execPath, [vitestEntry, ...args], {
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
