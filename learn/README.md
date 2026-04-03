# Learn App

This is a Vite app that lives in `learn/`.

The package now supports two tool-resolution modes:

- preferred: its own local `learn/node_modules`
- fallback: the already-installed frontend toolchain in `../web/node_modules`

Tiny local proxy scripts in `learn/scripts/` choose the local binaries first
and only fall back to the sibling `web/` install when needed.

## Why it is structured this way

- The learning app has its own source, entry HTML, tests, and build output.
- It can now be installed and built as a standalone package.
- Its `package.json` scripts call small local wrappers in `learn/scripts/`.
- Those wrappers prefer local Vite/Vitest binaries, but still work with the
  shared sibling install in `web/node_modules`.

That means:

- `cd learn && pnpm dev` works
- `cd learn && pnpm test` works
- `cd learn && pnpm build` works

## Vercel

For Vercel, treat `learn/` as its own project:

- root directory: `learn`
- install command: `pnpm install`
- build command: `pnpm build`
- output directory: `dist`

The app uses hash-based routing, so no SPA rewrite rule is required.

For a reproducible standalone deploy, commit a `learn/pnpm-lock.yaml` after
running `pnpm install` inside `learn/`.
