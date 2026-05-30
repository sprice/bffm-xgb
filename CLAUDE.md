# CLAUDE.md

Guidance for agents and contributors working in this repository.

## Always use `uv` — never a bare `python` / `pip`

This project's Python is managed entirely by [uv](https://docs.astral.sh/uv/).
**Do not invoke `python`, `python3`, or `pip` directly.** Always go through uv:

| Instead of               | Use                              |
| ------------------------ | -------------------------------- |
| `python script.py`       | `uv run python script.py`        |
| `python -m pytest`       | `uv run python -m pytest`        |
| `ruff` / `basedpyright`  | `uv run ruff` / `uv run basedpyright` |
| `pip install <pkg>`      | `uv add <pkg>` (dev: `uv add --dev <pkg>`) |
| `python -m venv .venv`   | `uv sync`                        |

- The interpreter is pinned by `.python-version` (3.14) and is downloaded/managed by uv. (`requires-python` is `>=3.11`, the supported floor; basedpyright type-checks against that floor.)
- Dependencies live in `pyproject.toml` (`[project.dependencies]` + `[dependency-groups] dev`) and are locked in `uv.lock`. Install/refresh with `uv sync`. **There is no `requirements.txt`.**
- The Makefile already routes all Python through uv (`PY := uv run python`); prefer the `make` targets below over ad-hoc commands.

## Common commands

```bash
make setup       # uv sync + TypeScript + web deps
make all         # full pipeline (download → figures)
make test        # lib + inference + web tests
make lint        # ruff check (kept at zero findings)
make format      # ruff format (apply)
make typecheck   # basedpyright (gated against .basedpyright/baseline.json)
```

One-offs go through uv too, e.g. `uv run python pipeline/03_compute_norms.py`.

## Code quality gates (enforced in CI)

- **Ruff** (`[tool.ruff]`, ruleset `E4/E7/E9/F/I/UP`): kept at **zero** findings — no baseline. `E402` is ignored in `pipeline/` and `scripts/` (intentional `sys.path.insert(...)` shim before the `lib.*` imports).
- **basedpyright** (`[tool.basedpyright]`, `standard` mode): gated against `.basedpyright/baseline.json` — fails on **new** diagnostics only. Fix genuine issues; never grow the baseline to hide one. Regenerate with `uv run basedpyright --writebaseline` only when shrinking it.
- **pytest**: `make test`.

## Repo conventions

- **Edit source, not generated artifacts.** Model cards, `output/**/README.md`, `notes/NOTES.md`, `research_summary.json`, and figures are produced by the pipeline — change the generators/templates (`pipeline/`, `templates/`, `scripts/`), then regenerate. (`output/README.md` and `notes/NOTES.md` are committed but generated.)
- Shared logic lives in `lib/`; numbered pipeline stages live in `pipeline/` (`01`–`13`).
- Remote training installs uv via instance cloud-init (`infra/{cpu,gpu}/main.tf`) and runs `uv sync` through `make remote-setup`.
- See `docs/pipeline.md` (reproduction, stages, lint/typecheck) and `docs/infrastructure.md` (remote AWS training).
