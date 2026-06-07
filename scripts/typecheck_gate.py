#!/usr/bin/env python3
"""Gate `make typecheck` on basedpyright errors AND warnings.

basedpyright's own exit code is non-zero only when it reports *errors*; new
*warning*-severity diagnostics leave it at exit 0, so a plain `basedpyright`
invocation would let new warnings through CI. This reads basedpyright's
``--outputjson`` report from stdin (which already has the committed baseline in
``.basedpyright/baseline.json`` applied, so only NEW diagnostics are counted),
prints any error/warning diagnostics for visibility, and exits non-zero if there
is at least one error OR warning. Notes (information) do not fail the gate.

Usage (see the Makefile `typecheck` target):
    uv run basedpyright --outputjson | uv run python scripts/typecheck_gate.py
"""

from __future__ import annotations

import json
import sys
from typing import Any


def main() -> int:
    raw = sys.stdin.read()
    try:
        report: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        sys.stderr.write(
            "typecheck gate: basedpyright did not emit valid JSON on stdout. "
            "Raw output follows:\n"
        )
        sys.stderr.write(raw[:4000])
        sys.stderr.write("\n")
        return 1

    summary = report.get("summary", {})
    errors = int(summary.get("errorCount", 0))
    warnings = int(summary.get("warningCount", 0))
    notes = int(summary.get("informationCount", 0))

    if errors or warnings:
        diagnostics = report.get("generalDiagnostics", [])
        for diag in diagnostics if isinstance(diagnostics, list) else []:
            if not isinstance(diag, dict):
                continue
            severity = diag.get("severity")
            if severity not in ("error", "warning"):
                continue
            start = diag.get("range", {}).get("start", {})
            line = int(start.get("line", 0)) + 1
            col = int(start.get("character", 0)) + 1
            # partition (not splitlines()[0]) is empty/None/multiline-safe:
            # "".splitlines() is [] and would IndexError.
            message = (diag.get("message") or "").partition("\n")[0]
            rule = diag.get("rule") or ""
            suffix = f" ({rule})" if rule else ""
            print(f"{diag.get('file')}:{line}:{col} {severity}: {message}{suffix}")

    print(
        f"basedpyright (post-baseline): {errors} error(s), "
        f"{warnings} warning(s), {notes} note(s)"
    )
    return 1 if (errors or warnings) else 0


if __name__ == "__main__":
    sys.exit(main())
