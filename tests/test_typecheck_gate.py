"""Unit tests for scripts/typecheck_gate.py (the `make typecheck` errors+warnings gate)."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from scripts.typecheck_gate import main


def _run(report: object, monkeypatch) -> int:
    """Feed `report` (serialized to JSON) to the gate via stdin and return its exit code."""
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(report)))
    return main()


def _summary(errors: int = 0, warnings: int = 0, notes: int = 0) -> dict:
    return {"errorCount": errors, "warningCount": warnings, "informationCount": notes}


def test_clean_passes(monkeypatch) -> None:
    assert _run({"summary": _summary()}, monkeypatch) == 0


def test_error_fails(monkeypatch) -> None:
    report = {"summary": _summary(errors=1), "generalDiagnostics": []}
    assert _run(report, monkeypatch) == 1


def test_warning_fails(monkeypatch) -> None:
    """The crux: a warning (which basedpyright's own exit code ignores) must fail the gate."""
    report = {"summary": _summary(warnings=1), "generalDiagnostics": []}
    assert _run(report, monkeypatch) == 1


def test_notes_only_passes(monkeypatch) -> None:
    """Notes (information) are advisory and must NOT fail the gate."""
    assert _run({"summary": _summary(notes=3)}, monkeypatch) == 0


def test_empty_message_diagnostic_does_not_crash(monkeypatch) -> None:
    """A diagnostic with an empty/absent message must not IndexError (regression: the gate
    used to call ''.splitlines()[0]). It still fails (warningCount > 0)."""
    report = {
        "summary": _summary(warnings=1),
        "generalDiagnostics": [
            {"severity": "warning", "message": "", "rule": "reportFoo",
             "range": {"start": {"line": 0, "character": 0}}, "file": "x.py"},
        ],
    }
    assert _run(report, monkeypatch) == 1


def test_multiline_message_uses_first_line(monkeypatch, capsys) -> None:
    report = {
        "summary": _summary(errors=1),
        "generalDiagnostics": [
            {"severity": "error", "message": "first line\nsecond line", "rule": "reportBar",
             "range": {"start": {"line": 4, "character": 2}}, "file": "y.py"},
        ],
    }
    assert _run(report, monkeypatch) == 1
    out = capsys.readouterr().out
    assert "y.py:5:3 error: first line (reportBar)" in out
    assert "second line" not in out


def test_invalid_json_fails(monkeypatch) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("not json (basedpyright crashed)"))
    assert main() == 1


def test_missing_summary_keys_default_to_zero(monkeypatch) -> None:
    assert _run({"summary": {}}, monkeypatch) == 0
