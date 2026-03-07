#!/usr/bin/env python3
"""Run a command, prefix each output line with a label, and tee to a log file."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _emit(prefix: str, message: str, log_handle: object | None) -> None:
    text = f"{prefix} {message}"
    print(text, flush=True)
    if log_handle is not None:
        log_handle.write(text + "\n")
        log_handle.flush()


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prefix command output with a stable label and write it to a log file."
    )
    parser.add_argument("--label", required=True, help="Short label for console/log prefixes.")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file path. Parent directories are created automatically.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run. Separate wrapper args from command args with --",
    )
    args = parser.parse_args()

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing command after --")

    prefix = f"[{args.label}]"
    log_handle = None
    try:
        if args.log_file is not None:
            args.log_file.parent.mkdir(parents=True, exist_ok=True)
            log_handle = args.log_file.open("w", encoding="utf-8", buffering=1)

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env["BFFM_LOG_LABEL"] = args.label

        _emit(
            prefix,
            f"[{_utc_timestamp()}] >>> starting: {' '.join(shlex.quote(part) for part in command)}",
            log_handle,
        )

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip("\n")
            _emit(prefix, line, log_handle)

        return_code = process.wait()
        _emit(prefix, f"[{_utc_timestamp()}] >>> exit={return_code}", log_handle)
        return return_code
    finally:
        if log_handle is not None:
            log_handle.close()


if __name__ == "__main__":
    sys.exit(main())
