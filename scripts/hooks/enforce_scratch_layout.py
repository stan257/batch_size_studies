#!/usr/bin/env python3
"""Keep exploratory artifacts out of root and canonical notebook paths."""

from __future__ import annotations

import subprocess
import sys

ALLOWED_ROOT_MARKDOWN = {"README.md"}
ALLOWED_ROOT_PYTHON = {"experiments.py", "sitecustomize.py"}


def _staged_name_status() -> list[tuple[str, str]]:
    cmd = ["git", "diff", "--cached", "--name-status", "--diff-filter=ACMR"]
    out = subprocess.check_output(cmd, text=True)
    rows: list[tuple[str, str]] = []
    for raw_line in out.splitlines():
        if not raw_line.strip():
            continue
        parts = raw_line.split("\t")
        status = parts[0]
        path = parts[-1]
        rows.append((status, path))
    return rows


def main() -> int:
    violations: list[str] = []

    for status, path in _staged_name_status():
        is_root = "/" not in path

        if is_root and path.endswith(".md") and path not in ALLOWED_ROOT_MARKDOWN:
            violations.append(f"{path} -> move exploratory markdown to notes/")

        if is_root and path.endswith(".py") and path not in ALLOWED_ROOT_PYTHON:
            violations.append(f"{path} -> keep ad-hoc scripts under scripts/ or notes/")

        if status.startswith("A") and path.endswith(".ipynb"):
            if path.startswith("notebooks/") and not path.startswith("notebooks/scratch/"):
                violations.append(f"{path} -> new notebooks should live in notebooks/scratch/")
            elif not path.startswith("notebooks/"):
                violations.append(f"{path} -> notebook files should live in notebooks/scratch/")

    if not violations:
        return 0

    print("Scratch layout check failed. Move exploratory files out of root paths:", file=sys.stderr)
    for item in violations:
        print(f"  - {item}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
