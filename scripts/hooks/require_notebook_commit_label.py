#!/usr/bin/env python3
"""Fail commit-msg when tracked notebook edits are staged without a label."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REQUIRED_LABEL = "[notebook-ok]"


def _staged_tracked_notebooks() -> list[str]:
    cmd = ["git", "diff", "--cached", "--name-status", "--diff-filter=MR", "--", "*.ipynb"]
    out = subprocess.check_output(cmd, text=True)
    notebooks: list[str] = []
    for raw_line in out.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("\t")
        status = parts[0]
        if status.startswith("R"):
            notebooks.append(parts[-1])
        elif status == "M":
            notebooks.append(parts[1])
    return notebooks


def main() -> int:
    if len(sys.argv) < 2:
        print("require-notebook-label: missing commit message path.", file=sys.stderr)
        return 1

    commit_msg_path = Path(sys.argv[1])
    commit_msg = commit_msg_path.read_text(encoding="utf-8")
    changed_notebooks = _staged_tracked_notebooks()

    if not changed_notebooks:
        return 0

    if REQUIRED_LABEL in commit_msg:
        return 0

    changed_list = "\n".join(f"  - {path}" for path in changed_notebooks)
    print(
        "Notebook hygiene check failed.\n"
        "Tracked notebook edits are staged, but the commit message does not include "
        f"the required label {REQUIRED_LABEL}.\n"
        "Staged notebook files:\n"
        f"{changed_list}\n"
        "If this commit intentionally changes tracked notebooks, add "
        f"{REQUIRED_LABEL} to the commit message.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
