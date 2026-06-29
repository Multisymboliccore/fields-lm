#!/usr/bin/env python3
"""Compile every tracked Python source in memory without creating bytecode files."""

from __future__ import annotations

import argparse
from pathlib import Path

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
}


def iter_python_files(root: Path):
    for path in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in path.parts):
            continue
        yield path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()

    checked = 0
    for path in iter_python_files(root):
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec", dont_inherit=True)
        checked += 1

    print(f"PYTHON_SYNTAX_CHECK=PASS files={checked}")


if __name__ == "__main__":
    main()
