#!/usr/bin/env python3
"""Fail the release when common secrets or private-machine artifacts are found."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

SKIP_PARTS = {".git", ".venv", "__pycache__", "dist", "build", ".pytest_cache"}
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".gz", ".safetensors", ".pt"}
PATTERNS = {
    "private_key": re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
    "pem_filename": re.compile(r"\.pem\b", re.IGNORECASE),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "hf_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "aws_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "windows_user_path": re.compile(r"C:\\Users\\[^\\\s]+", re.IGNORECASE),
    "active_cloud_ip": re.compile(r"\b192\.222\.54\.32\b"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    findings: list[str] = []

    for path in root.rglob("*"):
        if path.resolve() == Path(__file__).resolve():
            continue
        if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for name, pattern in PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{name}: {path.relative_to(root)}")

    if findings:
        print("PUBLIC TREE SCAN: FAIL")
        for finding in findings:
            print(" -", finding)
        raise SystemExit(1)
    print("PUBLIC TREE SCAN: PASS")


if __name__ == "__main__":
    main()
