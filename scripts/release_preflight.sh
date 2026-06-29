#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python}"

cd "$ROOT"
"$PYTHON" scripts/scan_public_tree.py .
"$PYTHON" scripts/check_python_syntax.py .
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m pytest -q -p no:cacheprovider

echo "RELEASE_PREFLIGHT=PASS"
