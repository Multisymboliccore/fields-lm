#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARCHIVE="${ARCHIVE:-/home/ubuntu/fields-lm-github-release.tar.gz}"
PYTHON="${PYTHON:-python}"

cd "$ROOT"
find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name "*.egg-info" \) -prune -exec rm -rf {} +
find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
"$PYTHON" scripts/scan_public_tree.py .
"$PYTHON" scripts/check_python_syntax.py .

find src/fields_official/reference -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > SHA256SUMS_SOURCE.txt
find . -type f \
  ! -path './.git/*' \
  ! -name 'SHA256SUMS_RELEASE.txt' \
  -print0 \
  | sort -z \
  | xargs -0 sha256sum > SHA256SUMS_RELEASE.txt

rm -f "$ARCHIVE"
tar -czf "$ARCHIVE" -C "$(dirname "$ROOT")" "$(basename "$ROOT")"
echo "SEALED_RELEASE=$ARCHIVE"
