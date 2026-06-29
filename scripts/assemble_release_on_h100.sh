#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="${DEST:-/home/ubuntu/fields-lm-github-release}"
EXPECTED="0454647a7f667cdf73f45029eee0f525c58c0788ecd433ec1a4db8c3937ac848"

find_canonical() {
  local candidate actual
  for candidate in \
    "${CANONICAL:-}" \
    /home/ubuntu/field_fusion_reengineering_r1_redundancy_map/field_only_v4_chunked_triton_wiki100.py \
    /home/ubuntu/field_fusion_final_closure_arena_H100/field_only_v4_chunked_triton_wiki100.py \
    /home/ubuntu/field_fusion_pg19_memory_validation_H100/field_only_v4_chunked_triton_wiki100.py \
    /home/ubuntu/field_only_v4_chunked_triton_wiki100.py; do
    [[ -n "$candidate" && -f "$candidate" ]] || continue
    actual="$(sha256sum "$candidate" | awk '{print $1}')"
    if [[ "$actual" == "$EXPECTED" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  while IFS= read -r candidate; do
    actual="$(sha256sum "$candidate" | awk '{print $1}')"
    if [[ "$actual" == "$EXPECTED" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done < <(find /home/ubuntu -type f -name field_only_v4_chunked_triton_wiki100.py 2>/dev/null | sort)
  return 1
}

CANONICAL_PATH="$(find_canonical)" || {
  echo "ERROR: canonical source with expected SHA-256 was not found under /home/ubuntu" >&2
  exit 2
}

echo "canonical_source=$CANONICAL_PATH"
echo "canonical_sha256=$EXPECTED"

rm -rf "$DEST"
cp -a "$REPO_ROOT" "$DEST"
mkdir -p "$DEST/src/fields_official/reference"
cp -a "$CANONICAL_PATH" "$DEST/src/fields_official/reference/field_only_v4_chunked_triton_wiki100.py"

printf 'canonical_source=%s\ncanonical_sha256=%s\n' \
  "src/fields_official/reference/field_only_v4_chunked_triton_wiki100.py" \
  "$EXPECTED" > "$DEST/SOURCE_IDENTITY.txt"

cd "$DEST"
find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name "*.egg-info" \) -prune -exec rm -rf {} +
find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
python scripts/scan_public_tree.py .
python scripts/check_python_syntax.py .

find src/fields_official/reference -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > SHA256SUMS_SOURCE.txt
find . -type f \
  ! -path './.git/*' \
  ! -name 'SHA256SUMS_RELEASE.txt' \
  -print0 \
  | sort -z \
  | xargs -0 sha256sum > SHA256SUMS_RELEASE.txt

TAR=/home/ubuntu/fields-lm-github-release.tar.gz
rm -f "$TAR"
tar -czf "$TAR" -C "$(dirname "$DEST")" "$(basename "$DEST")"

echo "FIELDS_GITHUB_TREE_READY=$DEST"
echo "FIELDS_GITHUB_ARCHIVE_READY=$TAR"
