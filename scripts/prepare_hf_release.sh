#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python}"
RUN_ROOT="${RUN_ROOT:-/home/ubuntu/pcaf_runs/field_fusion_final_closure_run}"
OUTPUT="${OUTPUT:-/home/ubuntu/fields_hf_export}"
TOKENIZER="${TOKENIZER:-/home/ubuntu/field_lab/field_fusion_pg19_official_data/tokenizer/tokenizer.json}"

CHECKPOINT="${CHECKPOINT:-}"
if [[ -z "$CHECKPOINT" ]]; then
  CHECKPOINT="$(find "$RUN_ROOT/quality" -type f \
    -path '*/seed0_1234/field_official_18f2m4r_pcaf_on/final_bf16.pt' \
    -print -quit 2>/dev/null || true)"
fi
[[ -n "$CHECKPOINT" && -f "$CHECKPOINT" ]] || {
  echo "ERROR: promoted seed-1234 checkpoint not found; set CHECKPOINT=/path/final_bf16.pt" >&2
  exit 2
}
[[ -f "$TOKENIZER" ]] || { echo "ERROR: tokenizer not found: $TOKENIZER" >&2; exit 3; }

rm -rf "$OUTPUT"
mkdir -p "$OUTPUT"

"$PYTHON" "$REPO_ROOT/scripts/export_hf_checkpoint.py" \
  --checkpoint "$CHECKPOINT" \
  --tokenizer "$TOKENIZER" \
  --output "$OUTPUT" \
  --model-card "$REPO_ROOT/hf/README.md" \
  --environment "$REPO_ROOT/ENVIRONMENT.txt"

"$PYTHON" "$REPO_ROOT/scripts/validate_hf_roundtrip.py" \
  --checkpoint "$CHECKPOINT" \
  --artifact "$OUTPUT" \
  --device cuda \
  --report "$OUTPUT/EQUIVALENCE_REPORT.json"

sha256sum "$OUTPUT"/* > "$OUTPUT/SHA256SUMS.txt"

echo "FIELDS_HF_ARTIFACT_READY=$OUTPUT"
