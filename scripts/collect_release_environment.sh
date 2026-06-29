#!/usr/bin/env bash
set -Eeuo pipefail
OUT="${1:-ENVIRONMENT.txt}"
{
  echo "generated_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "kernel=$(uname -srmo)"
  echo
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
  echo
  python - <<'PY'
import importlib
import platform
import sys

mods = [
    "torch",
    "triton",
    "mamba_ssm",
    "causal_conv1d",
    "transformers",
    "tokenizers",
    "huggingface_hub",
    "safetensors",
]
print("python", sys.version.replace("\n", " "))
print("platform", platform.platform())
for name in mods:
    try:
        mod = importlib.import_module(name)
        print(name, getattr(mod, "__version__", "unknown"))
    except Exception as exc:
        print(name, "UNAVAILABLE", type(exc).__name__)
if "torch" in sys.modules:
    import torch
    print("torch_cuda", torch.version.cuda)
    print("torch_cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("torch_gpu", torch.cuda.get_device_name(0))
PY
} > "$OUT"
echo "environment_report=$OUT"
