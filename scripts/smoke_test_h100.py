#!/usr/bin/env python3
"""Construct the promoted model and run a short CUDA forward/backward smoke test."""

from __future__ import annotations

import json

import torch

from fields_official import FieldsConfig, build_official_fields


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the release smoke test")

    device = torch.device("cuda")
    model = build_official_fields(FieldsConfig(), device=device).to(dtype=torch.bfloat16)
    model.train()
    tokens = torch.randint(0, 16_384, (1, 32), device=device, dtype=torch.long)
    labels = torch.roll(tokens, -1, 1)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(tokens, labels)
    output.loss.backward()

    report = {
        "status": "PASS",
        "loss": float(output.loss.detach().float().cpu()),
        "backend": model.backend_name,
        "pcaf_enabled": model.pcaf_enabled,
        "canonical_sha256": model.source_audit["canonical_sha256"],
        "topology": model.topology_audit,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
