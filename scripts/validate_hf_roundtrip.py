#!/usr/bin/env python3
"""Validate arena-checkpoint -> Hub-artifact numerical equivalence on GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fields_official import FieldsHubModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--sequence-length", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260629)
    p.add_argument("--report", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    artifact = args.artifact.expanduser().resolve()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if raw.get("format") != "field_fusion_pg19_official_bf16":
        raise RuntimeError(f"unexpected checkpoint format: {raw.get('format')!r}")
    metadata = dict(raw.get("metadata", {}))
    model_seed = int(metadata.get("model_seed", 1234))

    native = FieldsHubModel(model_seed=model_seed, pcaf_enabled=True, amp="bf16")
    incompatible = native.core_model.load_state_dict(raw["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"native checkpoint mismatch: {incompatible}")

    restored = FieldsHubModel.from_pretrained(str(artifact), map_location="cpu")
    native = native.to(device=device, dtype=torch.bfloat16).eval()
    restored = restored.to(device=device, dtype=torch.bfloat16).eval()

    tokens = torch.randint(0, 16_384, (1, args.sequence_length), device=device, dtype=torch.long)
    labels = torch.roll(tokens, shifts=-1, dims=1)

    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        native_logits = native(tokens).logits
        restored_logits = restored(tokens).logits
        native_loss = native(tokens, labels).loss
        restored_loss = restored(tokens, labels).loss

    logits_diff = float((native_logits.float() - restored_logits.float()).abs().max().cpu())
    loss_diff = float((native_loss.float() - restored_loss.float()).abs().cpu())
    exact_state = True
    native_state = native.state_dict()
    restored_state = restored.state_dict()
    if native_state.keys() != restored_state.keys():
        exact_state = False
    else:
        for key in native_state:
            if not torch.equal(native_state[key].detach().cpu(), restored_state[key].detach().cpu()):
                exact_state = False
                break

    passed = exact_state and logits_diff == 0.0 and loss_diff == 0.0
    report = {
        "status": "PASS" if passed else "FAIL",
        "checkpoint": str(checkpoint),
        "artifact": str(artifact),
        "device": str(device),
        "sequence_length": args.sequence_length,
        "state_dict_exact": exact_state,
        "max_abs_logits_diff": logits_diff,
        "abs_loss_diff": loss_diff,
        "pcaf_enabled": bool(restored.pcaf_enabled_at_build),
        "backend_name": restored.backend_name,
        "canonical_sha256": restored.source_audit["canonical_sha256"],
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.report:
        report_path = args.report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text, encoding="utf-8")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
