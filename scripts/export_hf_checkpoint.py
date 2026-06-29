#!/usr/bin/env python3
"""Convert a paper arena export into a safe Hugging Face model artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from tokenizers import Tokenizer

from fields_official import FieldsHubModel


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-seed", type=int, default=1234)
    p.add_argument("--embedding-seed", type=int, default=314159)
    p.add_argument("--model-card", type=Path, default=None)
    p.add_argument("--environment", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    tokenizer_path = args.tokenizer.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if raw.get("format") != "field_fusion_pg19_official_bf16":
        raise RuntimeError(f"unexpected checkpoint format: {raw.get('format')!r}")
    model_name = str(raw.get("model", ""))
    if "field" not in model_name or "pcaf_off" in model_name:
        raise RuntimeError(f"checkpoint is not the promoted Fields+PCAF arm: {model_name!r}")

    metadata = dict(raw.get("metadata", {}))
    model_seed = int(metadata.get("model_seed", args.model_seed))
    model = FieldsHubModel(
        model_seed=model_seed,
        embedding_seed=args.embedding_seed,
        pcaf_enabled=True,
        amp="bf16",
    )
    incompatible = model.core_model.load_state_dict(raw["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"state-dict mismatch: {incompatible}")
    model.eval()
    model._save_pretrained(output)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() != 16_384:
        raise RuntimeError(f"tokenizer vocab mismatch: {tokenizer.get_vocab_size()}")
    if tokenizer.token_to_id("<unk>") is None:
        raise RuntimeError("tokenizer is missing <unk>")
    shutil.copy2(tokenizer_path, output / "tokenizer.json")
    (output / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "model_max_length": 65536,
                "unk_token": "<unk>",
                "clean_up_tokenization_spaces": False,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    (output / "special_tokens_map.json").write_text(
        json.dumps({"unk_token": "<unk>"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.model_card:
        shutil.copy2(args.model_card.expanduser().resolve(), output / "README.md")
    if args.environment:
        shutil.copy2(args.environment.expanduser().resolve(), output / "ENVIRONMENT.txt")

    identity = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "tokenizer_sha256": sha256(tokenizer_path),
        "weights_sha256": sha256(output / "model.safetensors"),
        "model": model_name,
        "metadata": metadata,
        "canonical_sha256": model.source_audit["canonical_sha256"],
    }
    (output / "SOURCE_IDENTITY.txt").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(identity, indent=2, sort_keys=True))
    print(f"HF_ARTIFACT_READY={output}")


if __name__ == "__main__":
    main()
