#!/usr/bin/env python3
"""Minimal Fields LM inference example using a Hugging Face checkpoint."""

from __future__ import annotations

import argparse

import torch
from transformers import PreTrainedTokenizerFast

from fields_official import FieldsHubModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Multisymboliccore/fields-300m-pg19")
    parser.add_argument("--prompt", default="The history of computation")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model)
    model = FieldsHubModel.from_pretrained(args.model, map_location="cpu")
    model = model.to(device=args.device, dtype=torch.bfloat16).eval()

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(args.device)
    with torch.no_grad(), torch.autocast(
        device_type=torch.device(args.device).type,
        dtype=torch.bfloat16,
        enabled=torch.device(args.device).type == "cuda",
    ):
        output = model(input_ids)

    next_token = int(output.logits[:, -1].argmax(dim=-1).item())
    print("input_shape:", tuple(input_ids.shape))
    print("logits_shape:", tuple(output.logits.shape))
    print("greedy_next_token_id:", next_token)
    print("greedy_next_token:", tokenizer.decode([next_token]))


if __name__ == "__main__":
    main()
