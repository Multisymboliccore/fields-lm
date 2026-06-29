#!/usr/bin/env python3
"""One illustrative optimization step on a randomly initialized Fields model."""

from __future__ import annotations

import torch

from fields_official import FieldsConfig, build_official_fields


def main() -> None:
    device = torch.device("cuda")
    model = build_official_fields(
        FieldsConfig(pcaf_enabled=True),
        device=device,
    ).to(dtype=torch.bfloat16)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    tokens = torch.randint(0, 16_384, (1, 128), device=device, dtype=torch.long)
    labels = torch.roll(tokens, shifts=-1, dims=1)

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(tokens, labels)
    output.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    print("loss:", float(output.loss.detach().cpu()))


if __name__ == "__main__":
    main()
