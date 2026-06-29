<div align="center">

# Fields LM

**An experimental long-context language-model architecture built from recurrent Field blocks, displaced readout, PCAF successor memory, refresh attention, and Mamba-2 editors.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![Status](https://img.shields.io/badge/status-research%20release-orange.svg)](#project-status)
[![CI](https://github.com/Multisymboliccore/fields-lm/actions/workflows/ci.yml/badge.svg)](https://github.com/Multisymboliccore/fields-lm/actions/workflows/ci.yml)

Official research implementation by **MSC Technology**.

</div>

## Overview

Fields LM is a causal sequence architecture developed as an alternative long-context backbone. The promoted research configuration combines:

- **18 native Field blocks** using the validated complex recurrence, independent vacancy, causal writes, and displaced readout;
- **2 Mamba-2 editor blocks**;
- **4 local refresh-attention stations**;
- **PCAF successor memory** at the language-model output path.

The first public checkpoint targets approximately **300 million parameters** and was trained from scratch on a controlled **49,152,000-token sample of PG-19**, using a 16,384-token BPE vocabulary and a 2,048-token training window. It is a research checkpoint, not a frontier-scale general-purpose assistant.

> [!IMPORTANT]
> This repository separates the readable public API from the exact frozen source used for the paper experiments. The release builder verifies every frozen source file by SHA-256 before model construction. This avoids silently changing the recurrence, initialization, PCAF routing, or block topology while cleaning up the public interface.

## Architecture

```text
BPE-16K tokens → embedding (d_model = 1024)
  0–4    Native Field
  5      Refresh attention (window 256)
  6–9    Native Field
  10     Mamba-2 editor
  11     Refresh attention (window 512)
  12–16  Native Field
  17     Refresh attention (window 1024)
  18–21  Native Field
  22     Mamba-2 editor
  23     Refresh attention (window 1024)
→ final normalization → tied LM head → PCAF successor memory
```

The canonical topology is therefore **18F / 2M / 4R + PCAF**. See [Architecture](docs/ARCHITECTURE.md) for the block-level explanation and source identity rules.

## Project status

This is an **experimental research release**. The architecture and publication package are intended for reproduction, inspection, ablation, and further research.

What is established by the release:

- exact source identity and topology auditing;
- paired PG-19 experiments at a controlled token budget;
- evaluation code for contexts from 2K through 64K;
- a registered no-PCAF-from-initialization ablation;
- comparison infrastructure for PCAF-Conv, Transformer + YaRN, and official Mamba-2.

What is not claimed:

- universal superiority over Transformers or state-space models;
- production readiness;
- frontier language quality;
- safety for unsupervised deployment;
- 64K training—the promoted checkpoint is trained at 2K and evaluated beyond that window.

## Installation

### Validated CUDA environment

The exact GPU package versions are recorded by `scripts/collect_release_environment.sh` when the release is assembled on the paper machine.

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch
python -m pip install causal-conv1d --no-build-isolation
python -m pip install mamba-ssm --no-build-isolation
python -m pip install -e ".[hf]"
```

For development:

```bash
python -m pip install -e ".[dev,hf]"
pytest -q
```

## Quick start

### Build a randomly initialized promoted architecture

```python
import torch
from fields_official import FieldsConfig, build_official_fields

model = build_official_fields(
    FieldsConfig(pcaf_enabled=True),
    device="cuda",
).to(dtype=torch.bfloat16).eval()

input_ids = torch.randint(0, 16_384, (1, 128), device="cuda")

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    output = model(input_ids)

print(output.logits.shape)
```

### Load the released Hugging Face checkpoint

After the model repository is published:

```python
import torch
from fields_official import FieldsHubModel

model = FieldsHubModel.from_pretrained(
    "Multisymboliccore/fields-300m-pg19",
    map_location="cpu",
)
model = model.to(device="cuda", dtype=torch.bfloat16).eval()
```

The runtime is installed from this GitHub repository; weights and tokenizer files are distributed separately through the Hugging Face Hub in `safetensors` format.

## Repository layout

```text
fields-lm/
├── src/fields_official/       Public API and frozen reference source
├── examples/                  Minimal inference and training examples
├── scripts/                   Release, checkpoint, and HF utilities
├── tests/                     Source identity and API contract tests
├── docs/                      Architecture and reproducibility notes
├── hf/                        Hugging Face model-card template
└── .github/                   CI and contribution templates
```

## Reproducibility

The release embeds:

- a frozen source manifest;
- SHA-256 verification of the canonical Field kernel;
- topology checks for 18 Field, 2 Mamba-2, and 4 refresh blocks;
- exact `safetensors` round-trip validation for the published checkpoint;
- scripts for recording the CUDA/PyTorch/Triton/Mamba environment.

See [Reproducibility](docs/REPRODUCIBILITY.md).

## Results and paper

The repository intentionally does not mix preliminary single-seed values with the final paper table. Final three-seed results, the paper citation, and the central 64K comparison should be copied into [Results](docs/RESULTS.md) from the validated arena summary before the public announcement.

## Citation

Until the paper identifier is available, cite the software release using [`CITATION.cff`](CITATION.cff). The paper citation will be added when the manuscript is publicly posted.

## Contributing

Mathematical changes should include an equivalence test or a controlled benchmark. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Fields LM is released under the [Apache License 2.0](LICENSE). Third-party packages and source components remain subject to their respective licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
