---
library_name: fields-lm
license: apache-2.0
pipeline_tag: text-generation
tags:
- pytorch
- causal-lm
- long-context
- state-space-model
- mamba
- research
language:
- en
base_model: null
datasets:
- emozilla/pg19
---

# Fields 300M PG-19

This repository contains the first public checkpoint for **Fields LM**, an experimental causal language-model architecture combining native Field recurrence, displaced readout, PCAF successor memory, local refresh attention, and Mamba-2 editors.

## Model details

- architecture: Fields 18F/2M/4R + PCAF;
- scale: approximately 300M parameters;
- vocabulary: 16,384-token BPE;
- training data: controlled sample of PG-19;
- training budget: 49,152,000 tokens;
- training context: 2,048 tokens;
- evaluated contexts: 2K, 8K, 16K, 32K, and 64K;
- weights: `safetensors`;
- runtime: install from the official GitHub repository.

This is **not** a frontier-scale assistant and is not instruction-tuned. Long-context values beyond 2K measure evaluation-time extrapolation, not training at those lengths.

## Installation

```bash
pip install "git+https://github.com/Multisymboliccore/fields-lm.git"
```

Install the validated CUDA dependencies for Mamba-2 and causal convolution as documented in the GitHub repository.

## Loading

```python
import torch
from fields_official import FieldsHubModel

model = FieldsHubModel.from_pretrained(
    "Multisymboliccore/fields-300m-pg19",
    map_location="cpu",
)
model = model.to(device="cuda", dtype=torch.bfloat16).eval()
```

## Intended use

The checkpoint is intended for architecture research, reproducibility, controlled ablations, long-context analysis, and further pretraining or fine-tuning by qualified users.

## Limitations

- trained with a comparatively small token budget;
- English literary-book domain bias from PG-19;
- not instruction-tuned or safety-tuned;
- may generate inaccurate, biased, or harmful text;
- custom CUDA dependencies are required for the validated high-performance path;
- 64K evaluation does not imply equal quality across all long-context tasks.

## Evaluation

Final three-seed metrics and paper links will be synchronized from the validated release artifacts. Do not infer universal model quality from the architecture benchmark alone.

## License

Apache License 2.0. Dependency licenses continue to apply.
