# Third-party notices

Fields LM depends on third-party projects that retain their own copyright and license terms.

## Runtime dependencies

- PyTorch
- Triton
- Mamba / Mamba-2 (`mamba-ssm`)
- causal-conv1d
- Hugging Face Hub
- safetensors
- tokenizers / Transformers
- NumPy

The release does not relicense these dependencies under the Fields LM Apache 2.0 license. Users are responsible for complying with the license terms of the versions they install.

## Frozen research source

The frozen source directory contains the project’s own experiment-derived construction and audit code. External packages are imported as dependencies rather than vendored when practical. Any future vendored third-party source must preserve its original notice and license in this file and beside the relevant files.
