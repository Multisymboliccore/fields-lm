# Publishing workflow

## GitHub

The GitHub repository contains code, documentation, tests, and source-identity manifests. It should not contain trained checkpoints, datasets, private keys, cloud IP addresses, or experiment working directories.

Before a public release:

```bash
python scripts/scan_public_tree.py .
python scripts/check_python_syntax.py .
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider
```

Create a signed or annotated tag after the CUDA smoke test and Hugging Face round-trip validation pass.

## Hugging Face Hub

The model repository contains:

- `model.safetensors`;
- `config.json`;
- tokenizer files;
- model card (`README.md`);
- environment and source-identity reports;
- exact equivalence report.

Upload the generated folder with:

```bash
hf upload Multisymboliccore/fields-300m-pg19 /path/to/fields_hf_export .
```

The Hugging Face model card must state that the checkpoint was trained with a 2K context and evaluated at longer contexts.
