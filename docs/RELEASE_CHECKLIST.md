# Public release checklist

## Identity and repository

- [ ] Repository is named `Multisymboliccore/fields-lm`.
- [ ] Default branch is `main`.
- [ ] Apache 2.0 `LICENSE` and `NOTICE` are present.
- [ ] Repository description is accurate and avoids universal performance claims.

## Source and tests

- [ ] Canonical source SHA-256 passes.
- [ ] Frozen source manifest passes.
- [ ] `python -m compileall -q src scripts examples tests` passes.
- [ ] `pytest -q` passes.
- [ ] CUDA smoke test passes on the validated H100 environment.
- [ ] `ENVIRONMENT.txt` is generated from the release machine.

## Scientific reporting

- [ ] Final three-seed values are copied from generated outputs, not typed from memory.
- [ ] Training budget is stated as 49,152,000 tokens, not full-dataset training.
- [ ] Training context is stated as 2K; 64K is described as evaluation/extrapolation.
- [ ] Paper link and BibTeX are added when public.

## Security and privacy

- [ ] No private key, token, password, cookie, or cloud credential is present.
- [ ] No personal home address is present.
- [ ] No active instance IP address is present.
- [ ] No unpublished dataset or checkpoint is accidentally committed.

## Hugging Face

- [ ] Export uses `model.safetensors`.
- [ ] Tokenizer vocabulary is exactly 16,384.
- [ ] Round-trip report is `PASS`.
- [ ] Model card describes intended use, limits, training data, and evaluation.
- [ ] Remote download hash matches the local artifact.
