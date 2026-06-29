# Reproducibility and source identity

## Canonical Field source

The exact canonical Field kernel expected by this release has SHA-256:

```text
0454647a7f667cdf73f45029eee0f525c58c0788ecd433ec1a4db8c3937ac848
```

`verify_frozen_source()` checks this hash and every file listed in `OFFICIAL_SOURCE_MANIFEST.json` before model construction.

## Experimental protocol represented by the release

- dataset: PG-19 sampled training documents;
- tokenizer: BPE, vocabulary size 16,384;
- training budget: 49,152,000 tokens per model and seed;
- training sequence length: 2,048;
- paired models and data ordering within each seed;
- long-context scoring at 2K, 8K, 16K, 32K, and 64K;
- three-seed confirmation protocol: model seeds 1234, 2345, 3456.

The checkpoint is not described as “trained at 64K.” It is trained at 2K and evaluated at longer contexts.

## Release gates

Before a public tag:

1. canonical and manifest hashes pass;
2. Python sources compile under Python 3.10;
3. CPU contract tests pass;
4. a CUDA smoke test constructs all promoted components;
5. exported `safetensors` weights reload with exact state-dict equality;
6. logits and loss match the source checkpoint under the registered round-trip test;
7. the environment manifest is recorded;
8. no secrets, private keys, personal paths, or instance IP addresses are committed.
