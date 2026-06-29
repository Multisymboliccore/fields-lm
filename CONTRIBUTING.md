# Contributing to Fields LM

Fields LM is a research codebase whose central requirement is **scientific equivalence**. Readability and speed improvements are welcome, but changes to recurrence, initialization, memory routing, topology, precision, or kernels must not be merged on intuition alone.

## Development setup

```bash
python -m pip install -e ".[dev,hf]"
pytest -q
```

GPU-dependent changes should also be tested in the validated CUDA environment recorded in `ENVIRONMENT.txt`.

## Pull-request expectations

1. Explain the research or engineering motivation.
2. Keep the public API separate from frozen paper source.
3. Add or update tests.
4. For mathematical changes, provide one of:
   - exact tensor equivalence;
   - bounded numerical-error analysis;
   - a controlled quality/speed/memory benchmark.
5. Do not commit datasets, checkpoints, API tokens, private paths, IP addresses, or machine credentials.
6. Preserve third-party copyright and attribution notices.

## Scope

Good first contributions include documentation, additional contract tests, portable fallbacks, packaging improvements, profiling tools, and independent reproduction scripts.
