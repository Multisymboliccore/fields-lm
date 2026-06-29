# Fields LM architecture

## Promoted configuration

The public release reconstructs the exact architecture used by the promoted paper arm:

| Component | Count | Positions |
|---|---:|---|
| Native Field block | 18 | 0–4, 6–9, 12–16, 18–21 |
| Mamba-2 editor | 2 | 10, 22 |
| Refresh attention | 4 | 5, 11, 17, 23 |
| PCAF successor memory | 1 output path | enabled from initialization |

The model uses a 16,384-token BPE vocabulary, model width 1,024, 24 total backbone blocks, tied output embeddings, and an approximate 300M-parameter target.

## Native Field block

The native Field component is the architecture-specific recurrent unit. The frozen implementation contains the promoted forms of:

- independent vacancy;
- causal write dynamics;
- complex-valued recurrent state evolution;
- displaced readout;
- exact chunked execution;
- a Triton acceleration path.

The readable public API deliberately does not rewrite these equations. It verifies and imports the frozen implementation that produced the paper model.

## Refresh attention

Refresh stations provide bounded local token-token interaction at selected depths. The registered windows are 256, 512, 1,024, and 1,024 tokens. They are not full-context quadratic attention layers.

## Mamba-2 editors

Two official Mamba-2 blocks are inserted as state-space editors. Mamba-2 remains an external dependency and retains its own license and attribution.

## PCAF successor memory

PCAF is enabled in the promoted model from update zero. The registered ablation disables PCAF before the first optimization update; toggling it only after training is not considered an equivalent ablation.

## Why the release includes frozen source

The research path involved parameter matching, block replacement, optimizer grouping, cache routing, and custom kernels. A cosmetic rewrite can preserve shapes while changing behavior. The public release therefore has two layers:

1. a conventional documented `nn.Module` interface;
2. a hash-audited frozen source snapshot underneath it.

This design prioritizes reproducibility over aesthetic simplification.
