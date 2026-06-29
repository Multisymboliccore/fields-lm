#!/usr/bin/env python3
"""FIELD-FUSION CANONICAL 300M v25 — WikiText-103 100% three-way arena.

Competitors
-----------
* field_fusion_fast: promoted Field-Fusion from v22/v23, fast PCAF and exact
  field_half activation recomputation.
* transformer_flash_300m: matched-parameter Flash-SDPA Transformer.
* mamba2_official_300m: official state-spaces/mamba Mamba-2 kernels,
  d_state=128, d_conv=4, expand=2, headdim=64, chunk_size=256,
  use_mem_eff_path=True, with no outer torch checkpoint wrapper.

All three models use the same 16,384-token byte-level BPE, the same fixed token
budget, deterministic training windows, tied input/output embeddings, BF16 and
AdamW schedule. Mamba-2 width/depth are solved at runtime against the same 300M
parameter target using the installed official package.
"""
from __future__ import annotations

import gc
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import field_fusion_wiki100_canonical_v23 as v23

try:
    import causal_conv1d  # type: ignore
    import mamba_ssm  # type: ignore
    from mamba_ssm import Mamba2 as OfficialMamba2  # type: ignore
    MAMBA_VERSION = getattr(mamba_ssm, "__version__", "unknown")
    CAUSAL_CONV1D_VERSION = getattr(causal_conv1d, "__version__", "unknown")
    MAMBA_IMPORT_ERROR = ""
except Exception as exc:  # Keep --help usable before environment preparation.
    OfficialMamba2 = None
    MAMBA_VERSION = "unavailable"
    CAUSAL_CONV1D_VERSION = "unavailable"
    MAMBA_IMPORT_ERROR = repr(exc)

FUSION = v23.FUSION
TRANSFORMER = v23.TRANSFORMER
MAMBA2 = "mamba2_official_300m"
MODELS = (FUSION, TRANSFORMER, MAMBA2)
REQUIRED_MAMBA_VERSION = "2.3.2.post1"

# Save the v23 implementations before monkey-patching its module globals.
_ORIG_BUILD_MODEL = v23.build_model
_ORIG_LOSS_CALL = v23.loss_call
_ORIG_TOKEN_NLL = v23.token_nll
_ORIG_SET_DISTILL = v23.set_distill
_ORIG_CHECKPOINT_SIGNATURE = v23.checkpoint_signature
_ORIG_MAKE_SUMMARY = v23.make_summary


class FastRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class Mamba2ResidualBlock(nn.Module):
    def __init__(self, dim: int, args) -> None:
        super().__init__()
        if OfficialMamba2 is None:
            raise RuntimeError("Official Mamba-2 unavailable: " + MAMBA_IMPORT_ERROR)
        self.norm = FastRMSNorm(dim)
        self.mixer = OfficialMamba2(
            d_model=dim,
            d_state=args.mamba_d_state,
            d_conv=args.mamba_d_conv,
            expand=args.mamba_expand,
            headdim=args.mamba_headdim,
            ngroups=args.mamba_ngroups,
            chunk_size=args.mamba_chunk_size,
            use_mem_eff_path=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class OfficialMamba2LM(nn.Module):
    """Pure Mamba-2 LM using official fused kernels and tied embeddings."""

    def __init__(self, vocab: int, dim: int, layers: int, args) -> None:
        super().__init__()
        self.activation_dtype = torch.bfloat16
        self.emb = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([
            Mamba2ResidualBlock(dim, args) for _ in range(layers)
        ])
        self.norm = FastRMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.emb(tokens).to(self.activation_dtype)
        # Do not wrap official Mamba custom autograd kernels in
        # torch.utils.checkpoint: this caused CheckpointError in prior arenas.
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))


def nparams(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def mamba_block_params(dim: int, args) -> int:
    block = Mamba2ResidualBlock(dim, args)
    value = nparams(block)
    del block
    gc.collect()
    return value


def solve_mamba_shape(args) -> v23.Shape:
    """Find the closest official Mamba-2 width/depth at the 300M target."""
    candidates = []
    preferred_layers = int(args.mamba_preferred_layers)
    for dim in range(args.mamba_min_dim, args.mamba_max_dim + 1,
                     args.mamba_headdim):
        if (args.mamba_expand * dim) % args.mamba_headdim:
            continue
        try:
            per_block = mamba_block_params(dim, args)
        except Exception:
            continue
        # Tied embedding/head: one vocabulary matrix, not two.
        fixed = args.vocab_size * dim + dim
        for layers in range(args.mamba_min_layers, args.mamba_max_layers + 1):
            total = fixed + layers * per_block
            relative = abs(total - args.target_params) / args.target_params
            candidates.append((relative, abs(layers - preferred_layers),
                               abs(dim - 1024), dim, layers, total))
    if not candidates:
        raise RuntimeError("No valid official Mamba-2 parameter candidates")
    _, _, _, dim, layers, total = min(candidates)
    delta = 100.0 * (total - args.target_params) / args.target_params
    if abs(delta) > args.max_param_delta_pct:
        raise RuntimeError(
            f"Mamba-2 parameter mismatch {delta:+.3f}% exceeds "
            f"{args.max_param_delta_pct:.3f}%"
        )
    return v23.Shape(
        MAMBA2, int(total), int(dim), int(layers),
        int((args.mamba_expand * dim) // args.mamba_headdim), 0,
    )


def solve_shapes_v25(args, deps) -> Dict[str, v23.Shape]:
    arena, v3, canonical, bridge, optmod, epi, judge = deps
    out: Dict[str, v23.Shape] = {}
    for name in (FUSION, TRANSFORMER):
        # v23 names -> v22 public names -> v21 internal implementation names.
        # Calling v21 directly with "fusion_fast" raises KeyError because v21
        # only knows its internal constant (for example V21_FAST).
        internal_name = v23.v22.mapped(v23.TO_V22[name])
        raw = v23.v21.solve_shape(
            internal_name, args, arena, v3, canonical, bridge,
            optmod, epi, judge,
        )
        shape = v23.Shape(name, raw.params, raw.dim, raw.layers,
                          raw.heads, raw.ff_hidden)
        delta = 100.0 * (shape.params - args.target_params) / args.target_params
        if abs(delta) > args.max_param_delta_pct:
            raise RuntimeError(f"parameter mismatch {name}: {delta:+.3f}%")
        out[name] = shape
    out[MAMBA2] = solve_mamba_shape(args)
    return out


def build_model_v25(name: str, shape: v23.Shape, args, deps,
                    device: torch.device) -> nn.Module:
    if name != MAMBA2:
        return _ORIG_BUILD_MODEL(name, shape, args, deps, device)
    v23.core.seed_all(args.model_seed)
    model = OfficialMamba2LM(
        args.vocab_size, shape.dim, shape.layers, args
    ).to(device)
    v23.v21.tied_embedding_init(model, args.embedding_seed)
    return model


def set_distill_v25(model: nn.Module, value: float) -> None:
    if isinstance(model, OfficialMamba2LM):
        return
    _ORIG_SET_DISTILL(model, value)


def loss_call_v25(name: str, model: nn.Module,
                  x: torch.Tensor, y: torch.Tensor):
    if name != MAMBA2:
        return _ORIG_LOSS_CALL(name, model, x, y)
    logits = model(x)
    primary = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1)
    )
    return primary, primary.detach()


def _fast_successor_token_nll(cache: nn.Module, states: torch.Tensor,
                              logits: torch.Tensor, tokens: torch.Tensor,
                              targets: torch.Tensor) -> torch.Tensor:
    """Exact per-token NLL for the target-only FastSuccessorCacheV5 path."""
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    param_nll = F.cross_entropy(flat_logits.float(), flat_targets, reduction="none")
    if not bool(getattr(cache, "enabled", True)):
        return param_nll.view_as(targets)
    param_target = torch.exp(-param_nll)
    b, t, _ = states.shape
    cache_module = sys.modules.get(cache.__class__.__module__)
    fast_candidates = getattr(cache_module, "causal_recent_candidates_i32", None) if cache_module is not None else None
    if bool(getattr(cache, "use_i32", False)) and fast_candidates is not None:
        idx = fast_candidates(tokens, cache.order, cache.num_buckets, cache.top_k, cache._v3)
    else:
        idx = cache._v3.causal_recent_candidates(tokens, cache.order, cache.num_buckets, cache.top_k)
    valid = idx >= 0
    has = valid.any(-1)
    safe = idx.clamp_min(0)
    batch_idx = torch.arange(b, device=states.device)[:, None, None]
    proj = cache._v3.normalize_rows(F.linear(states.float(), cache.shared_weight.float()))
    q = proj[:, :, None, :]
    ck = proj[batch_idx, safe]
    scores = (ck * q).sum(-1) * (cache.memory_dim ** -0.5)
    recency = safe.float() / max(float(t - 1), 1.0)
    scores = scores + cache.recency_scale.float() * recency
    scores = scores.masked_fill(~valid, -1.0e9)
    weights = torch.softmax(scores.float(), dim=-1) * valid.float()
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
    cand_tokens = targets[batch_idx, safe]
    target_cache = (weights * (cand_tokens == targets[:, :, None]).float()).sum(-1)
    active = has.reshape(-1)
    state_logit = cache.state_gate(states).float().squeeze(-1)
    gate_flat = torch.zeros_like(state_logit.reshape(-1))
    if bool(active.any()):
        features = cache._features(
            scores.reshape(-1, cache.top_k)[active], weights.reshape(-1, cache.top_k)[active],
            valid.reshape(-1, cache.top_k)[active], cand_tokens.reshape(-1, cache.top_k)[active],
            recency.reshape(-1, cache.top_k)[active], flat_logits[active],
        )
        route = cache.router(features)
        if cache.router_mode == "v5":
            gate_logit_active = state_logit.reshape(-1)[active] + route[:, 0]
        else:
            cache_conf = features[:, 5].clamp(1e-4, 1.0 - 1e-4)
            param_conf = features[:, 8].clamp(1e-4, 1.0 - 1e-4)
            evidence = torch.logit(cache_conf) - torch.logit(param_conf)
            evidence = evidence + 1.25 * features[:, 6] - 0.50 * features[:, 7]
            evidence = evidence + 0.35 * features[:, 10] + 0.25 * features[:, 3]
            state_term = 0.0 if cache.router_mode == "confidence_nostate" else state_logit.reshape(-1)[active]
            gate_logit_active = state_term + route[:, 0] + cache.evidence_gain * evidence + cache.evidence_bias
        gate_flat[active] = torch.sigmoid(gate_logit_active).clamp(1e-5, 1.0 - 1e-5)
    mixed = (1.0 - gate_flat) * param_target + gate_flat * target_cache.reshape(-1)
    return -torch.log(mixed.clamp_min(1e-8)).view_as(targets)


def token_nll_v25(name: str, model: nn.Module,
                  x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if name in {TRANSFORMER, MAMBA2}:
        logits = model(x)
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none").view_as(y)
    states, logits = model.states_logits(x)
    if hasattr(model.cache, "token_nll"):
        return model.cache.token_nll(states, logits, x, y).float()
    return _fast_successor_token_nll(model.cache, states, logits, x, y).float()


def evaluation_preflight_v25(args, shape: v23.Shape, deps,
                            device: torch.device, root: Path) -> Dict[str, object]:
    """Exercise the exact validation path before any long training begins."""
    model = build_model_v25(FUSION, shape, args, deps, device).eval()
    length = max(17, int(args.selftest_tokens))
    x = (torch.arange(length, device=device)[None] * 37 + 11) % args.vocab_size
    y = (x + 1) % args.vocab_size
    with torch.no_grad(), v23.amp_ctx(device, args.amp):
        nll = token_nll_v25(FUSION, model, x, y)
    finite = bool(torch.isfinite(nll).all().item())
    result = {"shape": list(nll.shape), "expected_shape": list(y.shape), "finite": finite,
              "mean_nll": float(nll.float().mean().cpu()), "cache_class": model.cache.__class__.__name__,
              "has_native_token_nll": bool(hasattr(model.cache, "token_nll"))}
    v23.atomic_json(root / "evaluation_preflight.json", result)
    del model, x, y, nll
    v23.clear_cuda()
    if result["shape"] != result["expected_shape"] or not finite:
        raise RuntimeError(f"Fusion evaluation preflight failed: {result}")
    return result


def checkpoint_signature_v25(args, name: str,
                             shape: v23.Shape) -> Dict[str, object]:
    sig = _ORIG_CHECKPOINT_SIGNATURE(args, name, shape)
    sig["version"] = 25
    sig["arena_models"] = list(MODELS)
    if name == MAMBA2:
        sig.update({
            "mamba_version": MAMBA_VERSION,
            "causal_conv1d_version": CAUSAL_CONV1D_VERSION,
            "mamba_d_state": args.mamba_d_state,
            "mamba_d_conv": args.mamba_d_conv,
            "mamba_expand": args.mamba_expand,
            "mamba_headdim": args.mamba_headdim,
            "mamba_ngroups": args.mamba_ngroups,
            "mamba_chunk_size": args.mamba_chunk_size,
            "outer_checkpoint": False,
        })
    return sig


def initialization_audit_v25(args, shapes, deps, device: torch.device,
                             root: Path) -> Dict[str, object]:
    rows: Dict[str, object] = {}
    shared = []
    for name in MODELS:
        model = build_model_v25(name, shapes[name], args, deps, device)
        emb_hash = v23.tensor_hash(model.emb.weight)
        tied = model.lm_head.weight.data_ptr() == model.emb.weight.data_ptr()
        rows[name] = {
            "embedding_hash": emb_hash,
            "embedding_shape": list(model.emb.weight.shape),
            "head_tied": bool(tied),
        }
        if name in (FUSION, TRANSFORMER):
            shared.append(emb_hash)
        del model
        v23.clear_cuda()
    if len(set(shared)) != 1:
        raise AssertionError(f"Fusion/Transformer paired embedding mismatch: {rows}")
    if not all(bool(x["head_tied"]) for x in rows.values()):
        raise AssertionError(f"untied embedding detected: {rows}")
    out = {
        "models": rows,
        "fusion_transformer_shared_embedding_hash": shared[0],
        "mamba_embedding_is_dimension_specific": True,
    }
    v23.atomic_json(root / "initialization_audit.json", out)
    return out


def mamba_strict_preflight(args, shape: v23.Shape, deps,
                           device: torch.device, root: Path) -> Dict[str, object]:
    model = build_model_v25(MAMBA2, shape, args, deps, device).train()
    seq = int(args.mamba_preflight_seq)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.data_seed + 99173)
    x = torch.randint(0, args.vocab_size, (1, seq), generator=gen,
                      dtype=torch.long).to(device)
    y = torch.randint(0, args.vocab_size, (1, seq), generator=gen,
                      dtype=torch.long).to(device)
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    with v23.amp_ctx(device, args.amp):
        loss, _ = loss_call_v25(MAMBA2, model, x, y)
    loss.backward()
    finite = bool(torch.isfinite(loss).item()) and all(
        p.grad is None or bool(torch.isfinite(p.grad).all().item())
        for p in model.parameters()
    )
    result = {
        "loss": float(loss.detach().cpu()),
        "finite": finite,
        "peak_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "output_shape": [1, seq, args.vocab_size],
        "mamba_version": MAMBA_VERSION,
        "causal_conv1d_version": CAUSAL_CONV1D_VERSION,
        "outer_checkpoint": False,
    }
    v23.atomic_json(root / "mamba2_preflight.json", result)
    del model, x, y, loss
    v23.clear_cuda()
    if not finite:
        raise RuntimeError(f"Mamba-2 strict preflight failed: {result}")
    return result


def make_summary_v25(args, canonical_path: Path, tokenizer_path: Path,
                     shapes: Mapping[str, v23.Shape], corpora,
                     results: Mapping[str, v23.TrainResult], systems,
                     init_audit: Mapping[str, object]) -> str:
    # The v23 formatter already renders every item in the patched MODELS tuple.
    text = _ORIG_MAKE_SUMMARY(
        args, canonical_path, tokenizer_path, shapes, corpora,
        results, systems, init_audit,
    )
    text = text.replace(
        "FIELD-FUSION CANONICAL 300M v23 — WIKITEXT-103 100% / FIXED TOKEN BUDGET",
        "FIELD-FUSION CANONICAL 300M v25 — WIKITEXT-103 100% / THREE-WAY FIXED TOKEN ARENA",
    )
    f = results[FUSION]
    t = results[TRANSFORMER]
    m = results[MAMBA2]
    width = 210
    extra = [
        "",
        "MAMBA-2 COMPARATOR",
        f"backend=official mamba-ssm {MAMBA_VERSION} | causal-conv1d {CAUSAL_CONV1D_VERSION} | d_state={args.mamba_d_state} | d_conv={args.mamba_d_conv} | expand={args.mamba_expand} | headdim={args.mamba_headdim} | ngroups={args.mamba_ngroups} | chunk={args.mamba_chunk_size} | outer_checkpoint=off",
        f"Fusion vs Mamba-2: dNLL={f.final_test['nll']-m.final_test['nll']:+.5f} dPPL={100.0*(f.final_test['ppl']/m.final_test['ppl']-1.0):+.3f}%",
        f"Transformer vs Mamba-2: dNLL={t.final_test['nll']-m.final_test['nll']:+.5f} dPPL={100.0*(t.final_test['ppl']/m.final_test['ppl']-1.0):+.3f}%",
        "",
        "THREE-WAY SYSTEM RATIOS",
    ]

    def find(name: str, kind: str, context: int):
        return next((x for x in systems if x.model == name and x.kind == kind
                     and x.context == context and x.status == "ok"), None)

    for context in args.system_contexts:
        fr = find(FUSION, "train", int(context))
        tr = find(TRANSFORMER, "train", int(context))
        mr = find(MAMBA2, "train", int(context))
        if fr and tr and mr and fr.tokens_per_second and tr.tokens_per_second and mr.tokens_per_second:
            extra.append(
                f"train ctx={int(context):5d}: Fusion/Mamba2 speed={fr.tokens_per_second/mr.tokens_per_second:.3f}x peak={fr.peak_gib/mr.peak_gib:.3f}x | Transformer/Mamba2 speed={tr.tokens_per_second/mr.tokens_per_second:.3f}x peak={tr.peak_gib/mr.peak_gib:.3f}x"
            )
        fi = find(FUSION, "infer", int(context))
        ti = find(TRANSFORMER, "infer", int(context))
        mi = find(MAMBA2, "infer", int(context))
        if fi and ti and mi and fi.tokens_per_second and ti.tokens_per_second and mi.tokens_per_second:
            extra.append(
                f"infer ctx={int(context):5d}: Fusion/Mamba2 speed={fi.tokens_per_second/mi.tokens_per_second:.3f}x peak={fi.peak_gib/mi.peak_gib:.3f}x | Transformer/Mamba2 speed={ti.tokens_per_second/mi.tokens_per_second:.3f}x peak={ti.peak_gib/mi.peak_gib:.3f}x"
            )
    ranking = sorted(results.values(), key=lambda r: r.final_test["nll"])
    extra.extend([
        "",
        "FINAL TEST QUALITY RANKING: " + " < ".join(
            f"{r.model} ({r.final_test['nll']:.5f})" for r in ranking
        ),
        "=" * width,
    ])
    return text.rstrip() + "\n" + "\n".join(extra) + "\n"


def install_patches() -> None:
    v23.MODELS = MODELS
    v23.build_model = build_model_v25
    v23.loss_call = loss_call_v25
    v23.token_nll = token_nll_v25
    v23.set_distill = set_distill_v25
    v23.checkpoint_signature = checkpoint_signature_v25
    v23.make_summary = make_summary_v25


def add_mamba_defaults(args) -> None:
    # Fixed to the configuration family previously validated on the same H100.
    args.mamba_d_state = int(os.environ.get("MAMBA_D_STATE", "128"))
    args.mamba_d_conv = int(os.environ.get("MAMBA_D_CONV", "4"))
    args.mamba_expand = int(os.environ.get("MAMBA_EXPAND", "2"))
    args.mamba_headdim = int(os.environ.get("MAMBA_HEADDIM", "64"))
    args.mamba_ngroups = int(os.environ.get("MAMBA_NGROUPS", "1"))
    args.mamba_chunk_size = int(os.environ.get("MAMBA_CHUNK_SIZE", "256"))
    args.mamba_min_dim = int(os.environ.get("MAMBA_MIN_DIM", "768"))
    args.mamba_max_dim = int(os.environ.get("MAMBA_MAX_DIM", "1408"))
    args.mamba_min_layers = int(os.environ.get("MAMBA_MIN_LAYERS", "24"))
    args.mamba_max_layers = int(os.environ.get("MAMBA_MAX_LAYERS", "52"))
    args.mamba_preferred_layers = int(os.environ.get("MAMBA_PREFERRED_LAYERS", "36"))
    args.mamba_preflight_seq = int(os.environ.get("MAMBA_PREFLIGHT_SEQ", "256"))


def main() -> None:
    install_patches()
    args = v23.parse_args()
    add_mamba_defaults(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA H100 required")
    if OfficialMamba2 is None:
        raise RuntimeError(
            "Official Mamba-2 import failed. Run prepare_mamba2_env_v24.sh first. "
            + MAMBA_IMPORT_ERROR
        )
    if MAMBA_VERSION != REQUIRED_MAMBA_VERSION:
        raise RuntimeError(
            f"mamba-ssm version mismatch: required={REQUIRED_MAMBA_VERSION} "
            f"actual={MAMBA_VERSION}. Run prepare_mamba2_env_v24.sh."
        )
    if args.layers != 6 * len(args.refresh_windows):
        raise ValueError("layers must equal 6 * len(refresh_windows)")

    tokens_per_update = args.batch_size * args.accum * args.train_seq
    if tokens_per_update != 8192:
        v23.log(f"WARNING: tokens/update={tokens_per_update:,}, canonical protocol uses 8,192")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.set_float32_matmul_precision("high")

    arena = v23.base.import_module(v23.base.V15_PATH, "field_scale_50m_v15_for_v25")
    canonical_path = v23.base.locate_canonical(args.canonical_source)
    actual_sha = v23.sha256(canonical_path)
    if actual_sha != v23.EXPECTED_CANONICAL_SHA256:
        raise RuntimeError(
            f"canonical SHA mismatch expected={v23.EXPECTED_CANONICAL_SHA256} actual={actual_sha}"
        )
    v3 = arena.base.import_module(arena.base.V3_PATH, "v25_v3")
    bridge = arena.base.import_module(arena.base.BRIDGE_PATH, "v25_bridge")
    optmod = arena.base.import_module(arena.base.OPT_PATH, "v25_opt")
    epi = arena.base.import_module(arena.base.V9_PATH, "v25_epi")
    judge = arena.base.import_module(arena.base.JUDGE_PATH, "v25_judge")
    canonical = arena.base.import_module(canonical_path, "v25_canonical")
    optmod.v3_global = v3
    arena.base.install_fast_candidate_route(epi, optmod)
    changed = v23.core.patch_vocab(args.vocab_size, v23.HERE, canonical_path)
    v23.log(f"[vocab] patched VOCAB={args.vocab_size:,} in {len(changed)} modules")

    args.field_dim = args.dim
    args.field_layers = args.layers
    args.field_heads = args.heads
    args.field_ff_hidden = args.min_ff_hidden
    args.hybrid_ff_hidden = args.min_ff_hidden
    args.af_ff_hidden = args.min_ff_hidden
    args.tf_dim = args.dim
    args.tf_layers = args.layers
    args.tf_heads = args.heads
    args.tf_ff_hidden = args.min_ff_hidden
    args.conf_distill_ramp = args.distill_ramp

    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)

    # Fail fast before downloading, encoding, or moving the 100% corpus to GPU.
    deps = (arena, v3, canonical, bridge, optmod, epi, judge)
    shapes = solve_shapes_v25(args, deps)
    for name, shape in shapes.items():
        delta = 100.0 * (shape.params - args.target_params) / args.target_params
        v23.log(
            f"[shape] {name:32s} params={shape.params:,} dTarget={delta:+.3f}% "
            f"dim={shape.dim} layers={shape.layers} heads={shape.heads} ff={shape.ff_hidden}"
        )

    init_audit = initialization_audit_v25(args, shapes, deps, device, root)
    checkpoint_audit = v23.v22.checkpoint_exactness_audit(
        args, shapes[FUSION], deps, device, root
    )
    v23.log(f"[selftest] Fusion checkpoint exactness={checkpoint_audit}")
    evaluation_preflight = evaluation_preflight_v25(args, shapes[FUSION], deps, device, root)
    v23.log(f"[selftest] Fusion evaluation={evaluation_preflight}")
    mamba_preflight = mamba_strict_preflight(
        args, shapes[MAMBA2], deps, device, root
    )
    v23.log(f"[selftest] Mamba-2={mamba_preflight}")

    raw_rows = v23.core.load_raw_rows(args.cache_dir, args.data_frac)
    tokenizer = v23.base.copy_or_train_tokenizer(
        root, raw_rows[0], args.vocab_size,
        args.tokenizer_min_frequency, args.tokenizer_source,
    )
    train_c, val_c, test_c = v23.core.save_or_load_corpora(root, tokenizer, raw_rows)
    train = v23.core.place_tokens(train_c.tokens, device, args.data_device, "train")
    val = v23.core.place_tokens(val_c.tokens, device, args.data_device, "validation")
    test = v23.core.place_tokens(test_c.tokens, device, args.data_device, "test")
    corpora = {"train": train_c, "validation": val_c, "test": test_c}

    results: Dict[str, v23.TrainResult] = {}
    for name in MODELS:
        v23.log("=" * 180)
        v23.log(f"CANONICAL ARM: {name}")
        results[name] = v23.train_arm(
            name, shapes[name], args, deps, train, val_c, val,
            test_c, test, root, device,
        )
        v23.atomic_json(root / "train_results.json", {
            k: asdict(v) for k, v in results.items()
        })

    systems = []
    for context in args.system_contexts:
        batch = max(1, args.system_tokens_per_call // int(context))
        for name in MODELS:
            v23.log(f"[system/train] {name} ctx={context} batch={batch}")
            row = v23.benchmark_train(
                name, shapes[name], args, deps, train, int(context), batch,
                train_c.bytes_per_token, device,
            )
            systems.append(row)
            v23.log(asdict(row))
            v23.atomic_json(root / "system_rows.json", [asdict(x) for x in systems])
        for name in MODELS:
            v23.log(f"[system/infer] {name} ctx={context} batch={batch}")
            row = v23.benchmark_infer(
                name, shapes[name], args, deps, test, int(context), batch,
                test_c.bytes_per_token, device,
            )
            systems.append(row)
            v23.log(asdict(row))
            v23.atomic_json(root / "system_rows.json", [asdict(x) for x in systems])

    result = {
        "version": "25",
        "args": vars(args),
        "canonical_source": str(canonical_path),
        "canonical_sha256": actual_sha,
        "tokenizer": str(root / "tokenizer" / "tokenizer.json"),
        "tokenizer_sha256": v23.sha256(root / "tokenizer" / "tokenizer.json"),
        "shapes": {k: asdict(v) for k, v in shapes.items()},
        "initialization_audit": init_audit,
        "checkpoint_audit": checkpoint_audit,
        "evaluation_preflight": evaluation_preflight,
        "mamba2_preflight": mamba_preflight,
        "mamba_ssm": MAMBA_VERSION,
        "causal_conv1d": CAUSAL_CONV1D_VERSION,
        "train_results": {k: asdict(v) for k, v in results.items()},
        "system_rows": [asdict(x) for x in systems],
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    v23.atomic_json(root / "results.json", result)
    summary = make_summary_v25(
        args, canonical_path, root / "tokenizer" / "tokenizer.json",
        shapes, corpora, results, systems, init_audit,
    )
    v23.atomic_text(root / "summary.txt", summary)
    v23.log(summary)


if __name__ == "__main__":
    main()
