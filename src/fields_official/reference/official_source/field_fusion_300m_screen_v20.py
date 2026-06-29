#!/usr/bin/env python3
"""FIELD-FUSION 300M SCREEN v20.

A focused WikiText-103 5% token-level screen for three ~300M systems:

  * field_fusion_300m      20 canonical Field/PCAF blocks + 4 latent-GQA
                           refresh stations with causal chunk landmarks.
  * transformer_flash_300m strong 24-layer Flash-SDPA Transformer control.
  * field_pcaf_300m        24-layer strict attention-free Field/PCAF control.

The experiment is intentionally small in data and model count.  Its job is to
answer whether the new hybrid has the right *direction* at 300M before spending
on WikiText-103 100% or a larger corpus.

Fusion topology (24 residual/FFN blocks total):

    [Field x5 -> Refresh(w=256)]
    [Field x5 -> Refresh(w=512)]
    [Field x5 -> Refresh(w=1024)]
    [Field x5 -> Refresh(w=2048)]

Each refresh station uses:
  * full-width queries;
  * low-rank latent K/V projection;
  * grouped-query attention (many Q heads, few KV heads);
  * exact causal attention inside non-overlapping local windows;
  * one causal landmark per previous 256-token chunk;
  * a learned per-channel residual gate initialized conservatively;
  * SwiGLU FFN.

Both Field systems retain the validated chunk-parallel Triton recurrence,
FastSuccessorCacheV5/PCAF path, token-safe boundary mixer and selective episodic
corrective cache.  The canonical Field source is external and SHA-verified.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# The v19 module provides the already-validated token data/training/evaluation
# machinery and packs all recent Field/PCAF dependencies beside this file.
import field_token_efficiency_arena_v19 as base

core = base.core
Shape = core.Shape
Corpus = core.Corpus
LN2 = core.LN2

HERE = Path(__file__).resolve().parent
CANONICAL_NAME = base.CANONICAL_NAME
EXPECTED_CANONICAL_SHA256 = base.EXPECTED_CANONICAL_SHA256

FUSION = "field_fusion_300m"
TRANSFORMER = "transformer_flash_300m"
FIELD = "field_pcaf_300m"
MODEL_NAMES = (FUSION, TRANSFORMER, FIELD)
FIELD_NAMES = (FUSION, FIELD)


def log(x: object = "") -> None:
    print(str(x), flush=True)


def sha256(path: Path) -> str:
    return base.sha256(path)


def atomic_json(path: Path, obj: object) -> None:
    base.atomic_json(path, obj)


def atomic_text(path: Path, text: str) -> None:
    base.atomic_text(path, text)


def nparams(model: nn.Module) -> int:
    return core.nparams(model)


class NativeRMSNorm(nn.Module):
    """Fused/native RMSNorm that preserves the residual-stream dtype."""

    def __init__(self, dim: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.dim,), self.weight.to(dtype=x.dtype), self.eps)


class PackedSwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.gate_up = nn.Linear(dim, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


_ROPE_CACHE: Dict[Tuple[str, int, torch.dtype, int, int], Tuple[torch.Tensor, torch.Tensor]] = {}


def rope_cos_sin(device: torch.device, dtype: torch.dtype, length: int,
                  head_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (str(device), int(length), dtype, int(head_dim), 10000)
    cached = _ROPE_CACHE.get(key)
    if cached is not None:
        return cached
    if head_dim % 2:
        raise ValueError(f"RoPE head_dim must be even, got {head_dim}")
    inv = 1.0 / (10000.0 ** (
        torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
    ))
    pos = torch.arange(length, device=device, dtype=torch.float32)
    freq = torch.outer(pos, inv)
    cos = freq.cos().to(dtype)[None, None, :, :]
    sin = freq.sin().to(dtype)[None, None, :, :]
    _ROPE_CACHE[key] = (cos, sin)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # Half-split RoPE, matching the strong Transformer control.
    half = x.shape[-1] // 2
    a, b = x[..., :half], x[..., half:]
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)


def sdpa_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
             *, causal: bool, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Flash/efficient SDPA GQA when available, exact repeat fallback otherwise."""
    try:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal,
            dropout_p=0.0, enable_gqa=True
        )
    except TypeError:
        if q.shape[1] % k.shape[1]:
            raise ValueError("query heads must be divisible by KV heads")
        repeat = q.shape[1] // k.shape[1]
        return F.scaled_dot_product_attention(
            q, k.repeat_interleave(repeat, dim=1),
            v.repeat_interleave(repeat, dim=1),
            attn_mask=attn_mask, is_causal=causal, dropout_p=0.0,
        )


class LatentGQALandmarkAttention(nn.Module):
    """Linear-in-context practical approximation to Field-Fusion MLA.

    Recent exact detail is handled inside fixed local windows.  Older context is
    represented by one latent landmark per completed landmark chunk.  A query in
    chunk c may only read landmarks < c, so the branch is strictly causal.
    """

    def __init__(self, dim: int, q_heads: int, kv_heads: int, latent_dim: int,
                 local_window: int, landmark_chunk: int) -> None:
        super().__init__()
        if dim % q_heads:
            raise ValueError(f"dim={dim} must divide q_heads={q_heads}")
        if q_heads % kv_heads:
            raise ValueError("q_heads must be divisible by kv_heads")
        self.dim = int(dim)
        self.q_heads = int(q_heads)
        self.kv_heads = int(kv_heads)
        self.head_dim = dim // q_heads
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        self.latent_dim = int(latent_dim)
        self.local_window = int(local_window)
        self.landmark_chunk = int(landmark_chunk)

        self.q_proj = nn.Linear(dim, q_heads * self.head_dim, bias=False)
        self.kv_down = nn.Linear(dim, latent_dim, bias=False)
        self.k_up = nn.Linear(latent_dim, kv_heads * self.head_dim, bias=False)
        self.v_up = nn.Linear(latent_dim, kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.global_mix_logit = nn.Parameter(torch.tensor(-0.7))

    def _reshape_q(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.q_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reshape_kv(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.kv_heads, self.head_dim).transpose(1, 2).contiguous()

    def _local_attention(self, q: torch.Tensor, k: torch.Tensor,
                         v: torch.Tensor) -> torch.Tensor:
        """Vectorized independent causal windows in one SDPA launch."""
        batch, q_heads, length, head_dim = q.shape
        window = self.local_window
        pad = (-length) % window
        if pad:
            q = F.pad(q, (0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
        groups = q.shape[2] // window
        qg = q.view(batch, q_heads, groups, window, head_dim).permute(
            0, 2, 1, 3, 4
        ).reshape(batch * groups, q_heads, window, head_dim)
        kg = k.view(batch, self.kv_heads, groups, window, head_dim).permute(
            0, 2, 1, 3, 4
        ).reshape(batch * groups, self.kv_heads, window, head_dim)
        vg = v.view(batch, self.kv_heads, groups, window, head_dim).permute(
            0, 2, 1, 3, 4
        ).reshape(batch * groups, self.kv_heads, window, head_dim)
        yg = sdpa_gqa(qg, kg, vg, causal=True)
        y = yg.view(batch, groups, q_heads, window, head_dim).permute(
            0, 2, 1, 3, 4
        ).reshape(batch, q_heads, groups * window, head_dim)
        return y[:, :, :length]

    def _landmark_attention(self, q: torch.Tensor, latent: torch.Tensor,
                            cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, _, length, _ = q.shape
        chunk = self.landmark_chunk
        n_chunks = (length + chunk - 1) // chunk
        if n_chunks <= 1:
            return q.new_zeros((b, self.q_heads, length, self.head_dim))

        # Only completed chunks can become landmarks for later chunks.  Since
        # only the final sequence chunk may be incomplete, all landmarks read by
        # future chunks are averages of exactly `chunk` causal states.
        full_chunks = length // chunk
        if full_chunks == 0:
            return q.new_zeros((b, self.q_heads, length, self.head_dim))
        landmark_latent = latent[:, : full_chunks * chunk].reshape(
            b, full_chunks, chunk, self.latent_dim
        ).mean(dim=2)
        lk = self._reshape_kv(self.k_up(landmark_latent))
        lv = self._reshape_kv(self.v_up(landmark_latent))

        # Place each summary at the final position of its source chunk.
        positions = torch.arange(
            chunk - 1, full_chunks * chunk, chunk, device=q.device, dtype=torch.long
        )
        lk = apply_rope(lk, cos[:, :, positions], sin[:, :, positions])

        # One zero-valued null landmark is always visible.  Real landmark j is
        # visible only to tokens in chunks strictly after j.  This turns the
        # variable-prefix loop into one compact SDPA call with K=T/chunk.
        null_k = lk.new_zeros((b, self.kv_heads, 1, self.head_dim))
        null_v = lv.new_zeros((b, self.kv_heads, 1, self.head_dim))
        lk = torch.cat((null_k, lk), dim=2)
        lv = torch.cat((null_v, lv), dim=2)
        token_chunks = torch.arange(length, device=q.device) // chunk
        landmark_ids = torch.arange(full_chunks, device=q.device)
        allowed_real = landmark_ids[None, :] < token_chunks[:, None]
        allowed = torch.cat((
            torch.ones((length, 1), device=q.device, dtype=torch.bool),
            allowed_real,
        ), dim=1)
        return sdpa_gqa(
            q, lk, lv, causal=False, attn_mask=allowed[None, None]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, length, _ = x.shape
        q = self._reshape_q(self.q_proj(x))
        latent = self.kv_down(x)
        k = self._reshape_kv(self.k_up(latent))
        v = self._reshape_kv(self.v_up(latent))
        cos, sin = rope_cos_sin(x.device, x.dtype, length, self.head_dim)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        local = self._local_attention(q, k, v)
        global_summary = self._landmark_attention(q, latent, cos, sin)
        mix = torch.sigmoid(self.global_mix_logit).to(dtype=local.dtype)
        y = local + mix * global_summary
        y = y.transpose(1, 2).contiguous().view(b, length, self.dim)
        return self.out_proj(y)


class FusionRefreshBlock(nn.Module):
    def __init__(self, dim: int, q_heads: int, kv_heads: int, latent_dim: int,
                 local_window: int, landmark_chunk: int, ff_hidden: int) -> None:
        super().__init__()
        self.norm1 = NativeRMSNorm(dim)
        self.attn = LatentGQALandmarkAttention(
            dim, q_heads, kv_heads, latent_dim, local_window, landmark_chunk
        )
        self.residual_gate = nn.Linear(dim, dim, bias=True)
        nn.init.zeros_(self.residual_gate.weight)
        nn.init.constant_(self.residual_gate.bias, math.log(0.25 / 0.75))
        self.norm2 = NativeRMSNorm(dim)
        self.ff = PackedSwiGLU(dim, ff_hidden)
        self.window = int(local_window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm1(x)
        a = self.attn(z)
        gate = torch.sigmoid(self.residual_gate(z))
        x = x + gate * a
        return x + self.ff(self.norm2(x))


class StrongFlashBlock300M(nn.Module):
    def __init__(self, dim: int, heads: int, ff_hidden: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must divide heads")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = dim // heads
        if self.head_dim % 2:
            raise ValueError("head_dim must be even")
        self.norm1 = NativeRMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = NativeRMSNorm(dim)
        self.ff = PackedSwiGLU(dim, ff_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.qkv(self.norm1(x)).view(b, t, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        cos, sin = rope_cos_sin(x.device, x.dtype, t, self.head_dim)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(b, t, self.dim)
        x = x + self.proj(y)
        return x + self.ff(self.norm2(x))


class StrongFlashTransformer300M(nn.Module):
    def __init__(self, vocab: int, dim: int, heads: int, layers: int,
                 ff_hidden: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([
            StrongFlashBlock300M(dim, heads, ff_hidden) for _ in range(layers)
        ])
        self.final_norm = NativeRMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.emb(tokens)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))


class FieldTokenSystem300M(nn.Module):
    """Shared Field/PCAF token LM for the pure and fusion systems."""

    def __init__(self, *, fusion: bool, vocab: int, dim: int, layers: int,
                 ff_hidden: int, q_heads: int, kv_heads: int, latent_dim: int,
                 refresh_windows: Sequence[int], landmark_chunk: int,
                 field_chunk: int, triton_block_c: int, triton_chunk_t: int,
                 num_buckets: int, v3, canonical) -> None:
        super().__init__()
        if fusion:
            stages = len(refresh_windows)
            expected = stages * 6
            if layers != expected:
                raise ValueError(
                    f"fusion layers must equal 6*len(refresh_windows)={expected}, got {layers}"
                )
        self.fusion = bool(fusion)
        self.emb = nn.Embedding(vocab, dim)
        modules: List[nn.Module] = []
        if fusion:
            for window in refresh_windows:
                for _ in range(5):
                    modules.append(canonical.FieldBlock(
                        dim, "triton", field_chunk, triton_block_c,
                        triton_chunk_t, ff_hidden,
                    ))
                modules.append(FusionRefreshBlock(
                    dim, q_heads, kv_heads, latent_dim, int(window),
                    landmark_chunk, ff_hidden,
                ))
        else:
            for _ in range(layers):
                modules.append(canonical.FieldBlock(
                    dim, "triton", field_chunk, triton_block_c,
                    triton_chunk_t, ff_hidden,
                ))
        self.blocks = nn.ModuleList(modules)
        self.final_norm = v3.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.cache = v3.SuccessorCacheV5(
            dim, memory_dim=64, num_buckets=num_buckets, order=4, top_k=4,
            router_mode="confidence",
        )
        self.softpatch = v3.BoundaryStateMixer(dim, rank=64, learned=True)
        self.patch_position = max(0, layers // 2 - 1)
        self.locals = nn.ModuleDict()
        self.multiscales = nn.ModuleDict()
        self._patch_aux: Optional[torch.Tensor] = None

    @property
    def attention_free(self) -> bool:
        return not self.fusion

    def states_logits(self, tokens: torch.Tensor):
        x = self.emb(tokens)
        self._patch_aux = x.new_zeros(())
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == self.patch_position and self.softpatch is not None:
                x = self.softpatch(x, tokens)
                self._patch_aux = self.softpatch.last_aux
        return x, self.lm_head(self.final_norm(x))

    def loss_and_stats(self, tokens: torch.Tensor, targets: torch.Tensor,
                       compute_metrics: bool = False):
        states, logits = self.states_logits(tokens)
        loss, primary, stats = self.cache(
            states, logits, tokens, targets, compute_metrics
        )
        if self.training and self._patch_aux is not None:
            loss = loss + self._patch_aux
        return loss, primary, stats


def tied_embedding_init(model: nn.Module, seed: int, std: float = 0.02) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    weight = torch.empty(tuple(model.emb.weight.shape), dtype=torch.float32)
    weight.normal_(0.0, std, generator=generator)
    with torch.no_grad():
        model.emb.weight.copy_(weight.to(model.emb.weight.device, model.emb.weight.dtype))
    core.tie_embeddings(model, initialize=False)


def optimize_field_model(model: FieldTokenSystem300M, args, arena, v3,
                         optmod, device: torch.device) -> nn.Module:
    optmod.replace_softpatch(model, v3)
    optmod.replace_cache(model, v3, i32=True)
    model.cache.FEATURE_DIM = int(v3.SuccessorCacheV5.FEATURE_DIM)
    arena._install_cloud_fast_route(optmod)
    model.cache = arena.cloud.make_v10_cache(model.cache, args).to(device)
    return model


def build_model(name: str, shape: Shape, args, arena, v3, canonical, bridge,
                optmod, epi, judge, device: torch.device) -> nn.Module:
    del bridge, epi, judge
    core.seed_all(args.model_seed)
    if name == TRANSFORMER:
        model = StrongFlashTransformer300M(
            args.vocab_size, shape.dim, shape.heads, shape.layers, shape.ff_hidden
        ).to(device)
    elif name in FIELD_NAMES:
        model = FieldTokenSystem300M(
            fusion=(name == FUSION), vocab=args.vocab_size,
            dim=shape.dim, layers=shape.layers, ff_hidden=shape.ff_hidden,
            q_heads=args.fusion_q_heads, kv_heads=args.fusion_kv_heads,
            latent_dim=args.fusion_latent_dim,
            refresh_windows=args.refresh_windows,
            landmark_chunk=args.landmark_chunk,
            field_chunk=args.field_chunk,
            triton_block_c=args.triton_block_c,
            triton_chunk_t=args.triton_chunk_t,
            num_buckets=args.num_buckets,
            v3=v3, canonical=canonical,
        ).to(device)
        model = optimize_field_model(model, args, arena, v3, optmod, device)
    else:
        raise KeyError(name)
    tied_embedding_init(model, args.embedding_seed)
    return model


# Reuse all v18/v19 train and benchmark functions with the v20 builders.
core.TRANSFORMER = TRANSFORMER
core.MODEL_NAMES = MODEL_NAMES
core.build_model = build_model
base.TRANSFORMER = TRANSFORMER
base.MODEL_NAMES = MODEL_NAMES
base.FIELD_NAMES = FIELD_NAMES
base.build_model = build_model


def checkpoint_signature_v20(args, name: str, shape: Shape) -> Dict[str, object]:
    return {
        "arena_version": "v20",
        "name": name,
        "shape": asdict(shape),
        "vocab_size": args.vocab_size,
        "train_seq": args.train_seq,
        "batch": args.batch_size,
        "accum": args.accum,
        "train_steps": args.train_steps,
        "model_seed": args.model_seed,
        "embedding_seed": args.embedding_seed,
        "data_seed": args.data_seed,
        "refresh_windows": list(args.refresh_windows),
        "landmark_chunk": args.landmark_chunk,
        "fusion_q_heads": args.fusion_q_heads,
        "fusion_kv_heads": args.fusion_kv_heads,
        "fusion_latent_dim": args.fusion_latent_dim,
    }


core.checkpoint_signature = checkpoint_signature_v20


def make_shape(name: str, params: int, args, ff_hidden: int) -> Shape:
    return Shape(name, params, args.dim, args.layers, args.heads, ff_hidden)


def count_model(name: str, hidden: int, args, arena, v3, canonical, bridge,
                optmod, epi, judge) -> int:
    shape = Shape(name, 0, args.dim, args.layers, args.heads, hidden)
    model = build_model(
        name, shape, args, arena, v3, canonical, bridge, optmod, epi, judge,
        torch.device("cpu"),
    )
    value = nparams(model)
    del model
    gc.collect()
    return value


def solve_hidden(name: str, args, arena, v3, canonical, bridge, optmod,
                 epi, judge) -> Tuple[int, int]:
    # Every contestant has exactly `layers` SwiGLU blocks.  Each hidden unit
    # contributes 3*dim parameters per layer, so only one cheap count is needed.
    probe_hidden = 64
    probe_params = count_model(
        name, probe_hidden, args, arena, v3, canonical, bridge, optmod, epi, judge
    )
    slope = 3 * args.dim * args.layers
    intercept = probe_params - slope * probe_hidden
    raw = (args.target_params - intercept) / max(slope, 1)
    aligned = int(round(raw / args.ff_multiple) * args.ff_multiple)
    aligned = max(args.min_ff_hidden, min(args.max_ff_hidden, aligned))
    candidates = sorted(set(
        max(args.min_ff_hidden, min(args.max_ff_hidden,
            aligned + d * args.ff_multiple)) for d in range(-3, 4)
    ))
    rows = []
    for hidden in candidates:
        params = intercept + slope * hidden
        rows.append((hidden, int(params)))
    hidden, _ = min(rows, key=lambda hp: abs(hp[1] - args.target_params))
    # Construct the final system once to verify the analytical count and catch
    # any architecture-dependent non-linear parameter term.
    exact = count_model(
        name, hidden, args, arena, v3, canonical, bridge, optmod, epi, judge
    )
    return int(hidden), int(exact)


def resolve_shapes(args, arena, v3, canonical, bridge, optmod, epi, judge,
                   selected: Sequence[str]) -> Dict[str, Shape]:
    shapes: Dict[str, Shape] = {}
    for name in selected:
        hidden, params = solve_hidden(
            name, args, arena, v3, canonical, bridge, optmod, epi, judge
        )
        delta = 100.0 * (params - args.target_params) / args.target_params
        if abs(delta) > args.max_param_delta_pct:
            raise RuntimeError(
                f"parameter mismatch {name}: {params:,} ({delta:+.3f}%)"
            )
        shapes[name] = make_shape(name, params, args, hidden)
    return shapes


def tensor_hash(tensor: torch.Tensor) -> str:
    x = tensor.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str(tuple(x.shape)).encode())
    h.update(str(x.dtype).encode())
    h.update(x.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def architecture_audit(args, shapes, arena, v3, canonical, bridge, optmod,
                       epi, judge, device: torch.device, selected: Sequence[str],
                       root: Path) -> Dict[str, object]:
    log("[audit] architecture, tied embeddings, finite backward and causality")
    rows: Dict[str, object] = {}
    embedding_hashes = set()
    for name in selected:
        model = build_model(
            name, shapes[name], args, arena, v3, canonical, bridge, optmod,
            epi, judge, device,
        ).train()
        if not core.embeddings_tied(model):
            raise AssertionError(f"untied embedding/head: {name}")
        embedding_hashes.add(tensor_hash(model.emb.weight))
        x = torch.randint(0, args.vocab_size, (1, args.selftest_tokens), device=device)
        y = torch.randint(0, args.vocab_size, (1, args.selftest_tokens), device=device)
        core.set_distill(name, model, args.conf_distill_ramp, args)
        with core.amp_ctx(device, args.amp):
            loss, primary = core.training_loss(name, model, x, y)
        loss.backward()
        finite = bool(torch.isfinite(loss)) and all(
            p.grad is None or bool(torch.isfinite(p.grad).all())
            for p in model.parameters()
        )
        if not finite:
            raise AssertionError(f"non-finite backward: {name}")
        # Prefix causality: changing the suffix must not alter earlier NLLs.
        model.eval()
        cut = args.selftest_tokens // 2
        x2 = x.clone()
        x2[:, cut:] = torch.randint(
            0, args.vocab_size, x2[:, cut:].shape, device=device
        )
        with torch.no_grad(), core.amp_ctx(device, args.amp):
            n1 = core.token_nll(name, model, x, y)
            n2 = core.token_nll(name, model, x2, y)
        causal_max = float((n1[:, :cut] - n2[:, :cut]).abs().max().cpu())
        if causal_max > args.causal_tol:
            raise AssertionError(
                f"causality failed {name}: {causal_max} > {args.causal_tol}"
            )
        refresh_windows = []
        if name == FUSION:
            refresh_windows = [
                int(block.window) for block in model.blocks
                if isinstance(block, FusionRefreshBlock)
            ]
        rows[name] = {
            "params": nparams(model),
            "embedding_hash": tensor_hash(model.emb.weight),
            "loss": float(loss.detach().cpu()),
            "primary": float(primary.detach().cpu()),
            "causal_max": causal_max,
            "refresh_windows": refresh_windows,
            "attention_free": bool(getattr(model, "attention_free", False)),
        }
        log(
            f"[audit] {name:28s} params={nparams(model):,} "
            f"loss={float(loss):.5f} causal={causal_max:.3e} "
            f"refresh={refresh_windows}"
        )
        del model, x, x2, y, loss, primary, n1, n2
        gc.collect()
        torch.cuda.empty_cache()
    if len(embedding_hashes) != 1:
        raise AssertionError("paired embedding initialization mismatch")
    atomic_json(root / "architecture_audit.json", rows)
    log("[audit] PASS")
    return rows


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    vals = [float(v) for v in values]
    return statistics.fmean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0


def aggregate_results(per_seed: Mapping[int, Mapping[str, Mapping[str, object]]],
                      selected: Sequence[str], contexts: Sequence[int]) -> Dict[str, object]:
    out: Dict[str, object] = {"seeds": sorted(per_seed), "models": {}}
    for name in selected:
        rows = [per_seed[s][name] for s in sorted(per_seed)]
        nll_m, nll_s = mean_std([r["test"]["nll"] for r in rows])
        bpb_m, bpb_s = mean_std([r["test"]["bpb_norm"] for r in rows])
        tps_m, tps_s = mean_std([r["train_tokens_per_second"] for r in rows])
        peak_m, peak_s = mean_std([r["train_peak_gib"] for r in rows])
        mrow: Dict[str, object] = {
            "nll_mean": nll_m,
            "nll_std": nll_s,
            "ppl_from_mean_nll": math.exp(min(nll_m, 20.0)),
            "bits_per_token_mean": nll_m / LN2,
            "bpb_norm_mean": bpb_m,
            "bpb_norm_std": bpb_s,
            "train_tokens_per_second_mean": tps_m,
            "train_tokens_per_second_std": tps_s,
            "train_peak_gib_mean": peak_m,
            "train_peak_gib_std": peak_s,
            "matched_suffix": {},
            "individual": rows,
        }
        for ctx in contexts:
            vals = []
            for r in rows:
                idx = {int(x["context"]): x for x in r["matched_suffix"]}
                vals.append(float(idx[int(ctx)]["nll"]))
            mm, ss = mean_std(vals)
            mrow["matched_suffix"][str(ctx)] = {
                "nll_mean": mm, "nll_std": ss,
                "ppl_from_mean_nll": math.exp(min(mm, 20.0)),
            }
        out["models"][name] = mrow

    comparisons = []
    if FUSION in selected:
        for reference in (TRANSFORMER, FIELD):
            if reference not in selected:
                continue
            deltas = []
            wins = 0
            for seed in sorted(per_seed):
                a = per_seed[seed][FUSION]
                b = per_seed[seed][reference]
                dnll = float(a["test"]["nll"]) - float(b["test"]["nll"])
                wins += int(dnll < 0)
                deltas.append(dnll)
            md = statistics.fmean(deltas)
            comparisons.append({
                "candidate": FUSION,
                "reference": reference,
                "wins": wins,
                "seeds": len(deltas),
                "delta_nll_mean": md,
                "delta_bpb_norm_mean": (
                    out["models"][FUSION]["bpb_norm_mean"]
                    - out["models"][reference]["bpb_norm_mean"]
                ),
                "ppl_relative_pct": (math.exp(md) - 1.0) * 100.0,
            })
    out["comparisons"] = comparisons
    return out


def rows_by_context(rows: Sequence[Mapping[str, object]]) -> Dict[int, Dict[str, Mapping[str, object]]]:
    out: Dict[int, Dict[str, Mapping[str, object]]] = {}
    for row in rows:
        out.setdefault(int(row["context"]), {})[str(row["model"])] = row
    return out


def add_system_ratios(aggregate: Dict[str, object], systems, inference,
                      selected: Sequence[str]) -> None:
    aggregate["system_ratios"] = []
    aggregate["inference_ratios"] = []
    for kind, source, key in (
        ("train", systems, "system_ratios"),
        ("inference", inference, "inference_ratios"),
    ):
        by_ctx = rows_by_context(source)
        for ctx in sorted(by_ctx):
            rows = by_ctx[ctx]
            if FUSION not in rows or TRANSFORMER not in rows:
                continue
            f, t = rows[FUSION], rows[TRANSFORMER]
            if f.get("status") != "ok" or t.get("status") != "ok":
                continue
            aggregate[key].append({
                "kind": kind,
                "context": ctx,
                "speed_ratio_fusion_vs_transformer": (
                    float(f["tokens_per_second"]) / float(t["tokens_per_second"])
                ),
                "peak_ratio_fusion_vs_transformer": (
                    float(f["peak_gib"]) / float(t["peak_gib"])
                ),
            })


def fmt(v: Optional[float], pattern: str = ".4f") -> str:
    if v is None or not math.isfinite(float(v)):
        return "-"
    return format(float(v), pattern)


def make_summary(args, canonical_path: Path, tokenizer_path: Path,
                 shapes: Mapping[str, Shape], corpora: Mapping[str, Corpus],
                 aggregate: Mapping[str, object], systems, inference,
                 selected: Sequence[str]) -> str:
    width = 210
    lines = [
        "=" * width,
        "FIELD-FUSION 300M SCREEN v20 — WIKITEXT-103 5%",
        "=" * width,
        f"canonical_source={canonical_path} sha256={sha256(canonical_path)}",
        f"tokenizer={tokenizer_path} sha256={sha256(tokenizer_path)} vocab={args.vocab_size:,}",
        (
            f"protocol: WikiText-103 {args.data_frac:.1%} | train ctx={args.train_seq} | "
            f"steps={args.train_steps:,} | tokens/update={args.batch_size*args.accum*args.train_seq:,} | "
            f"seeds={list(args.model_seeds)} | BF16"
        ),
        (
            "fusion: [Field x5 -> latent-GQA refresh] x4 | windows="
            f"{list(args.refresh_windows)} | KV heads={args.fusion_kv_heads}/Q heads={args.fusion_q_heads} "
            f"| latent={args.fusion_latent_dim} | landmark chunk={args.landmark_chunk}"
        ),
        "PPL is exp(mean NLL). All models receive identical token windows per seed.",
        "",
        "TOKENIZED CORPORA",
    ]
    for name in ("train", "validation", "test"):
        c = corpora[name]
        lines.append(
            f"{name:12s} tokens={c.tokens.numel():12,d} raw_bytes={c.raw_bytes:12,d} "
            f"bytes/token={c.bytes_per_token:.4f}"
        )
    lines.extend([
        "", "MODEL SHAPES",
        f"{'model':30s} {'params':>14s} {'dTarget%':>10s} {'dim':>6s} {'layers':>7s} {'heads':>6s} {'ff':>7s}",
    ])
    for name in selected:
        s = shapes[name]
        delta = 100.0 * (s.params - args.target_params) / args.target_params
        lines.append(
            f"{name:30s} {s.params:14,d} {delta:+10.3f} {s.dim:6d} "
            f"{s.layers:7d} {s.heads:6d} {s.ff_hidden:7d}"
        )
    models = aggregate["models"]
    lines.extend([
        "", "FINAL TEST QUALITY",
        f"{'model':30s} {'PPL':>11s} {'NLL':>10s} {'NLL sd':>9s} {'bits/tok':>10s} {'BPB norm':>10s} {'train tok/s':>14s} {'peak GB':>9s}",
    ])
    for name in selected:
        r = models[name]
        lines.append(
            f"{name:30s} {r['ppl_from_mean_nll']:11.4f} {r['nll_mean']:10.5f} "
            f"{r['nll_std']:9.5f} {r['bits_per_token_mean']:10.5f} "
            f"{r['bpb_norm_mean']:10.5f} {r['train_tokens_per_second_mean']:14,.0f} "
            f"{r['train_peak_gib_mean']:9.2f}"
        )
    lines.extend(["", "FUSION QUALITY DELTAS"])
    for row in aggregate.get("comparisons", []):
        lines.append(
            f"vs {row['reference']:30s}: dNLL={row['delta_nll_mean']:+.5f} "
            f"dPPL={row['ppl_relative_pct']:+.3f}% wins={row['wins']}/{row['seeds']}"
        )
    lines.extend(["", "MATCHED-SUFFIX GENERALIZATION — SAME TARGET TOKENS"])
    lines.append(f"{'model':30s}" + "".join(f" {'PPL@'+str(c):>12s}" for c in args.matched_contexts))
    for name in selected:
        vals = models[name]["matched_suffix"]
        lines.append(
            f"{name:30s}" + "".join(
                f" {vals[str(c)]['ppl_from_mean_nll']:12.4f}" for c in args.matched_contexts
            )
        )
    lines.extend([
        "", "EQUAL NO-CHECKPOINT TRAINING SYSTEMS — REAL CORPUS",
        f"{'model':30s} {'ctx':>7s} {'batch':>6s} {'status':>8s} {'tok/s':>13s} {'step ms':>11s} {'peak GB':>9s}",
    ])
    for row in systems:
        lines.append(
            f"{row['model']:30s} {int(row['context']):7d} {int(row['batch']):6d} "
            f"{row['status']:>8s} {fmt(row.get('tokens_per_second'), ',.0f'):>13s} "
            f"{fmt(row.get('step_ms'), '.2f'):>11s} {fmt(row.get('peak_gib'), '.2f'):>9s}"
        )
    lines.extend([
        "", "FULL-PATH INFERENCE — REAL CORPUS, NO BACKWARD",
        f"{'model':30s} {'ctx':>7s} {'batch':>6s} {'status':>8s} {'tok/s':>13s} {'latency':>11s} {'peak GB':>9s}",
    ])
    for row in inference:
        lines.append(
            f"{row['model']:30s} {int(row['context']):7d} {int(row['batch']):6d} "
            f"{row['status']:>8s} {fmt(row.get('tokens_per_second'), ',.0f'):>13s} "
            f"{fmt(row.get('latency_ms'), '.2f'):>11s} {fmt(row.get('peak_gib'), '.2f'):>9s}"
        )
    lines.extend(["", "FUSION / TRANSFORMER SYSTEM RATIOS"])
    for row in aggregate.get("system_ratios", []):
        lines.append(
            f"train ctx={row['context']:5d}: speed={row['speed_ratio_fusion_vs_transformer']:.3f}x "
            f"peak={row['peak_ratio_fusion_vs_transformer']:.3f}x"
        )
    for row in aggregate.get("inference_ratios", []):
        lines.append(
            f"infer ctx={row['context']:5d}: speed={row['speed_ratio_fusion_vs_transformer']:.3f}x "
            f"peak={row['peak_ratio_fusion_vs_transformer']:.3f}x"
        )

    fnll = float(models[FUSION]["nll_mean"]) if FUSION in models else float("inf")
    tnll = float(models[TRANSFORMER]["nll_mean"]) if TRANSFORMER in models else float("inf")
    quality_delta = fnll - tnll
    train_long = {int(r["context"]): r for r in aggregate.get("system_ratios", [])}
    long_ctx = max(train_long) if train_long else None
    long_speed = train_long[long_ctx]["speed_ratio_fusion_vs_transformer"] if long_ctx else None
    long_peak = train_long[long_ctx]["peak_ratio_fusion_vs_transformer"] if long_ctx else None
    if quality_delta <= 0:
        quality_verdict = "QUALITY WIN: Fusion beat the Transformer on this screen."
    elif quality_delta <= args.promising_nll_gap:
        quality_verdict = (
            f"QUALITY PROMISING: Fusion is within {args.promising_nll_gap:.3f} NLL of the Transformer."
        )
    else:
        quality_verdict = "QUALITY MISS: Fusion remains materially behind the Transformer."
    if long_speed is None:
        system_verdict = "SYSTEMS UNKNOWN: no valid long-context comparison."
    elif long_speed >= 1.0 and long_peak <= 1.0:
        system_verdict = (
            f"SYSTEM WIN @{long_ctx}: speed={long_speed:.3f}x, peak={long_peak:.3f}x."
        )
    else:
        system_verdict = (
            f"SYSTEM MISS @{long_ctx}: speed={long_speed:.3f}x, peak={long_peak:.3f}x."
        )
    promote = quality_delta <= args.promising_nll_gap and (
        long_speed is not None and long_speed >= args.promote_long_speed
        and long_peak is not None and long_peak <= args.promote_long_peak
    )
    lines.extend([
        "", "AUTOMATIC SCREEN VERDICT", quality_verdict, system_verdict,
        (
            "PROMOTE TO LONGER 5% / THEN 100%" if promote else
            "DO NOT SCALE YET — REVISE THE FUSION MIXER OR SYSTEMS PATH"
        ),
        "This is a one-seed 5% screen by default; promotion is directional, not a final scientific claim.",
        "=" * width,
    ])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=("selftest", "train", "systems", "inference", "summary", "all"), default="all")
    p.add_argument("--outdir", default="/home/ubuntu/pcaf_runs/field_fusion_300m_v20")
    p.add_argument("--canonical-source", default="")
    p.add_argument("--cache-dir", default="/home/ubuntu/field_lab/hf_cache")
    p.add_argument("--tokenizer-source", default="")
    p.add_argument("--data-frac", type=float, default=0.05)
    p.add_argument("--data-device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--tokenizer-min-frequency", type=int, default=2)
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    p.add_argument("--target-params", type=int, default=300_000_000)
    p.add_argument("--max-param-delta-pct", type=float, default=0.75)

    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--layers", type=int, default=24)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--ff-multiple", type=int, default=64)
    p.add_argument("--min-ff-hidden", type=int, default=512)
    p.add_argument("--max-ff-hidden", type=int, default=8192)
    p.add_argument("--field-chunk", type=int, default=32)
    p.add_argument("--triton-block-c", type=int, default=16)
    p.add_argument("--triton-chunk-t", type=int, default=64)
    p.add_argument("--num-buckets", type=int, default=16384)
    p.add_argument("--local-chunk", type=int, default=1024)
    p.add_argument("--salience-floor", type=float, default=0.10)
    p.add_argument("--residual-limit", type=float, default=4.0)
    # Compatibility arguments consumed by the validated v10 cache constructor.
    p.add_argument("--span-tokens", type=int, default=2)
    p.add_argument("--address-dim", type=int, default=24)
    p.add_argument("--latent-top-k", type=int, default=4)
    p.add_argument("--score-limit", type=float, default=2.0)
    p.add_argument("--span-top-k", type=int, default=4)
    p.add_argument("--sidecar-max-mix", type=float, default=0.40)
    p.add_argument("--sidecar-gate-bias", type=float, default=1e-6)
    p.add_argument("--sidecar-aux-weight", type=float, default=0.01)
    p.add_argument("--gate-grad-scale", type=float, default=0.01)
    p.add_argument("--delta-heads", type=int, default=4)
    p.add_argument("--delta-block", type=int, default=16)
    p.add_argument("--phase-bands", type=int, default=8)
    p.add_argument("--phase-rank", type=int, default=16)

    p.add_argument("--fusion-q-heads", type=int, default=16)
    p.add_argument("--fusion-kv-heads", type=int, default=4)
    p.add_argument("--fusion-latent-dim", type=int, default=256)
    p.add_argument("--refresh-windows", type=int, nargs="+", default=[256, 512, 1024, 2048])
    p.add_argument("--landmark-chunk", type=int, default=256)

    p.add_argument("--train-seq", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--train-steps", type=int, default=1200)
    p.add_argument("--model-seeds", type=int, nargs="+", default=[1234])
    p.add_argument("--embedding-seed", type=int, default=314159)
    p.add_argument("--data-seed", type=int, default=5678)
    p.add_argument("--eval-seed", type=int, default=9012)

    p.add_argument("--amp", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--field-lr", type=float, default=3.0e-4)
    p.add_argument("--transformer-lr", type=float, default=3.0e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--min-lr-ratio", type=float, default=0.10)
    p.add_argument("--weight-decay", type=float, default=0.10)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--conf-distill-ramp", type=int, default=150)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=300)
    p.add_argument("--save-every", type=int, default=300)
    p.add_argument("--quick-eval-windows", type=int, default=6)
    p.add_argument("--test-token-budget", type=int, default=262144)
    p.add_argument("--matched-contexts", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    p.add_argument("--matched-score-tokens", type=int, default=128)
    p.add_argument("--matched-windows", type=int, default=6)

    p.add_argument("--system-contexts", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384])
    p.add_argument("--system-tokens-per-step", type=int, default=4096)
    p.add_argument("--system-warmup", type=int, default=1)
    p.add_argument("--system-steps", type=int, default=3)
    p.add_argument("--inference-contexts", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384])
    p.add_argument("--inference-tokens-per-call", type=int, default=4096)
    p.add_argument("--inference-warmup", type=int, default=1)
    p.add_argument("--inference-steps", type=int, default=3)

    p.add_argument("--selftest-tokens", type=int, default=65)
    p.add_argument("--causal-tol", type=float, default=0.005)
    p.add_argument("--promising-nll-gap", type=float, default=0.020)
    p.add_argument("--promote-long-speed", type=float, default=1.0)
    p.add_argument("--promote-long-peak", type=float, default=1.0)
    p.add_argument("--prune-completed-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def replicate_args(args, replicate_index: int, model_seed: int) -> argparse.Namespace:
    out = argparse.Namespace(**vars(args))
    out.model_seed = int(model_seed)
    out.embedding_seed = int(args.embedding_seed + replicate_index * 10_007)
    out.data_seed = int(args.data_seed + replicate_index * 1_000_003)
    # Compatibility widths used only by imported helper constructors.
    out.field_dim = args.dim
    out.field_layers = args.layers
    out.field_heads = args.heads
    out.field_ff_hidden = args.min_ff_hidden
    out.hybrid_ff_hidden = args.min_ff_hidden
    out.af_ff_hidden = args.min_ff_hidden
    out.tf_dim = args.dim
    out.tf_layers = args.layers
    out.tf_heads = args.heads
    out.tf_ff_hidden = args.min_ff_hidden
    return out


def main() -> None:
    args = parse_args()
    selected = tuple(name for name in MODEL_NAMES if name in set(args.models))
    if not selected:
        raise ValueError("no models selected")
    if len(args.refresh_windows) * 6 != args.layers:
        raise ValueError("layers must equal 6 * number of refresh windows")
    if args.dim % args.fusion_q_heads:
        raise ValueError("dim must divide fusion_q_heads")
    if args.fusion_q_heads % args.fusion_kv_heads:
        raise ValueError("fusion_q_heads must divide fusion_kv_heads")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/H100 required")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.set_float32_matmul_precision("high")

    arena = base.import_module(base.V15_PATH, "field_scale_50m_v15_for_v20")
    canonical_path = base.locate_canonical(args.canonical_source)
    actual_sha = sha256(canonical_path)
    if actual_sha != EXPECTED_CANONICAL_SHA256:
        raise RuntimeError(
            f"canonical SHA mismatch expected={EXPECTED_CANONICAL_SHA256} actual={actual_sha}"
        )
    v3 = arena.base.import_module(arena.base.V3_PATH, "v20_v3")
    bridge = arena.base.import_module(arena.base.BRIDGE_PATH, "v20_bridge")
    optmod = arena.base.import_module(arena.base.OPT_PATH, "v20_opt")
    epi = arena.base.import_module(arena.base.V9_PATH, "v20_epi")
    judge = arena.base.import_module(arena.base.JUDGE_PATH, "v20_judge")
    canonical = arena.base.import_module(canonical_path, "v20_canonical")
    optmod.v3_global = v3
    arena.base.install_fast_candidate_route(epi, optmod)
    changed = core.patch_vocab(args.vocab_size, HERE, canonical_path)
    log(f"[vocab] patched VOCAB={args.vocab_size:,} in {len(changed)} modules")

    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)
    raw_rows = core.load_raw_rows(args.cache_dir, args.data_frac)
    tokenizer = base.copy_or_train_tokenizer(
        root, raw_rows[0], args.vocab_size, args.tokenizer_min_frequency,
        args.tokenizer_source,
    )
    train_c, val_c, test_c = core.save_or_load_corpora(root, tokenizer, raw_rows)
    corpora = {"train": train_c, "validation": val_c, "test": test_c}
    train = core.place_tokens(train_c.tokens, device, args.data_device, "train")
    val = core.place_tokens(val_c.tokens, device, args.data_device, "validation")
    test = core.place_tokens(test_c.tokens, device, args.data_device, "test")

    first_args = replicate_args(args, 0, args.model_seeds[0])
    shapes = resolve_shapes(
        first_args, arena, v3, canonical, bridge, optmod, epi, judge, selected
    )
    config = {
        "args": vars(args),
        "selected_models": list(selected),
        "canonical_source": str(canonical_path),
        "canonical_sha256": actual_sha,
        "tokenizer": str(root / "tokenizer" / "tokenizer.json"),
        "tokenizer_sha256": sha256(root / "tokenizer" / "tokenizer.json"),
        "shapes": {name: asdict(shapes[name]) for name in selected},
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    atomic_json(root / "config.json", config)

    log("=" * 180)
    log("FIELD-FUSION 300M SCREEN v20 — WIKITEXT-103 5%")
    log(f"gpu={torch.cuda.get_device_name(0)} torch={torch.__version__} cuda={torch.version.cuda}")
    log(f"canonical={canonical_path} sha={actual_sha}")
    log(f"models={selected} seeds={args.model_seeds}")
    log(
        f"fusion topology=[Field x5 -> Refresh] x4 windows={args.refresh_windows} "
        f"q_heads={args.fusion_q_heads} kv_heads={args.fusion_kv_heads} "
        f"latent={args.fusion_latent_dim} landmark={args.landmark_chunk}"
    )
    for name in selected:
        s = shapes[name]
        delta = 100.0 * (s.params - args.target_params) / args.target_params
        log(f"{name:30s} params={s.params:,} dTarget={delta:+.3f}% ff={s.ff_hidden}")
    log("=" * 180)

    if args.mode in ("selftest", "all"):
        architecture_audit(
            first_args, shapes, arena, v3, canonical, bridge, optmod, epi,
            judge, device, selected, root,
        )
        if args.mode == "selftest":
            return

    per_seed_path = root / "per_seed_results.json"
    aggregate_path = root / "aggregate_results.json"
    per_seed: Dict[int, Dict[str, Dict[str, object]]] = {}
    if args.mode in ("train", "all"):
        for rep_index, seed in enumerate(args.model_seeds):
            run_args = replicate_args(args, rep_index, seed)
            rep_root = root / "replicates" / f"seed_{seed}"
            rep_root.mkdir(parents=True, exist_ok=True)
            per_seed[int(seed)] = {}
            log("-" * 180)
            log(
                f"[replicate] seed={seed} embedding_seed={run_args.embedding_seed} "
                f"data_seed={run_args.data_seed}"
            )
            for name in selected:
                per_seed[int(seed)][name] = core.train_one(
                    name, shapes[name], run_args, arena, v3, canonical, bridge,
                    optmod, epi, judge, train, val, test_c, test, rep_root, device,
                )
                atomic_json(per_seed_path, per_seed)
                if args.prune_completed_checkpoints:
                    ckpt = rep_root / "models" / name / "latest.pt"
                    if ckpt.is_file():
                        ckpt.unlink()
                        log(f"[{name}] pruned completed checkpoint")
        aggregate = aggregate_results(per_seed, selected, args.matched_contexts)
        atomic_json(aggregate_path, aggregate)
        if args.mode == "train":
            return
    else:
        if not per_seed_path.is_file():
            raise FileNotFoundError(per_seed_path)
        raw = json.loads(per_seed_path.read_text(encoding="utf-8"))
        per_seed = {int(k): v for k, v in raw.items()}
        aggregate = aggregate_results(per_seed, selected, args.matched_contexts)

    systems_path = root / "systems.json"
    if args.mode in ("systems", "all"):
        systems = []
        for context in args.system_contexts:
            for name in selected:
                row = base.training_systems_benchmark(
                    name, shapes[name], first_args, arena, v3, canonical, bridge,
                    optmod, epi, judge, int(context), train_c.bytes_per_token,
                    train, device,
                )
                systems.append(row)
                atomic_json(systems_path, systems)
                log(
                    f"[systems] {name:30s} ctx={context:5d} status={row['status']} "
                    f"tok/s={row['tokens_per_second']} peak={row['peak_gib']}"
                )
        if args.mode == "systems":
            return
    else:
        systems = json.loads(systems_path.read_text(encoding="utf-8"))

    inference_path = root / "inference.json"
    if args.mode in ("inference", "all"):
        inference = []
        for context in args.inference_contexts:
            for name in selected:
                row = base.inference_benchmark(
                    name, shapes[name], first_args, arena, v3, canonical, bridge,
                    optmod, epi, judge, int(context), test, device,
                )
                inference.append(row)
                atomic_json(inference_path, inference)
                log(
                    f"[inference] {name:30s} ctx={context:5d} status={row['status']} "
                    f"tok/s={row['tokens_per_second']} peak={row['peak_gib']}"
                )
        if args.mode == "inference":
            return
    else:
        inference = json.loads(inference_path.read_text(encoding="utf-8"))

    add_system_ratios(aggregate, systems, inference, selected)
    atomic_json(aggregate_path, aggregate)
    summary = make_summary(
        args, canonical_path, root / "tokenizer" / "tokenizer.json", shapes,
        corpora, aggregate, systems, inference, selected,
    )
    atomic_text(root / "summary.txt", summary)
    log(summary)


if __name__ == "__main__":
    main()
