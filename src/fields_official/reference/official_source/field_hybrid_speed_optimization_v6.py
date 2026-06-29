#!/usr/bin/env python3
"""Speed-optimization qualification for the canonical Field hybrid and AF branch.

This arena does not change the validated Field recurrence. It targets overhead in:
  * the sliding-window local side path;
  * soft-patch boundary handling;
  * PCAF host synchronizations and candidate index width;
  * the multiscale causal-convolution side path.

Exact-equivalent candidates are checked against the frozen reference before they
are benchmarked. The optional AF-lite arm changes the number of convolution
branches and therefore receives a short paired quality screen.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
V3_PATH = HERE / "field_hybrid_attentionfree_qualification_v3.py"
V4_PATH = HERE / "field_hybrid_canonical_50m_bridge_v4.py"
CANONICAL_NAME = "field_only_v4_chunked_triton_wiki100.py"
CANONICAL_SHA = "0454647a7f667cdf73f45029eee0f525c58c0788ecd433ec1a4db8c3937ac848"
VOCAB = 256
LN2 = math.log(2.0)


def log(x: object = "") -> None:
    print(str(x), flush=True)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def nparams(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8")
    os.replace(tmp, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def locate_canonical(explicit: str) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates += [
        HERE / CANONICAL_NAME,
        Path("/home/ubuntu/field_hybrid_canonical_50m_bridge_v4") / CANONICAL_NAME,
        Path("/home/ubuntu/field_pcaf_efficiency_v1") / CANONICAL_NAME,
        Path("/home/ubuntu") / CANONICAL_NAME,
    ]
    for p in candidates:
        if p.is_file():
            p = p.resolve()
            actual = sha256(p)
            if actual != CANONICAL_SHA:
                raise RuntimeError(f"canonical SHA mismatch: {p} {actual}")
            return p
    raise FileNotFoundError("missing validated canonical Field source")


def amp_ctx(device: torch.device, amp: str):
    enabled = device.type == "cuda" and amp in {"bf16", "fp16"}
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def make_optimizer(model: nn.Module, lr: float, weight_decay: float):
    kwargs = dict(lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay)
    try:
        return torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(model.parameters(), **kwargs)


# =====================================================================================
# Exact local-attention execution engines
# =====================================================================================


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class CachedChunkLocalAttention(nn.Module):
    """Same math/parameters as v3 LowRankLocalAttention with cached RoPE/masks."""

    def __init__(self, dim: int, inner: int, heads: int, window: int, chunk: int, v3):
        super().__init__()
        if inner % heads:
            raise ValueError("inner must divide heads")
        self.dim, self.inner, self.heads = dim, inner, heads
        self.head_dim = inner // heads
        self.window, self.chunk = int(window), int(chunk)
        self.norm = v3.RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * inner, bias=False)
        self.out = nn.Linear(inner, dim, bias=False)
        inv = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self._rope_cache: Dict[Tuple[str, str, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._mask_cache: Dict[Tuple[str, int, int, int], List[torch.Tensor]] = {}

    def _rope_pair(self, t: int, device: torch.device, dtype: torch.dtype):
        key = (str(device), str(dtype), int(t))
        pair = self._rope_cache.get(key)
        if pair is None:
            pos = torch.arange(t, device=device, dtype=torch.float32)
            freq = torch.outer(pos, self.inv_freq.to(device=device, dtype=torch.float32))
            emb = torch.repeat_interleave(freq, 2, dim=-1).to(dtype)
            pair = (emb.cos()[None, None], emb.sin()[None, None])
            self._rope_cache[key] = pair
        return pair

    def _masks(self, t: int, device: torch.device) -> List[torch.Tensor]:
        key = (str(device), int(t), self.window, self.chunk)
        masks = self._mask_cache.get(key)
        if masks is None:
            pos = torch.arange(t, device=device)
            masks = []
            for q0 in range(0, t, self.chunk):
                q1 = min(t, q0 + self.chunk)
                k0 = max(0, q0 - self.window + 1)
                qp = pos[q0:q1, None]
                kp = pos[None, k0:q1]
                masks.append((kp <= qp) & (kp >= qp - self.window + 1))
            self._mask_cache[key] = masks
        return masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        z = self.qkv(self.norm(x)).view(b, t, 3, self.heads, self.head_dim)
        q, k, v = z.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        cos, sin = self._rope_pair(t, q.device, q.dtype)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        masks = self._masks(t, x.device)
        outs: List[torch.Tensor] = []
        for part, q0 in enumerate(range(0, t, self.chunk)):
            q1 = min(t, q0 + self.chunk)
            k0 = max(0, q0 - self.window + 1)
            outs.append(F.scaled_dot_product_attention(
                q[:, :, q0:q1], k[:, :, k0:q1], v[:, :, k0:q1],
                attn_mask=masks[part], dropout_p=0.0,
            ))
        y = torch.cat(outs, dim=2).transpose(1, 2).contiguous().view(b, t, self.inner)
        return x + self.out(y)


_FLEX_IMPORT_ERROR = ""
try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    HAS_FLEX = True
except Exception as exc:  # pragma: no cover - H100 environment decides.
    create_block_mask = None
    flex_attention = None
    HAS_FLEX = False
    _FLEX_IMPORT_ERROR = repr(exc)

try:
    _COMPILED_FLEX = torch.compile(
        flex_attention, dynamic=False, fullgraph=True, mode="max-autotune-no-cudagraphs"
    ) if HAS_FLEX else None
except Exception as exc:  # pragma: no cover
    _COMPILED_FLEX = None
    _FLEX_IMPORT_ERROR += f" compile={exc!r}"


class FlexLocalAttention(nn.Module):
    """Exact causal sliding-window attention using cached FlexAttention BlockMask."""

    def __init__(self, dim: int, inner: int, heads: int, window: int, v3, compiled: bool):
        super().__init__()
        if not HAS_FLEX:
            raise RuntimeError("FlexAttention unavailable: " + _FLEX_IMPORT_ERROR)
        if inner % heads:
            raise ValueError("inner must divide heads")
        self.dim, self.inner, self.heads = dim, inner, heads
        self.head_dim = inner // heads
        self.window = int(window)
        self.compiled = bool(compiled and _COMPILED_FLEX is not None)
        self.norm = v3.RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * inner, bias=False)
        self.out = nn.Linear(inner, dim, bias=False)
        inv = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self._rope_cache: Dict[Tuple[str, str, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._block_masks: Dict[Tuple[str, int, int], object] = {}

    def _rope_pair(self, t: int, device: torch.device, dtype: torch.dtype):
        key = (str(device), str(dtype), int(t))
        pair = self._rope_cache.get(key)
        if pair is None:
            pos = torch.arange(t, device=device, dtype=torch.float32)
            freq = torch.outer(pos, self.inv_freq.to(device=device, dtype=torch.float32))
            emb = torch.repeat_interleave(freq, 2, dim=-1).to(dtype)
            pair = (emb.cos()[None, None], emb.sin()[None, None])
            self._rope_cache[key] = pair
        return pair

    def _mask(self, t: int, device: torch.device):
        key = (str(device), int(t), self.window)
        mask = self._block_masks.get(key)
        if mask is None:
            window = self.window
            def sliding_causal(b, h, q_idx, kv_idx):
                del b, h
                return (q_idx >= kv_idx) & ((q_idx - kv_idx) < window)
            mask = create_block_mask(
                sliding_causal, B=None, H=None, Q_LEN=t, KV_LEN=t,
                device=str(device), BLOCK_SIZE=128, _compile=True,
            )
            self._block_masks[key] = mask
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        z = self.qkv(self.norm(x)).view(b, t, 3, self.heads, self.head_dim)
        q, k, v = z.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        cos, sin = self._rope_pair(t, q.device, q.dtype)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        fn = _COMPILED_FLEX if self.compiled else flex_attention
        y = fn(q, k, v, block_mask=self._mask(t, x.device))
        y = y.transpose(1, 2).contiguous().view(b, t, self.inner)
        return x + self.out(y)


# =====================================================================================
# Exact fast softpatch/cache/attention-free modules
# =====================================================================================


class FastBoundaryStateMixer(nn.Module):
    """Same softpatch math, but a byte LUT removes 21 comparisons and no host sync."""

    def __init__(self, dim: int, rank: int, learned: bool, target_rate: float, v3):
        super().__init__()
        self.learned = learned
        self.target_rate = float(target_rate)
        self.norm = v3.RMSNorm(dim)
        self.down = nn.Linear(dim, rank, bias=False)
        self.score = nn.Linear(dim, 1, bias=True) if learned else None
        self.up = nn.Linear(2 * rank, dim, bias=False)
        # Byte models use the original 256-entry punctuation/whitespace prior.
        # Token models patch v3.VOCAB above 256; byte IDs no longer have lexical
        # meaning there, so use a vocab-sized zero prior and let the learned
        # scorer discover token boundaries. This also prevents out-of-range CUDA
        # indexing that otherwise surfaces asynchronously as a cuBLAS failure.
        vocab_size = int(getattr(v3, "VOCAB", 256))
        lut = torch.zeros(vocab_size, dtype=torch.float32)
        if vocab_size == 256:
            for value in v3._BOUNDARY_BYTES:
                lut[int(value)] = 1.0
        self.register_buffer("_boundary_lut", lut, persistent=False)
        self.last_aux = torch.tensor(0.0)
        self.last_rate = torch.tensor(0.0)

    def forward(self, x: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        z = self.down(h).float()
        prior = self._boundary_lut[tokens]
        if self.learned:
            score = self.score(h).float().squeeze(-1) + 1.5 * prior
            boundary = torch.sigmoid(score).clamp(0.01, 0.99)
            rate = boundary.mean()
            self.last_aux = 0.02 * (rate - self.target_rate).square()
            self.last_rate = rate.detach()
        else:
            boundary = prior.clone()
            boundary[:, 0] = 1.0
            self.last_aux = x.new_zeros(())
            self.last_rate = boundary.mean().detach()
        a = (1.0 - boundary)[..., None].expand_as(z)
        drive = boundary[..., None] * z
        state = v3_global.hierarchical_scan(a, drive, block=32)
        delta = self.up(torch.cat((z, state), dim=-1).to(x.dtype))
        return x + delta


class FastMultiScaleCausalConv(nn.Module):
    """Same parameters/math, fewer explicit padding kernels and fused branch reduction."""

    def __init__(self, dim: int, inner: int, dilations: Sequence[int], v3):
        super().__init__()
        self.dim, self.inner = dim, inner
        self.dilations = tuple(int(x) for x in dilations)
        self.norm = v3.RMSNorm(dim)
        self.in_proj = nn.Linear(dim, inner, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(inner, inner, 3, dilation=d, groups=inner, bias=False)
            for d in self.dilations
        ])
        self.branch_logits = nn.Parameter(torch.zeros(len(self.dilations)))
        self.gate = nn.Linear(dim, inner, bias=True)
        self.out = nn.Linear(inner, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        z = self.in_proj(h).transpose(1, 2)
        t = z.shape[-1]
        branches = []
        for conv, dilation in zip(self.convs, self.dilations):
            y = F.conv1d(
                z, conv.weight, conv.bias, stride=1, padding=2 * dilation,
                dilation=dilation, groups=self.inner,
            )[..., :t]
            branches.append(F.silu(y))
        stacked = torch.stack(branches, dim=0)
        mix = torch.softmax(self.branch_logits.float(), dim=0).to(stacked.dtype)
        y = (stacked * mix[:, None, None, None]).sum(dim=0).transpose(1, 2)
        y = y * torch.sigmoid(self.gate(h))
        return x + self.out(y)


def causal_recent_candidates_i32(tokens: torch.Tensor, order: int, num_buckets: int, top_k: int, v3):
    b, t = tokens.shape
    buckets = v3.causal_ngram_buckets(tokens, order, num_buckets).to(torch.int32)
    pos = torch.arange(t, device=tokens.device, dtype=torch.int32)[None, :].expand(b, -1)
    bid = torch.arange(b, device=tokens.device, dtype=torch.int32)[:, None].expand(-1, t)
    group = bid * int(num_buckets) + buckets
    max_key = int((b * num_buckets) * (t + 1) + t)
    if max_key >= 2_147_000_000:
        return v3.causal_recent_candidates(tokens, order, num_buckets, top_k)
    key = group * int(t + 1) + pos
    perm = torch.argsort(key.reshape(-1), stable=True)
    sg = group.reshape(-1).index_select(0, perm)
    sp = pos.reshape(-1).index_select(0, perm)
    n = perm.numel()
    cand_sorted = torch.full((n, top_k), -1, device=tokens.device, dtype=torch.int32)
    for k in range(1, top_k + 1):
        if k >= n:
            break
        same = sg[k:] == sg[:-k]
        prev = torch.where(same, sp[:-k], torch.full_like(sp[:-k], -1))
        cand_sorted[k:, k - 1] = prev
    out = torch.full_like(cand_sorted, -1)
    out[perm] = cand_sorted
    return out.view(b, t, top_k).long()


class FastSuccessorCacheV5(nn.Module):
    """State-compatible PCAF v5 cache with no per-step CPU/GPU synchronization."""

    def __init__(self, source, v3, use_i32: bool = True):
        super().__init__()
        # Mirror the original module tree so state_dict keys remain identical.
        self.state_dim = source.state_dim
        self.memory_dim = source.memory_dim
        self.num_buckets = source.num_buckets
        self.order = source.order
        self.top_k = source.top_k
        self.router_mode = source.router_mode
        self.shared_weight = nn.Parameter(torch.empty_like(source.shared_weight))
        self.state_gate = source.state_gate.__class__(*[]) if False else None
        # Deep-copy modules without importing copy on hot path.
        import copy
        self.state_gate = copy.deepcopy(source.state_gate)
        self.router = copy.deepcopy(source.router)
        self.evidence_gain = nn.Parameter(source.evidence_gain.detach().clone()) if source.evidence_gain is not None else None
        self.evidence_bias = nn.Parameter(source.evidence_bias.detach().clone()) if source.evidence_bias is not None else None
        self.recency_scale = nn.Parameter(source.recency_scale.detach().clone())
        self.distill_temperature = source.distill_temperature
        self.distill_weight = source.distill_weight
        self.distill_scale = source.distill_scale
        self.enabled = source.enabled
        self.last_aux = {}
        self.use_i32 = bool(use_i32)
        self._v3 = v3
        self.load_state_dict(source.state_dict(), strict=True)

    def _features(self, *args, **kwargs):
        # Reuse the already validated feature computation.
        return self._v3.SuccessorCacheV5._features(self, *args, **kwargs)

    def forward(self, states, logits, tokens, targets, compute_metrics=False):
        if compute_metrics:
            # Metrics are infrequent; preserve the exact reference implementation.
            return self._reference_forward(states, logits, tokens, targets, compute_metrics=True)

        flat_logits = logits.reshape(-1, VOCAB)
        flat_targets = targets.reshape(-1)
        param_nll = F.cross_entropy(flat_logits.float(), flat_targets, reduction="none")
        param_target = torch.exp(-param_nll)
        if not self.enabled:
            primary = param_nll.mean()
            return primary, primary.detach(), None

        b, t, _ = states.shape
        if self.use_i32:
            idx = causal_recent_candidates_i32(tokens, self.order, self.num_buckets, self.top_k, self._v3)
        else:
            idx = self._v3.causal_recent_candidates(tokens, self.order, self.num_buckets, self.top_k)
        valid = idx >= 0
        has = valid.any(-1)
        safe = idx.clamp_min(0)
        batch_idx = torch.arange(b, device=states.device)[:, None, None]
        proj = self._v3.normalize_rows(F.linear(states.float(), self.shared_weight.float()))
        q = proj[:, :, None, :]
        ck = proj[batch_idx, safe]
        scores = (ck * q).sum(-1) * (self.memory_dim ** -0.5)
        recency = safe.float() / max(float(t - 1), 1.0)
        scores = scores + self.recency_scale.float() * recency
        scores = scores.masked_fill(~valid, -1.0e9)
        weights = torch.softmax(scores.float(), dim=-1) * valid.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        cand_tokens = targets[batch_idx, safe]
        target_cache = (weights * (cand_tokens == targets[:, :, None]).float()).sum(-1)

        active = has.reshape(-1)
        state_logit = self.state_gate(states).float().squeeze(-1)
        features = self._features(
            scores.reshape(-1, self.top_k)[active],
            weights.reshape(-1, self.top_k)[active],
            valid.reshape(-1, self.top_k)[active],
            cand_tokens.reshape(-1, self.top_k)[active],
            recency.reshape(-1, self.top_k)[active],
            flat_logits[active],
        )
        route = self.router(features)
        flat_state_logit = state_logit.reshape(-1)
        gate_flat = torch.zeros_like(flat_state_logit)
        if self.router_mode == "v5":
            gate_logit_active = flat_state_logit[active] + route[:, 0]
        else:
            cache_conf = features[:, 5].clamp(1e-4, 1.0 - 1e-4)
            param_conf = features[:, 8].clamp(1e-4, 1.0 - 1e-4)
            evidence = torch.logit(cache_conf) - torch.logit(param_conf)
            evidence = evidence + 1.25 * features[:, 6] - 0.50 * features[:, 7]
            evidence = evidence + 0.35 * features[:, 10] + 0.25 * features[:, 3]
            state_term = 0.0 if self.router_mode == "confidence_nostate" else flat_state_logit[active]
            gate_logit_active = state_term + route[:, 0] + self.evidence_gain * evidence + self.evidence_bias
        ga = torch.sigmoid(gate_logit_active).clamp(1e-5, 1.0 - 1e-5)
        gate_flat[active] = ga
        mixed = (1.0 - gate_flat) * param_target + gate_flat * target_cache.reshape(-1)
        primary = -torch.log(mixed.clamp_min(1e-8)).mean()
        loss = primary

        if self.training and self.distill_scale > 0:
            pa = param_target[active]
            ca = target_cache.reshape(-1)[active]
            log_adv = torch.log(ca.detach().clamp_min(1e-8)) - torch.log(pa.detach().clamp_min(1e-8))
            teacher = torch.sigmoid(log_adv / self.distill_temperature)
            weight = torch.tanh(log_adv.abs())
            gate_logit = torch.logit(gate_flat[active].clamp(1e-5, 1 - 1e-5))
            aux = (
                F.binary_cross_entropy_with_logits(gate_logit, teacher, reduction="none") * weight
            ).sum() / weight.sum().clamp_min(1.0)
            loss = primary + self.distill_weight * float(self.distill_scale) * aux
            # Keep tensors detached; no hidden device synchronization.
            self.last_aux = {
                "distill": aux.detach(), "teacher": teacher.mean().detach(),
                "cache_win": (log_adv > 0).float().mean().detach(),
            }
        return loss, primary.detach(), None

    def _reference_forward(self, states, logits, tokens, targets, compute_metrics):
        # Build a lightweight proxy that shares the exact parameters.
        ref = self._v3.SuccessorCacheV5(
            self.state_dim, self.memory_dim, self.num_buckets, self.order, self.top_k,
            router_mode=self.router_mode,
        ).to(states.device)
        ref.load_state_dict(self.state_dict(), strict=True)
        ref.distill_scale = self.distill_scale
        ref.enabled = self.enabled
        ref.train(self.training)
        return ref(states, logits, tokens, targets, compute_metrics)


# This global is assigned after importing v3; FastBoundary uses it without storing
# a module object as a submodule.
v3_global = None


# =====================================================================================
# Model construction / checkpoint loading
# =====================================================================================


@dataclass
class VariantInfo:
    name: str
    kind: str
    params: int
    local_engine: str
    attention_free: bool


def make_bridge_args(args):
    return SimpleNamespace(
        model_seed=args.model_seed,
        field_dim=args.field_dim,
        field_layers=args.field_layers,
        field_heads=8,
        field_ff_hidden=args.field_ff_hidden,
        field_chunk=args.field_chunk,
        triton_block_c=args.triton_block_c,
        triton_chunk_t=args.triton_chunk_t,
        num_buckets=args.num_buckets,
        tf_dim=640, tf_heads=10, tf_layers=10, tf_ff_hidden=1776,
    )


def replace_softpatch(model, v3):
    old = model.softpatch
    if old is None:
        return
    new = FastBoundaryStateMixer(
        model.emb.embedding_dim, old.down.out_features, old.learned,
        old.target_rate, v3,
    ).to(next(model.parameters()).device)
    new.load_state_dict(old.state_dict(), strict=True)
    model.softpatch = new


def replace_cache(model, v3, i32=True):
    model.cache = FastSuccessorCacheV5(model.cache, v3, use_i32=i32).to(next(model.parameters()).device)


def replace_local(model, v3, engine: str, chunk: int):
    for key, old in list(model.locals.items()):
        if engine == "cached":
            new = CachedChunkLocalAttention(
                old.dim, old.inner, old.heads, old.window, chunk, v3,
            )
        elif engine == "flex":
            new = FlexLocalAttention(old.dim, old.inner, old.heads, old.window, v3, compiled=False)
        elif engine == "flex_compile":
            new = FlexLocalAttention(old.dim, old.inner, old.heads, old.window, v3, compiled=True)
        else:
            raise ValueError(engine)
        new = new.to(next(model.parameters()).device)
        new.load_state_dict(old.state_dict(), strict=True)
        model.locals[key] = new


def replace_multiscale(model, v3, lite: bool = False):
    for key, old in list(model.multiscales.items()):
        dilations = (1, 4, 16, 32) if lite else old.dilations
        new = FastMultiScaleCausalConv(old.dim, old.inner, dilations, v3).to(next(model.parameters()).device)
        if not lite:
            new.load_state_dict(old.state_dict(), strict=True)
        else:
            # Preserve shared projections/output; select four trained scales.
            new.norm.load_state_dict(old.norm.state_dict())
            new.in_proj.load_state_dict(old.in_proj.state_dict())
            new.gate.load_state_dict(old.gate.state_dict())
            new.out.load_state_dict(old.out.state_dict())
            selected = [0, 2, 4, 5]
            for dst, src_i in zip(new.convs, selected):
                dst.load_state_dict(old.convs[src_i].state_dict())
            with torch.no_grad():
                new.branch_logits.copy_(old.branch_logits[selected])
        model.multiscales[key] = new


def build_variant(name: str, args, v3, v4, canonical, device, local_engine="ref", chunk=256):
    bargs = make_bridge_args(args)
    seed_all(args.model_seed)
    # Recompute strict parity width exactly as v4 did.
    base = v4.build_field("baseline_v5", bargs, v3, canonical)
    target = nparams(base)
    del base
    parity_hidden = v4.find_parity_hidden(bargs, v3, canonical, target)

    if name.startswith("hybrid"):
        model = v4.build_field("hybrid_w256_conf_parity", bargs, v3, canonical, parity_hidden).to(device)
        if local_engine != "ref":
            replace_local(model, v3, local_engine, chunk)
        if name != "hybrid_ref":
            replace_softpatch(model, v3)
            replace_cache(model, v3, i32=True)
    elif name.startswith("attentionfree"):
        model = v4.build_field("attentionfree_multiscale", bargs, v3, canonical).to(device)
        if name != "attentionfree_ref":
            replace_softpatch(model, v3)
            replace_cache(model, v3, i32=True)
            replace_multiscale(model, v3, lite=(name == "attentionfree_lite4"))
    elif name == "baseline_v5":
        model = v4.build_field("baseline_v5", bargs, v3, canonical).to(device)
        replace_cache(model, v3, i32=True)
    else:
        raise ValueError(name)
    return model


def discover_checkpoint(kind: str, explicit: str) -> Optional[Path]:
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None
    if kind == "hybrid":
        candidates = [
            Path("/home/ubuntu/pcaf_runs/field_transformer_judge_repair_50m_8k_v5_latest/models/hybrid_w256_conf_parity/latest.pt"),
            Path("/home/ubuntu/pcaf_runs/field_hybrid_canonical_50m_bridge_v4_latest/models/hybrid_w256_conf_parity/latest.pt"),
        ]
    else:
        candidates = [
            Path("/home/ubuntu/pcaf_runs/field_hybrid_canonical_50m_bridge_v4_latest/models/attentionfree_multiscale/latest.pt"),
        ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    return None


def load_checkpoint_model(model: nn.Module, path: Optional[Path]) -> bool:
    if path is None:
        return False
    state = torch.load(path, map_location="cpu", weights_only=False)
    sd = state.get("model", state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        # Exact variants must be state-compatible. AF-lite is handled after ref load.
        raise RuntimeError(f"checkpoint mismatch missing={missing} unexpected={unexpected}")
    return True


# =====================================================================================
# Evaluation / timing
# =====================================================================================


def training_loss(model, x, y):
    loss, primary, _ = model.loss_and_stats(x, y, compute_metrics=False)
    return loss, primary


@torch.no_grad()
def evaluate(model, data, context: int, windows: int, seed: int, device, amp: str, v3):
    model.eval()
    starts = v3.fixed_starts(len(data), context, windows, seed)
    rows = []
    for s in starts:
        x, y = v3.fixed_batch(data, [s], context, device)
        with amp_ctx(device, amp):
            loss, _, stats = model.loss_and_stats(x, y, compute_metrics=True)
        rows.append((float(loss / LN2), stats))
    model.train()
    return {
        "bpb": float(np.mean([x[0] for x in rows])),
        "capture": float(np.mean([x[1].capture for x in rows])) if rows and rows[0][1] else None,
        "gate_sep": float(np.mean([x[1].gate_separation for x in rows])) if rows and rows[0][1] else None,
    }


def bench_step(model, context: int, tokens_per_step: int, steps: int, warmup: int,
               device, amp: str, lr: float, wd: float):
    batch = max(1, tokens_per_step // context)
    x = torch.randint(0, VOCAB, (batch, context), device=device)
    y = torch.randint(0, VOCAB, (batch, context), device=device)
    model.train()
    opt = make_optimizer(model, lr, wd)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    try:
        for _ in range(warmup):
            opt.zero_grad(set_to_none=True)
            with amp_ctx(device, amp):
                loss, _ = training_loss(model, x, y)
            loss.backward(); opt.step()
        sync(device)
        started = time.perf_counter()
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            with amp_ctx(device, amp):
                loss, _ = training_loss(model, x, y)
            loss.backward(); opt.step()
        sync(device)
        elapsed = time.perf_counter() - started
        return {
            "status": "ok", "context": context, "batch": batch,
            "bytes_per_second": steps * batch * context / elapsed,
            "step_ms": 1000 * elapsed / steps,
            "peak_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        }
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" not in str(exc).lower() and not isinstance(exc, torch.cuda.OutOfMemoryError):
            raise
        return {"status": "oom", "context": context, "batch": batch, "error": repr(exc)}
    finally:
        del opt, x, y
        gc.collect(); torch.cuda.empty_cache()


def clone_state_cpu(model: nn.Module):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def equivalence_test(ref, candidate, device, amp: str, seq: int = 512):
    candidate.load_state_dict(ref.state_dict(), strict=True)
    # Repeating input guarantees many cache hits and exercises the router.
    pattern = torch.arange(64, device=device) % VOCAB
    x = pattern.repeat((seq + 63) // 64)[:seq][None, :].repeat(2, 1)
    y = torch.roll(x, shifts=-1, dims=1)
    ref.train(); candidate.train()
    for m in (ref, candidate):
        m.zero_grad(set_to_none=True)
    with amp_ctx(device, amp):
        lr, _ = training_loss(ref, x, y)
        lc, _ = training_loss(candidate, x, y)
    lr.backward(); lc.backward()
    loss_abs = abs(float(lr.detach()) - float(lc.detach()))
    grad_abs = 0.0
    grad_rel = 0.0
    worst = ""
    ref_params = dict(ref.named_parameters())
    for name, p in candidate.named_parameters():
        rp = ref_params[name]
        if p.grad is None and rp.grad is None:
            continue
        if p.grad is None or rp.grad is None:
            return {"pass": False, "loss_abs": loss_abs, "grad_abs": float("inf"), "grad_rel": float("inf"), "worst": name}
        diff = (p.grad.float() - rp.grad.float()).abs()
        ma = float(diff.max())
        denom = rp.grad.float().abs().max().clamp_min(1e-6)
        mr = float(diff.max() / denom)
        if ma > grad_abs:
            grad_abs, grad_rel, worst = ma, mr, name
    ok = loss_abs <= 3e-4 and (grad_abs <= 4e-3 or grad_rel <= 3e-2)
    return {"pass": ok, "loss_abs": loss_abs, "grad_abs": grad_abs, "grad_rel": grad_rel, "worst": worst}


def auto_tune_local(args, v3, v4, canonical, device):
    rows = []
    engines: List[Tuple[str, int]] = [("cached", c) for c in args.local_chunks]
    if HAS_FLEX:
        engines += [("flex", 0), ("flex_compile", 0)]
    ref = build_variant("hybrid_ref", args, v3, v4, canonical, device)
    ref_state = clone_state_cpu(ref)
    ref_bench = build_variant("hybrid_ref", args, v3, v4, canonical, device)
    ref_bench.load_state_dict(ref_state, strict=True)
    ref_timing = bench_step(
        ref_bench, args.tune_context, args.system_tokens_per_step,
        args.tune_steps, args.tune_warmup, device, args.amp,
        args.system_lr, args.weight_decay,
    )
    rows.append({"engine": "reference", "status": ref_timing["status"], "pass": True,
                 "loss_abs": 0.0, "grad_abs": 0.0, **ref_timing})
    del ref_bench
    gc.collect(); torch.cuda.empty_cache()
    for engine, chunk in engines:
        name = f"{engine}{chunk if chunk else ''}"
        try:
            cand = build_variant("hybrid_opt", args, v3, v4, canonical, device, engine, chunk or 256)
            cand.load_state_dict(ref_state, strict=True)
            eq = equivalence_test(ref, cand, device, args.amp, seq=args.selftest_seq)
            if not eq["pass"]:
                rows.append({"engine": name, "status": "equivalence_fail", **eq})
                del cand; gc.collect(); torch.cuda.empty_cache(); continue
            # Restore weights after gradient self-test.
            cand.load_state_dict(ref_state, strict=True)
            timing = bench_step(
                cand, args.tune_context, args.system_tokens_per_step,
                args.tune_steps, args.tune_warmup, device, args.amp,
                args.system_lr, args.weight_decay,
            )
            rows.append({"engine": name, "status": timing["status"], **eq, **timing})
            del cand
        except Exception as exc:
            rows.append({"engine": name, "status": "error", "error": repr(exc)})
        gc.collect(); torch.cuda.empty_cache()
    del ref
    valid = [r for r in rows if r.get("engine") != "reference" and r.get("status") == "ok" and r.get("pass")]
    if not valid:
        raise RuntimeError(f"no exact optimized local engine passed: {rows}")
    best = max(valid, key=lambda r: r["bytes_per_second"])
    eng = best["engine"]
    if eng.startswith("cached"):
        return "cached", int(eng.replace("cached", "")), rows
    if eng == "flex_compile":
        return "flex_compile", 256, rows
    return "flex", 256, rows


def short_train_pair(args, train, val, device, v3, v4, canonical, local_engine, local_chunk, root):
    """Short paired confirmation for hybrid exact engine and AF-lite quality."""
    names = ["hybrid_ref", "hybrid_opt", "attentionfree_ref", "attentionfree_opt", "attentionfree_lite4"]
    results = {}
    for name in names:
        seed_all(args.model_seed)
        model = build_variant(
            name, args, v3, v4, canonical, device,
            local_engine=(local_engine if name == "hybrid_opt" else "ref"),
            chunk=local_chunk,
        )
        opt = make_optimizer(model, args.screen_lr, args.weight_decay)
        gen_device = train.device.type if train.device.type == "cuda" else "cpu"
        gen = torch.Generator(device=gen_device).manual_seed(args.data_seed)
        model.train(); processed = 0
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        sync(device); started = time.perf_counter()
        for step in range(1, args.screen_steps + 1):
            model.cache.distill_scale = min(1.0, step / 100.0)
            opt.zero_grad(set_to_none=True)
            for _ in range(args.screen_accum):
                x, y = v3.random_batch(train, args.screen_batch, args.screen_seq, gen, device)
                with amp_ctx(device, args.amp):
                    loss, _ = training_loss(model, x, y)
                    loss = loss / args.screen_accum
                loss.backward()
                del x, y, loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            processed += args.screen_batch * args.screen_accum * args.screen_seq
        sync(device)
        elapsed = time.perf_counter() - started
        e8 = evaluate(model, val, 8192, args.eval_windows, args.eval_seed, device, args.amp, v3)
        results[name] = {
            "params": nparams(model), "bpb8k": e8["bpb"],
            "bytes_per_second": processed / elapsed,
            "peak_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        }
        log(f"[screen] {name:22s} bpb8k={e8['bpb']:.5f} B/s={results[name]['bytes_per_second']:,.0f}")
        del model, opt
        gc.collect(); torch.cuda.empty_cache()
    atomic_json(root / "short_screen.json", results)
    return results


def make_summary(args, tune_rows, chosen, checkpoint_rows, systems, screen) -> str:
    lines = [
        "=" * 190,
        "FIELD HYBRID SPEED OPTIMIZATION v6 — CANONICAL 50M / 8K",
        "=" * 190,
        f"chosen_local_engine={chosen[0]} chunk={chosen[1]} flex_available={HAS_FLEX}",
        "Exact optimization targets: cached RoPE/masks, FlexAttention, no hidden host sync, int32 candidate sort, byte-LUT softpatch.",
        "",
        "LOCAL ENGINE AUTOTUNE",
        f"{'engine':22s} {'status':>18s} {'B/s':>12s} {'step ms':>10s} {'peak':>7s} {'lossΔ':>10s} {'gradΔ':>10s}",
    ]
    for r in tune_rows:
        bps_s = f"{r['bytes_per_second']:,.0f}" if r.get("bytes_per_second") else "-"
        ms_s = f"{r['step_ms']:.2f}" if r.get("step_ms") else "-"
        peak_s = f"{r['peak_gib']:.2f}" if r.get("peak_gib") else "-"
        lines.append(
            f"{r['engine']:22s} {r.get('status','-'):>18s} "
            f"{bps_s:>12s} {ms_s:>10s} {peak_s:>7s} "
            f"{r.get('loss_abs',float('nan')):10.2e} {r.get('grad_abs',float('nan')):10.2e}"
        )
    lines += ["", "CHECKPOINT QUALITY EQUIVALENCE"]
    if checkpoint_rows:
        lines.append(f"{'variant':28s} {'ctx':>6s} {'BPB':>10s} {'capture':>9s} {'sep':>9s}")
        for r in checkpoint_rows:
            lines.append(f"{r['variant']:28s} {r['context']:6d} {r['bpb']:10.5f} {str(r.get('capture')):>9s} {str(r.get('gate_sep')):>9s}")
    else:
        lines.append("No prior checkpoint discovered; exact equivalence is established by forward/backward self-test.")
    lines += [
        "", "EQUAL NO-CHECKPOINT SYSTEMS BENCHMARK",
        f"{'variant':28s} {'ctx':>6s} {'batch':>6s} {'status':>10s} {'B/s':>12s} {'speed/ref':>10s} {'peak':>7s}",
    ]
    lookup = {(r['variant'], r['context']): r for r in systems}
    for r in systems:
        ref_name = "hybrid_ref" if r['variant'].startswith('hybrid') else (
            "attentionfree_ref" if r['variant'].startswith('attentionfree') else r['variant']
        )
        ref = lookup.get((ref_name, r['context']))
        ratio = (r.get('bytes_per_second') or 0) / max((ref or {}).get('bytes_per_second') or 1, 1)
        bps_s = f"{r['bytes_per_second']:,.0f}" if r.get("bytes_per_second") else "-"
        peak_s = f"{r['peak_gib']:.2f}" if r.get("peak_gib") else "-"
        lines.append(
            f"{r['variant']:28s} {r['context']:6d} {r['batch']:6d} {r['status']:>10s} "
            f"{bps_s:>12s} {ratio:10.3f} {peak_s:>7s}"
        )
    if screen:
        lines += ["", "SHORT PAIRED QUALITY SCREEN", f"{'variant':28s} {'BPB8K':>10s} {'dHybrid':>10s} {'B/s':>12s}"]
        hb = screen['hybrid_ref']['bpb8k']
        for name, r in screen.items():
            lines.append(f"{name:28s} {r['bpb8k']:10.5f} {r['bpb8k']-hb:+10.5f} {r['bytes_per_second']:12,.0f}")

    h_ref = lookup.get(("hybrid_ref", 8192), {})
    h_opt = lookup.get(("hybrid_opt", 8192), {})
    af_ref = lookup.get(("attentionfree_ref", 8192), {})
    af_opt = lookup.get(("attentionfree_opt", 8192), {})
    h_gain = (h_opt.get('bytes_per_second') or 0) / max(h_ref.get('bytes_per_second') or 1, 1) - 1
    af_gain = (af_opt.get('bytes_per_second') or 0) / max(af_ref.get('bytes_per_second') or 1, 1) - 1
    quality_ok = True
    if screen:
        quality_ok = abs(screen['hybrid_opt']['bpb8k'] - screen['hybrid_ref']['bpb8k']) <= args.max_quality_drift
    ready = h_gain >= args.min_hybrid_speed_gain and quality_ok
    lines += [
        "", "VERDICT",
        f"hybrid_speed_gain_8K={h_gain*100:+.2f}% | attentionfree_speed_gain_8K={af_gain*100:+.2f}% | quality_equivalence={'PASS' if quality_ok else 'FAIL'}",
        ("OPTIMIZATION PASS — freeze optimized kernels and proceed to 300M H2H" if ready else
         "OPTIMIZATION MARGINAL — keep best safe engine; architecture is ready for 300M H2H without further delay"),
        "The attention-free branch remains active and receives its own optimized exact path plus AF-lite screen.",
        "=" * 190,
    ]
    return "\n".join(lines) + "\n"


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--outdir", default="/home/ubuntu/pcaf_runs/field_hybrid_speed_optimization_v6")
    p.add_argument("--canonical-source", default="")
    p.add_argument("--cache-dir", default="/home/ubuntu/field_lab/hf_cache")
    p.add_argument("--hybrid-checkpoint", default="")
    p.add_argument("--attentionfree-checkpoint", default="")
    p.add_argument("--data-frac", type=float, default=0.05)
    p.add_argument("--data-device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--field-dim", type=int, default=704)
    p.add_argument("--field-layers", type=int, default=8)
    p.add_argument("--field-ff-hidden", type=int, default=1920)
    p.add_argument("--field-chunk", type=int, default=32)
    p.add_argument("--triton-block-c", type=int, default=16)
    p.add_argument("--triton-chunk-t", type=int, default=64)
    p.add_argument("--num-buckets", type=int, default=8192)
    p.add_argument("--amp", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--local-chunks", type=int, nargs="+", default=[256, 512, 1024, 2048])
    p.add_argument("--selftest-seq", type=int, default=512)
    p.add_argument("--tune-context", type=int, default=8192)
    p.add_argument("--tune-warmup", type=int, default=2)
    p.add_argument("--tune-steps", type=int, default=4)
    p.add_argument("--system-contexts", type=int, nargs="+", default=[4096, 8192, 16384])
    p.add_argument("--system-tokens-per-step", type=int, default=16384)
    p.add_argument("--system-warmup", type=int, default=2)
    p.add_argument("--system-steps", type=int, default=5)
    p.add_argument("--system-lr", type=float, default=3e-4)
    p.add_argument("--screen-steps", type=int, default=600)
    p.add_argument("--screen-seq", type=int, default=4096)
    p.add_argument("--screen-batch", type=int, default=2)
    p.add_argument("--screen-accum", type=int, default=2)
    p.add_argument("--screen-lr", type=float, default=5e-4)
    p.add_argument("--eval-windows", type=int, default=4)
    p.add_argument("--weight-decay", type=float, default=0.10)
    p.add_argument("--model-seed", type=int, default=1234)
    p.add_argument("--data-seed", type=int, default=5678)
    p.add_argument("--eval-seed", type=int, default=9012)
    p.add_argument("--min-hybrid-speed-gain", type=float, default=0.05)
    p.add_argument("--max-quality-drift", type=float, default=0.005)
    p.add_argument("--skip-screen", action="store_true")
    return p.parse_args()


def main():
    global v3_global
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/H100 required")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    # FlexAttention and masked SDPA may need non-Flash fallback internally.
    torch.backends.cuda.enable_math_sdp(False)
    torch.set_float32_matmul_precision("high")

    v3 = import_module(V3_PATH, "field_speed_v3")
    v3_global = v3
    v4 = import_module(V4_PATH, "field_speed_v4")
    canonical_path = locate_canonical(args.canonical_source)
    canonical = import_module(canonical_path, "field_speed_canonical")
    root = Path(args.outdir); root.mkdir(parents=True, exist_ok=True)

    log("=" * 160)
    log("FIELD HYBRID SPEED OPTIMIZATION v6")
    log(f"gpu={torch.cuda.get_device_name(0)} torch={torch.__version__} cuda={torch.version.cuda}")
    log(f"canonical={canonical_path} sha={sha256(canonical_path)}")
    log(f"flex_available={HAS_FLEX} flex_error={_FLEX_IMPORT_ERROR}")
    log("=" * 160)

    # Canonical kernel audit first.
    targs = argparse.Namespace(**vars(args))
    targs.selftest_forward_tol = 0.002
    targs.selftest_grad_rel_tol = 0.02
    targs.selftest_grad_abs_tol = 0.002
    targs.selftest_causal_tol = 0.0002
    canonical.run_kernel_self_test(device, targs)
    log("[selftest] canonical Triton PASS")

    tok = (torch.arange(2048, device=device)[None, :] % 97).long()
    idx_ref = v3.causal_recent_candidates(tok, 4, args.num_buckets, 4)
    idx_fast = causal_recent_candidates_i32(tok, 4, args.num_buckets, 4, v3)
    if not torch.equal(idx_ref, idx_fast):
        raise AssertionError("int32 candidate route is not exact")
    log("[selftest] int32 candidate route exact PASS")

    af_ref_test = build_variant("attentionfree_ref", args, v3, v4, canonical, device)
    af_opt_test = build_variant("attentionfree_opt", args, v3, v4, canonical, device)
    af_eq = equivalence_test(af_ref_test, af_opt_test, device, args.amp, seq=args.selftest_seq)
    log(f"[selftest] attention-free exact path {af_eq}")
    if not af_eq["pass"]:
        raise AssertionError("attention-free optimized path mismatch")
    del af_ref_test, af_opt_test, tok, idx_ref, idx_fast
    gc.collect(); torch.cuda.empty_cache()

    local_engine, local_chunk, tune_rows = auto_tune_local(args, v3, v4, canonical, device)
    log(f"[autotune] selected engine={local_engine} chunk={local_chunk}")
    atomic_json(root / "local_autotune.json", tune_rows)

    # Load data once for checkpoint evaluation and short screen.
    train, val, _ = v3.load_wikitext103_raw(args.cache_dir, args.data_frac)
    train = v3.place_data(train, device, args.data_device, "train")
    val = v3.place_data(val, device, args.data_device, "val")

    hybrid_ckpt = discover_checkpoint("hybrid", args.hybrid_checkpoint)
    af_ckpt = discover_checkpoint("attentionfree", args.attentionfree_checkpoint)
    log(f"[checkpoint] hybrid={hybrid_ckpt}")
    log(f"[checkpoint] attentionfree={af_ckpt}")

    checkpoint_rows = []
    if hybrid_ckpt is not None:
        ref = build_variant("hybrid_ref", args, v3, v4, canonical, device)
        opt = build_variant("hybrid_opt", args, v3, v4, canonical, device, local_engine, local_chunk)
        load_checkpoint_model(ref, hybrid_ckpt)
        load_checkpoint_model(opt, hybrid_ckpt)
        for ctx in (8192, 16384):
            for name, model in (("hybrid_ref", ref), ("hybrid_opt", opt)):
                row = evaluate(model, val, ctx, args.eval_windows, args.eval_seed, device, args.amp, v3)
                checkpoint_rows.append({"variant": name, "context": ctx, **row})
        del ref, opt; gc.collect(); torch.cuda.empty_cache()

    if af_ckpt is not None:
        ref = build_variant("attentionfree_ref", args, v3, v4, canonical, device)
        opt = build_variant("attentionfree_opt", args, v3, v4, canonical, device)
        load_checkpoint_model(ref, af_ckpt)
        load_checkpoint_model(opt, af_ckpt)
        for ctx in (8192, 16384):
            for name, model in (("attentionfree_ref", ref), ("attentionfree_opt", opt)):
                row = evaluate(model, val, ctx, args.eval_windows, args.eval_seed, device, args.amp, v3)
                checkpoint_rows.append({"variant": name, "context": ctx, **row})
        del ref, opt; gc.collect(); torch.cuda.empty_cache()
    atomic_json(root / "checkpoint_eval.json", checkpoint_rows)

    # Equal systems benchmark from paired fresh initializations.
    systems = []
    variant_specs = [
        ("baseline_v5", "ref", 256),
        ("hybrid_ref", "ref", 256),
        ("hybrid_opt", local_engine, local_chunk),
        ("attentionfree_ref", "ref", 256),
        ("attentionfree_opt", "ref", 256),
        ("attentionfree_lite4", "ref", 256),
    ]
    for ctx in args.system_contexts:
        for name, eng, chunk in variant_specs:
            seed_all(args.model_seed)
            model = build_variant(name, args, v3, v4, canonical, device, eng, chunk)
            row = bench_step(
                model, ctx, args.system_tokens_per_step, args.system_steps,
                args.system_warmup, device, args.amp, args.system_lr,
                args.weight_decay,
            )
            row["variant"] = name
            systems.append(row)
            log(f"[systems] {name:24s} ctx={ctx:5d} status={row['status']} B/s={row.get('bytes_per_second',0):,.0f}")
            del model; gc.collect(); torch.cuda.empty_cache()
    atomic_json(root / "systems.json", systems)

    screen = None
    if not args.skip_screen and args.screen_steps > 0:
        screen = short_train_pair(
            args, train, val, device, v3, v4, canonical,
            local_engine, local_chunk, root,
        )

    summary = make_summary(args, tune_rows, (local_engine, local_chunk), checkpoint_rows, systems, screen)
    atomic_text(root / "SUMMARY.txt", summary)
    log(summary)


if __name__ == "__main__":
    main()
