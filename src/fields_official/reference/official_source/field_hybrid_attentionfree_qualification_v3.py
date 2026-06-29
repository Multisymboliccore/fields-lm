#!/usr/bin/env python3
"""Field hybrid / attention-free qualification arena v3.

V2 established a reproducible winner: soft-patch hierarchy + one short exact
local-attention path + confidence-aware PCAF routing.  V3 is a consolidation
round before any new 300M head-to-head.  It tests whether the gain survives:

* a longer 2K training context and 8K/16K evaluation,
* two additional paired confirmation seeds,
* strict parameter matching,
* a 4K-context bridge run,
* and attention-free replacements for the local-attention side path.

The attention-free replacement is a zero-initialized low-rank multiscale causal
depthwise-convolution mixer with dilations 1,2,4,8,16,32 (effective radius 127).
It has no query-key dot products, no softmax over positions, and no attention
matrix.  This remains a mechanism-ranking arena using the exact portable Field
reference math.  A final 300M run requires transplanting the winner into the
canonical optimized Triton Field-PCAF implementation.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LN2 = math.log(2.0)
VOCAB = 256
VAC_MAX = 0.90
ARMS = (
    "baseline_v5",
    "softpatch_conf",
    "local_w128_conf",
    "softpatch_local_w128_conf",
    "softpatch_local_w128_two_conf",
    "softpatch_local_w256_conf",
    "multiscale_conv_conf",
    "softpatch_multiscale_conf",
    "softpatch_local_w128_conf_parity",
)


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


def amp_ctx(device: torch.device, amp: str):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[amp]
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=(device.type == "cuda" and amp != "fp32"),
    )


def lr_at(step: int, total: int, warmup: int, peak: float, min_ratio: float) -> float:
    if step <= warmup:
        return peak * step / max(1, warmup)
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (min_ratio + (1.0 - min_ratio) * cosine)


def make_optimizer(model: nn.Module, lr: float, weight_decay: float):
    kwargs = dict(lr=lr, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=weight_decay)
    try:
        return torch.optim.AdamW(model.parameters(), fused=torch.cuda.is_available(), **kwargs)
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(model.parameters(), **kwargs)


# ======================================================================================
# Data
# ======================================================================================


def _join_text_rows(rows: Iterable[Dict[str, str]]) -> bytes:
    parts: List[bytes] = []
    for row in rows:
        text = row.get("text", "")
        if text:
            parts.append(text.encode("utf-8", errors="replace"))
            parts.append(b"\n")
    return b"".join(parts)


def load_wikitext103_raw(cache_dir: str, data_frac: float):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Missing datasets package") from exc
    log("[data] loading Salesforce/wikitext, wikitext-103-raw-v1")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", cache_dir=cache_dir)
    train_raw = _join_text_rows(ds["train"])
    val_raw = _join_text_rows(ds["validation"])
    test_raw = _join_text_rows(ds["test"])
    if not 0.0 < data_frac <= 1.0:
        raise ValueError("data_frac must be in (0,1]")
    train_raw = train_raw[: max(2, int(len(train_raw) * data_frac))]

    def as_u8(raw: bytes) -> torch.Tensor:
        return torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy())

    train, val, test = map(as_u8, (train_raw, val_raw, test_raw))
    log(f"[data] train={len(train):,} ({data_frac:.1%}) val={len(val):,} test={len(test):,}")
    return train, val, test


def place_data(x: torch.Tensor, device: torch.device, mode: str, name: str) -> torch.Tensor:
    if mode == "cuda" or (mode == "auto" and device.type == "cuda"):
        y = x.to(device=device, dtype=torch.uint8)
        log(f"[data] {name} -> GPU ({y.numel()/2**20:.1f} MiB)")
        return y
    y = x.contiguous()
    if device.type == "cuda":
        try:
            y = y.pin_memory()
        except RuntimeError:
            pass
    log(f"[data] {name} kept on CPU ({y.numel()/2**20:.1f} MiB)")
    return y


_OFFSET_CACHE: Dict[Tuple[str, int], torch.Tensor] = {}


def offsets_for(data: torch.Tensor, seq_len: int) -> torch.Tensor:
    key = (str(data.device), seq_len)
    if key not in _OFFSET_CACHE:
        _OFFSET_CACHE[key] = torch.arange(seq_len + 1, device=data.device, dtype=torch.long)
    return _OFFSET_CACHE[key]


def random_batch(data: torch.Tensor, batch: int, seq: int, gen: torch.Generator, device: torch.device):
    starts = torch.randint(0, len(data) - seq - 1, (batch,), generator=gen, device=data.device)
    win = data[starts[:, None] + offsets_for(data, seq)[None, :]].long()
    if win.device != device:
        win = win.to(device, non_blocking=True)
    return win[:, :-1], win[:, 1:]


def fixed_starts(data_len: int, seq: int, windows: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed + seq * 1009)
    return rng.integers(0, data_len - seq - 1, size=windows).tolist()


def fixed_batch(data: torch.Tensor, starts: Sequence[int], seq: int, device: torch.device):
    win = torch.stack([data[s : s + seq + 1] for s in starts]).long()
    if win.device != device:
        win = win.to(device, non_blocking=True)
    return win[:, :-1], win[:, 1:]


# ======================================================================================
# Field v4 portable exact reference
# ======================================================================================


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        y = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


class PackedSwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: Optional[int] = None):
        super().__init__()
        self.hidden = int(hidden or ((int(8 * dim / 3) + 63) // 64) * 64)
        self.w12 = nn.Linear(dim, 2 * self.hidden, bias=False)
        self.w3 = nn.Linear(self.hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, v = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(g) * v)


def assoc_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.clone()
    b = b.clone()
    shift = 1
    while shift < a.shape[1]:
        ap = torch.cat((torch.ones_like(a[:, :shift]), a[:, :-shift]), dim=1)
        bp = torch.cat((torch.zeros_like(b[:, :shift]), b[:, :-shift]), dim=1)
        b = a * bp + b
        a = a * ap
        shift *= 2
    return b


def hierarchical_scan(a: torch.Tensor, b: torch.Tensor, block: int = 32) -> torch.Tensor:
    if block < 1 or block > 64:
        raise ValueError(block)
    batch, length, channels = a.shape
    pad = (-length) % block
    if pad:
        a = torch.cat((a, a.new_ones(batch, pad, channels)), dim=1)
        b = torch.cat((b, b.new_zeros(batch, pad, channels)), dim=1)
    padded = a.shape[1]
    groups = padded // block
    ac = a.reshape(batch, groups, block, channels)
    bc = b.reshape(batch, groups, block, channels)
    la = ac.reshape(batch * groups, block, channels).clone()
    lb = bc.reshape(batch * groups, block, channels).clone()
    shift = 1
    while shift < block:
        ap = torch.cat((torch.ones_like(la[:, :shift]), la[:, :-shift]), dim=1)
        bp = torch.cat((torch.zeros_like(lb[:, :shift]), lb[:, :-shift]), dim=1)
        lb = la * bp + lb
        la = la * ap
        shift *= 2
    la = la.reshape(batch, groups, block, channels)
    lb = lb.reshape(batch, groups, block, channels)
    carry_out = assoc_scan(la[:, :, -1], lb[:, :, -1])
    carry_in = torch.cat((carry_out.new_zeros(batch, 1, channels), carry_out[:, :-1]), dim=1)
    states = la * carry_in[:, :, None, :] + lb
    return states.reshape(batch, padded, channels)[:, :length]


def reference_field_read(raw, transition_r, transition_i, gamma, field_chunk):
    channels = transition_r.numel()
    inj_r = torch.tanh(raw[..., :channels])
    inj_i = torch.tanh(raw[..., channels : 2 * channels])
    vacancy = torch.sigmoid(raw[..., 2 * channels :]) * VAC_MAX
    injection = torch.complex(inj_r, inj_i)
    transition = torch.complex(transition_r, transition_i)
    a = (1.0 - vacancy).to(torch.complex64) * transition
    b = gamma.to(torch.complex64) * vacancy.to(torch.complex64) * injection
    states = hierarchical_scan(a, b, field_chunk)
    previous = torch.cat((torch.zeros_like(states[:, :1]), states[:, :-1]), dim=1)
    moved = transition * previous
    displaced = vacancy.to(torch.complex64) * moved
    return torch.cat((states.real, states.imag, displaced.real, displaced.imag), dim=-1)


class IndependentVacancyField(nn.Module):
    def __init__(self, dim: int, field_chunk: int = 32):
        super().__init__()
        if dim % 2:
            raise ValueError("dim must be even")
        self.dim = dim
        self.channels = dim // 2
        self.field_chunk = field_chunk
        self.write_proj = nn.Linear(dim, dim + self.channels)
        ring = torch.linspace(0.85, 0.999, self.channels)
        self.radius_logit = nn.Parameter(torch.log(ring / (1.0 - ring)))
        self.theta = nn.Parameter(torch.linspace(0.03, math.pi * 0.97, self.channels))
        self.read_norm = RMSNorm(2 * dim)
        self.out_proj = nn.Linear(2 * dim, dim)
        self.gate_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.write_proj(x).contiguous().float()
        radius = torch.sigmoid(self.radius_logit).clamp(0.50, 0.99995).float()
        theta = self.theta.float()
        tr = radius * torch.cos(theta)
        ti = radius * torch.sin(theta)
        gamma = torch.sqrt((1.0 - radius.square()).clamp_min(1e-4))
        read = reference_field_read(raw, tr, ti, gamma, self.field_chunk)
        out = self.out_proj(self.read_norm(read))
        gate = torch.sigmoid(self.gate_proj(x))
        return x + (out * gate).to(x.dtype)


class FieldBlock(nn.Module):
    def __init__(self, dim: int, ff_hidden: int, field_chunk: int = 32):
        super().__init__()
        self.mixer = IndependentVacancyField(dim, field_chunk)
        self.ff_norm = RMSNorm(dim)
        self.ff = PackedSwiGLU(dim, ff_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mixer(x)
        return x + self.ff(self.ff_norm(x))


# ======================================================================================
# Experimental side paths
# ======================================================================================


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class LowRankLocalAttention(nn.Module):
    def __init__(self, dim: int, inner: int = 128, heads: int = 4, window: int = 128, chunk: int = 256):
        super().__init__()
        if inner % heads:
            raise ValueError("inner must divide heads")
        self.dim, self.inner, self.heads = dim, inner, heads
        self.head_dim = inner // heads
        self.window, self.chunk = window, chunk
        self.norm = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * inner, bias=False)
        self.out = nn.Linear(inner, dim, bias=False)
        nn.init.zeros_(self.out.weight)
        inv = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def _rope(self, z: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        freq = torch.outer(positions.float(), self.inv_freq.to(positions.device))
        emb = torch.repeat_interleave(freq, 2, dim=-1).to(z.dtype)[None, None]
        return z * emb.cos() + rotate_half(z) * emb.sin()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        z = self.qkv(self.norm(x)).view(b, t, 3, self.heads, self.head_dim)
        q, k, v = z.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        pos = torch.arange(t, device=x.device)
        q = self._rope(q, pos)
        k = self._rope(k, pos)
        outs: List[torch.Tensor] = []
        for q0 in range(0, t, self.chunk):
            q1 = min(t, q0 + self.chunk)
            k0 = max(0, q0 - self.window + 1)
            qa = q[:, :, q0:q1]
            ka = k[:, :, k0:q1]
            va = v[:, :, k0:q1]
            qp = pos[q0:q1, None]
            kp = pos[None, k0:q1]
            allowed = (kp <= qp) & (kp >= qp - self.window + 1)
            ya = F.scaled_dot_product_attention(qa, ka, va, attn_mask=allowed, dropout_p=0.0)
            outs.append(ya)
        y = torch.cat(outs, dim=2).transpose(1, 2).contiguous().view(b, t, self.inner)
        return x + self.out(y)


class MultiScaleCausalConv(nn.Module):
    """Attention-free low-rank local mixer with ~128-byte causal receptive field."""

    def __init__(self, dim: int, inner: int = 128, dilations=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.dim = dim
        self.inner = inner
        self.dilations = tuple(int(d) for d in dilations)
        self.norm = RMSNorm(dim)
        self.in_proj = nn.Linear(dim, inner, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(inner, inner, kernel_size=3, dilation=d, groups=inner, bias=False)
            for d in self.dilations
        ])
        self.branch_logits = nn.Parameter(torch.zeros(len(self.dilations)))
        self.gate = nn.Linear(dim, inner, bias=True)
        self.out = nn.Linear(inner, dim, bias=False)
        # Preserve paired initialization and make the arm start exactly at baseline.
        nn.init.zeros_(self.out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        z = self.in_proj(h).transpose(1, 2)
        branches = []
        for conv, dilation in zip(self.convs, self.dilations):
            # Left-only padding: strictly causal and same sequence length.
            za = F.pad(z, (2 * dilation, 0))
            branches.append(F.silu(conv(za)))
        mix = torch.softmax(self.branch_logits.float(), dim=0).to(z.dtype)
        y = sum(w * branch for w, branch in zip(mix, branches))
        y = y.transpose(1, 2)
        y = y * torch.sigmoid(self.gate(h))
        return x + self.out(y)


_BOUNDARY_BYTES = (9, 10, 13, 32, 33, 34, 39, 40, 41, 44, 45, 46, 47, 58, 59, 63, 91, 93, 123, 125)


def boundary_mask(tokens: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(tokens, dtype=torch.bool)
    for value in _BOUNDARY_BYTES:
        out |= tokens == value
    return out


class BoundaryStateMixer(nn.Module):
    def __init__(self, dim: int, rank: int = 64, learned: bool = False, target_rate: float = 0.16):
        super().__init__()
        self.learned = learned
        self.target_rate = target_rate
        self.norm = RMSNorm(dim)
        self.down = nn.Linear(dim, rank, bias=False)
        self.score = nn.Linear(dim, 1, bias=True) if learned else None
        if self.score is not None:
            nn.init.zeros_(self.score.weight)
            nn.init.constant_(self.score.bias, math.log(target_rate / (1.0 - target_rate)))
        self.up = nn.Linear(2 * rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)
        self.last_aux = torch.tensor(0.0)
        self.last_rate = 0.0

    def forward(self, x: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        z = self.down(h).float()
        prior = boundary_mask(tokens).float()
        if self.learned:
            score = self.score(h).float().squeeze(-1) + 1.5 * prior
            boundary = torch.sigmoid(score).clamp(0.01, 0.99)
            rate = boundary.mean()
            self.last_aux = 0.02 * (rate - self.target_rate).square()
            self.last_rate = float(rate.detach())
        else:
            boundary = prior
            # Always initialize the first state causally.
            boundary = boundary.clone()
            boundary[:, 0] = 1.0
            self.last_aux = x.new_zeros(())
            self.last_rate = float(boundary.mean())
        a = (1.0 - boundary)[..., None].expand_as(z)
        drive = boundary[..., None] * z
        state = hierarchical_scan(a, drive, block=32)
        delta = self.up(torch.cat((z, state), dim=-1).to(x.dtype))
        return x + delta


# ======================================================================================
# PCAF v5-like exact successor cache
# ======================================================================================


def causal_ngram_buckets(tokens: torch.Tensor, order: int, num_buckets: int) -> torch.Tensor:
    h = torch.zeros_like(tokens, dtype=torch.int64)
    for shift in range(order - 1, -1, -1):
        if shift == 0:
            shifted = tokens.long()
        else:
            shifted = torch.cat((torch.zeros_like(tokens[:, :shift]), tokens[:, :-shift]), dim=1).long()
        h = torch.remainder(h * 1_000_003 + shifted + 97, 2_147_483_647)
    return torch.remainder(h, num_buckets).long()


def causal_recent_candidates(tokens: torch.Tensor, order: int, num_buckets: int, top_k: int) -> torch.Tensor:
    b, t = tokens.shape
    buckets = causal_ngram_buckets(tokens, order, num_buckets)
    pos = torch.arange(t, device=tokens.device, dtype=torch.long)[None, :].expand(b, -1)
    bid = torch.arange(b, device=tokens.device, dtype=torch.long)[:, None].expand(-1, t)
    group = bid * num_buckets + buckets
    key = group * (t + 1) + pos
    flat_key = key.reshape(-1)
    perm = torch.argsort(flat_key, stable=True)
    sg = group.reshape(-1).index_select(0, perm)
    sp = pos.reshape(-1).index_select(0, perm)
    n = perm.numel()
    cand_sorted = torch.full((n, top_k), -1, device=tokens.device, dtype=torch.long)
    for k in range(1, top_k + 1):
        if k >= n:
            break
        same = sg[k:] == sg[:-k]
        prev = torch.where(same, sp[:-k], torch.full_like(sp[:-k], -1))
        cand_sorted[k:, k - 1] = prev
    out = torch.full_like(cand_sorted, -1)
    out[perm] = cand_sorted
    return out.view(b, t, top_k)


def normalize_rows(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.square().sum(dim=-1, keepdim=True).clamp_min(eps))


@dataclass
class CacheStats:
    coverage: float
    gate: float
    hit: float
    cache_prob: float
    param_bpb: float
    oracle_bpb: float
    capture: float
    cache_win_rate: float
    gate_when_cache_wins: float
    gate_when_cache_loses: float
    gate_separation: float


class SuccessorCacheV5(nn.Module):
    FEATURE_DIM = 15

    def __init__(self, state_dim: int, memory_dim: int = 64, num_buckets: int = 8192,
                 order: int = 4, top_k: int = 4, router_mode: str = "v5"):
        super().__init__()
        if router_mode not in {"v5", "confidence", "confidence_nostate"}:
            raise ValueError(router_mode)
        self.state_dim = state_dim
        self.memory_dim = memory_dim
        self.num_buckets = num_buckets
        self.order = order
        self.top_k = top_k
        self.router_mode = router_mode
        self.shared_weight = nn.Parameter(torch.empty(memory_dim, state_dim))
        nn.init.kaiming_uniform_(self.shared_weight, a=math.sqrt(5))
        self.state_gate = nn.Sequential(RMSNorm(state_dim), nn.Linear(state_dim, 1))
        if router_mode == "v5":
            self.router = nn.Sequential(
                nn.Linear(self.FEATURE_DIM, 32), nn.SiLU(), nn.Linear(32, 1, bias=False)
            )
            nn.init.zeros_(self.router[-1].weight)
            self.evidence_gain = None
            self.evidence_bias = None
            self.distill_weight = 0.02
        else:
            self.router = nn.Sequential(
                nn.LayerNorm(self.FEATURE_DIM),
                nn.Linear(self.FEATURE_DIM, 64), nn.SiLU(),
                nn.Linear(64, 32), nn.SiLU(),
                nn.Linear(32, 1),
            )
            nn.init.zeros_(self.router[-1].weight)
            nn.init.zeros_(self.router[-1].bias)
            # Direct causal evidence prior.  The MLP learns residual corrections,
            # while this term prevents the router from needing to rediscover the
            # basic cache-confidence versus parametric-confidence comparison.
            self.evidence_gain = nn.Parameter(torch.tensor(0.50))
            self.evidence_bias = nn.Parameter(torch.tensor(-0.75))
            self.distill_weight = 0.05
        self.recency_scale = nn.Parameter(torch.tensor(1.0))
        self.distill_temperature = 0.5
        self.distill_scale = 1.0
        self.enabled = True
        self.last_aux: Dict[str, float] = {}

    def _features(self, scores, weights, valid, cand_tokens, recency, logits):
        n, k = valid.shape
        masked = scores.masked_fill(~valid, -1.0e9)
        top2 = torch.topk(masked, k=min(2, k), dim=-1).values
        top1 = top2[:, 0]
        count = valid.float().sum(-1)
        margin = torch.where(count >= 2, top2[:, 0] - top2[:, 1], torch.zeros_like(top1)) if top2.size(-1) > 1 else torch.zeros_like(top1)
        cand_ent = -(weights * torch.log(weights.clamp_min(1e-8))).sum(-1)
        cand_ent = torch.where(count > 1, cand_ent / torch.log(count.clamp_min(2.0)), torch.zeros_like(cand_ent))
        wrec = (weights * recency).sum(-1)
        cl = cand_tokens.long()
        same = cl[:, :, None] == cl[:, None, :]
        token_mass = (same.float() * weights[:, None, :]).sum(-1)
        earlier = torch.tril(torch.ones((k, k), device=valid.device, dtype=torch.bool), diagonal=-1)
        unique = valid & ~(same & earlier[None]).any(-1)
        unique_mass = token_mass.masked_fill(~unique, 0.0)
        mass2, massidx = torch.topk(unique_mass, k=min(2, k), dim=-1)
        cache_conf = mass2[:, 0]
        cache_second = mass2[:, 1] if mass2.size(-1) > 1 else torch.zeros_like(cache_conf)
        cache_margin = cache_conf - cache_second
        cache_ent = -(unique_mass * torch.log(unique_mass.clamp_min(1e-8))).sum(-1) / math.log(max(k, 2))
        cache_top = cl.gather(1, massidx[:, :1]).squeeze(1)
        l32 = logits.float()
        logz = torch.logsumexp(l32, -1)
        ptop, ptok = torch.topk(l32, 2, dim=-1)
        pconf = torch.exp(ptop[:, 0] - logz)
        psecond = torch.exp(ptop[:, 1] - logz)
        pmargin = pconf - psecond
        ptoken = ptok[:, 0]
        p_cache_top = torch.exp(l32.gather(1, cache_top[:, None]).squeeze(1) - logz)
        cache_mass_ptop = (weights * (cl == ptoken[:, None]).float()).sum(-1)
        agree = (cache_top == ptoken).float()
        delta = cache_conf - pconf
        f = torch.stack((
            torch.tanh(top1), torch.tanh(margin), cand_ent.clamp(0, 1),
            (count / float(k)).clamp(0, 1), wrec.clamp(0, 1), cache_conf.clamp(0, 1),
            cache_margin.clamp(0, 1), cache_ent.clamp(0, 1), pconf.clamp(0, 1),
            pmargin.clamp(0, 1), agree, p_cache_top.clamp(0, 1),
            cache_mass_ptop.clamp(0, 1), delta.clamp(-1, 1), delta.abs().clamp(0, 1),
        ), dim=-1)
        return f.detach()

    def forward(self, states, logits, tokens, targets, compute_metrics=False):
        flat_logits = logits.reshape(-1, VOCAB)
        flat_targets = targets.reshape(-1)
        param_nll = F.cross_entropy(flat_logits.float(), flat_targets, reduction="none")
        param_target = torch.exp(-param_nll)
        if not self.enabled:
            primary = param_nll.mean()
            stats = CacheStats(0, 0, 0, 0, float(primary / LN2), float(primary / LN2), 0, 0, 0, 0, 0) if compute_metrics else None
            return primary, primary.detach(), stats

        b, t, d = states.shape
        idx = causal_recent_candidates(tokens, self.order, self.num_buckets, self.top_k)
        valid = idx >= 0
        has = valid.any(-1)
        safe = idx.clamp_min(0)
        batch_idx = torch.arange(b, device=states.device)[:, None, None]
        proj = normalize_rows(F.linear(states.float(), self.shared_weight.float()))
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
        ) if bool(active.any()) else states.new_zeros((0, self.FEATURE_DIM))
        route = self.router(features) if features.numel() else states.new_zeros((0, 1))
        flat_state_logit = state_logit.reshape(-1)
        gate_flat = torch.zeros_like(flat_state_logit)
        if bool(active.any()):
            if self.router_mode == "v5":
                gate_logit_active = flat_state_logit[active] + route[:, 0]
            else:
                # Feature layout: cache_conf=5, cache_margin=6, cache_entropy=7,
                # param_conf=8, param_margin=9, agreement=10, count_fraction=3.
                cache_conf = features[:, 5].clamp(1e-4, 1.0 - 1e-4)
                param_conf = features[:, 8].clamp(1e-4, 1.0 - 1e-4)
                evidence = torch.logit(cache_conf) - torch.logit(param_conf)
                evidence = evidence + 1.25 * features[:, 6] - 0.50 * features[:, 7]
                evidence = evidence + 0.35 * features[:, 10] + 0.25 * features[:, 3]
                state_term = 0.0 if self.router_mode == "confidence_nostate" else flat_state_logit[active]
                gate_logit_active = state_term + route[:, 0] + self.evidence_gain * evidence + self.evidence_bias
            ga = torch.sigmoid(gate_logit_active).clamp(1e-5, 1.0 - 1e-5)
            gate_flat[active] = ga
        gate = gate_flat.view(b, t)
        mixed = (1.0 - gate.reshape(-1)) * param_target + gate.reshape(-1) * target_cache.reshape(-1)
        primary = -torch.log(mixed.clamp_min(1e-8)).mean()
        loss = primary

        if self.training and bool(active.any()) and self.distill_scale > 0:
            pa = param_target[active]
            ca = target_cache.reshape(-1)[active]
            log_adv = torch.log(ca.detach().clamp_min(1e-8)) - torch.log(pa.detach().clamp_min(1e-8))
            teacher = torch.sigmoid(log_adv / self.distill_temperature)
            weight = torch.tanh(log_adv.abs())
            ga = gate_flat[active].clamp(1e-5, 1 - 1e-5)
            gate_logit = torch.logit(ga)
            aux = (F.binary_cross_entropy_with_logits(gate_logit, teacher, reduction="none") * weight).sum() / weight.sum().clamp_min(1.0)
            loss = primary + self.distill_weight * float(self.distill_scale) * aux
            self.last_aux = {"distill": float(aux.detach()), "teacher": float(teacher.mean()), "cache_win": float((log_adv > 0).float().mean())}

        stats = None
        if compute_metrics:
            with torch.no_grad():
                param_loss = param_nll.mean()
                oracle_target = torch.maximum(param_target, target_cache.reshape(-1))
                oracle_target = torch.where(active, oracle_target, param_target)
                oracle_loss = -torch.log(oracle_target.clamp_min(1e-8)).mean()
                denom = float((param_loss - oracle_loss).clamp_min(1e-12))
                capture = float((param_loss - primary) / max(denom, 1e-12))
                cache_target_flat = target_cache.reshape(-1)
                cache_win = active & (cache_target_flat > param_target)
                cache_lose = active & ~cache_win
                gate_win = float(gate_flat[cache_win].mean()) if bool(cache_win.any()) else 0.0
                gate_lose = float(gate_flat[cache_lose].mean()) if bool(cache_lose.any()) else 0.0
                stats = CacheStats(
                    coverage=float(has.float().mean()),
                    gate=float(gate_flat[active].mean()) if bool(active.any()) else 0.0,
                    hit=float((cache_target_flat > 0).float().mean()),
                    cache_prob=float(target_cache.mean()),
                    param_bpb=float(param_loss / LN2),
                    oracle_bpb=float(oracle_loss / LN2),
                    capture=capture,
                    cache_win_rate=float(cache_win.float().sum() / active.float().sum().clamp_min(1.0)),
                    gate_when_cache_wins=gate_win,
                    gate_when_cache_loses=gate_lose,
                    gate_separation=gate_win - gate_lose,
                )
        return loss, primary.detach(), stats


# ======================================================================================
# Full model
# ======================================================================================


def resolve_arm_spec(arm: str, layers: int) -> dict:
    if arm not in ARMS:
        raise ValueError(arm)
    middle = max(0, layers // 2 - 1)
    late = max(0, layers - 1)
    spec = {
        "window": None,
        "positions": (),
        "softpatch": False,
        "patch_position": middle,
        "router_mode": "v5",
        "multiscale": False,
        "multiscale_positions": (),
        "attention_free": True,
    }
    if arm == "baseline_v5":
        return spec

    if "softpatch" in arm:
        spec["softpatch"] = True
    if "_conf" in arm:
        spec["router_mode"] = "confidence"

    if "local_w128" in arm:
        spec["window"] = 128
        spec["positions"] = (middle, late) if "_two_" in arm else (middle,)
        spec["attention_free"] = False
    elif "local_w256" in arm:
        spec["window"] = 256
        spec["positions"] = (middle,)
        spec["attention_free"] = False

    if "multiscale" in arm:
        spec["multiscale"] = True
        spec["multiscale_positions"] = (middle,)
        spec["attention_free"] = True
    return spec


class FieldPCAFLM(nn.Module):
    def __init__(self, arm: str, dim: int, layers: int, heads: int, ff_hidden: int,
                 field_chunk: int, num_buckets: int):
        super().__init__()
        self.arm = arm
        self.spec = resolve_arm_spec(arm, layers)
        self.emb = nn.Embedding(VOCAB, dim)
        # Common modules are constructed first so paired seeds preserve the
        # backbone/head initialization independently of side paths.
        self.blocks = nn.ModuleList([FieldBlock(dim, ff_hidden, field_chunk) for _ in range(layers)])
        self.final_norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, VOCAB, bias=False)
        self.cache = SuccessorCacheV5(
            dim, 64, num_buckets, 4, 4, router_mode=self.spec["router_mode"]
        )
        self.locals = nn.ModuleDict()
        if self.spec["window"] is not None:
            for pos in self.spec["positions"]:
                local_inner = min(128, max(16, dim))
                local_heads = 4 if local_inner % 4 == 0 else 1
                self.locals[str(pos)] = LowRankLocalAttention(
                    dim, inner=local_inner, heads=local_heads, window=int(self.spec["window"])
                )
        self.multiscales = nn.ModuleDict()
        if self.spec["multiscale"]:
            for pos in self.spec["multiscale_positions"]:
                self.multiscales[str(pos)] = MultiScaleCausalConv(dim, inner=min(128, max(16, dim)))
        self.softpatch = BoundaryStateMixer(dim, 64, learned=True) if self.spec["softpatch"] else None
        self._last_patch_aux = torch.tensor(0.0)

    @property
    def attention_free(self) -> bool:
        return bool(self.spec["attention_free"])

    def set_train_phase(self, phase: str) -> None:
        if phase != "normal":
            raise ValueError(f"v3 has no staged phase: {phase}")
        self.cache.enabled = True
        for p in self.parameters():
            p.requires_grad_(True)

    def states_logits(self, tokens: torch.Tensor):
        x = self.emb(tokens)
        self._last_patch_aux = x.new_zeros(())
        patch_pos = int(self.spec["patch_position"])
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == patch_pos and self.softpatch is not None:
                x = self.softpatch(x, tokens)
                self._last_patch_aux = self.softpatch.last_aux
            key = str(i)
            if key in self.locals:
                x = self.locals[key](x)
            if key in self.multiscales:
                x = self.multiscales[key](x)
        logits = self.lm_head(self.final_norm(x))
        return x, logits

    def loss_and_stats(self, tokens, targets, compute_metrics=False):
        states, logits = self.states_logits(tokens)
        loss, primary, stats = self.cache(states, logits, tokens, targets, compute_metrics)
        if self.training and self._last_patch_aux is not None:
            loss = loss + self._last_patch_aux
        return loss, primary, stats


# ======================================================================================
# Train / eval
# ======================================================================================


@dataclass
class TrainResult:
    arm: str
    seed: int
    params: int
    attention_free: bool
    target_steps: int
    final_step: int
    bpb_short: float
    bpb_8k: float
    bpb_16k: float
    param_bpb_8k: float
    oracle_bpb_8k: float
    capture_8k: float
    coverage_8k: float
    gate_8k: float
    hit_8k: float
    cache_win_rate_8k: float
    gate_win_8k: float
    gate_lose_8k: float
    gate_sep_8k: float
    bytes_per_second: float
    peak_gib: float
    checkpoint: str


@torch.no_grad()
def evaluate(model: FieldPCAFLM, data: torch.Tensor, device: torch.device, amp: str,
             seq: int, starts: Sequence[int], batch_size: int = 1):
    model.eval()
    losses, stats_rows = [], []
    for base in range(0, len(starts), batch_size):
        x, y = fixed_batch(data, starts[base : base + batch_size], seq, device)
        with amp_ctx(device, amp):
            loss, _, stats = model.loss_and_stats(x, y, compute_metrics=True)
        losses.append(float(loss))
        if stats is not None:
            stats_rows.append(asdict(stats))
        del x, y, loss
    model.train()
    mean = lambda k: float(np.mean([r[k] for r in stats_rows])) if stats_rows else 0.0
    return {
        "bpb": float(np.mean(losses) / LN2),
        "param_bpb": mean("param_bpb"),
        "oracle_bpb": mean("oracle_bpb"),
        "capture": mean("capture"),
        "coverage": mean("coverage"),
        "gate": mean("gate"),
        "hit": mean("hit"),
        "cache_win_rate": mean("cache_win_rate"),
        "gate_win": mean("gate_when_cache_wins"),
        "gate_lose": mean("gate_when_cache_loses"),
        "gate_sep": mean("gate_separation"),
    }


def parse_csv_strings(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_csv_ints(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def distill_scale_for(arm: str, step: int) -> float:
    if "_conf" in arm:
        return min(1.0, step / 100.0)
    if step <= 100:
        return 0.0
    return min(1.0, (step - 100) / 200.0)


def default_ff_hidden(dim: int) -> int:
    return ((int(8 * dim / 3) + 63) // 64) * 64


def ff_hidden_for_arm(arm: str, args) -> int:
    base_hidden = default_ff_hidden(args.dim)
    if arm != "softpatch_local_w128_conf_parity":
        return base_hidden
    # Pay for patch + local path + richer confidence cache by reducing FFN width.
    baseline_cache = nparams(SuccessorCacheV5(args.dim, 64, args.num_buckets, 4, 4, "v5"))
    conf_cache = nparams(SuccessorCacheV5(args.dim, 64, args.num_buckets, 4, 4, "confidence"))
    local = nparams(LowRankLocalAttention(args.dim, inner=min(128, args.dim), heads=4, window=128))
    patch = nparams(BoundaryStateMixer(args.dim, 64, learned=True))
    extra = (conf_cache - baseline_cache) + local + patch
    reduction = round(extra / max(1, 3 * args.dim * args.layers))
    return max(32, base_hidden - reduction)


def train_arm(arm: str, seed: int, target_steps: int, args, train, val, device, outroot: Path,
              schedule_steps: Optional[int] = None) -> dict:
    schedule_steps = int(schedule_steps or target_steps)
    run = outroot / f"{arm}_seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    result_path = run / f"result_step{target_steps}.json"
    latest = run / "latest.pt"
    if args.resume and result_path.exists():
        return json.loads(result_path.read_text())

    seed_all(args.model_seed + seed)
    ff_hidden = ff_hidden_for_arm(arm, args)
    model = FieldPCAFLM(arm, args.dim, args.layers, args.heads, ff_hidden,
                        args.field_chunk, args.num_buckets).to(device)
    params = nparams(model)
    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    gen_device = train.device.type if train.device.type == "cuda" else "cpu"
    gen = torch.Generator(device=gen_device).manual_seed(args.data_seed + seed)
    start = 0
    history: List[dict] = []
    if args.resume and latest.exists():
        st = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        gen.set_state(st["rng"].cpu())
        start = int(st["step"])
        history = list(st.get("history", []))
        log(f"[resume] {arm} seed={seed} {start}->{target_steps}")

    log("\n" + "=" * 160)
    log(f"TRAIN {arm} seed={seed} params={params:,} attention_free={model.attention_free} "
        f"steps={start}->{target_steps} schedule={schedule_steps} seq={args.seq_len} "
        f"batch={args.batch_size} accum={args.accum} ff={ff_hidden}")
    log("=" * 160)
    model.set_train_phase("normal")
    model.train()
    torch.cuda.empty_cache()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    wall = time.perf_counter()
    excluded = 0.0
    processed = 0

    for step in range(start + 1, target_steps + 1):
        model.cache.distill_scale = distill_scale_for(arm, step)
        lr = lr_at(step, schedule_steps, args.warmup, args.lr, args.min_lr_ratio)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        train_primary = 0.0
        for _ in range(args.accum):
            x, y = random_batch(train, args.batch_size, args.seq_len, gen, device)
            with amp_ctx(device, args.amp):
                loss, primary, _ = model.loss_and_stats(x, y, compute_metrics=False)
                scaled = loss / args.accum
            if not torch.isfinite(scaled):
                raise FloatingPointError((arm, step, float(scaled)))
            scaled.backward()
            train_primary += float(primary) / args.accum
            del x, y, loss, scaled
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
        optimizer.step()
        processed += args.batch_size * args.accum * args.seq_len

        if step == 1 or step % args.log_every == 0 or step == target_steps:
            sync(device)
            active = max(1e-9, time.perf_counter() - wall - excluded)
            bps = processed / active
            peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
            log(f"[{arm}] step={step:04d}/{target_steps} bpb={train_primary/LN2:.4f} "
                f"grad={grad:.3f} lr={lr:.3e} B/s={bps:,.0f} peak={peak:.2f}G")
            history.append({"step": step, "train_bpb": train_primary/LN2, "grad": grad,
                            "lr": lr, "bps": bps, "peak": peak})

        if step % args.eval_every == 0 or step == target_steps:
            t0 = time.perf_counter()
            ev = evaluate(model, val, device, args.amp, args.seq_len,
                          fixed_starts(len(val), args.seq_len, args.eval_windows, args.eval_seed), 1)
            excluded += time.perf_counter() - t0
            log(f"[{arm}] EVAL step={step:04d} bpb={ev['bpb']:.5f} param={ev['param_bpb']:.5f} "
                f"oracle={ev['oracle_bpb']:.5f} cap={ev['capture']:.3f} cov={ev['coverage']:.3f} "
                f"gate={ev['gate']:.3f} hit={ev['hit']:.3f} gwin={ev['gate_win']:.3f} "
                f"glose={ev['gate_lose']:.3f} sep={ev['gate_sep']:+.3f}")
            history.append({"step": step, "eval": ev})

        if step % args.save_every == 0 or step == target_steps:
            tmp = latest.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step, "history": history, "rng": gen.get_state().cpu()}, tmp)
            os.replace(tmp, latest)

    sync(device)
    active = max(1e-9, time.perf_counter() - wall - excluded)
    bps = processed / active
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
    ev_short = evaluate(model, val, device, args.amp, args.seq_len,
                        fixed_starts(len(val), args.seq_len, args.final_windows, args.eval_seed + 100), 1)
    ev8 = evaluate(model, val, device, args.amp, args.long_context,
                   fixed_starts(len(val), args.long_context, args.long_windows, args.eval_seed + 200), 1)
    ev16 = evaluate(model, val, device, args.amp, args.very_long_context,
                    fixed_starts(len(val), args.very_long_context, args.very_long_windows, args.eval_seed + 300), 1)
    result = asdict(TrainResult(
        arm=arm, seed=seed, params=params, attention_free=model.attention_free,
        target_steps=target_steps, final_step=target_steps,
        bpb_short=ev_short["bpb"], bpb_8k=ev8["bpb"], bpb_16k=ev16["bpb"],
        param_bpb_8k=ev8["param_bpb"], oracle_bpb_8k=ev8["oracle_bpb"],
        capture_8k=ev8["capture"], coverage_8k=ev8["coverage"], gate_8k=ev8["gate"],
        hit_8k=ev8["hit"], cache_win_rate_8k=ev8["cache_win_rate"],
        gate_win_8k=ev8["gate_win"], gate_lose_8k=ev8["gate_lose"],
        gate_sep_8k=ev8["gate_sep"], bytes_per_second=bps, peak_gib=peak,
        checkpoint=str(latest),
    ))
    result["history"] = history
    atomic_json(result_path, result)
    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_selftest(args, device: torch.device) -> None:
    log("[selftest] paired initialization, attention-free audit, causality, finite backward")
    ff = 32
    seed_all(77)
    a = FieldPCAFLM("baseline_v5", 16, 1, 4, ff, 8, 64).to(device)
    seed_all(77)
    b = FieldPCAFLM("softpatch_local_w128_conf", 16, 1, 4, ff, 8, 64).to(device)
    keys = [k for k in a.state_dict() if k.startswith(("emb.", "blocks.", "final_norm.", "lm_head."))]
    max_pair = max(float((a.state_dict()[k] - b.state_dict()[k]).abs().max()) for k in keys)
    log(f"[selftest] paired backbone max_abs={max_pair:.3e}")
    if max_pair != 0.0:
        raise AssertionError("paired initialization failed")

    x = torch.randint(0, VOCAB, (1, 17), device=device)
    y = torch.randint(0, VOCAB, (1, 17), device=device)
    idx = causal_recent_candidates(x, 4, 64, 4)
    q = torch.arange(x.shape[1], device=device)[None, :, None]
    if bool(((idx >= q) & (idx >= 0)).any()):
        raise AssertionError("future/self candidate")
    log("[selftest] cache candidate causality PASS")

    reps = (
        "baseline_v5", "softpatch_conf", "local_w128_conf",
        "softpatch_local_w128_two_conf", "multiscale_conv_conf",
        "softpatch_multiscale_conf",
    )
    for arm in reps:
        seed_all(91)
        m = FieldPCAFLM(arm, 16, 1, 4, ff, 8, 64).to(device)
        with amp_ctx(device, args.amp):
            loss, primary, _ = m.loss_and_stats(x, y, False)
        loss.backward()
        finite = bool(torch.isfinite(loss)) and all(
            p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters()
        )
        log(f"[selftest] {arm:<42} attention_free={m.attention_free} "
            f"loss={float(loss.detach()):.5f} primary={float(primary.detach()):.5f} finite={finite}")
        if not finite:
            raise AssertionError(arm)
        del m

    # Strict prefix check on both hybrid and attention-free finalists.
    for arm in ("softpatch_local_w128_conf", "softpatch_multiscale_conf"):
        seed_all(123)
        m = FieldPCAFLM(arm, 16, 1, 4, ff, 8, 64).to(device).eval()
        p = 8
        x2 = x.clone(); x2[:, p:] = torch.randint(0, VOCAB, x2[:, p:].shape, device=device)
        with torch.no_grad(), amp_ctx(device, args.amp):
            z1 = m.states_logits(x)[1][:, :p]
            z2 = m.states_logits(x2)[1][:, :p]
        err = float((z1 - z2).abs().max())
        log(f"[selftest] {arm} strict prefix max_abs={err:.3e}")
        if err > 2e-4:
            raise AssertionError((arm, "prefix causality"))
    log("[selftest] PASS")


def rank_rows(rows: List[dict], baseline_bps: float, min_speed_ratio: float) -> List[dict]:
    valid = [r for r in rows if r["bytes_per_second"] >= baseline_bps * min_speed_ratio]
    return sorted(valid, key=lambda r: (r["bpb_8k"], r["bpb_16k"], r["bpb_short"]))


def seed_aggregate(stage_c: List[dict], arms: Sequence[str]) -> List[str]:
    lines = []
    for arm in arms:
        rows = [r for r in stage_c if r["arm"] == arm]
        if not rows:
            continue
        b8 = np.asarray([r["bpb_8k"] for r in rows], dtype=np.float64)
        b16 = np.asarray([r["bpb_16k"] for r in rows], dtype=np.float64)
        lines.append(f"{arm:<42} n={len(rows)} BPB8K={b8.mean():.5f}±{b8.std(ddof=0):.5f} "
                     f"BPB16K={b16.mean():.5f}±{b16.std(ddof=0):.5f}")
    return lines


def make_summary(stage_a: List[dict], stage_b: List[dict], stage_c: List[dict], args) -> str:
    base_a = next(r for r in stage_a if r["arm"] == "baseline_v5")
    lines = [
        "FIELD HYBRID / ATTENTION-FREE QUALIFICATION v3",
        "=" * 235,
        f"Protocol: WikiText-103 {args.data_frac:.1%} | train ctx={args.seq_len} | "
        f"eval={args.long_context}/{args.very_long_context} | bytes/update={args.batch_size*args.accum*args.seq_len:,}",
        "Portable exact Field reference; final production claims require the Triton transplant.",
        "AF=yes means no query-key attention path (Field/PCAF/softpatch/causal convolution only).",
        "",
        "STAGE A — PAIRED QUALIFICATION SCREEN",
        f"{'arm':<42} {'AF':>3} {'params':>12} {'d%':>7} {('BPB'+str(args.seq_len)):>9} {'BPB8K':>9} {'d8K':>9} {'BPB16K':>9} "
        f"{'oracle':>9} {'cap':>7} {'sep':>7} {'B/s':>11} {'speed':>7} {'peak':>7}",
    ]
    for r in sorted(stage_a, key=lambda x: x["bpb_8k"]):
        dp = 100 * (r["params"] - base_a["params"]) / base_a["params"]
        lines.append(
            f"{r['arm']:<42} {('yes' if r['attention_free'] else 'no'):>3} {r['params']:>12,d} {dp:>+7.2f} "
            f"{r['bpb_short']:>9.5f} {r['bpb_8k']:>9.5f} {r['bpb_8k']-base_a['bpb_8k']:>+9.5f} "
            f"{r['bpb_16k']:>9.5f} {r['oracle_bpb_8k']:>9.5f} {r['capture_8k']:>7.3f} "
            f"{r['gate_sep_8k']:>+7.3f} {r['bytes_per_second']:>11,.0f} "
            f"{r['bytes_per_second']/base_a['bytes_per_second']:>7.2f} {r['peak_gib']:>7.2f}"
        )

    pool = stage_a
    if stage_b:
        base_b = next(r for r in stage_b if r["arm"] == "baseline_v5")
        lines += [
            "", f"STAGE B — CONTINUED TO {args.stage_b_epochs:g} EPOCHS",
            f"{'arm':<42} {'AF':>3} {('BPB'+str(args.seq_len)):>9} {'BPB8K':>9} {'d8K':>9} {'BPB16K':>9} "
            f"{'oracle':>9} {'cap':>7} {'gate':>7} {'sep':>7} {'B/s':>11} {'speed':>7}",
        ]
        for r in sorted(stage_b, key=lambda x: x["bpb_8k"]):
            lines.append(
                f"{r['arm']:<42} {('yes' if r['attention_free'] else 'no'):>3} {r['bpb_short']:>9.5f} "
                f"{r['bpb_8k']:>9.5f} {r['bpb_8k']-base_b['bpb_8k']:>+9.5f} {r['bpb_16k']:>9.5f} "
                f"{r['oracle_bpb_8k']:>9.5f} {r['capture_8k']:>7.3f} {r['gate_8k']:>7.3f} "
                f"{r['gate_sep_8k']:>+7.3f} {r['bytes_per_second']:>11,.0f} "
                f"{r['bytes_per_second']/base_b['bytes_per_second']:>7.2f}"
            )
        pool = stage_b

    if stage_c:
        lines += [
            "", "STAGE C — EXTRA-SEED CONFIRMATION",
            f"{'arm':<42} {'seed':>6} {'BPB8K':>9} {'d8K':>9} {'BPB16K':>9} {'cap':>7} {'sep':>7} {'B/s':>11}",
        ]
        # Compare each confirmation seed to its same-seed baseline.
        base_by_seed = {r["seed"]: r for r in stage_c if r["arm"] == "baseline_v5"}
        for r in sorted(stage_c, key=lambda x: (x["seed"], x["arm"])):
            base = base_by_seed.get(r["seed"], r)
            lines.append(
                f"{r['arm']:<42} {r['seed']:>6} {r['bpb_8k']:>9.5f} "
                f"{r['bpb_8k']-base['bpb_8k']:>+9.5f} {r['bpb_16k']:>9.5f} "
                f"{r['capture_8k']:>7.3f} {r['gate_sep_8k']:>+7.3f} {r['bytes_per_second']:>11,.0f}"
            )
        lines += ["", "SEED AGGREGATES"]
        lines.extend(seed_aggregate(stage_c, sorted({r["arm"] for r in stage_c})))

    base_pool = next(r for r in pool if r["arm"] == "baseline_v5")
    winner = min(pool, key=lambda r: r["bpb_8k"])
    af_rows = [r for r in pool if r["attention_free"] and r["arm"] != "baseline_v5"]
    best_af = min(af_rows, key=lambda r: r["bpb_8k"]) if af_rows else None
    parity = next((r for r in pool if r["arm"] == "softpatch_local_w128_conf_parity"), None)
    lines += ["", "DECISION SUMMARY"]
    lines.append(f"overall winner: {winner['arm']} | d8K={winner['bpb_8k']-base_pool['bpb_8k']:+.5f} | "
                 f"d16K={winner['bpb_16k']-base_pool['bpb_16k']:+.5f} | speed={winner['bytes_per_second']/base_pool['bytes_per_second']:.2f}x")
    if best_af:
        lines.append(f"best strictly attention-free: {best_af['arm']} | d8K={best_af['bpb_8k']-base_pool['bpb_8k']:+.5f} | "
                     f"d16K={best_af['bpb_16k']-base_pool['bpb_16k']:+.5f} | speed={best_af['bytes_per_second']/base_pool['bytes_per_second']:.2f}x")
    if parity:
        lines.append(f"strict-param hybrid control: params={parity['params']:,} | dparams={100*(parity['params']-base_pool['params'])/base_pool['params']:+.3f}% | "
                     f"d8K={parity['bpb_8k']-base_pool['bpb_8k']:+.5f}")
    lines += [
        "", "Promotion gate before 300M:",
        "- hybrid: mean gain >=0.040 BPB across seeds, 16K gain preserved, speed >=0.80x;",
        "- attention-free branch: gain >=0.020 BPB, 16K gain preserved, speed >=0.85x;",
        "- strict-param control must retain at least 70% of the hybrid gain;",
        "- then transplant into canonical Triton Field-PCAF and run an intermediate 50M/10% systems check.",
        "=" * 235,
    ]
    return "\n".join(lines) + "\n"


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=("all", "selftest", "screen", "summary"), default="all")
    p.add_argument("--outdir", default="./field_hybrid_attentionfree_qualification_v3")
    p.add_argument("--cache-dir", default="./hf_cache")
    p.add_argument("--data-frac", type=float, default=0.05)
    p.add_argument("--data-device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--long-context", type=int, default=8192)
    p.add_argument("--very-long-context", type=int, default=16384)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--field-chunk", type=int, default=32)
    p.add_argument("--num-buckets", type=int, default=8192)
    p.add_argument("--amp", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--min-lr-ratio", type=float, default=0.10)
    p.add_argument("--weight-decay", type=float, default=0.10)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--stage-a-epochs", type=float, default=1.0)
    p.add_argument("--stage-b-epochs", type=float, default=2.0)
    p.add_argument("--finalists", type=int, default=4)
    p.add_argument("--confirm-finalists", type=int, default=2)
    p.add_argument("--min-speed-ratio", type=float, default=0.70)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-windows", type=int, default=4)
    p.add_argument("--final-windows", type=int, default=8)
    p.add_argument("--long-windows", type=int, default=4)
    p.add_argument("--very-long-windows", type=int, default=2)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--model-seed", type=int, default=1234)
    p.add_argument("--data-seed", type=int, default=5678)
    p.add_argument("--eval-seed", type=int, default=9012)
    p.add_argument("--screen-seed", type=int, default=0)
    p.add_argument("--confirm-seeds", default="1000,2000")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                    "args": vars(args)}, indent=2))
    if device.type == "cuda" and args.amp == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unavailable")

    selected_arms = parse_csv_strings(args.arms)
    unknown = [a for a in selected_arms if a not in ARMS]
    if unknown:
        raise ValueError(f"unknown arms: {unknown}")
    if "baseline_v5" not in selected_arms:
        selected_arms.insert(0, "baseline_v5")

    if args.mode in {"all", "selftest"}:
        run_selftest(args, device)
        if args.mode == "selftest":
            return

    if args.mode == "summary":
        stage_a = json.loads((out / "stage_a.json").read_text())
        stage_b = json.loads((out / "stage_b.json").read_text()) if (out / "stage_b.json").exists() else []
        stage_c = json.loads((out / "stage_c.json").read_text()) if (out / "stage_c.json").exists() else []
        summary = make_summary(stage_a, stage_b, stage_c, args)
        atomic_text(out / "SUMMARY.txt", summary); log(summary); return

    train, val, _ = load_wikitext103_raw(args.cache_dir, args.data_frac)
    train = place_data(train, device, args.data_device, "train")
    val = place_data(val, device, args.data_device, "validation")
    bytes_per_update = args.batch_size * args.accum * args.seq_len
    stage_a_steps = max(1, math.ceil(len(train) * args.stage_a_epochs / bytes_per_update))
    stage_b_steps = max(stage_a_steps, math.ceil(len(train) * args.stage_b_epochs / bytes_per_update))
    log(f"[protocol] arms={selected_arms} bytes/update={bytes_per_update:,} "
        f"stageA={stage_a_steps} stageB={stage_b_steps}")

    stage_a_path = out / "stage_a.json"
    if stage_a_path.exists() and args.resume:
        stage_a = json.loads(stage_a_path.read_text())
    else:
        stage_a = [train_arm(arm, args.screen_seed, stage_a_steps, args, train, val, device,
                             out / "runs", schedule_steps=stage_b_steps) for arm in selected_arms]
        atomic_json(stage_a_path, stage_a)

    if args.mode == "screen":
        summary = make_summary(stage_a, [], [], args)
        atomic_text(out / "SUMMARY.txt", summary); log("\n" + summary)
        return

    base = next(r for r in stage_a if r["arm"] == "baseline_v5")
    ranked = rank_rows(stage_a, base["bytes_per_second"], args.min_speed_ratio)
    finalists = [r["arm"] for r in ranked if r["arm"] != "baseline_v5"][:args.finalists]
    log(f"[selection] finalists={finalists}")

    stage_b_path = out / "stage_b.json"
    if stage_b_path.exists() and args.resume:
        stage_b = json.loads(stage_b_path.read_text())
    else:
        stage_b = [train_arm(arm, args.screen_seed, stage_b_steps, args, train, val, device,
                             out / "runs", schedule_steps=stage_b_steps)
                   for arm in ["baseline_v5", *finalists]]
        atomic_json(stage_b_path, stage_b)

    ranked_b = sorted(stage_b, key=lambda r: (r["bpb_8k"], r["bpb_16k"]))
    confirm = [r["arm"] for r in ranked_b if r["arm"] != "baseline_v5"][:args.confirm_finalists]
    confirm_seeds = parse_csv_ints(args.confirm_seeds)
    stage_c_path = out / "stage_c.json"
    if stage_c_path.exists() and args.resume:
        stage_c = json.loads(stage_c_path.read_text())
    else:
        stage_c = []
        for seed in confirm_seeds:
            for arm in ["baseline_v5", *confirm]:
                stage_c.append(train_arm(arm, seed, stage_a_steps, args, train, val, device,
                                         out / "confirm_runs", schedule_steps=stage_a_steps))
        atomic_json(stage_c_path, stage_c)

    summary = make_summary(stage_a, stage_b, stage_c, args)
    atomic_text(out / "SUMMARY.txt", summary)
    log("\n" + summary)

    rows = []
    for stage_name, stage in (("A", stage_a), ("B", stage_b), ("C", stage_c)):
        for r in stage:
            rows.append({"stage": stage_name, **{k: v for k, v in r.items() if k != "history"}})
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as f:
        fields = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
