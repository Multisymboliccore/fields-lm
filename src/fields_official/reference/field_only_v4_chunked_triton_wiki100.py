#!/usr/bin/env python3
"""Field-Only v4: chunk-parallel Triton recurrence + WikiText-103 100% arena.

Purpose
-------
Optimize the validated zero-attention Field architecture without changing its
mathematics, then compare it against:

  * field_reference : original exact hierarchical PyTorch/complex scan;
  * field_triton    : same parameters/equations, chunk-parallel Triton forward+backward;
  * transformer     : parameter-matched RoPE + PyTorch SDPA + SwiGLU control.

The discarded competitive-vacancy arm is intentionally absent. Both Field arms
use independent vacancy and displaced readout, with zero attention.

Default full run
----------------
* WikiText-103 raw-byte, 100% of the training split;
* one data epoch (automatic step count);
* dim=192, layers=6, train context=1024;
* effective 32,768 bytes/update (batch 16, accumulation 2);
* no activation checkpointing by default;
* exact same initialization and sampled windows for reference/Triton;
* preflight numerical/gradient equivalence test;
* training systems benchmark at contexts 512/1024/2048/4096;
* resumable checkpoints, curves, CSV/JSON and full text log.

The Triton implementation parallelizes time by affine chunks and fuses:
  raw projection nonlinearities -> independent vacancy -> complex recurrence
  -> displaced readout

Its custom reverse kernel computes the exact recurrent backward pass. Linear
layers, RMSNorm, SwiGLU and optimizer remain standard PyTorch operations.
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
from torch.utils.checkpoint import checkpoint

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
    TRITON_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - exercised on CPU-only machines.
    triton = None
    tl = None
    HAS_TRITON = False
    TRITON_IMPORT_ERROR = repr(exc)

LN2 = math.log(2.0)
BYTE_VOCAB = 256
VAC_MAX = 0.90
MODEL_NAMES = ("field_reference", "field_triton", "transformer")
_OFFSET_CACHE: Dict[Tuple[str, int], torch.Tensor] = {}


# =================================================================================================
# Utilities
# =================================================================================================


def log(message: str = "") -> None:
    print(message, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def nparams(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def choose_amp(requested: str) -> str:
    if requested != "auto":
        return requested
    if not torch.cuda.is_available():
        return "fp32"
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"


def amp_dtype(amp: str) -> torch.dtype:
    if amp == "bf16":
        return torch.bfloat16
    if amp == "fp16":
        return torch.float16
    return torch.float32


def autocast_context(device: torch.device, amp: str):
    enabled = device.type == "cuda" and amp in ("bf16", "fp16")
    return torch.autocast(
        device_type=device.type,
        dtype=amp_dtype(amp),
        enabled=enabled,
    )


def checkpoint_block(module: nn.Module, x: torch.Tensor, enabled: bool) -> torch.Tensor:
    if enabled and module.training and torch.is_grad_enabled():
        return checkpoint(
            module,
            x,
            use_reentrant=False,
            preserve_rng_state=False,
            determinism_check="none",
        )
    return module(x)


# =================================================================================================
# Data
# =================================================================================================


def _join_text_rows(rows: Iterable[Dict[str, str]]) -> bytes:
    parts: List[bytes] = []
    for row in rows:
        text = row.get("text", "")
        if text:
            parts.append(text.encode("utf-8", errors="replace"))
            parts.append(b"\n")
    return b"".join(parts)


def load_wikitext103_raw(
    cache_dir: str,
    data_frac: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pip install datasets") from exc

    log("[data] loading Salesforce/wikitext, wikitext-103-raw-v1")
    ds = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        cache_dir=cache_dir,
    )
    train_raw = _join_text_rows(ds["train"])
    val_raw = _join_text_rows(ds["validation"])
    test_raw = _join_text_rows(ds["test"])
    if not 0.0 < data_frac <= 1.0:
        raise ValueError("data_frac must be in (0, 1]")
    train_raw = train_raw[: max(2, int(len(train_raw) * data_frac))]

    def as_u8(raw: bytes) -> torch.Tensor:
        array = np.frombuffer(raw, dtype=np.uint8).copy()
        return torch.from_numpy(array)

    train, val, test = map(as_u8, (train_raw, val_raw, test_raw))
    log(
        f"[data] train={len(train):,} bytes ({data_frac:.1%}) "
        f"val={len(val):,} test={len(test):,}"
    )
    return train, val, test


def place_dataset(
    tensor: torch.Tensor,
    target: str,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if target == "cpu" or device.type != "cuda":
        result = tensor.contiguous()
        try:
            result = result.pin_memory()
        except RuntimeError:
            pass
        log(f"[data] {name} kept on CPU as uint8 ({result.numel()/2**20:.1f} MiB)")
        return result

    if target not in ("cuda", "auto"):
        raise ValueError(target)

    if target == "auto":
        total_gib = torch.cuda.get_device_properties(device).total_memory / 2**30
        if total_gib < 40:
            result = tensor.contiguous()
            try:
                result = result.pin_memory()
            except RuntimeError:
                pass
            log(
                f"[data] {name} kept on CPU: GPU has {total_gib:.1f} GiB, "
                "auto threshold is 40 GiB"
            )
            return result

    result = tensor.to(device=device, dtype=torch.uint8)
    log(f"[data] {name} moved to GPU as uint8 ({result.numel()/2**20:.1f} MiB)")
    return result


def generator_for(data: torch.Tensor, seed: int) -> torch.Generator:
    device_name = data.device.type if data.device.type == "cuda" else "cpu"
    generator = torch.Generator(device=device_name)
    generator.manual_seed(seed)
    return generator


def offsets_for(data: torch.Tensor, seq_len: int) -> torch.Tensor:
    key = (str(data.device), seq_len)
    cached = _OFFSET_CACHE.get(key)
    if cached is None or cached.device != data.device:
        cached = torch.arange(seq_len + 1, device=data.device, dtype=torch.long)
        _OFFSET_CACHE[key] = cached
    return cached


def random_byte_batch(
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    generator: torch.Generator,
    model_device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= seq_len + 1:
        raise ValueError(f"Dataset has {len(data)} bytes, needs > {seq_len + 1}")
    starts = torch.randint(
        0,
        len(data) - seq_len - 1,
        (batch_size,),
        generator=generator,
        device=data.device,
    )
    offsets = offsets_for(data, seq_len)
    windows = data[starts[:, None] + offsets[None, :]].long()
    if windows.device != model_device:
        windows = windows.to(model_device, non_blocking=True)
    return windows[:, :-1], windows[:, 1:]


def fixed_eval_starts(
    data_len: int,
    context: int,
    windows: int,
    seed: int,
) -> List[int]:
    if data_len <= context + 1:
        raise ValueError(f"Validation set too short for context {context}")
    rng = np.random.default_rng(seed + context * 1009)
    return rng.integers(0, data_len - context - 1, size=windows).tolist()


def fixed_eval_batch(
    data: torch.Tensor,
    starts: Sequence[int],
    context: int,
    model_device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    windows = torch.stack([data[s : s + context + 1] for s in starts]).long()
    if windows.device != model_device:
        windows = windows.to(model_device, non_blocking=True)
    return windows[:, :-1], windows[:, 1:]


# =================================================================================================
# Shared blocks
# =================================================================================================


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
        self.hidden = (
            int(hidden)
            if hidden is not None
            else ((int(8 * dim / 3) + 63) // 64) * 64
        )
        self.w12 = nn.Linear(dim, 2 * self.hidden, bias=False)
        self.w3 = nn.Linear(self.hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(gate) * value)


# =================================================================================================
# Reference exact scan
# =================================================================================================


def assoc_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Inclusive associative scan for s_t = a_t*s_(t-1)+b_t, s_-1=0."""
    a = a.clone()
    b = b.clone()
    shift = 1
    while shift < a.shape[1]:
        a_prev = torch.cat((torch.ones_like(a[:, :shift]), a[:, :-shift]), dim=1)
        b_prev = torch.cat((torch.zeros_like(b[:, :shift]), b[:, :-shift]), dim=1)
        b = a * b_prev + b
        a = a * a_prev
        shift *= 2
    return b


def hierarchical_scan(
    a: torch.Tensor,
    b: torch.Tensor,
    block: int = 32,
) -> torch.Tensor:
    """Numerically stable exact chunked affine scan without prefix division."""
    if block < 1 or block > 64:
        raise ValueError("field_chunk must be in [1, 64]")
    batch, length, channels = a.shape
    pad = (-length) % block
    if pad:
        a = torch.cat((a, a.new_ones(batch, pad, channels)), dim=1)
        b = torch.cat((b, b.new_zeros(batch, pad, channels)), dim=1)

    padded = a.shape[1]
    groups = padded // block
    ac = a.reshape(batch, groups, block, channels)
    bc = b.reshape(batch, groups, block, channels)
    local_a = ac.reshape(batch * groups, block, channels).clone()
    local_b = bc.reshape(batch * groups, block, channels).clone()

    shift = 1
    while shift < block:
        a_prev = torch.cat(
            (torch.ones_like(local_a[:, :shift]), local_a[:, :-shift]), dim=1
        )
        b_prev = torch.cat(
            (torch.zeros_like(local_b[:, :shift]), local_b[:, :-shift]), dim=1
        )
        local_b = local_a * b_prev + local_b
        local_a = local_a * a_prev
        shift *= 2

    local_a = local_a.reshape(batch, groups, block, channels)
    local_b = local_b.reshape(batch, groups, block, channels)
    block_a = local_a[:, :, -1]
    block_b = local_b[:, :, -1]
    carry_out = assoc_scan(block_a, block_b)
    carry_in = torch.cat(
        (carry_out.new_zeros(batch, 1, channels), carry_out[:, :-1]), dim=1
    )
    states = local_a * carry_in[:, :, None, :] + local_b
    return states.reshape(batch, padded, channels)[:, :length]


def reference_field_read(
    raw: torch.Tensor,
    transition_r: torch.Tensor,
    transition_i: torch.Tensor,
    gamma: torch.Tensor,
    field_chunk: int,
) -> torch.Tensor:
    channels = transition_r.numel()
    inj_r = torch.tanh(raw[..., :channels])
    inj_i = torch.tanh(raw[..., channels : 2 * channels])
    vacancy = torch.sigmoid(raw[..., 2 * channels :]) * VAC_MAX
    injection = torch.complex(inj_r, inj_i)
    transition = torch.complex(transition_r, transition_i)
    a = (1.0 - vacancy).to(torch.complex64) * transition
    b = gamma.to(torch.complex64) * vacancy.to(torch.complex64) * injection
    states = hierarchical_scan(a, b, field_chunk)
    previous = torch.cat(
        (torch.zeros_like(states[:, :1]), states[:, :-1]), dim=1
    )
    moved = transition * previous
    displaced = vacancy.to(torch.complex64) * moved
    return torch.cat(
        (states.real, states.imag, displaced.real, displaced.imag),
        dim=-1,
    )



# =================================================================================================
# Chunk-parallel Triton recurrence + custom backward
# =================================================================================================
#
# The v3 kernel assigned one program to an entire (batch, channel-block) sequence.
# That was efficient for short sequences, but left an H100 badly under-occupied as
# context grew. v4 splits time into independent chunks:
#
#   1) summarize every chunk as s_out = A_chunk*s_in + B_chunk;
#   2) scan the much shorter list of chunk summaries to obtain each carry-in;
#   3) materialize all chunks in parallel.
#
# Backward uses the dual affine recurrence:
#
#   g_before = P_chunk*g_after + Q_chunk
#
# followed by a reverse chunk scan and a parallel materialization pass. The
# equations, parameters, vacancy and displaced readout are unchanged.
# =================================================================================================


if HAS_TRITON:

    @triton.jit
    def _field_chunk_summary_kernel(
        raw_ptr,
        tr_ptr,
        ti_ptr,
        gamma_ptr,
        chunk_ar_ptr,
        chunk_ai_ptr,
        chunk_br_ptr,
        chunk_bi_ptr,
        n_ctx,
        n_chunks,
        channels,
        VACANCY_MAX: tl.constexpr,
        CHUNK_T: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_chunk = tl.program_id(1)
        pid_cb = tl.program_id(2)
        offs = pid_cb * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs < channels

        tr = tl.load(tr_ptr + offs, mask=mask_c, other=0.0).to(tl.float32)
        ti = tl.load(ti_ptr + offs, mask=mask_c, other=0.0).to(tl.float32)
        gm = tl.load(gamma_ptr + offs, mask=mask_c, other=0.0).to(tl.float32)

        # Affine identity: state_out = A*state_in + B.
        sum_ar = tl.full([BLOCK_C], 1.0, tl.float32)
        sum_ai = tl.zeros([BLOCK_C], tl.float32)
        sum_br = tl.zeros([BLOCK_C], tl.float32)
        sum_bi = tl.zeros([BLOCK_C], tl.float32)

        local_t = 0
        while local_t < CHUNK_T:
            t = pid_chunk * CHUNK_T + local_t
            valid = mask_c & (t < n_ctx)
            raw_base = (pid_b * n_ctx + t) * (3 * channels)
            zr = tl.load(raw_ptr + raw_base + offs, mask=valid, other=0.0).to(tl.float32)
            zi = tl.load(
                raw_ptr + raw_base + channels + offs,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            zv = tl.load(
                raw_ptr + raw_base + 2 * channels + offs,
                mask=valid,
                other=0.0,
            ).to(tl.float32)

            inj_r = 2.0 * tl.sigmoid(2.0 * zr) - 1.0
            inj_i = 2.0 * tl.sigmoid(2.0 * zi) - 1.0
            vacancy = VACANCY_MAX * tl.sigmoid(zv)
            keep = 1.0 - vacancy
            a_r = tl.where(valid, keep * tr, 1.0)
            a_i = tl.where(valid, keep * ti, 0.0)
            b_scale = gm * vacancy
            b_r = tl.where(valid, b_scale * inj_r, 0.0)
            b_i = tl.where(valid, b_scale * inj_i, 0.0)

            # New summary is current transition composed after prior summary.
            next_br = a_r * sum_br - a_i * sum_bi + b_r
            next_bi = a_r * sum_bi + a_i * sum_br + b_i
            next_ar = a_r * sum_ar - a_i * sum_ai
            next_ai = a_r * sum_ai + a_i * sum_ar
            sum_ar, sum_ai = next_ar, next_ai
            sum_br, sum_bi = next_br, next_bi
            local_t += 1

        base = (pid_b * n_chunks + pid_chunk) * channels + offs
        tl.store(chunk_ar_ptr + base, sum_ar, mask=mask_c)
        tl.store(chunk_ai_ptr + base, sum_ai, mask=mask_c)
        tl.store(chunk_br_ptr + base, sum_br, mask=mask_c)
        tl.store(chunk_bi_ptr + base, sum_bi, mask=mask_c)


    @triton.jit
    def _field_chunk_carry_kernel(
        chunk_ar_ptr,
        chunk_ai_ptr,
        chunk_br_ptr,
        chunk_bi_ptr,
        carry_r_ptr,
        carry_i_ptr,
        n_chunks,
        channels,
        BLOCK_C: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_cb = tl.program_id(1)
        offs = pid_cb * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs < channels
        state_r = tl.zeros([BLOCK_C], tl.float32)
        state_i = tl.zeros([BLOCK_C], tl.float32)

        chunk = 0
        while chunk < n_chunks:
            base = (pid_b * n_chunks + chunk) * channels + offs
            tl.store(carry_r_ptr + base, state_r, mask=mask_c)
            tl.store(carry_i_ptr + base, state_i, mask=mask_c)
            ar = tl.load(chunk_ar_ptr + base, mask=mask_c, other=1.0).to(tl.float32)
            ai = tl.load(chunk_ai_ptr + base, mask=mask_c, other=0.0).to(tl.float32)
            br = tl.load(chunk_br_ptr + base, mask=mask_c, other=0.0).to(tl.float32)
            bi = tl.load(chunk_bi_ptr + base, mask=mask_c, other=0.0).to(tl.float32)
            next_r = ar * state_r - ai * state_i + br
            next_i = ar * state_i + ai * state_r + bi
            state_r, state_i = next_r, next_i
            chunk += 1


    @triton.jit
    def _field_chunk_materialize_kernel(
        raw_ptr,
        tr_ptr,
        ti_ptr,
        gamma_ptr,
        carry_r_ptr,
        carry_i_ptr,
        read_ptr,
        n_ctx,
        n_chunks,
        channels,
        VACANCY_MAX: tl.constexpr,
        CHUNK_T: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_chunk = tl.program_id(1)
        pid_cb = tl.program_id(2)
        offs = pid_cb * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs < channels

        tr = tl.load(tr_ptr + offs, mask=mask_c, other=0.0).to(tl.float32)
        ti = tl.load(ti_ptr + offs, mask=mask_c, other=0.0).to(tl.float32)
        gm = tl.load(gamma_ptr + offs, mask=mask_c, other=0.0).to(tl.float32)
        carry_base = (pid_b * n_chunks + pid_chunk) * channels + offs
        state_r = tl.load(carry_r_ptr + carry_base, mask=mask_c, other=0.0).to(tl.float32)
        state_i = tl.load(carry_i_ptr + carry_base, mask=mask_c, other=0.0).to(tl.float32)

        local_t = 0
        while local_t < CHUNK_T:
            t = pid_chunk * CHUNK_T + local_t
            valid = mask_c & (t < n_ctx)
            raw_base = (pid_b * n_ctx + t) * (3 * channels)
            zr = tl.load(raw_ptr + raw_base + offs, mask=valid, other=0.0).to(tl.float32)
            zi = tl.load(
                raw_ptr + raw_base + channels + offs,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            zv = tl.load(
                raw_ptr + raw_base + 2 * channels + offs,
                mask=valid,
                other=0.0,
            ).to(tl.float32)

            inj_r = 2.0 * tl.sigmoid(2.0 * zr) - 1.0
            inj_i = 2.0 * tl.sigmoid(2.0 * zi) - 1.0
            vacancy = VACANCY_MAX * tl.sigmoid(zv)
            keep = 1.0 - vacancy

            prev_r = state_r
            prev_i = state_i
            moved_r = tr * prev_r - ti * prev_i
            moved_i = tr * prev_i + ti * prev_r
            disp_r = vacancy * moved_r
            disp_i = vacancy * moved_i

            a_r = keep * tr
            a_i = keep * ti
            b_scale = gm * vacancy
            state_r = a_r * prev_r - a_i * prev_i + b_scale * inj_r
            state_i = a_r * prev_i + a_i * prev_r + b_scale * inj_i

            out_base = (pid_b * n_ctx + t) * (4 * channels)
            tl.store(read_ptr + out_base + offs, state_r, mask=valid)
            tl.store(read_ptr + out_base + channels + offs, state_i, mask=valid)
            tl.store(read_ptr + out_base + 2 * channels + offs, disp_r, mask=valid)
            tl.store(read_ptr + out_base + 3 * channels + offs, disp_i, mask=valid)
            local_t += 1


    @triton.jit
    def _field_backward_chunk_summary_kernel(
        raw_ptr,
        tr_ptr,
        ti_ptr,
        grad_read_ptr,
        chunk_pr_ptr,
        chunk_pi_ptr,
        chunk_qr_ptr,
        chunk_qi_ptr,
        n_ctx,
        n_chunks,
        channels,
        VACANCY_MAX: tl.constexpr,
        CHUNK_T: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_chunk = tl.program_id(1)
        pid_cb = tl.program_id(2)
        offs = pid_cb * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs < channels
        tr = tl.load(tr_ptr + offs, mask=mask_c, other=0.0).to(tl.float32)
        ti = tl.load(ti_ptr + offs, mask=mask_c, other=0.0).to(tl.float32)

        # carry_before = P*carry_after + Q.
        sum_pr = tl.full([BLOCK_C], 1.0, tl.float32)
        sum_pi = tl.zeros([BLOCK_C], tl.float32)
        sum_qr = tl.zeros([BLOCK_C], tl.float32)
        sum_qi = tl.zeros([BLOCK_C], tl.float32)

        local_t = CHUNK_T
        while local_t > 0:
            local_t -= 1
            t = pid_chunk * CHUNK_T + local_t
            valid = mask_c & (t < n_ctx)
            raw_base = (pid_b * n_ctx + t) * (3 * channels)
            zv = tl.load(
                raw_ptr + raw_base + 2 * channels + offs,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            vacancy = VACANCY_MAX * tl.sigmoid(zv)
            keep = 1.0 - vacancy
            a_r = tl.where(valid, keep * tr, 1.0)
            a_i = tl.where(valid, keep * ti, 0.0)

            out_base = (pid_b * n_ctx + t) * (4 * channels)
            u_r = tl.load(
                grad_read_ptr + out_base + offs,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            u_i = tl.load(
                grad_read_ptr + out_base + channels + offs,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            gd_r = tl.load(
                grad_read_ptr + out_base + 2 * channels + offs,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            gd_i = tl.load(
                grad_read_ptr + out_base + 3 * channels + offs,
                mask=valid,
                other=0.0,
            ).to(tl.float32)

            direct_r = vacancy * (tr * gd_r + ti * gd_i)
            direct_i = vacancy * (-ti * gd_r + tr * gd_i)
            # p = conj(a), q = conj(a)*direct_state_grad + displaced contribution.
            q_r = a_r * u_r + a_i * u_i + direct_r
            q_i = -a_i * u_r + a_r * u_i + direct_i

            next_qr = a_r * sum_qr + a_i * sum_qi + q_r
            next_qi = -a_i * sum_qr + a_r * sum_qi + q_i
            next_pr = a_r * sum_pr + a_i * sum_pi
            next_pi = -a_i * sum_pr + a_r * sum_pi
            sum_pr, sum_pi = next_pr, next_pi
            sum_qr, sum_qi = next_qr, next_qi

        base = (pid_b * n_chunks + pid_chunk) * channels + offs
        tl.store(chunk_pr_ptr + base, sum_pr, mask=mask_c)
        tl.store(chunk_pi_ptr + base, sum_pi, mask=mask_c)
        tl.store(chunk_qr_ptr + base, sum_qr, mask=mask_c)
        tl.store(chunk_qi_ptr + base, sum_qi, mask=mask_c)


    @triton.jit
    def _field_backward_chunk_carry_kernel(
        chunk_pr_ptr,
        chunk_pi_ptr,
        chunk_qr_ptr,
        chunk_qi_ptr,
        carry_after_r_ptr,
        carry_after_i_ptr,
        n_chunks,
        channels,
        BLOCK_C: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_cb = tl.program_id(1)
        offs = pid_cb * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs < channels
        carry_r = tl.zeros([BLOCK_C], tl.float32)
        carry_i = tl.zeros([BLOCK_C], tl.float32)

        chunk = n_chunks
        while chunk > 0:
            chunk -= 1
            base = (pid_b * n_chunks + chunk) * channels + offs
            tl.store(carry_after_r_ptr + base, carry_r, mask=mask_c)
            tl.store(carry_after_i_ptr + base, carry_i, mask=mask_c)
            pr = tl.load(chunk_pr_ptr + base, mask=mask_c, other=1.0).to(tl.float32)
            pi = tl.load(chunk_pi_ptr + base, mask=mask_c, other=0.0).to(tl.float32)
            qr = tl.load(chunk_qr_ptr + base, mask=mask_c, other=0.0).to(tl.float32)
            qi = tl.load(chunk_qi_ptr + base, mask=mask_c, other=0.0).to(tl.float32)
            next_r = pr * carry_r - pi * carry_i + qr
            next_i = pr * carry_i + pi * carry_r + qi
            carry_r, carry_i = next_r, next_i


    @triton.jit
    def _field_backward_chunk_materialize_kernel(
        raw_ptr,
        tr_ptr,
        ti_ptr,
        gamma_ptr,
        read_ptr,
        grad_read_ptr,
        carry_after_r_ptr,
        carry_after_i_ptr,
        grad_raw_ptr,
        grad_tr_partial_ptr,
        grad_ti_partial_ptr,
        grad_gamma_partial_ptr,
        n_ctx,
        n_chunks,
        channels,
        VACANCY_MAX: tl.constexpr,
        CHUNK_T: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_chunk = tl.program_id(1)
        pid_cb = tl.program_id(2)
        offs = pid_cb * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs < channels

        tr = tl.load(tr_ptr + offs, mask=mask_c, other=0.0).to(tl.float32)
        ti = tl.load(ti_ptr + offs, mask=mask_c, other=0.0).to(tl.float32)
        gm = tl.load(gamma_ptr + offs, mask=mask_c, other=0.0).to(tl.float32)
        carry_base = (pid_b * n_chunks + pid_chunk) * channels + offs
        carry_r = tl.load(
            carry_after_r_ptr + carry_base,
            mask=mask_c,
            other=0.0,
        ).to(tl.float32)
        carry_i = tl.load(
            carry_after_i_ptr + carry_base,
            mask=mask_c,
            other=0.0,
        ).to(tl.float32)

        dtr_acc = tl.zeros([BLOCK_C], tl.float32)
        dti_acc = tl.zeros([BLOCK_C], tl.float32)
        dgm_acc = tl.zeros([BLOCK_C], tl.float32)

        local_t = CHUNK_T
        while local_t > 0:
            local_t -= 1
            t = pid_chunk * CHUNK_T + local_t
            valid = mask_c & (t < n_ctx)
            raw_base = (pid_b * n_ctx + t) * (3 * channels)
            zr = tl.load(raw_ptr + raw_base + offs, mask=valid, other=0.0).to(tl.float32)
            zi = tl.load(
                raw_ptr + raw_base + channels + offs,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            zv = tl.load(
                raw_ptr + raw_base + 2 * channels + offs,
                mask=valid,
                other=0.0,
            ).to(tl.float32)

            inj_r = 2.0 * tl.sigmoid(2.0 * zr) - 1.0
            inj_i = 2.0 * tl.sigmoid(2.0 * zi) - 1.0
            sig_v = tl.sigmoid(zv)
            vacancy = VACANCY_MAX * sig_v
            keep = 1.0 - vacancy

            out_base = (pid_b * n_ctx + t) * (4 * channels)
            grad_state_r = tl.load(
                grad_read_ptr + out_base + offs,
                mask=valid,
                other=0.0,
            ).to(tl.float32) + carry_r
            grad_state_i = tl.load(
                grad_read_ptr + out_base + channels + offs,
                mask=valid,
                other=0.0,
            ).to(tl.float32) + carry_i
            grad_disp_r = tl.load(
                grad_read_ptr + out_base + 2 * channels + offs,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            grad_disp_i = tl.load(
                grad_read_ptr + out_base + 3 * channels + offs,
                mask=valid,
                other=0.0,
            ).to(tl.float32)

            has_prev = valid & (t > 0)
            prev_base = (pid_b * n_ctx + (t - 1)) * (4 * channels)
            prev_r = tl.load(
                read_ptr + prev_base + offs,
                mask=has_prev,
                other=0.0,
            ).to(tl.float32)
            prev_i = tl.load(
                read_ptr + prev_base + channels + offs,
                mask=has_prev,
                other=0.0,
            ).to(tl.float32)

            a_r = keep * tr
            a_i = keep * ti
            moved_r = tr * prev_r - ti * prev_i
            moved_i = tr * prev_i + ti * prev_r

            d_ar = grad_state_r * prev_r + grad_state_i * prev_i
            d_ai = -grad_state_r * prev_i + grad_state_i * prev_r
            d_br = grad_state_r
            d_bi = grad_state_i

            next_carry_r = a_r * grad_state_r + a_i * grad_state_i
            next_carry_i = -a_i * grad_state_r + a_r * grad_state_i

            dv_disp = grad_disp_r * moved_r + grad_disp_i * moved_i
            dtr_disp = vacancy * (
                grad_disp_r * prev_r + grad_disp_i * prev_i
            )
            dti_disp = vacancy * (
                -grad_disp_r * prev_i + grad_disp_i * prev_r
            )
            next_carry_r += vacancy * (
                tr * grad_disp_r + ti * grad_disp_i
            )
            next_carry_i += vacancy * (
                -ti * grad_disp_r + tr * grad_disp_i
            )

            common_b = inj_r * d_br + inj_i * d_bi
            d_inj_r = gm * vacancy * d_br
            d_inj_i = gm * vacancy * d_bi
            d_v = -tr * d_ar - ti * d_ai + gm * common_b + dv_disp
            dtr_acc += tl.where(valid, keep * d_ar + dtr_disp, 0.0)
            dti_acc += tl.where(valid, keep * d_ai + dti_disp, 0.0)
            dgm_acc += tl.where(valid, vacancy * common_b, 0.0)

            d_zr = d_inj_r * (1.0 - inj_r * inj_r)
            d_zi = d_inj_i * (1.0 - inj_i * inj_i)
            d_zv = d_v * VACANCY_MAX * sig_v * (1.0 - sig_v)
            tl.store(grad_raw_ptr + raw_base + offs, d_zr, mask=valid)
            tl.store(
                grad_raw_ptr + raw_base + channels + offs,
                d_zi,
                mask=valid,
            )
            tl.store(
                grad_raw_ptr + raw_base + 2 * channels + offs,
                d_zv,
                mask=valid,
            )
            carry_r, carry_i = next_carry_r, next_carry_i

        partial_base = (
            (pid_b * n_chunks + pid_chunk) * channels + offs
        )
        tl.store(grad_tr_partial_ptr + partial_base, dtr_acc, mask=mask_c)
        tl.store(grad_ti_partial_ptr + partial_base, dti_acc, mask=mask_c)
        tl.store(grad_gamma_partial_ptr + partial_base, dgm_acc, mask=mask_c)


class _ChunkedFieldReadFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        raw: torch.Tensor,
        transition_r: torch.Tensor,
        transition_i: torch.Tensor,
        gamma: torch.Tensor,
        block_c: int,
        chunk_t: int,
    ) -> torch.Tensor:
        if not HAS_TRITON:
            raise RuntimeError("Triton unavailable: " + TRITON_IMPORT_ERROR)
        if not raw.is_cuda:
            raise RuntimeError("Chunked Field kernel requires CUDA")
        if raw.dtype != torch.float32:
            raise TypeError(f"raw must be float32, got {raw.dtype}")
        raw = raw.contiguous()
        transition_r = transition_r.contiguous().float()
        transition_i = transition_i.contiguous().float()
        gamma = gamma.contiguous().float()
        batch, length, three_channels = raw.shape
        if three_channels % 3:
            raise ValueError(raw.shape)
        channels = three_channels // 3
        if transition_r.numel() != channels:
            raise ValueError((raw.shape, transition_r.shape))
        n_chunks = triton.cdiv(length, chunk_t)

        shape = (batch, n_chunks, channels)
        chunk_ar = torch.empty(shape, device=raw.device, dtype=torch.float32)
        chunk_ai = torch.empty_like(chunk_ar)
        chunk_br = torch.empty_like(chunk_ar)
        chunk_bi = torch.empty_like(chunk_ar)
        carry_r = torch.empty_like(chunk_ar)
        carry_i = torch.empty_like(chunk_ar)
        read = torch.empty(
            (batch, length, 4 * channels),
            device=raw.device,
            dtype=torch.float32,
        )

        grid_chunks = (
            batch,
            n_chunks,
            triton.cdiv(channels, block_c),
        )
        _field_chunk_summary_kernel[grid_chunks](
            raw,
            transition_r,
            transition_i,
            gamma,
            chunk_ar,
            chunk_ai,
            chunk_br,
            chunk_bi,
            length,
            n_chunks,
            channels,
            VACANCY_MAX=VAC_MAX,
            CHUNK_T=chunk_t,
            BLOCK_C=block_c,
            num_warps=1,
            num_stages=1,
        )
        grid_carry = (batch, triton.cdiv(channels, block_c))
        _field_chunk_carry_kernel[grid_carry](
            chunk_ar,
            chunk_ai,
            chunk_br,
            chunk_bi,
            carry_r,
            carry_i,
            n_chunks,
            channels,
            BLOCK_C=block_c,
            num_warps=1,
            num_stages=1,
        )
        _field_chunk_materialize_kernel[grid_chunks](
            raw,
            transition_r,
            transition_i,
            gamma,
            carry_r,
            carry_i,
            read,
            length,
            n_chunks,
            channels,
            VACANCY_MAX=VAC_MAX,
            CHUNK_T=chunk_t,
            BLOCK_C=block_c,
            num_warps=1,
            num_stages=1,
        )
        ctx.save_for_backward(
            raw,
            transition_r,
            transition_i,
            gamma,
            read,
        )
        ctx.block_c = int(block_c)
        ctx.chunk_t = int(chunk_t)
        ctx.n_chunks = int(n_chunks)
        return read

    @staticmethod
    def backward(ctx, grad_read: torch.Tensor):
        raw, transition_r, transition_i, gamma, read = ctx.saved_tensors
        grad_read = grad_read.contiguous().float()
        batch, length, three_channels = raw.shape
        channels = three_channels // 3
        shape = (batch, ctx.n_chunks, channels)

        chunk_pr = torch.empty(shape, device=raw.device, dtype=torch.float32)
        chunk_pi = torch.empty_like(chunk_pr)
        chunk_qr = torch.empty_like(chunk_pr)
        chunk_qi = torch.empty_like(chunk_pr)
        carry_after_r = torch.empty_like(chunk_pr)
        carry_after_i = torch.empty_like(chunk_pr)
        grad_raw = torch.empty_like(raw)
        grad_tr_partial = torch.empty_like(chunk_pr)
        grad_ti_partial = torch.empty_like(chunk_pr)
        grad_gamma_partial = torch.empty_like(chunk_pr)

        grid_chunks = (
            batch,
            ctx.n_chunks,
            triton.cdiv(channels, ctx.block_c),
        )
        _field_backward_chunk_summary_kernel[grid_chunks](
            raw,
            transition_r,
            transition_i,
            grad_read,
            chunk_pr,
            chunk_pi,
            chunk_qr,
            chunk_qi,
            length,
            ctx.n_chunks,
            channels,
            VACANCY_MAX=VAC_MAX,
            CHUNK_T=ctx.chunk_t,
            BLOCK_C=ctx.block_c,
            num_warps=1,
            num_stages=1,
        )
        grid_carry = (batch, triton.cdiv(channels, ctx.block_c))
        _field_backward_chunk_carry_kernel[grid_carry](
            chunk_pr,
            chunk_pi,
            chunk_qr,
            chunk_qi,
            carry_after_r,
            carry_after_i,
            ctx.n_chunks,
            channels,
            BLOCK_C=ctx.block_c,
            num_warps=1,
            num_stages=1,
        )
        _field_backward_chunk_materialize_kernel[grid_chunks](
            raw,
            transition_r,
            transition_i,
            gamma,
            read,
            grad_read,
            carry_after_r,
            carry_after_i,
            grad_raw,
            grad_tr_partial,
            grad_ti_partial,
            grad_gamma_partial,
            length,
            ctx.n_chunks,
            channels,
            VACANCY_MAX=VAC_MAX,
            CHUNK_T=ctx.chunk_t,
            BLOCK_C=ctx.block_c,
            num_warps=1,
            num_stages=1,
        )
        return (
            grad_raw,
            grad_tr_partial.sum(dim=(0, 1)),
            grad_ti_partial.sum(dim=(0, 1)),
            grad_gamma_partial.sum(dim=(0, 1)),
            None,
            None,
        )


def fused_field_read(
    raw: torch.Tensor,
    transition_r: torch.Tensor,
    transition_i: torch.Tensor,
    gamma: torch.Tensor,
    block_c: int,
    chunk_t: int,
) -> torch.Tensor:
    return _ChunkedFieldReadFunction.apply(
        raw,
        transition_r,
        transition_i,
        gamma,
        block_c,
        chunk_t,
    )


# =================================================================================================
# Field and Transformer models
# =================================================================================================


class IndependentVacancyField(nn.Module):
    """Zero-attention Field with independent vacancy and displaced readout."""

    def __init__(
        self,
        dim: int,
        backend: str,
        field_chunk: int,
        triton_block_c: int,
        triton_chunk_t: int,
    ):
        super().__init__()
        if dim % 2:
            raise ValueError("Field dim must be even")
        if backend not in ("reference", "triton"):
            raise ValueError(backend)
        self.dim = dim
        self.channels = dim // 2
        self.backend = backend
        self.field_chunk = int(field_chunk)
        self.triton_block_c = int(triton_block_c)
        self.triton_chunk_t = int(triton_chunk_t)

        # [inj_real logits, inj_imag logits, vacancy logits]
        self.write_proj = nn.Linear(dim, dim + self.channels)
        ring = torch.linspace(0.85, 0.999, self.channels)
        self.radius_logit = nn.Parameter(torch.log(ring / (1.0 - ring)))
        self.theta = nn.Parameter(
            torch.linspace(0.03, math.pi * 0.97, self.channels)
        )

        # Read contains [state.real, state.imag, displaced.real, displaced.imag].
        self.read_norm = RMSNorm(2 * dim)
        self.out_proj = nn.Linear(2 * dim, dim)
        self.gate_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.write_proj(x).contiguous().float()
        radius = torch.sigmoid(self.radius_logit).clamp(0.50, 0.99995).float()
        theta = self.theta.float()
        transition_r = radius * torch.cos(theta)
        transition_i = radius * torch.sin(theta)
        gamma = torch.sqrt((1.0 - radius.square()).clamp_min(1e-4))

        if self.backend == "triton":
            read = fused_field_read(
                raw,
                transition_r,
                transition_i,
                gamma,
                self.triton_block_c,
                self.triton_chunk_t,
            )
        else:
            read = reference_field_read(
                raw,
                transition_r,
                transition_i,
                gamma,
                self.field_chunk,
            )

        out = self.out_proj(self.read_norm(read))
        gate = torch.sigmoid(self.gate_proj(x))
        return x + (out * gate).to(x.dtype)


class FieldBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        backend: str,
        field_chunk: int,
        triton_block_c: int,
        triton_chunk_t: int,
        ff_hidden: int,
    ):
        super().__init__()
        self.mixer = IndependentVacancyField(
            dim,
            backend,
            field_chunk,
            triton_block_c,
            triton_chunk_t,
        )
        self.ff_norm = RMSNorm(dim)
        self.ff = PackedSwiGLU(dim, hidden=ff_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mixer(x)
        return x + self.ff(self.ff_norm(x))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        inv = 1.0 / (
            10000.0
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32)
                / self.head_dim
            )
        )
        self.register_buffer("inv_freq", inv, persistent=False)
        self._cache_key = None
        self._cache_cos = None
        self._cache_sin = None

    def rope(self, length: int, device: torch.device, dtype: torch.dtype):
        key = (length, device.type, device.index, dtype)
        if self._cache_key != key:
            pos = torch.arange(length, device=device, dtype=torch.float32)
            freqs = torch.outer(pos, self.inv_freq.to(device))
            emb = torch.repeat_interleave(freqs, 2, dim=-1)
            self._cache_cos = emb.cos().to(dtype)[None, None]
            self._cache_sin = emb.sin().to(dtype)[None, None]
            self._cache_key = key
        return self._cache_cos, self._cache_sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rope(length, q.device, q.dtype)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(
            y.transpose(1, 2).contiguous().view(batch, length, self.dim)
        )


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_hidden: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, heads)
        self.ff_norm = RMSNorm(dim)
        self.ff = PackedSwiGLU(dim, hidden=ff_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm(x))
        return x + self.ff(self.ff_norm(x))


@dataclass
class ModelConfig:
    model: str
    dim: int = 192
    heads: int = 6
    layers: int = 6
    field_chunk: int = 32
    triton_block_c: int = 16
    triton_chunk_t: int = 128
    checkpoint_blocks: bool = False
    field_ff_hidden: int = 512
    transformer_ff_hidden: int = 544


class ByteLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(BYTE_VOCAB, cfg.dim)
        blocks: List[nn.Module] = []
        if cfg.model in ("field_reference", "field_triton"):
            backend = "reference" if cfg.model == "field_reference" else "triton"
            for _ in range(cfg.layers):
                blocks.append(
                    FieldBlock(
                        cfg.dim,
                        backend,
                        cfg.field_chunk,
                        cfg.triton_block_c,
                        cfg.triton_chunk_t,
                        cfg.field_ff_hidden,
                    )
                )
        elif cfg.model == "transformer":
            for _ in range(cfg.layers):
                blocks.append(
                    TransformerBlock(
                        cfg.dim,
                        cfg.heads,
                        cfg.transformer_ff_hidden,
                    )
                )
        else:
            raise ValueError(cfg.model)
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, BYTE_VOCAB, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.emb(tokens)
        for block in self.blocks:
            x = checkpoint_block(block, x, self.cfg.checkpoint_blocks)
        return self.lm_head(self.final_norm(x))


def default_hidden_sizes(dim: int) -> Tuple[int, int]:
    field_hidden = ((int(8 * dim / 3) + 63) // 64) * 64
    # Exact block-level matching would add (dim+11)/6 hidden units.
    extra = int(round((dim + 11) / 6))
    transformer_hidden = ((field_hidden + extra + 8) // 16) * 16
    return field_hidden, transformer_hidden


def make_model_config(args, model: str) -> ModelConfig:
    field_hidden, transformer_hidden = default_hidden_sizes(args.dim)
    return ModelConfig(
        model=model,
        dim=args.dim,
        heads=args.heads,
        layers=args.layers,
        field_chunk=args.field_chunk,
        triton_block_c=args.triton_block_c,
        triton_chunk_t=args.triton_chunk_t,
        checkpoint_blocks=args.checkpoint_blocks,
        field_ff_hidden=field_hidden,
        transformer_ff_hidden=transformer_hidden,
    )


# =================================================================================================
# Correctness / audit
# =================================================================================================


def audit_parameter_parity(args) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for model_name in args.models:
        set_seed(args.model_seed)
        model = ByteLM(make_model_config(args, model_name))
        counts[model_name] = nparams(model)
        del model
    smallest = min(counts.values())
    largest = max(counts.values())
    delta = 100.0 * (largest - smallest) / smallest
    log("[params] " + json.dumps(counts, indent=2))
    log(f"[params] maximum delta={delta:.4f}%")
    if delta > args.max_param_delta_pct:
        raise RuntimeError(
            f"Parameter mismatch {delta:.4f}% exceeds "
            f"{args.max_param_delta_pct:.4f}%"
        )
    return counts


def run_kernel_self_test(device: torch.device, args) -> Dict[str, float]:
    if device.type != "cuda":
        raise RuntimeError("Kernel self-test requires CUDA")
    if not HAS_TRITON:
        raise RuntimeError("Triton unavailable: " + TRITON_IMPORT_ERROR)

    log("[self-test] reference vs fused Triton forward/backward equivalence")
    test_args = argparse.Namespace(**vars(args))
    test_args.dim = 64
    test_args.heads = 4
    test_args.layers = 2
    test_args.field_chunk = 8
    test_args.triton_block_c = 8
    test_args.triton_chunk_t = 16
    test_args.checkpoint_blocks = False

    set_seed(7721)
    ref = ByteLM(make_model_config(test_args, "field_reference")).to(device)
    tri = ByteLM(make_model_config(test_args, "field_triton")).to(device)
    tri.load_state_dict(ref.state_dict(), strict=True)
    ref.train()
    tri.train()

    tokens = torch.randint(0, BYTE_VOCAB, (2, 37), device=device)
    targets = torch.randint(0, BYTE_VOCAB, (2, 37), device=device)

    out_ref = ref(tokens)
    out_tri = tri(tokens)
    forward_max = float((out_ref - out_tri).abs().max())
    forward_mean = float((out_ref - out_tri).abs().mean())

    loss_ref = F.cross_entropy(
        out_ref.float().reshape(-1, BYTE_VOCAB),
        targets.reshape(-1),
    )
    loss_tri = F.cross_entropy(
        out_tri.float().reshape(-1, BYTE_VOCAB),
        targets.reshape(-1),
    )
    loss_ref.backward()
    loss_tri.backward()

    grad_max_abs = 0.0
    grad_max_rel = 0.0
    worst_name = ""
    ref_params = dict(ref.named_parameters())
    tri_params = dict(tri.named_parameters())
    for name in ref_params:
        ga = ref_params[name].grad
        gb = tri_params[name].grad
        if ga is None or gb is None:
            continue
        abs_err = float((ga - gb).abs().max())
        denom = max(float(ga.abs().max()), 1e-6)
        rel_err = abs_err / denom
        if rel_err > grad_max_rel:
            grad_max_rel = rel_err
            grad_max_abs = abs_err
            worst_name = name

    # Causality check on the optimized model.
    tri.eval()
    with torch.no_grad():
        full = tri(tokens)
        prefix = tri(tokens[:, :23])
    causal_max = float((full[:, :23] - prefix).abs().max())

    result = {
        "forward_max_abs": forward_max,
        "forward_mean_abs": forward_mean,
        "loss_abs": abs(float(loss_ref) - float(loss_tri)),
        "grad_max_abs": grad_max_abs,
        "grad_max_relative": grad_max_rel,
        "grad_worst_parameter": worst_name,
        "causal_max_abs": causal_max,
    }
    log("[self-test] " + json.dumps(result, indent=2))

    if forward_max > args.selftest_forward_tol:
        raise AssertionError(
            f"Triton forward mismatch {forward_max:.3e} > "
            f"{args.selftest_forward_tol:.3e}"
        )
    if grad_max_rel > args.selftest_grad_rel_tol and grad_max_abs > args.selftest_grad_abs_tol:
        raise AssertionError(
            f"Triton gradient mismatch rel={grad_max_rel:.3e}, "
            f"abs={grad_max_abs:.3e}, parameter={worst_name}"
        )
    if causal_max > args.selftest_causal_tol:
        raise AssertionError(
            f"Triton causality mismatch {causal_max:.3e} > "
            f"{args.selftest_causal_tol:.3e}"
        )

    del ref, tri, tokens, targets, out_ref, out_tri, loss_ref, loss_tri
    gc.collect()
    torch.cuda.empty_cache()
    return result


# =================================================================================================
# Training / evaluation
# =================================================================================================


@dataclass
class TrainConfig:
    outdir: str
    steps: int
    train_seq: int
    batch_size: int
    accum: int
    lr: float
    min_lr_ratio: float
    warmup: int
    weight_decay: float
    grad_clip: float
    log_every: int
    eval_every: int
    save_every: int
    quick_eval_batches: int
    final_contexts: Tuple[int, ...]
    final_eval_windows: int
    final_eval_batch: int
    target_bytes: int
    model_seed: int
    data_seed: int
    eval_seed: int
    resume: bool


def lr_at(
    step: int,
    total: int,
    warmup: int,
    peak: float,
    min_ratio: float,
) -> float:
    if step <= warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (min_ratio + (1.0 - min_ratio) * cosine)


def make_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    kwargs = dict(
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
    )
    if torch.cuda.is_available():
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(model.parameters(), **kwargs)


@torch.no_grad()
def quick_eval(
    model: ByteLM,
    data: torch.Tensor,
    device: torch.device,
    amp: str,
    seq_len: int,
    batches: int,
    batch_size: int,
    seed: int,
) -> float:
    model.eval()
    generator = generator_for(data, seed)
    losses = []
    for _ in range(batches):
        x, y = random_byte_batch(
            data,
            batch_size,
            seq_len,
            generator,
            device,
        )
        with autocast_context(device, amp):
            logits = model(x)
            loss = F.cross_entropy(
                logits.float().reshape(-1, BYTE_VOCAB),
                y.reshape(-1),
            )
        losses.append(float(loss))
        del x, y, logits, loss
    return float(np.mean(losses) / LN2)


@torch.no_grad()
def final_eval(
    model: ByteLM,
    data: torch.Tensor,
    device: torch.device,
    amp: str,
    context: int,
    starts: Sequence[int],
    batch_size: int,
    target_bytes: int,
) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    total_targets = 0
    total_time = 0.0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for base in range(0, len(starts), batch_size):
        chunk = starts[base : base + batch_size]
        x, y = fixed_eval_batch(data, chunk, context, device)
        sync()
        started = time.perf_counter()
        with autocast_context(device, amp):
            logits = model(x)
            n_target = min(target_bytes, context)
            loss = F.cross_entropy(
                logits[:, -n_target:].float().reshape(-1, BYTE_VOCAB),
                y[:, -n_target:].reshape(-1),
            )
        sync()
        elapsed = time.perf_counter() - started
        losses.append(float(loss))
        total_targets += len(chunk) * min(target_bytes, context)
        total_time += elapsed
        del x, y, logits, loss

    mean_loss = float(np.mean(losses))
    return {
        "context": context,
        "bpb": mean_loss / LN2,
        "ppl": math.exp(mean_loss),
        "target_bytes": total_targets,
        "bytes_per_second": total_targets / max(total_time, 1e-9),
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
    }


def save_checkpoint(
    path: Path,
    model: ByteLM,
    optimizer: torch.optim.Optimizer,
    scaler,
    step: int,
    history: List[Dict[str, float]],
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    data_generator: torch.Generator,
) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "step": step,
            "history": history,
            "model_cfg": asdict(model_cfg),
            "train_cfg": asdict(train_cfg),
            "data_rng_state": data_generator.get_state().cpu(),
        },
        tmp,
    )
    os.replace(tmp, path)


def train_one_model(
    model_name: str,
    args,
    train_cfg: TrainConfig,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    device: torch.device,
    amp: str,
) -> Dict[str, object]:
    run_dir = Path(train_cfg.outdir) / model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = run_dir / "latest.pt"

    set_seed(train_cfg.model_seed)
    model_cfg = make_model_config(args, model_name)
    model = ByteLM(model_cfg).to(device)
    params = nparams(model)
    optimizer = make_optimizer(model, train_cfg.lr, train_cfg.weight_decay)
    use_scaler = amp == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_scaler else None
    generator = generator_for(train_data, train_cfg.data_seed)
    start_step = 0
    history: List[Dict[str, float]] = []

    if train_cfg.resume and latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scaler is not None and state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        start_step = int(state["step"])
        history = list(state.get("history", []))
        if state.get("data_rng_state") is not None:
            generator.set_state(state["data_rng_state"].cpu())
        log(f"[resume] {model_name} step={start_step}")

    log(
        f"\n{'=' * 120}\n"
        f"TRAIN {model_name} | params={params:,} | "
        f"steps={train_cfg.steps:,} | context={train_cfg.train_seq} | "
        f"effective bytes/update="
        f"{train_cfg.batch_size*train_cfg.accum*train_cfg.train_seq:,}\n"
        f"{'=' * 120}"
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    wall = time.perf_counter()
    bytes_per_update = (
        train_cfg.batch_size * train_cfg.accum * train_cfg.train_seq
    )
    bytes_at_start = start_step * bytes_per_update

    for step in range(start_step + 1, train_cfg.steps + 1):
        lr_now = lr_at(
            step,
            train_cfg.steps,
            train_cfg.warmup,
            train_cfg.lr,
            train_cfg.min_lr_ratio,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr_now

        ce_sum = 0.0
        for micro in range(train_cfg.accum):
            x, y = random_byte_batch(
                train_data,
                train_cfg.batch_size,
                train_cfg.train_seq,
                generator,
                device,
            )
            with autocast_context(device, amp):
                logits = model(x)
                ce = F.cross_entropy(
                    logits.float().reshape(-1, BYTE_VOCAB),
                    y.reshape(-1),
                )
                loss = ce / train_cfg.accum
            if not torch.isfinite(ce):
                raise FloatingPointError(
                    f"Non-finite CE: model={model_name}, step={step}, micro={micro}"
                )
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            ce_sum += float(ce.detach())
            del x, y, logits, ce, loss

        if scaler is not None:
            scaler.unscale_(optimizer)
        grad = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            train_cfg.grad_clip,
        )
        if not torch.isfinite(grad):
            raise FloatingPointError(
                f"Non-finite gradient: model={model_name}, step={step}"
            )
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        should_log = (
            step == 1
            or step % train_cfg.log_every == 0
            or step == train_cfg.steps
        )
        if should_log:
            sync()
            elapsed = max(1e-9, time.perf_counter() - wall)
            bytes_seen = step * bytes_per_update
            bps = (bytes_seen - bytes_at_start) / elapsed
            peak = torch.cuda.max_memory_allocated() / 2**30
            row = {
                "step": step,
                "train_bpb": (ce_sum / train_cfg.accum) / LN2,
                "grad": float(grad),
                "lr": lr_now,
                "bytes_per_second": bps,
                "peak_gib": peak,
            }
            history.append(row)
            log(
                f"step {step:6d}/{train_cfg.steps} "
                f"train_bpb={row['train_bpb']:.4f} "
                f"grad={row['grad']:.3f} lr={lr_now:.3e} "
                f"B/s={bps:,.0f} peak={peak:.2f}G"
            )

        if step % train_cfg.eval_every == 0 or step == train_cfg.steps:
            val_bpb = quick_eval(
                model,
                val_data,
                device,
                amp,
                train_cfg.train_seq,
                train_cfg.quick_eval_batches,
                max(1, min(train_cfg.batch_size, 4)),
                train_cfg.eval_seed,
            )
            if not history or history[-1].get("step") != step:
                history.append({"step": step})
            history[-1]["quick_val_bpb"] = val_bpb
            log(
                f"  quick_val step={step} bpb={val_bpb:.4f} "
                f"ppl={2 ** val_bpb:.3f}"
            )
            model.train()

        if step % train_cfg.save_every == 0 or step == train_cfg.steps:
            save_checkpoint(
                latest,
                model,
                optimizer,
                scaler,
                step,
                history,
                model_cfg,
                train_cfg,
                generator,
            )

    final_rows = []
    for context in train_cfg.final_contexts:
        starts = fixed_eval_starts(
            len(val_data),
            context,
            train_cfg.final_eval_windows,
            train_cfg.eval_seed,
        )
        try:
            row = final_eval(
                model,
                val_data,
                device,
                amp,
                context,
                starts,
                train_cfg.final_eval_batch,
                train_cfg.target_bytes,
            )
            row.update(status="ok", error="")
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            row = {
                "context": context,
                "status": "oom",
                "error": str(exc),
                "bpb": None,
                "ppl": None,
                "target_bytes": 0,
                "bytes_per_second": None,
                "peak_gib": None,
            }
        final_rows.append(row)
        log(
            f"  final context={context:5d} status={row['status']} "
            f"bpb={row.get('bpb')} B/s={row.get('bytes_per_second')} "
            f"peak={row.get('peak_gib')}"
        )

    best_quick = min(
        (
            row["quick_val_bpb"]
            for row in history
            if row.get("quick_val_bpb") is not None
        ),
        default=None,
    )
    result = {
        "model": model_name,
        "parameters": params,
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
        "amp": amp,
        "best_quick_val_bpb": best_quick,
        "history": history,
        "final": final_rows,
        "checkpoint": str(latest),
    }
    atomic_json(run_dir / "result.json", result)
    with (run_dir / "curve.csv").open("w", newline="", encoding="utf-8") as f:
        fields = sorted({key for row in history for key in row})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)

    del model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return result


# =================================================================================================
# Systems benchmark
# =================================================================================================


def benchmark_training_step(
    model_name: str,
    context: int,
    args,
    device: torch.device,
    amp: str,
) -> Dict[str, object]:
    set_seed(args.model_seed)
    cfg = make_model_config(args, model_name)
    cfg.checkpoint_blocks = False
    model = ByteLM(cfg).to(device).train()
    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    batch = args.bench_batch
    x = torch.randint(
        0,
        BYTE_VOCAB,
        (batch, context),
        device=device,
    )
    y = torch.randint(
        0,
        BYTE_VOCAB,
        (batch, context),
        device=device,
    )

    def one_step():
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            logits = model(x)
            loss = F.cross_entropy(
                logits.float().reshape(-1, BYTE_VOCAB),
                y.reshape(-1),
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        del logits, loss

    status = "ok"
    error = ""
    bps = None
    step_ms = None
    peak = None
    try:
        torch.cuda.empty_cache()
        for _ in range(args.bench_warmup):
            one_step()
        sync()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(args.bench_steps):
            one_step()
        sync()
        elapsed = time.perf_counter() - started
        bps = (
            args.bench_steps * batch * context / max(elapsed, 1e-9)
        )
        step_ms = elapsed * 1000.0 / args.bench_steps
        peak = torch.cuda.max_memory_allocated() / 2**30
    except torch.cuda.OutOfMemoryError as exc:
        status = "oom"
        error = str(exc)
        torch.cuda.empty_cache()
    except Exception as exc:
        status = "error"
        error = repr(exc)
        torch.cuda.empty_cache()

    result = {
        "model": model_name,
        "context": context,
        "batch": batch,
        "status": status,
        "bytes_per_second": bps,
        "step_ms": step_ms,
        "peak_gib": peak,
        "error": error,
    }
    del model, optimizer, x, y
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_benchmark(
    args,
    device: torch.device,
    amp: str,
    root: Path,
) -> List[Dict[str, object]]:
    rows = []
    log("\n" + "=" * 120)
    log("TRAINING-SYSTEMS BENCHMARK — FORWARD + BACKWARD + OPTIMIZER, NO CHECKPOINT")
    log("=" * 120)
    for context in args.bench_contexts:
        for model_name in args.models:
            row = benchmark_training_step(
                model_name,
                context,
                args,
                device,
                amp,
            )
            rows.append(row)
            log(
                f"{model_name:18s} ctx={context:5d} batch={args.bench_batch:2d} "
                f"status={row['status']:>5s} "
                f"B/s={row.get('bytes_per_second')} "
                f"step_ms={row.get('step_ms')} peak={row.get('peak_gib')}"
            )

    fields = [
        "model",
        "context",
        "batch",
        "status",
        "bytes_per_second",
        "step_ms",
        "peak_gib",
        "error",
    ]
    with (root / "systems_benchmark.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(root / "systems_benchmark.json", rows)
    return rows


# =================================================================================================
# Summaries
# =================================================================================================


def write_summary(
    root: Path,
    args,
    counts: Dict[str, int],
    selftest: Dict[str, float],
    bench_rows: Sequence[Dict[str, object]],
    train_results: Sequence[Dict[str, object]],
) -> None:
    final_rows = []
    for result in train_results:
        for row in result["final"]:
            final_rows.append(
                {
                    "model": result["model"],
                    "parameters": result["parameters"],
                    "best_quick_val_bpb": result["best_quick_val_bpb"],
                    **row,
                }
            )

    if final_rows:
        fields = [
            "model",
            "parameters",
            "best_quick_val_bpb",
            "context",
            "status",
            "bpb",
            "ppl",
            "target_bytes",
            "bytes_per_second",
            "peak_gib",
            "error",
        ]
        with (root / "quality_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fields,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(final_rows)

    lines = [
        "=" * 124,
        "FIELD-ONLY v4 — CHUNK-PARALLEL TRITON + WIKITEXT-103 100% SUMMARY",
        "=" * 124,
        "",
        "Audit:",
        f"  Triton: {getattr(triton, '__version__', 'unavailable')}",
        f"  zero-attention Field arms: field_reference, field_triton",
        f"  self-test forward max abs: {selftest.get('forward_max_abs')}",
        f"  self-test gradient max relative: {selftest.get('grad_max_relative')}",
        f"  self-test causality max abs: {selftest.get('causal_max_abs')}",
        "",
        "Parameters:",
    ]
    for name, value in counts.items():
        lines.append(f"  {name:18s} {value:12,d}")

    lines.extend(
        [
            "",
            "Systems benchmark (training step, no checkpoint):",
            f"{'model':18s} {'ctx':>7s} {'batch':>6s} {'status':>8s} "
            f"{'B/s':>13s} {'step ms':>11s} {'peak GiB':>10s}",
        ]
    )
    for row in bench_rows:
        bps = (
            "-"
            if row.get("bytes_per_second") is None
            else f"{float(row['bytes_per_second']):,.0f}"
        )
        ms = (
            "-"
            if row.get("step_ms") is None
            else f"{float(row['step_ms']):.2f}"
        )
        peak = (
            "-"
            if row.get("peak_gib") is None
            else f"{float(row['peak_gib']):.2f}"
        )
        lines.append(
            f"{row['model']:18s} {int(row['context']):7d} "
            f"{int(row['batch']):6d} {row['status']:>8s} "
            f"{bps:>13s} {ms:>11s} {peak:>10s}"
        )

    if final_rows:
        lines.extend(
            [
                "",
                "WikiText-103 quality:",
                f"{'model':18s} {'params':>12s} {'best quick':>11s} "
                f"{'ctx':>7s} {'status':>8s} {'BPB':>9s} "
                f"{'eval B/s':>12s} {'peak GiB':>10s}",
            ]
        )
        for row in final_rows:
            bpb = "-" if row.get("bpb") is None else f"{row['bpb']:.4f}"
            best = (
                "-"
                if row.get("best_quick_val_bpb") is None
                else f"{row['best_quick_val_bpb']:.4f}"
            )
            bps = (
                "-"
                if row.get("bytes_per_second") is None
                else f"{row['bytes_per_second']:,.0f}"
            )
            peak = (
                "-"
                if row.get("peak_gib") is None
                else f"{row['peak_gib']:.2f}"
            )
            lines.append(
                f"{row['model']:18s} {row['parameters']:12,d} "
                f"{best:>11s} {row['context']:7d} {row['status']:>8s} "
                f"{bpb:>9s} {bps:>12s} {peak:>10s}"
            )

    lines.extend(
        [
            "",
            "Artifacts:",
            f"  {root / 'summary.txt'}",
            f"  {root / 'systems_benchmark.csv'}",
            f"  {root / 'quality_summary.csv'}",
            f"  {root / 'full_console_log.txt'}",
            "=" * 124,
        ]
    )
    text = "\n".join(lines) + "\n"
    (root / "summary.txt").write_text(text, encoding="utf-8")
    log("\n" + text)


# =================================================================================================
# CLI / orchestration
# =================================================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--mode",
        choices=("full", "benchmark", "train", "selftest", "smoke"),
        default="full",
    )
    p.add_argument("--outdir", default="./field_only_v4_wiki100")
    p.add_argument("--cache-dir", default="./hf_cache")
    p.add_argument("--amp", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    p.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
    )
    p.add_argument("--data-frac", type=float, default=1.0)
    p.add_argument("--data-device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument(
        "--steps",
        type=int,
        default=0,
        help="0 computes steps from epochs and training-byte count",
    )
    p.add_argument("--train-seq", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--dim", type=int, default=192)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--field-chunk", type=int, default=32)
    p.add_argument("--triton-block-c", type=int, choices=(4, 8, 16, 32), default=16)
    p.add_argument(
        "--triton-chunk-t",
        type=int,
        choices=(16, 32, 64, 128, 256),
        default=128,
        help="time positions per parallel affine chunk",
    )
    p.add_argument(
        "--checkpoint-blocks",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.10)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--weight-decay", type=float, default=0.10)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--quick-eval-batches", type=int, default=8)
    p.add_argument(
        "--final-contexts",
        nargs="+",
        type=int,
        default=[512, 1024, 2048, 4096, 8192],
    )
    p.add_argument("--final-eval-windows", type=int, default=4)
    p.add_argument("--final-eval-batch", type=int, default=1)
    p.add_argument("--target-bytes", type=int, default=256)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument(
        "--bench-contexts",
        nargs="+",
        type=int,
        default=[512, 1024, 2048, 4096, 8192, 16384, 32768],
    )
    p.add_argument("--bench-batch", type=int, default=4)
    p.add_argument("--bench-warmup", type=int, default=3)
    p.add_argument("--bench-steps", type=int, default=8)

    p.add_argument("--model-seed", type=int, default=1234)
    p.add_argument("--data-seed", type=int, default=5678)
    p.add_argument("--eval-seed", type=int, default=9012)
    p.add_argument("--max-param-delta-pct", type=float, default=1.0)

    p.add_argument("--selftest-forward-tol", type=float, default=2e-3)
    p.add_argument("--selftest-grad-rel-tol", type=float, default=2e-2)
    p.add_argument("--selftest-grad-abs-tol", type=float, default=2e-3)
    p.add_argument("--selftest-causal-tol", type=float, default=2e-4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.outdir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    # Mirror stdout/stderr into a permanent log.
    log_path = root / "full_console_log.txt"
    import sys

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for stream in self.streams:
                stream.write(data)
                stream.flush()
            return len(data)

        def flush(self):
            for stream in self.streams:
                stream.flush()

    file_log = log_path.open("a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, file_log)
    sys.stderr = Tee(sys.__stderr__, file_log)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = choose_amp(args.amp)
    if device.type != "cuda":
        raise RuntimeError("This optimized arena requires an NVIDIA CUDA GPU")
    if not HAS_TRITON:
        raise RuntimeError("Triton unavailable: " + TRITON_IMPORT_ERROR)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.set_float32_matmul_precision("high")

    if args.dim % 2 or args.dim % args.heads:
        raise ValueError("dim must be even and divisible by heads")
    if args.triton_block_c > args.dim // 2:
        raise ValueError("triton_block_c cannot exceed dim/2")
    if args.triton_chunk_t < 1:
        raise ValueError("triton_chunk_t must be positive")

    if args.mode == "smoke":
        args.data_frac = 0.001
        args.epochs = 0.01
        args.steps = 4
        args.train_seq = 128
        args.batch_size = 2
        args.accum = 1
        args.dim = 64
        args.heads = 4
        args.layers = 2
        args.field_chunk = 8
        args.triton_block_c = 8
        args.triton_chunk_t = 16
        args.warmup = 2
        args.log_every = 1
        args.eval_every = 4
        args.save_every = 4
        args.quick_eval_batches = 1
        args.final_contexts = [128, 256]
        args.final_eval_windows = 1
        args.target_bytes = 64
        args.bench_contexts = [128]
        args.bench_batch = 2
        args.bench_warmup = 1
        args.bench_steps = 2
        args.resume = False
        args.checkpoint_blocks = False

    environment = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": getattr(triton, "__version__", "unknown"),
        "gpu": torch.cuda.get_device_name(0),
        "amp": amp,
        "mode": args.mode,
        "arguments": vars(args),
    }
    atomic_json(root / "environment.json", environment)
    log(json.dumps(environment, indent=2))

    counts = audit_parameter_parity(args)
    selftest = run_kernel_self_test(device, args)
    atomic_json(root / "kernel_selftest.json", selftest)
    if args.mode == "selftest":
        write_summary(root, args, counts, selftest, [], [])
        return

    bench_rows: List[Dict[str, object]] = []
    if args.mode in ("full", "benchmark", "smoke"):
        bench_rows = run_benchmark(args, device, amp, root)
    if args.mode == "benchmark":
        write_summary(root, args, counts, selftest, bench_rows, [])
        return

    train_results: List[Dict[str, object]] = []
    if args.mode in ("full", "train", "smoke"):
        train_data, val_data, _ = load_wikitext103_raw(
            args.cache_dir,
            args.data_frac,
        )
        train_data = place_dataset(
            train_data,
            args.data_device,
            device,
            "train",
        )
        val_data = place_dataset(
            val_data,
            args.data_device,
            device,
            "validation",
        )

        effective = args.batch_size * args.accum * args.train_seq
        steps = args.steps
        if steps <= 0:
            steps = max(1, math.ceil(args.epochs * len(train_data) / effective))
        warmup = min(args.warmup, max(1, steps // 5))
        log(
            f"[schedule] steps={steps:,}, effective bytes/update={effective:,}, "
            f"total sampled bytes={steps*effective:,}, "
            f"dataset bytes={len(train_data):,}, nominal epochs="
            f"{steps*effective/len(train_data):.4f}"
        )

        train_cfg = TrainConfig(
            outdir=str(root / "runs"),
            steps=steps,
            train_seq=args.train_seq,
            batch_size=args.batch_size,
            accum=args.accum,
            lr=args.lr,
            min_lr_ratio=args.min_lr_ratio,
            warmup=warmup,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            log_every=args.log_every,
            eval_every=min(args.eval_every, steps),
            save_every=min(args.save_every, steps),
            quick_eval_batches=args.quick_eval_batches,
            final_contexts=tuple(args.final_contexts),
            final_eval_windows=args.final_eval_windows,
            final_eval_batch=args.final_eval_batch,
            target_bytes=args.target_bytes,
            model_seed=args.model_seed,
            data_seed=args.data_seed,
            eval_seed=args.eval_seed,
            resume=args.resume,
        )
        for model_name in args.models:
            train_results.append(
                train_one_model(
                    model_name,
                    args,
                    train_cfg,
                    train_data,
                    val_data,
                    device,
                    amp,
                )
            )
        atomic_json(root / "all_training_results.json", train_results)

    write_summary(
        root,
        args,
        counts,
        selftest,
        bench_rows,
        train_results,
    )


if __name__ == "__main__":
    main()
