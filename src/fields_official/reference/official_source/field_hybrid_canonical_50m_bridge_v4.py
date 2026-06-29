#!/usr/bin/env python3
"""Canonical Triton Field + hierarchy/local-recall/confidence bridge at ~50M.

This arena transplants the v3-qualified side paths onto the validated Field-v4
Triton block.  It deliberately imports the frozen canonical source instead of
copying/reimplementing the recurrence.  The imported source SHA is checked.

Models
------
* baseline_v5                  canonical Triton Field + v5-like successor cache
* hybrid_w256_conf             softpatch + one local W256 path + confidence router
* hybrid_w128_two_conf         softpatch + two local W128 paths + confidence router
* hybrid_w256_conf_parity      W256 hybrid paid for by a narrower FFN
* attentionfree_multiscale     softpatch + multiscale causal conv + confidence router
* transformer_flash_sdpa       parameter-matched RoPE/SwiGLU/Flash-SDPA control

Protocol defaults
-----------------
* WikiText-103 raw bytes, 10% train split
* ~50M trainable parameters
* context 4096, 16,384 bytes/update, one data epoch
* BF16, no activation checkpointing
* all-position validation at 8K and 16K
* short LR calibration from fresh initializations
* separate equal no-checkpoint systems benchmark
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
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
V3_PATH = HERE / "field_hybrid_attentionfree_qualification_v3.py"
CANONICAL_NAME = "field_only_v4_chunked_triton_wiki100.py"
EXPECTED_CANONICAL_SHA256 = "0454647a7f667cdf73f45029eee0f525c58c0788ecd433ec1a4db8c3937ac848"
VOCAB = 256
LN2 = math.log(2.0)
FIELD_ARMS = (
    "baseline_v5",
    "hybrid_w256_conf",
    "hybrid_w128_two_conf",
    "hybrid_w256_conf_parity",
    "attentionfree_multiscale",
)
MODEL_NAMES = (*FIELD_ARMS, "transformer_flash_sdpa")


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
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def locate_canonical(explicit: str) -> Path:
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend([
        HERE / CANONICAL_NAME,
        Path("/home/ubuntu/field_pcaf_efficiency_v1") / CANONICAL_NAME,
        Path("/home/ubuntu/field_pcaf_pareto_v2") / CANONICAL_NAME,
        Path("/home/ubuntu/field_pcaf_quality_v4") / CANONICAL_NAME,
        Path("/home/ubuntu/field_pcaf_router_v5") / CANONICAL_NAME,
    ])
    for path in candidates:
        if path.is_file():
            return path.resolve()
    tried = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"missing canonical source {CANONICAL_NAME}. Tried:\n  {tried}\n"
        "Place the validated file beside this script or pass --canonical-source."
    )


def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def lr_at(step: int, total: int, warmup: int, peak: float, min_ratio: float) -> float:
    if step <= warmup:
        return peak * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return peak * (min_ratio + (1.0 - min_ratio) * cosine)


def batch_for_step(data: torch.Tensor, batch: int, seq: int, seed: int, step: int,
                   micro: int, device: torch.device, v3):
    source = data.device.type if data.device.type == "cuda" else "cpu"
    g = torch.Generator(device=source)
    g.manual_seed(seed + step * 1_000_003 + micro * 97)
    return v3.random_batch(data, batch, seq, g, device)


# =====================================================================================
# Modern Transformer control
# =====================================================================================


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


_ROPE_CACHE: Dict[Tuple[str, str, int, int], Tuple[torch.Tensor, torch.Tensor]] = {}


def rope_cache(device: torch.device, dtype: torch.dtype, seq: int, head_dim: int):
    key = (str(device), str(dtype), seq, head_dim)
    cached = _ROPE_CACHE.get(key)
    if cached is not None:
        return cached
    inv = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq, device=device).float()
    freq = torch.outer(pos, inv)
    emb = torch.repeat_interleave(freq, 2, dim=-1).to(dtype)
    cos, sin = emb.cos()[None, None], emb.sin()[None, None]
    _ROPE_CACHE[key] = (cos, sin)
    return cos, sin


class FlashAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_hidden: int, v3):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must divide heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.norm1 = v3.RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.norm2 = v3.RMSNorm(dim)
        self.ff = v3.PackedSwiGLU(dim, ff_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        z = self.qkv(self.norm1(x))
        q, k, v = z.chunk(3, dim=-1)
        q = q.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        cos, sin = rope_cache(q.device, q.dtype, t, self.head_dim)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(b, t, self.dim)
        x = x + self.out(y)
        return x + self.ff(self.norm2(x))


class FlashTransformerLM(nn.Module):
    def __init__(self, dim: int, heads: int, layers: int, ff_hidden: int, v3):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, dim)
        self.blocks = nn.ModuleList([
            FlashAttentionBlock(dim, heads, ff_hidden, v3) for _ in range(layers)
        ])
        self.final_norm = v3.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, VOCAB, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.emb(tokens)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))


# =====================================================================================
# Canonical Triton Field transplant
# =====================================================================================


def arm_spec(name: str, layers: int) -> dict:
    mid = max(0, layers // 2 - 1)
    late = layers - 1
    specs = {
        "baseline_v5": dict(softpatch=False, local_window=None, positions=(),
                            multiscale=False, router="v5", attention_free=True),
        "hybrid_w256_conf": dict(softpatch=True, local_window=256, positions=(mid,),
                                 multiscale=False, router="confidence", attention_free=False),
        "hybrid_w128_two_conf": dict(softpatch=True, local_window=128, positions=(mid, late),
                                     multiscale=False, router="confidence", attention_free=False),
        "hybrid_w256_conf_parity": dict(softpatch=True, local_window=256, positions=(mid,),
                                        multiscale=False, router="confidence", attention_free=False),
        "attentionfree_multiscale": dict(softpatch=True, local_window=None, positions=(),
                                         multiscale=True, router="confidence", attention_free=True),
    }
    return specs[name]


class CanonicalFieldPCAFLM(nn.Module):
    def __init__(self, name: str, dim: int, layers: int, ff_hidden: int,
                 field_chunk: int, triton_block_c: int, triton_chunk_t: int,
                 num_buckets: int, v3, v4):
        super().__init__()
        self.name = name
        self.spec = arm_spec(name, layers)
        self.emb = nn.Embedding(VOCAB, dim)
        # Common modules first: paired seeds preserve the baseline backbone/head.
        self.blocks = nn.ModuleList([
            v4.FieldBlock(
                dim, "triton", field_chunk, triton_block_c, triton_chunk_t, ff_hidden
            ) for _ in range(layers)
        ])
        self.final_norm = v3.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, VOCAB, bias=False)
        self.cache = v3.SuccessorCacheV5(
            dim, memory_dim=64, num_buckets=num_buckets, order=4, top_k=4,
            router_mode=self.spec["router"],
        )
        self.locals = nn.ModuleDict()
        if self.spec["local_window"] is not None:
            for pos in self.spec["positions"]:
                self.locals[str(pos)] = v3.LowRankLocalAttention(
                    dim, inner=128, heads=4, window=int(self.spec["local_window"]), chunk=256
                )
        self.multiscales = nn.ModuleDict()
        if self.spec["multiscale"]:
            mid = max(0, layers // 2 - 1)
            self.multiscales[str(mid)] = v3.MultiScaleCausalConv(dim, inner=128)
        self.softpatch = v3.BoundaryStateMixer(dim, rank=64, learned=True) \
            if self.spec["softpatch"] else None
        self.patch_position = max(0, layers // 2 - 1)
        self._patch_aux: Optional[torch.Tensor] = None

    @property
    def attention_free(self) -> bool:
        return bool(self.spec["attention_free"])

    def states_logits(self, tokens: torch.Tensor):
        x = self.emb(tokens)
        self._patch_aux = x.new_zeros(())
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == self.patch_position and self.softpatch is not None:
                x = self.softpatch(x, tokens)
                self._patch_aux = self.softpatch.last_aux
            key = str(i)
            if key in self.locals:
                x = self.locals[key](x)
            if key in self.multiscales:
                x = self.multiscales[key](x)
        return x, self.lm_head(self.final_norm(x))

    def loss_and_stats(self, tokens: torch.Tensor, targets: torch.Tensor,
                       compute_metrics: bool = False):
        states, logits = self.states_logits(tokens)
        loss, primary, stats = self.cache(states, logits, tokens, targets, compute_metrics)
        if self.training and self._patch_aux is not None:
            loss = loss + self._patch_aux
        return loss, primary, stats


@dataclass
class ModelShape:
    name: str
    params: int
    dim: int
    layers: int
    heads: int
    ff_hidden: int
    attention_free: bool


@dataclass
class EvalRow:
    context: int
    bpb: float
    param_bpb: Optional[float]
    oracle_bpb: Optional[float]
    capture: Optional[float]
    gate: Optional[float]
    gate_sep: Optional[float]


def build_field(name: str, args, v3, v4, ff_hidden: Optional[int] = None):
    hidden = int(ff_hidden if ff_hidden is not None else args.field_ff_hidden)
    return CanonicalFieldPCAFLM(
        name, args.field_dim, args.field_layers, hidden,
        args.field_chunk, args.triton_block_c, args.triton_chunk_t,
        args.num_buckets, v3, v4,
    )


def find_parity_hidden(args, v3, v4, target: int) -> int:
    best: Tuple[int, int] | None = None
    for hidden in range(max(256, args.field_ff_hidden - 512), args.field_ff_hidden + 1, 16):
        model = build_field("hybrid_w256_conf_parity", args, v3, v4, hidden)
        diff = abs(nparams(model) - target)
        if best is None or diff < best[0]:
            best = (diff, hidden)
        del model
    assert best is not None
    return best[1]


def resolve_shapes(args, v3, v4, device: torch.device):
    seed_all(args.model_seed)
    base = build_field("baseline_v5", args, v3, v4)
    target = nparams(base)
    del base
    parity_hidden = find_parity_hidden(args, v3, v4, target)
    shapes: Dict[str, ModelShape] = {}
    for name in FIELD_ARMS:
        seed_all(args.model_seed)
        hidden = parity_hidden if name.endswith("_parity") else args.field_ff_hidden
        model = build_field(name, args, v3, v4, hidden)
        shapes[name] = ModelShape(
            name=name, params=nparams(model), dim=args.field_dim,
            layers=args.field_layers, heads=args.field_heads, ff_hidden=hidden,
            attention_free=model.attention_free,
        )
        del model
    seed_all(args.model_seed)
    tf = FlashTransformerLM(
        args.tf_dim, args.tf_heads, args.tf_layers, args.tf_ff_hidden, v3
    )
    shapes["transformer_flash_sdpa"] = ModelShape(
        name="transformer_flash_sdpa", params=nparams(tf), dim=args.tf_dim,
        layers=args.tf_layers, heads=args.tf_heads, ff_hidden=args.tf_ff_hidden,
        attention_free=False,
    )
    del tf
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return shapes, parity_hidden


def build_model(name: str, args, v3, v4, shapes: Dict[str, ModelShape], device):
    seed_all(args.model_seed)
    if name == "transformer_flash_sdpa":
        s = shapes[name]
        return FlashTransformerLM(s.dim, s.heads, s.layers, s.ff_hidden, v3).to(device)
    return build_field(name, args, v3, v4, shapes[name].ff_hidden).to(device)


# =====================================================================================
# Evaluation / training
# =====================================================================================


@torch.no_grad()
def evaluate(model_name: str, model: nn.Module, data: torch.Tensor, context: int,
             windows: int, seed: int, device: torch.device, amp: str, v3) -> EvalRow:
    model.eval()
    starts = v3.fixed_starts(len(data), context, windows, seed + context * 17)
    losses: List[float] = []
    param: List[float] = []
    oracle: List[float] = []
    capture: List[float] = []
    gate: List[float] = []
    sep: List[float] = []
    for s in starts:
        x, y = v3.fixed_batch(data, [s], context, device)
        with amp_ctx(device, amp):
            if model_name == "transformer_flash_sdpa":
                logits = model(x)
                loss = F.cross_entropy(logits.float().reshape(-1, VOCAB), y.reshape(-1))
                losses.append(float(loss / LN2))
            else:
                _, primary, stats = model.loss_and_stats(x, y, compute_metrics=True)
                losses.append(float(primary / LN2))
                assert stats is not None
                param.append(stats.param_bpb)
                oracle.append(stats.oracle_bpb)
                capture.append(stats.capture)
                gate.append(stats.gate)
                sep.append(stats.gate_separation)
        del x, y
    return EvalRow(
        context=context,
        bpb=float(np.mean(losses)),
        param_bpb=float(np.mean(param)) if param else None,
        oracle_bpb=float(np.mean(oracle)) if oracle else None,
        capture=float(np.mean(capture)) if capture else None,
        gate=float(np.mean(gate)) if gate else None,
        gate_sep=float(np.mean(sep)) if sep else None,
    )


def distill_scale_for(model_name: str, step: int) -> float:
    if model_name == "transformer_flash_sdpa":
        return 0.0
    if "conf" in model_name or model_name == "attentionfree_multiscale":
        return min(1.0, step / 100.0)
    if step <= 100:
        return 0.0
    return min(1.0, (step - 100) / 200.0)


def set_distill_scale(model_name: str, model: nn.Module, step: int) -> None:
    cache = getattr(model, "cache", None)
    if cache is not None:
        cache.distill_scale = distill_scale_for(model_name, step)


def loss_for(model_name: str, model: nn.Module, x: torch.Tensor, y: torch.Tensor):
    if model_name == "transformer_flash_sdpa":
        logits = model(x)
        primary = F.cross_entropy(logits.float().reshape(-1, VOCAB), y.reshape(-1))
        return primary, primary.detach()
    return model.loss_and_stats(x, y, compute_metrics=False)[:2]


def calibration_run(model_name: str, lr: float, args, train, val, device, v3, v4, shapes):
    model = build_model(model_name, args, v3, v4, shapes, device)
    opt = make_optimizer(model, lr, args.weight_decay)
    model.train()
    torch.cuda.empty_cache()
    started = time.perf_counter()
    for step in range(1, args.calibration_steps + 1):
        set_distill_scale(model_name, model, step)
        opt.zero_grad(set_to_none=True)
        for micro in range(args.accum):
            x, y = batch_for_step(
                train, args.batch_size, args.train_seq, args.data_seed,
                step, micro, device, v3,
            )
            with amp_ctx(device, args.amp):
                loss, _ = loss_for(model_name, model, x, y)
                scaled = loss / args.accum
            scaled.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        now_lr = lr_at(step, args.calibration_steps, args.calibration_warmup,
                       lr, args.min_lr_ratio)
        for group in opt.param_groups:
            group["lr"] = now_lr
        opt.step()
    sync(device)
    elapsed = time.perf_counter() - started
    row = evaluate(model_name, model, val, args.train_seq,
                   args.calibration_eval_windows, args.eval_seed, device, args.amp, v3)
    result = {
        "model": model_name, "lr": lr, "bpb": row.bpb,
        "bytes_per_second": args.calibration_steps * args.batch_size * args.accum * args.train_seq / max(elapsed, 1e-9),
    }
    del model, opt
    gc.collect(); torch.cuda.empty_cache()
    return result


def run_calibration(args, train, val, device, v3, v4, shapes, root: Path):
    path = root / "lr_selection.json"
    if path.exists() and args.resume:
        return json.loads(path.read_text())
    groups = {
        "field": "baseline_v5",
        "hybrid": "hybrid_w256_conf",
        "transformer": "transformer_flash_sdpa",
    }
    result: Dict[str, object] = {"runs": [], "selected": {}}
    for family, name in groups.items():
        rows = []
        for lr in args.calibration_lrs:
            log(f"[calibration] family={family} model={name} lr={lr:.3e}")
            row = calibration_run(name, lr, args, train, val, device, v3, v4, shapes)
            rows.append(row); result["runs"].append(row)
            log(f"[calibration] {name} lr={lr:.3e} bpb={row['bpb']:.5f} B/s={row['bytes_per_second']:,.0f}")
        best = min(rows, key=lambda r: r["bpb"])
        result["selected"][family] = best
    atomic_json(path, result)
    return result


def family_for(name: str) -> str:
    if name == "transformer_flash_sdpa":
        return "transformer"
    if name == "baseline_v5":
        return "field"
    return "hybrid"


def checkpoint_signature(args, name: str, shape: ModelShape, lr: float, steps: int) -> dict:
    return {
        "name": name, "params": shape.params, "dim": shape.dim,
        "layers": shape.layers, "ff_hidden": shape.ff_hidden,
        "train_seq": args.train_seq, "batch": args.batch_size,
        "accum": args.accum, "lr": lr, "steps": steps,
        "data_frac": args.data_frac, "model_seed": args.model_seed,
    }


def full_train(name: str, lr: float, args, train, val, device, v3, v4,
               shapes, root: Path) -> dict:
    out = root / "models" / name
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "result.json"
    if result_path.exists() and args.resume:
        return json.loads(result_path.read_text())
    bytes_per_step = args.batch_size * args.accum * args.train_seq
    total_steps = args.train_steps if args.train_steps > 0 else math.ceil(len(train) * args.epochs / bytes_per_step)
    signature = checkpoint_signature(args, name, shapes[name], lr, total_steps)
    model = build_model(name, args, v3, v4, shapes, device)
    opt = make_optimizer(model, lr, args.weight_decay)
    start_step = 0
    ckpt = out / "latest.pt"
    if ckpt.exists() and args.resume:
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        if payload.get("signature") != signature:
            raise RuntimeError(f"checkpoint signature mismatch for {name}")
        model.load_state_dict(payload["model"])
        opt.load_state_dict(payload["optimizer"])
        start_step = int(payload["step"])
        log(f"[{name}] resume step={start_step}/{total_steps}")
    model.train()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    sync(device); started = time.perf_counter(); processed = 0; excluded = 0.0
    best = float("inf")
    for step in range(start_step + 1, total_steps + 1):
        set_distill_scale(name, model, step)
        opt.zero_grad(set_to_none=True)
        primary_acc = 0.0
        for micro in range(args.accum):
            x, y = batch_for_step(train, args.batch_size, args.train_seq,
                                  args.data_seed, step, micro, device, v3)
            with amp_ctx(device, args.amp):
                loss, primary = loss_for(name, model, x, y)
                scaled = loss / args.accum
            scaled.backward()
            primary_acc += float(primary.detach() / LN2)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        now_lr = lr_at(step, total_steps, args.warmup, lr, args.min_lr_ratio)
        for group in opt.param_groups:
            group["lr"] = now_lr
        opt.step()
        processed += bytes_per_step
        if step % args.log_every == 0 or step == total_steps:
            sync(device)
            elapsed = time.perf_counter() - started - excluded
            bps = processed / max(elapsed, 1e-9)
            peak = torch.cuda.max_memory_allocated() / 2**30
            log(f"[{name}] step={step:05d}/{total_steps} primary_bpb={primary_acc/args.accum:.4f} lr={now_lr:.3e} B/s={bps:,.0f} peak={peak:.2f}G")
        if step % args.eval_every == 0 or step == total_steps:
            eval_started = time.perf_counter()
            row = evaluate(name, model, val, 8192, args.quick_eval_windows,
                           args.eval_seed, device, args.amp, v3)
            best = min(best, row.bpb)
            log(f"[{name}] EVAL step={step:05d} bpb8k={row.bpb:.5f} best={best:.5f}")
            model.train()
            excluded += time.perf_counter() - eval_started
        if step % args.save_every == 0 or step == total_steps:
            tmp = ckpt.with_suffix(".tmp")
            torch.save({
                "signature": signature, "step": step,
                "model": model.state_dict(), "optimizer": opt.state_dict(),
            }, tmp)
            os.replace(tmp, ckpt)
    sync(device)
    elapsed = time.perf_counter() - started - excluded
    rows = [
        evaluate(name, model, val, ctx, args.final_eval_windows,
                 args.eval_seed, device, args.amp, v3)
        for ctx in args.final_contexts
    ]
    result = {
        "model": name, "params": shapes[name].params, "lr": lr,
        "steps": total_steps, "best_quick_8k": best,
        "train_bytes_per_second": processed / max(elapsed, 1e-9),
        "train_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "eval": [asdict(r) for r in rows],
        "attention_free": shapes[name].attention_free,
    }
    atomic_json(result_path, result)
    del model, opt
    gc.collect(); torch.cuda.empty_cache()
    return result


def system_benchmark(name: str, args, device, v3, v4, shapes, context: int) -> dict:
    model = build_model(name, args, v3, v4, shapes, device).train()
    set_distill_scale(name, model, 10_000)
    opt = make_optimizer(model, args.system_lr, args.weight_decay)
    batch = max(1, args.system_tokens_per_step // context)
    x = torch.randint(0, VOCAB, (batch, context), device=device)
    y = torch.randint(0, VOCAB, (batch, context), device=device)

    def step_once():
        opt.zero_grad(set_to_none=True)
        with amp_ctx(device, args.amp):
            loss, _ = loss_for(name, model, x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

    status, error = "ok", ""
    bps = ms = peak = None
    try:
        torch.cuda.empty_cache()
        for _ in range(args.system_warmup):
            step_once()
        sync(device); torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
        for _ in range(args.system_steps):
            step_once()
        sync(device); elapsed = time.perf_counter() - start
        bps = args.system_steps * batch * context / max(elapsed, 1e-9)
        ms = elapsed * 1000.0 / args.system_steps
        peak = torch.cuda.max_memory_allocated() / 2**30
    except torch.cuda.OutOfMemoryError as exc:
        status, error = "oom", str(exc)
    except Exception as exc:
        status, error = "error", repr(exc)
    row = {
        "model": name, "context": context, "batch": batch, "status": status,
        "bytes_per_second": bps, "step_ms": ms, "peak_gib": peak, "error": error,
    }
    del model, opt, x, y
    gc.collect(); torch.cuda.empty_cache()
    return row


def run_selftest(args, device, v3, v4, shapes):
    log("[selftest] canonical source and Triton kernel")
    if not getattr(v4, "HAS_TRITON", False):
        raise RuntimeError("canonical source imported without Triton support")
    # Canonical forward/backward test at its own strict tolerances.
    test_args = argparse.Namespace(**vars(args))
    test_args.dim = 64; test_args.heads = 4; test_args.layers = 2
    test_args.field_chunk = 8; test_args.triton_block_c = 8; test_args.triton_chunk_t = 16
    test_args.models = ["field_reference", "field_triton"]
    test_args.checkpoint_blocks = False
    test_args.max_param_delta_pct = 1.0
    # The canonical v4 self-test reads all four tolerance fields from the
    # Namespace passed by the caller.  Keep these explicit so this bridge does
    # not depend on the canonical script's CLI parser having populated them.
    test_args.selftest_forward_tol = 0.002
    test_args.selftest_grad_rel_tol = 0.02
    test_args.selftest_grad_abs_tol = 0.002
    test_args.selftest_causal_tol = 0.0002
    v4.run_kernel_self_test(device, test_args)
    log("[selftest] canonical Field reference/Triton PASS")

    # Paired baseline backbone initialization.
    seed_all(args.model_seed)
    base = build_model("baseline_v5", args, v3, v4, shapes, device)
    seed_all(args.model_seed)
    hybrid = build_model("hybrid_w256_conf", args, v3, v4, shapes, device)
    max_abs = 0.0
    for p, q in zip(base.blocks.parameters(), hybrid.blocks.parameters()):
        max_abs = max(max_abs, float((p - q).abs().max()))
    log(f"[selftest] paired canonical backbone max_abs={max_abs:.3e}")
    if max_abs != 0.0:
        raise AssertionError("paired backbone initialization mismatch")

    # Causality and finite backward on every arm.
    x = torch.randint(0, VOCAB, (1, 96), device=device)
    y = torch.randint(0, VOCAB, (1, 96), device=device)
    for name in FIELD_ARMS:
        model = build_model(name, args, v3, v4, shapes, device).train()
        with amp_ctx(device, args.amp):
            loss, primary = loss_for(name, model, x, y)
        loss.backward()
        finite = bool(torch.isfinite(loss)) and all(
            p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()
        )
        log(f"[selftest] {name} loss={float(loss):.5f} primary={float(primary):.5f} finite={finite}")
        if not finite:
            raise AssertionError(name)
        del model
    # Strict prefix causality on hybrid.
    model = build_model("hybrid_w256_conf", args, v3, v4, shapes, device).eval()
    a = torch.randint(0, VOCAB, (1, 128), device=device)
    b = a.clone(); b[:, 80:] = torch.randint(0, VOCAB, b[:, 80:].shape, device=device)
    with torch.no_grad(), amp_ctx(device, args.amp):
        sa, _ = model.states_logits(a)
        sb, _ = model.states_logits(b)
    err = float((sa[:, :80] - sb[:, :80]).abs().max())
    log(f"[selftest] strict prefix max_abs={err:.3e}")
    if err > 2e-3:
        raise AssertionError("prefix causality failure")
    log("[selftest] PASS")
    del base, hybrid, model, x, y, a, b
    gc.collect(); torch.cuda.empty_cache()


def get_eval(result: dict, context: int) -> dict:
    for row in result["eval"]:
        if int(row["context"]) == context:
            return row
    raise KeyError(context)


def make_summary(args, canonical_path: Path, shapes, selection, results, systems) -> str:
    lines = [
        "=" * 186,
        "CANONICAL TRITON FIELD HYBRID — 50M / WIKITEXT-103 10% / 4K BRIDGE",
        "=" * 186,
        f"canonical_source={canonical_path} sha256={sha256(canonical_path)}",
        f"protocol: ctx={args.train_seq} bytes/update={args.batch_size*args.accum*args.train_seq:,} epochs={args.epochs} BF16 no-checkpoint",
        "",
        "MODEL SHAPES",
        f"{'model':35s} {'params':>12s} {'d%':>8s} {'dim':>5s} {'layers':>6s} {'ff':>6s} {'AF':>4s}",
    ]
    target = shapes["baseline_v5"].params
    for name in MODEL_NAMES:
        s = shapes[name]
        lines.append(f"{name:35s} {s.params:12,d} {100*(s.params-target)/target:+8.3f} {s.dim:5d} {s.layers:6d} {s.ff_hidden:6d} {str(s.attention_free):>4s}")
    lines.extend(["", "LR SELECTION"])
    for fam, row in selection["selected"].items():
        lines.append(f"{fam:12s} model={row['model']:30s} lr={row['lr']:.3e} bpb={row['bpb']:.5f}")
    lines.extend([
        "",
        "FINAL QUALITY",
        f"{'model':35s} {'BPB8K':>9s} {'dBase':>9s} {'BPB16K':>9s} {'dBase':>9s} {'oracle8':>9s} {'cap8':>7s} {'sep8':>7s} {'train B/s':>12s} {'peak':>7s}",
    ])
    base8 = get_eval(results["baseline_v5"], 8192)["bpb"]
    base16 = get_eval(results["baseline_v5"], 16384)["bpb"]
    tf8 = get_eval(results["transformer_flash_sdpa"], 8192)["bpb"]
    for name in MODEL_NAMES:
        r = results[name]; e8 = get_eval(r,8192); e16=get_eval(r,16384)
        oracle = "-" if e8["oracle_bpb"] is None else f"{e8['oracle_bpb']:.5f}"
        cap = "-" if e8["capture"] is None else f"{e8['capture']:.3f}"
        sep = "-" if e8["gate_sep"] is None else f"{e8['gate_sep']:+.3f}"
        lines.append(f"{name:35s} {e8['bpb']:9.5f} {e8['bpb']-base8:+9.5f} {e16['bpb']:9.5f} {e16['bpb']-base16:+9.5f} {oracle:>9s} {cap:>7s} {sep:>7s} {r['train_bytes_per_second']:12,.0f} {r['train_peak_gib']:7.2f}")
    lines.extend([
        "",
        f"Best hybrid vs Transformer @8K: {min(get_eval(results[n],8192)['bpb'] for n in FIELD_ARMS[1:]):.5f} vs {tf8:.5f}",
        "",
        "EQUAL NO-CHECKPOINT SYSTEMS BENCHMARK",
        f"{'model':35s} {'ctx':>6s} {'batch':>6s} {'status':>8s} {'B/s':>12s} {'step ms':>10s} {'peak GB':>8s}",
    ])
    for row in systems:
        bps = "-" if row["bytes_per_second"] is None else f"{row['bytes_per_second']:,.0f}"
        ms = "-" if row["step_ms"] is None else f"{row['step_ms']:.2f}"
        peak = "-" if row["peak_gib"] is None else f"{row['peak_gib']:.2f}"
        lines.append(f"{row['model']:35s} {row['context']:6d} {row['batch']:6d} {row['status']:>8s} {bps:>12s} {ms:>10s} {peak:>8s}")

    hybrids = [n for n in ("hybrid_w256_conf","hybrid_w128_two_conf")]
    best_name = min(hybrids, key=lambda n: get_eval(results[n],8192)["bpb"])
    best8 = get_eval(results[best_name],8192)["bpb"]
    best16 = get_eval(results[best_name],16384)["bpb"]
    parity8 = get_eval(results["hybrid_w256_conf_parity"],8192)["bpb"]
    gain = base8 - best8
    parity_gain = base8 - parity8
    retention = parity_gain / max(gain, 1e-12)
    sys_lookup = {(r["model"],r["context"]):r for r in systems}
    bsys = sys_lookup.get(("baseline_v5",8192),{})
    hsys = sys_lookup.get((best_name,8192),{})
    speed_ratio = (hsys.get("bytes_per_second") or 0)/(bsys.get("bytes_per_second") or 1)
    quality_pass = gain >= args.promotion_gain_8k and (base16-best16) >= args.promotion_gain_16k
    parity_pass = retention >= args.parity_retention
    speed_pass = speed_ratio >= args.promotion_speed
    ready = quality_pass and parity_pass and speed_pass
    lines.extend([
        "",
        "PROMOTION VERDICT",
        f"best_hybrid={best_name} gain8K={gain:.5f} gain16K={base16-best16:.5f} speed8K={speed_ratio:.3f}x",
        f"strict_param_retention={retention:.3f}",
        f"quality_gate={'PASS' if quality_pass else 'FAIL'} | parity_gate={'PASS' if parity_pass else 'FAIL'} | systems_gate={'PASS' if speed_pass else 'FAIL'}",
        "VERDICT: " + ("READY FOR 300M H2H IMPLEMENTATION" if ready else "NOT YET READY FOR 300M"),
        "=" * 186,
    ])
    return "\n".join(lines) + "\n"


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=("selftest","calibrate","train","systems","summary","all"), default="all")
    p.add_argument("--outdir", default="/home/ubuntu/pcaf_runs/field_hybrid_canonical_50m_bridge_v4")
    p.add_argument("--canonical-source", default="")
    p.add_argument("--cache-dir", default="/home/ubuntu/field_lab/hf_cache")
    p.add_argument("--data-frac", type=float, default=0.10)
    p.add_argument("--data-device", choices=("auto","cpu","cuda"), default="auto")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--train-steps", type=int, default=0)
    p.add_argument("--train-seq", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--field-dim", type=int, default=704)
    p.add_argument("--field-layers", type=int, default=8)
    p.add_argument("--field-heads", type=int, default=8)
    p.add_argument("--field-ff-hidden", type=int, default=1920)
    p.add_argument("--field-chunk", type=int, default=32)
    p.add_argument("--triton-block-c", type=int, default=16)
    p.add_argument("--triton-chunk-t", type=int, default=64)
    p.add_argument("--num-buckets", type=int, default=8192)
    p.add_argument("--tf-dim", type=int, default=640)
    p.add_argument("--tf-heads", type=int, default=10)
    p.add_argument("--tf-layers", type=int, default=10)
    p.add_argument("--tf-ff-hidden", type=int, default=1776)
    p.add_argument("--amp", choices=("bf16","fp16","fp32"), default="bf16")
    p.add_argument("--calibration-steps", type=int, default=250)
    p.add_argument("--calibration-warmup", type=int, default=40)
    p.add_argument("--calibration-lrs", type=float, nargs="+", default=[2e-4,3e-4,4e-4])
    p.add_argument("--calibration-eval-windows", type=int, default=4)
    p.add_argument("--warmup", type=int, default=250)
    p.add_argument("--min-lr-ratio", type=float, default=0.10)
    p.add_argument("--weight-decay", type=float, default=0.10)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--quick-eval-windows", type=int, default=4)
    p.add_argument("--final-contexts", type=int, nargs="+", default=[8192,16384])
    p.add_argument("--final-eval-windows", type=int, default=8)
    p.add_argument("--system-contexts", type=int, nargs="+", default=[4096,8192,16384])
    p.add_argument("--system-tokens-per-step", type=int, default=16384)
    p.add_argument("--system-warmup", type=int, default=2)
    p.add_argument("--system-steps", type=int, default=5)
    p.add_argument("--system-lr", type=float, default=3e-4)
    p.add_argument("--model-seed", type=int, default=1234)
    p.add_argument("--data-seed", type=int, default=5678)
    p.add_argument("--eval-seed", type=int, default=9012)
    p.add_argument("--promotion-gain-8k", type=float, default=0.035)
    p.add_argument("--promotion-gain-16k", type=float, default=0.030)
    p.add_argument("--promotion-speed", type=float, default=0.75)
    p.add_argument("--parity-retention", type=float, default=0.70)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/H100 required")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.set_float32_matmul_precision("high")

    if not V3_PATH.is_file():
        raise FileNotFoundError(V3_PATH)
    canonical_path = locate_canonical(args.canonical_source)
    actual = sha256(canonical_path)
    if actual != EXPECTED_CANONICAL_SHA256:
        raise RuntimeError(
            f"canonical SHA mismatch expected={EXPECTED_CANONICAL_SHA256} actual={actual} path={canonical_path}"
        )
    v3 = import_module(V3_PATH, "field_bridge_v3_source")
    v4 = import_module(canonical_path, "field_bridge_canonical_v4")
    root = Path(args.outdir); root.mkdir(parents=True, exist_ok=True)
    shapes, parity_hidden = resolve_shapes(args, v3, v4, device)
    atomic_json(root / "config.json", {
        "args": vars(args), "canonical_source": str(canonical_path),
        "canonical_sha256": actual, "parity_hidden": parity_hidden,
        "shapes": {k: asdict(v) for k,v in shapes.items()},
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    })
    log("="*160)
    log("CANONICAL TRITON FIELD HYBRID 50M BRIDGE v4")
    log(f"gpu={torch.cuda.get_device_name(0)} torch={torch.__version__} cuda={torch.version.cuda}")
    log(f"canonical={canonical_path} sha={actual}")
    for name,s in shapes.items():
        log(f"{name:35s} params={s.params:,} dim={s.dim} layers={s.layers} ff={s.ff_hidden} AF={s.attention_free}")
    log("="*160)

    if args.mode in {"selftest","all"}:
        run_selftest(args, device, v3, v4, shapes)
        if args.mode == "selftest": return

    train = val = None
    if args.mode in {"calibrate","train","all"}:
        train, val, _ = v3.load_wikitext103_raw(args.cache_dir, args.data_frac)
        train = v3.place_data(train, device, args.data_device, "train")
        val = v3.place_data(val, device, args.data_device, "validation")

    selection_path = root / "lr_selection.json"
    if args.mode in {"calibrate","all"}:
        assert train is not None and val is not None
        selection = run_calibration(args, train, val, device, v3, v4, shapes, root)
        if args.mode == "calibrate": return
    else:
        if not selection_path.exists():
            raise FileNotFoundError(selection_path)
        selection = json.loads(selection_path.read_text())

    results: Dict[str,dict] = {}
    if args.mode in {"train","all"}:
        assert train is not None and val is not None
        for name in MODEL_NAMES:
            family = family_for(name)
            lr = float(selection["selected"][family]["lr"])
            results[name] = full_train(name, lr, args, train, val, device, v3, v4, shapes, root)
        atomic_json(root / "all_results.json", results)
        if args.mode == "train": return
    else:
        p = root / "all_results.json"
        if not p.exists(): raise FileNotFoundError(p)
        results = json.loads(p.read_text())

    systems_path = root / "systems.json"
    if args.mode in {"systems","all"}:
        systems = []
        for ctx in args.system_contexts:
            for name in MODEL_NAMES:
                row = system_benchmark(name, args, device, v3, v4, shapes, ctx)
                systems.append(row)
                log(f"[systems] {name} ctx={ctx} status={row['status']} B/s={row['bytes_per_second']} peak={row['peak_gib']}")
        atomic_json(systems_path, systems)
        if args.mode == "systems": return
    else:
        if not systems_path.exists(): raise FileNotFoundError(systems_path)
        systems = json.loads(systems_path.read_text())

    text = make_summary(args, canonical_path, shapes, selection, results, systems)
    atomic_text(root / "SUMMARY.txt", text)
    log(text)


if __name__ == "__main__":
    main()
