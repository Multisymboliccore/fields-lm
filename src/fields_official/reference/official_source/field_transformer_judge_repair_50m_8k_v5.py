from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = 256
LN2 = math.log(2.0)
HERE = Path(__file__).resolve().parent
V3_PATH = HERE / "field_hybrid_attentionfree_qualification_v3.py"
V4_PATH = HERE / "field_hybrid_canonical_50m_bridge_v4.py"
EXPECTED_CANONICAL_SHA256 = "0454647a7f667cdf73f45029eee0f525c58c0788ecd433ec1a4db8c3937ac848"
FIELD_NAMES = ("baseline_v5", "hybrid_w256_conf_parity")
TF_NAME = "transformer_flash_sdpa_strong"
MODEL_NAMES = (*FIELD_NAMES, TF_NAME)


def log(msg: str) -> None:
    print(msg, flush=True)


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def locate_canonical(explicit: str) -> Path:
    candidates = [
        Path(explicit) if explicit else None,
        HERE / "field_only_v4_chunked_triton_wiki100.py",
        Path("/home/ubuntu/field_hybrid_canonical_50m_bridge_v4/field_only_v4_chunked_triton_wiki100.py"),
        Path("/home/ubuntu/field_only_v4_chunked_triton_wiki100.py"),
    ]
    for p in candidates:
        if p and p.is_file():
            return p.resolve()
    raise FileNotFoundError("canonical Field source not found; pass --canonical-source")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def nparams(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def amp_ctx(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


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
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (min_ratio + (1.0 - min_ratio) * cosine)


def batch_for_step(data: torch.Tensor, batch: int, seq: int, seed: int, step: int,
                   micro: int, device: torch.device, v3):
    source = data.device.type if data.device.type == "cuda" else "cpu"
    g = torch.Generator(device=source)
    g.manual_seed(seed + step * 1_000_003 + micro * 97)
    return v3.random_batch(data, batch, seq, g, device)


# Exact native Flash-SDPA contestant used by the validated 300M arena.
_ROPE_CACHE: Dict[Tuple[str, int, torch.dtype, int, int, float], Tuple[torch.Tensor, torch.Tensor]] = {}


def rope_cache(device: torch.device, dtype: torch.dtype, seq: int, head_dim: int,
               position_scale: float = 1.0):
    key = (device.type, device.index or 0, dtype, seq, head_dim, float(position_scale))
    cached = _ROPE_CACHE.get(key)
    if cached is not None:
        return cached
    if head_dim % 2:
        raise ValueError(f"RoPE requires even head_dim, got {head_dim}")
    inv = 1.0 / (10000.0 ** (
        torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
    ))
    pos = torch.arange(seq, device=device, dtype=torch.float32) * float(position_scale)
    freqs = torch.outer(pos, inv)
    cos = freqs.cos().to(dtype)[None, None, :, :]
    sin = freqs.sin().to(dtype)[None, None, :, :]
    _ROPE_CACHE[key] = (cos, sin)
    return cos, sin


def apply_rope(x: torch.Tensor, position_scale: float = 1.0) -> torch.Tensor:
    d = x.shape[-1]
    cos, sin = rope_cache(x.device, x.dtype, x.shape[-2], d, position_scale)
    a, b = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)


class StrongFlashBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_hidden: int, v3) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim={dim} not divisible by heads={heads}")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        if self.head_dim > 256 or self.head_dim % 2:
            raise ValueError(f"unsupported head_dim={self.head_dim}")
        self.norm1 = v3.RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = v3.RMSNorm(dim)
        self.gate_up = nn.Linear(dim, 2 * ff_hidden, bias=False)
        self.down = nn.Linear(ff_hidden, dim, bias=False)

    def forward(self, x: torch.Tensor, position_scale: float = 1.0) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.qkv(self.norm1(x)).view(b, t, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = apply_rope(q.transpose(1, 2), position_scale)
        k = apply_rope(k.transpose(1, 2), position_scale)
        v = v.transpose(1, 2)
        if q.is_cuda:
            try:
                from torch.nn.attention import SDPBackend, sdpa_kernel
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
            except (ImportError, AttributeError):
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True, enable_math=False, enable_mem_efficient=False
                ):
                    a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        else:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        a = a.transpose(1, 2).contiguous().view(b, t, self.dim)
        x = x + self.proj(a)
        g, u = self.gate_up(self.norm2(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(g) * u)


class StrongFlashTransformerLM(nn.Module):
    def __init__(self, dim: int, heads: int, layers: int, ff_hidden: int, v3) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.layers = layers
        self.ff_hidden = ff_hidden
        self.emb = nn.Embedding(VOCAB, dim)
        self.blocks = nn.ModuleList([
            StrongFlashBlock(dim, heads, ff_hidden, v3) for _ in range(layers)
        ])
        self.norm = v3.RMSNorm(dim)
        self.head = nn.Linear(dim, VOCAB, bias=False)

    def forward(self, tokens: torch.Tensor, position_scale: float = 1.0) -> torch.Tensor:
        x = self.emb(tokens)
        for block in self.blocks:
            x = block(x, position_scale)
        return self.head(self.norm(x))


@dataclass(frozen=True)
class TFShape:
    name: str
    dim: int
    heads: int
    layers: int
    ff_hidden: int
    params: int = 0


@dataclass
class EvalRow:
    context: int
    rope_mode: str
    position_scale: float
    bpb: float
    param_bpb: Optional[float] = None
    oracle_bpb: Optional[float] = None
    capture: Optional[float] = None
    gate_sep: Optional[float] = None


def default_tf_shapes(v3) -> List[TFShape]:
    raw = [
        ("wide8", 704, 11, 8, 2048),
        ("balanced10", 640, 10, 10, 1776),
        ("deep12", 576, 9, 12, 1664),
    ]
    out = []
    for name, dim, heads, layers, ff in raw:
        model = StrongFlashTransformerLM(dim, heads, layers, ff, v3)
        out.append(TFShape(name, dim, heads, layers, ff, nparams(model)))
        del model
    return out


def build_field(name: str, args, v3, canonical, bridge):
    # Reuse the exact canonical transplant and strict-parity width search.
    base = bridge.build_field("baseline_v5", args, v3, canonical)
    target = nparams(base)
    del base
    if name == "baseline_v5":
        return bridge.build_field(name, args, v3, canonical)
    hidden = bridge.find_parity_hidden(args, v3, canonical, target)
    return bridge.build_field(name, args, v3, canonical, hidden)


def build_transformer(shape: TFShape, v3):
    return StrongFlashTransformerLM(
        shape.dim, shape.heads, shape.layers, shape.ff_hidden, v3
    )


def build_model(name: str, args, v3, canonical, bridge, tf_shape: TFShape,
                device: torch.device):
    seed_all(args.model_seed)
    if name == TF_NAME:
        return build_transformer(tf_shape, v3).to(device)
    return build_field(name, args, v3, canonical, bridge).to(device)


def set_distill_scale(name: str, model: nn.Module, step: int) -> None:
    if name == TF_NAME:
        return
    cache = getattr(model, "cache", None)
    if cache is not None:
        if "conf" in name:
            cache.distill_scale = min(1.0, step / 100.0)
        else:
            cache.distill_scale = 0.0 if step <= 100 else min(1.0, (step - 100) / 200.0)


def training_loss(name: str, model: nn.Module, x: torch.Tensor, y: torch.Tensor):
    if name == TF_NAME:
        logits = model(x, 1.0)
        primary = F.cross_entropy(logits.float().reshape(-1, VOCAB), y.reshape(-1))
        return primary, primary.detach()
    total, primary, _ = model.loss_and_stats(x, y, compute_metrics=False)
    return total, primary


@torch.no_grad()
def evaluate(name: str, model: nn.Module, data: torch.Tensor, context: int,
             windows: int, seed: int, device: torch.device, amp: str, v3,
             train_context: int, rope_mode: str = "raw") -> EvalRow:
    model.eval()
    starts = v3.fixed_starts(len(data), context, windows, seed + context * 17)
    vals: List[float] = []
    param: List[float] = []
    oracle: List[float] = []
    capture: List[float] = []
    sep: List[float] = []
    scale = 1.0
    if name == TF_NAME and rope_mode == "pi" and context > train_context:
        scale = train_context / context
    for s in starts:
        x, y = v3.fixed_batch(data, [s], context, device)
        with amp_ctx(device, amp):
            if name == TF_NAME:
                logits = model(x, scale)
                loss = F.cross_entropy(logits.float().reshape(-1, VOCAB), y.reshape(-1))
                vals.append(float(loss / LN2))
            else:
                _, primary_loss, stats = model.loss_and_stats(x, y, compute_metrics=True)
                vals.append(float(primary_loss / LN2))
                assert stats is not None
                param.append(stats.param_bpb)
                oracle.append(stats.oracle_bpb)
                capture.append(stats.capture)
                sep.append(stats.gate_separation)
        del x, y
    return EvalRow(
        context=context,
        rope_mode=rope_mode,
        position_scale=scale,
        bpb=float(np.mean(vals)),
        param_bpb=float(np.mean(param)) if param else None,
        oracle_bpb=float(np.mean(oracle)) if oracle else None,
        capture=float(np.mean(capture)) if capture else None,
        gate_sep=float(np.mean(sep)) if sep else None,
    )


def calibration_run(name: str, lr: float, steps: int, shape: TFShape, args,
                    train, val, device, v3, canonical, bridge) -> dict:
    model = build_model(name, args, v3, canonical, bridge, shape, device)
    opt = make_optimizer(model, lr, args.weight_decay)
    model.train()
    sync(device)
    started = time.perf_counter()
    for step in range(1, steps + 1):
        set_distill_scale(name, model, step)
        opt.zero_grad(set_to_none=True)
        for micro in range(args.accum):
            x, y = batch_for_step(
                train, args.batch_size, args.train_seq, args.data_seed,
                step, micro, device, v3,
            )
            with amp_ctx(device, args.amp):
                loss, _ = training_loss(name, model, x, y)
            (loss / args.accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        now_lr = lr_at(step, steps, args.calibration_warmup, lr, args.min_lr_ratio)
        for group in opt.param_groups:
            group["lr"] = now_lr
        opt.step()
    sync(device)
    elapsed = time.perf_counter() - started
    row = evaluate(
        name, model, val, args.train_seq, args.calibration_eval_windows,
        args.eval_seed, device, args.amp, v3, args.train_seq, "raw",
    )
    result = {
        "model": name,
        "shape": asdict(shape),
        "lr": lr,
        "steps": steps,
        "bpb": row.bpb,
        "bytes_per_second": steps * args.batch_size * args.accum * args.train_seq / max(elapsed, 1e-9),
    }
    del model, opt
    gc.collect(); torch.cuda.empty_cache()
    return result


def run_selection(args, train, val, device, v3, canonical, bridge, root: Path,
                  tf_shapes: List[TFShape]) -> dict:
    path = root / "selection.json"
    if path.exists() and args.resume:
        return json.loads(path.read_text())
    result = {"shape_screen": [], "lr_runs": [], "selected": {}}

    # Shape screen at the known-good central LR.
    for shape in tf_shapes:
        log(f"[shape-screen] {shape.name} params={shape.params:,} lr={args.shape_screen_lr:.2e}")
        row = calibration_run(
            TF_NAME, args.shape_screen_lr, args.shape_screen_steps, shape,
            args, train, val, device, v3, canonical, bridge,
        )
        result["shape_screen"].append(row)
        log(f"[shape-screen] {shape.name} bpb={row['bpb']:.5f} B/s={row['bytes_per_second']:,.0f}")
    best_shape_row = min(result["shape_screen"], key=lambda x: x["bpb"])
    best_shape_name = best_shape_row["shape"]["name"]
    best_shape = next(s for s in tf_shapes if s.name == best_shape_name)

    # Wider LR sweep for the winning shape, always from fresh initialization.
    tf_rows = []
    for lr in args.transformer_lrs:
        log(f"[lr-screen] transformer shape={best_shape.name} lr={lr:.2e}")
        row = calibration_run(
            TF_NAME, lr, args.lr_screen_steps, best_shape,
            args, train, val, device, v3, canonical, bridge,
        )
        tf_rows.append(row); result["lr_runs"].append(row)
        log(f"[lr-screen] transformer lr={lr:.2e} bpb={row['bpb']:.5f}")
    result["selected"]["transformer"] = min(tf_rows, key=lambda x: x["bpb"])

    # Each Field family receives its own small fair LR screen at 8K.
    for name, family in [("baseline_v5", "field"), ("hybrid_w256_conf_parity", "hybrid")]:
        rows = []
        for lr in args.field_lrs:
            log(f"[lr-screen] {name} lr={lr:.2e}")
            row = calibration_run(
                name, lr, args.field_lr_screen_steps, best_shape,
                args, train, val, device, v3, canonical, bridge,
            )
            rows.append(row); result["lr_runs"].append(row)
            log(f"[lr-screen] {name} lr={lr:.2e} bpb={row['bpb']:.5f}")
        result["selected"][family] = min(rows, key=lambda x: x["bpb"])

    atomic_json(path, result)
    return result


def family_for(name: str) -> str:
    if name == TF_NAME:
        return "transformer"
    if name == "baseline_v5":
        return "field"
    return "hybrid"


def checkpoint_signature(args, name: str, params: int, lr: float, total_steps: int,
                         tf_shape: TFShape) -> dict:
    return {
        "name": name, "params": params, "lr": lr, "steps": total_steps,
        "train_seq": args.train_seq, "batch": args.batch_size, "accum": args.accum,
        "data_frac": args.data_frac, "model_seed": args.model_seed,
        "tf_shape": asdict(tf_shape),
    }


def full_train(name: str, lr: float, tf_shape: TFShape, args, train, val, device,
               v3, canonical, bridge, root: Path) -> dict:
    out = root / "models" / name
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "result.json"
    if result_path.exists() and args.resume:
        return json.loads(result_path.read_text())
    bytes_per_step = args.batch_size * args.accum * args.train_seq
    total_steps = args.train_steps if args.train_steps > 0 else math.ceil(
        len(train) * args.epochs / bytes_per_step
    )
    model = build_model(name, args, v3, canonical, bridge, tf_shape, device)
    params = nparams(model)
    signature = checkpoint_signature(args, name, params, lr, total_steps, tf_shape)
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
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    sync(device); started = time.perf_counter(); processed = 0
    best = float("inf")
    for step in range(start_step + 1, total_steps + 1):
        set_distill_scale(name, model, step)
        opt.zero_grad(set_to_none=True)
        primary_sum = 0.0
        for micro in range(args.accum):
            x, y = batch_for_step(
                train, args.batch_size, args.train_seq, args.data_seed,
                step, micro, device, v3,
            )
            with amp_ctx(device, args.amp):
                loss, primary = training_loss(name, model, x, y)
            (loss / args.accum).backward()
            primary_sum += float(primary / LN2)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        now_lr = lr_at(step, total_steps, args.warmup, lr, args.min_lr_ratio)
        for group in opt.param_groups:
            group["lr"] = now_lr
        opt.step()
        processed += bytes_per_step
        if step % args.log_every == 0 or step == total_steps:
            sync(device)
            elapsed = time.perf_counter() - started
            bps = processed / max(elapsed, 1e-9)
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            log(f"[{name}] step={step:05d}/{total_steps} primary_bpb={primary_sum/args.accum:.4f} lr={now_lr:.3e} B/s={bps:,.0f} peak={peak:.2f}G")
        if step % args.eval_every == 0 or step == total_steps:
            row = evaluate(
                name, model, val, args.train_seq, args.quick_eval_windows,
                args.eval_seed, device, args.amp, v3, args.train_seq, "raw",
            )
            best = min(best, row.bpb)
            log(f"[{name}] EVAL8K step={step:05d} bpb={row.bpb:.5f} best={best:.5f}")
        if step % args.save_every == 0 or step == total_steps:
            torch.save({
                "signature": signature, "step": step,
                "model": model.state_dict(), "optimizer": opt.state_dict(),
            }, ckpt)
    sync(device)
    elapsed = time.perf_counter() - started
    train_bps = processed / max(elapsed, 1e-9)
    peak = torch.cuda.max_memory_allocated(device) / 2**30

    eval_rows = []
    for ctx in args.final_contexts:
        eval_rows.append(asdict(evaluate(
            name, model, val, ctx, args.final_eval_windows,
            args.eval_seed, device, args.amp, v3, args.train_seq, "raw",
        )))
        if name == TF_NAME and ctx > args.train_seq:
            eval_rows.append(asdict(evaluate(
                name, model, val, ctx, args.final_eval_windows,
                args.eval_seed, device, args.amp, v3, args.train_seq, "pi",
            )))
    result = {
        "name": name, "params": params, "lr": lr, "steps": total_steps,
        "best_8k": best, "train_bytes_per_second": train_bps,
        "peak_gib": peak, "eval": eval_rows,
    }
    atomic_json(result_path, result)
    del model, opt
    gc.collect(); torch.cuda.empty_cache()
    return result


def system_benchmark(name: str, args, device, v3, canonical, bridge,
                     tf_shape: TFShape, context: int) -> dict:
    batch = max(1, args.system_tokens_per_step // context)
    try:
        model = build_model(name, args, v3, canonical, bridge, tf_shape, device).train()
        opt = make_optimizer(model, args.system_lr, args.weight_decay)
        x = torch.randint(0, VOCAB, (batch, context), device=device)
        y = torch.randint(0, VOCAB, (batch, context), device=device)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        for _ in range(args.system_warmup):
            opt.zero_grad(set_to_none=True)
            with amp_ctx(device, args.amp):
                loss, _ = training_loss(name, model, x, y)
            loss.backward(); opt.step()
        sync(device); started = time.perf_counter()
        for _ in range(args.system_steps):
            opt.zero_grad(set_to_none=True)
            with amp_ctx(device, args.amp):
                loss, _ = training_loss(name, model, x, y)
            loss.backward(); opt.step()
        sync(device)
        elapsed = time.perf_counter() - started
        row = {
            "model": name, "context": context, "batch": batch, "status": "ok",
            "bytes_per_second": args.system_steps * batch * context / elapsed,
            "step_ms": elapsed * 1000 / args.system_steps,
            "peak_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        }
        del model, opt, x, y
        gc.collect(); torch.cuda.empty_cache()
        return row
    except torch.cuda.OutOfMemoryError:
        gc.collect(); torch.cuda.empty_cache()
        return {"model": name, "context": context, "batch": batch, "status": "oom"}


def selftest(args, device, v3, canonical, bridge, tf_shapes: List[TFShape]) -> None:
    # Canonical Triton conformance.
    test_args = argparse.Namespace(**vars(args))
    test_args.selftest_forward_tol = 0.002
    test_args.selftest_grad_rel_tol = 0.02
    test_args.selftest_grad_abs_tol = 0.002
    test_args.selftest_causal_tol = 0.0002
    test_args.dim = args.field_dim
    test_args.layers = 1
    test_args.ff_hidden = args.field_ff_hidden
    test_args.chunk = args.field_chunk
    test_args.backend = "triton"
    canonical.run_kernel_self_test(device, test_args)
    log("[selftest] canonical Triton PASS")

    # All candidate shapes must be finite and use the fused Flash kernel on H100.
    x = torch.randint(0, VOCAB, (1, 256), device=device)
    y = torch.randint(0, VOCAB, (1, 256), device=device)
    for shape in tf_shapes:
        seed_all(args.model_seed)
        model = build_transformer(shape, v3).to(device).train()
        with amp_ctx(device, args.amp):
            logits = model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, VOCAB), y.reshape(-1))
        loss.backward()
        finite = bool(torch.isfinite(loss)) and all(
            p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()
        )
        log(f"[selftest] transformer {shape.name} params={shape.params:,} loss={float(loss):.5f} finite={finite}")
        if not finite:
            raise AssertionError(shape.name)
        del model

    # Causal-prefix invariance must compare equal total sequence lengths.
    # FlashAttention may choose different BF16 tilings for T=512 and T=1024,
    # producing harmless round-off differences that are amplified by depth.
    # Keeping T fixed while changing only the future suffix catches real
    # future-token leakage without conflating it with kernel-shape numerics.
    shape = tf_shapes[0]
    seed_all(args.model_seed)
    model = build_transformer(shape, v3).to(device).eval()
    prefix = torch.randint(0, VOCAB, (1, 512), device=device)
    suffix_a = torch.randint(0, VOCAB, (1, 512), device=device)
    suffix_b = torch.randint(0, VOCAB, (1, 512), device=device)
    seq_a = torch.cat([prefix, suffix_a], dim=1)
    seq_b = torch.cat([prefix, suffix_b], dim=1)
    with torch.no_grad(), amp_ctx(device, args.amp):
        logits_a = model(seq_a)[:, :512]
        logits_b = model(seq_b)[:, :512]
    causal_err = float((logits_a - logits_b).abs().max())
    log(f"[selftest] transformer equal-shape suffix perturbation max_abs={causal_err:.3e}")
    if causal_err > 5e-3:
        raise AssertionError("transformer causal prefix invariance failure")

    # Non-fatal diagnostic: different sequence lengths can legitimately differ
    # slightly in BF16 because the fused Flash kernel changes its tiling.
    with torch.no_grad(), amp_ctx(device, args.amp):
        logits_short = model(prefix)
        logits_long = model(seq_a)[:, :512]
    shape_err = float((logits_short - logits_long).abs().max())
    log(f"[selftest] transformer cross-shape BF16 diagnostic max_abs={shape_err:.3e} (non-fatal)")

    del model, x, y, prefix, suffix_a, suffix_b, seq_a, seq_b, logits_a, logits_b, logits_short, logits_long
    gc.collect(); torch.cuda.empty_cache()
    log("[selftest] PASS")


def find_eval(result: dict, context: int, rope_mode: str = "raw") -> dict:
    for row in result["eval"]:
        if int(row["context"]) == context and row["rope_mode"] == rope_mode:
            return row
    raise KeyError((context, rope_mode))


def make_summary(args, canonical_path: Path, tf_shape: TFShape, selection,
                 results: Dict[str, dict], systems: List[dict]) -> str:
    lines = [
        "=" * 190,
        "CORRECTED 50M JUDGE — CANONICAL FIELD HYBRID vs STRONG FLASH-SDPA — 8K TRAINING",
        "=" * 190,
        f"canonical_source={canonical_path} sha256={sha256(canonical_path)}",
        f"protocol: WikiText-103 {args.data_frac*100:.1f}% | train ctx={args.train_seq} | bytes/update={args.batch_size*args.accum*args.train_seq:,} | epochs={args.epochs} | BF16 | no checkpoint",
        "Primary score is raw BPB@8K, inside the trained context. 16K raw and position-interpolated RoPE are diagnostics only.",
        "",
        "TRANSFORMER SELECTION",
    ]
    for row in selection["shape_screen"]:
        s = row["shape"]
        lines.append(f"shape={s['name']:12s} params={s['params']:,} dim={s['dim']} heads={s['heads']} layers={s['layers']} ff={s['ff_hidden']} bpb={row['bpb']:.5f}")
    sel_tf = selection["selected"]["transformer"]
    lines += [
        f"selected shape={tf_shape.name} params={tf_shape.params:,} lr={float(sel_tf['lr']):.3e} calibration_bpb={float(sel_tf['bpb']):.5f}",
        "",
        "FINAL QUALITY",
        f"{'model':36s} {'params':>12s} {'LR':>10s} {'BPB4K':>9s} {'BPB8K':>9s} {'BPB16K':>9s} {'16K-PI':>9s} {'train B/s':>12s} {'peak':>7s}",
    ]
    for name in MODEL_NAMES:
        r = results[name]
        b4 = find_eval(r, 4096)["bpb"]
        b8 = find_eval(r, 8192)["bpb"]
        b16 = find_eval(r, 16384)["bpb"]
        pi = find_eval(r, 16384, "pi")["bpb"] if name == TF_NAME else None
        lines.append(
            f"{name:36s} {int(r['params']):12,d} {float(r['lr']):10.3e} {b4:9.5f} {b8:9.5f} {b16:9.5f} "
            f"{(f'{pi:.5f}' if pi is not None else '-'):>9s} {float(r['train_bytes_per_second']):12,.0f} {float(r['peak_gib']):7.2f}"
        )
    base8 = find_eval(results["baseline_v5"], 8192)["bpb"]
    hybrid8 = find_eval(results["hybrid_w256_conf_parity"], 8192)["bpb"]
    tf8 = find_eval(results[TF_NAME], 8192)["bpb"]
    lines += [
        "",
        "QUALITY DELTAS @8K (negative is first model better)",
        f"hybrid - baseline:    {hybrid8-base8:+.5f} BPB",
        f"hybrid - transformer: {hybrid8-tf8:+.5f} BPB",
        f"baseline - transformer:{base8-tf8:+.5f} BPB",
        "",
        "EQUAL NO-CHECKPOINT SYSTEMS BENCHMARK",
        f"{'model':36s} {'ctx':>6s} {'batch':>6s} {'status':>8s} {'B/s':>12s} {'step ms':>10s} {'peak GB':>9s}",
    ]
    for row in systems:
        if row["status"] == "ok":
            lines.append(f"{row['model']:36s} {int(row['context']):6d} {int(row['batch']):6d} {'ok':>8s} {float(row['bytes_per_second']):12,.0f} {float(row['step_ms']):10.2f} {float(row['peak_gib']):9.2f}")
        else:
            lines.append(f"{row['model']:36s} {int(row['context']):6d} {int(row['batch']):6d} {row['status']:>8s}")
    verdict = "HYBRID WINS CORRECTED 50M QUALITY JUDGE" if hybrid8 < tf8 else "TRANSFORMER WINS CORRECTED 50M QUALITY JUDGE"
    lines += [
        "",
        "VERDICT",
        verdict,
        "This result is the valid 50M bridge because the primary 8K score is evaluated inside the 8K training context and all contestants use fresh initialization, equal data windows, equal bytes/update and near-equal parameters.",
        "=" * 190,
    ]
    return "\n".join(lines) + "\n"


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=("selftest", "select", "train", "systems", "summary", "all"), default="all")
    p.add_argument("--outdir", default="/home/ubuntu/pcaf_runs/field_transformer_judge_repair_50m_8k_v5")
    p.add_argument("--canonical-source", default="")
    p.add_argument("--cache-dir", default="/home/ubuntu/field_lab/hf_cache")
    p.add_argument("--data-frac", type=float, default=0.10)
    p.add_argument("--data-device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--train-steps", type=int, default=0)
    p.add_argument("--train-seq", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--field-dim", type=int, default=704)
    p.add_argument("--field-layers", type=int, default=8)
    p.add_argument("--field-heads", type=int, default=8)
    p.add_argument("--field-ff-hidden", type=int, default=1920)
    p.add_argument("--field-chunk", type=int, default=32)
    p.add_argument("--triton-block-c", type=int, default=16)
    p.add_argument("--triton-chunk-t", type=int, default=64)
    p.add_argument("--num-buckets", type=int, default=8192)
    p.add_argument("--amp", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--shape-screen-steps", type=int, default=250)
    p.add_argument("--shape-screen-lr", type=float, default=4e-4)
    p.add_argument("--lr-screen-steps", type=int, default=300)
    p.add_argument("--field-lr-screen-steps", type=int, default=250)
    p.add_argument("--transformer-lrs", type=float, nargs="+", default=[3e-4, 4e-4, 5e-4, 6e-4])
    p.add_argument("--field-lrs", type=float, nargs="+", default=[3e-4, 4e-4, 5e-4])
    p.add_argument("--calibration-warmup", type=int, default=40)
    p.add_argument("--calibration-eval-windows", type=int, default=4)
    p.add_argument("--warmup", type=int, default=250)
    p.add_argument("--min-lr-ratio", type=float, default=0.10)
    p.add_argument("--weight-decay", type=float, default=0.10)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--quick-eval-windows", type=int, default=4)
    p.add_argument("--final-contexts", type=int, nargs="+", default=[4096, 8192, 16384])
    p.add_argument("--final-eval-windows", type=int, default=8)
    p.add_argument("--system-contexts", type=int, nargs="+", default=[4096, 8192, 16384])
    p.add_argument("--system-tokens-per-step", type=int, default=16384)
    p.add_argument("--system-warmup", type=int, default=2)
    p.add_argument("--system-steps", type=int, default=5)
    p.add_argument("--system-lr", type=float, default=3e-4)
    p.add_argument("--model-seed", type=int, default=1234)
    p.add_argument("--data-seed", type=int, default=5678)
    p.add_argument("--eval-seed", type=int, default=9012)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/H100 required")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    torch.set_float32_matmul_precision("high")

    canonical_path = locate_canonical(args.canonical_source)
    actual = sha256(canonical_path)
    if actual != EXPECTED_CANONICAL_SHA256:
        raise RuntimeError(f"canonical SHA mismatch expected={EXPECTED_CANONICAL_SHA256} actual={actual}")
    v3 = import_module(V3_PATH, "judge_v3_source")
    bridge = import_module(V4_PATH, "judge_v4_bridge")
    canonical = import_module(canonical_path, "judge_canonical_field")
    root = Path(args.outdir); root.mkdir(parents=True, exist_ok=True)
    tf_shapes = default_tf_shapes(v3)
    atomic_json(root / "config.json", {
        "args": vars(args), "canonical_source": str(canonical_path),
        "canonical_sha256": actual, "tf_shapes": [asdict(x) for x in tf_shapes],
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    })
    log("=" * 160)
    log("CORRECTED 50M JUDGE v5 — 8K TRAINING")
    log(f"gpu={torch.cuda.get_device_name(0)} torch={torch.__version__} cuda={torch.version.cuda}")
    for s in tf_shapes:
        log(f"TF candidate {s.name}: params={s.params:,} dim={s.dim} heads={s.heads} layers={s.layers} ff={s.ff_hidden}")
    log("=" * 160)

    if args.mode in {"selftest", "all"}:
        selftest(args, device, v3, canonical, bridge, tf_shapes)
        if args.mode == "selftest":
            return

    train = val = None
    if args.mode in {"select", "train", "all"}:
        train, val, _ = v3.load_wikitext103_raw(args.cache_dir, args.data_frac)
        train = v3.place_data(train, device, args.data_device, "train")
        val = v3.place_data(val, device, args.data_device, "validation")

    selection_path = root / "selection.json"
    if args.mode in {"select", "all"}:
        selection = run_selection(
            args, train, val, device, v3, canonical, bridge, root, tf_shapes
        )
        if args.mode == "select":
            return
    else:
        if not selection_path.exists():
            raise FileNotFoundError(selection_path)
        selection = json.loads(selection_path.read_text())

    selected_shape_name = selection["selected"]["transformer"]["shape"]["name"]
    tf_shape = next(s for s in tf_shapes if s.name == selected_shape_name)

    results_path = root / "all_results.json"
    if args.mode in {"train", "all"}:
        results = {}
        for name in MODEL_NAMES:
            family = family_for(name)
            lr = float(selection["selected"][family]["lr"])
            results[name] = full_train(
                name, lr, tf_shape, args, train, val, device,
                v3, canonical, bridge, root,
            )
        atomic_json(results_path, results)
        if args.mode == "train":
            return
    else:
        if not results_path.exists():
            raise FileNotFoundError(results_path)
        results = json.loads(results_path.read_text())

    systems_path = root / "systems.json"
    if args.mode in {"systems", "all"}:
        systems = []
        for ctx in args.system_contexts:
            for name in MODEL_NAMES:
                row = system_benchmark(
                    name, args, device, v3, canonical, bridge, tf_shape, ctx
                )
                systems.append(row)
                log(f"[systems] {name} ctx={ctx} status={row['status']} B/s={row.get('bytes_per_second')} peak={row.get('peak_gib')}")
        atomic_json(systems_path, systems)
        if args.mode == "systems":
            return
    else:
        if not systems_path.exists():
            raise FileNotFoundError(systems_path)
        systems = json.loads(systems_path.read_text())

    text = make_summary(args, canonical_path, tf_shape, selection, results, systems)
    atomic_text(root / "SUMMARY.txt", text)
    log(text)


if __name__ == "__main__":
    main()
