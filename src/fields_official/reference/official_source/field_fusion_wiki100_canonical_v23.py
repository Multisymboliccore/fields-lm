#!/usr/bin/env python3
"""FIELD-FUSION CANONICAL 300M v23 — WikiText-103 100% fixed-token run.

This is the promoted two-arm run from v22:
  * field_fusion_fast: packed 20-Field/4-refresh Fusion, target-only Fast PCAF,
    and exact field_half activation recomputation during training.
  * transformer_flash_300m: matched-parameter Flash-SDPA Transformer control.

Protocol defaults:
  * shared 16,384 byte-level BPE tokenizer;
  * WikiText-103 100% (dataset fraction changes diversity, not step count);
  * context 2,048, physical batch 4, accumulation 1;
  * 8,192 tokens/update and 12,000 updates = 98,304,000 tokens/model;
  * paired initialization seeds and identical deterministic token windows;
  * BF16, fused AdamW, tied embeddings, fixed cosine schedule;
  * exact resume checkpoints, milestone validation, full final test,
    matched-suffix long-context evaluation, and final system sweep.

Checkpoint I/O and evaluation are excluded from compute throughput. Wall-clock
throughput is also reported. The output archive excludes large token tensors,
optimizer checkpoints, and final model weights; those remain on the H100.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import field_fusion_quality_memory_gate_v22 as v22

v21 = v22.v21
v20 = v22.v20
base = v22.base
core = v22.core
Shape = v22.Shape
Corpus = v22.Corpus
HERE = Path(__file__).resolve().parent
EXPECTED_CANONICAL_SHA256 = v22.EXPECTED_CANONICAL_SHA256

FUSION = "field_fusion_fast"
TRANSFORMER = "transformer_flash_300m"
MODELS = (FUSION, TRANSFORMER)
TO_V22 = {FUSION: v22.FAST, TRANSFORMER: v22.TRANSFORMER}


def log(x: object = "") -> None:
    print(str(x), flush=True)


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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def tensor_hash(t: torch.Tensor) -> str:
    x = t.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str(tuple(x.shape)).encode())
    h.update(str(x.dtype).encode())
    h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def amp_ctx(device: torch.device, amp: str):
    return v22.amp_ctx(device, amp)


def mapped(name: str) -> str:
    return TO_V22[name]


def set_distill(model: nn.Module, value: float) -> None:
    v22.set_distill(model, value)


def build_model(name: str, shape: Shape, args, deps, device: torch.device) -> nn.Module:
    policy = args.fusion_checkpoint_policy if name == FUSION else "none"
    return v22.build_model(mapped(name), shape, args, deps, device, policy)


def loss_call(name: str, model: nn.Module, x: torch.Tensor, y: torch.Tensor):
    return v22.loss_call(mapped(name), model, x, y)


def make_optimizer(model: nn.Module, args):
    return core.make_optimizer(model, args.lr, args.weight_decay)


def batch_for_step(data: torch.Tensor, batch: int, seq: int, seed: int,
                   step: int, micro: int, device: torch.device):
    return core.batch_for_step(data, batch, seq, seed, step, micro, device)


def lr_at(step: int, total: int, warmup: int, peak: float, min_ratio: float) -> float:
    return core.lr_at(step, total, warmup, peak, min_ratio)


def solve_shapes(args, deps) -> Dict[str, Shape]:
    arena, v3, canonical, bridge, optmod, epi, judge = deps
    out: Dict[str, Shape] = {}
    for name in MODELS:
        raw = v21.solve_shape(
            mapped(name), args, arena, v3, canonical, bridge, optmod, epi, judge
        )
        shape = Shape(name, raw.params, raw.dim, raw.layers, raw.heads, raw.ff_hidden)
        delta = 100.0 * (shape.params - args.target_params) / args.target_params
        if abs(delta) > args.max_param_delta_pct:
            raise RuntimeError(f"parameter mismatch {name}: {delta:+.3f}%")
        out[name] = shape
    return out


def token_nll(name: str, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if name == TRANSFORMER:
        logits = model(x)
        return F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            y.reshape(-1), reduction="none",
        ).view_as(y)
    states, logits = model.states_logits(x)
    return model.cache.token_nll(states, logits, x, y).float()


@torch.no_grad()
def evaluate_stream(name: str, model: nn.Module, corpus: Corpus,
                    data: torch.Tensor, context: int, token_budget: int,
                    device: torch.device, amp: str) -> Dict[str, float]:
    model.eval()
    usable = min(len(data) - 1, token_budget if token_budget > 0 else len(data) - 1)
    total_nll = 0.0
    total_tokens = 0
    for start in range(0, usable, context):
        length = min(context, usable - start)
        if length < 8:
            break
        win = data[start:start + length + 1].long()
        if win.device != device:
            win = win.to(device, non_blocking=True)
        x, y = win[:-1][None], win[1:][None]
        with amp_ctx(device, amp):
            nll = token_nll(name, model, x, y)
        total_nll += float(nll.sum().detach().cpu())
        total_tokens += int(nll.numel())
    mean = total_nll / max(total_tokens, 1)
    bytes_est = total_tokens * corpus.bytes_per_token
    return {
        "context": int(context), "nll": mean,
        "ppl": math.exp(min(mean, 20.0)),
        "bits_per_token": mean / math.log(2.0),
        "bpb_norm": (total_nll / math.log(2.0)) / max(bytes_est, 1e-9),
        "tokens": total_tokens, "bytes_est": bytes_est,
    }


@torch.no_grad()
def evaluate_fixed_windows(name: str, model: nn.Module, data: torch.Tensor,
                           context: int, windows: int, seed: int,
                           device: torch.device, amp: str) -> Dict[str, float]:
    model.eval()
    rng = np.random.default_rng(seed + context * 1009)
    starts = rng.integers(0, len(data) - context - 1, size=windows).tolist()
    total = 0.0
    count = 0
    for start in starts:
        win = data[start:start + context + 1].long()
        if win.device != device:
            win = win.to(device, non_blocking=True)
        x, y = win[:-1][None], win[1:][None]
        with amp_ctx(device, amp):
            nll = token_nll(name, model, x, y)
        total += float(nll.sum().detach().cpu())
        count += int(nll.numel())
    mean = total / max(count, 1)
    return {
        "context": int(context), "nll": mean,
        "ppl": math.exp(min(mean, 20.0)), "tokens": count,
    }


@torch.no_grad()
def evaluate_matched_suffix(name: str, model: nn.Module, data: torch.Tensor,
                            contexts: Sequence[int], score_tokens: int,
                            windows: int, seed: int, device: torch.device,
                            amp: str) -> List[Dict[str, float]]:
    max_ctx = max(contexts)
    if score_tokens >= min(contexts):
        raise ValueError("matched score_tokens must be smaller than every context")
    if len(data) <= max_ctx + 2:
        raise ValueError("test corpus too short for matched-suffix evaluation")
    rng = np.random.default_rng(seed)
    ends = rng.integers(max_ctx + 1, len(data) - 1, size=windows).tolist()
    rows: List[Dict[str, float]] = []
    model.eval()
    for context in contexts:
        total = 0.0
        count = 0
        for end in ends:
            start = end - int(context)
            win = data[start:end + 1].long()
            if win.device != device:
                win = win.to(device, non_blocking=True)
            x, y = win[:-1][None], win[1:][None]
            with amp_ctx(device, amp):
                nll = token_nll(name, model, x, y)[:, -score_tokens:]
            total += float(nll.sum().detach().cpu())
            count += int(nll.numel())
        mean = total / max(count, 1)
        rows.append({
            "context": int(context), "score_tokens": int(score_tokens),
            "windows": int(windows), "nll": mean,
            "ppl": math.exp(min(mean, 20.0)),
            "bits_per_token": mean / math.log(2.0),
        })
    return rows


def checkpoint_signature(args, name: str, shape: Shape) -> Dict[str, object]:
    return {
        "version": 23, "name": name, "shape": asdict(shape),
        "vocab_size": args.vocab_size, "train_seq": args.train_seq,
        "batch_size": args.batch_size, "accum": args.accum,
        "steps": args.steps, "data_frac": args.data_frac,
        "model_seed": args.model_seed, "embedding_seed": args.embedding_seed,
        "data_seed": args.data_seed,
        "checkpoint_policy": args.fusion_checkpoint_policy if name == FUSION else "none",
    }


def save_checkpoint(path: Path, model: nn.Module, optimizer, step: int,
                    history: List[Dict[str, object]], best_val: float,
                    processed_tokens: int, compute_seconds: float,
                    wall_seconds: float, signature: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    payload = {
        "signature": dict(signature), "step": int(step),
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "history": history, "best_validation_nll": float(best_val),
        "processed_tokens": int(processed_tokens),
        "compute_seconds": float(compute_seconds),
        "wall_seconds": float(wall_seconds),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer,
                    signature: Mapping[str, object]) -> Optional[Dict[str, object]]:
    if not path.is_file():
        return None
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if raw.get("signature") != dict(signature):
        raise RuntimeError(f"checkpoint signature mismatch: {path}")
    model.load_state_dict(raw["model"], strict=True)
    optimizer.load_state_dict(raw["optimizer"])
    if raw.get("torch_rng") is not None:
        torch.set_rng_state(raw["torch_rng"])
    if torch.cuda.is_available() and raw.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(raw["cuda_rng"])
    return raw


@dataclass
class TrainResult:
    model: str
    params: int
    steps: int
    train_tokens: int
    compute_seconds: float
    wall_seconds: float
    compute_tokens_per_second: float
    wall_tokens_per_second: float
    peak_gib: float
    best_validation_nll: float
    final_validation: Dict[str, float]
    final_test: Dict[str, float]
    matched_suffix: List[Dict[str, float]]
    history: List[Dict[str, object]]
    final_weights: str


def train_arm(name: str, shape: Shape, args, deps,
              train: torch.Tensor, val_c: Corpus, val: torch.Tensor,
              test_c: Corpus, test: torch.Tensor, root: Path,
              device: torch.device) -> TrainResult:
    out = root / "models" / name
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "result.json"
    if result_path.is_file() and args.resume:
        raw = json.loads(result_path.read_text(encoding="utf-8"))
        log(f"[{name}] completed result found; skipping training")
        return TrainResult(**raw)

    model = build_model(name, shape, args, deps, device).train()
    optimizer = make_optimizer(model, args)
    signature = checkpoint_signature(args, name, shape)
    checkpoint_path = out / "latest.pt"
    history: List[Dict[str, object]] = []
    best_val = float("inf")
    processed = 0
    prior_compute = 0.0
    prior_wall = 0.0
    start_step = 1

    if args.resume:
        raw = load_checkpoint(checkpoint_path, model, optimizer, signature)
        if raw is not None:
            start_step = int(raw["step"]) + 1
            history = list(raw.get("history", []))
            best_val = float(raw.get("best_validation_nll", float("inf")))
            processed = int(raw.get("processed_tokens", 0))
            prior_compute = float(raw.get("compute_seconds", 0.0))
            prior_wall = float(raw.get("wall_seconds", 0.0))
            log(f"[{name}] resumed from step={start_step-1} tokens={processed:,}")

    milestones = set(int(x) for x in args.milestones if int(x) <= args.steps)
    milestones.add(args.steps)
    primary_acc = torch.zeros((), device=device, dtype=torch.float32)
    grad_tensor = torch.zeros((), device=device, dtype=torch.float32)

    clear_cuda()
    torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    wall_started = time.perf_counter()
    compute_started = time.perf_counter()
    excluded = 0.0

    if start_step > args.steps:
        log(f"[{name}] checkpoint already at requested final step")

    for step in range(start_step, args.steps + 1):
        set_distill(model, min(1.0, step / max(args.distill_ramp, 1)))
        optimizer.zero_grad(set_to_none=True)
        primary_acc.zero_()
        for micro in range(args.accum):
            x, y = batch_for_step(
                train, args.batch_size, args.train_seq, args.data_seed,
                step, micro, device,
            )
            with amp_ctx(device, args.amp):
                loss, primary = loss_call(name, model, x, y)
                scaled = loss / args.accum
            scaled.backward()
            primary_acc.add_(primary.detach().float() / args.accum)
        grad_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        now_lr = lr_at(step, args.steps, args.warmup, args.lr, args.min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = now_lr
        optimizer.step()
        processed += args.batch_size * args.accum * args.train_seq

        do_log = step % args.log_every == 0 or step in milestones
        if do_log:
            sync(device)
            current_compute = prior_compute + time.perf_counter() - compute_started - excluded
            nll = float(primary_acc.detach().cpu())
            grad = float(grad_tensor.detach().cpu())
            row = {
                "step": int(step), "train_nll": nll,
                "train_ppl": math.exp(min(nll, 20.0)), "grad": grad,
                "lr": now_lr, "processed_tokens": int(processed),
                "compute_seconds": current_compute,
                "compute_tokens_per_second": processed / max(current_compute, 1e-9),
                "peak_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            }
            history.append(row)
            log(
                f"[{name}] step={step:05d}/{args.steps} nll={nll:.5f} "
                f"ppl={row['train_ppl']:.3f} grad={grad:.3f} lr={now_lr:.3e} "
                f"tok/s={row['compute_tokens_per_second']:,.0f} "
                f"peak={row['peak_gib']:.2f}G"
            )

        if step in milestones:
            sync(device)
            pause = time.perf_counter()
            val_row = evaluate_fixed_windows(
                name, model, val, args.train_seq, args.eval_windows,
                args.eval_seed, device, args.amp,
            )
            best_val = min(best_val, val_row["nll"])
            log(
                f"[{name}] VAL step={step:05d} nll={val_row['nll']:.5f} "
                f"ppl={val_row['ppl']:.3f} best={best_val:.5f}"
            )
            model.train()
            sync(device)
            eval_time = time.perf_counter() - pause
            excluded += eval_time

            current_compute = prior_compute + time.perf_counter() - compute_started - excluded
            pause = time.perf_counter()
            save_checkpoint(
                checkpoint_path, model, optimizer, step, history, best_val,
                processed, current_compute,
                prior_wall + time.perf_counter() - wall_started, signature,
            )
            sync(device)
            save_time = time.perf_counter() - pause
            excluded += save_time
            atomic_json(out / "history.json", history)
            log(f"[{name}] checkpoint saved step={step} io={save_time:.1f}s")

    sync(device)
    compute_seconds = prior_compute + time.perf_counter() - compute_started - excluded
    wall_seconds = prior_wall + time.perf_counter() - wall_started
    peak = torch.cuda.max_memory_allocated(device) / 2**30

    log(f"[{name}] final full validation")
    final_val = evaluate_stream(
        name, model, val_c, val, args.train_seq, args.validation_token_budget,
        device, args.amp,
    )
    log(f"[{name}] final full test")
    final_test = evaluate_stream(
        name, model, test_c, test, args.train_seq, args.test_token_budget,
        device, args.amp,
    )
    log(f"[{name}] matched-suffix contexts={args.matched_contexts}")
    matched = evaluate_matched_suffix(
        name, model, test, args.matched_contexts, args.matched_score_tokens,
        args.matched_windows, args.eval_seed + 70000, device, args.amp,
    )

    weights_dir = root / "final_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / f"{name}_step{args.steps}.pt"
    pause = time.perf_counter()
    torch.save({
        "version": 23, "model": name, "shape": asdict(shape),
        "step": args.steps, "state_dict": model.state_dict(),
        "args": vars(args), "final_test": final_test,
    }, weights_path)
    log(f"[{name}] final weights saved {weights_path} io={time.perf_counter()-pause:.1f}s")

    result = TrainResult(
        model=name, params=sum(p.numel() for p in model.parameters()),
        steps=args.steps, train_tokens=processed,
        compute_seconds=compute_seconds, wall_seconds=wall_seconds,
        compute_tokens_per_second=processed / max(compute_seconds, 1e-9),
        wall_tokens_per_second=processed / max(wall_seconds, 1e-9),
        peak_gib=peak, best_validation_nll=best_val,
        final_validation=final_val, final_test=final_test,
        matched_suffix=matched, history=history,
        final_weights=str(weights_path),
    )
    atomic_json(result_path, asdict(result))
    del model, optimizer
    clear_cuda()
    return result


@dataclass
class BenchRow:
    model: str
    policy: str
    kind: str
    context: int
    batch: int
    tokens_per_call: int
    status: str
    tokens_per_second: Optional[float]
    bytes_per_second_est: Optional[float]
    latency_ms: Optional[float]
    baseline_gib: Optional[float]
    peak_gib: Optional[float]
    activation_gib: Optional[float]
    error: str = ""


def benchmark_train(name: str, shape: Shape, args, deps,
                    data: torch.Tensor, context: int, batch: int,
                    bytes_per_token: float, device: torch.device) -> BenchRow:
    model = build_model(name, shape, args, deps, device).train()
    set_distill(model, 1.0)
    optimizer = make_optimizer(model, args)
    x, y = batch_for_step(
        data, batch, context, args.system_seed + context * 17,
        1, 0, device,
    )

    def step_once() -> None:
        optimizer.zero_grad(set_to_none=True)
        with amp_ctx(device, args.amp):
            loss, _ = loss_call(name, model, x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

    status, error = "ok", ""
    tps = bps = latency = baseline = peak = activation = None
    try:
        for _ in range(args.system_warmup):
            step_once()
        sync(device)
        baseline = torch.cuda.memory_allocated(device) / 2**30
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(args.system_steps):
            step_once()
        sync(device)
        elapsed = time.perf_counter() - started
        tokens = args.system_steps * batch * context
        tps = tokens / max(elapsed, 1e-9)
        bps = tps * bytes_per_token
        latency = elapsed * 1000.0 / args.system_steps
        peak = torch.cuda.max_memory_allocated(device) / 2**30
        activation = max(0.0, peak - baseline)
    except torch.cuda.OutOfMemoryError as exc:
        status, error = "oom", str(exc)
    except Exception as exc:
        status, error = "error", repr(exc)
    row = BenchRow(
        model=name, policy=args.fusion_checkpoint_policy if name == FUSION else "none",
        kind="train", context=context, batch=batch,
        tokens_per_call=batch * context, status=status,
        tokens_per_second=tps, bytes_per_second_est=bps,
        latency_ms=latency, baseline_gib=baseline, peak_gib=peak,
        activation_gib=activation, error=error,
    )
    del model, optimizer, x, y
    clear_cuda()
    return row


def benchmark_infer(name: str, shape: Shape, args, deps,
                    data: torch.Tensor, context: int, batch: int,
                    bytes_per_token: float, device: torch.device) -> BenchRow:
    # Recompute policy automatically disables itself in eval/no-grad.
    model = build_model(name, shape, args, deps, device).eval()
    set_distill(model, 1.0)
    x, y = batch_for_step(
        data, batch, context, args.system_seed + context * 19,
        1, 0, device,
    )

    def call_once() -> None:
        with torch.inference_mode(), amp_ctx(device, args.amp):
            loss, _ = loss_call(name, model, x, y)
            _ = loss + 0.0

    status, error = "ok", ""
    tps = bps = latency = baseline = peak = activation = None
    try:
        for _ in range(args.infer_warmup):
            call_once()
        sync(device)
        baseline = torch.cuda.memory_allocated(device) / 2**30
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(args.infer_steps):
            call_once()
        sync(device)
        elapsed = time.perf_counter() - started
        tokens = args.infer_steps * batch * context
        tps = tokens / max(elapsed, 1e-9)
        bps = tps * bytes_per_token
        latency = elapsed * 1000.0 / args.infer_steps
        peak = torch.cuda.max_memory_allocated(device) / 2**30
        activation = max(0.0, peak - baseline)
    except torch.cuda.OutOfMemoryError as exc:
        status, error = "oom", str(exc)
    except Exception as exc:
        status, error = "error", repr(exc)
    row = BenchRow(
        model=name, policy="none", kind="infer", context=context,
        batch=batch, tokens_per_call=batch * context, status=status,
        tokens_per_second=tps, bytes_per_second_est=bps,
        latency_ms=latency, baseline_gib=baseline, peak_gib=peak,
        activation_gib=activation, error=error,
    )
    del model, x, y
    clear_cuda()
    return row


def initialization_audit(args, shapes, deps, device: torch.device, root: Path) -> Dict[str, object]:
    rows: Dict[str, object] = {}
    embedding_hashes = set()
    for name in MODELS:
        model = build_model(name, shapes[name], args, deps, device)
        emb = tensor_hash(model.emb.weight)
        head_tied = model.lm_head.weight.data_ptr() == model.emb.weight.data_ptr()
        rows[name] = {"embedding_hash": emb, "head_tied": bool(head_tied)}
        embedding_hashes.add(emb)
        del model
        clear_cuda()
    if len(embedding_hashes) != 1:
        raise AssertionError(f"paired embedding mismatch: {rows}")
    if not all(bool(x["head_tied"]) for x in rows.values()):
        raise AssertionError(f"untied embedding detected: {rows}")
    out = {"models": rows, "shared_embedding_hash": next(iter(embedding_hashes))}
    atomic_json(root / "initialization_audit.json", out)
    return out


def make_summary(args, canonical_path: Path, tokenizer_path: Path,
                 shapes: Mapping[str, Shape], corpora: Mapping[str, Corpus],
                 results: Mapping[str, TrainResult], systems: Sequence[BenchRow],
                 init_audit: Mapping[str, object]) -> str:
    width = 210
    f = results[FUSION]
    t = results[TRANSFORMER]
    lines = [
        "=" * width,
        "FIELD-FUSION CANONICAL 300M v23 — WIKITEXT-103 100% / FIXED TOKEN BUDGET",
        "=" * width,
        f"canonical_source={canonical_path} sha256={sha256(canonical_path)}",
        f"tokenizer={tokenizer_path} sha256={sha256(tokenizer_path)} vocab={args.vocab_size:,}",
        f"protocol: WikiText-103 {args.data_frac*100:.1f}% | ctx={args.train_seq} | batch={args.batch_size} | accum={args.accum} | steps={args.steps:,} | tokens/update={args.batch_size*args.accum*args.train_seq:,} | total/model={args.steps*args.batch_size*args.accum*args.train_seq:,} | BF16",
        f"Fusion: fast PCAF | checkpoint_policy={args.fusion_checkpoint_policy} | [Field x5 -> latent-GQA refresh] x4 | windows={args.refresh_windows}",
        "Compute tok/s excludes evaluation and checkpoint I/O. Wall tok/s includes them.",
        "",
        "TOKENIZED CORPORA",
    ]
    for split in ("train", "validation", "test"):
        c = corpora[split]
        lines.append(
            f"{split:12s} tokens={int(c.tokens.numel()):12,d} raw_bytes={c.raw_bytes:12,d} bytes/token={c.bytes_per_token:.4f}"
        )
    lines.extend(["", "MODEL SHAPES", f"{'model':32s} {'params':>15s} {'dTarget%':>10s} {'dim':>6s} {'layers':>7s} {'heads':>7s} {'ff':>7s}"])
    for name in MODELS:
        s = shapes[name]
        delta = 100.0 * (s.params - args.target_params) / args.target_params
        lines.append(f"{name:32s} {s.params:15,d} {delta:+10.3f} {s.dim:6d} {s.layers:7d} {s.heads:7d} {s.ff_hidden:7d}")

    lines.extend(["", "FINAL QUALITY", f"{'model':32s} {'PPL':>11s} {'NLL':>10s} {'BPB norm':>11s} {'compute tok/s':>15s} {'wall tok/s':>13s} {'peak GB':>10s}"])
    for name in MODELS:
        r = results[name]
        lines.append(
            f"{name:32s} {r.final_test['ppl']:11.4f} {r.final_test['nll']:10.5f} {r.final_test['bpb_norm']:11.5f} {r.compute_tokens_per_second:15,.0f} {r.wall_tokens_per_second:13,.0f} {r.peak_gib:10.2f}"
        )
    dnll = f.final_test["nll"] - t.final_test["nll"]
    dppl = 100.0 * (f.final_test["ppl"] / t.final_test["ppl"] - 1.0)
    lines.extend(["", "QUALITY DELTA", f"Fusion vs Transformer: dNLL={dnll:+.5f} dPPL={dppl:+.3f}%"])

    lines.extend(["", "MATCHED-SUFFIX GENERALIZATION — SAME TARGET TOKENS"])
    contexts = [int(x) for x in args.matched_contexts]
    lines.append(f"{'model':32s}" + "".join(f" {'PPL@'+str(c):>12s}" for c in contexts))
    for name in MODELS:
        idx = {int(x["context"]): x for x in results[name].matched_suffix}
        lines.append(f"{name:32s}" + "".join(f" {idx[c]['ppl']:12.4f}" for c in contexts))

    lines.extend(["", "SYSTEM SWEEP — COMPLETE LOSS PATH", f"{'model':32s} {'kind':>7s} {'ctx':>7s} {'batch':>6s} {'tok/s':>13s} {'MB/s est':>12s} {'peak GB':>10s} {'act~GB':>10s} {'status':>8s}"])
    for row in systems:
        ts = "-" if row.tokens_per_second is None else f"{row.tokens_per_second:,.0f}"
        mb = "-" if row.bytes_per_second_est is None else f"{row.bytes_per_second_est/1e6:.2f}"
        pg = "-" if row.peak_gib is None else f"{row.peak_gib:.2f}"
        ag = "-" if row.activation_gib is None else f"{row.activation_gib:.2f}"
        lines.append(f"{row.model:32s} {row.kind:>7s} {row.context:7d} {row.batch:6d} {ts:>13s} {mb:>12s} {pg:>10s} {ag:>10s} {row.status:>8s}")

    def find(name: str, kind: str, context: int) -> Optional[BenchRow]:
        return next((x for x in systems if x.model == name and x.kind == kind and x.context == context and x.status == "ok"), None)

    lines.extend(["", "FUSION / TRANSFORMER RATIOS"])
    long_passes = []
    for context in args.system_contexts:
        fr = find(FUSION, "train", int(context)); tr = find(TRANSFORMER, "train", int(context))
        if fr and tr and fr.tokens_per_second and tr.tokens_per_second and fr.peak_gib and tr.peak_gib:
            sr = fr.tokens_per_second / tr.tokens_per_second
            mr = fr.peak_gib / tr.peak_gib
            lines.append(f"train ctx={context:5d}: speed={sr:.3f}x peak={mr:.3f}x")
            if int(context) >= args.long_context_gate:
                long_passes.append(sr >= 1.0 and mr <= 1.0)
        fi = find(FUSION, "infer", int(context)); ti = find(TRANSFORMER, "infer", int(context))
        if fi and ti and fi.tokens_per_second and ti.tokens_per_second and fi.peak_gib and ti.peak_gib:
            lines.append(f"infer ctx={context:5d}: speed={fi.tokens_per_second/ti.tokens_per_second:.3f}x peak={fi.peak_gib/ti.peak_gib:.3f}x")

    quality_win = dnll < 0.0
    system_win = bool(long_passes) and all(long_passes)
    lines.extend(["", "AUTOMATIC VERDICT"])
    if quality_win and system_win:
        lines.append("CANONICAL PASS: Field-Fusion beat the Transformer in final test quality and in long-context training speed+memory.")
    elif quality_win:
        lines.append("QUALITY PASS / SYSTEM PARTIAL: Field-Fusion won final quality, but not every long-context speed+memory gate.")
    else:
        lines.append("QUALITY MISS: the Transformer recovered the final quality lead at this fixed-token budget.")
    lines.append(f"initialization shared_embedding_hash={init_audit['shared_embedding_hash']}")
    lines.append(f"final_weights_fusion={f.final_weights}")
    lines.append(f"final_weights_transformer={t.final_weights}")
    lines.append("=" * width)
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--outdir", required=True)
    p.add_argument("--canonical-source", default="")
    p.add_argument("--cache-dir", default="/home/ubuntu/field_lab/hf_cache")
    p.add_argument("--tokenizer-source", default="")
    p.add_argument("--data-frac", type=float, default=1.0)
    p.add_argument("--data-device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--tokenizer-min-frequency", type=int, default=2)

    p.add_argument("--target-params", type=int, default=300_000_000)
    p.add_argument("--max-param-delta-pct", type=float, default=0.75)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--layers", type=int, default=24)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--min-ff-hidden", type=int, default=1024)
    p.add_argument("--max-ff-hidden", type=int, default=4096)
    p.add_argument("--ff-multiple", type=int, default=64)

    p.add_argument("--field-chunk", type=int, default=32)
    p.add_argument("--triton-block-c", type=int, default=16)
    p.add_argument("--triton-chunk-t", type=int, default=64)
    p.add_argument("--num-buckets", type=int, default=16384)
    p.add_argument("--salience-floor", type=float, default=0.10)
    p.add_argument("--residual-limit", type=float, default=4.0)
    p.add_argument("--address-dim", type=int, default=64)
    p.add_argument("--latent-top-k", type=int, default=4)
    p.add_argument("--score-limit", type=float, default=8.0)
    p.add_argument("--sidecar-aux-weight", type=float, default=0.0)
    p.add_argument("--sidecar-max-mix", type=float, default=0.25)
    p.add_argument("--sidecar-gate-bias", type=float, default=-2.0)
    p.add_argument("--gate-grad-scale", type=float, default=1.0)
    p.add_argument("--delta-heads", type=int, default=4)
    p.add_argument("--delta-block", type=int, default=64)
    p.add_argument("--span-top-k", type=int, default=4)
    p.add_argument("--phase-bands", type=int, default=4)
    p.add_argument("--phase-rank", type=int, default=32)

    p.add_argument("--fusion-q-heads", type=int, default=16)
    p.add_argument("--fusion-kv-heads", type=int, default=4)
    p.add_argument("--fusion-latent-dim", type=int, default=256)
    p.add_argument("--refresh-windows", nargs="+", type=int, default=[256, 512, 1024, 2048])
    p.add_argument("--landmark-chunk", type=int, default=256)
    p.add_argument("--sparse-surprise-mix", type=float, default=0.25)
    p.add_argument("--fusion-checkpoint-policy", choices=v22.POLICIES, default="field_half")

    p.add_argument("--model-seed", type=int, default=1234)
    p.add_argument("--embedding-seed", type=int, default=314159)
    p.add_argument("--data-seed", type=int, default=5678)
    p.add_argument("--eval-seed", type=int, default=9012)
    p.add_argument("--system-seed", type=int, default=44021)
    p.add_argument("--amp", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.10)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--min-lr-ratio", type=float, default=0.10)
    p.add_argument("--warmup", type=int, default=600)
    p.add_argument("--distill-ramp", type=int, default=600)

    p.add_argument("--train-seq", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--milestones", nargs="+", type=int, default=[1500, 3000, 6000, 9000, 12000])
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-windows", type=int, default=16)
    p.add_argument("--validation-token-budget", type=int, default=0)
    p.add_argument("--test-token-budget", type=int, default=0)
    p.add_argument("--matched-contexts", nargs="+", type=int, default=[256, 512, 1024, 2048, 4096, 8192, 16384])
    p.add_argument("--matched-score-tokens", type=int, default=128)
    p.add_argument("--matched-windows", type=int, default=8)

    p.add_argument("--system-contexts", nargs="+", type=int, default=[1024, 2048, 4096, 8192, 16384])
    p.add_argument("--system-tokens-per-call", type=int, default=8192)
    p.add_argument("--system-warmup", type=int, default=2)
    p.add_argument("--system-steps", type=int, default=5)
    p.add_argument("--infer-warmup", type=int, default=3)
    p.add_argument("--infer-steps", type=int, default=10)
    p.add_argument("--long-context-gate", type=int, default=8192)
    p.add_argument("--selftest-tokens", type=int, default=65)
    p.add_argument("--exact-loss-tol", type=float, default=5e-4)
    p.add_argument("--exact-grad-rel-tol", type=float, default=0.03)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA H100 required")
    if args.layers != 6 * len(args.refresh_windows):
        raise ValueError("layers must equal 6 * len(refresh_windows)")
    tokens_per_update = args.batch_size * args.accum * args.train_seq
    if tokens_per_update != 8192:
        log(f"WARNING: tokens/update={tokens_per_update:,}, promoted protocol used 8,192")
    if args.data_frac != 1.0:
        log(f"WARNING: canonical run requested with data_frac={args.data_frac}")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.set_float32_matmul_precision("high")

    arena = base.import_module(base.V15_PATH, "field_scale_50m_v15_for_v23")
    canonical_path = base.locate_canonical(args.canonical_source)
    actual_sha = sha256(canonical_path)
    if actual_sha != EXPECTED_CANONICAL_SHA256:
        raise RuntimeError(
            f"canonical SHA mismatch expected={EXPECTED_CANONICAL_SHA256} actual={actual_sha}"
        )
    v3 = arena.base.import_module(arena.base.V3_PATH, "v23_v3")
    bridge = arena.base.import_module(arena.base.BRIDGE_PATH, "v23_bridge")
    optmod = arena.base.import_module(arena.base.OPT_PATH, "v23_opt")
    epi = arena.base.import_module(arena.base.V9_PATH, "v23_epi")
    judge = arena.base.import_module(arena.base.JUDGE_PATH, "v23_judge")
    canonical = arena.base.import_module(canonical_path, "v23_canonical")
    optmod.v3_global = v3
    arena.base.install_fast_candidate_route(epi, optmod)
    changed = core.patch_vocab(args.vocab_size, HERE, canonical_path)
    log(f"[vocab] patched VOCAB={args.vocab_size:,} in {len(changed)} modules")

    # Compatibility fields consumed by inherited builders.
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
    raw_rows = core.load_raw_rows(args.cache_dir, args.data_frac)
    tokenizer = base.copy_or_train_tokenizer(
        root, raw_rows[0], args.vocab_size,
        args.tokenizer_min_frequency, args.tokenizer_source,
    )
    train_c, val_c, test_c = core.save_or_load_corpora(root, tokenizer, raw_rows)
    train = core.place_tokens(train_c.tokens, device, args.data_device, "train")
    val = core.place_tokens(val_c.tokens, device, args.data_device, "validation")
    test = core.place_tokens(test_c.tokens, device, args.data_device, "test")
    corpora = {"train": train_c, "validation": val_c, "test": test_c}

    deps = (arena, v3, canonical, bridge, optmod, epi, judge)
    shapes = solve_shapes(args, deps)
    for name, shape in shapes.items():
        delta = 100.0 * (shape.params - args.target_params) / args.target_params
        log(f"[shape] {name:32s} params={shape.params:,} dTarget={delta:+.3f}% ff={shape.ff_hidden}")

    init_audit = initialization_audit(args, shapes, deps, device, root)
    # Reuse the v22 exactness test for field_half checkpointing.
    checkpoint_audit = v22.checkpoint_exactness_audit(args, shapes[FUSION], deps, device, root)
    log(f"[selftest] checkpoint exactness={checkpoint_audit}")

    results: Dict[str, TrainResult] = {}
    for name in MODELS:
        log("=" * 180)
        log(f"CANONICAL ARM: {name}")
        results[name] = train_arm(
            name, shapes[name], args, deps, train, val_c, val,
            test_c, test, root, device,
        )
        atomic_json(root / "train_results.json", {k: asdict(v) for k, v in results.items()})

    systems: List[BenchRow] = []
    for context in args.system_contexts:
        batch = max(1, args.system_tokens_per_call // int(context))
        for name in MODELS:
            log(f"[system/train] {name} ctx={context} batch={batch}")
            row = benchmark_train(
                name, shapes[name], args, deps, train, int(context), batch,
                train_c.bytes_per_token, device,
            )
            systems.append(row)
            log(asdict(row))
            atomic_json(root / "system_rows.json", [asdict(x) for x in systems])
        for name in MODELS:
            log(f"[system/infer] {name} ctx={context} batch={batch}")
            row = benchmark_infer(
                name, shapes[name], args, deps, test, int(context), batch,
                test_c.bytes_per_token, device,
            )
            systems.append(row)
            log(asdict(row))
            atomic_json(root / "system_rows.json", [asdict(x) for x in systems])

    result = {
        "args": vars(args), "canonical_source": str(canonical_path),
        "canonical_sha256": actual_sha,
        "tokenizer": str(root / "tokenizer" / "tokenizer.json"),
        "tokenizer_sha256": sha256(root / "tokenizer" / "tokenizer.json"),
        "shapes": {k: asdict(v) for k, v in shapes.items()},
        "initialization_audit": init_audit,
        "checkpoint_audit": checkpoint_audit,
        "train_results": {k: asdict(v) for k, v in results.items()},
        "system_rows": [asdict(x) for x in systems],
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda": torch.version.cuda,
    }
    atomic_json(root / "results.json", result)
    summary = make_summary(
        args, canonical_path, root / "tokenizer" / "tokenizer.json",
        shapes, corpora, results, systems, init_audit,
    )
    atomic_text(root / "summary.txt", summary)
    log(summary)


if __name__ == "__main__":
    main()
