#!/usr/bin/env python3
"""FIELD-FUSION QUALITY + MEMORY GATE v22.

A short, paired 300M gate before the WikiText-103 100% canonical run.

Stage A — quality guard on WikiText-103 5%
  * fusion_dense: v21 packed refresh + the v20 dense episodic PCAF wrapper.
  * fusion_fast:  v21 packed refresh + validated target-only Fast PCAF.
  * fusion_sparse: v21 packed refresh + scalable surprise-weighted sparse PCAF.
  * transformer: same strong Flash-SDPA 300M control.

All arms start from paired initial weights and consume identical BPE windows.
The training protocol fixes the v20 under-utilization: batch=8, accum=1 at
ctx=1024 (8192 tokens/update), no per-step GPU->CPU loss synchronization, and
checkpoint I/O excluded from compute throughput.

Stage B — exact recompute memory sweep
  The best quality-preserving PCAF candidate is benchmarked with:
    none       : normal autograd
    field_half : checkpoint alternate Field blocks
    field_all  : checkpoint all 20 Field blocks
    all        : checkpoint all 24 blocks
  Checkpointing is an exact training-memory mode: forward math and parameters
  do not change. It is disabled automatically in evaluation/inference.

Promotion requires:
  * candidate test NLL within --quality-gap of dense;
  * candidate still beats the Transformer test NLL;
  * at least one recompute policy uses <= Transformer peak VRAM at the long
    system context while retaining >= --min-long-speed-ratio throughput.

This script does not launch WikiText-103 100% automatically. It emits the exact
recommended cache and recompute policy for the canonical run.
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
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import field_fusion_system_smoke_v21 as v21

v20 = v21.v20
base = v21.base
core = v21.core
Shape = v21.Shape
Corpus = core.Corpus
HERE = Path(__file__).resolve().parent
EXPECTED_CANONICAL_SHA256 = v21.EXPECTED_CANONICAL_SHA256

DENSE = "fusion_dense"
FAST = "fusion_fast"
SPARSE = "fusion_sparse"
TRANSFORMER = "transformer_flash_300m"
QUALITY_MODELS = (DENSE, FAST, SPARSE, TRANSFORMER)
CACHE_CANDIDATES = (FAST, SPARSE)
POLICIES = ("none", "field_half", "field_all", "all")
NAME_TO_V21 = {
    DENSE: v21.V21_EXACT_DENSE,
    FAST: v21.V21_FAST,
    SPARSE: v21.V21_SPARSE,
    TRANSFORMER: v21.TRANSFORMER,
}


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


def module_hash(model: nn.Module, *, exclude_prefixes: Sequence[str] = ()) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if any(key.startswith(p) for p in exclude_prefixes):
            continue
        h.update(key.encode())
        x = value.detach().contiguous().cpu()
        h.update(str(tuple(x.shape)).encode())
        h.update(str(x.dtype).encode())
        h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def amp_ctx(device: torch.device, amp: str):
    return v21.amp_ctx(device, amp)


def set_distill(model: nn.Module, value: float) -> None:
    v21.set_distill(model, value)


def mapped(name: str) -> str:
    return NAME_TO_V21[name]


def build_model(name: str, shape: Shape, args, deps, device: torch.device,
                checkpoint_policy: str = "none") -> nn.Module:
    arena, v3, canonical, bridge, optmod, epi, judge = deps
    model = v21.build_model(
        mapped(name), shape, args, arena, v3, canonical, bridge, optmod, epi,
        judge, device,
    )
    if name != TRANSFORMER and checkpoint_policy != "none":
        install_checkpoint_policy(model, checkpoint_policy)
    return model


def solve_shapes(args, deps) -> Dict[str, Shape]:
    arena, v3, canonical, bridge, optmod, epi, judge = deps
    shapes: Dict[str, Shape] = {}
    for name in QUALITY_MODELS:
        raw = v21.solve_shape(
            mapped(name), args, arena, v3, canonical, bridge, optmod, epi, judge
        )
        shapes[name] = Shape(name, raw.params, raw.dim, raw.layers, raw.heads, raw.ff_hidden)
        delta = 100.0 * (raw.params - args.target_params) / args.target_params
        if abs(delta) > args.max_param_delta_pct:
            raise RuntimeError(f"parameter mismatch {name}: {delta:+.3f}%")
    return shapes


def install_checkpoint_policy(model: nn.Module, policy: str) -> None:
    if policy not in POLICIES or policy == "none":
        if policy != "none":
            raise ValueError(policy)
        return
    model._v22_checkpoint_policy = policy

    def states_logits(self, tokens: torch.Tensor):
        x = self.emb(tokens)
        self._patch_aux = x.new_zeros(())
        field_index = 0
        for i, block in enumerate(self.blocks):
            is_refresh = isinstance(block, v21.FusionRefreshBlockV21)
            use = False
            if self.training and torch.is_grad_enabled():
                if policy == "all":
                    use = True
                elif policy == "field_all":
                    use = not is_refresh
                elif policy == "field_half":
                    use = (not is_refresh) and (field_index % 2 == 0)
            if use:
                x = checkpoint(
                    block, x, use_reentrant=False, preserve_rng_state=False
                )
            else:
                x = block(x)
            if not is_refresh:
                field_index += 1
            if i == self.patch_position and self.softpatch is not None:
                x = self.softpatch(x, tokens)
                self._patch_aux = self.softpatch.last_aux
        return x, self.lm_head(self.final_norm(x))

    model.states_logits = types.MethodType(states_logits, model)


def make_optimizer(model: nn.Module, args):
    return core.make_optimizer(model, args.lr, args.weight_decay)


def lr_at(step: int, total: int, warmup: int, peak: float, min_ratio: float) -> float:
    return core.lr_at(step, total, warmup, peak, min_ratio)


def loss_call(name: str, model: nn.Module, x: torch.Tensor, y: torch.Tensor):
    return v21.loss_call(mapped(name), model, x, y, mode="full")


def batch_for_step(data: torch.Tensor, batch: int, seq: int, seed: int,
                   step: int, micro: int, device: torch.device):
    return core.batch_for_step(data, batch, seq, seed, step, micro, device)


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
            _, primary = loss_call(name, model, x, y)
        total_nll += float(primary.detach().cpu()) * int(y.numel())
        total_tokens += int(y.numel())
    mean = total_nll / max(total_tokens, 1)
    bytes_est = total_tokens * corpus.bytes_per_token
    return {
        "context": int(context),
        "nll": mean,
        "ppl": math.exp(min(mean, 20.0)),
        "bits_per_token": mean / math.log(2.0),
        "bpb_norm": (total_nll / math.log(2.0)) / max(bytes_est, 1e-9),
        "tokens": total_tokens,
        "bytes_est": bytes_est,
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
            _, primary = loss_call(name, model, x, y)
        total += float(primary.detach().cpu()) * int(y.numel())
        count += int(y.numel())
    nll = total / max(count, 1)
    return {"nll": nll, "ppl": math.exp(min(nll, 20.0)), "tokens": count}


@dataclass
class QualityResult:
    model: str
    params: int
    steps: int
    train_tokens_per_second: float
    train_peak_gib: float
    best_validation_nll: float
    test: Dict[str, float]
    history: List[Dict[str, float]]


def train_quality_arm(name: str, shape: Shape, args, deps,
                      train: torch.Tensor, val: torch.Tensor,
                      test_c: Corpus, test: torch.Tensor,
                      root: Path, device: torch.device) -> QualityResult:
    out = root / "quality" / name
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "result.json"
    if result_path.is_file() and args.resume:
        raw = json.loads(result_path.read_text(encoding="utf-8"))
        return QualityResult(**raw)

    model = build_model(name, shape, args, deps, device).train()
    opt = make_optimizer(model, args)
    history: List[Dict[str, float]] = []
    processed = 0
    excluded = 0.0
    primary_acc = torch.zeros((), device=device, dtype=torch.float32)
    grad_tensor = torch.zeros((), device=device, dtype=torch.float32)
    best_val = float("inf")

    clear_cuda()
    torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    started = time.perf_counter()

    for step in range(1, args.quality_steps + 1):
        set_distill(model, min(1.0, step / max(args.distill_ramp, 1)))
        opt.zero_grad(set_to_none=True)
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
        now_lr = lr_at(
            step, args.quality_steps, args.warmup, args.lr, args.min_lr_ratio
        )
        for group in opt.param_groups:
            group["lr"] = now_lr
        opt.step()
        processed += args.batch_size * args.accum * args.train_seq

        do_log = step % args.log_every == 0 or step == args.quality_steps
        if do_log:
            sync(device)
            elapsed = time.perf_counter() - started - excluded
            nll = float(primary_acc.detach().cpu())
            grad = float(grad_tensor.detach().cpu())
            row = {
                "step": float(step), "nll": nll,
                "ppl": math.exp(min(nll, 20.0)), "grad": grad,
                "lr": now_lr, "tokens_per_second": processed / max(elapsed, 1e-9),
                "peak_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            }
            history.append(row)
            log(
                f"[{name}] step={step:04d}/{args.quality_steps} "
                f"nll={nll:.5f} ppl={row['ppl']:.3f} grad={grad:.3f} "
                f"tok/s={row['tokens_per_second']:,.0f} peak={row['peak_gib']:.2f}G"
            )

        if step % args.eval_every == 0 or step == args.quality_steps:
            t0 = time.perf_counter()
            row = evaluate_fixed_windows(
                name, model, val, args.train_seq, args.eval_windows,
                args.eval_seed, device, args.amp,
            )
            best_val = min(best_val, row["nll"])
            log(
                f"[{name}] VAL step={step:04d} nll={row['nll']:.5f} "
                f"ppl={row['ppl']:.3f} best_nll={best_val:.5f}"
            )
            model.train()
            excluded += time.perf_counter() - t0

    sync(device)
    elapsed = time.perf_counter() - started - excluded
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    test_row = evaluate_stream(
        name, model, test_c, test, args.train_seq, args.test_token_budget,
        device, args.amp,
    )
    result = QualityResult(
        model=name, params=sum(p.numel() for p in model.parameters()),
        steps=args.quality_steps,
        train_tokens_per_second=processed / max(elapsed, 1e-9),
        train_peak_gib=peak, best_validation_nll=best_val,
        test=test_row, history=history,
    )
    atomic_json(result_path, asdict(result))
    del model, opt
    clear_cuda()
    return result


@dataclass
class BenchRow:
    model: str
    cache: str
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


def benchmark_train(name: str, policy: str, shape: Shape, args, deps,
                    data: torch.Tensor, context: int, batch: int,
                    bytes_per_token: float, device: torch.device) -> BenchRow:
    model = build_model(name, shape, args, deps, device, policy).train()
    set_distill(model, 1.0)
    opt = make_optimizer(model, args)
    x, y = batch_for_step(
        data, batch, context, args.system_seed + context * 17, 1, 0, device
    )

    def step_once():
        opt.zero_grad(set_to_none=True)
        with amp_ctx(device, args.amp):
            loss, _ = loss_call(name, model, x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

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
        model=f"{name}:{policy}", cache=name, policy=policy,
        kind="train", context=context, batch=batch,
        tokens_per_call=batch * context, status=status,
        tokens_per_second=tps, bytes_per_second_est=bps,
        latency_ms=latency, baseline_gib=baseline, peak_gib=peak,
        activation_gib=activation, error=error,
    )
    del model, opt, x, y
    clear_cuda()
    return row


def benchmark_infer(name: str, shape: Shape, args, deps,
                    data: torch.Tensor, context: int, batch: int,
                    bytes_per_token: float, device: torch.device) -> BenchRow:
    model = build_model(name, shape, args, deps, device).eval()
    set_distill(model, 1.0)
    x, y = batch_for_step(
        data, batch, context, args.system_seed + context * 19, 1, 0, device
    )

    def call_once():
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
        model=name, cache=name, policy="none", kind="infer",
        context=context, batch=batch, tokens_per_call=batch * context,
        status=status, tokens_per_second=tps, bytes_per_second_est=bps,
        latency_ms=latency, baseline_gib=baseline, peak_gib=peak,
        activation_gib=activation, error=error,
    )
    del model, x, y
    clear_cuda()
    return row


def paired_initialization_audit(args, shapes, deps, device: torch.device,
                                root: Path) -> Dict[str, object]:
    rows: Dict[str, object] = {}
    backbone_hashes = set()
    embedding_hashes = set()
    for name in (DENSE, FAST, SPARSE):
        model = build_model(name, shapes[name], args, deps, device).eval()
        emb = tensor_hash(model.emb.weight)
        backbone = module_hash(model, exclude_prefixes=("cache.",))
        embedding_hashes.add(emb)
        backbone_hashes.add(backbone)
        rows[name] = {"embedding_hash": emb, "backbone_hash": backbone}
        del model
        clear_cuda()
    if len(embedding_hashes) != 1:
        raise AssertionError("paired embedding initialization mismatch")
    if len(backbone_hashes) != 1:
        raise AssertionError("paired Fusion backbone initialization mismatch")
    atomic_json(root / "paired_initialization.json", rows)
    return rows


def checkpoint_exactness_audit(args, shape: Shape, deps,
                               device: torch.device, root: Path) -> Dict[str, float]:
    base_model = build_model(FAST, shape, args, deps, device, "none").train()
    test_model = build_model(FAST, shape, args, deps, device, "field_all").train()
    test_model.load_state_dict(base_model.state_dict(), strict=True)
    x = torch.randint(0, args.vocab_size, (1, args.selftest_tokens), device=device)
    y = torch.randint(0, args.vocab_size, (1, args.selftest_tokens), device=device)
    set_distill(base_model, 1.0)
    set_distill(test_model, 1.0)
    with amp_ctx(device, args.amp):
        lb, _ = loss_call(FAST, base_model, x, y)
        lt, _ = loss_call(FAST, test_model, x, y)
    lb.backward()
    lt.backward()
    loss_abs = float((lb.detach() - lt.detach()).abs().cpu())
    grad_abs = 0.0
    grad_rel = 0.0
    checked = 0
    for (nb, pb), (nt, pt) in zip(base_model.named_parameters(), test_model.named_parameters()):
        if nb != nt or pb.grad is None or pt.grad is None:
            continue
        da = (pb.grad.float() - pt.grad.float()).abs()
        grad_abs = max(grad_abs, float(da.max().cpu()))
        denom = pb.grad.float().abs().mean().clamp_min(1e-8)
        grad_rel = max(grad_rel, float((da.mean() / denom).cpu()))
        checked += 1
    row = {
        "loss_abs": loss_abs, "grad_max_abs": grad_abs,
        "grad_max_mean_relative": grad_rel, "parameters_checked": checked,
    }
    if loss_abs > args.exact_loss_tol or grad_rel > args.exact_grad_rel_tol:
        raise AssertionError(f"checkpoint exactness failed: {row}")
    atomic_json(root / "checkpoint_exactness.json", row)
    del base_model, test_model, x, y, lb, lt
    clear_cuda()
    return row


def quality_decision(results: Mapping[str, QualityResult], args) -> Dict[str, object]:
    dense_nll = results[DENSE].test["nll"]
    tf_nll = results[TRANSFORMER].test["nll"]
    candidates = []
    for name in CACHE_CANDIDATES:
        nll = results[name].test["nll"]
        gap = nll - dense_nll
        passes = gap <= args.quality_gap and nll < tf_nll
        candidates.append({
            "name": name, "nll": nll, "gap_vs_dense": gap,
            "beats_transformer": nll < tf_nll, "quality_pass": passes,
            "train_tokens_per_second": results[name].train_tokens_per_second,
            "train_peak_gib": results[name].train_peak_gib,
        })
    passed = [x for x in candidates if x["quality_pass"]]
    # Prefer the fastest candidate when quality passes; use NLL as tie-breaker.
    winner = None
    if passed:
        winner = max(passed, key=lambda r: (r["train_tokens_per_second"], -r["nll"]))["name"]
    return {
        "dense_nll": dense_nll, "transformer_nll": tf_nll,
        "candidates": candidates, "quality_winner": winner,
    }


def system_decision(quality: Mapping[str, object], rows: Sequence[BenchRow],
                    args) -> Dict[str, object]:
    winner = quality.get("quality_winner")
    if not winner:
        return {"promote": False, "reason": "no quality-preserving cache candidate"}
    long_ctx = max(args.system_contexts)
    valid = [r for r in rows if r.kind == "train" and r.context == long_ctx and r.status == "ok"]
    tf = next((r for r in valid if r.cache == TRANSFORMER), None)
    options = [r for r in valid if r.cache == winner]
    ranked = []
    if tf is not None and tf.tokens_per_second and tf.peak_gib:
        for r in options:
            if r.tokens_per_second is None or r.peak_gib is None:
                continue
            speed = r.tokens_per_second / tf.tokens_per_second
            peak = r.peak_gib / tf.peak_gib
            passes = speed >= args.min_long_speed_ratio and peak <= args.max_long_peak_ratio
            ranked.append({
                "cache": winner, "policy": r.policy,
                "speed_ratio": speed, "peak_ratio": peak,
                "tokens_per_second": r.tokens_per_second,
                "peak_gib": r.peak_gib, "passes": passes,
            })
    passing = [x for x in ranked if x["passes"]]
    recommended = None
    if passing:
        # Highest speed among policies that actually beat the memory gate.
        recommended = max(passing, key=lambda x: x["speed_ratio"])
    return {
        "promote": recommended is not None,
        "long_context": long_ctx,
        "transformer": None if tf is None else asdict(tf),
        "options": ranked,
        "recommended": recommended,
        "reason": "quality and long-context system gates passed" if recommended else
                  "no recompute policy passed both long-context speed and memory gates",
    }


def fnum(x: Optional[float], fmt: str = ".3f") -> str:
    if x is None or not math.isfinite(float(x)):
        return "-"
    return format(float(x), fmt)


def make_summary(args, canonical_path: Path, tokenizer_path: Path,
                 shapes: Mapping[str, Shape], quality_results: Mapping[str, QualityResult],
                 quality: Mapping[str, object], rows: Sequence[BenchRow],
                 system: Mapping[str, object], init_audit, checkpoint_audit,
                 corpora: Mapping[str, Corpus]) -> str:
    width = 210
    lines = [
        "=" * width,
        "FIELD-FUSION QUALITY + MEMORY GATE v22 — 300M / WIKITEXT-103 5%",
        "=" * width,
        f"canonical_source={canonical_path} sha256={sha256(canonical_path)}",
        f"tokenizer={tokenizer_path} sha256={sha256(tokenizer_path)} vocab={args.vocab_size:,}",
        (
            f"quality protocol: ctx={args.train_seq} batch={args.batch_size} accum={args.accum} "
            f"steps={args.quality_steps} tokens/update={args.train_seq*args.batch_size*args.accum:,} BF16"
        ),
        "System memory modes are exact activation recomputation; they do not change forward mathematics or parameters.",
        "",
        "TOKENIZED CORPORA",
    ]
    for name in ("train", "validation", "test"):
        c = corpora[name]
        lines.append(
            f"{name:12s} tokens={c.tokens.numel():12,d} raw_bytes={c.raw_bytes:12,d} bytes/token={c.bytes_per_token:.4f}"
        )
    lines.extend([
        "", "SELFTESTS",
        f"paired backbone hash={next(iter(init_audit.values()))['backbone_hash']}",
        f"checkpoint loss_abs={checkpoint_audit['loss_abs']:.3e} grad_rel={checkpoint_audit['grad_max_mean_relative']:.3e}",
        "", "MODEL SHAPES",
        f"{'model':28s} {'params':>14s} {'dTarget%':>10s} {'ff':>7s}",
    ])
    for name in QUALITY_MODELS:
        s = shapes[name]
        delta = 100.0 * (s.params - args.target_params) / args.target_params
        lines.append(f"{name:28s} {s.params:14,d} {delta:+10.3f} {s.ff_hidden:7d}")
    lines.extend([
        "", "PAIRED QUALITY GUARD",
        f"{'model':28s} {'PPL':>11s} {'NLL':>10s} {'BPB norm':>10s} {'train tok/s':>14s} {'peak GB':>9s} {'dNLL dense':>11s}",
    ])
    dense_nll = quality_results[DENSE].test["nll"]
    for name in QUALITY_MODELS:
        r = quality_results[name]
        lines.append(
            f"{name:28s} {r.test['ppl']:11.4f} {r.test['nll']:10.5f} {r.test['bpb_norm']:10.5f} "
            f"{r.train_tokens_per_second:14,.0f} {r.train_peak_gib:9.2f} {r.test['nll']-dense_nll:+11.5f}"
        )
    lines.extend(["", "QUALITY DECISION"])
    for c in quality["candidates"]:
        lines.append(
            f"{c['name']:28s} gap_dense={c['gap_vs_dense']:+.5f} "
            f"beats_tf={c['beats_transformer']} pass={c['quality_pass']}"
        )
    lines.append(f"quality_winner={quality.get('quality_winner')}")
    lines.extend([
        "", "SYSTEM SWEEP — COMPLETE LOSS PATH",
        f"{'model':34s} {'kind':>7s} {'ctx':>7s} {'batch':>6s} {'tok/s':>13s} {'MB/s est':>11s} {'peak GB':>9s} {'act~GB':>9s} {'status':>8s}",
    ])
    for r in rows:
        lines.append(
            f"{r.model:34s} {r.kind:>7s} {r.context:7d} {r.batch:6d} "
            f"{fnum(r.tokens_per_second, ',.0f'):>13s} "
            f"{fnum(None if r.bytes_per_second_est is None else r.bytes_per_second_est/1e6, '.2f'):>11s} "
            f"{fnum(r.peak_gib, '.2f'):>9s} {fnum(r.activation_gib, '.2f'):>9s} {r.status:>8s}"
        )
    lines.extend(["", "LONG-CONTEXT SYSTEM DECISION"])
    for option in system.get("options", []):
        lines.append(
            f"cache={option['cache']} policy={option['policy']:<10s} "
            f"speed={option['speed_ratio']:.3f}x peak={option['peak_ratio']:.3f}x pass={option['passes']}"
        )
    lines.extend([
        f"recommended={system.get('recommended')}",
        f"PROMOTE={system.get('promote')} reason={system.get('reason')}",
        "",
        "AUTOMATIC VERDICT",
    ])
    if system.get("promote"):
        rec = system["recommended"]
        lines.extend([
            "PASS: quality-preserving cache and a memory-winning recompute policy were found.",
            f"Canonical WikiText-103 100% candidate: cache={rec['cache']} checkpoint_policy={rec['policy']}.",
            "Proceed to the 100% run with a fixed step budget; do not use epoch-count scaling.",
        ])
    elif quality.get("quality_winner"):
        lines.extend([
            "PARTIAL PASS: a low-cost cache preserved quality, but no memory policy passed the full long-context system gate.",
            "Do not launch the canonical 100% run yet; profile the remaining activation peak or relax only the speed gate, not quality.",
        ])
    else:
        lines.extend([
            "QUALITY FAIL: neither scalable PCAF candidate stayed within the dense quality tolerance.",
            "Keep the dense model as the scientific baseline and redesign the cache before scaling.",
        ])
    lines.append("=" * width)
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--outdir", required=True)
    p.add_argument("--canonical-source", default="")
    p.add_argument("--cache-dir", default="/home/ubuntu/field_lab/hf_cache")
    p.add_argument("--tokenizer-source", default="")
    p.add_argument("--data-frac", type=float, default=0.05)
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
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--distill-ramp", type=int, default=200)

    p.add_argument("--train-seq", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--quality-steps", type=int, default=600)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=150)
    p.add_argument("--eval-windows", type=int, default=8)
    p.add_argument("--test-token-budget", type=int, default=131072)
    p.add_argument("--quality-gap", type=float, default=0.010)

    p.add_argument("--system-contexts", nargs="+", type=int, default=[1024, 4096, 8192, 16384])
    p.add_argument("--system-tokens-per-call", type=int, default=8192)
    p.add_argument("--system-warmup", type=int, default=2)
    p.add_argument("--system-steps", type=int, default=5)
    p.add_argument("--infer-warmup", type=int, default=3)
    p.add_argument("--infer-steps", type=int, default=10)
    p.add_argument("--min-long-speed-ratio", type=float, default=1.00)
    p.add_argument("--max-long-peak-ratio", type=float, default=1.00)

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
    if args.batch_size * args.accum * args.train_seq != 8192:
        log("WARNING: quality tokens/update is not 8192")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.set_float32_matmul_precision("high")

    arena = base.import_module(base.V15_PATH, "field_scale_50m_v15_for_v22")
    canonical_path = base.locate_canonical(args.canonical_source)
    actual_sha = sha256(canonical_path)
    if actual_sha != EXPECTED_CANONICAL_SHA256:
        raise RuntimeError(
            f"canonical SHA mismatch expected={EXPECTED_CANONICAL_SHA256} actual={actual_sha}"
        )
    v3 = arena.base.import_module(arena.base.V3_PATH, "v22_v3")
    bridge = arena.base.import_module(arena.base.BRIDGE_PATH, "v22_bridge")
    optmod = arena.base.import_module(arena.base.OPT_PATH, "v22_opt")
    epi = arena.base.import_module(arena.base.V9_PATH, "v22_epi")
    judge = arena.base.import_module(arena.base.JUDGE_PATH, "v22_judge")
    canonical = arena.base.import_module(canonical_path, "v22_canonical")
    optmod.v3_global = v3
    arena.base.install_fast_candidate_route(epi, optmod)
    changed = core.patch_vocab(args.vocab_size, HERE, canonical_path)
    log(f"[vocab] patched VOCAB={args.vocab_size:,} in {len(changed)} modules")

    # Compatibility fields used by imported constructors.
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
        root, raw_rows[0], args.vocab_size, args.tokenizer_min_frequency,
        args.tokenizer_source,
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
        log(f"[shape] {name:28s} params={shape.params:,} dTarget={delta:+.3f}% ff={shape.ff_hidden}")

    init_audit = paired_initialization_audit(args, shapes, deps, device, root)
    checkpoint_audit = checkpoint_exactness_audit(args, shapes[FAST], deps, device, root)
    log(f"[selftest] checkpoint exactness={checkpoint_audit}")

    quality_results: Dict[str, QualityResult] = {}
    for name in QUALITY_MODELS:
        log("=" * 160)
        log(f"QUALITY ARM: {name}")
        quality_results[name] = train_quality_arm(
            name, shapes[name], args, deps, train, val, test_c, test, root, device
        )
        atomic_json(root / "quality_results.json", {k: asdict(v) for k, v in quality_results.items()})

    quality = quality_decision(quality_results, args)
    atomic_json(root / "quality_decision.json", quality)
    log(f"[quality] {quality}")

    # Benchmark both scalable cache candidates so the result remains useful if
    # their quality ordering is close. Dense was already diagnosed in v21.
    system_rows: List[BenchRow] = []
    for context in args.system_contexts:
        batch = max(1, args.system_tokens_per_call // context)
        for candidate in CACHE_CANDIDATES:
            for policy in POLICIES:
                log(f"[system/train] cache={candidate} policy={policy} ctx={context} batch={batch}")
                row = benchmark_train(
                    candidate, policy, shapes[candidate], args, deps, train,
                    int(context), batch, train_c.bytes_per_token, device,
                )
                system_rows.append(row)
                log(asdict(row))
                atomic_json(root / "system_rows.json", [asdict(x) for x in system_rows])
        log(f"[system/train] transformer ctx={context} batch={batch}")
        row = benchmark_train(
            TRANSFORMER, "none", shapes[TRANSFORMER], args, deps, train,
            int(context), batch, train_c.bytes_per_token, device,
        )
        system_rows.append(row)
        log(asdict(row))
        # Inference is independent of checkpoint policy; compare only base caches.
        for name in (*CACHE_CANDIDATES, TRANSFORMER):
            log(f"[system/infer] {name} ctx={context} batch={batch}")
            row = benchmark_infer(
                name, shapes[name], args, deps, test, int(context), batch,
                test_c.bytes_per_token, device,
            )
            system_rows.append(row)
            log(asdict(row))
        atomic_json(root / "system_rows.json", [asdict(x) for x in system_rows])

    system = system_decision(quality, system_rows, args)
    atomic_json(root / "system_decision.json", system)
    result = {
        "args": vars(args),
        "canonical_source": str(canonical_path),
        "canonical_sha256": actual_sha,
        "tokenizer": str(root / "tokenizer" / "tokenizer.json"),
        "tokenizer_sha256": sha256(root / "tokenizer" / "tokenizer.json"),
        "shapes": {k: asdict(v) for k, v in shapes.items()},
        "initialization_audit": init_audit,
        "checkpoint_audit": checkpoint_audit,
        "quality_results": {k: asdict(v) for k, v in quality_results.items()},
        "quality_decision": quality,
        "system_rows": [asdict(x) for x in system_rows],
        "system_decision": system,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    atomic_json(root / "results.json", result)
    summary = make_summary(
        args, canonical_path, root / "tokenizer" / "tokenizer.json", shapes,
        quality_results, quality, system_rows, system, init_audit,
        checkpoint_audit, corpora,
    )
    atomic_text(root / "summary.txt", summary)
    log(summary)


if __name__ == "__main__":
    main()
