#!/usr/bin/env python3
"""FIELD-FUSION v28 — one-seed 98.304M-token scaling confirmation.

This run freezes every architecture and promotes only the best fair recipe found
in v27:

* Field-Fusion: batch 4, WSD, refresh/PCAF routing LR x2, field_half recompute.
* Transformer: batch 4, WSD.
* Official Mamba-2: batch 4, WSD.

All three receive the same 48,000 WikiText-103 windows, in the same order, at
context 2048 (98,304,000 tokens/model).  Full validation and test NLL are
measured with the exact streaming readout at 25.165824M, 49.152M, 73.728M and
98.304M tokens.  The program saves resumable checkpoints, BF16 inference
exports, final long-context matched-suffix measurements and an explicit next-
step decision.  It never auto-launches another experiment.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

import field_fusion_recipe_memory_v27 as v27
import field_fusion_final_ablation_v26 as v26
import field_fusion_wiki100_canonical_v23 as v23
import field_fusion_wiki100_mamba2_v25 as v25

FUSION = v27.FUSION
TRANSFORMER = v27.TRANSFORMER
MAMBA2 = v27.MAMBA2
VERSION = 28
EXPECTED_CANONICAL_SHA256 = v27.EXPECTED_CANONICAL_SHA256
Variant = v26.Variant

VARIANTS: Tuple[Variant, ...] = (
    Variant(
        "field_long_wsd_gate2x_b4", FUSION, 4, "wsd", 2.0,
        "Field-Fusion v27 winner: WSD, more updates, routing LR x2.",
    ),
    Variant(
        "transformer_long_wsd_b4", TRANSFORMER, 4, "wsd", 1.0,
        "Best fair Transformer recipe from v27: WSD and more updates.",
    ),
    Variant(
        "mamba2_long_wsd_b4", MAMBA2, 4, "wsd", 1.0,
        "Best fair official Mamba-2 recipe from v27: WSD and more updates.",
    ),
)


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


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_args() -> argparse.Namespace:
    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument("--train-token-budget", type=int, default=98_304_000)
    custom.add_argument(
        "--eval-token-milestones", nargs="+", type=int,
        default=[25_165_824, 49_152_000, 73_728_000, 98_304_000],
    )
    custom.add_argument("--checkpoint-every-updates", type=int, default=1000)
    custom.add_argument("--profile-log-every-updates", type=int, default=128)
    custom.add_argument("--stream-readout-chunk", type=int, default=512)
    custom.add_argument("--milestone-validation-token-budget", type=int, default=0)
    custom.add_argument("--milestone-test-token-budget", type=int, default=293_944)
    custom.add_argument(
        "--final-contexts", nargs="+", type=int,
        default=[2048, 4096, 8192, 16384],
    )
    custom.add_argument("--final-context-score-tokens", type=int, default=128)
    custom.add_argument("--final-context-windows", type=int, default=4)
    custom.add_argument("--run-final-context-eval", action=argparse.BooleanOptionalAction, default=True)
    custom.add_argument("--export-bf16", action=argparse.BooleanOptionalAction, default=True)
    custom.add_argument("--prefix-audit-v27", action=argparse.BooleanOptionalAction, default=True)
    custom.add_argument("--equivalence-nll", type=float, default=0.020)
    custom.add_argument("--narrow-lead-nll", type=float, default=0.050)
    custom_args, remaining = custom.parse_known_args()

    old = sys.argv
    try:
        sys.argv = [old[0], *remaining]
        args = v27.parse_args()
    finally:
        sys.argv = old
    for key, value in vars(custom_args).items():
        setattr(args, key, value)
    # v27's short-screen fields are not used for training here, but keeping them
    # synchronized makes all inherited selftests and metadata unambiguous.
    args.quality_token_budget = args.train_token_budget
    v25.add_mamba_defaults(args)
    return args


def configure(args) -> None:
    v27.configure(args)
    v26.VERSION = VERSION
    v26.VARIANTS = VARIANTS
    v26.FIELD_VARIANTS = tuple(v for v in VARIANTS if v.model == FUSION)


def validate_plan(args) -> List[int]:
    if args.train_token_budget <= 0:
        raise ValueError("train-token-budget must be positive")
    if args.train_token_budget % args.train_seq:
        raise ValueError("train-token-budget must be divisible by train-seq")
    total_sequences = args.train_token_budget // args.train_seq
    for variant in VARIANTS:
        if total_sequences % variant.batch:
            raise ValueError(f"total sequences not divisible by batch for {variant.name}")
    milestones = sorted(set(int(x) for x in args.eval_token_milestones))
    if not milestones or milestones[-1] != args.train_token_budget:
        raise ValueError("eval-token-milestones must end exactly at train-token-budget")
    quantum = args.train_seq * VARIANTS[0].batch
    for token_count in milestones:
        if token_count <= 0 or token_count > args.train_token_budget:
            raise ValueError(f"invalid milestone {token_count}")
        if token_count % quantum:
            raise ValueError(
                f"milestone {token_count} must be divisible by batch*sequence={quantum}"
            )
    if args.checkpoint_every_updates <= 0 or args.profile_log_every_updates <= 0:
        raise ValueError("checkpoint/log intervals must be positive")
    return milestones


@torch.inference_mode()
def evaluate_streaming_corpus(
    name: str,
    model: nn.Module,
    corpus,
    data: torch.Tensor,
    context: int,
    token_budget: int,
    readout_chunk: int,
    device: torch.device,
    amp: str,
) -> Dict[str, float]:
    model.eval()
    usable = min(len(data) - 1, token_budget if token_budget > 0 else len(data) - 1)
    total_nll = 0.0
    total_tokens = 0
    started = time.perf_counter()
    peak_before = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    for start in range(0, usable, context):
        length = min(context, usable - start)
        if length < 8:
            break
        win = data[start:start + length + 1].long()
        if win.device != device:
            win = win.to(device, non_blocking=True)
        x, y = win[:-1][None], win[1:][None]
        with v23.amp_ctx(device, amp):
            mean_nll = v27.streaming_nll(name, model, x, y, readout_chunk, False)
        count = int(y.numel())
        total_nll += float(mean_nll.detach().float().cpu()) * count
        total_tokens += count
    sync(device)
    seconds = time.perf_counter() - started
    mean = total_nll / max(total_tokens, 1)
    bytes_est = total_tokens * corpus.bytes_per_token
    peak_after = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return {
        "context": int(context),
        "readout_chunk": int(readout_chunk),
        "nll": mean,
        "ppl": math.exp(min(mean, 20.0)),
        "bits_per_token": mean / math.log(2.0),
        "bpb_norm": (total_nll / math.log(2.0)) / max(bytes_est, 1e-9),
        "tokens": total_tokens,
        "bytes_est": bytes_est,
        "seconds": seconds,
        "tokens_per_second": total_tokens / max(seconds, 1e-9),
        "peak_gib": max(peak_before, peak_after) / 2**30,
    }


@torch.inference_mode()
def evaluate_matched_suffix_streaming(
    name: str,
    model: nn.Module,
    data: torch.Tensor,
    contexts: Sequence[int],
    score_tokens: int,
    windows: int,
    seed: int,
    readout_chunk: int,
    device: torch.device,
    amp: str,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    out: Dict[str, Dict[str, float]] = {}
    max_context = max(int(c) for c in contexts)
    if len(data) <= max_context + 2:
        raise ValueError("not enough test tokens for final context evaluation")
    rng = np.random.default_rng(seed)
    # Use the same end positions for every context so only available history changes.
    end_positions = rng.integers(max_context, len(data) - 1, size=windows).tolist()
    for context in contexts:
        context = int(context)
        total = 0.0
        count = 0
        started = time.perf_counter()
        for end in end_positions:
            start = end - context
            win = data[start:end + 1].long()
            if win.device != device:
                win = win.to(device, non_blocking=True)
            x, y = win[:-1][None], win[1:][None]
            with v23.amp_ctx(device, amp):
                token_nll = v27.streaming_nll(
                    name, model, x, y, min(readout_chunk, context), True
                ).float()
            tail = token_nll[:, -min(score_tokens, context):]
            total += float(tail.sum().cpu())
            count += int(tail.numel())
        sync(device)
        seconds = time.perf_counter() - started
        mean = total / max(count, 1)
        out[str(context)] = {
            "context": context,
            "score_tokens": int(min(score_tokens, context)),
            "windows": int(windows),
            "nll": mean,
            "ppl": math.exp(min(mean, 20.0)),
            "tokens": count,
            "seconds": seconds,
            "tokens_per_second": count / max(seconds, 1e-9),
        }
        log(
            f"[{name}] CONTEXT ctx={context:5d} score={min(score_tokens, context):4d} "
            f"nll={mean:.5f} ppl={out[str(context)]['ppl']:.3f}"
        )
    return out


def checkpoint_signature(args, variant: Variant, shape, total_sequences: int, milestones: Sequence[int]) -> Dict[str, object]:
    return {
        "version": VERSION,
        "variant": asdict(variant),
        "shape": asdict(shape),
        "train_token_budget": int(args.train_token_budget),
        "total_sequences": int(total_sequences),
        "train_seq": int(args.train_seq),
        "milestones": [int(x) for x in milestones],
        "model_seed": int(args.model_seed),
        "embedding_seed": int(args.embedding_seed),
        "data_seed": int(args.data_seed),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "min_lr_ratio": float(args.min_lr_ratio),
        "warmup_fraction": float(args.warmup_fraction),
        "wsd_stable_fraction": float(args.wsd_stable_fraction),
        "checkpoint_policy": "field_half" if variant.model == FUSION else "none",
        "stream_readout_chunk": int(args.stream_readout_chunk),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    sequence_index: int,
    history: List[Dict[str, object]],
    compute_seconds: float,
    signature: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "signature": dict(signature),
            "sequence_index": int(sequence_index),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "compute_seconds": float(compute_seconds),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        },
        tmp,
    )
    os.replace(tmp, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer, signature: Mapping[str, object]) -> Optional[Dict[str, object]]:
    if not path.is_file():
        return None
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if raw.get("signature") != dict(signature):
        raise RuntimeError(f"checkpoint signature mismatch: {path}")
    model.load_state_dict(raw["model"], strict=True)
    optimizer.load_state_dict(raw["optimizer"])
    if raw.get("torch_rng") is not None:
        torch.set_rng_state(raw["torch_rng"])
    if raw.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(raw["cuda_rng"])
    return raw


def export_bf16_model(path: Path, name: str, model: nn.Module, shape, variant: Variant, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    state = {}
    for key, value in model.state_dict().items():
        cpu = value.detach().cpu()
        state[key] = cpu.to(torch.bfloat16) if cpu.is_floating_point() else cpu
    torch.save(
        {
            "format": "field_fusion_v28_bf16_inference",
            "version": VERSION,
            "model": name,
            "variant": asdict(variant),
            "shape": asdict(shape),
            "train_tokens": int(args.train_token_budget),
            "state_dict": state,
            "args": vars(args),
        },
        tmp,
    )
    os.replace(tmp, path)


@dataclass
class LongResult:
    variant: str
    model: str
    description: str
    batch: int
    updates: int
    schedule: str
    gate_lr_multiplier: float
    train_tokens: int
    compute_seconds: float
    tokens_per_second: float
    peak_gib: float
    evaluations: List[Dict[str, object]]
    final_validation: Dict[str, float]
    final_test: Dict[str, float]
    final_contexts: Dict[str, Dict[str, float]]
    checkpoint: str
    bf16_export: str


def train_long_variant(
    variant: Variant,
    shape,
    args,
    deps,
    train: torch.Tensor,
    val_c,
    val: torch.Tensor,
    test_c,
    test: torch.Tensor,
    starts: np.ndarray,
    milestones: Sequence[int],
    root: Path,
    device: torch.device,
) -> LongResult:
    out = root / "scaling" / variant.name
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "result.json"
    if result_path.is_file() and args.resume:
        return LongResult(**json.loads(result_path.read_text(encoding="utf-8")))

    old_policy = args.fusion_checkpoint_policy
    args.fusion_checkpoint_policy = "field_half" if variant.model == FUSION else "none"
    try:
        model = v25.build_model_v25(variant.model, shape, args, deps, device).train()
    finally:
        args.fusion_checkpoint_policy = old_policy
    optimizer = v26.make_optimizer(model, args.lr, args.weight_decay, variant.gate_lr_multiplier)

    total_sequences = len(starts)
    total_updates = total_sequences // variant.batch
    signature = checkpoint_signature(args, variant, shape, total_sequences, milestones)
    checkpoint_path = out / "latest.pt"
    history_path = out / "history.json"
    eval_path = out / "evaluations.json"
    sequence_index = 0
    history: List[Dict[str, object]] = []
    evaluations: List[Dict[str, object]] = []
    prior_compute = 0.0

    if eval_path.is_file():
        evaluations = list(json.loads(eval_path.read_text(encoding="utf-8")))
    if args.resume:
        raw = load_checkpoint(checkpoint_path, model, optimizer, signature)
        if raw is not None:
            sequence_index = int(raw["sequence_index"])
            history = list(raw.get("history", []))
            prior_compute = float(raw.get("compute_seconds", 0.0))
            log(
                f"[{variant.name}] resume update={sequence_index // variant.batch:,}/"
                f"{total_updates:,} tokens={sequence_index * args.train_seq:,}"
            )

    milestone_sequences = {int(t // args.train_seq): int(t) for t in milestones}
    completed_eval_tokens = {int(row["train_tokens"]) for row in evaluations}
    clear_cuda()
    torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    started = time.perf_counter()
    excluded = 0.0
    primary_value = float("nan")
    grad_value = float("nan")

    def perform_milestone_eval(processed_after: int, update: int, base_lr: float) -> None:
        nonlocal excluded
        if processed_after in completed_eval_tokens:
            return
        sync(device)
        pause = time.perf_counter()
        log(f"[{variant.name}] MILESTONE {processed_after:,}: full streaming validation")
        val_row = evaluate_streaming_corpus(
            variant.model, model, val_c, val, args.train_seq,
            args.milestone_validation_token_budget, args.stream_readout_chunk,
            device, args.amp,
        )
        log(
            f"[{variant.name}] VAL tokens={processed_after:,} "
            f"nll={val_row['nll']:.5f} ppl={val_row['ppl']:.3f}"
        )
        log(f"[{variant.name}] MILESTONE {processed_after:,}: full streaming test")
        test_row = evaluate_streaming_corpus(
            variant.model, model, test_c, test, args.train_seq,
            args.milestone_test_token_budget, args.stream_readout_chunk,
            device, args.amp,
        )
        log(
            f"[{variant.name}] TEST tokens={processed_after:,} "
            f"nll={test_row['nll']:.5f} ppl={test_row['ppl']:.3f}"
        )
        evaluations.append(
            {
                "train_tokens": int(processed_after),
                "update": int(update),
                "lr": float(base_lr),
                "validation": val_row,
                "test": test_row,
            }
        )
        evaluations.sort(key=lambda r: int(r["train_tokens"]))
        completed_eval_tokens.add(int(processed_after))
        atomic_json(eval_path, evaluations)
        model.train()
        sync(device)
        excluded += time.perf_counter() - pause

    # A crash during a milestone evaluation leaves a valid training checkpoint
    # exactly at that milestone.  Complete the missing evaluation before taking
    # another optimizer step so resume cannot silently skip a curve point.
    resumed_tokens = sequence_index * args.train_seq
    if sequence_index in milestone_sequences and resumed_tokens not in completed_eval_tokens:
        resumed_update = sequence_index // variant.batch
        resumed_lr = v26.lr_for_tokens(
            resumed_tokens, args.train_token_budget, args, variant.schedule
        )
        perform_milestone_eval(resumed_tokens, resumed_update, resumed_lr)

    while sequence_index < total_sequences:
        batch = variant.batch
        x, y = v26.paired_batch(train, starts, sequence_index, batch, args.train_seq, device)
        processed_after = (sequence_index + batch) * args.train_seq
        distill_progress = min(
            1.0,
            processed_after / max(args.train_token_budget * args.warmup_fraction, 1.0),
        )
        v25.set_distill_v25(model, distill_progress)
        optimizer.zero_grad(set_to_none=True)
        with v23.amp_ctx(device, args.amp):
            loss, primary = v25.loss_call_v25(variant.model, model, x, y)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        base_lr = v26.lr_for_tokens(processed_after, args.train_token_budget, args, variant.schedule)
        v26.set_optimizer_lr(optimizer, base_lr)
        optimizer.step()
        sequence_index += batch
        update = sequence_index // variant.batch
        primary_value = float(primary.detach().float().cpu())
        grad_value = float(grad.detach().float().cpu())

        if update % args.profile_log_every_updates == 0 or sequence_index in milestone_sequences:
            sync(device)
            compute = prior_compute + time.perf_counter() - started - excluded
            row = {
                "sequence_index": int(sequence_index),
                "update": int(update),
                "train_tokens": int(processed_after),
                "train_nll": primary_value,
                "train_ppl": math.exp(min(primary_value, 20.0)),
                "grad": grad_value,
                "lr": base_lr,
                "special_lr": base_lr * variant.gate_lr_multiplier,
                "tokens_per_second": processed_after / max(compute, 1e-9),
                "peak_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            }
            history.append(row)
            atomic_json(history_path, history)
            log(
                f"[{variant.name}] update={update:05d}/{total_updates} "
                f"tokens={processed_after:,}/{args.train_token_budget:,} "
                f"nll={primary_value:.5f} ppl={row['train_ppl']:.2f} "
                f"lr={base_lr:.3e} special_lr={row['special_lr']:.3e} "
                f"tok/s={row['tokens_per_second']:,.0f} peak={row['peak_gib']:.2f}G"
            )

        periodic_checkpoint = update % args.checkpoint_every_updates == 0
        at_milestone = sequence_index in milestone_sequences
        at_end = sequence_index == total_sequences
        if periodic_checkpoint or at_milestone or at_end:
            sync(device)
            compute = prior_compute + time.perf_counter() - started - excluded
            pause = time.perf_counter()
            save_checkpoint(
                checkpoint_path, model, optimizer, sequence_index, history, compute, signature
            )
            sync(device)
            io_seconds = time.perf_counter() - pause
            excluded += io_seconds
            log(f"[{variant.name}] checkpoint update={update} io={io_seconds:.1f}s")

        if at_milestone:
            perform_milestone_eval(processed_after, update, base_lr)

    sync(device)
    compute_seconds = prior_compute + time.perf_counter() - started - excluded
    peak_gib = torch.cuda.max_memory_allocated(device) / 2**30
    if not evaluations or int(evaluations[-1]["train_tokens"]) != args.train_token_budget:
        raise RuntimeError(f"missing final evaluation for {variant.name}")
    final_validation = dict(evaluations[-1]["validation"])
    final_test = dict(evaluations[-1]["test"])

    final_contexts: Dict[str, Dict[str, float]] = {}
    if args.run_final_context_eval:
        final_contexts = evaluate_matched_suffix_streaming(
            variant.model, model, test, args.final_contexts,
            args.final_context_score_tokens, args.final_context_windows,
            args.eval_seed + 280_000, args.stream_readout_chunk,
            device, args.amp,
        )
        atomic_json(out / "final_contexts.json", final_contexts)

    export_path = root / "exports" / f"{variant.name}_step{total_updates}_BF16.pt"
    if args.export_bf16:
        pause = time.perf_counter()
        export_bf16_model(export_path, variant.model, model, shape, variant, args)
        log(
            f"[{variant.name}] BF16 export={export_path} "
            f"size={export_path.stat().st_size / 2**20:.1f} MiB "
            f"io={time.perf_counter() - pause:.1f}s"
        )
    else:
        export_path = Path("")

    result = LongResult(
        variant=variant.name,
        model=variant.model,
        description=variant.description,
        batch=variant.batch,
        updates=total_updates,
        schedule=variant.schedule,
        gate_lr_multiplier=variant.gate_lr_multiplier,
        train_tokens=args.train_token_budget,
        compute_seconds=compute_seconds,
        tokens_per_second=args.train_token_budget / max(compute_seconds, 1e-9),
        peak_gib=peak_gib,
        evaluations=evaluations,
        final_validation=final_validation,
        final_test=final_test,
        final_contexts=final_contexts,
        checkpoint=str(checkpoint_path),
        bf16_export=str(export_path) if args.export_bf16 else "",
    )
    atomic_json(result_path, asdict(result))
    del model, optimizer
    clear_cuda()
    return result


def fit_log_scaling(evaluations: Sequence[Mapping[str, object]], split: str) -> Dict[str, float]:
    xs = np.log(np.asarray([float(r["train_tokens"]) for r in evaluations], dtype=np.float64))
    ys = np.asarray([float(r[split]["nll"]) for r in evaluations], dtype=np.float64)
    if len(xs) < 2:
        return {"intercept": float("nan"), "slope": float("nan"), "r2": float("nan")}
    slope, intercept = np.polyfit(xs, ys, 1)
    pred = intercept + slope * xs
    ss_res = float(np.square(ys - pred).sum())
    ss_tot = float(np.square(ys - ys.mean()).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {"intercept": float(intercept), "slope": float(slope), "r2": float(r2)}


def crossover_tokens(fit_a: Mapping[str, float], fit_b: Mapping[str, float]) -> Optional[float]:
    denom = float(fit_a["slope"]) - float(fit_b["slope"])
    if not math.isfinite(denom) or abs(denom) < 1e-12:
        return None
    log_tokens = (float(fit_b["intercept"]) - float(fit_a["intercept"])) / denom
    if not math.isfinite(log_tokens) or log_tokens < math.log(1e6) or log_tokens > math.log(1e13):
        return None
    return float(math.exp(log_tokens))


def make_decision(results: Mapping[str, LongResult], args) -> Dict[str, object]:
    by_model = {r.model: r for r in results.values()}
    field = by_model[FUSION]
    transformer = by_model[TRANSFORMER]
    mamba = by_model[MAMBA2]
    fits = {
        name: {
            "validation": fit_log_scaling(row.evaluations, "validation"),
            "test": fit_log_scaling(row.evaluations, "test"),
        }
        for name, row in by_model.items()
    }
    curve = []
    tokens = [int(r["train_tokens"]) for r in field.evaluations]
    for token_count in tokens:
        item: Dict[str, object] = {"train_tokens": token_count}
        for name, row in by_model.items():
            match = next(e for e in row.evaluations if int(e["train_tokens"]) == token_count)
            item[name] = {
                "validation_nll": float(match["validation"]["nll"]),
                "test_nll": float(match["test"]["nll"]),
            }
        item["field_minus_transformer_val"] = item[FUSION]["validation_nll"] - item[TRANSFORMER]["validation_nll"]
        item["field_minus_mamba2_val"] = item[FUSION]["validation_nll"] - item[MAMBA2]["validation_nll"]
        item["field_minus_transformer_test"] = item[FUSION]["test_nll"] - item[TRANSFORMER]["test_nll"]
        item["field_minus_mamba2_test"] = item[FUSION]["test_nll"] - item[MAMBA2]["test_nll"]
        curve.append(item)

    f_val = field.final_validation["nll"]
    f_test = field.final_test["nll"]
    rivals = [transformer, mamba]
    best_rival_val = min(rivals, key=lambda r: r.final_validation["nll"])
    best_rival_test = min(rivals, key=lambda r: r.final_test["nll"])
    final_gap_val = f_val - best_rival_val.final_validation["nll"]
    final_gap_test = f_test - best_rival_test.final_test["nll"]
    field_wins = final_gap_val < 0.0 and final_gap_test < 0.0
    equivalent = abs(final_gap_val) <= args.equivalence_nll and abs(final_gap_test) <= args.equivalence_nll

    mamba_gaps = [float(row["field_minus_mamba2_val"]) for row in curve]
    mamba_gap_change = mamba_gaps[-1] - mamba_gaps[0]
    narrowing = mamba_gap_change < -0.010

    if field_wins:
        action = "RUN_THREE_SEEDS_AND_CANONIZE"
        reason = "Field finishes ahead of both fair baselines on full validation and test."
    elif equivalent:
        action = "RUN_THREE_SEEDS_EQUIVALENCE"
        reason = "Field is within the predeclared ±0.020 NLL equivalence band of the best rival."
    elif best_rival_val.model == MAMBA2 and final_gap_val <= args.narrow_lead_nll and narrowing:
        action = "EXTEND_ONE_SEED_TO_196M"
        reason = "Mamba-2 retains a narrow lead, but the Field gap is shrinking across the measured scaling curve."
    else:
        action = "TARGETED_FIELD_QUALITY_ABLATION"
        reason = "A fair baseline retains a material lead without evidence that the gap will close safely by scaling alone."

    out = {
        "action": action,
        "reason": reason,
        "field_final_validation_nll": f_val,
        "field_final_test_nll": f_test,
        "best_rival_validation_model": best_rival_val.model,
        "best_rival_validation_nll": best_rival_val.final_validation["nll"],
        "best_rival_test_model": best_rival_test.model,
        "best_rival_test_nll": best_rival_test.final_test["nll"],
        "field_minus_best_rival_validation_nll": final_gap_val,
        "field_minus_best_rival_test_nll": final_gap_test,
        "field_minus_transformer_validation_nll": f_val - transformer.final_validation["nll"],
        "field_minus_mamba2_validation_nll": f_val - mamba.final_validation["nll"],
        "field_minus_transformer_test_nll": f_test - transformer.final_test["nll"],
        "field_minus_mamba2_test_nll": f_test - mamba.final_test["nll"],
        "mamba_gap_change_first_to_final_validation": mamba_gap_change,
        "mamba_gap_narrowing": narrowing,
        "equivalence_nll": args.equivalence_nll,
        "narrow_lead_nll": args.narrow_lead_nll,
        "scaling_fits": fits,
        "heuristic_validation_crossover_tokens": {
            "field_vs_transformer": crossover_tokens(fits[FUSION]["validation"], fits[TRANSFORMER]["validation"]),
            "field_vs_mamba2": crossover_tokens(fits[FUSION]["validation"], fits[MAMBA2]["validation"]),
        },
        "curve": curve,
    }
    return out


def starts_prefix_audit(starts: np.ndarray, root: Path, enabled: bool) -> Dict[str, object]:
    row: Dict[str, object] = {
        "new_count": int(len(starts)),
        "new_sha256": hashlib.sha256(starts.tobytes()).hexdigest(),
        "v27_found": False,
        "prefix_equal": None,
    }
    candidates = [
        Path("/home/ubuntu/pcaf_runs/field_fusion_recipe_memory_v27_run/paired_example_starts.npy"),
        root.parent / "field_fusion_recipe_memory_v27_run" / "paired_example_starts.npy",
    ]
    if enabled:
        for path in candidates:
            if path.is_file():
                old = np.load(path)
                row.update({
                    "v27_found": True,
                    "v27_path": str(path),
                    "v27_count": int(len(old)),
                    "v27_sha256": hashlib.sha256(old.tobytes()).hexdigest(),
                    "prefix_equal": bool(np.array_equal(starts[:len(old)], old)),
                })
                if not row["prefix_equal"]:
                    raise AssertionError("v28 paired examples do not preserve the v27 prefix")
                break
    atomic_json(root / "paired_prefix_audit.json", row)
    log(f"[selftest] paired_prefix={row}")
    return row


def write_scaling_csv(path: Path, decision: Mapping[str, object]) -> None:
    lines = [
        "train_tokens,field_val,transformer_val,mamba2_val,field_test,transformer_test,mamba2_test,field_minus_transformer_val,field_minus_mamba2_val"
    ]
    for row in decision["curve"]:
        lines.append(
            ",".join(
                [
                    str(row["train_tokens"]),
                    f"{row[FUSION]['validation_nll']:.8f}",
                    f"{row[TRANSFORMER]['validation_nll']:.8f}",
                    f"{row[MAMBA2]['validation_nll']:.8f}",
                    f"{row[FUSION]['test_nll']:.8f}",
                    f"{row[TRANSFORMER]['test_nll']:.8f}",
                    f"{row[MAMBA2]['test_nll']:.8f}",
                    f"{row['field_minus_transformer_val']:.8f}",
                    f"{row['field_minus_mamba2_val']:.8f}",
                ]
            )
        )
    atomic_text(path, "\n".join(lines) + "\n")


def summary(args, canonical_path: Path, actual_sha: str, shapes, results: Mapping[str, LongResult], decision: Mapping[str, object], stream_audit, prefix_audit) -> str:
    width = 210
    by_model = {r.model: r for r in results.values()}
    lines = [
        "=" * width,
        "FIELD-FUSION v28 — ONE-SEED 98.304M-TOKEN SCALING CONFIRMATION",
        "=" * width,
        f"canonical={canonical_path} sha256={actual_sha}",
        f"WikiText-103 100% | tokenizer=16,384 BPE | train_seq={args.train_seq} | paired tokens/model={args.train_token_budget:,}",
        f"milestones={','.join(f'{x:,}' for x in args.eval_token_milestones)} | WSD stable_fraction={args.wsd_stable_fraction:.2f}",
        f"v27_prefix_found={prefix_audit.get('v27_found')} prefix_equal={prefix_audit.get('prefix_equal')}",
        "",
        "FINAL RESULTS",
        f"{'variant':36s} {'model':28s} {'updates':>7s} {'val NLL':>10s} {'test NLL':>10s} {'tok/s':>10s} {'peakGB':>8s}",
    ]
    for variant in VARIANTS:
        r = results[variant.name]
        lines.append(
            f"{r.variant:36s} {r.model:28s} {r.updates:7d} "
            f"{r.final_validation['nll']:10.5f} {r.final_test['nll']:10.5f} "
            f"{r.tokens_per_second:10,.0f} {r.peak_gib:8.2f}"
        )
    lines += ["", "SCALING CURVE — FULL STREAMING VALIDATION NLL"]
    lines.append(
        f"{'tokens':>14s} {'Field':>10s} {'Transformer':>12s} {'Mamba-2':>10s} {'F-T':>10s} {'F-M':>10s}"
    )
    for row in decision["curve"]:
        lines.append(
            f"{int(row['train_tokens']):14,d} {row[FUSION]['validation_nll']:10.5f} "
            f"{row[TRANSFORMER]['validation_nll']:12.5f} {row[MAMBA2]['validation_nll']:10.5f} "
            f"{row['field_minus_transformer_val']:+10.5f} {row['field_minus_mamba2_val']:+10.5f}"
        )
    lines += [
        "",
        "FINAL GAPS",
        f"Field minus Transformer validation={decision['field_minus_transformer_validation_nll']:+.5f}",
        f"Field minus Mamba-2 validation={decision['field_minus_mamba2_validation_nll']:+.5f}",
        f"Field minus Transformer test={decision['field_minus_transformer_test_nll']:+.5f}",
        f"Field minus Mamba-2 test={decision['field_minus_mamba2_test_nll']:+.5f}",
        f"Mamba gap change first→final validation={decision['mamba_gap_change_first_to_final_validation']:+.5f}",
        "",
        "EXACT STREAMING READOUT AUDIT",
    ]
    for name, row in stream_audit.items():
        lines.append(
            f"{name:28s} max_abs={row['max_abs']:.3e} mean_abs={row['mean_abs']:.3e} pass={row['pass']}"
        )
    lines += ["", "FINAL MATCHED-SUFFIX CONTEXT NLL"]
    for context in args.final_contexts:
        vals = []
        for name in (FUSION, TRANSFORMER, MAMBA2):
            row = by_model[name].final_contexts.get(str(context), {})
            vals.append(f"{name}={row.get('nll', float('nan')):.5f}")
        lines.append(f"ctx={context:5d}: " + " | ".join(vals))
    lines += [
        "",
        "AUTOMATIC NEXT STEP",
        f"action={decision['action']}",
        f"reason={decision['reason']}",
        "No follow-up run is launched automatically.",
        "=" * width,
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    milestones = validate_plan(args)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required")
    configure(args)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.set_float32_matmul_precision("high")

    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)
    canonical_path, actual_sha, deps = v27.load_dependencies(args)
    shapes = v27.solve_shapes(args, deps)
    for name, shape in shapes.items():
        delta = 100.0 * (shape.params - args.target_params) / args.target_params
        log(
            f"[shape] {name:30s} params={shape.params:,} dTarget={delta:+.3f}% "
            f"dim={shape.dim} layers={shape.layers} ff={shape.ff_hidden}"
        )

    paired = v26.paired_initialization_audit(args, shapes[FUSION], deps, device, root)
    init = v25.initialization_audit_v25(args, shapes, deps, device, root)
    ckpt = v23.v22.checkpoint_exactness_audit(args, shapes[FUSION], deps, device, root)
    eval_pre = v25.evaluation_preflight_v25(args, shapes[FUSION], deps, device, root)
    mamba_pre = v25.mamba_strict_preflight(args, shapes[MAMBA2], deps, device, root)
    stream_audit = v27.streaming_exactness_audit(args, shapes, deps, device, root)
    log(f"[selftest] paired={paired}")
    log(f"[selftest] initialization={init}")
    log(f"[selftest] checkpoint={ckpt}")
    log(f"[selftest] evaluation={eval_pre}")
    log(f"[selftest] mamba={mamba_pre}")

    raw_rows = v23.core.load_raw_rows(args.cache_dir, args.data_frac)
    tokenizer = v23.base.copy_or_train_tokenizer(
        root, raw_rows[0], args.vocab_size,
        args.tokenizer_min_frequency, args.tokenizer_source,
    )
    train_c, val_c, test_c = v23.core.save_or_load_corpora(root, tokenizer, raw_rows)
    train = v23.core.place_tokens(train_c.tokens, device, args.data_device, "train")
    val = v23.core.place_tokens(val_c.tokens, device, args.data_device, "validation")
    test = v23.core.place_tokens(test_c.tokens, device, args.data_device, "test")

    total_sequences = args.train_token_budget // args.train_seq
    starts = v26.make_example_starts(
        total_sequences, len(train) - args.train_seq - 1,
        args.data_seed, root / "paired_example_starts.npy",
    )
    prefix_audit = starts_prefix_audit(starts, root, args.prefix_audit_v27)

    results: Dict[str, LongResult] = {}
    for variant in VARIANTS:
        log("=" * 200)
        log(f"SCALING ARM: {variant.name} — {variant.description}")
        results[variant.name] = train_long_variant(
            variant, shapes[variant.model], args, deps,
            train, val_c, val, test_c, test, starts, milestones, root, device,
        )
        atomic_json(root / "scaling_results.json", {k: asdict(v) for k, v in results.items()})

    decision = make_decision(results, args)
    atomic_json(root / "decision.json", decision)
    write_scaling_csv(root / "scaling_curve.csv", decision)

    result = {
        "version": VERSION,
        "args": vars(args),
        "canonical_source": str(canonical_path),
        "canonical_sha256": actual_sha,
        "shapes": {k: asdict(v) for k, v in shapes.items()},
        "selftests": {
            "paired": paired,
            "initialization": init,
            "checkpoint": ckpt,
            "evaluation": eval_pre,
            "mamba": mamba_pre,
            "streaming_exactness": stream_audit,
            "paired_prefix": prefix_audit,
        },
        "results": {k: asdict(v) for k, v in results.items()},
        "decision": decision,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "mamba_ssm": v25.MAMBA_VERSION,
        "causal_conv1d": v25.CAUSAL_CONV1D_VERSION,
    }
    atomic_json(root / "results.json", result)
    text = summary(
        args, canonical_path, actual_sha, shapes, results,
        decision, stream_audit, prefix_audit,
    )
    atomic_text(root / "summary.txt", text)
    log(text)


if __name__ == "__main__":
    main()
