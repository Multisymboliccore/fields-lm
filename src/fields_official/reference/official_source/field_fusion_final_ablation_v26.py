#!/usr/bin/env python3
"""FIELD-FUSION FINAL ABLATION v26 — quality, memory and speed gate.

Purpose
-------
Freeze the promoted v25 architecture and test only low-risk changes:
  * more optimizer updates at the same token budget;
  * WSD versus cosine learning-rate schedules;
  * a conservative 2x LR multiplier for refresh/PCAF routing parameters;
  * exact activation-recompute policies and physical-batch sweeps.

The quality stage uses WikiText-103 100%, paired initialization, identical token
windows and exactly 25,165,824 training tokens per arm.  It does not auto-launch
a new 98M-token canonical run.  It promotes at most one recipe and prints the
exact next-run recommendation.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import hashlib
import json
import math
import os
import statistics
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import field_fusion_wiki100_canonical_v23 as v23
import field_fusion_wiki100_mamba2_v25 as v25

FUSION = v23.FUSION
TRANSFORMER = v23.TRANSFORMER
VERSION = 26
EXPECTED_CANONICAL_SHA256 = v23.EXPECTED_CANONICAL_SHA256


@dataclass(frozen=True)
class Variant:
    name: str
    model: str
    batch: int
    schedule: str
    gate_lr_multiplier: float
    description: str


VARIANTS: Tuple[Variant, ...] = (
    Variant("field_ref_cos_b8", FUSION, 8, "cosine", 1.0,
            "Exact v25 recipe scaled to the 25.17M-token screen."),
    Variant("field_updates_cos_b4", FUSION, 4, "cosine", 1.0,
            "Twice as many optimizer updates; same examples and token budget."),
    Variant("field_wsd_b8", FUSION, 8, "wsd", 1.0,
            "Warmup-stable-decay schedule; canonical batch."),
    Variant("field_updates_wsd_b4", FUSION, 4, "wsd", 1.0,
            "More updates plus WSD."),
    Variant("field_updates_wsd_gate2x_b4", FUSION, 4, "wsd", 2.0,
            "More updates plus WSD and 2x LR only for refresh/PCAF routing."),
    Variant("transformer_ref_cos_b8", TRANSFORMER, 8, "cosine", 1.0,
            "Matched Transformer anchor at the same examples and token budget."),
)

FIELD_VARIANTS = tuple(v for v in VARIANTS if v.model == FUSION)
REF_NAME = "field_ref_cos_b8"
TF_NAME = "transformer_ref_cos_b8"
POLICIES = ("none", "field_half", "field_all", "all")


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


def parse_args() -> argparse.Namespace:
    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument("--quality-token-budget", type=int, default=25_165_824)
    custom.add_argument("--quality-gate", type=float, default=0.010)
    custom.add_argument("--quality-safe-gap", type=float, default=0.005)
    custom.add_argument("--warmup-fraction", type=float, default=0.05)
    custom.add_argument("--wsd-stable-fraction", type=float, default=0.70)
    custom.add_argument("--eval-fractions", nargs="+", type=float,
                        default=[0.25, 0.50, 0.75, 1.00])
    custom.add_argument("--diagnostic-checkpoint", default="")
    custom.add_argument("--diagnostic-windows", type=int, default=8)
    custom.add_argument("--system-batches", nargs="+", type=int,
                        default=[1, 2, 4, 8, 12, 16])
    custom.add_argument("--max-system-tokens-per-call", type=int, default=65_536)
    custom.add_argument("--max-vram-fraction", type=float, default=0.92)
    custom.add_argument("--keep-loser-checkpoints",
                        action=argparse.BooleanOptionalAction, default=True)
    custom.add_argument("--run-diagnostics",
                        action=argparse.BooleanOptionalAction, default=True)
    custom.add_argument("--run-quality",
                        action=argparse.BooleanOptionalAction, default=True)
    custom.add_argument("--run-system",
                        action=argparse.BooleanOptionalAction, default=True)
    custom.add_argument("--profile-log-every-tokens", type=int, default=1_048_576)
    custom.add_argument("--screen-validation-token-budget", type=int, default=0)
    custom.add_argument("--screen-test-token-budget", type=int, default=293_944)
    custom_args, remaining = custom.parse_known_args()
    old = sys.argv
    try:
        sys.argv = [old[0], *remaining]
        args = v23.parse_args()
    finally:
        sys.argv = old
    for key, value in vars(custom_args).items():
        setattr(args, key, value)
    return args


def configure_builder_args(args) -> None:
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


def load_dependencies(args):
    arena = v23.base.import_module(v23.base.V15_PATH, "field_scale_50m_v15_for_v26")
    canonical_path = v23.base.locate_canonical(args.canonical_source)
    actual_sha = sha256(canonical_path)
    if actual_sha != EXPECTED_CANONICAL_SHA256:
        raise RuntimeError(
            f"canonical SHA mismatch expected={EXPECTED_CANONICAL_SHA256} "
            f"actual={actual_sha} path={canonical_path}"
        )
    v3 = arena.base.import_module(arena.base.V3_PATH, "v26_v3")
    bridge = arena.base.import_module(arena.base.BRIDGE_PATH, "v26_bridge")
    optmod = arena.base.import_module(arena.base.OPT_PATH, "v26_opt")
    epi = arena.base.import_module(arena.base.V9_PATH, "v26_epi")
    judge = arena.base.import_module(arena.base.JUDGE_PATH, "v26_judge")
    canonical = arena.base.import_module(canonical_path, "v26_canonical")
    optmod.v3_global = v3
    arena.base.install_fast_candidate_route(epi, optmod)
    changed = v23.core.patch_vocab(args.vocab_size, Path(__file__).resolve().parent, canonical_path)
    log(f"[vocab] patched VOCAB={args.vocab_size:,} in {len(changed)} modules")
    return canonical_path, actual_sha, (arena, v3, canonical, bridge, optmod, epi, judge)


def _internal_solver_name(name: str) -> str:
    """Translate v26 public model names to the implementation names known by v21."""
    try:
        v22_public = v23.TO_V22[name]
    except KeyError as exc:
        raise KeyError(f"v26 has no v23->v22 mapping for model {name!r}") from exc
    internal = v23.v22.mapped(v22_public)
    if name == FUSION and internal == name:
        raise AssertionError(
            "Field-Fusion alias was not translated before entering the v21 solver"
        )
    return internal


def solve_shapes(args, deps) -> Dict[str, v23.Shape]:
    """Solve only the Field-Fusion and Transformer shapes with explicit aliases.

    v23.solve_shapes forwards the public name ``fusion_fast`` into v21, whose
    builder only knows ``fusion_v21_fast_pcaf``.  Resolve that full chain here
    instead of mutating v23.MODELS and relying on the older wrapper.
    """
    arena, v3, canonical, bridge, optmod, epi, judge = deps
    out: Dict[str, v23.Shape] = {}
    alias_audit: Dict[str, str] = {}
    for name in (FUSION, TRANSFORMER):
        internal_name = _internal_solver_name(name)
        alias_audit[name] = internal_name
        raw = v23.v21.solve_shape(
            internal_name, args, arena, v3, canonical, bridge,
            optmod, epi, judge,
        )
        shape = v23.Shape(
            name, raw.params, raw.dim, raw.layers, raw.heads, raw.ff_hidden,
        )
        delta = 100.0 * (shape.params - args.target_params) / args.target_params
        if abs(delta) > args.max_param_delta_pct:
            raise RuntimeError(f"parameter mismatch {name}: {delta:+.3f}%")
        out[name] = shape
    log(f"[selftest] solver_aliases={alias_audit}")
    return out


def model_fingerprint(model: nn.Module) -> str:
    return v23.v22.module_hash(model)


def paired_initialization_audit(args, shape: v23.Shape, deps,
                                device: torch.device, root: Path) -> Dict[str, object]:
    hashes: Dict[str, str] = {}
    embeddings: Dict[str, str] = {}
    original_policy = args.fusion_checkpoint_policy
    try:
        args.fusion_checkpoint_policy = "field_half"
        # Two independent builds are sufficient: every quality arm uses this
        # same constructor and differs only after optimizer creation.
        for label in ("reference_build", "repeat_build"):
            model = v23.build_model(FUSION, shape, args, deps, device)
            hashes[label] = model_fingerprint(model)
            embeddings[label] = v23.tensor_hash(model.emb.weight)
            del model
            clear_cuda()
    finally:
        args.fusion_checkpoint_policy = original_policy
    if len(set(hashes.values())) != 1 or len(set(embeddings.values())) != 1:
        raise AssertionError("Field arm initialization mismatch")
    out = {
        "full_model_hash": next(iter(hashes.values())),
        "embedding_hash": next(iter(embeddings.values())),
        "builds": hashes,
        "applies_to_arms": [v.name for v in FIELD_VARIANTS],
    }
    atomic_json(root / "paired_initialization_audit.json", out)
    return out


def make_example_starts(total_sequences: int, max_start: int, seed: int,
                        path: Path) -> np.ndarray:
    if path.is_file():
        starts = np.load(path)
        if starts.shape == (total_sequences,) and int(starts.max()) < max_start:
            return starts.astype(np.int64, copy=False)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, max_start, size=total_sequences, dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, starts)
    return starts


def paired_batch(data: torch.Tensor, starts: np.ndarray, first: int, batch: int,
                 seq: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    s = torch.as_tensor(starts[first:first + batch], dtype=torch.long, device=device)
    offsets = torch.arange(seq + 1, device=device, dtype=torch.long)
    indices = s[:, None] + offsets[None, :]
    if data.device == device:
        windows = data[indices].long()
    else:
        cpu_indices = indices.cpu()
        windows = data[cpu_indices].long().to(device, non_blocking=True)
    return windows[:, :-1], windows[:, 1:]


def is_special_parameter(name: str) -> bool:
    keys = (
        "residual_gate", "global_mix_logit", ".cache.router",
        ".cache.state_gate", ".cache.evidence_", ".cache.recency_scale",
    )
    return any(key in name for key in keys)


def make_optimizer(model: nn.Module, lr: float, weight_decay: float,
                   gate_multiplier: float):
    groups: Dict[Tuple[bool, bool], List[nn.Parameter]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        special = is_special_parameter(name)
        decay = param.ndim >= 2
        groups.setdefault((special, decay), []).append(param)
    param_groups = []
    for (special, decay), params in groups.items():
        param_groups.append({
            "params": params,
            "weight_decay": weight_decay if decay else 0.0,
            "lr_scale": gate_multiplier if special else 1.0,
        })
    kwargs = dict(lr=lr, betas=(0.9, 0.95), eps=1.0e-8)
    try:
        return torch.optim.AdamW(param_groups, fused=True, **kwargs)
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(param_groups, **kwargs)


def lr_for_tokens(processed_tokens: int, total_tokens: int, args,
                  schedule: str) -> float:
    progress = min(max(processed_tokens / max(total_tokens, 1), 0.0), 1.0)
    warm = float(args.warmup_fraction)
    if progress <= warm:
        return args.lr * progress / max(warm, 1e-9)
    if schedule == "cosine":
        p = (progress - warm) / max(1.0 - warm, 1e-9)
        c = 0.5 * (1.0 + math.cos(math.pi * p))
        return args.lr * (args.min_lr_ratio + (1.0 - args.min_lr_ratio) * c)
    if schedule == "wsd":
        stable = max(warm, min(float(args.wsd_stable_fraction), 0.95))
        if progress <= stable:
            return args.lr
        p = (progress - stable) / max(1.0 - stable, 1e-9)
        c = 0.5 * (1.0 + math.cos(math.pi * p))
        return args.lr * (args.min_lr_ratio + (1.0 - args.min_lr_ratio) * c)
    raise ValueError(schedule)


def set_optimizer_lr(optimizer, base_lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = base_lr * float(group.get("lr_scale", 1.0))


def checkpoint_signature(args, variant: Variant, shape: v23.Shape,
                         total_sequences: int) -> Dict[str, object]:
    return {
        "version": VERSION,
        "variant": asdict(variant),
        "shape": asdict(shape),
        "quality_token_budget": args.quality_token_budget,
        "total_sequences": total_sequences,
        "train_seq": args.train_seq,
        "model_seed": args.model_seed,
        "embedding_seed": args.embedding_seed,
        "data_seed": args.data_seed,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "min_lr_ratio": args.min_lr_ratio,
        "warmup_fraction": args.warmup_fraction,
        "wsd_stable_fraction": args.wsd_stable_fraction,
        "checkpoint_policy": "field_half" if variant.model == FUSION else "none",
    }


def save_training_checkpoint(path: Path, model: nn.Module, optimizer,
                             sequence_index: int, history, best_nll: float,
                             compute_seconds: float, signature) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({
        "signature": signature,
        "sequence_index": int(sequence_index),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "history": history,
        "best_validation_nll": float(best_nll),
        "compute_seconds": float(compute_seconds),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }, tmp)
    os.replace(tmp, path)


def load_training_checkpoint(path: Path, model: nn.Module, optimizer,
                             signature) -> Optional[Dict[str, object]]:
    if not path.is_file():
        return None
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if raw.get("signature") != signature:
        raise RuntimeError(f"checkpoint signature mismatch: {path}")
    model.load_state_dict(raw["model"], strict=True)
    optimizer.load_state_dict(raw["optimizer"])
    if raw.get("torch_rng") is not None:
        torch.set_rng_state(raw["torch_rng"])
    if raw.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(raw["cuda_rng"])
    return raw


@dataclass
class QualityResult:
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
    best_validation_nll: float
    final_validation: Dict[str, float]
    final_test: Dict[str, float]
    history: List[Dict[str, object]]
    checkpoint: str


def train_variant(variant: Variant, shape: v23.Shape, args, deps,
                  train: torch.Tensor, val_c, val: torch.Tensor,
                  test_c, test: torch.Tensor, starts: np.ndarray,
                  root: Path, device: torch.device) -> QualityResult:
    out = root / "quality" / variant.name
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "result.json"
    if result_path.is_file() and args.resume:
        return QualityResult(**json.loads(result_path.read_text(encoding="utf-8")))

    old_policy = args.fusion_checkpoint_policy
    args.fusion_checkpoint_policy = "field_half" if variant.model == FUSION else "none"
    try:
        model = v23.build_model(variant.model, shape, args, deps, device).train()
    finally:
        args.fusion_checkpoint_policy = old_policy
    optimizer = make_optimizer(model, args.lr, args.weight_decay,
                               variant.gate_lr_multiplier)
    total_sequences = len(starts)
    if total_sequences % variant.batch:
        raise ValueError(f"total sequences {total_sequences} not divisible by batch {variant.batch}")
    total_updates = total_sequences // variant.batch
    signature = checkpoint_signature(args, variant, shape, total_sequences)
    checkpoint_path = out / "latest.pt"
    sequence_index = 0
    history: List[Dict[str, object]] = []
    best_nll = float("inf")
    prior_compute = 0.0
    if args.resume:
        raw = load_training_checkpoint(checkpoint_path, model, optimizer, signature)
        if raw is not None:
            sequence_index = int(raw["sequence_index"])
            history = list(raw.get("history", []))
            best_nll = float(raw.get("best_validation_nll", float("inf")))
            prior_compute = float(raw.get("compute_seconds", 0.0))
            log(f"[{variant.name}] resume sequences={sequence_index:,}/{total_sequences:,}")

    milestones = sorted({
        min(total_sequences, max(variant.batch,
            int(round(total_sequences * f / variant.batch)) * variant.batch))
        for f in args.eval_fractions
    })
    milestone_set = set(milestones)
    next_log_tokens = ((sequence_index * args.train_seq // args.profile_log_every_tokens) + 1) * args.profile_log_every_tokens
    excluded = 0.0
    clear_cuda()
    torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    started = time.perf_counter()
    primary_value = float("nan")
    grad_value = float("nan")

    while sequence_index < total_sequences:
        batch = min(variant.batch, total_sequences - sequence_index)
        x, y = paired_batch(train, starts, sequence_index, batch,
                            args.train_seq, device)
        processed_after = (sequence_index + batch) * args.train_seq
        distill_progress = min(1.0, processed_after /
                               max(args.quality_token_budget * args.warmup_fraction, 1.0))
        v23.set_distill(model, distill_progress)
        optimizer.zero_grad(set_to_none=True)
        with v23.amp_ctx(device, args.amp):
            loss, primary = v23.loss_call(variant.model, model, x, y)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        base_lr = lr_for_tokens(processed_after, args.quality_token_budget,
                                args, variant.schedule)
        set_optimizer_lr(optimizer, base_lr)
        optimizer.step()
        sequence_index += batch
        primary_value = float(primary.detach().float().cpu())
        grad_value = float(grad.detach().float().cpu())

        do_log = processed_after >= next_log_tokens or sequence_index in milestone_set
        if do_log:
            sync(device)
            compute = prior_compute + time.perf_counter() - started - excluded
            row = {
                "sequence_index": sequence_index,
                "update": sequence_index // variant.batch,
                "train_tokens": processed_after,
                "train_nll": primary_value,
                "train_ppl": math.exp(min(primary_value, 20.0)),
                "grad": grad_value,
                "lr": base_lr,
                "special_lr": base_lr * variant.gate_lr_multiplier,
                "tokens_per_second": processed_after / max(compute, 1e-9),
                "peak_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            }
            history.append(row)
            log(
                f"[{variant.name}] update={row['update']:04d}/{total_updates} "
                f"tokens={processed_after:,}/{args.quality_token_budget:,} "
                f"nll={primary_value:.5f} ppl={row['train_ppl']:.2f} "
                f"lr={base_lr:.3e} gate_lr={row['special_lr']:.3e} "
                f"tok/s={row['tokens_per_second']:,.0f} peak={row['peak_gib']:.2f}G"
            )
            while next_log_tokens <= processed_after:
                next_log_tokens += args.profile_log_every_tokens

        if sequence_index in milestone_set:
            sync(device)
            pause = time.perf_counter()
            val_row = v23.evaluate_fixed_windows(
                variant.model, model, val, args.train_seq, args.eval_windows,
                args.eval_seed, device, args.amp,
            )
            model.train()
            best_nll = min(best_nll, val_row["nll"])
            log(f"[{variant.name}] VAL tokens={processed_after:,} nll={val_row['nll']:.5f} ppl={val_row['ppl']:.3f}")
            sync(device)
            excluded += time.perf_counter() - pause
            compute = prior_compute + time.perf_counter() - started - excluded
            pause = time.perf_counter()
            save_training_checkpoint(
                checkpoint_path, model, optimizer, sequence_index, history,
                best_nll, compute, signature,
            )
            sync(device)
            excluded += time.perf_counter() - pause
            atomic_json(out / "history.json", history)

    sync(device)
    compute_seconds = prior_compute + time.perf_counter() - started - excluded
    peak_gib = torch.cuda.max_memory_allocated(device) / 2**30
    final_validation = v23.evaluate_stream(
        variant.model, model, val_c, val, args.train_seq,
        args.screen_validation_token_budget, device, args.amp,
    )
    final_test = v23.evaluate_stream(
        variant.model, model, test_c, test, args.train_seq,
        args.screen_test_token_budget, device, args.amp,
    )
    result = QualityResult(
        variant=variant.name,
        model=variant.model,
        description=variant.description,
        batch=variant.batch,
        updates=total_updates,
        schedule=variant.schedule,
        gate_lr_multiplier=variant.gate_lr_multiplier,
        train_tokens=args.quality_token_budget,
        compute_seconds=compute_seconds,
        tokens_per_second=args.quality_token_budget / max(compute_seconds, 1e-9),
        peak_gib=peak_gib,
        best_validation_nll=best_nll,
        final_validation=final_validation,
        final_test=final_test,
        history=history,
        checkpoint=str(checkpoint_path),
    )
    atomic_json(result_path, asdict(result))
    del model, optimizer
    clear_cuda()
    return result


def percentile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.float().reshape(-1), q).cpu())


def cache_diagnostics(cache: nn.Module, states: torch.Tensor,
                      logits: torch.Tensor, tokens: torch.Tensor,
                      targets: torch.Tensor) -> Dict[str, float]:
    b, t, _ = states.shape
    module = sys.modules.get(cache.__class__.__module__)
    fast = getattr(module, "causal_recent_candidates_i32", None) if module else None
    if bool(getattr(cache, "use_i32", False)) and fast is not None:
        idx = fast(tokens, cache.order, cache.num_buckets, cache.top_k, cache._v3)
    else:
        idx = cache._v3.causal_recent_candidates(tokens, cache.order,
                                                  cache.num_buckets, cache.top_k)
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
    scores = scores.masked_fill(~valid, -1e9)
    weights = torch.softmax(scores.float(), dim=-1) * valid.float()
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
    cand_tokens = targets[batch_idx, safe]
    target_cache = (weights * (cand_tokens == targets[:, :, None]).float()).sum(-1)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    param_nll = F.cross_entropy(flat_logits.float(), flat_targets, reduction="none")
    param_target = torch.exp(-param_nll)
    active = has.reshape(-1)
    gate = torch.zeros_like(param_target)
    if bool(active.any()):
        state_logit = cache.state_gate(states).float().squeeze(-1).reshape(-1)
        features = cache._features(
            scores.reshape(-1, cache.top_k)[active],
            weights.reshape(-1, cache.top_k)[active],
            valid.reshape(-1, cache.top_k)[active],
            cand_tokens.reshape(-1, cache.top_k)[active],
            recency.reshape(-1, cache.top_k)[active],
            flat_logits[active],
        )
        route = cache.router(features)
        if cache.router_mode == "v5":
            gl = state_logit[active] + route[:, 0]
        else:
            cache_conf = features[:, 5].clamp(1e-4, 1 - 1e-4)
            param_conf = features[:, 8].clamp(1e-4, 1 - 1e-4)
            evidence = torch.logit(cache_conf) - torch.logit(param_conf)
            evidence = evidence + 1.25 * features[:, 6] - 0.50 * features[:, 7]
            evidence = evidence + 0.35 * features[:, 10] + 0.25 * features[:, 3]
            state_term = 0.0 if cache.router_mode == "confidence_nostate" else state_logit[active]
            gl = state_term + route[:, 0] + cache.evidence_gain * evidence + cache.evidence_bias
        gate[active] = torch.sigmoid(gl)
    tc = target_cache.reshape(-1)
    win = tc > param_target
    return {
        "candidate_coverage": float(has.float().mean().cpu()),
        "mean_candidates": float(valid.float().sum(-1).mean().cpu()),
        "gate_mean": float(gate.mean().cpu()),
        "gate_on_active": float(gate[active].mean().cpu()) if bool(active.any()) else 0.0,
        "cache_win_rate": float(win[active].float().mean().cpu()) if bool(active.any()) else 0.0,
        "mean_log_advantage_active": float((torch.log(tc[active].clamp_min(1e-8)) - torch.log(param_target[active].clamp_min(1e-8))).mean().cpu()) if bool(active.any()) else 0.0,
    }


def load_checkpoint_state(path: Path) -> Mapping[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, Mapping) and "state_dict" in raw:
        return raw["state_dict"]
    if isinstance(raw, Mapping) and "model" in raw:
        return raw["model"]
    if isinstance(raw, Mapping) and all(torch.is_tensor(v) for v in raw.values()):
        return raw
    raise RuntimeError(f"unrecognized checkpoint format: {path}")


def run_diagnostics(args, shape: v23.Shape, deps, val: torch.Tensor,
                    root: Path, device: torch.device) -> Dict[str, object]:
    path = Path(args.diagnostic_checkpoint).expanduser()
    if not path.is_file():
        out = {"status": "skipped", "reason": f"checkpoint not found: {path}"}
        atomic_json(root / "diagnostics.json", out)
        return out
    old_policy = args.fusion_checkpoint_policy
    args.fusion_checkpoint_policy = "none"
    try:
        model = v23.build_model(FUSION, shape, args, deps, device).eval()
    finally:
        args.fusion_checkpoint_policy = old_policy
    state = load_checkpoint_state(path)
    model.load_state_dict(state, strict=True)
    del state
    clear_cuda()

    baseline = v23.evaluate_fixed_windows(
        FUSION, model, val, args.train_seq, args.diagnostic_windows,
        args.eval_seed + 300_000, device, args.amp,
    )
    refresh_blocks = [b for b in model.blocks if isinstance(b, v23.v21.FusionRefreshBlockV21)]
    gate_rows = []
    hooks = []
    captured: Dict[int, List[torch.Tensor]] = {i: [] for i in range(len(refresh_blocks))}

    for i, block in enumerate(refresh_blocks):
        def hook(mod, inputs, idx=i):
            x = inputs[0]
            z = mod.norm1(x)
            if mod.light_gate:
                g = torch.sigmoid(mod.residual_gate_logit).expand_as(z)
            else:
                g = torch.sigmoid(mod.residual_gate(z))
            captured[idx].append(g.detach())
        hooks.append(block.register_forward_pre_hook(hook))

    start = int((args.eval_seed * 7919) % max(1, len(val) - args.train_seq - 1))
    win = val[start:start + args.train_seq + 1].long().to(device)
    x, y = win[:-1][None], win[1:][None]
    with torch.no_grad(), v23.amp_ctx(device, args.amp):
        states, logits = model.states_logits(x)
        cache_row = cache_diagnostics(model.cache, states, logits, x, y)
    for h in hooks:
        h.remove()
    for i, block in enumerate(refresh_blocks):
        g = torch.cat([z.reshape(-1) for z in captured[i]])
        gate_rows.append({
            "index": i,
            "window": int(block.window),
            "gate_mean": float(g.float().mean().cpu()),
            "gate_std": float(g.float().std().cpu()),
            "gate_p10": percentile(g, 0.10),
            "gate_p50": percentile(g, 0.50),
            "gate_p90": percentile(g, 0.90),
            "global_mix": float(torch.sigmoid(block.attn.global_mix_logit).cpu()),
        })

    ablations = []
    cache_enabled = bool(model.cache.enabled)
    model.cache.enabled = False
    row = v23.evaluate_fixed_windows(
        FUSION, model, val, args.train_seq, args.diagnostic_windows,
        args.eval_seed + 300_000, device, args.amp,
    )
    ablations.append({"ablation": "pcaf_off", "nll": row["nll"],
                      "delta_nll": row["nll"] - baseline["nll"]})
    model.cache.enabled = cache_enabled

    for i, block in enumerate(refresh_blocks):
        if block.light_gate:
            saved = block.residual_gate_logit.detach().clone()
            block.residual_gate_logit.data.fill_(-20.0)
        else:
            saved_w = block.residual_gate.weight.detach().clone()
            saved_b = block.residual_gate.bias.detach().clone()
            block.residual_gate.weight.data.zero_()
            block.residual_gate.bias.data.fill_(-20.0)
        row = v23.evaluate_fixed_windows(
            FUSION, model, val, args.train_seq, args.diagnostic_windows,
            args.eval_seed + 300_000, device, args.amp,
        )
        ablations.append({"ablation": f"refresh_{i}_window_{block.window}_off",
                          "nll": row["nll"], "delta_nll": row["nll"] - baseline["nll"]})
        if block.light_gate:
            block.residual_gate_logit.data.copy_(saved)
        else:
            block.residual_gate.weight.data.copy_(saved_w)
            block.residual_gate.bias.data.copy_(saved_b)

    saved_mix = [b.attn.global_mix_logit.detach().clone() for b in refresh_blocks]
    for b in refresh_blocks:
        b.attn.global_mix_logit.data.fill_(-20.0)
    row = v23.evaluate_fixed_windows(
        FUSION, model, val, args.train_seq, args.diagnostic_windows,
        args.eval_seed + 300_000, device, args.amp,
    )
    ablations.append({"ablation": "all_landmarks_off", "nll": row["nll"],
                      "delta_nll": row["nll"] - baseline["nll"]})
    for b, saved in zip(refresh_blocks, saved_mix):
        b.attn.global_mix_logit.data.copy_(saved)

    out = {
        "status": "ok", "checkpoint": str(path), "baseline": baseline,
        "refresh_gates": gate_rows, "pcaf": cache_row, "ablations": ablations,
    }
    atomic_json(root / "diagnostics.json", out)
    del model, x, y, states, logits
    clear_cuda()
    return out


def quality_decision(results: Mapping[str, QualityResult], args) -> Dict[str, object]:
    ref = results[REF_NAME]
    tf = results[TF_NAME]
    rows = []
    for variant in FIELD_VARIANTS:
        r = results[variant.name]
        delta = r.final_validation["nll"] - ref.final_validation["nll"]
        rows.append({
            "variant": variant.name,
            "validation_nll": r.final_validation["nll"],
            "test_nll": r.final_test["nll"],
            "delta_nll_vs_ref": delta,
            "beats_transformer_validation": r.final_validation["nll"] < tf.final_validation["nll"],
            "speed_ratio_vs_ref": r.tokens_per_second / ref.tokens_per_second,
            "peak_ratio_vs_ref": r.peak_gib / ref.peak_gib,
            "strict_pass": delta <= -args.quality_gate,
            "safe": delta <= args.quality_safe_gap,
        })
    best = min(rows, key=lambda x: (x["validation_nll"], -x["speed_ratio_vs_ref"]))
    promote = bool(best["strict_pass"])
    recommended = best["variant"] if promote else REF_NAME
    return {
        "reference": REF_NAME,
        "transformer_anchor": TF_NAME,
        "rows": rows,
        "best_field": best["variant"],
        "promote": promote,
        "recommended": recommended,
        "reason": (f"promote: validation NLL improved by {-best['delta_nll_vs_ref']:.5f}"
                   if promote else
                   f"hold v25 recipe: best improvement {-best['delta_nll_vs_ref']:.5f} < gate {args.quality_gate:.5f}"),
        "field_vs_transformer_validation_nll": results[recommended].final_validation["nll"] - tf.final_validation["nll"],
    }


def run_system_sweep(args, shapes, deps, train, test, bytes_train: float,
                     bytes_test: float, root: Path, device: torch.device):
    rows = []
    total_vram = torch.cuda.get_device_properties(device).total_memory / 2**30
    original_policy = args.fusion_checkpoint_policy
    for context in args.system_contexts:
        for batch in args.system_batches:
            if batch * int(context) > args.max_system_tokens_per_call:
                continue
            for policy in POLICIES:
                args.fusion_checkpoint_policy = policy
                log(f"[system/train] Field policy={policy} ctx={context} batch={batch}")
                row = v23.benchmark_train(
                    FUSION, shapes[FUSION], args, deps, train, int(context),
                    int(batch), bytes_train, device,
                )
                rows.append(asdict(row))
                atomic_json(root / "system_rows.json", rows)
            args.fusion_checkpoint_policy = "none"
            log(f"[system/train] Transformer ctx={context} batch={batch}")
            row = v23.benchmark_train(
                TRANSFORMER, shapes[TRANSFORMER], args, deps, train,
                int(context), int(batch), bytes_train, device,
            )
            rows.append(asdict(row))
            atomic_json(root / "system_rows.json", rows)
        infer_batches = [b for b in args.system_batches
                         if b * int(context) <= args.max_system_tokens_per_call]
        for batch in infer_batches:
            args.fusion_checkpoint_policy = "none"
            for name in (FUSION, TRANSFORMER):
                log(f"[system/infer] {name} ctx={context} batch={batch}")
                row = v23.benchmark_infer(
                    name, shapes[name], args, deps, test, int(context),
                    int(batch), bytes_test, device,
                )
                rows.append(asdict(row))
                atomic_json(root / "system_rows.json", rows)
    args.fusion_checkpoint_policy = original_policy

    best = []
    for kind in ("train", "infer"):
        for model in (FUSION, TRANSFORMER):
            for context in args.system_contexts:
                candidates = [r for r in rows if r["kind"] == kind and r["model"] == model
                              and r["context"] == int(context) and r["status"] == "ok"
                              and r["peak_gib"] is not None
                              and r["peak_gib"] <= args.max_vram_fraction * total_vram]
                if not candidates:
                    continue
                winner = max(candidates, key=lambda r: r["tokens_per_second"] or 0.0)
                best.append(winner)
    decision = {
        "gpu_total_gib": total_vram,
        "max_vram_fraction": args.max_vram_fraction,
        "best_rows": best,
    }
    atomic_json(root / "system_decision.json", decision)
    return rows, decision


def make_summary(args, canonical_path: Path, actual_sha: str, shapes,
                 corpora, diagnostics, quality_results, quality, system):
    width = 190
    lines = [
        "=" * width,
        "FIELD-FUSION FINAL ABLATION v26 — QUALITY / MEMORY / SPEED",
        "=" * width,
        f"canonical={canonical_path} sha256={actual_sha}",
        f"dataset=WikiText-103 100% | tokenizer=16,384 BPE | train_seq={args.train_seq}",
        f"quality_budget/arm={args.quality_token_budget:,} tokens | exact paired examples",
        "",
        "QUALITY ARMS",
        f"{'variant':38s} {'batch':>5s} {'updates':>7s} {'schedule':>8s} {'gateLR':>7s} {'val NLL':>10s} {'test NLL':>10s} {'tok/s':>10s} {'peakGB':>8s}",
    ]
    for variant in VARIANTS:
        r = quality_results.get(variant.name)
        if r is None:
            continue
        lines.append(
            f"{r.variant:38s} {r.batch:5d} {r.updates:7d} {r.schedule:>8s} "
            f"{r.gate_lr_multiplier:7.2f} {r.final_validation['nll']:10.5f} "
            f"{r.final_test['nll']:10.5f} {r.tokens_per_second:10,.0f} {r.peak_gib:8.2f}"
        )
    lines += [
        "",
        "QUALITY DECISION",
        f"recommended={quality.get('recommended')} promote={quality.get('promote')}",
        f"reason={quality.get('reason')}",
        f"recommended Field minus Transformer validation NLL={quality.get('field_vs_transformer_validation_nll', float('nan')):+.5f}",
    ]
    if diagnostics:
        lines += ["", "V25 CHECKPOINT DIAGNOSTICS", f"status={diagnostics.get('status')}"]
        if diagnostics.get("status") == "ok":
            lines.append(f"baseline diagnostic NLL={diagnostics['baseline']['nll']:.5f}")
            for row in diagnostics.get("refresh_gates", []):
                lines.append(
                    f"refresh {row['index']} window={row['window']:4d} gate={row['gate_mean']:.3f} "
                    f"p10/p90={row['gate_p10']:.3f}/{row['gate_p90']:.3f} global_mix={row['global_mix']:.3f}"
                )
            p = diagnostics.get("pcaf", {})
            lines.append(
                f"PCAF coverage={p.get('candidate_coverage', 0):.3f} candidates={p.get('mean_candidates', 0):.2f} "
                f"gate={p.get('gate_mean', 0):.3f} cache_win={p.get('cache_win_rate', 0):.3f}"
            )
            for row in diagnostics.get("ablations", []):
                lines.append(f"{row['ablation']:34s} dNLL={row['delta_nll']:+.5f}")
    if system:
        lines += ["", "SYSTEM PARETO — FASTEST ROW UNDER VRAM LIMIT"]
        for row in system.get("best_rows", []):
            lines.append(
                f"{row['kind']:5s} {row['model']:28s} ctx={row['context']:5d} "
                f"batch={row['batch']:2d} policy={row['policy']:10s} "
                f"tok/s={row['tokens_per_second']:,.0f} peak={row['peak_gib']:.2f}G"
            )
    lines += [
        "",
        "AUTOMATIC NEXT STEP",
        ("Promote the recommended recipe to a 98.304M-token confirmation against Transformer and Mamba-2."
         if quality.get("promote") else
         "Keep the v25 training recipe; use diagnostics to choose one structural change before another long run."),
        "This program never auto-launches the long confirmation run.",
        "=" * width,
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable GPU required")
    if args.layers != 6 * len(args.refresh_windows):
        raise ValueError("layers must equal 6 * len(refresh_windows)")
    if args.train_seq != 2048:
        log(f"WARNING: v25 reference trained at context 2048, requested {args.train_seq}")
    if args.quality_token_budget % args.train_seq:
        raise ValueError("quality-token-budget must be divisible by train-seq")
    total_sequences = args.quality_token_budget // args.train_seq
    for variant in VARIANTS:
        if total_sequences % variant.batch:
            raise ValueError(f"token budget not divisible by batch for {variant.name}")

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.set_float32_matmul_precision("high")
    # v23 evaluation had a FastSuccessorCacheV5 token_nll compatibility gap;
    # use the exact v25 implementation that was validated against training loss.
    v23.token_nll = v25.token_nll_v25

    configure_builder_args(args)
    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)
    canonical_path, actual_sha, deps = load_dependencies(args)
    shapes = solve_shapes(args, deps)
    for name, shape in shapes.items():
        delta = 100.0 * (shape.params - args.target_params) / args.target_params
        log(f"[shape] {name:30s} params={shape.params:,} dTarget={delta:+.3f}% dim={shape.dim} layers={shape.layers} ff={shape.ff_hidden}")

    paired_audit = paired_initialization_audit(
        args, shapes[FUSION], deps, device, root,
    )
    initialization_audit = v23.initialization_audit(
        args, shapes, deps, device, root,
    )
    checkpoint_audit = v23.v22.checkpoint_exactness_audit(
        args, shapes[FUSION], deps, device, root,
    )
    evaluation_preflight = v25.evaluation_preflight_v25(
        args, shapes[FUSION], deps, device, root,
    )
    log(f"[selftest] paired={paired_audit}")
    log(f"[selftest] initialization={initialization_audit}")
    log(f"[selftest] checkpoint={checkpoint_audit}")
    log(f"[selftest] evaluation={evaluation_preflight}")

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

    if not args.diagnostic_checkpoint:
        default = Path("/home/ubuntu/pcaf_runs/field_fusion_fast_step6000_BF16.pt")
        args.diagnostic_checkpoint = str(default)
    diagnostics = run_diagnostics(
        args, shapes[FUSION], deps, val, root, device,
    ) if args.run_diagnostics else {"status": "disabled"}

    starts = make_example_starts(
        total_sequences, len(train) - args.train_seq - 1,
        args.data_seed, root / "paired_example_starts.npy",
    )
    quality_results: Dict[str, QualityResult] = {}
    if args.run_quality:
        for variant in VARIANTS:
            log("=" * 180)
            log(f"QUALITY ARM: {variant.name} — {variant.description}")
            shape = shapes[variant.model]
            quality_results[variant.name] = train_variant(
                variant, shape, args, deps, train, val_c, val,
                test_c, test, starts, root, device,
            )
            atomic_json(root / "quality_results.json", {
                k: asdict(v) for k, v in quality_results.items()
            })
        quality = quality_decision(quality_results, args)
    else:
        existing = root / "quality_results.json"
        if not existing.is_file():
            raise RuntimeError("--no-run-quality requires existing quality_results.json")
        quality_results = {k: QualityResult(**v) for k, v in json.loads(existing.read_text()).items()}
        quality = quality_decision(quality_results, args)
    atomic_json(root / "quality_decision.json", quality)
    log(f"[quality decision] {quality}")

    system_rows = []
    system_decision = {}
    if args.run_system:
        system_rows, system_decision = run_system_sweep(
            args, shapes, deps, train, test, train_c.bytes_per_token,
            test_c.bytes_per_token, root, device,
        )

    result = {
        "version": VERSION,
        "args": vars(args),
        "canonical_source": str(canonical_path),
        "canonical_sha256": actual_sha,
        "tokenizer": str(root / "tokenizer" / "tokenizer.json"),
        "tokenizer_sha256": sha256(root / "tokenizer" / "tokenizer.json"),
        "shapes": {k: asdict(v) for k, v in shapes.items()},
        "paired_initialization_audit": paired_audit,
        "initialization_audit": initialization_audit,
        "checkpoint_audit": checkpoint_audit,
        "evaluation_preflight": evaluation_preflight,
        "diagnostics": diagnostics,
        "quality_results": {k: asdict(v) for k, v in quality_results.items()},
        "quality_decision": quality,
        "system_rows": system_rows,
        "system_decision": system_decision,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    atomic_json(root / "results.json", result)
    summary = make_summary(
        args, canonical_path, actual_sha, shapes, corpora, diagnostics,
        quality_results, quality, system_decision,
    )
    atomic_text(root / "summary.txt", summary)
    log(summary)


if __name__ == "__main__":
    main()
