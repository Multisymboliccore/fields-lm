#!/usr/bin/env python3
"""FIELD-FUSION v27 — fair recipe control and memory closure.

This experiment has two deliberately separate goals:

1. Fair recipe control
   Compare the promoted Field recipe against Transformer and official Mamba-2
   after giving every architecture the same paired WikiText-103 examples, token
   budget, and access to cosine/WSD plus the same-update/more-update choices.

2. Memory closure
   Measure exact Field activation-recompute policies at 2K/8K/16K and test an
   exact time-chunked vocabulary readout.  The latter avoids keeping B*T*V
   logits resident during full-sequence NLL evaluation; it does not change the
   model, probabilities, parameters, or autoregressive generation semantics.

The program never auto-launches a 98.304M-token confirmation run.
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
import torch.nn.functional as F

import field_fusion_final_ablation_v26 as v26
import field_fusion_wiki100_canonical_v23 as v23
import field_fusion_wiki100_mamba2_v25 as v25

FUSION = v23.FUSION
TRANSFORMER = v23.TRANSFORMER
MAMBA2 = v25.MAMBA2
VERSION = 27
EXPECTED_CANONICAL_SHA256 = v23.EXPECTED_CANONICAL_SHA256
Variant = v26.Variant
QualityResult = v26.QualityResult

VARIANTS: Tuple[Variant, ...] = (
    Variant("field_updates_wsd_gate2x_b4", FUSION, 4, "wsd", 2.0,
            "Promoted v26 Field recipe: more updates, WSD, routing LR x2."),
    Variant("transformer_ref_cos_b8", TRANSFORMER, 8, "cosine", 1.0,
            "Transformer v25 reference recipe."),
    Variant("transformer_wsd_b8", TRANSFORMER, 8, "wsd", 1.0,
            "Transformer with WSD at canonical update count."),
    Variant("transformer_updates_wsd_b4", TRANSFORMER, 4, "wsd", 1.0,
            "Transformer with more updates plus WSD."),
    Variant("mamba2_ref_cos_b8", MAMBA2, 8, "cosine", 1.0,
            "Official Mamba-2 v25 reference recipe."),
    Variant("mamba2_wsd_b8", MAMBA2, 8, "wsd", 1.0,
            "Official Mamba-2 with WSD."),
    Variant("mamba2_updates_wsd_b4", MAMBA2, 4, "wsd", 1.0,
            "Official Mamba-2 with more updates plus WSD."),
)
FIELD_VARIANTS = tuple(v for v in VARIANTS if v.model == FUSION)
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


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_args() -> argparse.Namespace:
    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument("--quality-token-budget", type=int, default=25_165_824)
    custom.add_argument("--warmup-fraction", type=float, default=0.05)
    custom.add_argument("--wsd-stable-fraction", type=float, default=0.70)
    custom.add_argument("--eval-fractions", nargs="+", type=float,
                        default=[0.25, 0.50, 0.75, 1.00])
    custom.add_argument("--profile-log-every-tokens", type=int, default=1_048_576)
    custom.add_argument("--screen-validation-token-budget", type=int, default=0)
    custom.add_argument("--screen-test-token-budget", type=int, default=293_944)
    custom.add_argument("--readout-chunks", nargs="+", type=int,
                        default=[64, 128, 256, 512])
    custom.add_argument("--memory-contexts", nargs="+", type=int,
                        default=[2048, 8192, 16384])
    custom.add_argument("--memory-batches", nargs="+", type=int,
                        default=[8, 2, 1])
    custom.add_argument("--memory-warmup", type=int, default=1)
    custom.add_argument("--memory-steps", type=int, default=3)
    custom.add_argument("--stream-tolerance", type=float, default=3.0e-2)
    custom.add_argument("--run-quality", action=argparse.BooleanOptionalAction, default=True)
    custom.add_argument("--run-memory", action=argparse.BooleanOptionalAction, default=True)
    custom_args, remaining = custom.parse_known_args()
    old = sys.argv
    try:
        sys.argv = [old[0], *remaining]
        args = v23.parse_args()
    finally:
        sys.argv = old
    for key, value in vars(custom_args).items():
        setattr(args, key, value)
    v25.add_mamba_defaults(args)
    return args


def configure(args) -> None:
    v26.configure_builder_args(args)
    # Reuse the battle-tested v26 training machinery with v25's three-model
    # builder/loss/evaluation compatibility layer.
    v23.build_model = v25.build_model_v25
    v23.loss_call = v25.loss_call_v25
    v23.set_distill = v25.set_distill_v25
    v23.token_nll = v25.token_nll_v25
    v26.v23.build_model = v25.build_model_v25
    v26.v23.loss_call = v25.loss_call_v25
    v26.v23.set_distill = v25.set_distill_v25
    v26.v23.token_nll = v25.token_nll_v25
    v26.VERSION = VERSION
    v26.VARIANTS = VARIANTS
    v26.FIELD_VARIANTS = FIELD_VARIANTS


def load_dependencies(args):
    return v26.load_dependencies(args)


def solve_shapes(args, deps) -> Dict[str, v23.Shape]:
    shapes = v25.solve_shapes_v25(args, deps)
    expected = {FUSION, TRANSFORMER, MAMBA2}
    if set(shapes) != expected:
        raise AssertionError(f"shape solver returned {set(shapes)}, expected {expected}")
    return shapes


def best_by_model(results: Mapping[str, QualityResult], model: str) -> QualityResult:
    rows = [r for r in results.values() if r.model == model]
    if not rows:
        raise RuntimeError(f"no quality rows for {model}")
    return min(rows, key=lambda r: (r.final_validation["nll"], r.final_test["nll"]))


def quality_decision(results: Mapping[str, QualityResult]) -> Dict[str, object]:
    best_f = best_by_model(results, FUSION)
    best_t = best_by_model(results, TRANSFORMER)
    best_m = best_by_model(results, MAMBA2)
    refs = {
        TRANSFORMER: results["transformer_ref_cos_b8"],
        MAMBA2: results["mamba2_ref_cos_b8"],
    }
    recipe_gains = {
        "transformer": refs[TRANSFORMER].final_validation["nll"] - best_t.final_validation["nll"],
        "mamba2": refs[MAMBA2].final_validation["nll"] - best_m.final_validation["nll"],
    }
    field_wins = (
        best_f.final_validation["nll"] < best_t.final_validation["nll"]
        and best_f.final_validation["nll"] < best_m.final_validation["nll"]
    )
    out = {
        "best": {
            "field": asdict(best_f),
            "transformer": asdict(best_t),
            "mamba2": asdict(best_m),
        },
        "recipe_gain_nll": recipe_gains,
        "field_minus_best_transformer_val_nll": best_f.final_validation["nll"] - best_t.final_validation["nll"],
        "field_minus_best_mamba2_val_nll": best_f.final_validation["nll"] - best_m.final_validation["nll"],
        "field_minus_best_transformer_test_nll": best_f.final_test["nll"] - best_t.final_test["nll"],
        "field_minus_best_mamba2_test_nll": best_f.final_test["nll"] - best_m.final_test["nll"],
        "promote_long_confirmation": bool(field_wins),
        "recommended_field_recipe": best_f.variant,
        "recommended_transformer_recipe": best_t.variant,
        "recommended_mamba2_recipe": best_m.variant,
    }
    return out


def hidden_for_readout(name: str, model: nn.Module,
                       tokens: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Return cache states (Field only) and normalized hidden states.

    This deliberately skips lm_head, allowing vocabulary projection to be done
    in time chunks without changing the hidden computation.
    """
    if name == FUSION:
        x = model.emb(tokens)
        model._patch_aux = x.new_zeros(())
        for i, block in enumerate(model.blocks):
            x = block(x)
            if i == model.patch_position and model.softpatch is not None:
                x = model.softpatch(x, tokens)
                model._patch_aux = model.softpatch.last_aux
        return x, model.final_norm(x)
    if name == TRANSFORMER:
        x = model.emb(tokens)
        for block in model.blocks:
            x = block(x)
        return None, model.final_norm(x)
    if name == MAMBA2:
        x = model.emb(tokens).to(model.activation_dtype)
        for block in model.blocks:
            x = block(x)
        return None, model.norm(x)
    raise KeyError(name)


def generic_chunked_ce(model: nn.Module, hidden: torch.Tensor,
                       targets: torch.Tensor, chunk: int,
                       return_tokens: bool = False):
    pieces: List[torch.Tensor] = []
    total = hidden.new_zeros((), dtype=torch.float32)
    count = 0
    for start in range(0, hidden.shape[1], chunk):
        stop = min(hidden.shape[1], start + chunk)
        logits = model.lm_head(hidden[:, start:stop])
        nll = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            targets[:, start:stop].reshape(-1), reduction="none",
        ).view(targets.shape[0], stop - start)
        if return_tokens:
            pieces.append(nll)
        else:
            total = total + nll.sum()
            count += nll.numel()
        del logits
    if return_tokens:
        return torch.cat(pieces, dim=1)
    return total / max(count, 1)


def field_chunked_pcaf_nll(model: nn.Module, states: torch.Tensor,
                           hidden: torch.Tensor, tokens: torch.Tensor,
                           targets: torch.Tensor, chunk: int,
                           return_tokens: bool = False):
    """Exact FastSuccessorCacheV5 target NLL without persistent B*T*V logits."""
    cache = model.cache
    b, t, _ = states.shape
    module = sys.modules.get(cache.__class__.__module__)
    fast = getattr(module, "causal_recent_candidates_i32", None) if module else None
    if bool(getattr(cache, "use_i32", False)) and fast is not None:
        idx = fast(tokens, cache.order, cache.num_buckets, cache.top_k, cache._v3)
    else:
        idx = cache._v3.causal_recent_candidates(
            tokens, cache.order, cache.num_buckets, cache.top_k
        )
    valid = idx >= 0
    has = valid.any(-1)
    safe = idx.clamp_min(0)
    batch_idx = torch.arange(b, device=states.device)[:, None, None]
    proj = cache._v3.normalize_rows(
        F.linear(states.float(), cache.shared_weight.float())
    )
    q = proj[:, :, None, :]
    ck = proj[batch_idx, safe]
    scores = (ck * q).sum(-1) * (cache.memory_dim ** -0.5)
    recency = safe.float() / max(float(t - 1), 1.0)
    scores = scores + cache.recency_scale.float() * recency
    scores = scores.masked_fill(~valid, -1.0e9)
    weights = torch.softmax(scores.float(), dim=-1) * valid.float()
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
    cand_tokens = targets[batch_idx, safe]
    target_cache = (
        weights * (cand_tokens == targets[:, :, None]).float()
    ).sum(-1)
    state_logit = cache.state_gate(states).float().squeeze(-1)

    pieces: List[torch.Tensor] = []
    total = states.new_zeros((), dtype=torch.float32)
    count = 0
    for start in range(0, t, chunk):
        stop = min(t, start + chunk)
        logits = model.lm_head(hidden[:, start:stop])
        flat_logits = logits.reshape(-1, logits.shape[-1])
        target_slice = targets[:, start:stop]
        param_nll = F.cross_entropy(
            flat_logits.float(), target_slice.reshape(-1), reduction="none"
        )
        if not bool(getattr(cache, "enabled", True)):
            nll = param_nll.view(b, stop - start)
        else:
            active = has[:, start:stop].reshape(-1)
            gate_flat = torch.zeros_like(param_nll)
            if bool(active.any()):
                k = cache.top_k
                feat = cache._features(
                    scores[:, start:stop].reshape(-1, k)[active],
                    weights[:, start:stop].reshape(-1, k)[active],
                    valid[:, start:stop].reshape(-1, k)[active],
                    cand_tokens[:, start:stop].reshape(-1, k)[active],
                    recency[:, start:stop].reshape(-1, k)[active],
                    flat_logits[active],
                )
                route = cache.router(feat)
                flat_state = state_logit[:, start:stop].reshape(-1)
                if cache.router_mode == "v5":
                    gl = flat_state[active] + route[:, 0]
                else:
                    cache_conf = feat[:, 5].clamp(1e-4, 1.0 - 1e-4)
                    param_conf = feat[:, 8].clamp(1e-4, 1.0 - 1e-4)
                    evidence = torch.logit(cache_conf) - torch.logit(param_conf)
                    evidence = evidence + 1.25 * feat[:, 6] - 0.50 * feat[:, 7]
                    evidence = evidence + 0.35 * feat[:, 10] + 0.25 * feat[:, 3]
                    state_term = 0.0 if cache.router_mode == "confidence_nostate" else flat_state[active]
                    gl = state_term + route[:, 0] + cache.evidence_gain * evidence + cache.evidence_bias
                gate_flat[active] = torch.sigmoid(gl).clamp(1e-5, 1.0 - 1e-5)
            mixed = (
                (1.0 - gate_flat) * torch.exp(-param_nll)
                + gate_flat * target_cache[:, start:stop].reshape(-1)
            )
            nll = -torch.log(mixed.clamp_min(1e-8)).view(b, stop - start)
        if return_tokens:
            pieces.append(nll)
        else:
            total = total + nll.sum()
            count += nll.numel()
        del logits, flat_logits, param_nll
    if return_tokens:
        return torch.cat(pieces, dim=1)
    return total / max(count, 1)


def streaming_nll(name: str, model: nn.Module, x: torch.Tensor,
                  y: torch.Tensor, chunk: int, return_tokens: bool = False):
    states, hidden = hidden_for_readout(name, model, x)
    if name == FUSION:
        assert states is not None
        return field_chunked_pcaf_nll(
            model, states, hidden, x, y, chunk, return_tokens=return_tokens
        )
    return generic_chunked_ce(model, hidden, y, chunk, return_tokens=return_tokens)


@torch.inference_mode()
def streaming_exactness_audit(args, shapes, deps, device: torch.device,
                              root: Path) -> Dict[str, object]:
    rows: Dict[str, object] = {}
    length = max(65, int(args.selftest_tokens))
    x = ((torch.arange(length, device=device)[None] * 37 + 11) % args.vocab_size).long()
    y = ((x + 17) % args.vocab_size).long()
    for name in (FUSION, TRANSFORMER, MAMBA2):
        model = v25.build_model_v25(name, shapes[name], args, deps, device).eval()
        with v23.amp_ctx(device, args.amp):
            full = v25.token_nll_v25(name, model, x, y).float()
            streamed = streaming_nll(
                name, model, x, y, min(31, length), return_tokens=True
            ).float()
        max_abs = float((full - streamed).abs().max().cpu())
        mean_abs = float((full - streamed).abs().mean().cpu())
        passed = max_abs <= args.stream_tolerance
        rows[name] = {
            "max_abs": max_abs, "mean_abs": mean_abs,
            "tolerance": args.stream_tolerance, "pass": passed,
        }
        log(f"[selftest/stream] {name} max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} pass={passed}")
        if not passed:
            raise AssertionError(f"streaming readout mismatch for {name}: {max_abs}")
        del model, full, streamed
        clear_cuda()
    atomic_json(root / "streaming_exactness_audit.json", rows)
    return rows


@dataclass
class MemoryRow:
    model: str
    kind: str
    mode: str
    policy: str
    context: int
    batch: int
    readout_chunk: int
    status: str
    tokens_per_second: Optional[float]
    peak_gib: Optional[float]
    baseline_gib: Optional[float]
    activation_gib: Optional[float]
    error: str


@torch.inference_mode()
def benchmark_readout(name: str, shape: v23.Shape, args, deps,
                      data: torch.Tensor, context: int, batch: int,
                      chunk: int, mode: str, device: torch.device) -> MemoryRow:
    model = v25.build_model_v25(name, shape, args, deps, device).eval()
    x, y = v23.batch_for_step(
        data, batch, context, args.system_seed + context * 31 + chunk,
        1, 0, device,
    )

    def call_once():
        with v23.amp_ctx(device, args.amp):
            if mode == "full":
                loss = v25.token_nll_v25(name, model, x, y).mean()
            elif mode == "stream":
                loss = streaming_nll(name, model, x, y, chunk, False)
            else:
                raise ValueError(mode)
            _ = loss + 0.0

    status, error = "ok", ""
    tps = peak = baseline = activation = None
    try:
        for _ in range(args.memory_warmup):
            call_once()
        sync(device)
        baseline = torch.cuda.memory_allocated(device) / 2**30
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(args.memory_steps):
            call_once()
        sync(device)
        elapsed = time.perf_counter() - started
        tps = args.memory_steps * batch * context / max(elapsed, 1e-9)
        peak = torch.cuda.max_memory_allocated(device) / 2**30
        activation = max(0.0, peak - baseline)
    except torch.cuda.OutOfMemoryError as exc:
        status, error = "oom", str(exc)
    except Exception as exc:
        status, error = "error", repr(exc)
    row = MemoryRow(
        model=name, kind="infer_loss", mode=mode, policy="none",
        context=context, batch=batch, readout_chunk=(0 if mode == "full" else chunk),
        status=status, tokens_per_second=tps, peak_gib=peak,
        baseline_gib=baseline, activation_gib=activation, error=error,
    )
    del model, x, y
    clear_cuda()
    return row


def run_memory_suite(args, shapes, deps, train: torch.Tensor,
                     test: torch.Tensor, root: Path,
                     device: torch.device) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    if len(args.memory_contexts) != len(args.memory_batches):
        raise ValueError("memory-contexts and memory-batches must have equal length")
    rows: List[Dict[str, object]] = []
    old_policy = args.fusion_checkpoint_policy
    for context, batch in zip(args.memory_contexts, args.memory_batches):
        context, batch = int(context), int(batch)
        for policy in POLICIES:
            args.fusion_checkpoint_policy = policy
            log(f"[memory/train] Field ctx={context} batch={batch} policy={policy}")
            r = v23.benchmark_train(
                FUSION, shapes[FUSION], args, deps, train, context, batch,
                1.0, device,
            )
            rows.append(asdict(r))
            atomic_json(root / "memory_rows.json", rows)
        args.fusion_checkpoint_policy = "none"
        for name in (TRANSFORMER, MAMBA2):
            log(f"[memory/train] {name} ctx={context} batch={batch}")
            r = v23.benchmark_train(
                name, shapes[name], args, deps, train, context, batch,
                1.0, device,
            )
            rows.append(asdict(r))
            atomic_json(root / "memory_rows.json", rows)

        for name in (FUSION, TRANSFORMER, MAMBA2):
            log(f"[memory/infer-full] {name} ctx={context} batch={batch}")
            row = benchmark_readout(
                name, shapes[name], args, deps, test, context, batch,
                int(args.readout_chunks[0]), "full", device,
            )
            rows.append(asdict(row))
            atomic_json(root / "memory_rows.json", rows)
            for chunk in args.readout_chunks:
                log(f"[memory/infer-stream] {name} ctx={context} batch={batch} chunk={chunk}")
                row = benchmark_readout(
                    name, shapes[name], args, deps, test, context, batch,
                    int(chunk), "stream", device,
                )
                rows.append(asdict(row))
                atomic_json(root / "memory_rows.json", rows)
    args.fusion_checkpoint_policy = old_policy

    def valid(kind: str, model: str, context: int, mode: Optional[str] = None):
        out = [r for r in rows if r.get("kind") == kind and r.get("model") == model
               and int(r.get("context", -1)) == context and r.get("status") == "ok"]
        if mode is not None:
            out = [r for r in out if r.get("mode") == mode]
        return out

    decisions = {"contexts": {}}
    for context in map(int, args.memory_contexts):
        tf_train = valid("train", TRANSFORMER, context)
        field_train = valid("train", FUSION, context)
        tf_ref = max(tf_train, key=lambda r: r.get("tokens_per_second") or 0.0) if tf_train else None
        balanced_candidates = [r for r in field_train if r.get("peak_gib") is not None
                               and tf_ref is not None and r["peak_gib"] <= tf_ref["peak_gib"]]
        balanced = max(balanced_candidates, key=lambda r: r.get("tokens_per_second") or 0.0) if balanced_candidates else None
        stream_best = {}
        for name in (FUSION, TRANSFORMER, MAMBA2):
            cand = valid("infer_loss", name, context, "stream")
            stream_best[name] = max(cand, key=lambda r: r.get("tokens_per_second") or 0.0) if cand else None
        decisions["contexts"][str(context)] = {
            "transformer_train_reference": tf_ref,
            "field_memory_balanced": balanced,
            "stream_best": stream_best,
        }
    longest = str(max(map(int, args.memory_contexts)))
    long_row = decisions["contexts"][longest]
    ft = long_row.get("field_memory_balanced")
    tt = long_row.get("transformer_train_reference")
    fs = long_row.get("stream_best", {}).get(FUSION)
    ts = long_row.get("stream_best", {}).get(TRANSFORMER)
    decisions["long_context_training_memory_closed"] = bool(
        ft and tt and ft["peak_gib"] <= tt["peak_gib"]
    )
    decisions["long_context_streaming_infer_memory_closed"] = bool(
        fs and ts and fs["peak_gib"] <= ts["peak_gib"] * 1.05
    )
    atomic_json(root / "memory_decision.json", decisions)
    return rows, decisions


def summary(args, canonical_path: Path, actual_sha: str, shapes,
            quality_results, quality, memory, stream_audit) -> str:
    width = 200
    lines = [
        "=" * width,
        "FIELD-FUSION v27 — FAIR RECIPE CONTROL + MEMORY CLOSURE",
        "=" * width,
        f"canonical={canonical_path} sha256={actual_sha}",
        f"WikiText-103 100% source | paired budget/arm={args.quality_token_budget:,} | context={args.train_seq}",
        "",
        "FAIR QUALITY CONTROL",
        f"{'variant':38s} {'model':28s} {'batch':>5s} {'updates':>7s} {'schedule':>8s} {'val NLL':>10s} {'test NLL':>10s} {'tok/s':>10s} {'peakGB':>8s}",
    ]
    for variant in VARIANTS:
        r = quality_results.get(variant.name)
        if r:
            lines.append(
                f"{r.variant:38s} {r.model:28s} {r.batch:5d} {r.updates:7d} "
                f"{r.schedule:>8s} {r.final_validation['nll']:10.5f} "
                f"{r.final_test['nll']:10.5f} {r.tokens_per_second:10,.0f} {r.peak_gib:8.2f}"
            )
    lines += [
        "",
        "FAIRNESS DECISION",
        f"best Field={quality.get('recommended_field_recipe')}",
        f"best Transformer={quality.get('recommended_transformer_recipe')}",
        f"best Mamba-2={quality.get('recommended_mamba2_recipe')}",
        f"Field minus best Transformer val NLL={quality.get('field_minus_best_transformer_val_nll', float('nan')):+.5f}",
        f"Field minus best Mamba-2 val NLL={quality.get('field_minus_best_mamba2_val_nll', float('nan')):+.5f}",
        f"promote_long_confirmation={quality.get('promote_long_confirmation')}",
        "",
        "STREAMING READOUT EXACTNESS",
    ]
    for name, row in stream_audit.items():
        lines.append(
            f"{name:28s} max_abs={row['max_abs']:.3e} mean_abs={row['mean_abs']:.3e} pass={row['pass']}"
        )
    lines += ["", "MEMORY CLOSURE — RECOMMENDED ROWS"]
    for context, row in memory.get("contexts", {}).items():
        ft = row.get("field_memory_balanced")
        tt = row.get("transformer_train_reference")
        if ft and tt:
            lines.append(
                f"train ctx={int(context):5d}: Field policy={ft['policy']:10s} "
                f"tok/s={ft['tokens_per_second']:,.0f} peak={ft['peak_gib']:.2f}G | "
                f"Transformer tok/s={tt['tokens_per_second']:,.0f} peak={tt['peak_gib']:.2f}G"
            )
        sb = row.get("stream_best", {})
        fs, ts, ms = sb.get(FUSION), sb.get(TRANSFORMER), sb.get(MAMBA2)
        if fs and ts:
            text = (
                f"stream ctx={int(context):5d}: Field chunk={fs['readout_chunk']} "
                f"tok/s={fs['tokens_per_second']:,.0f} peak={fs['peak_gib']:.2f}G | "
                f"Transformer chunk={ts['readout_chunk']} tok/s={ts['tokens_per_second']:,.0f} peak={ts['peak_gib']:.2f}G"
            )
            if ms:
                text += f" | Mamba chunk={ms['readout_chunk']} tok/s={ms['tokens_per_second']:,.0f} peak={ms['peak_gib']:.2f}G"
            lines.append(text)
    lines += [
        "",
        f"long_context_training_memory_closed={memory.get('long_context_training_memory_closed')}",
        f"long_context_streaming_infer_memory_closed={memory.get('long_context_streaming_infer_memory_closed')}",
        "",
        "AUTOMATIC NEXT STEP",
    ]
    if quality.get("promote_long_confirmation"):
        lines.append("Run one 98.304M-token confirmation using each architecture's best fair recipe.")
    else:
        lines.append("Do not start the long confirmation; inspect which baseline recovered the quality lead.")
    lines.append("No architecture mutation is promoted by this test.")
    lines.append("=" * width)
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required")
    if args.quality_token_budget % args.train_seq:
        raise ValueError("quality-token-budget must be divisible by train-seq")
    total_sequences = args.quality_token_budget // args.train_seq
    for variant in VARIANTS:
        if total_sequences % variant.batch:
            raise ValueError(f"token budget not divisible by batch for {variant.name}")
    configure(args)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.set_float32_matmul_precision("high")

    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)
    canonical_path, actual_sha, deps = load_dependencies(args)
    shapes = solve_shapes(args, deps)
    for name, shape in shapes.items():
        delta = 100.0 * (shape.params - args.target_params) / args.target_params
        log(f"[shape] {name:30s} params={shape.params:,} dTarget={delta:+.3f}% dim={shape.dim} layers={shape.layers} ff={shape.ff_hidden}")

    paired = v26.paired_initialization_audit(args, shapes[FUSION], deps, device, root)
    init = v25.initialization_audit_v25(args, shapes, deps, device, root)
    ckpt = v23.v22.checkpoint_exactness_audit(args, shapes[FUSION], deps, device, root)
    eval_pre = v25.evaluation_preflight_v25(args, shapes[FUSION], deps, device, root)
    mamba_pre = v25.mamba_strict_preflight(args, shapes[MAMBA2], deps, device, root)
    stream_audit = streaming_exactness_audit(args, shapes, deps, device, root)
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

    starts = v26.make_example_starts(
        total_sequences, len(train) - args.train_seq - 1,
        args.data_seed, root / "paired_example_starts.npy",
    )
    quality_results: Dict[str, QualityResult] = {}
    if args.run_quality:
        for variant in VARIANTS:
            log("=" * 190)
            log(f"QUALITY ARM: {variant.name} — {variant.description}")
            quality_results[variant.name] = v26.train_variant(
                variant, shapes[variant.model], args, deps, train,
                val_c, val, test_c, test, starts, root, device,
            )
            atomic_json(root / "quality_results.json", {
                k: asdict(v) for k, v in quality_results.items()
            })
    else:
        raw = json.loads((root / "quality_results.json").read_text(encoding="utf-8"))
        quality_results = {k: QualityResult(**v) for k, v in raw.items()}
    quality = quality_decision(quality_results)
    atomic_json(root / "quality_decision.json", quality)
    log(f"[quality decision] {quality}")

    memory_rows: List[Dict[str, object]] = []
    memory_decision: Dict[str, object] = {}
    if args.run_memory:
        memory_rows, memory_decision = run_memory_suite(
            args, shapes, deps, train, test, root, device,
        )

    result = {
        "version": VERSION,
        "args": vars(args),
        "canonical_source": str(canonical_path),
        "canonical_sha256": actual_sha,
        "shapes": {k: asdict(v) for k, v in shapes.items()},
        "quality_results": {k: asdict(v) for k, v in quality_results.items()},
        "quality_decision": quality,
        "streaming_exactness_audit": stream_audit,
        "memory_rows": memory_rows,
        "memory_decision": memory_decision,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "mamba_ssm": v25.MAMBA_VERSION,
        "causal_conv1d": v25.CAUSAL_CONV1D_VERSION,
    }
    atomic_json(root / "results.json", result)
    text = summary(
        args, canonical_path, actual_sha, shapes, quality_results,
        quality, memory_decision, stream_audit,
    )
    atomic_text(root / "summary.txt", text)
    log(text)


if __name__ == "__main__":
    main()
