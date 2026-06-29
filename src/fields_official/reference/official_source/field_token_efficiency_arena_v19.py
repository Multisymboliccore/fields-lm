#!/usr/bin/env python3
"""FIELD TOKEN EFFICIENCY ARENA v19.

A paired, two-lane token-level screen designed to answer one question:
which validated Field mechanisms are actually efficient after the BPE transfer?

The arena compares seven ~50M parameter systems on one shared byte-level BPE:

Hybrid lane (local query-key path allowed)
  * hybrid_selective_token          control
  * hybrid_span2_token              selective residual + verified span2
  * hybrid_surface_span2_token      surface multiview + verified span2

Strict attention-free lane (no query-key path)
  * attentionfree_selective_token   control
  * attentionfree_span2_token       selective residual + verified span2
  * attentionfree_surface_span2_token

External reference
  * transformer_flash_token

Fairness / reliability additions over v18
-----------------------------------------
* Multiple paired seeds; every model in a replicate sees the same token windows.
* Tied embedding/head initialization uses a dedicated CPU generator, independent
  of sidecar construction order. This fixes the v18 pairing ambiguity.
* Initialization hashes verify identical embeddings across comparable models and
  identical backbones within each lane.
* Quality is aggregated in NLL space; PPL is exp(mean NLL), not mean(PPL).
* Equal-token no-checkpoint training benchmarks and full-path inference
  benchmarks measure throughput and peak allocated VRAM across context lengths.
* Automatic lane deltas, gate decisions and a three-objective Pareto frontier
  (lower NLL, higher throughput, lower VRAM).

The validated Field recurrence and memory dependencies are packed beside this
file. The canonical Triton Field source remains external and SHA-verified.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import hashlib
import importlib.util
import json
import math
import os
import shutil
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
CORE_PATH = HERE / "field_token_ppl_pilot_v18.py"
V15_PATH = HERE / "field_scale_50m_v15.py"
CANONICAL_NAME = "field_only_v4_chunked_triton_wiki100.py"
EXPECTED_CANONICAL_SHA256 = "0454647a7f667cdf73f45029eee0f525c58c0788ecd433ec1a4db8c3937ac848"

HYBRID_CONTROL = "hybrid_selective_token"
HYBRID_PARETO = "hybrid_span2_token"
HYBRID_QUALITY = "hybrid_surface_span2_token"
AF_CONTROL = "attentionfree_selective_token"
AF_PARETO = "attentionfree_span2_token"
AF_QUALITY = "attentionfree_surface_span2_token"
TRANSFORMER = "transformer_flash_token"

HYBRID_NAMES = (HYBRID_CONTROL, HYBRID_PARETO, HYBRID_QUALITY)
AF_NAMES = (AF_CONTROL, AF_PARETO, AF_QUALITY)
FIELD_NAMES = (*HYBRID_NAMES, *AF_NAMES)
MODEL_NAMES = (*FIELD_NAMES, TRANSFORMER)
LANE_REFERENCES = {
    HYBRID_CONTROL: HYBRID_CONTROL,
    HYBRID_PARETO: HYBRID_CONTROL,
    HYBRID_QUALITY: HYBRID_CONTROL,
    AF_CONTROL: AF_CONTROL,
    AF_PARETO: AF_CONTROL,
    AF_QUALITY: AF_CONTROL,
}


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


core = import_module(CORE_PATH, "field_token_ppl_pilot_v18_core_for_v19")
Shape = core.Shape
Corpus = core.Corpus
LN2 = core.LN2


def log(msg: object = "") -> None:
    print(str(msg), flush=True)


def atomic_json(path: Path, obj: object) -> None:
    core.atomic_json(path, obj)


def atomic_text(path: Path, text: str) -> None:
    core.atomic_text(path, text)


def sha256(path: Path) -> str:
    return core.sha256(path)


def model_lane(name: str) -> str:
    if name in HYBRID_NAMES:
        return "hybrid"
    if name in AF_NAMES:
        return "attentionfree"
    if name == TRANSFORMER:
        return "transformer"
    raise KeyError(name)


def model_profile(name: str) -> str:
    if name in (HYBRID_CONTROL, AF_CONTROL):
        return "selective"
    if name in (HYBRID_PARETO, AF_PARETO):
        return "span"
    if name in (HYBRID_QUALITY, AF_QUALITY):
        return "surface_span"
    if name == TRANSFORMER:
        return "transformer"
    raise KeyError(name)


def is_attention_free(name: str) -> bool:
    return name in AF_NAMES


def locate_canonical(explicit: str) -> Path:
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend([
        HERE / CANONICAL_NAME,
        Path("/home/ubuntu") / CANONICAL_NAME,
        Path("/home/ubuntu/field_memory_consolidation_canonical_50m_v13") / CANONICAL_NAME,
        Path("/home/ubuntu/field_hybrid_canonical_50m_bridge_v4") / CANONICAL_NAME,
        Path("/home/ubuntu/field_hybrid_300m_h2h_v7") / CANONICAL_NAME,
        Path("/home/ubuntu/field_pcaf_efficiency_v1") / CANONICAL_NAME,
    ])
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError("validated canonical Field source not found")


def copy_or_train_tokenizer(root: Path, train_rows: Sequence[str], vocab_size: int,
                            min_frequency: int, source: str):
    """Reuse the v18 tokenizer when available; otherwise train exactly once."""
    target = root / "tokenizer" / "tokenizer.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file() and source:
        src = Path(source).expanduser()
        if not src.is_file():
            raise FileNotFoundError(f"tokenizer source not found: {src}")
        shutil.copy2(src, target)
        log(f"[tokenizer] copied shared tokenizer from {src}")
    return core.build_or_load_tokenizer(root, train_rows, vocab_size, min_frequency)


def tied_embedding_init(model: nn.Module, seed: int, std: float = 0.02) -> None:
    """Initialize tied embeddings independently of model construction RNG.

    The matrix is generated on CPU so every architecture receives byte-identical
    values when shape and seed match, then copied to the target device.
    """
    emb = getattr(model, "emb", None)
    if emb is None:
        raise AttributeError("model has no emb")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    cpu_weight = torch.empty(tuple(emb.weight.shape), dtype=torch.float32, device="cpu")
    cpu_weight.normal_(mean=0.0, std=std, generator=generator)
    with torch.no_grad():
        emb.weight.copy_(cpu_weight.to(device=emb.weight.device, dtype=emb.weight.dtype))
    core.tie_embeddings(model, initialize=False)


def tensor_hash(tensor: torch.Tensor) -> str:
    h = hashlib.sha256()
    x = tensor.detach().contiguous().cpu()
    h.update(str(tuple(x.shape)).encode())
    h.update(str(x.dtype).encode())
    h.update(x.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def state_hash(model: nn.Module, prefixes: Sequence[str]) -> str:
    h = hashlib.sha256()
    state = model.state_dict()
    selected = [k for k in sorted(state) if any(k.startswith(p) for p in prefixes)]
    if not selected:
        raise RuntimeError(f"no state keys matched prefixes={prefixes}")
    for key in selected:
        x = state[key].detach().contiguous().cpu()
        h.update(key.encode())
        h.update(str(tuple(x.shape)).encode())
        h.update(str(x.dtype).encode())
        h.update(x.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def make_runtime_args(args, field_hidden: int, tf_hidden: int) -> argparse.Namespace:
    ns = argparse.Namespace(**vars(args))
    ns.hybrid_ff_hidden = int(field_hidden)
    ns.field_ff_hidden = int(field_hidden)
    ns.af_ff_hidden = int(field_hidden)
    ns.tf_ff_hidden = int(tf_hidden)
    return ns


def build_field(name: str, args, arena, v3, canonical, bridge, optmod, epi,
                device: torch.device, hidden: int):
    del epi
    arena.base.seed_all(args.model_seed)
    bargs = arena.base.make_bridge_args(args, hidden)
    arm = "attentionfree_multiscale" if is_attention_free(name) else "hybrid_w256_conf_parity"
    model = bridge.build_field(arm, bargs, v3, canonical, hidden).to(device)

    # Validated v6 optimized systems path.
    optmod.replace_softpatch(model, v3)
    optmod.replace_cache(model, v3, i32=True)
    model.cache.FEATURE_DIM = int(v3.SuccessorCacheV5.FEATURE_DIM)
    if is_attention_free(name):
        optmod.replace_multiscale(model, v3, lite=False)
    else:
        optmod.replace_local(model, v3, "cached", args.local_chunk)

    arena._install_cloud_fast_route(optmod)
    raw_cache = model.cache
    profile = model_profile(name)
    if profile == "selective":
        wrapped = arena.cloud.make_v10_cache(raw_cache, args)
    elif profile == "span":
        selective = arena.cloud.make_v10_cache(raw_cache, args)
        wrapped = arena.cloud.CloudMechanismCache(
            selective,
            delta_rank=None,
            span_max=int(args.span_tokens),
            phase=False,
            args=args,
        )
    elif profile == "surface_span":
        surface = arena.cloud.make_surface_multiview_cache(raw_cache, args)
        wrapped = arena.cloud.CloudMechanismCache(
            surface,
            delta_rank=None,
            span_max=int(args.span_tokens),
            phase=False,
            args=args,
        )
    else:
        raise KeyError(profile)
    model.cache = wrapped.to(device)
    return model


def build_model(name: str, shape: Shape, args, arena, v3, canonical, bridge,
                optmod, epi, judge, device: torch.device):
    field_hidden = shape.ff_hidden if name != TRANSFORMER else args.field_ff_hidden
    tf_hidden = shape.ff_hidden if name == TRANSFORMER else args.tf_ff_hidden
    run_args = make_runtime_args(args, field_hidden, tf_hidden)
    core.seed_all(args.model_seed)
    if name == TRANSFORMER:
        run_args.tf_dim = shape.dim
        run_args.tf_heads = shape.heads
        run_args.tf_layers = shape.layers
        run_args.tf_ff_hidden = shape.ff_hidden
        model = arena.base.build_transformer(run_args, judge, v3, device)
    elif name in FIELD_NAMES:
        model = build_field(name, run_args, arena, v3, canonical, bridge, optmod, epi,
                            device, shape.ff_hidden)
    else:
        raise KeyError(name)
    tied_embedding_init(model, seed=args.embedding_seed)
    return model


# Reuse the thoroughly tested v18 train/eval/system routines with v19 builders.
core.TRANSFORMER = TRANSFORMER
core.MODEL_NAMES = MODEL_NAMES
core.build_model = build_model
_core_checkpoint_signature = core.checkpoint_signature

def checkpoint_signature_v19(args, name: str, shape: Shape) -> Dict[str, object]:
    signature = _core_checkpoint_signature(args, name, shape)
    signature.update({
        "arena_version": "v19",
        "embedding_seed": int(args.embedding_seed),
        "lane": model_lane(name),
        "profile": model_profile(name),
    })
    return signature

core.checkpoint_signature = checkpoint_signature_v19


def count_model(name: str, hidden: int, args, arena, v3, canonical, bridge,
                optmod, epi, judge) -> int:
    if name == TRANSFORMER:
        shape = Shape(name, 0, args.tf_dim, args.tf_layers, args.tf_heads, hidden)
    else:
        shape = Shape(name, 0, args.field_dim, args.field_layers, args.field_heads, hidden)
    model = build_model(name, shape, args, arena, v3, canonical, bridge, optmod, epi,
                        judge, torch.device("cpu"))
    value = core.nparams(model)
    del model
    gc.collect()
    return value


def solve_hidden(name: str, target: int, args, arena, v3, canonical, bridge,
                 optmod, epi, judge) -> Tuple[int, int]:
    lo, hi = 512, 4096
    p0 = count_model(name, lo, args, arena, v3, canonical, bridge, optmod, epi, judge)
    p1 = count_model(name, lo + 16, args, arena, v3, canonical, bridge, optmod, epi, judge)
    slope = (p1 - p0) / 16.0
    if slope <= 0:
        raise RuntimeError(f"non-positive parameter slope for {name}")
    guess = int(round((lo + (target - p0) / slope) / 16.0) * 16)
    guess = max(lo, min(hi, guess))
    candidates = sorted(set(max(lo, min(hi, guess + 16 * d)) for d in range(-5, 6)))
    rows = [
        (hidden, count_model(name, hidden, args, arena, v3, canonical, bridge,
                             optmod, epi, judge))
        for hidden in candidates
    ]
    return min(rows, key=lambda hp: abs(hp[1] - target))


def resolve_shapes(args, arena, v3, canonical, bridge, optmod, epi, judge,
                   selected: Sequence[str]) -> Dict[str, Shape]:
    shapes: Dict[str, Shape] = {}
    for lane_names in (HYBRID_NAMES, AF_NAMES):
        active = [name for name in lane_names if name in selected]
        if not active:
            continue
        reference = lane_names[0]
        ref_hidden, ref_params = solve_hidden(reference, args.target_params, args, arena,
                                              v3, canonical, bridge, optmod, epi, judge)
        if reference in selected:
            shapes[reference] = Shape(reference, ref_params, args.field_dim,
                                      args.field_layers, args.field_heads, ref_hidden)
        for name in lane_names[1:]:
            if name not in selected:
                continue
            same_params = count_model(name, ref_hidden, args, arena, v3, canonical,
                                      bridge, optmod, epi, judge)
            delta_pct = abs(same_params - args.target_params) / args.target_params * 100.0
            if delta_pct <= args.max_param_delta_pct:
                hidden, params = ref_hidden, same_params
            else:
                hidden, params = solve_hidden(name, args.target_params, args, arena, v3,
                                              canonical, bridge, optmod, epi, judge)
            shapes[name] = Shape(name, params, args.field_dim, args.field_layers,
                                 args.field_heads, hidden)
    if TRANSFORMER in selected:
        hidden, params = solve_hidden(TRANSFORMER, args.target_params, args, arena, v3,
                                      canonical, bridge, optmod, epi, judge)
        shapes[TRANSFORMER] = Shape(TRANSFORMER, params, args.tf_dim, args.tf_layers,
                                    args.tf_heads, hidden)
    for shape in shapes.values():
        delta = 100.0 * (shape.params - args.target_params) / args.target_params
        if abs(delta) > args.max_param_delta_pct:
            raise RuntimeError(f"parameter mismatch {shape.name}: {delta:+.3f}%")
    return shapes


def replicate_args(args, replicate_index: int, model_seed: int) -> argparse.Namespace:
    out = argparse.Namespace(**vars(args))
    out.model_seed = int(model_seed)
    out.embedding_seed = int(args.embedding_seed + replicate_index * 10_007)
    out.data_seed = int(args.data_seed + replicate_index * 1_000_003)
    return out


def initialization_audit(args, shapes, arena, v3, canonical, bridge, optmod, epi,
                         judge, device: torch.device, selected: Sequence[str], root: Path):
    log("[audit] verifying deterministic tied embeddings and paired lane backbones")
    rows: Dict[str, Dict[str, object]] = {}
    for name in selected:
        model = build_model(name, shapes[name], args, arena, v3, canonical, bridge,
                            optmod, epi, judge, device).eval()
        rows[name] = {
            "embedding_hash": tensor_hash(model.emb.weight),
            "embedding_shape": list(model.emb.weight.shape),
            "embedding_std": float(model.emb.weight.detach().float().std().cpu()),
            "backbone_hash": None,
        }
        if name in FIELD_NAMES:
            rows[name]["backbone_hash"] = state_hash(
                model, ("emb.", "blocks.", "final_norm.", "locals.", "multiscales.", "softpatch.")
            )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    by_shape: Dict[Tuple[int, ...], List[str]] = {}
    for name, row in rows.items():
        by_shape.setdefault(tuple(row["embedding_shape"]), []).append(name)
    for shape, names in by_shape.items():
        hashes = {rows[name]["embedding_hash"] for name in names}
        log(f"[audit] embedding shape={shape} models={len(names)} unique_hashes={len(hashes)}")
        if len(hashes) != 1:
            raise AssertionError(f"embedding initialization mismatch for shape={shape}: {names}")

    for lane in (HYBRID_NAMES, AF_NAMES):
        names = [name for name in lane if name in selected]
        if len(names) < 2:
            continue
        hashes = {rows[name]["backbone_hash"] for name in names}
        log(f"[audit] lane={model_lane(names[0])} models={names} unique_backbone_hashes={len(hashes)}")
        if len(hashes) != 1:
            raise AssertionError(f"paired backbone mismatch in lane {names}")

    atomic_json(root / "initialization_audit.json", rows)
    log("[audit] PASS")
    return rows


def run_selftest(args, shapes, arena, v3, canonical, bridge, optmod, epi, judge,
                 device: torch.device, selected: Sequence[str], root: Path):
    # The v18 selftest already validates token-safe softpatch, finite backward,
    # tied heads, plausible initial NLL and causal token NLL.
    old_names = core.MODEL_NAMES
    core.MODEL_NAMES = tuple(selected)
    try:
        core.run_selftest(args, shapes, arena, v3, canonical, bridge, optmod, epi,
                          judge, device)
    finally:
        core.MODEL_NAMES = old_names
    return initialization_audit(args, shapes, arena, v3, canonical, bridge, optmod,
                                epi, judge, device, selected, root)


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    vals = [float(x) for x in values]
    if not vals:
        return float("nan"), float("nan")
    return statistics.fmean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0


def aggregate_results(per_seed: Mapping[int, Mapping[str, Mapping[str, object]]],
                      selected: Sequence[str], contexts: Sequence[int]) -> Dict[str, object]:
    aggregate: Dict[str, object] = {"seeds": sorted(per_seed), "models": {}}
    for name in selected:
        rows = [per_seed[seed][name] for seed in sorted(per_seed)]
        nll_mean, nll_std = mean_std([row["test"]["nll"] for row in rows])
        bpb_mean, bpb_std = mean_std([row["test"]["bpb_norm"] for row in rows])
        tps_mean, tps_std = mean_std([row["train_tokens_per_second"] for row in rows])
        peak_mean, peak_std = mean_std([row["train_peak_gib"] for row in rows])
        model_row: Dict[str, object] = {
            "nll_mean": nll_mean,
            "nll_std": nll_std,
            "ppl_from_mean_nll": math.exp(min(nll_mean, 20.0)),
            "bits_per_token_mean": nll_mean / LN2,
            "bpb_norm_mean": bpb_mean,
            "bpb_norm_std": bpb_std,
            "train_tokens_per_second_mean": tps_mean,
            "train_tokens_per_second_std": tps_std,
            "train_peak_gib_mean": peak_mean,
            "train_peak_gib_std": peak_std,
            "matched_suffix": {},
            "individual": rows,
        }
        for context in contexts:
            vals = []
            for row in rows:
                by_ctx = {int(item["context"]): item for item in row["matched_suffix"]}
                vals.append(float(by_ctx[int(context)]["nll"]))
            m, s = mean_std(vals)
            model_row["matched_suffix"][str(context)] = {
                "nll_mean": m,
                "nll_std": s,
                "ppl_from_mean_nll": math.exp(min(m, 20.0)),
            }
        aggregate["models"][name] = model_row

    comparisons = []
    for candidate in (*HYBRID_NAMES[1:], *AF_NAMES[1:]):
        if candidate not in selected:
            continue
        reference = LANE_REFERENCES[candidate]
        if reference not in selected:
            continue
        deltas = []
        wins = 0
        for seed in sorted(per_seed):
            cand = per_seed[seed][candidate]
            ref = per_seed[seed][reference]
            dnll = float(cand["test"]["nll"]) - float(ref["test"]["nll"])
            if dnll < 0:
                wins += 1
            deltas.append({
                "seed": seed,
                "delta_nll": dnll,
                "delta_bpb_norm": float(cand["test"]["bpb_norm"]) - float(ref["test"]["bpb_norm"]),
                "ppl_relative_pct": (math.exp(dnll) - 1.0) * 100.0,
                "train_speed_ratio": float(cand["train_tokens_per_second"]) / float(ref["train_tokens_per_second"]),
                "train_peak_ratio": float(cand["train_peak_gib"]) / float(ref["train_peak_gib"]),
            })
        comparisons.append({
            "candidate": candidate,
            "reference": reference,
            "wins": wins,
            "seeds": len(deltas),
            "delta_nll_mean": statistics.fmean(x["delta_nll"] for x in deltas),
            "delta_bpb_norm_mean": statistics.fmean(x["delta_bpb_norm"] for x in deltas),
            "ppl_relative_pct_from_mean_delta_nll": (math.exp(statistics.fmean(x["delta_nll"] for x in deltas)) - 1.0) * 100.0,
            "train_speed_ratio_mean": statistics.fmean(x["train_speed_ratio"] for x in deltas),
            "train_peak_ratio_mean": statistics.fmean(x["train_peak_ratio"] for x in deltas),
            "individual": deltas,
        })
    aggregate["paired_comparisons"] = comparisons
    return aggregate



def training_systems_benchmark(name: str, shape: Shape, args, arena, v3, canonical,
                               bridge, optmod, epi, judge, context: int,
                               bytes_per_token: float, data: torch.Tensor,
                               device: torch.device) -> Dict[str, object]:
    """Forward + backward + AdamW on a fixed real-corpus batch."""
    model = build_model(name, shape, args, arena, v3, canonical, bridge, optmod,
                        epi, judge, device).train()
    core.set_distill(name, model, args.conf_distill_ramp, args)
    lr = args.transformer_lr if name == TRANSFORMER else args.field_lr
    optimizer = core.make_optimizer(model, lr, args.weight_decay)
    batch = max(1, args.system_tokens_per_step // context)
    x, y = core.batch_for_step(
        data, batch, context, args.eval_seed + context * 97, 1, 0, device
    )

    def step_once():
        optimizer.zero_grad(set_to_none=True)
        with core.amp_ctx(device, args.amp):
            loss, _ = core.training_loss(name, model, x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

    status, error = "ok", ""
    tps = step_ms = peak = None
    try:
        for _ in range(args.system_warmup):
            step_once()
        core.sync(device)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(args.system_steps):
            step_once()
        core.sync(device)
        elapsed = time.perf_counter() - started
        tps = args.system_steps * batch * context / max(elapsed, 1e-9)
        step_ms = elapsed * 1000.0 / args.system_steps
        peak = torch.cuda.max_memory_allocated() / 2**30
    except torch.cuda.OutOfMemoryError as exc:
        status, error = "oom", str(exc)
    except Exception as exc:
        status, error = "error", repr(exc)
    row = {
        "model": name,
        "context": int(context),
        "batch": int(batch),
        "status": status,
        "tokens_per_second": tps,
        "bytes_per_second_est": None if tps is None else tps * bytes_per_token,
        "step_ms": step_ms,
        "peak_gib": peak,
        "input": "fixed_real_corpus_batch",
        "error": error,
    }
    del model, optimizer, x, y
    gc.collect()
    torch.cuda.empty_cache()
    return row

def inference_benchmark(name: str, shape: Shape, args, arena, v3, canonical, bridge,
                        optmod, epi, judge, context: int, data: torch.Tensor,
                        device: torch.device) -> Dict[str, object]:
    model = build_model(name, shape, args, arena, v3, canonical, bridge, optmod, epi,
                        judge, device).eval()
    core.set_distill(name, model, args.conf_distill_ramp, args)
    batch = max(1, args.inference_tokens_per_call // context)
    x, y = core.batch_for_step(
        data, batch, context, args.eval_seed + context * 193, 1, 0, device
    )

    def call_once():
        with torch.inference_mode(), core.amp_ctx(device, args.amp):
            nll = core.token_nll(name, model, x, y)
            return nll.mean()

    status, error = "ok", ""
    tps = latency_ms = peak = None
    try:
        for _ in range(args.inference_warmup):
            call_once()
        core.sync(device)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(args.inference_steps):
            call_once()
        core.sync(device)
        elapsed = time.perf_counter() - started
        tokens = args.inference_steps * batch * context
        tps = tokens / max(elapsed, 1e-9)
        latency_ms = elapsed * 1000.0 / args.inference_steps
        peak = torch.cuda.max_memory_allocated() / 2**30
    except torch.cuda.OutOfMemoryError as exc:
        status, error = "oom", str(exc)
    except Exception as exc:
        status, error = "error", repr(exc)
    row = {
        "model": name,
        "context": int(context),
        "batch": int(batch),
        "status": status,
        "tokens_per_second": tps,
        "latency_ms": latency_ms,
        "peak_gib": peak,
        "input": "fixed_real_corpus_batch",
        "error": error,
    }
    del model, x, y
    gc.collect()
    torch.cuda.empty_cache()
    return row


def system_row_index(rows: Sequence[Mapping[str, object]], context: int) -> Dict[str, Mapping[str, object]]:
    return {str(row["model"]): row for row in rows if int(row["context"]) == int(context)}


def add_efficiency_decisions(args, aggregate: Dict[str, object], systems: Sequence[Mapping[str, object]],
                             train_context: int, selected: Sequence[str]) -> None:
    by_model = system_row_index(systems, train_context)
    decisions = []
    for comparison in aggregate.get("paired_comparisons", []):
        candidate = comparison["candidate"]
        reference = comparison["reference"]
        c = by_model.get(candidate)
        r = by_model.get(reference)
        speed_ratio = None
        peak_ratio = None
        if c and r and c.get("status") == "ok" and r.get("status") == "ok":
            speed_ratio = float(c["tokens_per_second"]) / float(r["tokens_per_second"])
            peak_ratio = float(c["peak_gib"]) / float(r["peak_gib"])
        dnll = float(comparison["delta_nll_mean"])
        wins = int(comparison["wins"])
        seeds = int(comparison["seeds"])
        quality_win = dnll < 0.0 and wins >= math.ceil(seeds / 2)
        pareto_gate = bool(
            quality_win and speed_ratio is not None and peak_ratio is not None
            and speed_ratio >= args.pareto_min_speed_ratio
            and peak_ratio <= args.max_peak_ratio
        )
        quality_gate = bool(
            quality_win and speed_ratio is not None
            and speed_ratio >= args.quality_min_speed_ratio
        )
        decisions.append({
            **comparison,
            "system_speed_ratio_at_train_context": speed_ratio,
            "system_peak_ratio_at_train_context": peak_ratio,
            "quality_win": quality_win,
            "pareto_gate": pareto_gate,
            "quality_gate": quality_gate,
        })
    aggregate["efficiency_decisions"] = decisions

    # Three-objective frontier at the training context.
    points = []
    models = aggregate["models"]
    for name in selected:
        sysrow = by_model.get(name)
        if not sysrow or sysrow.get("status") != "ok":
            continue
        points.append({
            "model": name,
            "nll": float(models[name]["nll_mean"]),
            "tokens_per_second": float(sysrow["tokens_per_second"]),
            "peak_gib": float(sysrow["peak_gib"]),
        })
    frontier = []
    for point in points:
        dominated = False
        dominators = []
        for other in points:
            if other["model"] == point["model"]:
                continue
            no_worse = (
                other["nll"] <= point["nll"]
                and other["tokens_per_second"] >= point["tokens_per_second"]
                and other["peak_gib"] <= point["peak_gib"]
            )
            strict = (
                other["nll"] < point["nll"]
                or other["tokens_per_second"] > point["tokens_per_second"]
                or other["peak_gib"] < point["peak_gib"]
            )
            if no_worse and strict:
                dominated = True
                dominators.append(other["model"])
        point["dominated"] = dominated
        point["dominators"] = dominators
        if not dominated:
            frontier.append(point["model"])
    aggregate["pareto_points"] = points
    aggregate["pareto_frontier"] = frontier


def fmt(value: Optional[float], pattern: str = ".4f") -> str:
    if value is None or not math.isfinite(float(value)):
        return "-"
    return format(float(value), pattern)


def make_summary(args, canonical_path: Path, tokenizer_path: Path,
                 shapes: Mapping[str, Shape], corpora: Mapping[str, Corpus],
                 aggregate: Mapping[str, object], systems: Sequence[Mapping[str, object]],
                 inference: Sequence[Mapping[str, object]], selected: Sequence[str]) -> str:
    width = 224
    lines = [
        "=" * width,
        "FIELD TOKEN EFFICIENCY ARENA v19 — PAIRED HYBRID / ATTENTION-FREE SCREEN",
        "=" * width,
        f"canonical_source={canonical_path} sha256={sha256(canonical_path)}",
        f"tokenizer={tokenizer_path} sha256={sha256(tokenizer_path)} vocab={args.vocab_size:,}",
        (
            f"protocol: WikiText-103 {args.data_frac:.1%} | ctx={args.train_seq} | "
            f"steps={args.train_steps:,} | tokens/update={args.batch_size*args.accum*args.train_seq:,} | "
            f"paired seeds={list(args.model_seeds)} | span={args.span_tokens} BPE tokens | {args.amp.upper()}"
        ),
        "PPL is exp(mean seed NLL). Models in each replicate receive identical token windows.",
        "Training-system and inference rows execute the complete Field memory path.",
        "",
        "TOKENIZED CORPORA",
    ]
    for name in ("train", "validation", "test"):
        c = corpora[name]
        lines.append(
            f"{name:12s} tokens={c.tokens.numel():12,d} raw_bytes={c.raw_bytes:12,d} "
            f"bytes/token={c.bytes_per_token:.4f}"
        )

    lines.extend([
        "",
        "MODEL SHAPES",
        f"{'model':40s} {'lane':14s} {'profile':14s} {'params':>13s} {'dTarget%':>10s} {'dim':>6s} {'layers':>7s} {'heads':>6s} {'ff':>7s}",
    ])
    for name in selected:
        shape = shapes[name]
        delta = 100.0 * (shape.params - args.target_params) / args.target_params
        lines.append(
            f"{name:40s} {model_lane(name):14s} {model_profile(name):14s} "
            f"{shape.params:13,d} {delta:+10.3f} {shape.dim:6d} {shape.layers:7d} "
            f"{shape.heads:6d} {shape.ff_hidden:7d}"
        )

    lines.extend([
        "",
        "PAIRED-SEED TEST QUALITY",
        f"{'model':40s} {'PPL':>10s} {'NLL mean':>10s} {'NLL sd':>9s} {'bits/tok':>10s} {'BPB norm':>10s} {'train tok/s':>14s} {'train peak':>11s}",
    ])
    models = aggregate["models"]
    for name in selected:
        row = models[name]
        lines.append(
            f"{name:40s} {row['ppl_from_mean_nll']:10.4f} {row['nll_mean']:10.5f} "
            f"{row['nll_std']:9.5f} {row['bits_per_token_mean']:10.5f} "
            f"{row['bpb_norm_mean']:10.5f} {row['train_tokens_per_second_mean']:14,.0f} "
            f"{row['train_peak_gib_mean']:10.2f}G"
        )

    lines.extend(["", "MATCHED-SUFFIX CONTEXT GENERALIZATION — SAME TARGET TOKENS"])
    lines.append(f"{'model':40s}" + "".join(f" {'PPL@'+str(c):>12s}" for c in args.matched_contexts))
    for name in selected:
        vals = models[name]["matched_suffix"]
        lines.append(
            f"{name:40s}" + "".join(
                f" {vals[str(c)]['ppl_from_mean_nll']:12.4f}" for c in args.matched_contexts
            )
        )

    lines.extend([
        "",
        "EQUAL NO-CHECKPOINT TRAINING SYSTEMS — FIXED REAL-CORPUS BATCH",
        f"{'model':40s} {'ctx':>7s} {'batch':>6s} {'status':>8s} {'tok/s':>13s} {'step ms':>11s} {'peak GB':>9s}",
    ])
    for row in systems:
        lines.append(
            f"{row['model']:40s} {int(row['context']):7d} {int(row['batch']):6d} "
            f"{str(row['status']):>8s} {fmt(row.get('tokens_per_second'), ',.0f'):>13s} "
            f"{fmt(row.get('step_ms'), '.2f'):>11s} {fmt(row.get('peak_gib'), '.2f'):>9s}"
        )

    lines.extend([
        "",
        "FULL-PATH INFERENCE — FIXED REAL-CORPUS BATCH, TOKEN NLL, NO BACKWARD",
        f"{'model':40s} {'ctx':>7s} {'batch':>6s} {'status':>8s} {'tok/s':>13s} {'latency ms':>12s} {'peak GB':>9s}",
    ])
    for row in inference:
        lines.append(
            f"{row['model']:40s} {int(row['context']):7d} {int(row['batch']):6d} "
            f"{str(row['status']):>8s} {fmt(row.get('tokens_per_second'), ',.0f'):>13s} "
            f"{fmt(row.get('latency_ms'), '.2f'):>12s} {fmt(row.get('peak_gib'), '.2f'):>9s}"
        )

    lines.extend([
        "",
        "LANE EFFICIENCY DECISIONS",
        f"{'candidate':40s} {'reference':36s} {'wins':>7s} {'dNLL':>10s} {'dPPL%':>10s} {'sys speed':>10s} {'sys peak':>9s} {'Pareto':>8s} {'Quality':>8s}",
    ])
    for row in aggregate.get("efficiency_decisions", []):
        lines.append(
            f"{row['candidate']:40s} {row['reference']:36s} "
            f"{str(row['wins'])+'/'+str(row['seeds']):>7s} {row['delta_nll_mean']:+10.5f} "
            f"{row['ppl_relative_pct_from_mean_delta_nll']:+10.3f} "
            f"{fmt(row.get('system_speed_ratio_at_train_context'), '.3f'):>10s} "
            f"{fmt(row.get('system_peak_ratio_at_train_context'), '.3f'):>9s} "
            f"{str(row['pareto_gate']):>8s} {str(row['quality_gate']):>8s}"
        )

    frontier = aggregate.get("pareto_frontier", [])
    lines.extend([
        "",
        f"THREE-OBJECTIVE PARETO FRONTIER @ ctx={args.train_seq}: " + (", ".join(frontier) if frontier else "none"),
        (
            f"Pareto gate: quality win + system speed >= {args.pareto_min_speed_ratio:.2f}x control "
            f"+ peak <= {args.max_peak_ratio:.2f}x control."
        ),
        f"Quality gate: quality win + system speed >= {args.quality_min_speed_ratio:.2f}x control.",
        "",
        "AUTOMATIC VERDICT",
    ])
    passed_pareto = [r["candidate"] for r in aggregate.get("efficiency_decisions", []) if r["pareto_gate"]]
    passed_quality = [r["candidate"] for r in aggregate.get("efficiency_decisions", []) if r["quality_gate"]]
    if passed_pareto:
        lines.append("Pareto-efficient overlays: " + ", ".join(passed_pareto))
    else:
        lines.append("No overlay cleared the strict Pareto gate; retain the selective controls for the next scale step.")
    if passed_quality:
        lines.append("Quality-efficient overlays: " + ", ".join(passed_quality))
    else:
        lines.append("No overlay cleared the quality gate in this screen.")
    lines.append(
        "Use this screen to nominate one hybrid and one attention-free candidate for the longer canonical token run; "
        "do not scale every arm."
    )
    lines.append("=" * width)
    return "\n".join(lines) + "\n"


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=("selftest", "train", "systems", "inference", "summary", "all"), default="all")
    p.add_argument("--outdir", default="/home/ubuntu/pcaf_runs/field_token_efficiency_v19")
    p.add_argument("--canonical-source", default="")
    p.add_argument("--cache-dir", default="/home/ubuntu/field_lab/hf_cache")
    p.add_argument("--tokenizer-source", default="")
    p.add_argument("--data-frac", type=float, default=0.05)
    p.add_argument("--data-device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--tokenizer-min-frequency", type=int, default=2)
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    p.add_argument("--target-params", type=int, default=50_000_000)
    p.add_argument("--max-param-delta-pct", type=float, default=0.75)
    p.add_argument("--train-seq", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--train-steps", type=int, default=1500)
    p.add_argument("--model-seeds", type=int, nargs="+", default=[1234, 2345])
    p.add_argument("--embedding-seed", type=int, default=314159)
    p.add_argument("--data-seed", type=int, default=5678)
    p.add_argument("--eval-seed", type=int, default=9012)

    p.add_argument("--field-dim", type=int, default=704)
    p.add_argument("--field-layers", type=int, default=8)
    p.add_argument("--field-heads", type=int, default=8)
    p.add_argument("--field-ff-hidden", type=int, default=1200)
    p.add_argument("--hybrid-ff-hidden", type=int, default=1200)
    p.add_argument("--af-ff-hidden", type=int, default=1200)
    p.add_argument("--field-chunk", type=int, default=32)
    p.add_argument("--triton-block-c", type=int, default=16)
    p.add_argument("--triton-chunk-t", type=int, default=64)
    p.add_argument("--num-buckets", type=int, default=16384)
    p.add_argument("--local-chunk", type=int, default=1024)
    p.add_argument("--salience-floor", type=float, default=0.10)
    p.add_argument("--residual-limit", type=float, default=4.0)
    p.add_argument("--span-tokens", type=int, default=2)
    p.add_argument("--address-dim", type=int, default=24)
    p.add_argument("--latent-top-k", type=int, default=4)
    p.add_argument("--score-limit", type=float, default=2.0)
    p.add_argument("--span-top-k", type=int, default=4)
    p.add_argument("--sidecar-max-mix", type=float, default=0.40)
    p.add_argument("--sidecar-gate-bias", type=float, default=1e-6)
    p.add_argument("--sidecar-aux-weight", type=float, default=0.01)
    p.add_argument("--gate-grad-scale", type=float, default=0.01)
    p.add_argument("--delta-heads", type=int, default=4)
    p.add_argument("--delta-block", type=int, default=16)
    p.add_argument("--phase-bands", type=int, default=8)
    p.add_argument("--phase-rank", type=int, default=16)

    p.add_argument("--tf-dim", type=int, default=704)
    p.add_argument("--tf-heads", type=int, default=11)
    p.add_argument("--tf-layers", type=int, default=8)
    p.add_argument("--tf-ff-hidden", type=int, default=1408)

    p.add_argument("--amp", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--field-lr", type=float, default=4e-4)
    p.add_argument("--transformer-lr", type=float, default=5e-4)
    p.add_argument("--warmup", type=int, default=125)
    p.add_argument("--min-lr-ratio", type=float, default=0.10)
    p.add_argument("--weight-decay", type=float, default=0.10)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--conf-distill-ramp", type=int, default=175)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=300)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--quick-eval-windows", type=int, default=8)
    p.add_argument("--test-token-budget", type=int, default=262144)
    p.add_argument("--matched-contexts", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    p.add_argument("--matched-score-tokens", type=int, default=128)
    p.add_argument("--matched-windows", type=int, default=8)

    p.add_argument("--system-contexts", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    p.add_argument("--system-tokens-per-step", type=int, default=4096)
    p.add_argument("--system-warmup", type=int, default=2)
    p.add_argument("--system-steps", type=int, default=5)
    p.add_argument("--inference-contexts", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    p.add_argument("--inference-tokens-per-call", type=int, default=4096)
    p.add_argument("--inference-warmup", type=int, default=2)
    p.add_argument("--inference-steps", type=int, default=5)

    p.add_argument("--pareto-min-speed-ratio", type=float, default=0.95)
    p.add_argument("--quality-min-speed-ratio", type=float, default=0.85)
    p.add_argument("--max-peak-ratio", type=float, default=1.05)
    p.add_argument("--causal-tol", type=float, default=0.005)
    p.add_argument("--prune-completed-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main():
    args = parse_args()
    selected = tuple(name for name in MODEL_NAMES if name in set(args.models))
    if not selected:
        raise ValueError("no models selected")
    if len(set(args.model_seeds)) != len(args.model_seeds):
        raise ValueError("model seeds must be unique")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/H100 required")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.set_float32_matmul_precision("high")

    arena = import_module(V15_PATH, "field_scale_50m_v15_for_v19")
    canonical_path = locate_canonical(args.canonical_source)
    actual_sha = sha256(canonical_path)
    if actual_sha != EXPECTED_CANONICAL_SHA256:
        raise RuntimeError(
            f"canonical SHA mismatch expected={EXPECTED_CANONICAL_SHA256} actual={actual_sha}"
        )

    base = arena.base
    v3 = base.import_module(base.V3_PATH, "v19_v3")
    bridge = base.import_module(base.BRIDGE_PATH, "v19_bridge")
    optmod = base.import_module(base.OPT_PATH, "v19_opt")
    epi = base.import_module(base.V9_PATH, "v19_epi")
    judge = base.import_module(base.JUDGE_PATH, "v19_judge")
    canonical = base.import_module(canonical_path, "v19_canonical")
    optmod.v3_global = v3
    base.install_fast_candidate_route(epi, optmod)

    changed = core.patch_vocab(args.vocab_size, HERE, canonical_path)
    log(f"[vocab] patched VOCAB={args.vocab_size:,} in {len(changed)} modules")

    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)
    raw_rows = core.load_raw_rows(args.cache_dir, args.data_frac)
    tokenizer = copy_or_train_tokenizer(
        root, raw_rows[0], args.vocab_size, args.tokenizer_min_frequency,
        args.tokenizer_source,
    )
    train_c, val_c, test_c = core.save_or_load_corpora(root, tokenizer, raw_rows)
    corpora = {"train": train_c, "validation": val_c, "test": test_c}
    train = core.place_tokens(train_c.tokens, device, args.data_device, "train")
    val = core.place_tokens(val_c.tokens, device, args.data_device, "validation")
    test = core.place_tokens(test_c.tokens, device, args.data_device, "test")

    # Parameter solving uses the first paired seed; counts are seed-invariant.
    first_args = replicate_args(args, 0, args.model_seeds[0])
    shapes = resolve_shapes(first_args, arena, v3, canonical, bridge, optmod, epi,
                            judge, selected)
    config = {
        "args": vars(args),
        "selected_models": list(selected),
        "canonical_source": str(canonical_path),
        "canonical_sha256": actual_sha,
        "tokenizer": str(root / "tokenizer" / "tokenizer.json"),
        "tokenizer_sha256": sha256(root / "tokenizer" / "tokenizer.json"),
        "shapes": {name: asdict(shapes[name]) for name in selected},
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    atomic_json(root / "config.json", config)

    log("=" * 180)
    log("FIELD TOKEN EFFICIENCY ARENA v19 — PAIRED HYBRID / ATTENTION-FREE SCREEN")
    log(f"gpu={torch.cuda.get_device_name(0)} torch={torch.__version__} cuda={torch.version.cuda}")
    log(f"canonical={canonical_path} sha={actual_sha}")
    log(f"seeds={args.model_seeds} embedding_seed_base={args.embedding_seed}")
    for name in selected:
        shape = shapes[name]
        delta = 100.0 * (shape.params - args.target_params) / args.target_params
        log(
            f"{name:40s} params={shape.params:,} dTarget={delta:+.3f}% "
            f"lane={model_lane(name)} profile={model_profile(name)} ff={shape.ff_hidden}"
        )
    log("=" * 180)

    if args.mode in ("selftest", "all"):
        run_selftest(first_args, shapes, arena, v3, canonical, bridge, optmod, epi,
                     judge, device, selected, root)
        if args.mode == "selftest":
            return

    per_seed: Dict[int, Dict[str, Dict[str, object]]] = {}
    aggregate_path = root / "aggregate_results.json"
    per_seed_path = root / "per_seed_results.json"

    if args.mode in ("train", "all"):
        for rep_index, seed in enumerate(args.model_seeds):
            run_args = replicate_args(args, rep_index, seed)
            rep_root = root / "replicates" / f"seed_{seed}"
            rep_root.mkdir(parents=True, exist_ok=True)
            log("-" * 180)
            log(
                f"[replicate] index={rep_index} model_seed={run_args.model_seed} "
                f"embedding_seed={run_args.embedding_seed} data_seed={run_args.data_seed}"
            )
            per_seed[int(seed)] = {}
            for name in selected:
                per_seed[int(seed)][name] = core.train_one(
                    name, shapes[name], run_args, arena, v3, canonical, bridge, optmod,
                    epi, judge, train, val, test_c, test, rep_root, device,
                )
                atomic_json(per_seed_path, per_seed)
                if args.prune_completed_checkpoints:
                    checkpoint = rep_root / "models" / name / "latest.pt"
                    if checkpoint.is_file():
                        checkpoint.unlink()
                        log(f"[{name}] pruned completed checkpoint {checkpoint}")
        aggregate = aggregate_results(per_seed, selected, args.matched_contexts)
        atomic_json(aggregate_path, aggregate)
        if args.mode == "train":
            return
    else:
        if not per_seed_path.is_file():
            raise FileNotFoundError(f"missing training results: {per_seed_path}")
        raw = json.loads(per_seed_path.read_text(encoding="utf-8"))
        per_seed = {int(seed): value for seed, value in raw.items()}
        aggregate = aggregate_results(per_seed, selected, args.matched_contexts)

    systems_path = root / "systems.json"
    if args.mode in ("systems", "all"):
        system_args = first_args
        systems: List[Dict[str, object]] = []
        for context in args.system_contexts:
            for name in selected:
                row = training_systems_benchmark(
                    name, shapes[name], system_args, arena, v3, canonical, bridge,
                    optmod, epi, judge, int(context), train_c.bytes_per_token, train, device,
                )
                systems.append(row)
                atomic_json(systems_path, systems)
                log(
                    f"[systems] {name:40s} ctx={context:5d} status={row['status']} "
                    f"tok/s={row['tokens_per_second']} peak={row['peak_gib']}"
                )
        if args.mode == "systems":
            return
    else:
        if not systems_path.is_file():
            raise FileNotFoundError(f"missing systems results: {systems_path}")
        systems = json.loads(systems_path.read_text(encoding="utf-8"))

    inference_path = root / "inference.json"
    if args.mode in ("inference", "all"):
        inference: List[Dict[str, object]] = []
        for context in args.inference_contexts:
            for name in selected:
                row = inference_benchmark(
                    name, shapes[name], first_args, arena, v3, canonical, bridge,
                    optmod, epi, judge, int(context), test, device,
                )
                inference.append(row)
                atomic_json(inference_path, inference)
                log(
                    f"[inference] {name:40s} ctx={context:5d} status={row['status']} "
                    f"tok/s={row['tokens_per_second']} peak={row['peak_gib']}"
                )
        if args.mode == "inference":
            return
    else:
        if not inference_path.is_file():
            raise FileNotFoundError(f"missing inference results: {inference_path}")
        inference = json.loads(inference_path.read_text(encoding="utf-8"))

    add_efficiency_decisions(args, aggregate, systems, args.train_seq, selected)
    atomic_json(aggregate_path, aggregate)
    summary = make_summary(
        args, canonical_path, root / "tokenizer" / "tokenizer.json", shapes, corpora,
        aggregate, systems, inference, selected,
    )
    atomic_text(root / "summary.txt", summary)
    log(summary)


if __name__ == "__main__":
    main()
