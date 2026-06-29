#!/usr/bin/env python3
"""FIELD-FUSION OFFICIAL R1 — rebuilt definitive three-seed arena.

This controller loads the bundled, already-validated current source snapshot,
reconstructs the promoted 18 native Field / 2 official Mamba-2 / 4
refresh-attention model, and compares it from scratch against:

* a parameter-matched Flash-SDPA Transformer comparator from the validated v25
  arena stack; and
* the official mamba-ssm Mamba-2 comparator from that same stack.

The quality arena is paired by seed and by exact training-window order.  Each
model receives 49,152,000 tokens per seed at context 2048, for three seeds by
default.  Final validation/test loss, matched-suffix long-context loss, training
throughput, inference throughput, peak memory and paired aggregates are saved.

The program deliberately refuses to fall back to an older Field topology.  If
it cannot locate and audit the exact current 18F/2M/4R control constructor, it
stops before update zero.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import importlib
import inspect
import json
import math
import os
import random
import shutil
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "official_source"
CANONICAL_SHA256 = "0454647a7f667cdf73f45029eee0f525c58c0788ecd433ec1a4db8c3937ac848"
OFFICIAL_FIELD = "field_official_18f2m4r"
TRANSFORMER = "transformer_flash_elite_300m"
MAMBA2 = "mamba2_official_300m"
DISPLAY_NAMES = {
    OFFICIAL_FIELD: "Fields official 18F/2M/4R",
    TRANSFORMER: "Transformer Flash Elite",
    MAMBA2: "Official Mamba-2",
}
MODELS = (OFFICIAL_FIELD, TRANSFORMER, MAMBA2)
OFFICIAL_R1_SPEC = "r1_18f_2m_4r"
CONTROL_NAMES = (OFFICIAL_R1_SPEC,)


def log(value: object = "") -> None:
    print(str(value), flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8")
    os.replace(tmp, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def nparams(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def amp_ctx(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def import_current_stack() -> Dict[str, Any]:
    """Import only the promoted R1 source and validated comparator stack."""
    if not VENDOR.is_dir():
        raise FileNotFoundError(f"official source snapshot missing: {VENDOR}")
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    r1 = importlib.import_module("field_fusion_reengineering_r1_redundancy_map")
    v25 = importlib.import_module("field_fusion_wiki100_mamba2_v25")
    v27 = importlib.import_module("field_fusion_recipe_memory_v27")
    v23 = importlib.import_module("field_fusion_wiki100_canonical_v23")
    v29 = importlib.import_module("field_fusion_delta_quality_ablation_v29")
    return {"r1": r1, "v25": v25, "v27": v27, "v23": v23, "v29": v29}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--outdir", default="/home/ubuntu/pcaf_runs/field_fusion_official_final_arena_rebuilt_run")
    p.add_argument("--canonical-source", default="/home/ubuntu/field_fusion_reengineering_r1_redundancy_map/field_only_v4_chunked_triton_wiki100.py")
    p.add_argument("--cache-dir", default="/home/ubuntu/field_lab/hf_cache")
    p.add_argument("--shared-data-root", default="/home/ubuntu/field_lab/field_fusion_official_arena_data")
    p.add_argument("--tokenizer-source", default="", help="Optional existing 16,384-vocab tokenizer. If absent or invalid, the arena trains and caches its own tokenizer.")
    p.add_argument("--data-device", choices=("auto", "cpu", "cuda"), default="cuda")
    p.add_argument("--amp", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--seeds", nargs="+", type=int, default=[1234, 2345, 3456])
    p.add_argument("--data-seeds", nargs="+", type=int, default=[5678, 6789, 7890])
    p.add_argument("--eval-seed", type=int, default=9012)
    p.add_argument("--system-seed", type=int, default=44021)
    p.add_argument("--train-token-budget", type=int, default=49_152_000)
    p.add_argument("--train-seq", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.10)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-fraction", type=float, default=0.02)
    p.add_argument("--wsd-stable-fraction", type=float, default=0.70)
    p.add_argument("--min-lr-ratio", type=float, default=0.10)
    p.add_argument("--log-every-updates", type=int, default=1, help="Retained for compatibility; training PPL is always printed every optimizer update.")
    p.add_argument("--ppl-ema-beta", type=float, default=0.98, help="EMA coefficient for the smoothed per-step training perplexity shown beside instantaneous PPL.")
    p.add_argument("--checkpoint-every-updates", type=int, default=1000)
    p.add_argument("--eval-milestones", nargs="+", type=int, default=[25_165_824, 49_152_000])
    p.add_argument("--validation-token-budget", type=int, default=0)
    p.add_argument("--test-token-budget", type=int, default=0)
    p.add_argument("--stream-readout-chunk", type=int, default=512)
    p.add_argument("--long-contexts", nargs="+", type=int, default=[2048, 8192, 16384, 32768, 65536])
    p.add_argument("--long-context-windows", type=int, default=6)
    p.add_argument("--long-context-score-tokens", type=int, default=128)
    p.add_argument("--system-contexts", nargs="+", type=int, default=[2048, 4096, 8192, 16384, 32768, 65536])
    p.add_argument("--system-batches", nargs="+", type=int, default=[4, 2, 1, 1, 1, 1])
    p.add_argument("--system-warmup", type=int, default=3)
    p.add_argument("--system-steps", type=int, default=8)
    p.add_argument("--max-param-delta-pct", type=float, default=0.75)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--keep-final-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run-systems", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--package-selftest", action="store_true")
    p.add_argument("--gpu-selftest", action="store_true")
    p.add_argument("--only-model", choices=MODELS)
    p.add_argument("--only-seed-index", type=int)
    p.add_argument("--worker-task", type=Path)
    args = p.parse_args()
    if len(args.seeds) != len(args.data_seeds):
        p.error("--seeds and --data-seeds must have equal lengths")
    if len(args.system_contexts) != len(args.system_batches):
        p.error("--system-contexts and --system-batches must have equal lengths")
    if args.train_token_budget % args.train_seq:
        p.error("train-token-budget must divide train-seq")
    if args.train_token_budget % (args.train_seq * args.batch_size):
        p.error("train-token-budget must divide train-seq*batch-size")
    if sorted(set(args.eval_milestones))[-1] != args.train_token_budget:
        p.error("eval milestones must end at train-token-budget")
    if not 0.0 <= args.ppl_ema_beta < 1.0:
        p.error("--ppl-ema-beta must satisfy 0 <= beta < 1")
    return args


def create_base_args(args: argparse.Namespace, stack: Mapping[str, Any]) -> argparse.Namespace:
    r1 = stack["r1"]
    old = sys.argv
    try:
        # The frozen R1 parser predates this arena and requires --outdir.
        # Supply the arena output directory explicitly while parsing its
        # defaults; do not leak the arena CLI into the legacy parser.
        sys.argv = [old[0], "--outdir", str(args.outdir)]
        base = r1.parse_args()
    finally:
        sys.argv = old

    overrides = {
        "outdir": args.outdir,
        "canonical_source": args.canonical_source,
        "cache_dir": args.cache_dir,
        "tokenizer_source": "",
        "data_frac": 1.0,
        "data_device": args.data_device,
        "vocab_size": 16_384,
        "tokenizer_min_frequency": 2,
        "target_params": 300_000_000,
        "max_param_delta_pct": args.max_param_delta_pct,
        "param_tolerance_pct": args.max_param_delta_pct,
        "dim": 1024,
        "layers": 24,
        "heads": 16,
        "field_chunk": 32,
        "triton_block_c": 32,
        "triton_chunk_t": 64,
        "num_buckets": 16_384,
        "salience_floor": 0.10,
        "residual_limit": 4.0,
        "fusion_q_heads": 16,
        "fusion_kv_heads": 4,
        "fusion_latent_dim": 256,
        "refresh_windows": [256, 512, 1024, 1024],
        "landmark_chunk": 256,
        "fusion_checkpoint_policy": "field_half",
        "model_seed": args.seeds[0],
        "embedding_seed": 314159,
        "data_seed": args.data_seeds[0],
        "eval_seed": args.eval_seed,
        "system_seed": args.system_seed,
        "amp": args.amp,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "min_lr_ratio": args.min_lr_ratio,
        "warmup_fraction": args.warmup_fraction,
        "wsd_stable_fraction": args.wsd_stable_fraction,
        "batch_size": args.batch_size,
        "train_seq": args.train_seq,
        "quality_token_budget": args.train_token_budget,
        "screen_token_budget": args.train_token_budget,
        "ablation_token_budget": args.train_token_budget,
        "target_v28_results": "/home/ubuntu/pcaf_runs/field_fusion_scaling_confirmation_v28_run/results.json",
        "target_v28_starts": "/home/ubuntu/pcaf_runs/field_fusion_scaling_confirmation_v28_run/paired_example_starts.npy",
        "target_v30_root": "/home/ubuntu/pcaf_runs/field_fusion_finalists_49m_v30_run",
        "target_v31_root": "/home/ubuntu/pcaf_runs/field_fusion_pcaf_long_infer_v31_run",
        "source_v32_root": "/home/ubuntu/pcaf_runs/field_fusion_gap_kernel_v32_run",
        "target_v34_root": "/home/ubuntu/pcaf_runs/field_fusion_lr_calibration_v34_run",
    }
    for key, value in overrides.items():
        setattr(base, key, value)

    # Official Mamba defaults are added by the validated comparator module.
    if hasattr(stack["v25"], "add_mamba_defaults"):
        stack["v25"].add_mamba_defaults(base)
    return base


def configure_current_stack(base_args: argparse.Namespace, stack: Mapping[str, Any]) -> None:
    """Install the frozen R1 constructor and validated runtime hooks."""
    r1, v29 = stack["r1"], stack["v29"]
    r1.configure_r1(base_args)
    v29.configure(base_args)

def unpack_prepared(prepared: Any, current: Any, r1: Any, base_args: argparse.Namespace) -> Dict[str, Any]:
    if not isinstance(prepared, tuple):
        raise TypeError(f"prepare_training returned {type(prepared).__name__}, expected tuple")
    tensors = [x for x in prepared if isinstance(x, torch.Tensor) and x.ndim == 1]
    if len(tensors) < 3:
        raise RuntimeError(f"prepare_training exposed only {len(tensors)} one-dimensional corpora")
    train, val, test = tensors[-3:]

    deps = next((x for x in prepared if isinstance(x, tuple) and len(x) >= 5), None)
    if deps is None:
        raise RuntimeError("dependency tuple not found in prepare_training output")

    shapes = None
    for x in prepared:
        if isinstance(x, dict) and x and all(hasattr(v, "params") for v in x.values()):
            shapes = x
    if shapes is None:
        solved = current.solve_shapes(base_args, deps)
        if isinstance(solved, tuple):
            shapes = next((x for x in solved if isinstance(x, dict)), None)
        elif isinstance(solved, dict):
            shapes = solved
    if shapes is None:
        raise RuntimeError("candidate shape map not found")

    corpora = [x for x in prepared if hasattr(x, "bytes_per_token")]
    train_c = corpora[-3] if len(corpora) >= 3 else None
    val_c = corpora[-2] if len(corpora) >= 2 else None
    test_c = corpora[-1] if len(corpora) >= 1 else None
    canonical_path = next((x for x in prepared if isinstance(x, Path) and x.name.endswith(".py")), Path(base_args.canonical_source))
    canonical_hash = next((x for x in prepared if isinstance(x, str) and len(x) == 64 and all(c in "0123456789abcdef" for c in x.lower())), sha256(Path(base_args.canonical_source)))
    return {
        "prepared": prepared,
        "deps": deps,
        "shapes": shapes,
        "train": train,
        "val": val,
        "test": test,
        "train_c": train_c,
        "val_c": val_c,
        "test_c": test_c,
        "canonical_path": canonical_path,
        "canonical_hash": canonical_hash,
    }


def find_control_spec(current: Any, r1: Any) -> Tuple[str, Any]:
    del current
    table = getattr(r1, "SPEC_BY_NAME", None)
    if not isinstance(table, Mapping) or OFFICIAL_R1_SPEC not in table:
        raise RuntimeError(f"promoted spec {OFFICIAL_R1_SPEC!r} not found")
    arm = getattr(r1, "R1_BY_NAME", {}).get(OFFICIAL_R1_SPEC)
    if arm is None or int(arm.field_count) != 18 or len(arm.mamba_positions) != 2 or len(arm.refresh_positions) != 4:
        raise RuntimeError(f"promoted R1 arm metadata is invalid: {arm!r}")
    return OFFICIAL_R1_SPEC, table[OFFICIAL_R1_SPEC]

def find_shape(shapes: Mapping[str, Any], control_name: str) -> Any:
    if control_name in shapes:
        return shapes[control_name]
    for name in CONTROL_NAMES:
        if name in shapes:
            return shapes[name]
    # Some solvers return a single current-control shape under an implementation name.
    candidates = [(k, v) for k, v in shapes.items() if "18f" in k.lower() and "2m" in k.lower()]
    if len(candidates) == 1:
        return candidates[0][1]
    raise KeyError(f"control shape absent; available={sorted(shapes)}")


def build_official_field(stack: Mapping[str, Any], spec: Any, shape: Any, base_args: argparse.Namespace, deps: Any, device: torch.device) -> nn.Module:
    r1 = stack["r1"]
    if getattr(spec, "name", None) != OFFICIAL_R1_SPEC:
        raise RuntimeError(f"refusing non-official Field spec: {getattr(spec, 'name', None)!r}")
    model = r1.build_candidate_r1(spec, shape, base_args, deps, device)
    if getattr(model, "_r1_arm_name", None) != OFFICIAL_R1_SPEC:
        raise RuntimeError("R1 builder did not stamp the promoted arm name")
    return model

def comparator_shapes(stack: Mapping[str, Any], base_args: argparse.Namespace, deps: Any) -> Dict[str, Any]:
    v25 = stack["v25"]
    shapes = v25.solve_shapes_v25(base_args, deps)
    return {
        TRANSFORMER: shapes[v25.TRANSFORMER],
        MAMBA2: shapes[v25.MAMBA2],
    }


def build_model(model_name: str, stack: Mapping[str, Any], control_spec: Any, field_shape: Any, comp_shapes: Mapping[str, Any], base_args: argparse.Namespace, deps: Any, device: torch.device) -> Tuple[nn.Module, str]:
    if model_name == OFFICIAL_FIELD:
        model = build_official_field(stack, control_spec, field_shape, base_args, deps, device)
        return model, stack["v25"].FUSION
    v25 = stack["v25"]
    backend = v25.TRANSFORMER if model_name == TRANSFORMER else v25.MAMBA2
    model = v25.build_model_v25(backend, comp_shapes[model_name], base_args, deps, device)
    return model, backend


def topology_audit(model: nn.Module) -> Dict[str, Any]:
    blocks = list(getattr(model, "blocks", []))
    classes = [type(b).__name__ for b in blocks]
    mamba_idx = [i for i, name in enumerate(classes) if "mamba" in name.lower()]
    refresh_idx = [i for i, name in enumerate(classes) if "refresh" in name.lower() or "attention" in name.lower()]
    native_idx = [i for i in range(len(blocks)) if i not in set(mamba_idx) | set(refresh_idx)]
    rejected_markers = (
        "GroupedGateConditioned",
        "JointResidualGeometry",
        "SummaryRefresh",
        "PCAFWriteRead",
        "PackedWriteGate",
    )
    rejected = []
    for module_name, module in model.named_modules():
        descriptor = f"{module_name} {type(module).__name__} {type(module).__module__}".lower()
        if any(marker.lower() in descriptor for marker in rejected_markers):
            rejected.append({"name": module_name, "class": type(module).__name__})
    out = {
        "block_count": len(blocks),
        "classes": classes,
        "native_field_indices": native_idx,
        "mamba_indices": mamba_idx,
        "refresh_indices": refresh_idx,
        "native_field_count": len(native_idx),
        "mamba_count": len(mamba_idx),
        "refresh_count": len(refresh_idx),
        "rejected_module_names": rejected,
    }
    if len(blocks) != 24 or len(native_idx) != 18 or len(mamba_idx) != 2 or len(refresh_idx) != 4 or rejected:
        raise AssertionError("official topology audit failed: " + json.dumps(out, indent=2))
    return out


def optimizer_for_field(model: nn.Module, base_args: argparse.Namespace, stack: Mapping[str, Any]):
    """Use the exact grouped optimizer recipe attached to promoted R1."""
    r1 = stack["r1"]
    fn = getattr(getattr(r1, "v35", None), "make_optimizer_v35", None)
    if not callable(fn):
        fn = getattr(getattr(r1, "v29", None), "make_candidate_optimizer", None)
    if not callable(fn):
        raise RuntimeError("promoted grouped optimizer unavailable")
    last_error = None
    for call in (lambda: fn(model, base_args), lambda: fn(model, base_args.lr, base_args.weight_decay)):
        try:
            opt = call()
            if isinstance(opt, torch.optim.Optimizer):
                for group in opt.param_groups:
                    scaled = float(base_args.lr) * float(group.get("lr_scale", 1.0))
                    group["lr"] = scaled
                    group["arena_base_lr"] = scaled
                return opt, "official_r1_grouped"
        except Exception as exc:
            last_error = exc
    raise RuntimeError("unable to instantiate promoted grouped optimizer") from last_error

def optimizer_for_comparator(model: nn.Module, base_args: argparse.Namespace):
    kwargs = dict(lr=base_args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=base_args.weight_decay)
    try:
        opt = torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
    except (TypeError, RuntimeError):
        opt = torch.optim.AdamW(model.parameters(), **kwargs)
    for group in opt.param_groups:
        group["arena_base_lr"] = float(group["lr"])
    return opt


def wsd_multiplier(update: int, total: int, warmup_fraction: float, stable_fraction: float, min_ratio: float) -> float:
    warmup = max(1, int(round(total * warmup_fraction)))
    stable_end = max(warmup, int(round(total * stable_fraction)))
    if update <= warmup:
        return update / warmup
    if update <= stable_end:
        return 1.0
    progress = (update - stable_end) / max(total - stable_end, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def apply_lr(optimizer: torch.optim.Optimizer, multiplier: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(group.get("arena_base_lr", group["lr"])) * multiplier


def make_starts(train: torch.Tensor, sequences: int, seq: int, seed: int, path: Path) -> np.ndarray:
    max_start = int(train.numel()) - seq - 1
    if max_start <= 0:
        raise ValueError("training corpus is shorter than one sequence")
    if path.is_file():
        starts = np.load(path, allow_pickle=False)
        if starts.shape == (sequences,) and int(starts.max()) < max_start:
            return starts.astype(np.int64, copy=False)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, max_start, size=sequences, dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, starts)
    return starts


def batch_from_starts(train: torch.Tensor, starts: np.ndarray, sequence_index: int, batch: int, seq: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    rows = []
    for start in starts[sequence_index:sequence_index + batch].tolist():
        rows.append(train[int(start):int(start) + seq + 1].long())
    window = torch.stack(rows, dim=0)
    if window.device != device:
        window = window.to(device, non_blocking=True)
    return window[:, :-1], window[:, 1:]


def loss_call(stack: Mapping[str, Any], backend_name: str, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    v25 = stack["v25"]
    if hasattr(v25, "loss_call_v25"):
        loss, _ = v25.loss_call_v25(backend_name, model, x, y)
        return loss
    loss, _ = v25.loss_call(backend_name, model, x, y)
    return loss


def streaming_token_nll(stack: Mapping[str, Any], backend_name: str, model: nn.Module, x: torch.Tensor, y: torch.Tensor, chunk: int, return_tokens: bool) -> torch.Tensor:
    v27 = stack["v27"]
    if hasattr(v27, "streaming_nll"):
        return v27.streaming_nll(backend_name, model, x, y, chunk, return_tokens)
    v25 = stack["v25"]
    token = v25.token_nll_v25(backend_name, model, x, y)
    return token if return_tokens else token.mean()


@torch.inference_mode()
def evaluate_corpus(stack: Mapping[str, Any], backend_name: str, model: nn.Module, corpus: Any, data: torch.Tensor, context: int, token_budget: int, readout_chunk: int, device: torch.device, amp: str) -> Dict[str, float]:
    model.eval()
    usable = min(int(data.numel()) - 1, token_budget if token_budget > 0 else int(data.numel()) - 1)
    total_nll = 0.0
    total_tokens = 0
    started = time.perf_counter()
    for start in range(0, usable, context):
        length = min(context, usable - start)
        if length < 8:
            break
        win = data[start:start + length + 1].long()
        if win.device != device:
            win = win.to(device, non_blocking=True)
        x, y = win[:-1][None], win[1:][None]
        with amp_ctx(device, amp):
            mean_nll = streaming_token_nll(stack, backend_name, model, x, y, min(readout_chunk, length), False)
        count = int(y.numel())
        total_nll += float(mean_nll.detach().float().cpu()) * count
        total_tokens += count
    sync(device)
    seconds = time.perf_counter() - started
    mean = total_nll / max(total_tokens, 1)
    bytes_per_token = float(getattr(corpus, "bytes_per_token", 1.0)) if corpus is not None else 1.0
    result = {
        "context": int(context),
        "nll": mean,
        "ppl": math.exp(min(mean, 20.0)),
        "bpb_norm": (total_nll / math.log(2.0)) / max(total_tokens * bytes_per_token, 1e-9),
        "tokens": total_tokens,
        "seconds": seconds,
        "tokens_per_second": total_tokens / max(seconds, 1e-9),
    }
    model.train()
    return result


@torch.inference_mode()
def matched_suffix_contexts(stack: Mapping[str, Any], backend_name: str, model: nn.Module, data: torch.Tensor, contexts: Sequence[int], score_tokens: int, windows: int, seed: int, readout_chunk: int, device: torch.device, amp: str) -> Dict[str, Dict[str, Any]]:
    """Matched-suffix quality sweep that records OOM/error rows instead of aborting.

    The same end positions are used for every context.  This makes the context
    delta interpretable while allowing a quadratic comparator to report an OOM
    at 64K without destroying the completed quality run.
    """
    model.eval()
    max_context = max(map(int, contexts))
    if int(data.numel()) <= max_context + 2:
        raise ValueError("test corpus cannot support requested maximum context")
    rng = np.random.default_rng(seed)
    ends = rng.integers(max_context, int(data.numel()) - 1, size=windows).tolist()
    out: Dict[str, Dict[str, Any]] = {}
    for context in contexts:
        context = int(context)
        total = 0.0
        count = 0
        started = time.perf_counter()
        try:
            for end_pos in ends:
                win = data[end_pos - context:end_pos + 1].long()
                if win.device != device:
                    win = win.to(device, non_blocking=True)
                x, y = win[:-1][None], win[1:][None]
                with amp_ctx(device, amp):
                    token_nll = streaming_token_nll(
                        stack, backend_name, model, x, y,
                        min(readout_chunk, context), True,
                    ).float()
                tail = token_nll[:, -min(score_tokens, context):]
                total += float(tail.sum().cpu())
                count += int(tail.numel())
            sync(device)
            seconds = time.perf_counter() - started
            mean = total / max(count, 1)
            row: Dict[str, Any] = {
                "status": "ok",
                "context": context,
                "nll": mean,
                "ppl": math.exp(min(mean, 20.0)),
                "tokens": count,
                "seconds": seconds,
                "score_tokens": min(score_tokens, context),
                "windows": windows,
            }
            log(
                f"[context] backend={backend_name} ctx={context} "
                f"nll={mean:.5f} ppl={row['ppl']:.4f}"
            )
        except torch.cuda.OutOfMemoryError as exc:
            clear_cuda()
            row = {
                "status": "oom",
                "context": context,
                "nll": None,
                "ppl": None,
                "tokens": count,
                "seconds": time.perf_counter() - started,
                "score_tokens": min(score_tokens, context),
                "windows": windows,
                "error": str(exc),
            }
            log(f"[context] backend={backend_name} ctx={context} status=OOM")
        except Exception as exc:
            clear_cuda()
            row = {
                "status": "error",
                "context": context,
                "nll": None,
                "ppl": None,
                "tokens": count,
                "seconds": time.perf_counter() - started,
                "score_tokens": min(score_tokens, context),
                "windows": windows,
                "error": repr(exc),
            }
            log(f"[context] backend={backend_name} ctx={context} status=ERROR error={exc!r}")
        out[str(context)] = row
    model.train()
    return out


def save_checkpoint(path: Path, signature: Mapping[str, Any], model: nn.Module, optimizer: torch.optim.Optimizer, sequence_index: int, compute_seconds: float, history: Sequence[Mapping[str, Any]], ppl_ema: Optional[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({
        "signature": dict(signature),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "sequence_index": int(sequence_index),
        "compute_seconds": float(compute_seconds),
        "history": list(history),
        "ppl_ema": None if ppl_ema is None else float(ppl_ema),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, signature: Mapping[str, Any], model: nn.Module, optimizer: torch.optim.Optimizer) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if raw.get("signature") != dict(signature):
        raise RuntimeError(f"checkpoint signature mismatch: {path}")
    model.load_state_dict(raw["model"], strict=True)
    optimizer.load_state_dict(raw["optimizer"])
    if raw.get("torch_rng") is not None:
        torch.set_rng_state(raw["torch_rng"])
    if raw.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(raw["cuda_rng"])
    return raw


def export_bf16(path: Path, model_name: str, model: nn.Module, metadata: Mapping[str, Any]) -> None:
    state = {}
    for key, value in model.state_dict().items():
        cpu = value.detach().cpu()
        state[key] = cpu.to(torch.bfloat16) if cpu.is_floating_point() else cpu
    tmp = path.with_suffix(".tmp")
    torch.save({
        "format": "field_fusion_official_final_arena_bf16",
        "model": model_name,
        "metadata": dict(metadata),
        "state_dict": state,
    }, tmp)
    os.replace(tmp, path)


@dataclass
class TrainResult:
    model: str
    display_name: str
    seed_index: int
    model_seed: int
    data_seed: int
    backend_name: str
    parameters: int
    optimizer_recipe: str
    train_tokens: int
    updates: int
    compute_seconds: float
    train_tokens_per_second: float
    peak_gib: float
    validation: Dict[str, float]
    test: Dict[str, float]
    contexts: Dict[str, Dict[str, float]]
    milestones: List[Dict[str, Any]]
    checkpoint: str
    export: str
    topology: Optional[Dict[str, Any]]


def train_one(args: argparse.Namespace, stack: Mapping[str, Any], base_args: argparse.Namespace, prepared: Mapping[str, Any], control_name: str, control_spec: Any, field_shape: Any, comp_shapes: Mapping[str, Any], model_name: str, seed_index: int, device: torch.device, runroot: Path) -> TrainResult:
    model_seed = int(args.seeds[seed_index])
    data_seed = int(args.data_seeds[seed_index])
    out = runroot / "quality" / f"seed{seed_index}_{model_seed}" / model_name
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "result.json"
    if result_path.is_file() and args.resume:
        return TrainResult(**json.loads(result_path.read_text(encoding="utf-8")))

    base_args.model_seed = model_seed
    base_args.data_seed = data_seed
    seed_all(model_seed)
    model, backend = build_model(
        model_name, stack, control_spec, field_shape, comp_shapes,
        base_args, prepared["deps"], device,
    )
    topology = topology_audit(model) if model_name == OFFICIAL_FIELD else None
    params = nparams(model)

    if model_name == OFFICIAL_FIELD:
        optimizer, optimizer_recipe = optimizer_for_field(model, base_args, stack)
    else:
        optimizer = optimizer_for_comparator(model, base_args)
        optimizer_recipe = "adamw_uniform"

    sequences = args.train_token_budget // args.train_seq
    starts = make_starts(
        prepared["train"], sequences, args.train_seq, data_seed,
        runroot / "paired_starts" / f"seed{seed_index}_{data_seed}.npy",
    )
    updates = sequences // args.batch_size
    milestones = sorted(set(map(int, args.eval_milestones)))
    signature = {
        "arena_version": 2,
        "model": model_name,
        "control_name": control_name,
        "model_seed": model_seed,
        "data_seed": data_seed,
        "train_tokens": args.train_token_budget,
        "train_seq": args.train_seq,
        "batch": args.batch_size,
        "canonical_sha256": prepared["canonical_hash"],
        "source_manifest_sha256": sha256(VENDOR / "OFFICIAL_SOURCE_MANIFEST.json"),
        "parameters": params,
        "ppl_ema_beta": float(args.ppl_ema_beta),
        "per_step_ppl_logging": True,
    }
    checkpoint = out / "latest.pt"
    sequence_index = 0
    prior_compute = 0.0
    history: List[Dict[str, Any]] = []
    raw = load_checkpoint(checkpoint, signature, model, optimizer) if args.resume else None
    ppl_ema: Optional[float] = None
    if raw:
        sequence_index = int(raw["sequence_index"])
        prior_compute = float(raw.get("compute_seconds", 0.0))
        history = list(raw.get("history", []))
        if raw.get("ppl_ema") is not None:
            ppl_ema = float(raw["ppl_ema"])

    torch.cuda.reset_peak_memory_stats(device)
    train_compute_seconds = float(prior_compute)
    model.train()
    milestone_set = set(milestones)
    already = {int(x["train_tokens"]) for x in history if "train_tokens" in x}
    update = sequence_index // args.batch_size
    while sequence_index < sequences:
        update += 1
        multiplier = wsd_multiplier(
            update, updates, args.warmup_fraction,
            args.wsd_stable_fraction, args.min_lr_ratio,
        )
        apply_lr(optimizer, multiplier)
        step_started = time.perf_counter()
        x, y = batch_from_starts(
            prepared["train"], starts, sequence_index,
            args.batch_size, args.train_seq, device,
        )
        optimizer.zero_grad(set_to_none=True)
        with amp_ctx(device, args.amp):
            loss = loss_call(stack, backend, model, x, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss {model_name} seed={model_seed} update={update}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
        optimizer.step()
        # Per-step PPL is a required live metric.  Synchronizing here makes the
        # displayed step throughput and accumulated training-only time honest.
        sync(device)
        step_seconds = time.perf_counter() - step_started
        train_compute_seconds += step_seconds
        sequence_index += args.batch_size
        trained_tokens = sequence_index * args.train_seq

        # Mandatory per-update visibility: instantaneous training perplexity is
        # printed on every optimizer step, plus a smoothed trend value.
        loss_value = float(loss.detach().float().cpu())
        train_ppl = math.exp(min(loss_value, 20.0))
        if ppl_ema is None:
            ppl_ema = train_ppl
        else:
            beta = float(args.ppl_ema_beta)
            ppl_ema = beta * ppl_ema + (1.0 - beta) * train_ppl
        elapsed = train_compute_seconds
        tps = trained_tokens / max(elapsed, 1e-9)
        step_tps = (args.batch_size * args.train_seq) / max(step_seconds, 1e-9)
        current_lr = max(float(group["lr"]) for group in optimizer.param_groups)
        log(
            f"[{model_name} seed={model_seed}] step={update}/{updates} "
            f"tokens={trained_tokens:,} train_loss={loss_value:.5f} "
            f"train_ppl={train_ppl:.4f} ppl_ema={ppl_ema:.4f} "
            f"lr={current_lr:.3e} lr_mult={multiplier:.4f} "
            f"grad={grad_norm:.3f} step_tok/s={step_tps:,.0f} avg_tok/s={tps:,.0f}"
        )

        reached = [m for m in milestones if trained_tokens >= m and m not in already]
        for milestone in reached:
            val = evaluate_corpus(
                stack, backend, model, prepared["val_c"], prepared["val"],
                args.train_seq, args.validation_token_budget,
                args.stream_readout_chunk, device, args.amp,
            )
            test = evaluate_corpus(
                stack, backend, model, prepared["test_c"], prepared["test"],
                args.train_seq, args.test_token_budget,
                args.stream_readout_chunk, device, args.amp,
            )
            row = {
                "train_tokens": milestone,
                "update": update,
                "validation": val,
                "test": test,
            }
            history.append(row)
            already.add(milestone)
            atomic_json(out / "milestones.json", history)
            log(
                f"[{model_name} seed={model_seed}] MILESTONE {milestone:,} "
                f"val={val['nll']:.5f} val_ppl={val['ppl']:.4f} "
                f"test={test['nll']:.5f} test_ppl={test['ppl']:.4f}"
            )

        if update % args.checkpoint_every_updates == 0 or trained_tokens == args.train_token_budget:
            save_checkpoint(
                checkpoint, signature, model, optimizer, sequence_index,
                train_compute_seconds, history, ppl_ema,
            )

        del x, y, loss

    sync(device)
    compute_seconds = train_compute_seconds
    peak_gib = torch.cuda.max_memory_allocated(device) / 2**30
    validation = evaluate_corpus(
        stack, backend, model, prepared["val_c"], prepared["val"],
        args.train_seq, args.validation_token_budget,
        args.stream_readout_chunk, device, args.amp,
    )
    test = evaluate_corpus(
        stack, backend, model, prepared["test_c"], prepared["test"],
        args.train_seq, args.test_token_budget,
        args.stream_readout_chunk, device, args.amp,
    )
    contexts = matched_suffix_contexts(
        stack, backend, model, prepared["test"], args.long_contexts,
        args.long_context_score_tokens, args.long_context_windows,
        args.eval_seed + seed_index * 1009, args.stream_readout_chunk,
        device, args.amp,
    )
    export_path = out / "final_bf16.pt"
    export_bf16(export_path, model_name, model, signature)
    result = TrainResult(
        model=model_name,
        display_name=DISPLAY_NAMES[model_name],
        seed_index=seed_index,
        model_seed=model_seed,
        data_seed=data_seed,
        backend_name=backend,
        parameters=params,
        optimizer_recipe=optimizer_recipe,
        train_tokens=args.train_token_budget,
        updates=updates,
        compute_seconds=compute_seconds,
        train_tokens_per_second=args.train_token_budget / max(compute_seconds, 1e-9),
        peak_gib=peak_gib,
        validation=validation,
        test=test,
        contexts=contexts,
        milestones=history,
        checkpoint=str(checkpoint),
        export=str(export_path),
        topology=topology,
    )
    atomic_json(result_path, asdict(result))

    if not args.keep_final_checkpoints and checkpoint.is_file():
        checkpoint.unlink()
    del model, optimizer
    clear_cuda()
    return result


def benchmark_one(args: argparse.Namespace, stack: Mapping[str, Any], base_args: argparse.Namespace, prepared: Mapping[str, Any], control_spec: Any, field_shape: Any, comp_shapes: Mapping[str, Any], model_name: str, device: torch.device) -> List[Dict[str, Any]]:
    """Isolated systems sweep using a fresh model for every row.

    Training rows include forward, exact loss path, backward, clipping and
    optimizer.  Inference is split into raw full-sequence forward and exact
    scoring (including PCAF for Fields).  No inference row retains optimizer or
    gradient tensors from a prior training benchmark.
    """
    rows: List[Dict[str, Any]] = []

    def fresh_model():
        seed_all(args.system_seed)
        return build_model(
            model_name, stack, control_spec, field_shape, comp_shapes,
            base_args, prepared["deps"], device,
        )

    for context, batch in zip(args.system_contexts, args.system_batches):
        context, batch = int(context), int(batch)

        # ---------------- training: isolated fresh model + optimizer ----------------
        train_row: Dict[str, Any] = {
            "model": model_name, "kind": "train", "context": context,
            "batch": batch,
        }
        model = optimizer = x = y = None
        try:
            clear_cuda()
            model, backend = fresh_model()
            params = nparams(model)
            train_row["parameters"] = params
            if model_name == OFFICIAL_FIELD:
                optimizer, _ = optimizer_for_field(model, base_args, stack)
            else:
                optimizer = optimizer_for_comparator(model, base_args)
            x = torch.randint(0, 16_384, (batch, context), device=device)
            y = torch.randint(0, 16_384, (batch, context), device=device)

            def train_step() -> None:
                assert model is not None and optimizer is not None and x is not None and y is not None
                model.train()
                optimizer.zero_grad(set_to_none=True)
                with amp_ctx(device, args.amp):
                    step_loss = loss_call(stack, backend, model, x, y)
                step_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            for _ in range(args.system_warmup):
                train_step()
            sync(device)
            baseline = torch.cuda.memory_allocated(device) / 2**30
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            for _ in range(args.system_steps):
                train_step()
            sync(device)
            seconds = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            train_row.update({
                "status": "ok",
                "tokens_per_second": args.system_steps * batch * context / max(seconds, 1e-9),
                "latency_ms": seconds * 1000 / args.system_steps,
                "baseline_gib": baseline,
                "peak_gib": peak,
                "activation_like_gib": max(0.0, peak - baseline),
            })
        except torch.cuda.OutOfMemoryError as exc:
            train_row.update({"status": "oom", "error": str(exc)})
        except Exception as exc:
            train_row.update({"status": "error", "error": repr(exc)})
        rows.append(train_row)
        del model, optimizer, x, y
        clear_cuda()

        # ---------------- raw forward: no optimizer / no retained grads -------------
        forward_row: Dict[str, Any] = {
            "model": model_name, "kind": "infer_forward", "context": context,
            "batch": batch,
        }
        model = x = None
        try:
            model, _ = fresh_model()
            forward_row["parameters"] = nparams(model)
            model.eval()
            x = torch.randint(0, 16_384, (batch, context), device=device)
            with torch.inference_mode():
                for _ in range(args.system_warmup):
                    with amp_ctx(device, args.amp):
                        output = model(x)
                    del output
                sync(device)
                baseline = torch.cuda.memory_allocated(device) / 2**30
                torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                for _ in range(args.system_steps):
                    with amp_ctx(device, args.amp):
                        output = model(x)
                    del output
                sync(device)
            seconds = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            forward_row.update({
                "status": "ok",
                "tokens_per_second": args.system_steps * batch * context / max(seconds, 1e-9),
                "latency_ms": seconds * 1000 / args.system_steps,
                "baseline_gib": baseline,
                "peak_gib": peak,
                "activation_like_gib": max(0.0, peak - baseline),
            })
        except torch.cuda.OutOfMemoryError as exc:
            forward_row.update({"status": "oom", "error": str(exc)})
        except Exception as exc:
            forward_row.update({"status": "error", "error": repr(exc)})
        rows.append(forward_row)
        del model, x
        clear_cuda()

        # ---------------- exact scoring: includes Fields PCAF path -------------------
        score_row: Dict[str, Any] = {
            "model": model_name, "kind": "infer_exact_score", "context": context,
            "batch": batch,
        }
        model = x = y = None
        try:
            model, backend = fresh_model()
            score_row["parameters"] = nparams(model)
            model.eval()
            x = torch.randint(0, 16_384, (batch, context), device=device)
            y = torch.randint(0, 16_384, (batch, context), device=device)
            with torch.inference_mode():
                for _ in range(args.system_warmup):
                    with amp_ctx(device, args.amp):
                        value = streaming_token_nll(
                            stack, backend, model, x, y,
                            min(args.stream_readout_chunk, context), False,
                        )
                    del value
                sync(device)
                baseline = torch.cuda.memory_allocated(device) / 2**30
                torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                for _ in range(args.system_steps):
                    with amp_ctx(device, args.amp):
                        value = streaming_token_nll(
                            stack, backend, model, x, y,
                            min(args.stream_readout_chunk, context), False,
                        )
                    del value
                sync(device)
            seconds = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            score_row.update({
                "status": "ok",
                "tokens_per_second": args.system_steps * batch * context / max(seconds, 1e-9),
                "latency_ms": seconds * 1000 / args.system_steps,
                "baseline_gib": baseline,
                "peak_gib": peak,
                "activation_like_gib": max(0.0, peak - baseline),
            })
        except torch.cuda.OutOfMemoryError as exc:
            score_row.update({"status": "oom", "error": str(exc)})
        except Exception as exc:
            score_row.update({"status": "error", "error": repr(exc)})
        rows.append(score_row)
        del model, x, y
        clear_cuda()

    return rows


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(x) for x in values]
    return {
        "mean": statistics.fmean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        "n": len(vals),
    }


def aggregate_results(results: Sequence[TrainResult], systems: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_model: Dict[str, List[TrainResult]] = {name: [] for name in MODELS}
    for row in results:
        by_model[row.model].append(row)
    aggregates: Dict[str, Any] = {}
    for name, rows in by_model.items():
        rows = sorted(rows, key=lambda r: r.seed_index)
        aggregates[name] = {
            "display_name": DISPLAY_NAMES[name],
            "validation_nll": mean_std([r.validation["nll"] for r in rows]),
            "test_nll": mean_std([r.test["nll"] for r in rows]),
            "train_tokens_per_second": mean_std([r.train_tokens_per_second for r in rows]),
            "peak_gib": mean_std([r.peak_gib for r in rows]),
            "parameters": sorted(set(r.parameters for r in rows)),
            "seeds": [r.model_seed for r in rows],
        }
        contexts = sorted({k for r in rows for k in r.contexts}, key=int)
        context_summary: Dict[str, Any] = {}
        for ctx in contexts:
            valid = [
                float(r.contexts[ctx]["nll"])
                for r in rows
                if r.contexts.get(ctx, {}).get("status", "ok") == "ok"
                and r.contexts.get(ctx, {}).get("nll") is not None
                and math.isfinite(float(r.contexts[ctx]["nll"]))
            ]
            statuses = [r.contexts.get(ctx, {}).get("status", "missing") for r in rows]
            context_summary[ctx] = {
                "nll": mean_std(valid) if valid else None,
                "statuses": statuses,
                "ok_runs": len(valid),
                "total_runs": len(rows),
            }
        aggregates[name]["contexts"] = context_summary

    paired: Dict[str, Any] = {}
    field_rows = {r.seed_index: r for r in by_model[OFFICIAL_FIELD]}
    for rival in (TRANSFORMER, MAMBA2):
        rival_rows = {r.seed_index: r for r in by_model[rival]}
        common = sorted(set(field_rows) & set(rival_rows))
        # Positive gain means Fields has lower NLL.
        val_gain = [rival_rows[i].validation["nll"] - field_rows[i].validation["nll"] for i in common]
        test_gain = [rival_rows[i].test["nll"] - field_rows[i].test["nll"] for i in common]
        speed_ratio = [field_rows[i].train_tokens_per_second / max(rival_rows[i].train_tokens_per_second, 1e-9) for i in common]
        memory_ratio = [field_rows[i].peak_gib / max(rival_rows[i].peak_gib, 1e-9) for i in common]
        paired[rival] = {
            "validation_gain_fields": mean_std(val_gain),
            "test_gain_fields": mean_std(test_gain),
            "train_speed_fields_over_rival": mean_std(speed_ratio),
            "train_memory_fields_over_rival": mean_std(memory_ratio),
            "fields_test_wins": sum(x > 0 for x in test_gain),
            "paired_seeds": common,
        }

    field_mean = aggregates[OFFICIAL_FIELD]["test_nll"]["mean"]
    best_model = min(MODELS, key=lambda n: aggregates[n]["test_nll"]["mean"])
    verdict = {
        "lowest_mean_test_nll": best_model,
        "lowest_mean_test_nll_display": DISPLAY_NAMES[best_model],
        "fields_is_quality_winner": best_model == OFFICIAL_FIELD,
        "interpretation": (
            "Fields has the lowest three-seed mean test NLL. Inspect paired dispersion and systems rows before making a broader claim."
            if best_model == OFFICIAL_FIELD else
            f"{DISPLAY_NAMES[best_model]} has the lowest three-seed mean test NLL; Fields is not the quality winner in this arena."
        ),
    }
    return {
        "aggregates": aggregates,
        "paired": paired,
        "systems": list(systems),
        "verdict": verdict,
    }


def summary_text(args: argparse.Namespace, results: Sequence[TrainResult], aggregate: Mapping[str, Any]) -> str:
    ag = aggregate["aggregates"]
    lines = [
        "=" * 190,
        "FIELD-FUSION OFFICIAL R1 — DEFINITIVE THREE-SEED ARENA",
        "=" * 190,
        "Official Fields topology: 18 native Field / 2 official Mamba-2 / 4 refresh-attention; original promoted runtime.",
        f"seeds={args.seeds} tokens/model/seed={args.train_token_budget:,} seq={args.train_seq} batch={args.batch_size} amp={args.amp}",
        "Transformer comparator: parameter-matched Flash-SDPA elite stack. Mamba comparator: official mamba-ssm Mamba-2.",
        "",
        f"{'model':34s} {'params':>14s} {'val mean':>11s} {'val sd':>10s} {'test mean':>11s} {'test sd':>10s} {'tok/s':>12s} {'GB':>8s}",
    ]
    for name in MODELS:
        row = ag[name]
        params = row["parameters"][0] if len(row["parameters"]) == 1 else -1
        lines.append(
            f"{DISPLAY_NAMES[name]:34s} {params:14,d} "
            f"{row['validation_nll']['mean']:11.5f} {row['validation_nll']['std']:10.5f} "
            f"{row['test_nll']['mean']:11.5f} {row['test_nll']['std']:10.5f} "
            f"{row['train_tokens_per_second']['mean']:12,.0f} {row['peak_gib']['mean']:8.2f}"
        )
    lines += ["", "PAIRED FIELDS DELTAS (positive quality gain = lower Fields NLL)"]
    for rival in (TRANSFORMER, MAMBA2):
        row = aggregate["paired"][rival]
        lines.append(
            f"vs {DISPLAY_NAMES[rival]}: dVal={row['validation_gain_fields']['mean']:+.5f}±{row['validation_gain_fields']['std']:.5f} "
            f"dTest={row['test_gain_fields']['mean']:+.5f}±{row['test_gain_fields']['std']:.5f} "
            f"speed={row['train_speed_fields_over_rival']['mean']:.3f}x "
            f"memory={row['train_memory_fields_over_rival']['mean']:.3f}x "
            f"test_wins={row['fields_test_wins']}/{len(row['paired_seeds'])}"
        )
    systems = list(aggregate.get("systems", []))
    if systems:
        lines += [
            "",
            "ISOLATED SYSTEMS HIGHLIGHTS — fresh model per row",
            f"{'kind':18s} {'ctx':>7s} {'model':34s} {'status':>8s} {'tok/s':>12s} {'peak GB':>9s}",
        ]
        highlight_contexts = {2048, 16384, 65536}
        for kind in ("train", "infer_forward", "infer_exact_score"):
            for context in sorted(highlight_contexts):
                for name in MODELS:
                    row = next((
                        r for r in systems
                        if r.get("kind") == kind
                        and int(r.get("context", -1)) == context
                        and r.get("model") == name
                    ), None)
                    if row is None:
                        continue
                    status = str(row.get("status", "missing"))
                    speed = "-" if row.get("tokens_per_second") is None else f"{float(row['tokens_per_second']):,.0f}"
                    peak = "-" if row.get("peak_gib") is None else f"{float(row['peak_gib']):.2f}"
                    lines.append(
                        f"{kind:18s} {context:7d} {DISPLAY_NAMES[name]:34s} "
                        f"{status:>8s} {speed:>12s} {peak:>9s}"
                    )
    lines += [
        "",
        "VERDICT",
        f"lowest_mean_test_nll={aggregate['verdict']['lowest_mean_test_nll']}",
        aggregate["verdict"]["interpretation"],
        "The result is an architecture-scale benchmark, not by itself proof of general superiority across datasets or scales.",
        "=" * 190,
    ]
    return "\n".join(lines) + "\n"


def _valid_tokenizer_file(v23: Any, path: Path, vocab_size: int) -> bool:
    if not path.is_file():
        return False
    try:
        tok = v23.core.Tokenizer.from_file(str(path))
        return int(tok.get_vocab_size()) == int(vocab_size)
    except Exception:
        return False


def prepare_shared_data(args: argparse.Namespace, base_args: argparse.Namespace, stack: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    """Prepare a shared WT103 tokenizer/corpus cache without a mandatory legacy path."""
    v23 = stack["v23"]
    data_root = Path(args.shared_data_root).expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    raw_rows = v23.core.load_raw_rows(base_args.cache_dir, 1.0)
    target = data_root / "tokenizer" / "tokenizer.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    source_used = None
    tokenized_dir = data_root / "tokenized"
    manifest_path = data_root / "OFFICIAL_ARENA_DATA_MANIFEST.json"
    if target.is_file() and not _valid_tokenizer_file(v23, target, base_args.vocab_size):
        bad = target.with_suffix(".invalid.json")
        target.replace(bad)
        if tokenized_dir.exists():
            shutil.rmtree(tokenized_dir)
        log(f"[tokenizer] quarantined invalid cached tokenizer as {bad}")
    if target.is_file() and manifest_path.is_file():
        try:
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
            if prior.get("tokenizer_sha256") != sha256(target) and tokenized_dir.exists():
                shutil.rmtree(tokenized_dir)
                log("[tokenizer] cleared tokenized cache after tokenizer hash change")
        except Exception:
            if tokenized_dir.exists():
                shutil.rmtree(tokenized_dir)
    if not target.is_file():
        if tokenized_dir.exists():
            shutil.rmtree(tokenized_dir)
        candidates: List[Path] = []
        if args.tokenizer_source:
            candidates.append(Path(args.tokenizer_source).expanduser())
        candidates.extend([
            Path("/home/ubuntu/field_lab/wt103_bpe16384_full_v23/tokenizer/tokenizer.json"),
            Path("/home/ubuntu/pcaf_runs/field_fusion_official_final_arena_rebuilt_run/shared_data/tokenizer/tokenizer.json"),
        ])
        seen = set()
        for candidate in candidates:
            if str(candidate) in seen:
                continue
            seen.add(str(candidate))
            if _valid_tokenizer_file(v23, candidate, base_args.vocab_size):
                shutil.copy2(candidate, target)
                source_used = str(candidate.resolve())
                log(f"[tokenizer] imported valid tokenizer from {source_used}")
                break
    tokenizer = v23.core.build_or_load_tokenizer(
        data_root, raw_rows[0], int(base_args.vocab_size), int(base_args.tokenizer_min_frequency)
    )
    if int(tokenizer.get_vocab_size()) != int(base_args.vocab_size):
        raise RuntimeError(f"tokenizer vocab mismatch: {tokenizer.get_vocab_size()} != {base_args.vocab_size}")
    train_c, val_c, test_c = v23.core.save_or_load_corpora(data_root, tokenizer, raw_rows)
    train = v23.core.place_tokens(train_c.tokens, device, base_args.data_device, "train")
    val = v23.core.place_tokens(val_c.tokens, device, base_args.data_device, "validation")
    test = v23.core.place_tokens(test_c.tokens, device, base_args.data_device, "test")
    manifest = {
        "shared_data_root": str(data_root), "tokenizer": str(target),
        "tokenizer_sha256": sha256(target), "tokenizer_source_imported": source_used,
        "vocab_size": int(tokenizer.get_vocab_size()),
        "train_tokens": int(train_c.tokens.numel()), "validation_tokens": int(val_c.tokens.numel()),
        "test_tokens": int(test_c.tokens.numel()),
        "train_bytes_per_token": float(train_c.bytes_per_token),
        "validation_bytes_per_token": float(val_c.bytes_per_token),
        "test_bytes_per_token": float(test_c.bytes_per_token),
    }
    atomic_json(data_root / "OFFICIAL_ARENA_DATA_MANIFEST.json", manifest)
    log("autonomous_shared_tokenizer=PASS")
    log("shared_wt103_corpora=PASS")
    return {"train": train, "val": val, "test": test, "train_c": train_c, "val_c": val_c, "test_c": test_c, "data_manifest": manifest}


def package_selftest(args: argparse.Namespace, stack: Mapping[str, Any]) -> None:
    canonical = Path(args.canonical_source)
    if not canonical.is_file(): raise FileNotFoundError(canonical)
    if sha256(canonical) != CANONICAL_SHA256: raise RuntimeError("canonical Field source SHA mismatch")
    manifest = VENDOR / "OFFICIAL_SOURCE_MANIFEST.json"
    if not manifest.is_file(): raise FileNotFoundError(manifest)
    r1, v25 = stack["r1"], stack["v25"]
    control_name, _ = find_control_spec(None, r1)
    required = {
        "r1_module": r1.__name__, "control_name": control_name, "direct_r1_import": True,
        "macro_c_imported": "field_fusion_macroblock_c_joint_residual_screen" in sys.modules,
        "official_mamba_available": getattr(v25, "OfficialMamba2", None) is not None,
        "mamba_version": getattr(v25, "MAMBA_VERSION", "unknown"),
        "causal_conv1d_version": getattr(v25, "CAUSAL_CONV1D_VERSION", "unknown"),
        "canonical_sha256": sha256(canonical), "manifest_sha256": sha256(manifest),
        "autonomous_shared_data_root": str(Path(args.shared_data_root).expanduser()),
    }
    if required["macro_c_imported"]: raise RuntimeError("rejected Macroblock C was imported")
    if not required["official_mamba_available"]: raise RuntimeError("official Mamba-2 unavailable")
    train_source = inspect.getsource(train_one)
    required["per_step_training_ppl_source_audit"] = all(token in train_source for token in (
        "train_ppl=", "ppl_ema=", "step={update}/{updates}", "sync(device)", "step_tok/s=", "avg_tok/s="))
    if not required["per_step_training_ppl_source_audit"]: raise AssertionError("per-step PPL audit failed")
    atomic_json(Path(args.outdir) / "package_selftest.json", required)
    log("official_source_snapshot=PASS")
    log("direct_promoted_r1_spec=PASS")
    log("rejected_macro_c_not_imported=PASS")
    log("autonomous_tokenizer_design=PASS")
    log("transformer_flash_comparator_import=PASS")
    log("official_mamba2_import=PASS")
    log("canonical_sha256=PASS")
    log("per_step_training_ppl=PASS")
    log("[package-selftest] PASS")

def prepare_training(args: argparse.Namespace, stack: Mapping[str, Any], device: torch.device, root: Path) -> Tuple[argparse.Namespace, Dict[str, Any], str, Any, Any, Dict[str, Any]]:
    """Prepare the frozen official spec, comparators and shared data directly."""
    base_args = create_base_args(args, stack)
    base_args.candidate = [OFFICIAL_R1_SPEC]
    configure_current_stack(base_args, stack)
    r1, v27 = stack["r1"], stack["v27"]
    control_name, control_spec = find_control_spec(None, r1)
    canonical_path, canonical_hash, deps = v27.load_dependencies(base_args)
    if canonical_hash != CANONICAL_SHA256:
        raise RuntimeError(f"canonical SHA mismatch expected={CANONICAL_SHA256} actual={canonical_hash}")
    base_shape, shapes, accounting = r1.solve_candidate_shapes_r1(base_args, deps)
    if set(shapes) != {OFFICIAL_R1_SPEC}:
        raise RuntimeError(f"official shape solver leaked arms: {sorted(shapes)}")
    field_shape = shapes[OFFICIAL_R1_SPEC]
    atomic_json(root / "official_component_accounting.json", accounting)
    probe = r1.build_candidate_r1(control_spec, field_shape, base_args, deps, device)
    topology = topology_audit(probe)
    if getattr(probe, "_r1_arm_name", None) != OFFICIAL_R1_SPEC:
        raise RuntimeError("official probe has wrong R1 stamp")
    del probe
    clear_cuda()
    shared = prepare_shared_data(args, base_args, stack, device)
    prepared: Dict[str, Any] = {**shared, "deps": deps, "canonical_path": canonical_path, "canonical_hash": canonical_hash, "base_shape": base_shape, "official_accounting": accounting, "official_topology": topology}
    comp_shapes = comparator_shapes(stack, base_args, deps)
    log(f"official_r1_spec={OFFICIAL_R1_SPEC} direct_source=PASS")
    return base_args, prepared, control_name, control_spec, field_shape, comp_shapes

def gpu_selftest(args: argparse.Namespace, stack: Mapping[str, Any]) -> None:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable CUDA GPU required")
    device = torch.device("cuda")
    root = Path(args.outdir)
    base_args, prepared, control_name, control_spec, field_shape, comp_shapes = prepare_training(args, stack, device, root)
    audit: Dict[str, Any] = {}
    counts = {}
    for model_name in MODELS:
        seed_all(args.seeds[0])
        model, backend = build_model(model_name, stack, control_spec, field_shape, comp_shapes, base_args, prepared["deps"], device)
        counts[model_name] = nparams(model)
        if model_name == OFFICIAL_FIELD:
            audit["topology"] = topology_audit(model)
        x = torch.randint(0, 16_384, (1, 64), device=device)
        y = torch.randint(0, 16_384, (1, 64), device=device)
        model.train()
        with amp_ctx(device, args.amp):
            loss = loss_call(stack, backend, model, x, y)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise AssertionError(f"non-finite/missing gradients in {model_name}")
        model.eval()
        with torch.inference_mode(), amp_ctx(device, args.amp):
            token = streaming_token_nll(stack, backend, model, x, y, 64, True)
        if token.shape != y.shape or not torch.isfinite(token).all():
            raise AssertionError(f"evaluation path failed for {model_name}")
        del model, x, y, loss, token
        clear_cuda()
    smallest, largest = min(counts.values()), max(counts.values())
    delta = 100.0 * (largest - smallest) / smallest
    if delta > args.max_param_delta_pct:
        raise AssertionError(f"parameter delta {delta:.4f}% exceeds {args.max_param_delta_pct}%")
    audit["parameter_counts"] = counts
    audit["parameter_delta_pct"] = delta
    atomic_json(root / "gpu_selftest.json", audit)
    log("official_topology_18f_2m_4r=PASS")
    log("parameter_parity=PASS")
    log("all_models_forward_backward=PASS")
    log("all_models_exact_eval_path=PASS")
    log("[gpu-selftest] PASS")


def run_arena(args: argparse.Namespace, stack: Mapping[str, Any]) -> None:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable CUDA GPU required")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.set_float32_matmul_precision("high")

    root = Path(args.outdir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "args.json", vars(args))
    base_args, prepared, control_name, control_spec, field_shape, comp_shapes = prepare_training(args, stack, device, root)

    # Pre-run full-size audit, one model at a time.
    param_counts: Dict[str, int] = {}
    topology = None
    for model_name in MODELS:
        model, _ = build_model(model_name, stack, control_spec, field_shape, comp_shapes, base_args, prepared["deps"], device)
        param_counts[model_name] = nparams(model)
        if model_name == OFFICIAL_FIELD:
            topology = topology_audit(model)
        del model
        clear_cuda()
    delta = 100.0 * (max(param_counts.values()) - min(param_counts.values())) / min(param_counts.values())
    if delta > args.max_param_delta_pct:
        raise RuntimeError(f"parameter parity failed: {delta:.4f}%")
    preflight = {
        "control_name": control_name,
        "official_spec_source": "direct_r1",
        "shared_data_manifest": prepared.get("data_manifest"),
        "topology": topology,
        "parameter_counts": param_counts,
        "parameter_delta_pct": delta,
        "canonical_source": str(prepared["canonical_path"]),
        "canonical_sha256": prepared["canonical_hash"],
        "source_manifest_sha256": sha256(VENDOR / "OFFICIAL_SOURCE_MANIFEST.json"),
        "models": MODELS,
        "seeds": args.seeds,
        "data_seeds": args.data_seeds,
        "train_token_budget_per_model_seed": args.train_token_budget,
        "total_quality_tokens": args.train_token_budget * len(args.seeds) * len(MODELS),
        "per_step_training_ppl": True,
        "ppl_ema_beta": float(args.ppl_ema_beta),
    }
    atomic_json(root / "preflight.json", preflight)
    log("=" * 180)
    log("FIELD-FUSION OFFICIAL R1 — FINAL ARENA PRE-RUN AUDIT")
    log(f"control={control_name} direct_r1 topology=18F/2M/4R original runtime")
    log(f"parameters={param_counts} delta={delta:.4f}%")
    log(f"seeds={args.seeds} data_seeds={args.data_seeds}")
    log(f"tokens/model/seed={args.train_token_budget:,} total={preflight['total_quality_tokens']:,}")
    log(f"per_step_training_ppl=PASS ema_beta={args.ppl_ema_beta:.3f}")
    log("=" * 180)

    results: List[TrainResult] = []
    seed_indices = range(len(args.seeds))
    if args.only_seed_index is not None:
        if not 0 <= args.only_seed_index < len(args.seeds):
            raise IndexError(args.only_seed_index)
        seed_indices = [args.only_seed_index]
    models = [args.only_model] if args.only_model else list(MODELS)

    for seed_index in seed_indices:
        if args.only_model:
            ordered_models = models
        else:
            # Rotate execution order by seed to reduce systematic thermal/cache/order bias.
            shift = seed_index % len(MODELS)
            ordered_models = list(MODELS[shift:] + MODELS[:shift])
        for model_name in ordered_models:
            result = train_one(
                args, stack, base_args, prepared, control_name, control_spec,
                field_shape, comp_shapes, model_name, seed_index, device, root,
            )
            results.append(result)

    # Include previously completed rows when resuming or running subsets.
    for path in sorted((root / "quality").glob("seed*/*/result.json")):
        raw = TrainResult(**json.loads(path.read_text(encoding="utf-8")))
        if not any(r.model == raw.model and r.seed_index == raw.seed_index for r in results):
            results.append(raw)
    results.sort(key=lambda r: (r.seed_index, MODELS.index(r.model)))

    systems: List[Dict[str, Any]] = []
    if args.run_systems and not args.only_model and args.only_seed_index is None:
        for model_name in MODELS:
            systems.extend(benchmark_one(
                args, stack, base_args, prepared, control_spec,
                field_shape, comp_shapes, model_name, device,
            ))
        atomic_json(root / "systems.json", systems)
        with (root / "systems.csv").open("w", newline="", encoding="utf-8") as f:
            keys = sorted({k for row in systems for k in row})
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(systems)

    expected = len(args.seeds) * len(MODELS)
    if len(results) == expected:
        aggregate = aggregate_results(results, systems)
        atomic_json(root / "results.json", {
            "preflight": preflight,
            "runs": [asdict(r) for r in results],
            **aggregate,
        })
        summary = summary_text(args, results, aggregate)
        atomic_text(root / "summary.txt", summary)
        log(summary)
    else:
        atomic_json(root / "partial_results.json", [asdict(r) for r in results])
        log(f"partial arena complete: {len(results)}/{expected} quality runs")


def main() -> None:
    args = parse_args()
    stack = import_current_stack()
    if args.package_selftest:
        package_selftest(args, stack)
        return
    if args.gpu_selftest:
        gpu_selftest(args, stack)
        return
    run_arena(args, stack)


if __name__ == "__main__":
    main()
