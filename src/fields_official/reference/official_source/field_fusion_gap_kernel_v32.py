#!/usr/bin/env python3
"""FIELD-FUSION v32 — isolated 16K–64K systems closure, kernel audit, and gap ablation.

Phases
------
A) Inference-only, process-isolated prefill benchmark at 16K/32K/64K for:
     * v31 Field-Mamba4-refresh1024x2 + PCAF checkpoint;
     * frozen v28 Transformer checkpoint;
     * frozen v28 official Mamba-2 checkpoint.
   Every model/context/batch attempt runs in a fresh Python/CUDA process.  A CUDA
   illegal access or OOM therefore cannot poison the controller process.  The
   report contains both each model's maximum safe batch and a matched-batch
   comparison.

B) Exact runtime/kernel sweep on the frozen v31 hybrid:
     * Field/Triton launch geometry (field_chunk, BLOCK_C, CHUNK_T);
     * cached landmark metadata (same weights/math);
     * torch.compile default / reduce-overhead / max-autotune-no-cudagraphs.
   Candidate outputs are compared with eager baseline logits before promotion.
   A component CUDA-event profile identifies Field, Mamba, refresh, softpatch,
   and final-head shares of prefill time.

C) Short paired quality screen (no rival retraining) for the remaining ~0.03 NLL:
     * WSD stable fraction 0.55 / 0.60 / 0.65 versus current 0.70;
     * Mamba LR 0.75x / 1.25x;
     * refresh LR 1.50x;
     * moving the four localized Mamba blocks from stage-end to stage-middle.
   All arms use the same first 25,165,824 v28 token windows and ~300M params.

The script never starts a long follow-up run automatically.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

import field_fusion_pcaf_long_infer_v31 as v31
import field_fusion_delta_quality_ablation_v29 as v29
import field_fusion_scaling_confirmation_v28 as v28
import field_fusion_recipe_memory_v27 as v27
import field_fusion_final_ablation_v26 as v26
import field_fusion_wiki100_canonical_v23 as v23
import field_fusion_wiki100_mamba2_v25 as v25

VERSION = 32
FUSION = v29.FUSION
TRANSFORMER = v29.TRANSFORMER
MAMBA2 = v29.MAMBA2
EXPECTED_CANONICAL_SHA256 = v29.EXPECTED_CANONICAL_SHA256
CandidateSpec = v29.CandidateSpec

CURRENT_MAMBA = (4, 10, 16, 22)
MIDDLE_MAMBA = (2, 8, 14, 20)
HYBRID_V31_SPEC = CandidateSpec(
    "field_mamba4_refresh1024x2_selected_98m",
    "v31 frozen hybrid",
    refresh_1024x2=True,
    mamba_replace=CURRENT_MAMBA,
)


@dataclass(frozen=True)
class GapArm:
    name: str
    description: str
    stable_fraction: float = 0.70
    mamba_lr_scale: float = 1.0
    refresh_lr_scale: float = 1.0
    mamba_positions: Tuple[int, ...] = CURRENT_MAMBA

    def candidate(self) -> CandidateSpec:
        return CandidateSpec(
            self.name,
            self.description,
            refresh_1024x2=True,
            mamba_replace=self.mamba_positions,
        )


GAP_ARMS: Tuple[GapArm, ...] = (
    GapArm("gap_ref_stable70", "Current v31 topology and WSD stable fraction 0.70."),
    GapArm("gap_decay55", "Start WSD decay earlier: stable fraction 0.55.", stable_fraction=0.55),
    GapArm("gap_decay60", "Start WSD decay earlier: stable fraction 0.60.", stable_fraction=0.60),
    GapArm("gap_decay65", "Start WSD decay earlier: stable fraction 0.65.", stable_fraction=0.65),
    GapArm("gap_mamba_lr075", "Use 0.75x LR for localized Mamba blocks.", mamba_lr_scale=0.75),
    GapArm("gap_mamba_lr125", "Use 1.25x LR for localized Mamba blocks.", mamba_lr_scale=1.25),
    GapArm("gap_refresh_lr150", "Use 1.50x LR for four attention refresh blocks.", refresh_lr_scale=1.50),
    GapArm(
        "gap_mamba_middle",
        "Move localized Mamba blocks from the end to the middle of each five-Field stage.",
        mamba_positions=MIDDLE_MAMBA,
    ),
)
ARM_BY_NAME = {x.name: x for x in GAP_ARMS}
SPEC_BY_NAME = {x.name: x.candidate() for x in GAP_ARMS}

# v31 already patched v29's builder to switch PCAF by candidate name.  Wrap that
# builder rather than bypassing it, so every v32 arm retains canonical PCAF-on.
_BUILD_BEFORE_V32 = v29.build_candidate
_SIGNATURE_BEFORE_V32 = v29.checkpoint_signature


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


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        with contextlib.suppress(Exception):
            torch.cuda.synchronize()
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_args() -> argparse.Namespace:
    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument(
        "--target-v31-root",
        default="/home/ubuntu/pcaf_runs/field_fusion_pcaf_long_infer_v31_run",
    )
    custom.add_argument(
        "--target-v28-results",
        default="/home/ubuntu/pcaf_runs/field_fusion_scaling_confirmation_v28_run/results.json",
    )
    custom.add_argument(
        "--target-v28-starts",
        default="/home/ubuntu/pcaf_runs/field_fusion_scaling_confirmation_v28_run/paired_example_starts.npy",
    )
    custom.add_argument("--run-inference", action=argparse.BooleanOptionalAction, default=True)
    custom.add_argument("--run-kernel-sweep", action=argparse.BooleanOptionalAction, default=True)
    custom.add_argument("--run-gap-ablation", action=argparse.BooleanOptionalAction, default=True)
    custom.add_argument("--infer-contexts", nargs="+", type=int, default=[16384, 32768, 65536])
    custom.add_argument("--infer-max-batch-tokens", type=int, default=524_288)
    custom.add_argument("--infer-max-batch", type=int, default=64)
    custom.add_argument("--infer-warmup", type=int, default=1)
    custom.add_argument("--infer-steps", type=int, default=3)
    custom.add_argument("--infer-timeout-seconds", type=int, default=900)
    custom.add_argument("--infer-min-free-gib", type=float, default=1.0)
    custom.add_argument("--infer-seed", type=int, default=732_032)
    custom.add_argument("--kernel-context", type=int, default=16384)
    custom.add_argument("--kernel-min-speedup", type=float, default=1.02)
    custom.add_argument("--kernel-max-memory-ratio", type=float, default=1.05)
    custom.add_argument("--kernel-logit-tolerance", type=float, default=0.02)
    custom.add_argument("--kernel-profile-steps", type=int, default=3)
    custom.add_argument("--ablation-token-budget", type=int, default=25_165_824)
    custom.add_argument("--ablation-eval-fractions", nargs="+", type=float, default=[0.50, 1.0])
    custom.add_argument("--ablation-validation-token-budget", type=int, default=0)
    custom.add_argument("--ablation-test-token-budget", type=int, default=293_944)
    custom.add_argument("--ablation-checkpoint-every", type=int, default=512)
    custom.add_argument("--ablation-log-every", type=int, default=128)
    custom.add_argument("--ablation-min-gain", type=float, default=0.010)
    custom.add_argument("--ablation-min-speed-ratio", type=float, default=0.93)
    custom.add_argument("--ablation-max-memory-ratio", type=float, default=1.05)
    custom.add_argument("--ablation-max-context-drift", type=float, default=0.10)
    custom.add_argument("--ablation-long-contexts", nargs="+", type=int, default=[2048, 16384, 65536])
    custom.add_argument("--ablation-long-windows", type=int, default=2)
    custom.add_argument("--gap-arm", action="append", default=[])
    custom_args, remaining = custom.parse_known_args()

    old = sys.argv
    try:
        sys.argv = [old[0], *remaining]
        args = v29.parse_args()
    finally:
        sys.argv = old
    for key, value in vars(custom_args).items():
        setattr(args, key, value)
    args.screen_token_budget = int(args.ablation_token_budget)
    args.quality_token_budget = int(args.ablation_token_budget)
    args.eval_fractions = list(map(float, args.ablation_eval_fractions))
    args.checkpoint_every_updates = int(args.ablation_checkpoint_every)
    args.profile_log_every_updates = int(args.ablation_log_every)
    args.screen_validation_token_budget = int(args.ablation_validation_token_budget)
    args.screen_test_token_budget = int(args.ablation_test_token_budget)
    args.target_v28_starts = str(args.target_v28_starts)
    args.export_winner_bf16 = False
    v25.add_mamba_defaults(args)
    return args


def selected_arms(args) -> Tuple[GapArm, ...]:
    if not args.gap_arm:
        return GAP_ARMS
    requested = set(args.gap_arm)
    rows = tuple(x for x in GAP_ARMS if x.name in requested)
    missing = requested - {x.name for x in rows}
    if missing:
        raise ValueError(f"unknown --gap-arm: {sorted(missing)}")
    if "gap_ref_stable70" not in requested:
        rows = (ARM_BY_NAME["gap_ref_stable70"],) + rows
    return rows


def build_candidate_v32(spec: CandidateSpec, shape: v23.Shape, args, deps,
                        device: torch.device) -> nn.Module:
    model = _BUILD_BEFORE_V32(spec, shape, args, deps, device)
    model._v32_arm_name = spec.name
    return model


def checkpoint_signature_v32(args, spec: CandidateSpec, shape: v23.Shape,
                             total_sequences: int) -> Dict[str, object]:
    row = dict(_SIGNATURE_BEFORE_V32(args, spec, shape, total_sequences))
    arm = ARM_BY_NAME.get(spec.name)
    if arm is not None:
        row["v32_arm"] = asdict(arm)
    row["v32_version"] = VERSION
    return row


def parse_block_index(name: str) -> Optional[int]:
    match = re.match(r"blocks\.(\d+)\.", name)
    return None if match is None else int(match.group(1))


def make_optimizer_v32(model: nn.Module, args):
    arm_name = str(getattr(model, "_v32_arm_name", "gap_ref_stable70"))
    arm = ARM_BY_NAME.get(arm_name, ARM_BY_NAME["gap_ref_stable70"])
    groups: Dict[Tuple[float, bool], List[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        scale = 2.0 if v26.is_special_parameter(name) else 1.0
        index = parse_block_index(name)
        if index is not None and index < len(model.blocks):
            block = model.blocks[index]
            if v29.is_official_mamba_block(block):
                scale *= float(arm.mamba_lr_scale)
            elif v29.is_refresh_block(block):
                scale *= float(arm.refresh_lr_scale)
        decay = parameter.ndim >= 2
        groups.setdefault((round(scale, 6), decay), []).append(parameter)
    param_groups = [
        {
            "params": params,
            "weight_decay": args.weight_decay if decay else 0.0,
            "lr_scale": float(scale),
        }
        for (scale, decay), params in groups.items()
    ]
    kwargs = dict(lr=args.lr, betas=(0.9, 0.95), eps=1.0e-8)
    try:
        return torch.optim.AdamW(param_groups, fused=True, **kwargs)
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(param_groups, **kwargs)


# Install per-arm hooks used internally by v29.train_candidate.
v29.build_candidate = build_candidate_v32
v29.checkpoint_signature = checkpoint_signature_v32
v29.make_candidate_optimizer = make_optimizer_v32


# =====================================================================================
# Frozen checkpoint discovery / model loading
# =====================================================================================


def result_row_for_model(raw: Mapping[str, object], model_name: str) -> Mapping[str, object]:
    for row in raw.get("results", {}).values():
        if row.get("model") == model_name:
            return row
    raise KeyError(model_name)


def state_from_checkpoint(path: Path) -> Mapping[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        state = obj.get("model")
        if state is None:
            state = obj.get("state_dict")
        if state is not None:
            return state
    raise RuntimeError(f"no state dict in {path}")


def first_existing(paths: Iterable[str]) -> Path:
    tried = []
    for text in paths:
        if not text:
            continue
        path = Path(text)
        tried.append(str(path))
        if path.is_file():
            return path
    raise FileNotFoundError("none of the checkpoint paths exist: " + "; ".join(tried))


def locate_v31_hybrid_result(root: Path) -> Path:
    candidates = [
        root / "candidates" / HYBRID_V31_SPEC.name / "result.json",
        root / "long_results.json",
    ]
    if candidates[0].is_file():
        return candidates[0]
    if candidates[1].is_file():
        raw = read_json(candidates[1])
        row = raw.get(HYBRID_V31_SPEC.name)
        if row is None:
            raise KeyError(HYBRID_V31_SPEC.name)
        out = root / "_v32_hybrid_result.json"
        atomic_json(out, row)
        return out
    raise FileNotFoundError(f"v31 hybrid result missing under {root}")


def worker_base_args(task: Mapping[str, object]) -> argparse.Namespace:
    old = sys.argv
    try:
        sys.argv = [
            old[0], "--outdir", str(task["scratch_outdir"]),
            "--target-v28-results", str(task["target_v28_results"]),
            "--target-v28-starts", str(task["target_v28_starts"]),
        ]
        args = v31.parse_args()
    finally:
        sys.argv = old
    args.field_chunk = int(task.get("field_chunk", args.field_chunk))
    args.triton_block_c = int(task.get("triton_block_c", args.triton_block_c))
    args.triton_chunk_t = int(task.get("triton_chunk_t", args.triton_chunk_t))
    args.fusion_checkpoint_policy = "none"
    return args


def load_worker_model(task: Mapping[str, object], args, deps,
                      device: torch.device) -> Tuple[str, nn.Module, str]:
    label = str(task["model_label"])
    if label == "field_hybrid_v31":
        result = read_json(Path(task["v31_hybrid_result"]))
        shape = v23.Shape(
            HYBRID_V31_SPEC.name,
            int(result["params"]), 1024, 24, 16, int(result["ff_hidden"]),
        )
        v31._PCAF_ENABLED[HYBRID_V31_SPEC.name] = True
        model = v29.build_candidate(HYBRID_V31_SPEC, shape, args, deps, device)
        checkpoint = Path(str(result["checkpoint"]))
        model.load_state_dict(state_from_checkpoint(checkpoint), strict=True)
        return FUSION, model.eval(), str(checkpoint)

    raw28 = read_json(Path(task["target_v28_results"]))
    model_name = TRANSFORMER if label == "transformer_v28" else MAMBA2
    shape = v23.Shape(**raw28["shapes"][model_name])
    model = v25.build_model_v25(model_name, shape, args, deps, device).eval()
    row = result_row_for_model(raw28, model_name)
    checkpoint = first_existing([str(row.get("checkpoint", "")), str(row.get("bf16_export", ""))])
    model.load_state_dict(state_from_checkpoint(checkpoint), strict=True)
    return model_name, model, str(checkpoint)


# =====================================================================================
# Exact runtime patches / prefill runner
# =====================================================================================


def install_cached_landmark_metadata(model: nn.Module) -> int:
    """Cache only length/device-dependent tensors; weights and attention math stay unchanged."""
    patched = 0
    v20 = v23.v21.v20
    for block in model.blocks:
        target = block.refresh if isinstance(block, v29.RefreshWithEditor) else block
        if not isinstance(target, v23.v21.FusionRefreshBlockV21):
            continue
        attn = target.attn
        attn._v32_landmark_cache = {}

        def cached_landmark(self, q, latent, cos, sin, _v20=v20):
            b, _, length, _ = q.shape
            chunk = self.landmark_chunk
            full_chunks = length // chunk
            if full_chunks == 0:
                return q.new_zeros((b, self.q_heads, length, self.head_dim))
            landmark_latent = latent[:, : full_chunks * chunk].reshape(
                b, full_chunks, chunk, self.latent_dim
            ).mean(dim=2)
            kup = self.kv_up(landmark_latent)
            lk_raw, lv_raw = kup.split(self.kv_heads * self.head_dim, dim=-1)
            lk = self._reshape_kv(lk_raw)
            lv = self._reshape_kv(lv_raw)
            key = (str(q.device), int(length), int(full_chunks), int(chunk))
            meta = self._v32_landmark_cache.get(key)
            if meta is None:
                positions = torch.arange(
                    chunk - 1, full_chunks * chunk, chunk,
                    device=q.device, dtype=torch.long,
                )
                token_chunks = torch.arange(length, device=q.device) // chunk
                landmark_ids = torch.arange(full_chunks, device=q.device)
                allowed_real = landmark_ids[None, :] < token_chunks[:, None]
                allowed = torch.cat((
                    torch.ones((length, 1), device=q.device, dtype=torch.bool),
                    allowed_real,
                ), dim=1)[None, None]
                meta = (positions, allowed)
                self._v32_landmark_cache[key] = meta
            positions, allowed = meta
            lk = _v20.apply_rope(lk, cos[:, :, positions], sin[:, :, positions])
            null_k = lk.new_zeros((b, self.kv_heads, 1, self.head_dim))
            null_v = lv.new_zeros((b, self.kv_heads, 1, self.head_dim))
            lk = torch.cat((null_k, lk), dim=2)
            lv = torch.cat((null_v, lv), dim=2)
            return _v20.sdpa_gqa(q, lk, lv, causal=False, attn_mask=allowed)

        attn._landmark_attention = types.MethodType(cached_landmark, attn)
        patched += 1
    return patched


class PrefillModule(nn.Module):
    def __init__(self, model_name: str, model: nn.Module) -> None:
        super().__init__()
        self.model_name = str(model_name)
        self.model = model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        model = self.model
        if self.model_name == FUSION:
            h = model.emb(tokens)
            model._patch_aux = h.new_zeros(())
            for i, block in enumerate(model.blocks):
                h = block(h)
                if i == model.patch_position and model.softpatch is not None:
                    h = model.softpatch(h, tokens)
                    model._patch_aux = model.softpatch.last_aux
            h = model.final_norm(h)
        elif self.model_name == TRANSFORMER:
            h = model.emb(tokens)
            for block in model.blocks:
                h = block(h)
            h = model.final_norm(h)
        elif self.model_name == MAMBA2:
            h = model.emb(tokens).to(model.activation_dtype)
            for block in model.blocks:
                h = block(h)
            h = model.norm(h)
        else:
            raise KeyError(self.model_name)
        return model.lm_head(h[:, -1:, :])


def maybe_compile(module: nn.Module, mode: str) -> nn.Module:
    if mode == "eager":
        return module
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile unavailable")
    return torch.compile(module, mode=mode, fullgraph=False, dynamic=False)


def make_random_tokens(batch: int, context: int, vocab: int,
                       seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + int(context) * 131 + int(batch) * 17)
    return torch.randint(0, vocab, (batch, context), device=device, generator=generator)


def component_profile(model: nn.Module, tokens: torch.Tensor, amp: str,
                      steps: int) -> Dict[str, object]:
    device = tokens.device
    categories: Dict[str, float] = {k: 0.0 for k in (
        "embedding", "field_blocks", "mamba_blocks", "refresh_blocks",
        "softpatch", "final_norm_head",
    )}
    counts: Dict[str, int] = {k: 0 for k in categories}
    with torch.inference_mode(), v23.amp_ctx(device, amp):
        # Untimed warmup.
        runner = PrefillModule(FUSION, model)
        _ = runner(tokens)
        sync(device)
        for _ in range(max(1, int(steps))):
            h = None
            events: List[Tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
            def record(category: str, fn):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                out = fn()
                end.record()
                events.append((category, start, end))
                return out
            h = record("embedding", lambda: model.emb(tokens))
            for i, block in enumerate(model.blocks):
                if v29.is_official_mamba_block(block):
                    category = "mamba_blocks"
                elif v29.is_refresh_block(block):
                    category = "refresh_blocks"
                else:
                    category = "field_blocks"
                h = record(category, lambda b=block, z=h: b(z))
                if i == model.patch_position and model.softpatch is not None:
                    h = record("softpatch", lambda z=h: model.softpatch(z, tokens))
            _ = record("final_norm_head", lambda z=h: model.lm_head(model.final_norm(z)[:, -1:, :]))
            sync(device)
            for category, start, end in events:
                categories[category] += float(start.elapsed_time(end))
                counts[category] += 1
    total = sum(categories.values())
    return {
        "milliseconds": categories,
        "percent": {k: 100.0 * v / max(total, 1e-9) for k, v in categories.items()},
        "event_counts": counts,
        "total_ms": total,
        "steps": max(1, int(steps)),
    }


def worker_main(task_path: Path) -> int:
    task = read_json(task_path)
    output = Path(task["output"])
    result: Dict[str, object] = {
        "status": "error",
        "task": task,
        "error": "worker did not complete",
    }
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 CUDA GPU required")
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        torch.set_float32_matmul_precision("high")
        args = worker_base_args(task)
        v31.configure(args)
        canonical_path, canonical_sha, deps = v27.load_dependencies(args)
        if canonical_sha != EXPECTED_CANONICAL_SHA256:
            raise RuntimeError(f"canonical SHA mismatch: {canonical_sha}")
        model_name, model, checkpoint = load_worker_model(task, args, deps, device)
        patched = install_cached_landmark_metadata(model) if task.get("cache_landmark") else 0
        runner: nn.Module = PrefillModule(model_name, model).eval()
        compile_mode = str(task.get("compile_mode", "eager"))
        runner = maybe_compile(runner, compile_mode)
        context = int(task["context"])
        batch = int(task["batch"])
        x = make_random_tokens(batch, context, int(args.vocab_size), int(task["seed"]), device)
        warmup = int(task.get("warmup", 1))
        steps = int(task.get("steps", 3))
        with torch.inference_mode(), v23.amp_ctx(device, args.amp):
            logits = None
            for _ in range(warmup):
                logits = runner(x)
                sync(device)
            baseline_gib = torch.cuda.memory_allocated(device) / 2**30
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            for _ in range(steps):
                logits = runner(x)
            sync(device)
            elapsed = time.perf_counter() - start
        assert logits is not None
        sample = logits[0, 0, :128].detach().float().cpu().tolist()
        peak = torch.cuda.max_memory_allocated(device) / 2**30
        free = torch.cuda.mem_get_info(device)[0] / 2**30
        status = "ok" if free >= float(task.get("min_free_gib", 1.0)) else "low_headroom"
        result = {
            "status": status,
            "model_label": task["model_label"],
            "model_name": model_name,
            "checkpoint": checkpoint,
            "context": context,
            "batch": batch,
            "tokens_per_step": batch * context,
            "seconds": elapsed,
            "steps": steps,
            "tokens_per_second": steps * batch * context / max(elapsed, 1e-9),
            "sequences_per_second": steps * batch / max(elapsed, 1e-9),
            "peak_gib": peak,
            "baseline_gib": baseline_gib,
            "free_after_gib": free,
            "compile_mode": compile_mode,
            "field_chunk": int(args.field_chunk),
            "triton_block_c": int(args.triton_block_c),
            "triton_chunk_t": int(args.triton_chunk_t),
            "cache_landmark": bool(task.get("cache_landmark")),
            "cached_refreshes": patched,
            "logit_sample": sample,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "error": "",
        }
        if task.get("profile_components") and model_name == FUSION:
            result["component_profile"] = component_profile(
                model, x, args.amp, int(task.get("profile_steps", 3))
            )
    except torch.cuda.OutOfMemoryError as exc:
        result["status"] = "oom"
        result["error"] = str(exc).splitlines()[0]
    except Exception as exc:
        result["status"] = "error"
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
    with contextlib.suppress(Exception):
        atomic_json(output, result)
    return 0 if result.get("status") in {"ok", "low_headroom", "oom"} else 2


# =====================================================================================
# Process-isolated benchmark controller
# =====================================================================================


def standard_batch_candidates(max_batch: int) -> List[int]:
    standards = [64, 48, 40, 32, 24, 20, 16, 12, 10, 8, 6, 5, 4, 3, 2, 1]
    out = [x for x in standards if x <= max_batch]
    if max_batch not in out:
        out.insert(0, max_batch)
    return sorted(set(out), reverse=True)


def package_executable() -> str:
    archive = os.environ.get("FIELD_FUSION_V32_ARCHIVE", "")
    if archive and Path(archive).is_file():
        return archive
    return str(Path(__file__).resolve())


def run_worker(task: MutableMapping[str, object], workdir: Path,
               timeout: int) -> Dict[str, object]:
    workdir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()[:16]
    task_path = workdir / f"task_{key}.json"
    output = workdir / f"result_{key}.json"
    task["output"] = str(output)
    atomic_json(task_path, task)
    if output.is_file():
        return read_json(output)
    env = dict(os.environ)
    env["CUDA_LAUNCH_BLOCKING"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    command = [sys.executable, package_executable(), "--worker-task", str(task_path)]
    try:
        completed = subprocess.run(
            command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=int(timeout), check=False,
        )
        stdout_path = workdir / f"stdout_{key}.txt"
        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        if output.is_file():
            row = read_json(output)
        else:
            row = {
                "status": "crash",
                "error": f"worker exited {completed.returncode} without result",
                "stdout": str(stdout_path),
            }
        row["worker_returncode"] = completed.returncode
        row["worker_stdout"] = str(stdout_path)
        return row
    except subprocess.TimeoutExpired as exc:
        stdout_path = workdir / f"stdout_{key}.txt"
        text = exc.stdout or ""
        if isinstance(text, bytes):
            text = text.decode(errors="replace")
        stdout_path.write_text(text, encoding="utf-8")
        return {"status": "timeout", "error": f"timeout after {timeout}s", "worker_stdout": str(stdout_path)}


def common_worker_task(args, root: Path) -> Dict[str, object]:
    return {
        "v32_version": VERSION,
        "scratch_outdir": str(root / "worker_scratch"),
        "target_v31_root": str(args.target_v31_root),
        "v31_hybrid_result": str(locate_v31_hybrid_result(Path(args.target_v31_root))),
        "target_v28_results": str(args.target_v28_results),
        "target_v28_starts": str(args.target_v28_starts),
        "warmup": int(args.infer_warmup),
        "steps": int(args.infer_steps),
        "min_free_gib": float(args.infer_min_free_gib),
        "seed": int(args.infer_seed),
        "compile_mode": "eager",
        "field_chunk": int(args.field_chunk),
        "triton_block_c": int(args.triton_block_c),
        "triton_chunk_t": int(args.triton_chunk_t),
        "cache_landmark": False,
    }


def benchmark_isolated(args, root: Path) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    workdir = root / "workers" / "inference"
    base = common_worker_task(args, root)
    labels = ("field_hybrid_v31", "transformer_v28", "mamba2_v28")
    rows: List[Dict[str, object]] = []
    selected: Dict[str, Dict[str, Dict[str, object]]] = {x: {} for x in labels}
    for label in labels:
        for context in map(int, args.infer_contexts):
            max_by_tokens = max(1, int(args.infer_max_batch_tokens) // context)
            max_batch = min(int(args.infer_max_batch), max_by_tokens)
            choice = None
            for batch in standard_batch_candidates(max_batch):
                task = dict(base)
                task.update(model_label=label, context=context, batch=batch)
                log(f"[isolated/search] {label} ctx={context} batch={batch}")
                row = run_worker(task, workdir, args.infer_timeout_seconds)
                row.update(label=label, context=context, batch=batch, comparison="max_safe_search")
                rows.append(row)
                if row.get("status") == "ok":
                    choice = row
                    break
                if row.get("status") == "low_headroom":
                    continue
            if choice is None:
                valid = [x for x in rows if x.get("label") == label and x.get("context") == context
                         and x.get("status") in {"ok", "low_headroom"}]
                if valid:
                    choice = valid[-1]
            if choice is None:
                log(f"[isolated/result] {label} ctx={context} NO_SAFE_BATCH")
            else:
                choice["selected"] = True
                selected[label][str(context)] = choice
                log(
                    f"[isolated/result] {label} ctx={context} batch={choice['batch']} "
                    f"tok/s={choice['tokens_per_second']:,.0f} peak={choice['peak_gib']:.2f}G"
                )

    matched_batches: Dict[str, int] = {}
    for context in map(int, args.infer_contexts):
        key = str(context)
        available = [selected[x].get(key) for x in labels]
        if all(available):
            matched_batches[key] = min(int(x["batch"]) for x in available if x is not None)
    for label in labels:
        for context in map(int, args.infer_contexts):
            key = str(context)
            if key not in matched_batches:
                continue
            batch = matched_batches[key]
            task = dict(base)
            task.update(model_label=label, context=context, batch=batch)
            log(f"[isolated/matched] {label} ctx={context} batch={batch}")
            row = run_worker(task, workdir / "matched", args.infer_timeout_seconds)
            row.update(label=label, context=context, batch=batch, comparison="matched_batch")
            row["selected_matched"] = row.get("status") in {"ok", "low_headroom"}
            rows.append(row)
            if row["selected_matched"]:
                log(
                    f"[isolated/matched-result] {label} ctx={context} batch={batch} "
                    f"tok/s={row['tokens_per_second']:,.0f} peak={row['peak_gib']:.2f}G"
                )

    matched = {(str(r["label"]), str(r["context"])): r for r in rows if r.get("selected_matched")}
    comparisons: List[Dict[str, object]] = []
    for context in map(int, args.infer_contexts):
        key = str(context)
        f = matched.get(("field_hybrid_v31", key))
        t = matched.get(("transformer_v28", key))
        m = matched.get(("mamba2_v28", key))
        if not (f and t and m):
            continue
        ftps, ttps, mtps = map(float, (f["tokens_per_second"], t["tokens_per_second"], m["tokens_per_second"]))
        fmem, tmem, mmem = map(float, (f["peak_gib"], t["peak_gib"], m["peak_gib"]))
        comparisons.append({
            "context": context,
            "batch": int(f["batch"]),
            "field_tokens_per_second": ftps,
            "transformer_tokens_per_second": ttps,
            "mamba_tokens_per_second": mtps,
            "field_over_transformer_speed": ftps / max(ttps, 1e-9),
            "field_over_mamba_speed": ftps / max(mtps, 1e-9),
            "field_peak_gib": fmem,
            "transformer_peak_gib": tmem,
            "mamba_peak_gib": mmem,
            "field_over_transformer_memory": fmem / max(tmem, 1e-9),
            "field_over_mamba_memory": fmem / max(mmem, 1e-9),
            "field_vs_transformer_pareto": ftps >= ttps and fmem <= tmem,
            "field_vs_mamba_pareto": ftps >= mtps and fmem <= mmem,
        })
    decision = {
        "mode": "isolated_backbone_prefill_plus_last_token_head",
        "note": "Matched-batch full-context prefill; not incremental autoregressive decode.",
        "matched_batches": matched_batches,
        "comparisons": comparisons,
        "field_beats_transformer_speed_all": bool(comparisons) and all(x["field_over_transformer_speed"] > 1 for x in comparisons),
        "field_beats_mamba_speed_all": bool(comparisons) and all(x["field_over_mamba_speed"] > 1 for x in comparisons),
        "field_pareto_transformer_all": bool(comparisons) and all(x["field_vs_transformer_pareto"] for x in comparisons),
        "field_pareto_mamba_all": bool(comparisons) and all(x["field_vs_mamba_pareto"] for x in comparisons),
    }
    atomic_json(root / "inference_rows.json", rows)
    atomic_json(root / "inference_decision.json", decision)
    return rows, decision


KERNEL_VARIANTS: Tuple[Dict[str, object], ...] = (
    {"name": "eager_current", "field_chunk": 32, "triton_block_c": 16, "triton_chunk_t": 64, "cache_landmark": False, "compile_mode": "eager"},
    {"name": "eager_cached_landmark", "field_chunk": 32, "triton_block_c": 16, "triton_chunk_t": 64, "cache_landmark": True, "compile_mode": "eager"},
    {"name": "field_chunk16", "field_chunk": 16, "triton_block_c": 16, "triton_chunk_t": 64, "cache_landmark": True, "compile_mode": "eager"},
    {"name": "field_chunk64", "field_chunk": 64, "triton_block_c": 16, "triton_chunk_t": 64, "cache_landmark": True, "compile_mode": "eager"},
    {"name": "blockc32_t64", "field_chunk": 32, "triton_block_c": 32, "triton_chunk_t": 64, "cache_landmark": True, "compile_mode": "eager"},
    {"name": "blockc16_t128", "field_chunk": 32, "triton_block_c": 16, "triton_chunk_t": 128, "cache_landmark": True, "compile_mode": "eager"},
    {"name": "blockc32_t128", "field_chunk": 32, "triton_block_c": 32, "triton_chunk_t": 128, "cache_landmark": True, "compile_mode": "eager"},
    {"name": "compile_default", "field_chunk": 32, "triton_block_c": 16, "triton_chunk_t": 64, "cache_landmark": True, "compile_mode": "default"},
    {"name": "compile_reduce_overhead", "field_chunk": 32, "triton_block_c": 16, "triton_chunk_t": 64, "cache_landmark": True, "compile_mode": "reduce-overhead"},
    {"name": "compile_max_autotune", "field_chunk": 32, "triton_block_c": 16, "triton_chunk_t": 64, "cache_landmark": True, "compile_mode": "max-autotune-no-cudagraphs"},
)


def max_abs_sample(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return float("inf")
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


def kernel_sweep(args, root: Path, infer_decision: Mapping[str, object]) -> Dict[str, object]:
    context = int(args.kernel_context)
    matched_batches = infer_decision.get("matched_batches", {})
    batch = int(matched_batches.get(str(context), 1))
    base_task = common_worker_task(args, root)
    base_task.update(model_label="field_hybrid_v31", context=context, batch=batch,
                     warmup=max(2, int(args.infer_warmup)), steps=max(3, int(args.infer_steps)))
    rows: List[Dict[str, object]] = []
    baseline: Optional[Dict[str, object]] = None
    workdir = root / "workers" / "kernel"
    for variant in KERNEL_VARIANTS:
        task = dict(base_task)
        task.update(variant)
        task.pop("name", None)
        task["profile_components"] = variant["name"] == "eager_current"
        task["profile_steps"] = int(args.kernel_profile_steps)
        log(f"[kernel] {variant['name']} ctx={context} batch={batch}")
        row = run_worker(task, workdir, args.infer_timeout_seconds)
        row["variant"] = variant["name"]
        rows.append(row)
        if variant["name"] == "eager_current" and row.get("status") in {"ok", "low_headroom"}:
            baseline = row
    if baseline is None:
        decision = {"status": "no_baseline", "rows": rows, "winner": None}
        atomic_json(root / "kernel_decision.json", decision)
        return decision
    for row in rows:
        if row.get("status") not in {"ok", "low_headroom"}:
            row["exact"] = False
            continue
        diff = max_abs_sample(baseline.get("logit_sample", []), row.get("logit_sample", []))
        row["max_abs_logit_sample"] = diff
        row["exact"] = diff <= float(args.kernel_logit_tolerance)
        row["speed_ratio"] = float(row["tokens_per_second"]) / max(float(baseline["tokens_per_second"]), 1e-9)
        row["memory_ratio"] = float(row["peak_gib"]) / max(float(baseline["peak_gib"]), 1e-9)
        row["eligible"] = (
            row["exact"]
            and row["speed_ratio"] >= float(args.kernel_min_speedup)
            and row["memory_ratio"] <= float(args.kernel_max_memory_ratio)
        )
    eligible = [x for x in rows if x.get("eligible")]
    winner = max(eligible, key=lambda x: float(x["tokens_per_second"])) if eligible else baseline
    decision = {
        "status": "ok",
        "context": context,
        "batch": batch,
        "baseline_variant": "eager_current",
        "winner": winner.get("variant"),
        "winner_speed_ratio": float(winner.get("tokens_per_second", 0.0)) / max(float(baseline["tokens_per_second"]), 1e-9),
        "winner_memory_ratio": float(winner.get("peak_gib", 0.0)) / max(float(baseline["peak_gib"]), 1e-9),
        "promote_runtime_patch": winner.get("variant") != "eager_current",
        "rows": rows,
        "component_profile": baseline.get("component_profile"),
    }
    atomic_json(root / "kernel_rows.json", rows)
    atomic_json(root / "kernel_decision.json", decision)
    return decision


# =====================================================================================
# Paired quality gap screen
# =====================================================================================


def make_starts(count: int, upper: int, seed: int, path: Path) -> np.ndarray:
    return v28.make_or_load_starts(path, count, upper, seed)


def audit_prefix(starts: np.ndarray, args, root: Path) -> Dict[str, object]:
    old_path = Path(args.target_v28_starts)
    if not old_path.is_file():
        raise FileNotFoundError(old_path)
    old = np.load(old_path)
    if len(old) < len(starts):
        raise RuntimeError(f"v28 starts too short: {len(old)} < {len(starts)}")
    equal = bool(np.array_equal(starts, old[:len(starts)]))
    row = {
        "v28_path": str(old_path),
        "count": int(len(starts)),
        "prefix_equal": equal,
        "v28_prefix_sha256": hashlib.sha256(old[:len(starts)].tobytes()).hexdigest(),
        "v32_sha256": hashlib.sha256(starts.tobytes()).hexdigest(),
    }
    if not equal:
        raise AssertionError("v32 paired starts do not match v28 prefix")
    atomic_json(root / "paired_prefix_audit.json", row)
    return row


def run_gap_ablation(args, root: Path, device: torch.device) -> Dict[str, object]:
    arms = selected_arms(args)
    specs = tuple(SPEC_BY_NAME[x.name] for x in arms)
    v29.CANDIDATES = specs
    v29.VERSION = VERSION
    v26.VERSION = VERSION
    v29.configure(args)
    canonical_path, canonical_sha, deps = v27.load_dependencies(args)
    if canonical_sha != EXPECTED_CANONICAL_SHA256:
        raise RuntimeError(f"canonical SHA mismatch: {canonical_sha}")
    base_shape, shapes, accounting = v29.solve_candidate_shapes(args, deps)
    atomic_json(root / "gap_component_accounting.json", accounting)
    v29.architecture_audit(specs, shapes, args, deps, device, root / "gap_architecture")
    v29.causality_and_backward_preflight(specs, shapes, args, deps, device, root / "gap_preflight")

    raw_rows = v23.core.load_raw_rows(args.cache_dir, args.data_frac)
    tokenizer = v23.base.copy_or_train_tokenizer(
        root, raw_rows[0], args.vocab_size,
        args.tokenizer_min_frequency, args.tokenizer_source,
    )
    train_c, val_c, test_c = v23.core.save_or_load_corpora(root, tokenizer, raw_rows)
    train = v23.core.place_tokens(train_c.tokens, device, args.data_device, "train")
    val = v23.core.place_tokens(val_c.tokens, device, args.data_device, "validation")
    test = v23.core.place_tokens(test_c.tokens, device, args.data_device, "test")
    sequences = int(args.ablation_token_budget) // int(args.train_seq)
    starts = make_starts(
        sequences, len(train) - int(args.train_seq) - 1,
        int(args.data_seed), root / "gap_paired_starts.npy",
    )
    prefix = audit_prefix(starts, args, root)

    results: Dict[str, v29.ScreenResult] = {}
    original_stable = float(args.wsd_stable_fraction)
    for arm in arms:
        spec = SPEC_BY_NAME[arm.name]
        args.wsd_stable_fraction = float(arm.stable_fraction)
        log("=" * 220)
        log(
            f"GAP ARM: {arm.name} stable={arm.stable_fraction:.2f} "
            f"mambaLR={arm.mamba_lr_scale:.2f} refreshLR={arm.refresh_lr_scale:.2f} "
            f"positions={arm.mamba_positions}"
        )
        results[arm.name] = v29.train_candidate(
            spec, shapes[arm.name], args, deps,
            train, val_c, val, test_c, test, starts, root, device,
        )
        atomic_json(root / "gap_results.json", {k: asdict(v) for k, v in results.items()})
    args.wsd_stable_fraction = original_stable

    baseline = results["gap_ref_stable70"]
    ranked = sorted(results.values(), key=lambda x: (x.final_validation["nll"], x.final_test["nll"]))
    top = ranked[:2]
    contexts: Dict[str, Dict[str, Dict[str, float]]] = {}
    args.long_contexts = list(map(int, args.ablation_long_contexts))
    args.long_context_score_tokens = 128
    args.long_context_windows = int(args.ablation_long_windows)
    for result in top:
        arm = ARM_BY_NAME[result.candidate]
        args.wsd_stable_fraction = float(arm.stable_fraction)
        model = v29.load_model_from_result(
            SPEC_BY_NAME[result.candidate], shapes[result.candidate], result,
            args, deps, device,
        )
        contexts[result.candidate] = v29.long_context_eval(model, test, args, device)
        del model
        clear_cuda()
    args.wsd_stable_fraction = original_stable

    rows = []
    for result in ranked:
        ctx = contexts.get(result.candidate, {})
        c2 = float(ctx.get("2048", {}).get("nll", float("nan")))
        c64 = float(ctx.get("65536", {}).get("nll", float("nan")))
        drift = c64 - c2 if math.isfinite(c2) and math.isfinite(c64) else None
        gain_val = float(baseline.final_validation["nll"]) - float(result.final_validation["nll"])
        gain_test = float(baseline.final_test["nll"]) - float(result.final_test["nll"])
        speed_ratio = float(result.tokens_per_second) / max(float(baseline.tokens_per_second), 1e-9)
        memory_ratio = float(result.peak_gib) / max(float(baseline.peak_gib), 1e-9)
        eligible = (
            result.candidate != baseline.candidate
            and gain_val >= float(args.ablation_min_gain)
            and gain_test >= 0.0
            and speed_ratio >= float(args.ablation_min_speed_ratio)
            and memory_ratio <= float(args.ablation_max_memory_ratio)
            and (drift is None or drift <= float(args.ablation_max_context_drift))
        )
        rows.append({
            "candidate": result.candidate,
            "validation_nll": result.final_validation["nll"],
            "test_nll": result.final_test["nll"],
            "tokens_per_second": result.tokens_per_second,
            "peak_gib": result.peak_gib,
            "validation_gain_vs_baseline": gain_val,
            "test_gain_vs_baseline": gain_test,
            "speed_ratio_vs_baseline": speed_ratio,
            "memory_ratio_vs_baseline": memory_ratio,
            "context_2k_to_64k": drift,
            "eligible": eligible,
            "arm": asdict(ARM_BY_NAME[result.candidate]),
        })
    eligible = [x for x in rows if x["eligible"]]
    winner = min(eligible, key=lambda x: (x["validation_nll"], x["test_nll"])) if eligible else None
    decision = {
        "action": "PROMOTE_GAP_WINNER_TO_49M" if winner else "NO_GAP_ARM_PROMOTED",
        "winner": None if winner is None else winner["candidate"],
        "baseline": baseline.candidate,
        "rows": rows,
        "paired_prefix": prefix,
        "contexts": contexts,
    }
    atomic_json(root / "gap_decision.json", decision)
    return decision


# =====================================================================================
# Summary / main
# =====================================================================================


def make_summary(args, infer_decision, kernel_decision, gap_decision) -> str:
    width = 240
    lines = [
        "=" * width,
        "FIELD-FUSION v32 — 16K–64K ISOLATED SYSTEMS / KERNEL AUDIT / QUALITY GAP SCREEN",
        "=" * width,
        "No Transformer or Mamba model was retrained. Frozen v28 checkpoints are used only for inference.",
        "Inference mode=full-context backbone prefill + one last-token vocabulary projection; not incremental decode.",
        "",
        "MATCHED-BATCH PREFILL",
        f"{'ctx':>8s} {'batch':>7s} {'Field tok/s':>14s} {'TF tok/s':>14s} {'Mamba tok/s':>14s} {'F/TF':>8s} {'F/M':>8s} {'FieldGB':>9s} {'TFGB':>9s} {'MambaGB':>9s}",
    ]
    for row in (infer_decision or {}).get("comparisons", []):
        lines.append(
            f"{row['context']:8,d} {row['batch']:7d} {row['field_tokens_per_second']:14,.0f} "
            f"{row['transformer_tokens_per_second']:14,.0f} {row['mamba_tokens_per_second']:14,.0f} "
            f"{row['field_over_transformer_speed']:8.3f} {row['field_over_mamba_speed']:8.3f} "
            f"{row['field_peak_gib']:9.2f} {row['transformer_peak_gib']:9.2f} {row['mamba_peak_gib']:9.2f}"
        )
    if kernel_decision:
        lines += [
            "",
            "KERNEL / RUNTIME SWEEP",
            f"winner={kernel_decision.get('winner')} speed={kernel_decision.get('winner_speed_ratio', float('nan')):.3f}x "
            f"memory={kernel_decision.get('winner_memory_ratio', float('nan')):.3f}x promote={kernel_decision.get('promote_runtime_patch')}",
        ]
        profile = kernel_decision.get("component_profile") or {}
        for name, pct in sorted((profile.get("percent") or {}).items(), key=lambda x: -x[1]):
            lines.append(f"profile {name:24s} {pct:7.2f}%")
    if gap_decision:
        lines += [
            "",
            "25.165824M PAIRED QUALITY GAP SCREEN",
            f"{'candidate':30s} {'val':>9s} {'test':>9s} {'dVal':>9s} {'tok/s':>11s} {'speed':>8s} {'GB':>7s} {'eligible':>9s}",
        ]
        for row in gap_decision.get("rows", []):
            lines.append(
                f"{row['candidate']:30s} {row['validation_nll']:9.5f} {row['test_nll']:9.5f} "
                f"{row['validation_gain_vs_baseline']:+9.5f} {row['tokens_per_second']:11,.0f} "
                f"{row['speed_ratio_vs_baseline']:8.3f} {row['peak_gib']:7.2f} {str(row['eligible']):>9s}"
            )
        lines += [
            "",
            f"gap_action={gap_decision.get('action')}",
            f"gap_winner={gap_decision.get('winner')}",
        ]
    lines += ["", "No follow-up or rival training is launched automatically.", "=" * width]
    return "\n".join(lines) + "\n"


def validate_paths(args) -> None:
    if not Path(args.target_v28_results).is_file():
        raise FileNotFoundError(args.target_v28_results)
    if not Path(args.target_v28_starts).is_file():
        raise FileNotFoundError(args.target_v28_starts)
    locate_v31_hybrid_result(Path(args.target_v31_root))


def main() -> None:
    args = parse_args()
    validate_paths(args)
    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "args.json", vars(args))

    infer_decision: Dict[str, object] = {}
    kernel_decision: Dict[str, object] = {}
    gap_decision: Dict[str, object] = {}

    # The controller intentionally runs all subprocess-isolated GPU work before
    # initializing CUDA in this process.
    if args.run_inference:
        log("=" * 220)
        log("PHASE A — PROCESS-ISOLATED 16K/32K/64K PREFILL")
        _, infer_decision = benchmark_isolated(args, root)
    if args.run_kernel_sweep:
        if not infer_decision:
            infer_path = root / "inference_decision.json"
            if not infer_path.is_file():
                raise RuntimeError("kernel sweep needs inference_decision.json")
            infer_decision = read_json(infer_path)
        log("=" * 220)
        log("PHASE B — EXACT KERNEL / RUNTIME SWEEP")
        kernel_decision = kernel_sweep(args, root, infer_decision)

    if args.run_gap_ablation:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16-capable CUDA GPU required for gap ablation")
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        torch.set_float32_matmul_precision("high")
        log("=" * 220)
        log("PHASE C — PAIRED 25.165824M QUALITY GAP ABLATION")
        gap_decision = run_gap_ablation(args, root, device)

    payload = {
        "version": VERSION,
        "args": vars(args),
        "inference_decision": infer_decision,
        "kernel_decision": kernel_decision,
        "gap_decision": gap_decision,
        "v31_hybrid_result": str(locate_v31_hybrid_result(Path(args.target_v31_root))),
        "v28_results": str(args.target_v28_results),
        "v28_results_sha256": sha256(Path(args.target_v28_results)),
    }
    atomic_json(root / "results.json", payload)
    text = make_summary(args, infer_decision, kernel_decision, gap_decision)
    atomic_text(root / "summary.txt", text)
    log(text)


if __name__ == "__main__":
    if "--worker-task" in sys.argv:
        index = sys.argv.index("--worker-task")
        sys.exit(worker_main(Path(sys.argv[index + 1])))
    main()
