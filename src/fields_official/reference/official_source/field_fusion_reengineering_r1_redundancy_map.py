#!/usr/bin/env python3
"""FIELD-FUSION R1 — structural redundancy map for the promoted v35r1 system.

Goal
----
Measure whether the classical support modules accumulated around the native
Field are still necessary.  Every arm is trained from scratch with paired data
windows and is parameter-matched near 300M parameters.

Arms
----
* r1_control_16f_4m_4r: exact v35r1 topology, including late Field FFN reallocation.
* r1_18f_2m_4r: keep one Mamba-2 editor per half (block positions 10 and 22).
* r1_18f_4m_2r: keep all four Mamba-2 editors, but only two refresh stations
  (blocks 11 and 23); removed refreshes become native Field blocks.
* r1_20f_2m_2r: combine both reductions; 20 native Fields, two Mamba-2 editors,
  and two refresh stations.

This screen never launches a 50M confirmation automatically.  It emits
ADVANCE_R1_ARM_TO_WIKI5_50M only when an arm preserves quality and improves the
speed/memory frontier, or clearly improves quality without an excessive systems
cost.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

import field_fusion_structural_gap_v35r1 as v35
import field_fusion_lr_calibration_v34 as v34
import field_fusion_gap_kernel_v32 as v32
import field_fusion_delta_quality_ablation_v29 as v29
import field_fusion_recipe_memory_v27 as v27
import field_fusion_wiki100_canonical_v23 as v23
import field_fusion_wiki100_mamba2_v25 as v25

VERSION = 3701
BASELINE_NAME = "r1_control_16f_4m_4r"
REFRESH_POSITIONS = (5, 11, 17, 23)
CANONICAL_MAMBA_POSITIONS = (4, 10, 16, 22)


@dataclass(frozen=True)
class R1Arm:
    name: str
    field_count: int
    mamba_positions: Tuple[int, ...]
    refresh_positions: Tuple[int, ...]
    description: str


R1_ARMS: Tuple[R1Arm, ...] = (
    R1Arm(
        BASELINE_NAME,
        16,
        CANONICAL_MAMBA_POSITIONS,
        REFRESH_POSITIONS,
        "Exact promoted v35r1 topology: 16 Field, 4 Mamba-2, 4 refresh.",
    ),
    R1Arm(
        "r1_18f_2m_4r",
        18,
        (10, 22),
        REFRESH_POSITIONS,
        "Remove the stage-1 and stage-3 Mamba editors; retain one editor per half.",
    ),
    R1Arm(
        "r1_18f_4m_2r",
        18,
        CANONICAL_MAMBA_POSITIONS,
        (11, 23),
        "Replace refresh stations 0 and 2 with native Field blocks.",
    ),
    R1Arm(
        "r1_20f_2m_2r",
        20,
        (10, 22),
        (11, 23),
        "Lean hybrid: 20 Field, 2 Mamba-2, 2 refresh.",
    ),
)
R1_BY_NAME: Dict[str, R1Arm] = {arm.name: arm for arm in R1_ARMS}

# v35's optimizer only requires these structural fields.  The custom builder
# below owns the topology changes.
STRUCTURAL_ARMS: Tuple[v35.StructuralArm, ...] = tuple(
    v35.StructuralArm(
        name=arm.name,
        description=arm.description,
        kind="r1_redundancy",
        stable_fraction=0.70,
        mamba_lr_scale=1.10,
        refresh_lr_scale=1.35,
        mamba_positions=arm.mamba_positions,
        late_ff_delta=256,
    )
    for arm in R1_ARMS
)
STRUCT_BY_NAME = {arm.name: arm for arm in STRUCTURAL_ARMS}
SPEC_BY_NAME = {
    arm.name: v29.CandidateSpec(
        arm.name,
        arm.description,
        refresh_1024x2=True,
        mamba_replace=arm.mamba_positions,
    )
    for arm in R1_ARMS
}


def log(value: object = "") -> None:
    print(str(value), flush=True)


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


def nparams(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def classify_block(block: nn.Module) -> str:
    if v29.is_official_mamba_block(block):
        return "mamba"
    if v29.is_refresh_block(block):
        return "refresh"
    return "field"


def architecture_counts(model: nn.Module) -> Dict[str, object]:
    kinds = [classify_block(block) for block in model.blocks]
    return {
        "field": kinds.count("field"),
        "mamba": kinds.count("mamba"),
        "refresh": kinds.count("refresh"),
        "sequence": kinds,
        "mamba_positions": [i for i, kind in enumerate(kinds) if kind == "mamba"],
        "refresh_positions": [i for i, kind in enumerate(kinds) if kind == "refresh"],
    }


_BUILD_PRE_R1 = v29.build_candidate
_SIGNATURE_PRE_R1 = v29.checkpoint_signature


def _new_native_field(shape: v23.Shape, args, deps, device: torch.device) -> nn.Module:
    canonical = deps[2]
    block = canonical.FieldBlock(
        int(shape.dim),
        "triton",
        int(args.field_chunk),
        int(args.triton_block_c),
        int(args.triton_chunk_t),
        int(shape.ff_hidden),
    )
    return block.to(device)


def _install_generalized_late_reallocation(
    model: nn.Module,
    shape: v23.Shape,
    device: torch.device,
    delta: int = 256,
) -> Dict[str, object]:
    native = [
        i for i, block in enumerate(model.blocks)
        if classify_block(block) == "field"
    ]
    if len(native) < 8:
        raise AssertionError(f"late reallocation needs at least 8 Field blocks, got {native}")
    early_hidden = int(shape.ff_hidden) - int(delta)
    late_hidden = int(shape.ff_hidden) + int(delta)
    if early_hidden <= 0 or early_hidden % 32 or late_hidden % 32:
        raise ValueError((early_hidden, late_hidden))
    packed = v23.v21.v20.PackedSwiGLU
    for index in native[:4]:
        model.blocks[index].ff = packed(int(shape.dim), early_hidden).to(device)
    for index in native[-4:]:
        model.blocks[index].ff = packed(int(shape.dim), late_hidden).to(device)
    model._r1_early_ff_hidden = early_hidden
    model._r1_late_ff_hidden = late_hidden
    return {
        "native_indices": native,
        "early_indices": native[:4],
        "late_indices": native[-4:],
        "early_hidden": early_hidden,
        "late_hidden": late_hidden,
    }


def build_candidate_r1(spec, shape, args, deps, device: torch.device) -> nn.Module:
    # The inherited v29 constructor installs the exact Mamba-2 blocks, converts
    # the last refresh window from 2048 to 1024, and installs the validated
    # checkpoint policy.
    model = _BUILD_PRE_R1(spec, shape, args, deps, device)
    arm = R1_BY_NAME.get(spec.name)
    if arm is None:
        return model

    # Replace selected refresh stations with native Field blocks.  Use an arm-
    # independent seed per physical station so shared replacements are paired.
    for position in REFRESH_POSITIONS:
        if position in arm.refresh_positions:
            continue
        if classify_block(model.blocks[position]) != "refresh":
            raise AssertionError(f"expected refresh at {position}, got {classify_block(model.blocks[position])}")
        seed = int(args.model_seed) + 370_000 + int(position)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model.blocks[position] = _new_native_field(shape, args, deps, device)

    realloc = _install_generalized_late_reallocation(model, shape, device, delta=256)
    counts = architecture_counts(model)
    expected = {
        "field": int(arm.field_count),
        "mamba": len(arm.mamba_positions),
        "refresh": len(arm.refresh_positions),
    }
    for key, value in expected.items():
        if int(counts[key]) != int(value):
            raise AssertionError(f"{arm.name} {key}: expected={value} got={counts[key]} counts={counts}")
    model._r1_arm_name = arm.name
    model._r1_arm = asdict(arm)
    model._r1_counts = counts
    model._r1_reallocation = realloc
    # v35 optimizer reads this attribute.
    model._v35_arm_name = arm.name
    return model


def solve_candidate_shapes_r1(args, deps):
    base_shape = v25.solve_shapes_v25(args, deps)[v29.FUSION]
    dim = int(base_shape.dim)
    multiple = int(args.shape_multiple)
    shapes: Dict[str, v23.Shape] = {}
    accounting: Dict[str, object] = {
        "target_params": int(args.target_params),
        "base_shape": asdict(base_shape),
        "arms": {},
    }

    for spec in v29.selected_candidates(args):
        arm = R1_BY_NAME[spec.name]
        provisional = v23.Shape(
            spec.name,
            int(base_shape.params),
            dim,
            int(base_shape.layers),
            int(base_shape.heads),
            int(base_shape.ff_hidden),
        )
        model = build_candidate_r1(spec, provisional, args, deps, torch.device("cpu"))
        provisional_params = nparams(model)
        counts = architecture_counts(model)
        ff_blocks = 24 - int(counts["mamba"])
        slope = 3 * dim * ff_blocks
        fixed = int(provisional_params) - slope * int(provisional.ff_hidden)
        del model
        gc.collect()

        raw_hidden = (int(args.target_params) - fixed) / max(slope, 1)
        hidden = int(round(raw_hidden / multiple) * multiple)
        hidden = max(int(args.min_ff_hidden), min(int(args.max_ff_hidden), hidden))
        shape = v23.Shape(spec.name, 0, dim, int(base_shape.layers), int(base_shape.heads), hidden)
        verify = build_candidate_r1(spec, shape, args, deps, torch.device("cpu"))
        actual = nparams(verify)
        verified_counts = architecture_counts(verify)
        del verify
        gc.collect()

        delta_pct = 100.0 * (actual - int(args.target_params)) / int(args.target_params)
        if abs(delta_pct) > float(args.param_tolerance_pct):
            raise RuntimeError(
                f"R1 parameter mismatch {spec.name}: {delta_pct:+.4f}% "
                f"actual={actual:,} hidden={hidden} fixed={fixed:,} slope={slope:,}"
            )
        shapes[spec.name] = v23.Shape(
            spec.name, actual, dim, int(base_shape.layers), int(base_shape.heads), hidden
        )
        accounting["arms"][spec.name] = {
            "topology": asdict(arm),
            "counts": verified_counts,
            "ff_hidden": hidden,
            "params": actual,
            "delta_pct": delta_pct,
            "fixed_params": fixed,
            "ff_slope": slope,
        }
    return base_shape, shapes, accounting


def configure_r1(args: argparse.Namespace) -> None:
    v35.BASELINE_NAME = BASELINE_NAME
    v35.STRUCTURAL_ARMS = STRUCTURAL_ARMS
    v35.configure_tables(STRUCTURAL_ARMS)
    v35.ARM_BY_NAME = STRUCT_BY_NAME
    v35.SPEC_BY_NAME = SPEC_BY_NAME

    def signature(args_, spec, shape, total_sequences):
        row = dict(_SIGNATURE_PRE_R1(args_, spec, shape, total_sequences))
        arm = R1_BY_NAME.get(spec.name)
        row.update({
            "r1_version": VERSION,
            "r1_arm": None if arm is None else asdict(arm),
            "r1_runtime_geometry": {
                "field_chunk": int(args_.field_chunk),
                "triton_block_c": int(args_.triton_block_c),
                "triton_chunk_t": int(args_.triton_chunk_t),
            },
            "r1_paired_starts_seed": int(args_.data_seed),
        })
        return row

    v29.CANDIDATES = tuple(SPEC_BY_NAME.values())
    v29.build_candidate = build_candidate_r1
    v29.solve_candidate_shapes = solve_candidate_shapes_r1
    v29.make_candidate_optimizer = v35.make_optimizer_v35
    v29.checkpoint_signature = signature
    v29.is_refresh_block = v35.is_refresh_block_v35

    v32.GAP_ARMS = STRUCTURAL_ARMS
    v32.ARM_BY_NAME = STRUCT_BY_NAME
    v32.SPEC_BY_NAME = SPEC_BY_NAME
    v32.make_optimizer_v32 = v35.make_optimizer_v35
    v32.v29.CANDIDATES = tuple(SPEC_BY_NAME.values())
    v32.v29.build_candidate = build_candidate_r1
    v32.v29.solve_candidate_shapes = solve_candidate_shapes_r1
    v32.v29.make_candidate_optimizer = v35.make_optimizer_v35
    v32.v29.checkpoint_signature = signature
    v32.v29.is_refresh_block = v35.is_refresh_block_v35
    v34.LR_ARMS = STRUCTURAL_ARMS
    v34.BASELINE_NAME = BASELINE_NAME
    v29.VERSION = VERSION
    v32.VERSION = VERSION


def deterministic_starts(
    count: int,
    upper: int,
    seed: int,
    path: Path,
) -> np.ndarray:
    if count < 1 or upper < 1:
        raise ValueError((count, upper))
    rng = np.random.default_rng(int(seed))
    starts = rng.integers(0, int(upper) + 1, size=int(count), dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        existing = np.load(path, allow_pickle=False)
        if existing.dtype != starts.dtype or not np.array_equal(existing, starts):
            raise AssertionError(f"paired starts mismatch: {path}")
    else:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as stream:
            np.save(stream, starts, allow_pickle=False)
        os.replace(tmp, path)
    return starts


def drift_2k_64k(context: Mapping[str, Mapping[str, float]]) -> Optional[float]:
    low = float(context.get("2048", {}).get("nll", float("nan")))
    high = float(context.get("65536", {}).get("nll", float("nan")))
    if not math.isfinite(low) or not math.isfinite(high):
        return None
    return high - low


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--screen-token-budget-r1", type=int, default=12_582_912)
    p.add_argument("--context-windows-r1", type=int, default=3)
    p.add_argument("--quality-tolerance-r1", type=float, default=0.006)
    p.add_argument("--quality-gain-r1", type=float, default=0.004)
    p.add_argument("--speed-gain-r1", type=float, default=0.03)
    p.add_argument("--memory-gain-r1", type=float, default=0.05)
    p.add_argument("--max-drift-delta-r1", type=float, default=0.015)
    p.add_argument("--package-selftest", action="store_true")
    custom, remaining = p.parse_known_args()

    old = sys.argv
    try:
        sys.argv = [old[0], *remaining]
        args = v35.parse_args()
    finally:
        sys.argv = old

    budget = int(custom.screen_token_budget_r1)
    if budget % int(args.train_seq):
        raise ValueError(f"screen budget {budget} must divide train_seq={args.train_seq}")
    args.screen_token_budget = budget
    args.screen_token_budget_v34 = budget
    args.screen_token_budget_v35 = budget
    args.quality_token_budget = budget
    args.ablation_token_budget = budget
    args.eval_fractions = [0.50, 1.0]
    args.ablation_eval_fractions = [0.50, 1.0]
    args.ablation_long_contexts = [2048, 16384, 65536]
    args.ablation_long_windows = int(custom.context_windows_r1)
    args.long_contexts = [2048, 16384, 65536]
    args.long_context_score_tokens = 128
    args.long_context_windows = int(custom.context_windows_r1)
    args.context_windows_v34 = int(custom.context_windows_r1)
    args.context_windows_v35 = int(custom.context_windows_r1)
    args.run_promotion = False
    args.run_promotion_v35 = False
    args.package_selftest = bool(custom.package_selftest)
    args.r1_quality_tolerance = float(custom.quality_tolerance_r1)
    args.r1_quality_gain = float(custom.quality_gain_r1)
    args.r1_speed_gain = float(custom.speed_gain_r1)
    args.r1_memory_gain = float(custom.memory_gain_r1)
    args.r1_max_drift_delta = float(custom.max_drift_delta_r1)
    args.candidate = []
    return args


def prepare_training(args, root: Path, device: torch.device):
    v29.configure(args)
    canonical_path, canonical_sha, deps = v27.load_dependencies(args)
    if canonical_sha != v29.EXPECTED_CANONICAL_SHA256:
        raise RuntimeError(f"canonical SHA mismatch: {canonical_sha}")

    base_shape, shapes, accounting = solve_candidate_shapes_r1(args, deps)
    atomic_json(root / "component_accounting.json", accounting)
    specs = tuple(SPEC_BY_NAME[arm.name] for arm in R1_ARMS)
    v29.architecture_audit(specs, shapes, args, deps, device, root / "architecture")
    v29.causality_and_backward_preflight(specs, shapes, args, deps, device, root / "preflight")

    raw_rows = v23.core.load_raw_rows(args.cache_dir, args.data_frac)
    tokenizer = v23.base.copy_or_train_tokenizer(
        root,
        raw_rows[0],
        args.vocab_size,
        args.tokenizer_min_frequency,
        args.tokenizer_source,
    )
    train_c, val_c, test_c = v23.core.save_or_load_corpora(root, tokenizer, raw_rows)
    train = v23.core.place_tokens(train_c.tokens, device, args.data_device, "train")
    val = v23.core.place_tokens(val_c.tokens, device, args.data_device, "validation")
    test = v23.core.place_tokens(test_c.tokens, device, args.data_device, "test")
    return canonical_path, canonical_sha, deps, base_shape, shapes, train_c, train, val_c, val, test_c, test


def long_context_all(args, results, shapes, deps, test, device, path: Path):
    contexts = {}
    for arm in R1_ARMS:
        result = results[arm.name]
        model = v29.load_model_from_result(
            SPEC_BY_NAME[arm.name], shapes[arm.name], result, args, deps, device
        )
        contexts[arm.name] = v29.long_context_eval(model, test, args, device)
        del model
        clear_cuda()
        atomic_json(path, contexts)
    return contexts


def run_screen(args, root, deps, shapes, train, val_c, val, test_c, test, device):
    count = int(args.screen_token_budget) // int(args.train_seq)
    starts = deterministic_starts(
        count,
        len(train) - int(args.train_seq) - 1,
        int(args.data_seed),
        root / "screen_paired_starts.npy",
    )
    start_sha = hashlib.sha256(starts.tobytes()).hexdigest()
    results = {}
    original_stable = float(args.wsd_stable_fraction)
    args.wsd_stable_fraction = 0.70
    for arm in R1_ARMS:
        log("=" * 220)
        log(f"R1 SCREEN ARM: {arm.name} F/M/R={arm.field_count}/{len(arm.mamba_positions)}/{len(arm.refresh_positions)}")
        results[arm.name] = v29.train_candidate(
            SPEC_BY_NAME[arm.name],
            shapes[arm.name],
            args,
            deps,
            train,
            val_c,
            val,
            test_c,
            test,
            starts,
            root,
            device,
        )
        atomic_json(root / "screen_results.json", {k: asdict(v) for k, v in results.items()})
    args.wsd_stable_fraction = original_stable

    contexts = long_context_all(
        args, results, shapes, deps, test, device, root / "screen_long_contexts.json"
    )
    baseline = results[BASELINE_NAME]
    baseline_drift = drift_2k_64k(contexts[BASELINE_NAME])
    rows = []
    for arm in R1_ARMS:
        result = results[arm.name]
        val_gain = float(baseline.final_validation["nll"]) - float(result.final_validation["nll"])
        test_gain = float(baseline.final_test["nll"]) - float(result.final_test["nll"])
        speed = float(result.tokens_per_second) / max(float(baseline.tokens_per_second), 1e-9)
        memory = float(result.peak_gib) / max(float(baseline.peak_gib), 1e-9)
        drift = drift_2k_64k(contexts[arm.name])
        quality_ok = (
            val_gain >= -float(args.r1_quality_tolerance)
            and test_gain >= -float(args.r1_quality_tolerance)
        )
        drift_ok = (
            drift is not None
            and baseline_drift is not None
            and drift <= baseline_drift + float(args.r1_max_drift_delta)
        )
        efficiency_signal = (
            speed >= 1.0 + float(args.r1_speed_gain)
            or memory <= 1.0 - float(args.r1_memory_gain)
        )
        quality_signal = (
            val_gain >= float(args.r1_quality_gain)
            and test_gain >= 0.0
            and speed >= 0.95
            and memory <= 1.03
        )
        eligible = arm.name != BASELINE_NAME and drift_ok and (
            (quality_ok and efficiency_signal) or quality_signal
        )
        rows.append({
            "candidate": arm.name,
            "topology": asdict(arm),
            "params": int(shapes[arm.name].params),
            "ff_hidden": int(shapes[arm.name].ff_hidden),
            "validation_nll": float(result.final_validation["nll"]),
            "test_nll": float(result.final_test["nll"]),
            "validation_gain_vs_baseline": val_gain,
            "test_gain_vs_baseline": test_gain,
            "tokens_per_second": float(result.tokens_per_second),
            "speed_ratio_vs_baseline": speed,
            "peak_gib": float(result.peak_gib),
            "memory_ratio_vs_baseline": memory,
            "context_2k_to_64k": drift,
            "quality_ok": quality_ok,
            "drift_ok": drift_ok,
            "efficiency_signal": efficiency_signal,
            "quality_signal": quality_signal,
            "eligible": eligible,
        })
    rows.sort(key=lambda row: (row["validation_nll"], row["test_nll"], -row["speed_ratio_vs_baseline"]))
    eligible = [row for row in rows if row["eligible"]]
    winner = min(
        eligible,
        key=lambda row: (
            row["validation_nll"] + 0.25 * row["test_nll"],
            -row["speed_ratio_vs_baseline"],
            row["memory_ratio_vs_baseline"],
        ),
    ) if eligible else None
    decision = {
        "version": VERSION,
        "action": "ADVANCE_R1_ARM_TO_WIKI5_50M" if winner else "KEEP_V35R1_TOPOLOGY",
        "winner": None if winner is None else winner["candidate"],
        "baseline": BASELINE_NAME,
        "rows": rows,
        "contexts": contexts,
        "paired_starts_count": int(count),
        "paired_starts_sha256": start_sha,
        "automatic_launch": False,
        "next_protocol": {
            "data_fraction": 0.05,
            "token_budget": 49_152_000,
            "paired_control": BASELINE_NAME,
            "candidate": None if winner is None else winner["candidate"],
        },
    }
    atomic_json(root / "r1_decision.json", decision)
    return decision


def make_summary(args, accounting, decision) -> str:
    width = 220
    lines = [
        "=" * width,
        "FIELD-FUSION R1 — STRUCTURAL REDUNDANCY MAP",
        "=" * width,
        "From-scratch paired screen of the classical support modules around the native Field.",
        "No equations, Field Triton kernel, PCAF, softpatch, or optimizer recipe are changed.",
        f"screen_tokens_per_arm={int(args.screen_token_budget):,} data_frac={float(args.data_frac):.4f} train_seq={int(args.train_seq)} batch={int(args.batch_size)}",
        f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'} architecture={platform.machine()}",
        "",
        "SCREEN",
        f"{'candidate':22s} {'F/M/R':>7s} {'ff':>5s} {'params':>12s} {'val':>9s} {'test':>9s} {'dVal':>9s} {'dTest':>9s} {'tok/s':>11s} {'speed':>7s} {'GB':>7s} {'mem':>7s} {'2K→64K':>10s} {'signal':>7s}",
    ]
    for row in decision.get("rows", []):
        topology = row["topology"]
        counts = f"{topology['field_count']}/{len(topology['mamba_positions'])}/{len(topology['refresh_positions'])}"
        drift = row.get("context_2k_to_64k")
        drift_text = "n/a" if drift is None else f"{float(drift):+.5f}"
        lines.append(
            f"{row['candidate']:22s} {counts:>7s} {int(row['ff_hidden']):5d} {int(row['params']):12,d} "
            f"{float(row['validation_nll']):9.5f} {float(row['test_nll']):9.5f} "
            f"{float(row['validation_gain_vs_baseline']):+9.5f} {float(row['test_gain_vs_baseline']):+9.5f} "
            f"{float(row['tokens_per_second']):11,.0f} {float(row['speed_ratio_vs_baseline']):7.3f} "
            f"{float(row['peak_gib']):7.2f} {float(row['memory_ratio_vs_baseline']):7.3f} "
            f"{drift_text:>10s} {str(bool(row['eligible'])):>7s}"
        )
    lines += [
        "",
        "DECISION",
        f"action={decision.get('action')}",
        f"winner={decision.get('winner')}",
        "",
        "INTERPRETATION",
        "A winner is not trained further automatically.  If promoted, confirm only control versus winner on WT103 5% / 49.152M paired tokens.",
        "If no arm passes, retain v35r1 and move to Field-native summary refresh / Field-FFN-PCAF co-design rather than adding more support modules.",
        "=" * width,
    ]
    return "\n".join(lines) + "\n"


def package_selftest(args) -> Dict[str, object]:
    rows = {
        "version": VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "bf16": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        "arms": [asdict(arm) for arm in R1_ARMS],
    }
    if torch.cuda.is_available():
        rows["gpu"] = torch.cuda.get_device_name(0)
        rows["capability"] = list(torch.cuda.get_device_capability(0))
        rows["total_gib"] = torch.cuda.get_device_properties(0).total_memory / 2**30
    return rows


def main() -> None:
    args = parse_args()
    configure_r1(args)
    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "args.json", vars(args))
    selftest = package_selftest(args)
    atomic_json(root / "r1_package_selftest.json", selftest)

    log("=" * 220)
    log("FIELD-FUSION R1 — PRE-RUN AUDIT")
    log(f"arms={[arm.name for arm in R1_ARMS]}")
    log(f"topologies={[f'{a.field_count}F/{len(a.mamba_positions)}M/{len(a.refresh_positions)}R' for a in R1_ARMS]}")
    log(f"torch={torch.__version__} cuda={torch.version.cuda} machine={platform.machine()}")

    if args.package_selftest:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16-capable CUDA GPU required")
        v29.configure(args)
        canonical_path, canonical_sha, deps = v27.load_dependencies(args)
        base_shape, shapes, accounting = solve_candidate_shapes_r1(args, deps)
        audit = {
            "canonical_source": str(canonical_path),
            "canonical_sha256": canonical_sha,
            "base_shape": asdict(base_shape),
            "component_accounting": accounting,
            "shapes": {name: asdict(shape) for name, shape in shapes.items()},
        }
        atomic_json(root / "r1_deep_package_selftest.json", audit)
        for arm in R1_ARMS:
            row = accounting["arms"][arm.name]
            log(
                f"selftest_arm={arm.name} "
                f"F/M/R={row['counts']['field']}/{row['counts']['mamba']}/{row['counts']['refresh']} "
                f"params={row['params']:,} ff_hidden={row['ff_hidden']}"
            )
        log("r1_bootstrap=PASS")
        log("r1_topology_parameter_audit=PASS")
        log("[package-selftest] PASS")
        return

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable CUDA GPU required")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.set_float32_matmul_precision("high")

    (
        canonical_path,
        canonical_sha,
        deps,
        base_shape,
        shapes,
        train_c,
        train,
        val_c,
        val,
        test_c,
        test,
    ) = prepare_training(args, root, device)

    accounting = json.loads((root / "component_accounting.json").read_text(encoding="utf-8"))
    log("=" * 220)
    log("PHASE A — R1 PAIRED STRUCTURAL REDUNDANCY SCREEN")
    decision = run_screen(args, root, deps, shapes, train, val_c, val, test_c, test, device)

    payload = {
        "version": VERSION,
        "args": vars(args),
        "canonical_source": str(canonical_path),
        "canonical_sha256": canonical_sha,
        "base_shape": asdict(base_shape),
        "component_accounting": accounting,
        "decision": decision,
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "machine": platform.machine(),
    }
    atomic_json(root / "results.json", payload)
    summary = make_summary(args, accounting, decision)
    atomic_text(root / "summary.txt", summary)
    log(summary)


if __name__ == "__main__":
    main()
