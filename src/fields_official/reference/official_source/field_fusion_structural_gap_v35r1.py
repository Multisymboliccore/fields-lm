#!/usr/bin/env python3
"""FIELD-FUSION v35r1 — structural gap ablation and 49M confirmation (optimizer registry fix).

This run starts from the v34 canonical recipe:
  * 16 native Field blocks + four localized Mamba-2 blocks;
  * refresh windows 256/512/1024/1024;
  * refresh LR 1.35x and Mamba LR 1.10x;
  * PCAF enabled;
  * exact runtime geometry field_chunk=32, BLOCK_C=32, CHUNK_T=64.

It screens four small structural changes at 25.165824M paired tokens and promotes
only the best strictly eligible arm to a from-scratch 49.152M confirmation.
No Transformer or pure Mamba model is trained or benchmarked again.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import field_fusion_lr_calibration_v34 as v34
import field_fusion_gap_closure_v33 as v33
import field_fusion_gap_kernel_v32 as v32
import field_fusion_delta_quality_ablation_v29 as v29
import field_fusion_recipe_memory_v27 as v27
import field_fusion_wiki100_canonical_v23 as v23
import field_fusion_wiki100_mamba2_v25 as v25

VERSION = 351
FUSION = v29.FUSION
BASELINE_NAME = "struct_ref_lr135_m110"
V34_BASELINE_WINNER = "gap_refresh135_mamba110_49m"
CURRENT_MAMBA = v32.CURRENT_MAMBA
# Keep the canonical four end-of-stage editors and add one mid-stage editor in
# each of the two deeper stages.  This preserves a majority-Field backbone.
MAMBA6_LATE = (4, 10, 14, 16, 20, 22)


@dataclass(frozen=True)
class StructuralArm:
    name: str
    description: str
    kind: str = "baseline"
    stable_fraction: float = 0.70
    mamba_lr_scale: float = 1.10
    refresh_lr_scale: float = 1.35
    mamba_positions: Tuple[int, ...] = CURRENT_MAMBA
    small_mamba_d_state: int = 64
    small_mamba_expand: int = 1
    late_ff_delta: int = 256

    def candidate(self) -> v29.CandidateSpec:
        return v29.CandidateSpec(
            self.name,
            self.description,
            refresh_1024x2=True,
            mamba_replace=self.mamba_positions,
        )


STRUCTURAL_ARMS: Tuple[StructuralArm, ...] = (
    StructuralArm(
        BASELINE_NAME,
        "v34 canonical LR recipe and topology; structural control.",
    ),
    StructuralArm(
        "struct_mamba6_small",
        "Six distributed smaller Mamba-2 editors (d_state=64, expand=1), parameter matched.",
        kind="mamba6_small",
        mamba_positions=MAMBA6_LATE,
    ),
    StructuralArm(
        "struct_refresh_ff_gate",
        "Identity-initialized per-channel learned scale on each refresh FFN residual.",
        kind="refresh_ff_gate",
    ),
    StructuralArm(
        "struct_late_field_realloc",
        "Move equal FFN capacity from first four native Field blocks to last four.",
        kind="late_field_realloc",
        late_ff_delta=256,
    ),
    StructuralArm(
        "struct_resident_readout",
        "Cheap causal depthwise resident readout before the LM head, parallel to PCAF states.",
        kind="resident_readout",
    ),
)
ARM_BY_NAME: Dict[str, StructuralArm] = {arm.name: arm for arm in STRUCTURAL_ARMS}
SPEC_BY_NAME: Dict[str, v29.CandidateSpec] = {
    arm.name: arm.candidate() for arm in STRUCTURAL_ARMS
}


def log(value: object = "") -> None:
    print(str(value), flush=True)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nparams(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


class SmallMamba2ResidualBlock(v25.Mamba2ResidualBlock):
    """Official fused Mamba-2 with a smaller state/expansion footprint."""

    def __init__(self, dim: int, args, *, d_state: int, expand: int) -> None:
        nn.Module.__init__(self)
        if v25.OfficialMamba2 is None:
            raise RuntimeError("Official Mamba-2 unavailable: " + v25.MAMBA_IMPORT_ERROR)
        if (int(expand) * int(dim)) % int(args.mamba_headdim):
            raise ValueError("small Mamba expanded width must divide mamba_headdim")
        self.norm = v25.FastRMSNorm(dim)
        self.mixer = v25.OfficialMamba2(
            d_model=dim,
            d_state=int(d_state),
            d_conv=int(args.mamba_d_conv),
            expand=int(expand),
            headdim=int(args.mamba_headdim),
            ngroups=int(args.mamba_ngroups),
            chunk_size=int(args.mamba_chunk_size),
            use_mem_eff_path=True,
        )
        self._v35_small_mamba = True
        self._v35_d_state = int(d_state)
        self._v35_expand = int(expand)


class RefreshFFResidualGate(nn.Module):
    """Preserve the original refresh attention and learn only its FFN strength."""

    def __init__(self, refresh: nn.Module, dim: int) -> None:
        super().__init__()
        self.refresh = refresh
        # 2*sigmoid(0)=1, so the wrapper is mathematically identical at init.
        self.ff_mix_raw = nn.Parameter(torch.zeros(dim))
        self.window = int(getattr(refresh, "window", 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        block = self.refresh
        z = block.norm1(x)
        attention = block.attn(z)
        if bool(block.light_gate):
            gate = torch.sigmoid(block.residual_gate_logit).to(z.dtype)
        else:
            if block.residual_gate is None:
                raise RuntimeError("refresh residual gate missing")
            gate = torch.sigmoid(block.residual_gate(z))
        x = x + gate * attention
        ff = block.ff(block.norm2(x))
        scale = (2.0 * torch.sigmoid(self.ff_mix_raw)).to(ff.dtype)
        return x + scale * ff


class ResidentCausalReadout(nn.Module):
    """Parameter-light local resident path used only by the parametric logits."""

    def __init__(self, base_norm: nn.Module, dim: int) -> None:
        super().__init__()
        self.base_norm = base_norm
        self.pre_norm = v25.FastRMSNorm(dim)
        self.depthwise = nn.Conv1d(
            dim, dim, kernel_size=3, groups=dim, bias=False
        )
        nn.init.zeros_(self.depthwise.weight)
        # Starts close to off while preserving gradients into the zero conv.
        self.mix_raw = nn.Parameter(torch.full((dim,), -2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pre_norm(x).transpose(1, 2)
        local = self.depthwise(F.pad(z, (2, 0))).transpose(1, 2)
        mix = torch.sigmoid(self.mix_raw).to(local.dtype)
        return self.base_norm(x + mix * local)


_BUILD_BEFORE_V35 = v29.build_candidate
_IS_REFRESH_BEFORE_V35 = v29.is_refresh_block


def is_refresh_block_v35(block: nn.Module) -> bool:
    return isinstance(block, RefreshFFResidualGate) or _IS_REFRESH_BEFORE_V35(block)


def configure_tables(arms: Sequence[StructuralArm]) -> None:
    global ARM_BY_NAME, SPEC_BY_NAME
    arms = tuple(arms)
    ARM_BY_NAME = {arm.name: arm for arm in arms}
    SPEC_BY_NAME = {arm.name: arm.candidate() for arm in arms}
    # v32's specialized optimizer reads these tables dynamically.
    v32.GAP_ARMS = arms
    v32.ARM_BY_NAME = ARM_BY_NAME
    v32.SPEC_BY_NAME = SPEC_BY_NAME
    v29.CANDIDATES = tuple(SPEC_BY_NAME.values())
    # Reuse v34's validated screen machinery with v35 arms/baseline.
    v34.LR_ARMS = arms
    v34.BASELINE_NAME = BASELINE_NAME


def make_optimizer_v35(model: nn.Module, args):
    """Structural-arm-aware AdamW without v32's hard-coded fallback key.

    v32's optimizer assumes that the registry always contains
    ``gap_ref_stable70``.  v35 deliberately replaces that registry with
    structural arm names, so the old fallback can raise before update zero.
    This implementation resolves the current arm explicitly and falls back to
    v35's own baseline, while preserving the exact LR-group semantics.
    """
    arm_name = str(
        getattr(
            model,
            "_v35_arm_name",
            getattr(model, "_v32_arm_name", BASELINE_NAME),
        )
    )
    arm = ARM_BY_NAME.get(arm_name)
    if arm is None:
        arm = ARM_BY_NAME.get(BASELINE_NAME)
    if arm is None:
        raise KeyError(
            f"missing structural optimizer arm {arm_name!r}; "
            f"available={sorted(ARM_BY_NAME)} baseline={BASELINE_NAME!r}"
        )

    groups: Dict[Tuple[float, bool], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        scale = 2.0 if v32.v26.is_special_parameter(name) else 1.0
        index = v32.parse_block_index(name)
        if index is not None and index < len(model.blocks):
            block = model.blocks[index]
            if v29.is_official_mamba_block(block):
                scale *= float(arm.mamba_lr_scale)
            elif is_refresh_block_v35(block):
                scale *= float(arm.refresh_lr_scale)
        decay = parameter.ndim >= 2
        groups.setdefault((round(scale, 6), decay), []).append(parameter)

    if not groups:
        raise RuntimeError(f"optimizer for {arm_name!r} has no trainable parameters")
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


def optimizer_registry_selftest(args: argparse.Namespace) -> Dict[str, object]:
    """Exercise the exact optimizer entry point and LR routing for every arm."""

    class DummyMamba(v25.Mamba2ResidualBlock):
        def __init__(self) -> None:
            nn.Module.__init__(self)
            self.weight = nn.Parameter(torch.ones(2, 2))

    class DummyRefreshCore(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.window = 256

    class Probe(nn.Module):
        def __init__(self, arm_name: str) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(
                [DummyMamba(), RefreshFFResidualGate(DummyRefreshCore(), 2)]
            )
            self.weight = nn.Parameter(torch.ones(2, 2))
            self.bias = nn.Parameter(torch.zeros(2))
            self._v35_arm_name = arm_name
            self._v32_arm_name = arm_name

    rows = {}
    for arm in STRUCTURAL_ARMS:
        probe = Probe(arm.name)
        optimizer = make_optimizer_v35(probe, args)
        lr_scales = sorted(
            {float(group.get("lr_scale", 1.0)) for group in optimizer.param_groups}
        )
        expected = {1.0, float(arm.mamba_lr_scale), float(arm.refresh_lr_scale)}
        if not expected.issubset({round(value, 6) for value in lr_scales}):
            raise AssertionError(
                f"optimizer LR routing failed for {arm.name}: "
                f"expected={sorted(expected)} got={lr_scales}"
            )
        rows[arm.name] = {
            "groups": len(optimizer.param_groups),
            "lr_scales": lr_scales,
            "expected_mamba": float(arm.mamba_lr_scale),
            "expected_refresh": float(arm.refresh_lr_scale),
        }

    # Prove that the exact legacy name from the failed run resolves safely.
    fallback = Probe("gap_ref_stable70")
    fallback_optimizer = make_optimizer_v35(fallback, args)
    fallback_scales = sorted(
        {float(group.get("lr_scale", 1.0)) for group in fallback_optimizer.param_groups}
    )
    expected_fallback = {
        1.0,
        float(ARM_BY_NAME[BASELINE_NAME].mamba_lr_scale),
        float(ARM_BY_NAME[BASELINE_NAME].refresh_lr_scale),
    }
    if not expected_fallback.issubset({round(value, 6) for value in fallback_scales}):
        raise AssertionError(
            f"legacy optimizer fallback routing failed: {fallback_scales}"
        )
    rows["legacy_fallback"] = {
        "requested": "gap_ref_stable70",
        "resolved": BASELINE_NAME,
        "groups": len(fallback_optimizer.param_groups),
        "lr_scales": fallback_scales,
    }
    return rows


def replace_small_mamba(model: nn.Module, arm: StructuralArm, args, device: torch.device) -> None:
    torch.manual_seed(int(args.model_seed) + 350_610)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.model_seed) + 350_610)
    for index in arm.mamba_positions:
        if not v29.is_official_mamba_block(model.blocks[index]):
            raise AssertionError(f"expected Mamba block at index {index}")
        model.blocks[index] = SmallMamba2ResidualBlock(
            model.emb.embedding_dim,
            args,
            d_state=arm.small_mamba_d_state,
            expand=arm.small_mamba_expand,
        ).to(device)


def install_refresh_ff_gates(model: nn.Module, device: torch.device) -> None:
    count = 0
    for index, block in enumerate(list(model.blocks)):
        if _IS_REFRESH_BEFORE_V35(block):
            model.blocks[index] = RefreshFFResidualGate(
                block, model.emb.embedding_dim
            ).to(device)
            count += 1
    if count != 4:
        raise AssertionError(f"expected four refreshes, wrapped {count}")


def install_late_field_reallocation(
    model: nn.Module, shape: v23.Shape, arm: StructuralArm, device: torch.device
) -> None:
    native_indices = [
        index
        for index, block in enumerate(model.blocks)
        if not is_refresh_block_v35(block) and not v29.is_official_mamba_block(block)
    ]
    if len(native_indices) != 16:
        raise AssertionError(f"expected 16 native Field blocks, got {len(native_indices)}")
    delta = int(arm.late_ff_delta)
    early_hidden = int(shape.ff_hidden) - delta
    late_hidden = int(shape.ff_hidden) + delta
    if early_hidden <= 0 or early_hidden % 32 or late_hidden % 32:
        raise ValueError((early_hidden, late_hidden))
    packed = v23.v21.v20.PackedSwiGLU
    for index in native_indices[:4]:
        model.blocks[index].ff = packed(
            model.emb.embedding_dim, early_hidden
        ).to(device)
    for index in native_indices[-4:]:
        model.blocks[index].ff = packed(
            model.emb.embedding_dim, late_hidden
        ).to(device)
    model._v35_early_ff_hidden = early_hidden
    model._v35_late_ff_hidden = late_hidden


def build_candidate_v35(
    spec: v29.CandidateSpec,
    shape: v23.Shape,
    args,
    deps,
    device: torch.device,
) -> nn.Module:
    model = _BUILD_BEFORE_V35(spec, shape, args, deps, device)
    arm = ARM_BY_NAME.get(spec.name)
    if arm is None:
        return model
    if arm.kind == "mamba6_small":
        replace_small_mamba(model, arm, args, device)
    elif arm.kind == "refresh_ff_gate":
        install_refresh_ff_gates(model, device)
    elif arm.kind == "late_field_realloc":
        install_late_field_reallocation(model, shape, arm, device)
    elif arm.kind == "resident_readout":
        model.final_norm = ResidentCausalReadout(
            model.final_norm, model.emb.embedding_dim
        ).to(device)
    elif arm.kind != "baseline":
        raise ValueError(f"unknown structural arm kind {arm.kind!r}")
    model._v35_arm_name = arm.name
    model._v35_arm = asdict(arm)
    return model


def component_accounting_v35(args, deps, base_shape: v23.Shape) -> Dict[str, int]:
    base = v29.build_base_field(base_shape, args, deps, torch.device("cpu"))
    base_total = nparams(base)
    refreshes = [block for block in base.blocks if _IS_REFRESH_BEFORE_V35(block)]
    native = [block for block in base.blocks if not _IS_REFRESH_BEFORE_V35(block)]
    if len(refreshes) != 4 or len(native) != 20:
        raise AssertionError((len(native), len(refreshes)))
    field_block_total = nparams(native[0])
    ff_params = 3 * int(base_shape.dim) * int(base_shape.ff_hidden)
    field_mixer_fixed = field_block_total - ff_params
    del base
    gc.collect()

    canonical_mamba = v25.Mamba2ResidualBlock(base_shape.dim, args)
    canonical_mamba_params = nparams(canonical_mamba)
    del canonical_mamba
    small_mamba = SmallMamba2ResidualBlock(
        base_shape.dim, args, d_state=64, expand=1
    )
    small_mamba_params = nparams(small_mamba)
    del small_mamba
    gc.collect()

    dim = int(base_shape.dim)
    return {
        "base_total": int(base_total),
        "base_hidden": int(base_shape.ff_hidden),
        "field_block_total": int(field_block_total),
        "field_mixer_fixed": int(field_mixer_fixed),
        "canonical_mamba": int(canonical_mamba_params),
        "small_mamba": int(small_mamba_params),
        "refresh_ff_gate_extra": int(4 * dim),
        "resident_readout_extra": int(5 * dim),
    }


def solve_candidate_shapes_v35(args, deps):
    base_shape = v25.solve_shapes_v25(args, deps)[FUSION]
    accounting = component_accounting_v35(args, deps, base_shape)
    dim = int(base_shape.dim)
    base_fixed = int(accounting["base_total"]) - 3 * dim * 24 * int(base_shape.ff_hidden)
    multiple = int(args.shape_multiple)
    shapes: Dict[str, v23.Shape] = {}
    for spec in v29.selected_candidates(args):
        arm = ARM_BY_NAME[spec.name]
        replaced = len(arm.mamba_positions)
        ff_blocks = 24 - replaced
        fixed = base_fixed - replaced * int(accounting["field_mixer_fixed"])
        if arm.kind == "mamba6_small":
            fixed += replaced * int(accounting["small_mamba"])
        else:
            fixed += replaced * int(accounting["canonical_mamba"])
        if arm.kind == "refresh_ff_gate":
            fixed += int(accounting["refresh_ff_gate_extra"])
        elif arm.kind == "resident_readout":
            fixed += int(accounting["resident_readout_extra"])
        slope = 3 * dim * ff_blocks
        raw_hidden = (int(args.target_params) - fixed) / max(slope, 1)
        hidden = int(round(raw_hidden / multiple) * multiple)
        hidden = max(int(args.min_ff_hidden), min(int(args.max_ff_hidden), hidden))
        estimated = int(fixed + slope * hidden)
        delta_pct = 100.0 * (estimated - int(args.target_params)) / int(args.target_params)
        if abs(delta_pct) > float(args.param_tolerance_pct):
            raise RuntimeError(
                f"parameter mismatch {spec.name}: {delta_pct:+.3f}% "
                f"hidden={hidden} estimated={estimated:,}"
            )
        shapes[spec.name] = v23.Shape(
            spec.name,
            estimated,
            dim,
            base_shape.layers,
            base_shape.heads,
            hidden,
        )
    return base_shape, shapes, accounting


def install_v35_hooks(args: argparse.Namespace) -> None:
    configure_tables(STRUCTURAL_ARMS)
    # Install exact starts/runtime signature from v34/v33 first.
    v34.install_v34_signature(args)
    prior_signature = v29.checkpoint_signature

    def signature(args_, spec, shape, total_sequences):
        row = dict(prior_signature(args_, spec, shape, total_sequences))
        arm = ARM_BY_NAME.get(spec.name)
        row["v35_version"] = VERSION
        row["v35_structural_arm"] = None if arm is None else asdict(arm)
        row["v35_runtime_geometry"] = {
            "field_chunk": int(args_.field_chunk),
            "triton_block_c": int(args_.triton_block_c),
            "triton_chunk_t": int(args_.triton_chunk_t),
        }
        row["v35_target_v34_sha256"] = sha256(
            Path(args_.target_v34_root) / "promotion_decision.json"
        )
        return row

    v29.build_candidate = build_candidate_v35
    v29.is_refresh_block = is_refresh_block_v35
    v29.solve_candidate_shapes = solve_candidate_shapes_v35
    v29.checkpoint_signature = signature
    v29.make_candidate_optimizer = make_optimizer_v35
    # Keep all aliases synchronized because train_candidate resolves the global
    # function from the shared v29 module at runtime.
    v32.make_optimizer_v32 = make_optimizer_v35
    v32.v29.make_candidate_optimizer = make_optimizer_v35
    v32.v29.build_candidate = build_candidate_v35
    v32.v29.is_refresh_block = is_refresh_block_v35
    v32.v29.solve_candidate_shapes = solve_candidate_shapes_v35
    v32.v29.checkpoint_signature = signature
    v32.VERSION = VERSION
    v29.VERSION = VERSION


def parse_args() -> argparse.Namespace:
    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument("--screen-token-budget-v35", type=int, default=25_165_824)
    custom.add_argument("--promotion-token-budget-v35", type=int, default=49_152_000)
    custom.add_argument(
        "--promotion-eval-fractions-v35",
        nargs="+",
        type=float,
        default=[25_165_824 / 49_152_000, 1.0],
    )
    custom.add_argument(
        "--target-v34-root",
        default="/home/ubuntu/pcaf_runs/field_fusion_lr_calibration_v34_run",
    )
    custom.add_argument("--screen-min-gain-v35", type=float, default=0.010)
    custom.add_argument("--promotion-min-gain-v35", type=float, default=0.008)
    custom.add_argument("--max-context-drift-v35", type=float, default=0.100)
    custom.add_argument("--context-windows-v35", type=int, default=4)
    custom.add_argument(
        "--run-promotion-v35", action=argparse.BooleanOptionalAction, default=True
    )
    custom_args, remaining = custom.parse_known_args()

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining]
        args = v34.parse_args()
    finally:
        sys.argv = old_argv

    args.screen_token_budget_v34 = int(custom_args.screen_token_budget_v35)
    args.screen_token_budget_v35 = int(custom_args.screen_token_budget_v35)
    args.promotion_token_budget = int(custom_args.promotion_token_budget_v35)
    args.promotion_token_budget_v35 = int(custom_args.promotion_token_budget_v35)
    args.promotion_eval_fractions = list(custom_args.promotion_eval_fractions_v35)
    args.target_v34_root = str(custom_args.target_v34_root)
    args.screen_min_gain = float(custom_args.screen_min_gain_v35)
    args.promotion_min_gain = float(custom_args.promotion_min_gain_v35)
    args.screen_max_context_drift = float(custom_args.max_context_drift_v35)
    args.promotion_max_context_drift = float(custom_args.max_context_drift_v35)
    args.context_windows_v34 = int(custom_args.context_windows_v35)
    args.context_windows_v35 = int(custom_args.context_windows_v35)
    args.run_promotion = bool(custom_args.run_promotion_v35)
    args.ablation_token_budget = int(args.screen_token_budget_v35)
    args.screen_token_budget = int(args.screen_token_budget_v35)
    args.quality_token_budget = int(args.screen_token_budget_v35)
    args.eval_fractions = [0.50, 1.0]
    args.ablation_eval_fractions = [0.50, 1.0]
    args.ablation_long_contexts = [2048, 16384, 65536]
    args.ablation_long_windows = int(args.context_windows_v35)
    args.long_contexts = [2048, 16384, 65536]
    args.long_context_score_tokens = 128
    args.long_context_windows = int(args.context_windows_v35)
    args.export_winner_bf16 = True
    return args


def read_v34_baseline(root: Path) -> Dict[str, object]:
    decision_path = root / "promotion_decision.json"
    result_path = root / "promotion_result.json"
    if not decision_path.is_file():
        raise FileNotFoundError(decision_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    winner = str(decision.get("winner", ""))
    if winner != V34_BASELINE_WINNER:
        raise AssertionError(f"unexpected v34 promotion winner: {winner!r}")
    row = {
        "path": str(decision_path),
        "sha256": sha256(decision_path),
        "candidate": winner,
        "validation_nll": float(decision["validation_nll"]),
        "test_nll": float(decision["test_nll"]),
        "tokens_per_second": float(decision["tokens_per_second"]),
        "peak_gib": float(decision["peak_gib"]),
        "context_2k_to_64k": float(decision["context_2k_to_64k"]),
        "checkpoint": str(decision.get("checkpoint", "")),
    }
    if result_path.is_file():
        row["result_path"] = str(result_path)
        row["result_sha256"] = sha256(result_path)
    for key in (
        "validation_nll",
        "test_nll",
        "tokens_per_second",
        "peak_gib",
        "context_2k_to_64k",
    ):
        if not math.isfinite(float(row[key])):
            raise RuntimeError(f"non-finite v34 baseline {key}: {row[key]}")
    return row


def promotion_spec_and_arm_v35(winner_name: str):
    source = ARM_BY_NAME[winner_name]
    promotion_name = winner_name + "_49m"
    arm = StructuralArm(
        name=promotion_name,
        description=source.description + " Confirmed from scratch at 49.152M tokens.",
        kind=source.kind,
        stable_fraction=source.stable_fraction,
        mamba_lr_scale=source.mamba_lr_scale,
        refresh_lr_scale=source.refresh_lr_scale,
        mamba_positions=source.mamba_positions,
        small_mamba_d_state=source.small_mamba_d_state,
        small_mamba_expand=source.small_mamba_expand,
        late_ff_delta=source.late_ff_delta,
    )
    ARM_BY_NAME[promotion_name] = arm
    SPEC_BY_NAME[promotion_name] = arm.candidate()
    v32.ARM_BY_NAME[promotion_name] = arm
    v32.SPEC_BY_NAME[promotion_name] = SPEC_BY_NAME[promotion_name]
    return arm, SPEC_BY_NAME[promotion_name]


def drift_2k_64k(context: Mapping[str, Mapping[str, float]]) -> Optional[float]:
    low = float(context.get("2048", {}).get("nll", float("nan")))
    high = float(context.get("65536", {}).get("nll", float("nan")))
    if not math.isfinite(low) or not math.isfinite(high):
        return None
    return high - low


def exact_starts(args: argparse.Namespace, train: torch.Tensor, budget: int, path: Path) -> np.ndarray:
    if int(budget) % int(args.train_seq):
        raise ValueError(f"token budget {budget} must divide train_seq={args.train_seq}")
    count = int(budget) // int(args.train_seq)
    return v33.make_starts_from_v28(
        count,
        len(train) - int(args.train_seq) - 1,
        int(args.data_seed),
        path,
    )


def run_structural_screen(
    args,
    root: Path,
    deps,
    shapes,
    train,
    val_c,
    val,
    test_c,
    test,
    device,
) -> Dict[str, object]:
    # v34's screen implementation is already validated and now sees v35 tables.
    decision = v34.run_screen(
        args, root, deps, shapes, train, val_c, val, test_c, test, device
    )
    decision["action"] = (
        "PROMOTE_STRUCTURAL_ARM_TO_49M" if decision.get("winner") else "NO_STRICT_STRUCTURAL_ARM"
    )
    decision["version"] = VERSION
    atomic_json(root / "screen_decision.json", decision)
    return decision


def run_structural_promotion(
    args,
    root: Path,
    deps,
    train,
    val_c,
    val,
    test_c,
    test,
    screen_decision: Mapping[str, object],
    baseline: Mapping[str, object],
    device,
) -> Dict[str, object]:
    winner_name = screen_decision.get("winner")
    if not args.run_promotion or not winner_name:
        decision = {
            "action": "NOT_RUN",
            "reason": "promotion disabled or no strictly eligible screen winner",
            "winner": winner_name,
        }
        atomic_json(root / "promotion_decision.json", decision)
        return decision

    arm, spec = promotion_spec_and_arm_v35(str(winner_name))
    v29.CANDIDATES = (spec,)
    original_budget = int(args.screen_token_budget)
    original_quality = int(args.quality_token_budget)
    original_fractions = list(args.eval_fractions)
    original_stable = float(args.wsd_stable_fraction)

    args.screen_token_budget = int(args.promotion_token_budget_v35)
    args.quality_token_budget = int(args.promotion_token_budget_v35)
    args.eval_fractions = list(map(float, args.promotion_eval_fractions))
    args.wsd_stable_fraction = 0.70

    _, shapes, accounting = solve_candidate_shapes_v35(args, deps)
    atomic_json(root / "promotion_component_accounting.json", accounting)
    v29.architecture_audit((spec,), shapes, args, deps, device, root / "promotion_architecture")
    v29.causality_and_backward_preflight(
        (spec,), shapes, args, deps, device, root / "promotion_preflight"
    )
    starts = exact_starts(
        args,
        train,
        int(args.promotion_token_budget_v35),
        root / "promotion_paired_starts.npy",
    )
    log("=" * 220)
    log(
        f"49M STRUCTURAL CONFIRMATION: {spec.name} kind={arm.kind} "
        f"refreshLR={arm.refresh_lr_scale:.2f} mambaLR={arm.mamba_lr_scale:.2f}"
    )
    result = v29.train_candidate(
        spec,
        shapes[spec.name],
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
    atomic_json(root / "promotion_result.json", asdict(result))

    model = v29.load_model_from_result(
        spec, shapes[spec.name], result, args, deps, device
    )
    args.long_contexts = [2048, 16384, 65536]
    args.long_context_score_tokens = 128
    args.long_context_windows = int(args.context_windows_v35)
    contexts = v29.long_context_eval(model, test, args, device)
    del model
    v32.clear_cuda()
    atomic_json(root / "promotion_long_contexts.json", contexts)

    drift = drift_2k_64k(contexts)
    gain_val = float(baseline["validation_nll"]) - float(result.final_validation["nll"])
    gain_test = float(baseline["test_nll"]) - float(result.final_test["nll"])
    speed_ratio = float(result.tokens_per_second) / max(float(baseline["tokens_per_second"]), 1e-9)
    memory_ratio = float(result.peak_gib) / max(float(baseline["peak_gib"]), 1e-9)
    eligible = (
        gain_val >= float(args.promotion_min_gain)
        and gain_test >= 0.0
        and speed_ratio >= float(args.promotion_min_speed_ratio)
        and memory_ratio <= float(args.promotion_max_memory_ratio)
        and drift is not None
        and drift <= float(args.promotion_max_context_drift)
    )
    decision = {
        "action": "PROMOTE_STRUCTURAL_RECIPE_TO_98M" if eligible else "STOP_AFTER_49M",
        "winner": spec.name,
        "source_screen_winner": winner_name,
        "eligible": eligible,
        "validation_nll": float(result.final_validation["nll"]),
        "test_nll": float(result.final_test["nll"]),
        "tokens_per_second": float(result.tokens_per_second),
        "peak_gib": float(result.peak_gib),
        "validation_gain_vs_v34": gain_val,
        "test_gain_vs_v34": gain_test,
        "speed_ratio_vs_v34": speed_ratio,
        "memory_ratio_vs_v34": memory_ratio,
        "context_2k_to_64k": drift,
        "contexts": contexts,
        "v34_baseline": dict(baseline),
        "arm": asdict(arm),
        "checkpoint": result.checkpoint,
    }
    atomic_json(root / "promotion_decision.json", decision)

    args.screen_token_budget = original_budget
    args.quality_token_budget = original_quality
    args.eval_fractions = original_fractions
    args.wsd_stable_fraction = original_stable
    return decision


def make_summary(
    reused_inference: Mapping[str, object],
    reused_kernel: Mapping[str, object],
    baseline: Mapping[str, object],
    screen: Mapping[str, object],
    promotion: Mapping[str, object],
) -> str:
    width = 220
    lines = [
        "=" * width,
        "FIELD-FUSION v35r1 — STRUCTURAL GAP ABLATION + 49M CONFIRMATION",
        "=" * width,
        "No Transformer or pure Mamba model was trained or benchmarked again.",
        "All arms use v34 LR recipe: refresh=1.35x, Mamba=1.10x, PCAF on.",
        f"runtime={reused_kernel.get('winner')} speed={float(reused_kernel.get('winner_speed_ratio', float('nan'))):.4f}x "
        "geometry=field32/blockC32/chunkT64",
        "",
        "REUSED MATCHED-BATCH LONG-CONTEXT PREFILL",
        f"{'ctx':>8s} {'batch':>7s} {'Field tok/s':>14s} {'TF tok/s':>14s} {'Mamba tok/s':>14s} {'F/TF':>8s} {'F/M':>8s}",
    ]
    for row in reused_inference.get("comparisons", []):
        lines.append(
            f"{int(row['context']):8,d} {int(row['batch']):7d} "
            f"{float(row['field_tokens_per_second']):14,.0f} "
            f"{float(row['transformer_tokens_per_second']):14,.0f} "
            f"{float(row['mamba_tokens_per_second']):14,.0f} "
            f"{float(row['field_over_transformer_speed']):8.3f} "
            f"{float(row['field_over_mamba_speed']):8.3f}"
        )
    lines += [
        "",
        "25.165824M STRUCTURAL SCREEN",
        f"{'candidate':31s} {'val':>9s} {'test':>9s} {'dVal':>9s} {'tok/s':>11s} {'speed':>8s} {'GB':>7s} {'2K→64K':>10s} {'eligible':>9s}",
    ]
    for row in screen.get("rows", []):
        drift = row.get("context_2k_to_64k")
        drift_text = "n/a" if drift is None else f"{float(drift):+.5f}"
        lines.append(
            f"{str(row['candidate']):31s} {float(row['validation_nll']):9.5f} "
            f"{float(row['test_nll']):9.5f} {float(row['validation_gain_vs_baseline']):+9.5f} "
            f"{float(row['tokens_per_second']):11,.0f} {float(row['speed_ratio_vs_baseline']):8.3f} "
            f"{float(row['peak_gib']):7.2f} {drift_text:>10s} {str(bool(row['eligible'])):>9s}"
        )
    lines += [
        "",
        f"screen_action={screen.get('action')}",
        f"screen_winner={screen.get('winner')}",
        "",
        "FROZEN v34 49M BASELINE",
        f"val={float(baseline['validation_nll']):.5f} test={float(baseline['test_nll']):.5f} "
        f"tok/s={float(baseline['tokens_per_second']):,.0f} peak={float(baseline['peak_gib']):.2f}G "
        f"drift={float(baseline['context_2k_to_64k']):+.5f}",
        "",
        "49M STRUCTURAL CONFIRMATION",
    ]
    if promotion.get("action") == "NOT_RUN":
        lines.append(f"not run: {promotion.get('reason')}")
    else:
        drift = promotion.get("context_2k_to_64k")
        drift_text = "n/a" if drift is None else f"{float(drift):+.5f}"
        lines += [
            f"candidate={promotion.get('winner')}",
            f"val={float(promotion.get('validation_nll', float('nan'))):.5f} "
            f"test={float(promotion.get('test_nll', float('nan'))):.5f} "
            f"dVal_vs_v34={float(promotion.get('validation_gain_vs_v34', float('nan'))):+.5f} "
            f"dTest_vs_v34={float(promotion.get('test_gain_vs_v34', float('nan'))):+.5f}",
            f"tok/s={float(promotion.get('tokens_per_second', float('nan'))):,.0f} "
            f"speed={float(promotion.get('speed_ratio_vs_v34', float('nan'))):.3f}x "
            f"peak={float(promotion.get('peak_gib', float('nan'))):.2f}G "
            f"drift2K→64K={drift_text}",
            f"eligible={promotion.get('eligible')}",
        ]
    lines += [
        "",
        "AUTOMATIC NEXT STEP",
        f"action={promotion.get('action')}",
        "No 98M run is launched automatically.",
        "=" * width,
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    configure_tables(STRUCTURAL_ARMS)
    v32.validate_paths(args)
    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)

    reused_inference, reused_kernel = v33.validate_reused_phases(args, root)
    baseline = read_v34_baseline(Path(args.target_v34_root))
    install_v35_hooks(args)
    starts_audit = v33.paired_starts_selftest(args, root)
    frozen_starts = np.load(Path(args.target_v28_starts), allow_pickle=False)
    promotion_count = int(args.promotion_token_budget_v35) // int(args.train_seq)
    if frozen_starts.ndim != 1 or len(frozen_starts) < promotion_count:
        raise RuntimeError(
            f"v28 starts cannot support promotion: shape={frozen_starts.shape} need={promotion_count}"
        )

    optimizer_audit = optimizer_registry_selftest(args)
    preflight = {
        "version": VERSION,
        "arms": [asdict(arm) for arm in STRUCTURAL_ARMS],
        "optimizer_registry": optimizer_audit,
        "v34_baseline": baseline,
        "starts": starts_audit,
        "promotion_start_count": promotion_count,
        "promotion_prefix_sha256": hashlib.sha256(
            np.asarray(frozen_starts[:promotion_count], dtype=np.int64).tobytes()
        ).hexdigest(),
        "runtime": {
            "field_chunk": int(args.field_chunk),
            "triton_block_c": int(args.triton_block_c),
            "triton_chunk_t": int(args.triton_chunk_t),
        },
    }
    atomic_json(root / "v35_preflight.json", preflight)
    atomic_json(root / "args.json", vars(args))

    log("=" * 220)
    log("FIELD-FUSION v35r1 — PRE-RUN AUDIT")
    log(f"arms={[arm.name for arm in STRUCTURAL_ARMS]}")
    log(
        f"v34_baseline val={baseline['validation_nll']:.5f} "
        f"test={baseline['test_nll']:.5f} drift={baseline['context_2k_to_64k']:+.5f}"
    )
    log(
        f"paired_starts count={starts_audit['count']} "
        f"sha={starts_audit['prefix_bytes_sha256']} exact=True"
    )
    log(
        f"runtime field_chunk={args.field_chunk} BLOCK_C={args.triton_block_c} "
        f"CHUNK_T={args.triton_chunk_t}"
    )
    log(
        "optimizer_registry=PASS arms="
        + str([arm.name for arm in STRUCTURAL_ARMS])
        + f" legacy_fallback->{BASELINE_NAME}"
    )

    if args.package_selftest:
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
    ) = v34.prepare_training(args, root, device)

    log("=" * 220)
    log("PHASE A — 25.165824M STRUCTURAL GAP ABLATION")
    screen = run_structural_screen(
        args, root, deps, shapes, train, val_c, val, test_c, test, device
    )

    log("=" * 220)
    log("PHASE B — AUTOMATIC 49.152M STRUCTURAL CONFIRMATION")
    promotion = run_structural_promotion(
        args,
        root,
        deps,
        train,
        val_c,
        val,
        test_c,
        test,
        screen,
        baseline,
        device,
    )

    payload = {
        "version": VERSION,
        "args": vars(args),
        "canonical_source": str(canonical_path),
        "canonical_sha256": canonical_sha,
        "base_shape": asdict(base_shape),
        "reused_inference": reused_inference,
        "reused_kernel": reused_kernel,
        "v34_baseline": baseline,
        "screen": screen,
        "promotion": promotion,
        "target_v28_starts_sha256": sha256(Path(args.target_v28_starts)),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    atomic_json(root / "results.json", payload)
    summary = make_summary(
        reused_inference, reused_kernel, baseline, screen, promotion
    )
    atomic_text(root / "summary.txt", summary)
    log(summary)


if __name__ == "__main__":
    main()
