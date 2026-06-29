#!/usr/bin/env python3
"""FIELD-FUSION v29 — targeted quality ablation against frozen v28 targets.

No Transformer or pure Mamba-2 baseline is retrained.  Every candidate is a
Field-Fusion mutation trained on exactly the first 25,165,824 paired tokens from
v28.  The v28 results.json is treated as a frozen scoreboard and is SHA-recorded.

Candidates test four evidence-driven hypotheses:
  * replacing the weak final 2048 refresh with a second 1024 refresh;
  * causal low-rank block-Delta state editing after the strongest refreshes;
  * independent scalar/vector erase and write controls;
  * replacing four Field blocks with official fused Mamba-2 blocks as a hybrid
    control (not as a new baseline).

The run never launches a longer confirmation automatically.  A candidate earns
promotion only if it beats both frozen 25M rival targets on validation and test,
keeps parameters matched, preserves long-context stability, and stays inside
predeclared speed/memory guardrails.
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
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import field_fusion_scaling_confirmation_v28 as v28
import field_fusion_recipe_memory_v27 as v27
import field_fusion_final_ablation_v26 as v26
import field_fusion_wiki100_canonical_v23 as v23
import field_fusion_wiki100_mamba2_v25 as v25

VERSION = 29
FUSION = v27.FUSION
TRANSFORMER = v27.TRANSFORMER
MAMBA2 = v27.MAMBA2
EXPECTED_CANONICAL_SHA256 = v27.EXPECTED_CANONICAL_SHA256


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    description: str
    refresh_1024x2: bool = False
    delta_refresh_ids: Tuple[int, ...] = ()
    delta_mode: str = "none"          # none | coupled | scalar | vector
    delta_rank: int = 16
    delta_heads: int = 4
    delta_block: int = 64
    mamba_replace: Tuple[int, ...] = ()


# Block indices in the 24-block topology:
# [F0..F4,R0,F5..F9,R1,F10..F14,R2,F15..F19,R3]
MAMBA4_INDICES = (4, 10, 16, 22)
CANDIDATES: Tuple[CandidateSpec, ...] = (
    CandidateSpec(
        "field_refresh_1024x2",
        "Replace the weak 2048-token refresh window with a second 1024 window.",
        refresh_1024x2=True,
    ),
    CandidateSpec(
        "field_delta_coupled_r16_2",
        "Standard gated delta write after refreshes 512 and 1024.",
        delta_refresh_ids=(1, 2), delta_mode="coupled",
    ),
    CandidateSpec(
        "field_delta_scalar_r16_2",
        "Independent scalar retain/write after refreshes 512 and 1024.",
        delta_refresh_ids=(1, 2), delta_mode="scalar",
    ),
    CandidateSpec(
        "field_delta_vector_r16_2",
        "Per-channel retain/write after refreshes 512 and 1024.",
        delta_refresh_ids=(1, 2), delta_mode="vector",
    ),
    CandidateSpec(
        "field_delta_vector_r16_4",
        "Per-channel delta editors after all four refresh stations.",
        delta_refresh_ids=(0, 1, 2, 3), delta_mode="vector",
    ),
    CandidateSpec(
        "field_delta_vector_refresh1024x2",
        "Per-channel delta editors plus a duplicated 1024 refresh.",
        refresh_1024x2=True,
        delta_refresh_ids=(1, 2), delta_mode="vector",
    ),
    CandidateSpec(
        "field_mamba4_replace",
        "Hybrid control: replace the final Field block before each refresh with official Mamba-2.",
        mamba_replace=MAMBA4_INDICES,
    ),
    CandidateSpec(
        "field_mamba4_delta_vector",
        "Hybrid control plus vector delta editors after the 512/1024 refreshes.",
        delta_refresh_ids=(1, 2), delta_mode="vector",
        mamba_replace=MAMBA4_INDICES,
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


def nparams(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_args() -> argparse.Namespace:
    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument("--screen-token-budget", type=int, default=25_165_824)
    custom.add_argument("--eval-fractions", nargs="+", type=float, default=[0.50, 1.00])
    custom.add_argument("--checkpoint-every-updates", type=int, default=512)
    custom.add_argument("--profile-log-every-updates", type=int, default=128)
    custom.add_argument("--stream-readout-chunk", type=int, default=512)
    custom.add_argument("--screen-validation-token-budget", type=int, default=0)
    custom.add_argument("--screen-test-token-budget", type=int, default=293_944)
    custom.add_argument("--target-v28-results", default="/home/ubuntu/pcaf_runs/field_fusion_scaling_confirmation_v28_run/results.json")
    custom.add_argument("--target-v28-starts", default="/home/ubuntu/pcaf_runs/field_fusion_scaling_confirmation_v28_run/paired_example_starts.npy")
    custom.add_argument("--param-tolerance-pct", type=float, default=0.75)
    custom.add_argument("--min-speed-ratio", type=float, default=0.92)
    custom.add_argument("--max-memory-ratio", type=float, default=1.05)
    custom.add_argument("--max-context-drift", type=float, default=0.035)
    custom.add_argument("--long-contexts", nargs="+", type=int, default=[2048, 8192, 16384])
    custom.add_argument("--long-context-score-tokens", type=int, default=128)
    custom.add_argument("--long-context-windows", type=int, default=4)
    custom.add_argument("--shape-multiple", type=int, default=32)
    custom.add_argument("--delta-max-mix", type=float, default=0.50)
    custom.add_argument("--delta-gate-bias", type=float, default=-2.20)
    custom.add_argument("--delta-write-bias", type=float, default=-2.00)
    custom.add_argument("--delta-retain-init", type=float, default=0.985)
    custom.add_argument("--candidate", action="append", default=[])
    custom.add_argument("--run-preflight", action=argparse.BooleanOptionalAction, default=True)
    custom.add_argument("--export-winner-bf16", action=argparse.BooleanOptionalAction, default=True)
    custom_args, remaining = custom.parse_known_args()
    old = sys.argv
    try:
        sys.argv = [old[0], *remaining]
        args = v27.parse_args()
    finally:
        sys.argv = old
    for key, value in vars(custom_args).items():
        setattr(args, key, value)
    args.quality_token_budget = args.screen_token_budget
    v25.add_mamba_defaults(args)
    return args


def configure(args) -> None:
    v27.configure(args)
    v26.VERSION = VERSION


class BlockDeltaStateEditor(nn.Module):
    """Strictly causal block-level low-rank delta memory.

    Tokens in block j read a memory containing only completed blocks < j.  The
    matrix update follows a delta correction, with optional independent scalar
    or vector retention/write controls.  The branch is initialized near zero so
    every candidate begins close to the canonical Field model.
    """

    def __init__(self, dim: int, rank: int, heads: int, block_size: int,
                 mode: str, max_mix: float, gate_bias: float,
                 write_bias: float, retain_init: float) -> None:
        super().__init__()
        if dim <= 0 or rank < 2 or heads < 1 or block_size < 2:
            raise ValueError((dim, rank, heads, block_size))
        if mode not in {"coupled", "scalar", "vector"}:
            raise ValueError(mode)
        self.dim = int(dim)
        self.rank = int(rank)
        self.heads = int(heads)
        self.block_size = int(block_size)
        self.mode = str(mode)
        self.max_mix = float(max_mix)
        width = self.rank * self.heads

        self.norm = v25.FastRMSNorm(dim)
        self.q_proj = nn.Linear(dim, width, bias=False)
        self.k_proj = nn.Linear(dim, width, bias=False)
        self.v_proj = nn.Linear(dim, width, bias=False)
        gate_width = heads if mode != "vector" else width
        self.write_proj = nn.Linear(dim, gate_width, bias=True)
        self.retain_proj = None if mode == "coupled" else nn.Linear(dim, gate_width, bias=True)
        self.out_proj = nn.Linear(width, dim, bias=False)
        self.mix_proj = nn.Linear(dim, dim, bias=True)

        nn.init.xavier_uniform_(self.q_proj.weight, gain=0.50)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.50)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=0.50)
        nn.init.normal_(self.out_proj.weight, std=1.0e-3)
        nn.init.zeros_(self.write_proj.weight)
        nn.init.constant_(self.write_proj.bias, float(write_bias))
        if self.retain_proj is not None:
            nn.init.zeros_(self.retain_proj.weight)
            retain_init = min(max(float(retain_init), 1e-4), 1.0 - 1e-4)
            nn.init.constant_(self.retain_proj.bias, math.log(retain_init / (1.0 - retain_init)))
        nn.init.zeros_(self.mix_proj.weight)
        nn.init.constant_(self.mix_proj.bias, float(gate_bias))
        self.last_aux: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _normalize(z: torch.Tensor) -> torch.Tensor:
        return z * torch.rsqrt(z.square().sum(-1, keepdim=True).clamp_min(1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        h, r, bs = self.heads, self.rank, self.block_size
        z = self.norm(x)
        q = self._normalize(self.q_proj(z).view(b, t, h, r).float())

        pad = (-t) % bs
        zp = F.pad(z, (0, 0, 0, pad)) if pad else z
        nb = zp.shape[1] // bs
        summaries = zp.view(b, nb, bs, self.dim).mean(dim=2)
        k = self._normalize(self.k_proj(summaries).view(b, nb, h, r).float())
        v = torch.tanh(self.v_proj(summaries).view(b, nb, h, r).float())

        write_raw = torch.sigmoid(self.write_proj(summaries).float())
        if self.mode == "vector":
            write = write_raw.view(b, nb, h, r)
        else:
            write = write_raw.view(b, nb, h, 1)
        if self.mode == "coupled":
            retain = None
        else:
            assert self.retain_proj is not None
            retain_raw = torch.sigmoid(self.retain_proj(summaries).float())
            retain = retain_raw.view(b, nb, h, r if self.mode == "vector" else 1)

        memory = torch.zeros((b, h, r, r), device=x.device, dtype=torch.float32)
        exclusive: List[torch.Tensor] = []
        write_means: List[torch.Tensor] = []
        retain_means: List[torch.Tensor] = []
        for j in range(nb):
            exclusive.append(memory)
            kj = k[:, j]
            vj = v[:, j]
            pred = torch.einsum("bhij,bhj->bhi", memory, kj)
            error = vj - pred
            wj = write[:, j]
            delta = (wj * error)[..., :, None] * kj[..., None, :]
            if retain is None:
                memory = memory + delta
                retain_means.append((1.0 - wj).mean())
            else:
                rj = retain[:, j]
                memory = rj[..., :, None] * memory + delta
                retain_means.append(rj.mean())
            write_means.append(wj.mean())

        memory_by_block = torch.stack(exclusive, dim=1)
        block_ids = torch.div(
            torch.arange(t, device=x.device, dtype=torch.long), bs,
            rounding_mode="floor",
        ).clamp_max(nb - 1)
        token_memory = memory_by_block.index_select(1, block_ids)
        read = torch.einsum("bthij,bthj->bthi", token_memory, q)
        read = read.reshape(b, t, h * r).to(dtype=x.dtype)
        correction = self.out_proj(read)
        mix = self.max_mix * torch.sigmoid(self.mix_proj(z))
        out = x + mix * correction
        self.last_aux = {
            "mix": mix.detach().float().mean(),
            "write": torch.stack(write_means).detach().float().mean(),
            "retain": torch.stack(retain_means).detach().float().mean(),
            "blocks": torch.tensor(float(nb), device=x.device),
        }
        return out


class RefreshWithEditor(nn.Module):
    def __init__(self, refresh: nn.Module, editor: BlockDeltaStateEditor) -> None:
        super().__init__()
        self.refresh = refresh
        self.editor = editor
        self.window = int(getattr(refresh, "window", 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.editor(self.refresh(x))



def is_refresh_block(block: nn.Module) -> bool:
    return isinstance(block, (v23.v21.FusionRefreshBlockV21, RefreshWithEditor))


def is_official_mamba_block(block: nn.Module) -> bool:
    return isinstance(block, v25.Mamba2ResidualBlock)


def set_refresh_window(block: nn.Module, window: int) -> None:
    target = block.refresh if isinstance(block, RefreshWithEditor) else block
    if not isinstance(target, v23.v21.FusionRefreshBlockV21):
        raise TypeError(type(target))
    target.window = int(window)
    target.attn.local_window = int(window)


def install_candidate_checkpoint_policy(model: nn.Module, policy: str = "field_half") -> None:
    if policy not in {"none", "field_half", "field_all", "all"}:
        raise ValueError(policy)
    model._v29_checkpoint_policy = policy

    def states_logits(self, tokens: torch.Tensor):
        x = self.emb(tokens)
        self._patch_aux = x.new_zeros(())
        field_index = 0
        for i, block in enumerate(self.blocks):
            refresh = is_refresh_block(block)
            mamba = is_official_mamba_block(block)
            native_field = not refresh and not mamba
            use = False
            if self.training and torch.is_grad_enabled():
                if policy == "all":
                    # Official Mamba custom autograd is intentionally excluded.
                    use = not mamba
                elif policy == "field_all":
                    use = native_field
                elif policy == "field_half":
                    use = native_field and field_index % 2 == 0
            if use:
                x = checkpoint(block, x, use_reentrant=False, preserve_rng_state=False)
            else:
                x = block(x)
            if native_field:
                field_index += 1
            if i == self.patch_position and self.softpatch is not None:
                x = self.softpatch(x, tokens)
                self._patch_aux = self.softpatch.last_aux
        return x, self.lm_head(self.final_norm(x))

    model.states_logits = types.MethodType(states_logits, model)


def patch_candidate(model: nn.Module, spec: CandidateSpec, args, device: torch.device) -> nn.Module:
    if spec.refresh_1024x2:
        refreshes = [b for b in model.blocks if isinstance(b, v23.v21.FusionRefreshBlockV21)]
        if len(refreshes) != 4:
            raise AssertionError(f"expected four refreshes, got {len(refreshes)}")
        set_refresh_window(refreshes[-1], 1024)

    if spec.mamba_replace:
        torch.manual_seed(args.model_seed + 290_400)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.model_seed + 290_400)
        for index in spec.mamba_replace:
            if is_refresh_block(model.blocks[index]):
                raise AssertionError(f"cannot replace refresh block {index}")
            model.blocks[index] = v25.Mamba2ResidualBlock(model.emb.embedding_dim, args).to(device)

    if spec.delta_refresh_ids:
        torch.manual_seed(args.model_seed + 290_700)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.model_seed + 290_700)
        refresh_seen = 0
        for index, block in enumerate(list(model.blocks)):
            if isinstance(block, v23.v21.FusionRefreshBlockV21):
                if refresh_seen in spec.delta_refresh_ids:
                    editor = BlockDeltaStateEditor(
                        model.emb.embedding_dim, spec.delta_rank, spec.delta_heads,
                        spec.delta_block, spec.delta_mode, args.delta_max_mix,
                        args.delta_gate_bias, args.delta_write_bias,
                        args.delta_retain_init,
                    ).to(device)
                    model.blocks[index] = RefreshWithEditor(block, editor)
                refresh_seen += 1
        if refresh_seen != 4:
            raise AssertionError(f"expected four refresh stations, got {refresh_seen}")

    install_candidate_checkpoint_policy(model, args.fusion_checkpoint_policy)
    return model


def build_base_field(shape: v23.Shape, args, deps, device: torch.device) -> nn.Module:
    old = args.fusion_checkpoint_policy
    args.fusion_checkpoint_policy = "none"
    try:
        model = v25.build_model_v25(FUSION, shape, args, deps, device)
    finally:
        args.fusion_checkpoint_policy = old
    return model


def build_candidate(spec: CandidateSpec, shape: v23.Shape, args, deps,
                    device: torch.device) -> nn.Module:
    model = build_base_field(shape, args, deps, device)
    return patch_candidate(model, spec, args, device)


def component_accounting(args, deps, base_shape: v23.Shape) -> Dict[str, int]:
    """Count fixed components once, then solve every candidate analytically."""
    model = build_base_field(base_shape, args, deps, torch.device("cpu"))
    total = nparams(model)
    refresh_cls = v23.v21.FusionRefreshBlockV21
    field_blocks = [b for b in model.blocks if not isinstance(b, refresh_cls)]
    if len(field_blocks) != 20:
        raise AssertionError(f"expected 20 Field blocks, got {len(field_blocks)}")
    field_block_total = nparams(field_blocks[0])
    ff_params = 3 * base_shape.dim * base_shape.ff_hidden
    field_mixer_fixed = field_block_total - ff_params
    if field_mixer_fixed <= 0:
        raise AssertionError((field_block_total, ff_params))
    del model
    gc.collect()

    mamba = v25.Mamba2ResidualBlock(base_shape.dim, args)
    mamba_params = nparams(mamba)
    del mamba
    editor_counts = {}
    for mode in ("coupled", "scalar", "vector"):
        editor = BlockDeltaStateEditor(
            base_shape.dim, 16, 4, 64, mode,
            args.delta_max_mix, args.delta_gate_bias,
            args.delta_write_bias, args.delta_retain_init,
        )
        editor_counts[mode] = nparams(editor)
        del editor
    gc.collect()
    return {
        "base_total": int(total),
        "base_hidden": int(base_shape.ff_hidden),
        "field_block_total": int(field_block_total),
        "field_mixer_fixed": int(field_mixer_fixed),
        "mamba_block": int(mamba_params),
        **{f"editor_{k}": int(v) for k, v in editor_counts.items()},
    }


def solve_candidate_shapes(args, deps) -> Tuple[v23.Shape, Dict[str, v23.Shape], Dict[str, int]]:
    base_shape = v25.solve_shapes_v25(args, deps)[FUSION]
    accounting = component_accounting(args, deps, base_shape)
    dim = base_shape.dim
    # Baseline contains 24 SwiGLU FFNs (20 Field + four refresh).
    base_fixed = accounting["base_total"] - 3 * dim * 24 * base_shape.ff_hidden
    shapes: Dict[str, v23.Shape] = {}
    multiple = int(args.shape_multiple)
    for spec in selected_candidates(args):
        replaced = len(spec.mamba_replace)
        ff_blocks = 24 - replaced
        fixed = base_fixed
        fixed -= replaced * accounting["field_mixer_fixed"]
        fixed += replaced * accounting["mamba_block"]
        if spec.delta_mode != "none":
            key = f"editor_{spec.delta_mode}"
            # Rank/head defaults are frozen in this v29 screen.
            fixed += len(spec.delta_refresh_ids) * accounting[key]
        slope = 3 * dim * ff_blocks
        raw = (args.target_params - fixed) / max(slope, 1)
        hidden = int(round(raw / multiple) * multiple)
        hidden = max(args.min_ff_hidden, min(args.max_ff_hidden, hidden))
        estimated = int(fixed + slope * hidden)
        shape = v23.Shape(spec.name, estimated, dim, base_shape.layers,
                          base_shape.heads, hidden)
        delta = 100.0 * (estimated - args.target_params) / args.target_params
        if abs(delta) > args.param_tolerance_pct:
            raise RuntimeError(
                f"parameter mismatch {spec.name}: {delta:+.3f}% "
                f"hidden={hidden} estimated={estimated:,}"
            )
        shapes[spec.name] = shape
    return base_shape, shapes, accounting


def selected_candidates(args) -> Tuple[CandidateSpec, ...]:
    if not args.candidate:
        return CANDIDATES
    requested = set(args.candidate)
    out = tuple(spec for spec in CANDIDATES if spec.name in requested)
    missing = requested - {spec.name for spec in out}
    if missing:
        raise ValueError(f"unknown candidates: {sorted(missing)}")
    return out


def load_frozen_targets(args, root: Path) -> Dict[str, object]:
    path = Path(args.target_v28_results)
    if not path.is_file():
        raise FileNotFoundError(
            f"v28 frozen scoreboard missing: {path}. Keep the v28 run directory."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("results", {})
    target_tokens = int(args.screen_token_budget)
    by_model: Dict[str, Dict[str, float]] = {}
    for result in rows.values():
        model = result.get("model")
        for evaluation in result.get("evaluations", []):
            if int(evaluation.get("train_tokens", -1)) == target_tokens:
                by_model[model] = {
                    "validation_nll": float(evaluation["validation"]["nll"]),
                    "test_nll": float(evaluation["test"]["nll"]),
                }
                break
    for model in (FUSION, TRANSFORMER, MAMBA2):
        if model not in by_model:
            raise RuntimeError(f"v28 scoreboard has no {target_tokens:,}-token row for {model}")
    final = {
        model: {
            "validation_nll": float(result["final_validation"]["nll"]),
            "test_nll": float(result["final_test"]["nll"]),
            "tokens_per_second": float(result["tokens_per_second"]),
            "peak_gib": float(result["peak_gib"]),
        }
        for result in rows.values()
        for model in [result.get("model")]
        if model in {FUSION, TRANSFORMER, MAMBA2}
    }
    out = {
        "path": str(path),
        "sha256": sha256(path),
        "screen_tokens": target_tokens,
        "screen": by_model,
        "final_98m": final,
        "quality_meta_validation": min(by_model[TRANSFORMER]["validation_nll"], by_model[MAMBA2]["validation_nll"]),
        "quality_meta_test": min(by_model[TRANSFORMER]["test_nll"], by_model[MAMBA2]["test_nll"]),
        "field_speed_reference": float(final[FUSION]["tokens_per_second"]),
        "field_memory_reference": float(final[FUSION]["peak_gib"]),
    }
    atomic_json(root / "frozen_v28_targets.json", out)
    return out


def audit_starts(starts: np.ndarray, args, root: Path) -> Dict[str, object]:
    path = Path(args.target_v28_starts)
    if not path.is_file():
        raise FileNotFoundError(f"v28 paired starts missing: {path}")
    old = np.load(path)
    if len(old) < len(starts):
        raise RuntimeError(f"v28 starts shorter than v29 screen: {len(old)} < {len(starts)}")
    equal = bool(np.array_equal(starts, old[:len(starts)]))
    row = {
        "v28_path": str(path),
        "v28_sha256": hashlib.sha256(old.tobytes()).hexdigest(),
        "v29_sha256": hashlib.sha256(starts.tobytes()).hexdigest(),
        "count": int(len(starts)),
        "prefix_equal": equal,
    }
    if not equal:
        raise AssertionError("v29 paired starts do not equal the v28 prefix")
    atomic_json(root / "paired_prefix_audit.json", row)
    return row


def candidate_fingerprint(model: nn.Module) -> str:
    return v23.v22.module_hash(model)


def architecture_audit(specs: Sequence[CandidateSpec], shapes: Mapping[str, v23.Shape],
                       args, deps, device: torch.device, root: Path) -> Dict[str, object]:
    rows: Dict[str, object] = {}
    for spec in specs:
        shape = shapes[spec.name]
        model = build_candidate(spec, shape, args, deps, device)
        actual = nparams(model)
        delta = 100.0 * (actual - args.target_params) / args.target_params
        refresh_windows = []
        delta_editors = 0
        mamba_blocks = 0
        for block in model.blocks:
            if isinstance(block, RefreshWithEditor):
                refresh_windows.append(int(block.refresh.attn.local_window))
                delta_editors += 1
            elif isinstance(block, v23.v21.FusionRefreshBlockV21):
                refresh_windows.append(int(block.attn.local_window))
            elif is_official_mamba_block(block):
                mamba_blocks += 1
        if actual != shape.params:
            raise AssertionError(f"shape estimate mismatch {spec.name}: {shape.params} vs {actual}")
        rows[spec.name] = {
            "params": actual,
            "delta_pct": delta,
            "ff_hidden": shape.ff_hidden,
            "refresh_windows": refresh_windows,
            "delta_editors": delta_editors,
            "mamba_blocks": mamba_blocks,
            "fingerprint": candidate_fingerprint(model),
        }
        log(
            f"[architecture] {spec.name:38s} params={actual:,} d={delta:+.3f}% "
            f"ff={shape.ff_hidden} refresh={refresh_windows} delta={delta_editors} mamba={mamba_blocks}"
        )
        del model
        clear_cuda()
    atomic_json(root / "architecture_audit.json", rows)
    return rows


def causality_and_backward_preflight(specs: Sequence[CandidateSpec], shapes,
                                     args, deps, device, root) -> Dict[str, object]:
    rows = {}
    tokens = 129
    for spec in specs:
        model = build_candidate(spec, shapes[spec.name], args, deps, device).train()
        torch.manual_seed(args.eval_seed + 29)
        x = torch.randint(0, args.vocab_size, (1, tokens), device=device)
        y = torch.randint(0, args.vocab_size, (1, tokens), device=device)
        prefix = tokens // 2
        x2 = x.clone()
        x2[:, prefix:] = torch.randint(0, args.vocab_size, x2[:, prefix:].shape, device=device)
        with v23.amp_ctx(device, args.amp):
            s1, _ = model.states_logits(x)
            s2, _ = model.states_logits(x2)
            causal_max = float((s1[:, :prefix] - s2[:, :prefix]).abs().max().float().cpu())
            loss, primary = v25.loss_call_v25(FUSION, model, x, y)
        loss.backward()
        finite = bool(torch.isfinite(loss).item()) and all(
            p.grad is None or bool(torch.isfinite(p.grad).all().item())
            for p in model.parameters()
        )
        special_grad = 0.0
        for name, parameter in model.named_parameters():
            if ("editor" in name or "mixer" in name) and parameter.grad is not None:
                special_grad += float(parameter.grad.detach().float().norm().cpu())
        row = {
            "causal_max_abs": causal_max,
            "finite_backward": finite,
            "loss": float(loss.detach().float().cpu()),
            "primary": float(primary.detach().float().cpu()),
            "special_grad_norm_sum": special_grad,
            "pass": finite and causal_max <= 5e-3,
        }
        if not row["pass"]:
            raise AssertionError(f"preflight failed {spec.name}: {row}")
        rows[spec.name] = row
        log(f"[preflight] {spec.name:38s} causal={causal_max:.3e} finite={finite} special_grad={special_grad:.3e}")
        del model, loss, primary, s1, s2
        clear_cuda()
    atomic_json(root / "candidate_preflight.json", rows)
    return rows


def make_candidate_optimizer(model: nn.Module, args):
    groups: Dict[Tuple[bool, bool], List[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        delta_gate = any(key in name for key in (
            ".editor.write_proj", ".editor.retain_proj", ".editor.mix_proj",
        ))
        special = v26.is_special_parameter(name) or delta_gate
        decay = parameter.ndim >= 2
        groups.setdefault((special, decay), []).append(parameter)
    param_groups = [
        {
            "params": params,
            "weight_decay": args.weight_decay if decay else 0.0,
            "lr_scale": 2.0 if special else 1.0,
        }
        for (special, decay), params in groups.items()
    ]
    kwargs = dict(lr=args.lr, betas=(0.9, 0.95), eps=1.0e-8)
    try:
        return torch.optim.AdamW(param_groups, fused=True, **kwargs)
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(param_groups, **kwargs)


def checkpoint_signature(args, spec: CandidateSpec, shape: v23.Shape,
                         total_sequences: int) -> Dict[str, object]:
    return {
        "version": VERSION,
        "candidate": asdict(spec),
        "shape": asdict(shape),
        "screen_token_budget": int(args.screen_token_budget),
        "total_sequences": int(total_sequences),
        "train_seq": int(args.train_seq),
        "model_seed": int(args.model_seed),
        "embedding_seed": int(args.embedding_seed),
        "data_seed": int(args.data_seed),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "warmup_fraction": float(args.warmup_fraction),
        "wsd_stable_fraction": float(args.wsd_stable_fraction),
        "checkpoint_policy": str(args.fusion_checkpoint_policy),
    }


def save_checkpoint(path: Path, model: nn.Module, optimizer, sequence_index: int,
                    history: List[Dict[str, object]], evaluations: List[Dict[str, object]],
                    compute_seconds: float, signature: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({
        "signature": dict(signature),
        "sequence_index": int(sequence_index),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "history": history,
        "evaluations": evaluations,
        "compute_seconds": float(compute_seconds),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }, tmp)
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
    if raw.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(raw["cuda_rng"])
    return raw


def evaluate_streaming(model: nn.Module, corpus, data: torch.Tensor, args,
                       device: torch.device, token_budget: int) -> Dict[str, float]:
    return v28.evaluate_streaming_corpus(
        FUSION, model, corpus, data, args.train_seq, token_budget,
        args.stream_readout_chunk, device, args.amp,
    )


@dataclass
class ScreenResult:
    candidate: str
    description: str
    params: int
    param_delta_pct: float
    ff_hidden: int
    updates: int
    train_tokens: int
    compute_seconds: float
    tokens_per_second: float
    peak_gib: float
    evaluations: List[Dict[str, object]]
    final_validation: Dict[str, float]
    final_test: Dict[str, float]
    checkpoint: str


def train_candidate(spec: CandidateSpec, shape: v23.Shape, args, deps,
                    train: torch.Tensor, val_c, val: torch.Tensor,
                    test_c, test: torch.Tensor, starts: np.ndarray,
                    root: Path, device: torch.device) -> ScreenResult:
    out = root / "candidates" / spec.name
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "result.json"
    if result_path.is_file() and args.resume:
        return ScreenResult(**json.loads(result_path.read_text(encoding="utf-8")))

    model = build_candidate(spec, shape, args, deps, device).train()
    optimizer = make_candidate_optimizer(model, args)
    total_sequences = len(starts)
    total_updates = total_sequences // args.batch_size
    signature = checkpoint_signature(args, spec, shape, total_sequences)
    checkpoint_path = out / "latest.pt"
    sequence_index = 0
    history: List[Dict[str, object]] = []
    evaluations: List[Dict[str, object]] = []
    prior_compute = 0.0
    if args.resume:
        raw = load_checkpoint(checkpoint_path, model, optimizer, signature)
        if raw is not None:
            sequence_index = int(raw["sequence_index"])
            history = list(raw.get("history", []))
            evaluations = list(raw.get("evaluations", []))
            prior_compute = float(raw.get("compute_seconds", 0.0))
            log(f"[{spec.name}] resume update={sequence_index // args.batch_size}/{total_updates}")

    milestone_tokens = sorted(set(
        int(round(args.screen_token_budget * f / (args.train_seq * args.batch_size)))
        * args.train_seq * args.batch_size
        for f in args.eval_fractions
    ))
    if milestone_tokens[-1] != args.screen_token_budget:
        milestone_tokens[-1] = args.screen_token_budget
    milestone_sequences = {t // args.train_seq: t for t in milestone_tokens}
    completed = {int(x["train_tokens"]) for x in evaluations}

    clear_cuda()
    torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    started = time.perf_counter()
    excluded = 0.0

    def run_eval(tokens: int, update: int, lr: float) -> None:
        nonlocal excluded
        if tokens in completed:
            return
        sync(device)
        pause = time.perf_counter()
        model.eval()
        vr = evaluate_streaming(model, val_c, val, args, device,
                                args.screen_validation_token_budget)
        tr = evaluate_streaming(model, test_c, test, args, device,
                                args.screen_test_token_budget)
        row = {"train_tokens": int(tokens), "update": int(update), "lr": float(lr),
               "validation": vr, "test": tr}
        evaluations.append(row)
        evaluations.sort(key=lambda x: int(x["train_tokens"]))
        completed.add(int(tokens))
        atomic_json(out / "evaluations.json", evaluations)
        log(f"[{spec.name}] VAL tokens={tokens:,} nll={vr['nll']:.5f} ppl={vr['ppl']:.3f}")
        log(f"[{spec.name}] TEST tokens={tokens:,} nll={tr['nll']:.5f} ppl={tr['ppl']:.3f}")
        model.train()
        sync(device)
        excluded += time.perf_counter() - pause

    resumed_tokens = sequence_index * args.train_seq
    if sequence_index in milestone_sequences and resumed_tokens not in completed:
        lr = v26.lr_for_tokens(resumed_tokens, args.screen_token_budget, args, "wsd")
        run_eval(resumed_tokens, sequence_index // args.batch_size, lr)

    while sequence_index < total_sequences:
        x, y = v26.paired_batch(
            train, starts, sequence_index, args.batch_size, args.train_seq, device
        )
        processed_after = (sequence_index + args.batch_size) * args.train_seq
        distill = min(1.0, processed_after / max(args.screen_token_budget * args.warmup_fraction, 1.0))
        v25.set_distill_v25(model, distill)
        optimizer.zero_grad(set_to_none=True)
        with v23.amp_ctx(device, args.amp):
            loss, primary = v25.loss_call_v25(FUSION, model, x, y)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        lr = v26.lr_for_tokens(processed_after, args.screen_token_budget, args, "wsd")
        v26.set_optimizer_lr(optimizer, lr)
        optimizer.step()
        sequence_index += args.batch_size
        update = sequence_index // args.batch_size

        if update % args.profile_log_every_updates == 0 or sequence_index in milestone_sequences:
            sync(device)
            compute = prior_compute + time.perf_counter() - started - excluded
            row = {
                "update": update,
                "train_tokens": processed_after,
                "train_nll": float(primary.detach().float().cpu()),
                "grad": float(grad.detach().float().cpu()),
                "lr": lr,
                "tokens_per_second": processed_after / max(compute, 1e-9),
                "peak_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            }
            history.append(row)
            atomic_json(out / "history.json", history)
            log(
                f"[{spec.name}] update={update:04d}/{total_updates} tokens={processed_after:,}/"
                f"{args.screen_token_budget:,} nll={row['train_nll']:.5f} lr={lr:.3e} "
                f"tok/s={row['tokens_per_second']:,.0f} peak={row['peak_gib']:.2f}G"
            )

        at_milestone = sequence_index in milestone_sequences
        if update % args.checkpoint_every_updates == 0 or at_milestone or sequence_index == total_sequences:
            sync(device)
            compute = prior_compute + time.perf_counter() - started - excluded
            pause = time.perf_counter()
            save_checkpoint(checkpoint_path, model, optimizer, sequence_index,
                            history, evaluations, compute, signature)
            sync(device)
            excluded += time.perf_counter() - pause
        if at_milestone:
            run_eval(processed_after, update, lr)

    sync(device)
    compute_seconds = prior_compute + time.perf_counter() - started - excluded
    peak_gib = torch.cuda.max_memory_allocated(device) / 2**30
    if not evaluations or int(evaluations[-1]["train_tokens"]) != args.screen_token_budget:
        raise RuntimeError(f"missing final evaluation for {spec.name}")
    actual = nparams(model)
    result = ScreenResult(
        candidate=spec.name,
        description=spec.description,
        params=actual,
        param_delta_pct=100.0 * (actual - args.target_params) / args.target_params,
        ff_hidden=shape.ff_hidden,
        updates=total_updates,
        train_tokens=args.screen_token_budget,
        compute_seconds=compute_seconds,
        tokens_per_second=args.screen_token_budget / max(compute_seconds, 1e-9),
        peak_gib=peak_gib,
        evaluations=evaluations,
        final_validation=dict(evaluations[-1]["validation"]),
        final_test=dict(evaluations[-1]["test"]),
        checkpoint=str(checkpoint_path),
    )
    atomic_json(result_path, asdict(result))
    del model, optimizer
    clear_cuda()
    return result


def load_model_from_result(spec: CandidateSpec, shape: v23.Shape,
                           result: ScreenResult, args, deps, device) -> nn.Module:
    model = build_candidate(spec, shape, args, deps, device)
    raw = torch.load(result.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(raw["model"], strict=True)
    return model.eval()


def long_context_eval(model: nn.Module, test: torch.Tensor, args,
                      device: torch.device) -> Dict[str, Dict[str, float]]:
    return v28.evaluate_matched_suffix_streaming(
        FUSION, model, test, args.long_contexts,
        args.long_context_score_tokens, args.long_context_windows,
        args.eval_seed + 290_000, args.stream_readout_chunk,
        device, args.amp,
    )


def make_decision(results: Mapping[str, ScreenResult], targets: Mapping[str, object],
                  contexts: Mapping[str, Mapping[str, Dict[str, float]]], args) -> Dict[str, object]:
    val_meta = float(targets["quality_meta_validation"])
    test_meta = float(targets["quality_meta_test"])
    field_speed = float(targets["field_speed_reference"])
    field_memory = float(targets["field_memory_reference"])
    ranked = sorted(results.values(), key=lambda r: (r.final_validation["nll"], r.final_test["nll"]))
    rows = []
    for result in ranked:
        ctx = contexts.get(result.candidate, {})
        c2 = float(ctx.get("2048", {}).get("nll", float("nan")))
        c16 = float(ctx.get("16384", {}).get("nll", float("nan")))
        drift = c16 - c2 if math.isfinite(c2) and math.isfinite(c16) else float("inf")
        checks = {
            "beats_transformer_validation": result.final_validation["nll"] < targets["screen"][TRANSFORMER]["validation_nll"],
            "beats_mamba_validation": result.final_validation["nll"] < targets["screen"][MAMBA2]["validation_nll"],
            "beats_transformer_test": result.final_test["nll"] < targets["screen"][TRANSFORMER]["test_nll"],
            "beats_mamba_test": result.final_test["nll"] < targets["screen"][MAMBA2]["test_nll"],
            "speed_guard": result.tokens_per_second >= field_speed * args.min_speed_ratio,
            "memory_guard": result.peak_gib <= field_memory * args.max_memory_ratio,
            "parameter_guard": abs(result.param_delta_pct) <= args.param_tolerance_pct,
            "context_guard": drift <= args.max_context_drift,
        }
        rows.append({
            "candidate": result.candidate,
            "validation_nll": result.final_validation["nll"],
            "test_nll": result.final_test["nll"],
            "validation_minus_meta": result.final_validation["nll"] - val_meta,
            "test_minus_meta": result.final_test["nll"] - test_meta,
            "tokens_per_second": result.tokens_per_second,
            "peak_gib": result.peak_gib,
            "context_drift_2k_to_16k": drift,
            "checks": checks,
            "eligible": bool(all(checks.values())),
        })
    eligible = [row for row in rows if row["eligible"]]
    winner = eligible[0]["candidate"] if eligible else ranked[0].candidate
    if eligible:
        action = "PROMOTE_WINNER_TO_50M_THEN_98M_FULL_ROUND"
        reason = "At least one Field candidate beat both frozen v28 rivals on validation/test and passed speed, memory, parameter, and long-context guards."
    else:
        action = "NO_FULL_ROUND_REFINE_TOP_FIELD_MECHANISM"
        reason = "No candidate cleared every frozen v28 quality and systems target; do not rerun Transformer or Mamba-2."
    return {
        "action": action,
        "reason": reason,
        "winner": winner,
        "eligible_candidates": [row["candidate"] for row in eligible],
        "frozen_targets": targets,
        "ranked": rows,
    }


def export_bf16(path: Path, spec: CandidateSpec, shape: v23.Shape,
                result: ScreenResult, args, deps, device) -> None:
    model = load_model_from_result(spec, shape, result, args, deps, device)
    state = {}
    for key, value in model.state_dict().items():
        cpu = value.detach().cpu()
        state[key] = cpu.to(torch.bfloat16) if cpu.is_floating_point() else cpu
    tmp = path.with_suffix(".tmp")
    torch.save({
        "format": "field_fusion_v29_bf16_candidate",
        "version": VERSION,
        "candidate": asdict(spec),
        "shape": asdict(shape),
        "train_tokens": result.train_tokens,
        "state_dict": state,
        "args": vars(args),
    }, tmp)
    os.replace(tmp, path)
    del model
    clear_cuda()


def summary(args, canonical_path, canonical_sha, base_shape, shapes, accounting,
            targets, architecture, results, contexts, decision, prefix) -> str:
    width = 220
    lines = [
        "=" * width,
        "FIELD-FUSION v29 — TARGETED DELTA / HYBRID QUALITY ABLATION",
        "=" * width,
        f"canonical={canonical_path} sha256={canonical_sha}",
        f"paired screen tokens/candidate={args.screen_token_budget:,} context={args.train_seq} batch={args.batch_size} WSD gateLR=2x",
        f"v28_scoreboard={targets['path']} sha256={targets['sha256']} prefix_equal={prefix['prefix_equal']}",
        "No Transformer or pure Mamba-2 model was retrained.",
        "",
        "FROZEN v28 TARGETS AT MATCHED 25.165824M TOKENS",
        f"Transformer val={targets['screen'][TRANSFORMER]['validation_nll']:.5f} test={targets['screen'][TRANSFORMER]['test_nll']:.5f}",
        f"Mamba-2    val={targets['screen'][MAMBA2]['validation_nll']:.5f} test={targets['screen'][MAMBA2]['test_nll']:.5f}",
        f"Field-v28  val={targets['screen'][FUSION]['validation_nll']:.5f} test={targets['screen'][FUSION]['test_nll']:.5f}",
        "",
        "CANDIDATE RESULTS",
        f"{'candidate':40s} {'params':>12s} {'d%':>7s} {'ff':>5s} {'val NLL':>10s} {'test NLL':>10s} {'tok/s':>10s} {'peakGB':>8s} {'2K→16K':>10s}",
    ]
    for row in decision["ranked"]:
        r = results[row["candidate"]]
        lines.append(
            f"{r.candidate:40s} {r.params:12,d} {r.param_delta_pct:+7.3f} {r.ff_hidden:5d} "
            f"{r.final_validation['nll']:10.5f} {r.final_test['nll']:10.5f} "
            f"{r.tokens_per_second:10,.0f} {r.peak_gib:8.2f} {row['context_drift_2k_to_16k']:+10.5f}"
        )
    lines += ["", "PROMOTION CHECKS"]
    for row in decision["ranked"]:
        c = row["checks"]
        lines.append(
            f"{row['candidate']:40s} eligible={row['eligible']} "
            f"Tval={c['beats_transformer_validation']} Mval={c['beats_mamba_validation']} "
            f"Ttest={c['beats_transformer_test']} Mtest={c['beats_mamba_test']} "
            f"speed={c['speed_guard']} memory={c['memory_guard']} params={c['parameter_guard']} ctx={c['context_guard']}"
        )
    lines += [
        "",
        "AUTOMATIC NEXT STEP",
        f"action={decision['action']}",
        f"winner={decision['winner']}",
        f"eligible={','.join(decision['eligible_candidates']) if decision['eligible_candidates'] else 'none'}",
        f"reason={decision['reason']}",
        "No longer run is launched automatically.",
        "=" * width,
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required")
    if args.screen_token_budget % (args.train_seq * args.batch_size):
        raise ValueError("screen-token-budget must divide batch*sequence exactly")
    configure(args)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.set_float32_matmul_precision("high")

    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)
    canonical_path, canonical_sha, deps = v27.load_dependencies(args)
    specs = selected_candidates(args)
    base_shape, shapes, accounting = solve_candidate_shapes(args, deps)
    atomic_json(root / "component_accounting.json", accounting)
    targets = load_frozen_targets(args, root)

    architecture = architecture_audit(specs, shapes, args, deps, device, root)
    preflight = causality_and_backward_preflight(specs, shapes, args, deps, device, root) if args.run_preflight else {}

    raw_rows = v23.core.load_raw_rows(args.cache_dir, args.data_frac)
    tokenizer = v23.base.copy_or_train_tokenizer(
        root, raw_rows[0], args.vocab_size,
        args.tokenizer_min_frequency, args.tokenizer_source,
    )
    train_c, val_c, test_c = v23.core.save_or_load_corpora(root, tokenizer, raw_rows)
    train = v23.core.place_tokens(train_c.tokens, device, args.data_device, "train")
    val = v23.core.place_tokens(val_c.tokens, device, args.data_device, "validation")
    test = v23.core.place_tokens(test_c.tokens, device, args.data_device, "test")

    total_sequences = args.screen_token_budget // args.train_seq
    starts = v26.make_example_starts(
        total_sequences, len(train) - args.train_seq - 1,
        args.data_seed, root / "paired_example_starts.npy",
    )
    prefix = audit_starts(starts, args, root)

    results: Dict[str, ScreenResult] = {}
    for spec in specs:
        log("=" * 200)
        log(f"FIELD CANDIDATE: {spec.name} — {spec.description}")
        results[spec.name] = train_candidate(
            spec, shapes[spec.name], args, deps,
            train, val_c, val, test_c, test, starts, root, device,
        )
        atomic_json(root / "candidate_results.json", {k: asdict(v) for k, v in results.items()})

    preliminary = sorted(results.values(), key=lambda r: (r.final_validation["nll"], r.final_test["nll"]))
    # Always profile the two quality leaders, plus every candidate that already
    # clears the frozen validation/test and basic speed/memory/parameter gates.
    context_names = {row.candidate for row in preliminary[:2]}
    for row in preliminary:
        quality_ok = (
            row.final_validation["nll"] < targets["quality_meta_validation"]
            and row.final_test["nll"] < targets["quality_meta_test"]
        )
        systems_ok = (
            row.tokens_per_second >= targets["field_speed_reference"] * args.min_speed_ratio
            and row.peak_gib <= targets["field_memory_reference"] * args.max_memory_ratio
            and abs(row.param_delta_pct) <= args.param_tolerance_pct
        )
        if quality_ok and systems_ok:
            context_names.add(row.candidate)
    context_names = [row.candidate for row in preliminary if row.candidate in context_names]
    contexts: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name in context_names:
        spec = next(s for s in specs if s.name == name)
        model = load_model_from_result(spec, shapes[name], results[name], args, deps, device)
        contexts[name] = long_context_eval(model, test, args, device)
        atomic_json(root / "long_context_results.json", contexts)
        del model
        clear_cuda()

    decision = make_decision(results, targets, contexts, args)
    atomic_json(root / "decision.json", decision)

    if args.export_winner_bf16:
        winner = decision["winner"]
        spec = next(s for s in specs if s.name == winner)
        export_dir = root / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        export_bf16(
            export_dir / f"{winner}_step{results[winner].updates}_BF16.pt",
            spec, shapes[winner], results[winner], args, deps, device,
        )

    payload = {
        "version": VERSION,
        "args": vars(args),
        "canonical_source": str(canonical_path),
        "canonical_sha256": canonical_sha,
        "base_shape": asdict(base_shape),
        "candidate_shapes": {k: asdict(v) for k, v in shapes.items()},
        "component_accounting": accounting,
        "frozen_targets": targets,
        "architecture_audit": architecture,
        "candidate_preflight": preflight,
        "paired_prefix_audit": prefix,
        "results": {k: asdict(v) for k, v in results.items()},
        "long_contexts": contexts,
        "decision": decision,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "mamba_ssm": v25.MAMBA_VERSION,
    }
    atomic_json(root / "results.json", payload)
    text = summary(
        args, canonical_path, canonical_sha, base_shape, shapes, accounting,
        targets, architecture, results, contexts, decision, prefix,
    )
    atomic_text(root / "summary.txt", text)
    log(text)


if __name__ == "__main__":
    main()
