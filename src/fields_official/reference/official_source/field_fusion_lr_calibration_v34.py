#!/usr/bin/env python3
"""FIELD-FUSION v34 — specialized-LR calibration and 49M confirmation.

This run is deliberately narrow and reuses the validated v32 systems results and
exact v33 runtime geometry.  It performs:

A) a paired 25.165824M-token screen of four LR calibrations plus the canonical
   baseline, evaluating every arm at 2K/16K/64K;
B) an automatic from-scratch 49.152M confirmation of the best strictly eligible
   arm, compared against the frozen v30 49M hybrid baseline.

No Transformer or pure Mamba model is trained or benchmarked again.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

import field_fusion_gap_closure_v33 as v33
import field_fusion_gap_kernel_v32 as v32
import field_fusion_delta_quality_ablation_v29 as v29
import field_fusion_recipe_memory_v27 as v27
import field_fusion_wiki100_canonical_v23 as v23

VERSION = 34
BASELINE_NAME = "gap_ref_stable70"
V30_BASELINE_NAME = "field_mamba4_refresh1024x2_49m"

# The screen requested after v33: preserve the canonical schedule/topology and
# calibrate only the specialized learning-rate multipliers.
LR_ARMS: Tuple[v32.GapArm, ...] = (
    v32.GapArm(
        BASELINE_NAME,
        "Canonical v31/v33 recipe: refresh LR 1.00x, Mamba LR 1.00x.",
    ),
    v32.GapArm(
        "gap_refresh_lr125",
        "Refresh blocks at 1.25x LR.",
        refresh_lr_scale=1.25,
    ),
    v32.GapArm(
        "gap_refresh_lr135",
        "Refresh blocks at 1.35x LR.",
        refresh_lr_scale=1.35,
    ),
    v32.GapArm(
        "gap_refresh_lr140",
        "Refresh blocks at 1.40x LR.",
        refresh_lr_scale=1.40,
    ),
    v32.GapArm(
        "gap_refresh135_mamba110",
        "Refresh blocks at 1.35x LR plus localized Mamba blocks at 1.10x LR.",
        refresh_lr_scale=1.35,
        mamba_lr_scale=1.10,
    ),
)


def log(value: object = "") -> None:
    print(str(value), flush=True)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    os.replace(temp, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def configure_arm_tables(arms: Sequence[v32.GapArm]) -> None:
    table = tuple(arms)
    v32.GAP_ARMS = table
    v32.ARM_BY_NAME = {arm.name: arm for arm in table}
    v32.SPEC_BY_NAME = {arm.name: arm.candidate() for arm in table}


configure_arm_tables(LR_ARMS)


def parse_args() -> argparse.Namespace:
    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument("--screen-token-budget-v34", type=int, default=25_165_824)
    custom.add_argument("--promotion-token-budget", type=int, default=49_152_000)
    custom.add_argument(
        "--promotion-eval-fractions",
        nargs="+",
        type=float,
        default=[25_165_824 / 49_152_000, 1.0],
    )
    custom.add_argument(
        "--target-v30-root",
        default="/home/ubuntu/pcaf_runs/field_fusion_finalists_49m_v30_run",
    )
    custom.add_argument("--run-promotion", action=argparse.BooleanOptionalAction, default=True)
    custom.add_argument("--screen-min-gain", type=float, default=0.010)
    custom.add_argument("--screen-min-speed-ratio", type=float, default=0.95)
    custom.add_argument("--screen-max-memory-ratio", type=float, default=1.05)
    custom.add_argument("--screen-max-context-drift", type=float, default=0.100)
    custom.add_argument("--promotion-min-gain", type=float, default=0.010)
    custom.add_argument("--promotion-min-speed-ratio", type=float, default=0.95)
    custom.add_argument("--promotion-max-memory-ratio", type=float, default=1.05)
    custom.add_argument("--promotion-max-context-drift", type=float, default=0.100)
    custom.add_argument("--context-windows-v34", type=int, default=4)
    custom_args, remaining = custom.parse_known_args()

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining]
        args = v33.parse_args()
    finally:
        sys.argv = old_argv

    for key, value in vars(custom_args).items():
        setattr(args, key, value)

    args.ablation_token_budget = int(args.screen_token_budget_v34)
    args.screen_token_budget = int(args.screen_token_budget_v34)
    args.quality_token_budget = int(args.screen_token_budget_v34)
    args.eval_fractions = [0.50, 1.0]
    args.ablation_eval_fractions = [0.50, 1.0]
    args.ablation_long_contexts = [2048, 16384, 65536]
    args.ablation_long_windows = int(args.context_windows_v34)
    args.long_contexts = [2048, 16384, 65536]
    args.long_context_score_tokens = 128
    args.long_context_windows = int(args.context_windows_v34)
    args.export_winner_bf16 = True
    return args


def read_v30_baseline(root: Path) -> Dict[str, object]:
    path = root / "results.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    candidates = raw.get("results") or raw.get("candidate_results") or {}
    row = candidates.get(V30_BASELINE_NAME)
    if row is None:
        raise KeyError(f"{V30_BASELINE_NAME!r} not found in {path}")
    final_val = row.get("final_validation", {})
    final_test = row.get("final_test", {})
    result = {
        "path": str(path),
        "sha256": sha256(path),
        "candidate": V30_BASELINE_NAME,
        "validation_nll": float(final_val["nll"]),
        "test_nll": float(final_test["nll"]),
        "tokens_per_second": float(row["tokens_per_second"]),
        "peak_gib": float(row["peak_gib"]),
        "params": int(row["params"]),
    }
    for key in ("validation_nll", "test_nll", "tokens_per_second", "peak_gib"):
        if not math.isfinite(float(result[key])):
            raise RuntimeError(f"non-finite v30 baseline field {key}: {result[key]}")
    return result


def install_v34_signature(args: argparse.Namespace) -> None:
    # v33 installs exact starts + runtime geometry into the signature first.
    v33.install_v33_hooks(args)
    prior = v32.v29.checkpoint_signature

    def signature(args_, spec, shape, total_sequences):
        row = dict(prior(args_, spec, shape, total_sequences))
        arm = v32.ARM_BY_NAME.get(spec.name)
        row["v34_version"] = VERSION
        row["v34_lr_arm"] = None if arm is None else asdict(arm)
        row["v34_budget"] = int(args_.screen_token_budget)
        return row

    v32.v29.checkpoint_signature = signature
    v29.checkpoint_signature = signature
    v32.VERSION = VERSION
    v32.v29.VERSION = VERSION


def prepare_training(args: argparse.Namespace, root: Path, device: torch.device):
    specs = tuple(v32.SPEC_BY_NAME[arm.name] for arm in LR_ARMS)
    v29.CANDIDATES = specs
    v29.configure(args)
    canonical_path, canonical_sha, deps = v27.load_dependencies(args)
    if canonical_sha != v32.EXPECTED_CANONICAL_SHA256:
        raise RuntimeError(f"canonical SHA mismatch: {canonical_sha}")

    base_shape, shapes, accounting = v29.solve_candidate_shapes(args, deps)
    atomic_json(root / "component_accounting.json", accounting)
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


def exact_starts(args: argparse.Namespace, train: torch.Tensor, budget: int, path: Path) -> np.ndarray:
    if budget % int(args.train_seq):
        raise ValueError(f"token budget {budget} must divide train_seq={args.train_seq}")
    count = int(budget) // int(args.train_seq)
    return v33.make_starts_from_v28(
        count,
        len(train) - int(args.train_seq) - 1,
        int(args.data_seed),
        path,
    )


def context_eval_all(
    args: argparse.Namespace,
    arms: Sequence[v32.GapArm],
    specs: Mapping[str, object],
    shapes: Mapping[str, object],
    results: Mapping[str, v29.ScreenResult],
    deps,
    test: torch.Tensor,
    device: torch.device,
    output: Path,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    contexts: Dict[str, Dict[str, Dict[str, float]]] = {}
    args.long_contexts = [2048, 16384, 65536]
    args.long_context_score_tokens = 128
    args.long_context_windows = int(args.context_windows_v34)
    for arm in arms:
        result = results[arm.name]
        model = v29.load_model_from_result(
            specs[arm.name], shapes[arm.name], result, args, deps, device
        )
        contexts[arm.name] = v29.long_context_eval(model, test, args, device)
        del model
        v32.clear_cuda()
        atomic_json(output, contexts)
    return contexts


def drift_2k_64k(context: Mapping[str, Mapping[str, float]]) -> Optional[float]:
    c2 = float(context.get("2048", {}).get("nll", float("nan")))
    c64 = float(context.get("65536", {}).get("nll", float("nan")))
    if not math.isfinite(c2) or not math.isfinite(c64):
        return None
    return c64 - c2


def run_screen(
    args: argparse.Namespace,
    root: Path,
    deps,
    shapes,
    train: torch.Tensor,
    val_c,
    val: torch.Tensor,
    test_c,
    test: torch.Tensor,
    device: torch.device,
) -> Dict[str, object]:
    arms = LR_ARMS
    specs = v32.SPEC_BY_NAME
    args.screen_token_budget = int(args.screen_token_budget_v34)
    args.quality_token_budget = int(args.screen_token_budget_v34)
    args.eval_fractions = [0.50, 1.0]
    starts = exact_starts(args, train, args.screen_token_budget, root / "screen_paired_starts.npy")

    results: Dict[str, v29.ScreenResult] = {}
    original_stable = float(args.wsd_stable_fraction)
    args.wsd_stable_fraction = 0.70
    for arm in arms:
        log("=" * 220)
        log(
            f"LR SCREEN ARM: {arm.name} refreshLR={arm.refresh_lr_scale:.2f} "
            f"mambaLR={arm.mamba_lr_scale:.2f}"
        )
        results[arm.name] = v29.train_candidate(
            specs[arm.name],
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

    contexts = context_eval_all(
        args,
        arms,
        specs,
        shapes,
        results,
        deps,
        test,
        device,
        root / "screen_long_contexts.json",
    )

    baseline = results[BASELINE_NAME]
    rows = []
    for arm in arms:
        result = results[arm.name]
        drift = drift_2k_64k(contexts[arm.name])
        gain_val = float(baseline.final_validation["nll"]) - float(result.final_validation["nll"])
        gain_test = float(baseline.final_test["nll"]) - float(result.final_test["nll"])
        speed_ratio = float(result.tokens_per_second) / max(float(baseline.tokens_per_second), 1e-9)
        memory_ratio = float(result.peak_gib) / max(float(baseline.peak_gib), 1e-9)
        eligible = (
            arm.name != BASELINE_NAME
            and gain_val >= float(args.screen_min_gain)
            and gain_test >= 0.0
            and speed_ratio >= float(args.screen_min_speed_ratio)
            and memory_ratio <= float(args.screen_max_memory_ratio)
            and drift is not None
            and drift <= float(args.screen_max_context_drift)
        )
        rows.append(
            {
                "candidate": arm.name,
                "validation_nll": float(result.final_validation["nll"]),
                "test_nll": float(result.final_test["nll"]),
                "tokens_per_second": float(result.tokens_per_second),
                "peak_gib": float(result.peak_gib),
                "validation_gain_vs_baseline": gain_val,
                "test_gain_vs_baseline": gain_test,
                "speed_ratio_vs_baseline": speed_ratio,
                "memory_ratio_vs_baseline": memory_ratio,
                "context_2k_to_64k": drift,
                "eligible": eligible,
                "arm": asdict(arm),
            }
        )
    rows.sort(key=lambda row: (row["validation_nll"], row["test_nll"]))
    eligible_rows = [row for row in rows if row["eligible"]]
    winner = min(
        eligible_rows,
        key=lambda row: (row["validation_nll"], row["test_nll"]),
    ) if eligible_rows else None
    decision = {
        "action": "PROMOTE_LR_ARM_TO_49M" if winner else "NO_STRICT_LR_ARM",
        "winner": None if winner is None else winner["candidate"],
        "baseline": BASELINE_NAME,
        "rows": rows,
        "contexts": contexts,
        "screen_token_budget": int(args.screen_token_budget),
    }
    atomic_json(root / "screen_decision.json", decision)
    return decision


def promotion_spec_and_arm(winner_name: str):
    source_arm = v32.ARM_BY_NAME[winner_name]
    promotion_name = winner_name + "_49m"
    arm = v32.GapArm(
        promotion_name,
        source_arm.description + " Confirmed from scratch at 49.152M tokens.",
        stable_fraction=source_arm.stable_fraction,
        mamba_lr_scale=source_arm.mamba_lr_scale,
        refresh_lr_scale=source_arm.refresh_lr_scale,
        mamba_positions=source_arm.mamba_positions,
    )
    spec = arm.candidate()
    v32.ARM_BY_NAME[promotion_name] = arm
    v32.SPEC_BY_NAME[promotion_name] = spec
    return arm, spec


def run_promotion(
    args: argparse.Namespace,
    root: Path,
    deps,
    train: torch.Tensor,
    val_c,
    val: torch.Tensor,
    test_c,
    test: torch.Tensor,
    screen_decision: Mapping[str, object],
    v30_baseline: Mapping[str, object],
    device: torch.device,
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

    arm, spec = promotion_spec_and_arm(str(winner_name))
    v29.CANDIDATES = (spec,)
    original_budget = int(args.screen_token_budget)
    original_quality = int(args.quality_token_budget)
    original_fractions = list(args.eval_fractions)
    original_stable = float(args.wsd_stable_fraction)

    args.screen_token_budget = int(args.promotion_token_budget)
    args.quality_token_budget = int(args.promotion_token_budget)
    args.eval_fractions = list(map(float, args.promotion_eval_fractions))
    args.wsd_stable_fraction = 0.70

    _, shapes, accounting = v29.solve_candidate_shapes(args, deps)
    atomic_json(root / "promotion_component_accounting.json", accounting)
    v29.architecture_audit((spec,), shapes, args, deps, device, root / "promotion_architecture")
    v29.causality_and_backward_preflight((spec,), shapes, args, deps, device, root / "promotion_preflight")

    starts = exact_starts(
        args,
        train,
        int(args.promotion_token_budget),
        root / "promotion_paired_starts.npy",
    )
    log("=" * 220)
    log(
        f"49M CONFIRMATION: {spec.name} refreshLR={arm.refresh_lr_scale:.2f} "
        f"mambaLR={arm.mamba_lr_scale:.2f}"
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

    model = v29.load_model_from_result(spec, shapes[spec.name], result, args, deps, device)
    args.long_contexts = [2048, 16384, 65536]
    args.long_context_score_tokens = 128
    args.long_context_windows = int(args.context_windows_v34)
    contexts = v29.long_context_eval(model, test, args, device)
    del model
    v32.clear_cuda()
    atomic_json(root / "promotion_long_contexts.json", contexts)

    drift = drift_2k_64k(contexts)
    gain_val = float(v30_baseline["validation_nll"]) - float(result.final_validation["nll"])
    gain_test = float(v30_baseline["test_nll"]) - float(result.final_test["nll"])
    speed_ratio = float(result.tokens_per_second) / max(float(v30_baseline["tokens_per_second"]), 1e-9)
    memory_ratio = float(result.peak_gib) / max(float(v30_baseline["peak_gib"]), 1e-9)
    eligible = (
        gain_val >= float(args.promotion_min_gain)
        and gain_test >= 0.0
        and speed_ratio >= float(args.promotion_min_speed_ratio)
        and memory_ratio <= float(args.promotion_max_memory_ratio)
        and drift is not None
        and drift <= float(args.promotion_max_context_drift)
    )
    decision = {
        "action": "PROMOTE_LR_RECIPE_TO_98M" if eligible else "STOP_AFTER_49M",
        "winner": spec.name,
        "source_screen_winner": winner_name,
        "eligible": eligible,
        "validation_nll": float(result.final_validation["nll"]),
        "test_nll": float(result.final_test["nll"]),
        "tokens_per_second": float(result.tokens_per_second),
        "peak_gib": float(result.peak_gib),
        "validation_gain_vs_v30": gain_val,
        "test_gain_vs_v30": gain_test,
        "speed_ratio_vs_v30": speed_ratio,
        "memory_ratio_vs_v30": memory_ratio,
        "context_2k_to_64k": drift,
        "contexts": contexts,
        "v30_baseline": dict(v30_baseline),
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
    v30_baseline: Mapping[str, object],
    screen: Mapping[str, object],
    promotion: Mapping[str, object],
) -> str:
    width = 220
    lines = [
        "=" * width,
        "FIELD-FUSION v34 — SPECIALIZED-LR CALIBRATION + 49M CONFIRMATION",
        "=" * width,
        "No Transformer or pure Mamba model was trained or benchmarked again.",
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
        "25.165824M SPECIALIZED-LR SCREEN",
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
        "FROZEN v30 49M BASELINE",
        f"val={float(v30_baseline['validation_nll']):.5f} test={float(v30_baseline['test_nll']):.5f} "
        f"tok/s={float(v30_baseline['tokens_per_second']):,.0f} peak={float(v30_baseline['peak_gib']):.2f}G",
        "",
        "49M CONFIRMATION",
    ]
    if promotion.get("action") == "NOT_RUN":
        lines.append(f"not run: {promotion.get('reason')}")
    else:
        promotion_drift = promotion.get("context_2k_to_64k")
        promotion_drift_text = (
            "n/a" if promotion_drift is None else f"{float(promotion_drift):+.5f}"
        )
        lines += [
            f"candidate={promotion.get('winner')}",
            f"val={float(promotion.get('validation_nll', float('nan'))):.5f} "
            f"test={float(promotion.get('test_nll', float('nan'))):.5f} "
            f"dVal_vs_v30={float(promotion.get('validation_gain_vs_v30', float('nan'))):+.5f} "
            f"dTest_vs_v30={float(promotion.get('test_gain_vs_v30', float('nan'))):+.5f}",
            f"tok/s={float(promotion.get('tokens_per_second', float('nan'))):,.0f} "
            f"speed={float(promotion.get('speed_ratio_vs_v30', float('nan'))):.3f}x "
            f"peak={float(promotion.get('peak_gib', float('nan'))):.2f}G "
            f"drift2K→64K={promotion_drift_text}",
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
    configure_arm_tables(LR_ARMS)
    v32.validate_paths(args)
    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)

    reused_inference, reused_kernel = v33.validate_reused_phases(args, root)
    v30_baseline = read_v30_baseline(Path(args.target_v30_root))
    install_v34_signature(args)
    starts_audit = v33.paired_starts_selftest(args, root)
    frozen_starts = np.load(Path(args.target_v28_starts), allow_pickle=False)
    promotion_start_count = int(args.promotion_token_budget) // int(args.train_seq)
    if frozen_starts.ndim != 1 or len(frozen_starts) < promotion_start_count:
        raise RuntimeError(
            f"v28 starts cannot support promotion: shape={frozen_starts.shape} "
            f"need={promotion_start_count}"
        )
    atomic_json(
        root / "v34_preflight.json",
        {
            "version": VERSION,
            "arms": [asdict(arm) for arm in LR_ARMS],
            "v30_baseline": v30_baseline,
            "starts": starts_audit,
            "promotion_start_count": promotion_start_count,
            "promotion_prefix_sha256": hashlib.sha256(
                np.asarray(frozen_starts[:promotion_start_count], dtype=np.int64).tobytes()
            ).hexdigest(),
            "runtime": {
                "field_chunk": int(args.field_chunk),
                "triton_block_c": int(args.triton_block_c),
                "triton_chunk_t": int(args.triton_chunk_t),
            },
        },
    )
    atomic_json(root / "args.json", vars(args))

    log("=" * 220)
    log("FIELD-FUSION v34 — PRE-RUN AUDIT")
    log(f"arms={[arm.name for arm in LR_ARMS]}")
    log(
        f"v30_baseline val={v30_baseline['validation_nll']:.5f} "
        f"test={v30_baseline['test_nll']:.5f}"
    )
    log(
        f"paired_starts count={starts_audit['count']} "
        f"sha={starts_audit['prefix_bytes_sha256']} exact=True"
    )
    log(
        f"runtime field_chunk={args.field_chunk} BLOCK_C={args.triton_block_c} "
        f"CHUNK_T={args.triton_chunk_t}"
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
    ) = prepare_training(args, root, device)

    log("=" * 220)
    log("PHASE A — 25.165824M SPECIALIZED-LR CALIBRATION")
    screen = run_screen(
        args, root, deps, shapes, train, val_c, val, test_c, test, device
    )

    log("=" * 220)
    log("PHASE B — AUTOMATIC 49.152M CONFIRMATION")
    promotion = run_promotion(
        args,
        root,
        deps,
        train,
        val_c,
        val,
        test_c,
        test,
        screen,
        v30_baseline,
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
        "v30_baseline": v30_baseline,
        "screen": screen,
        "promotion": promotion,
        "target_v28_starts_sha256": sha256(Path(args.target_v28_starts)),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    atomic_json(root / "results.json", payload)
    summary = make_summary(
        reused_inference, reused_kernel, v30_baseline, screen, promotion
    )
    atomic_text(root / "summary.txt", summary)
    log(summary)


if __name__ == "__main__":
    main()
