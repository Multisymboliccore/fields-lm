#!/usr/bin/env python3
"""FIELD-FUSION v30 — 49.152M finalist confirmation without rival retraining.

Three evidence-driven Field finalists are trained from scratch on the exact
prefix of paired WikiText-103 windows used by v28/v29:

  1) pure Field with the weak 2048 refresh replaced by a second 1024 refresh;
  2) four localized official Mamba-2 blocks plus the duplicated 1024 refresh;
  3) the same hybrid plus two causal vector Delta editors.

Transformer and pure Mamba-2 are NOT retrained. Their exact 49.152M-token v28
validation/test results are loaded as a frozen scoreboard. A finalist is
eligible for a 98.304M no-rival continuation only if it beats both frozen rivals
and passes speed, memory, parameter and long-context guards.

PCAF is kept active during training. At the end, every finalist is evaluated
with the same checkpoint both with and without PCAF, making its current marginal
contribution explicit rather than guessing from older models.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch

import field_fusion_delta_quality_ablation_v29 as v29
import field_fusion_final_ablation_v26 as v26
import field_fusion_recipe_memory_v27 as v27
import field_fusion_scaling_confirmation_v28 as v28
import field_fusion_wiki100_canonical_v23 as v23
import field_fusion_wiki100_mamba2_v25 as v25

VERSION = 30
FUSION = v29.FUSION
TRANSFORMER = v29.TRANSFORMER
MAMBA2 = v29.MAMBA2
EXPECTED_CANONICAL_SHA256 = v29.EXPECTED_CANONICAL_SHA256

# The four Mamba blocks replace F4/F9/F14/F19 — the last Field block before
# each attention refresh. The four refresh blocks remain present.
MAMBA4_INDICES = v29.MAMBA4_INDICES
CandidateSpec = v29.CandidateSpec
CANDIDATES = (
    CandidateSpec(
        "field_refresh_1024x2_49m",
        "Pure Field control: refresh windows 256/512/1024/1024.",
        refresh_1024x2=True,
    ),
    CandidateSpec(
        "field_mamba4_refresh1024x2_49m",
        "Pareto finalist: four localized Mamba-2 blocks plus duplicated 1024 refresh.",
        refresh_1024x2=True,
        mamba_replace=MAMBA4_INDICES,
    ),
    CandidateSpec(
        "field_mamba4_delta_vector_refresh1024x2_49m",
        "Quality finalist: localized Mamba-2, vector Delta editors, duplicated 1024 refresh.",
        refresh_1024x2=True,
        delta_refresh_ids=(1, 2),
        delta_mode="vector",
        mamba_replace=MAMBA4_INDICES,
    ),
)


def log(x: object = "") -> None:
    print(str(x), flush=True)


def atomic_json(path: Path, obj: object) -> None:
    v29.atomic_json(path, obj)


def atomic_text(path: Path, text: str) -> None:
    v29.atomic_text(path, text)


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument("--screen-token-budget", type=int, default=49_152_000)
    custom.add_argument("--eval-fractions", nargs="+", type=float, default=[0.512, 1.0])
    custom.add_argument(
        "--target-v29-results",
        default="/home/ubuntu/pcaf_runs/field_fusion_delta_quality_ablation_v29_run/results.json",
    )
    custom.add_argument("--max-context-drift", type=float, default=0.10)
    custom.add_argument("--pcaf-diagnostic-token-budget", type=int, default=293_944)
    custom.add_argument("--export-promoted-bf16", action=argparse.BooleanOptionalAction, default=True)
    custom_args, remaining = custom.parse_known_args()

    old = sys.argv
    try:
        sys.argv = [old[0], *remaining]
        args = v29.parse_args()
    finally:
        sys.argv = old
    for key, value in vars(custom_args).items():
        setattr(args, key, value)
    args.quality_token_budget = int(args.screen_token_budget)
    args.export_winner_bf16 = False  # v30 exports up to two promoted finalists itself.
    return args


def configure(args) -> None:
    v29.CANDIDATES = CANDIDATES
    v29.VERSION = VERSION
    v26.VERSION = VERSION
    v29.configure(args)


def selected_candidates(args) -> Sequence[CandidateSpec]:
    if not args.candidate:
        return CANDIDATES
    wanted = set(args.candidate)
    rows = tuple(x for x in CANDIDATES if x.name in wanted)
    missing = wanted - {x.name for x in rows}
    if missing:
        raise ValueError(f"unknown candidate(s): {sorted(missing)}")
    return rows


def load_v29_provenance(args, root: Path) -> Dict[str, object]:
    path = Path(args.target_v29_results)
    if not path.is_file():
        row = {"found": False, "path": str(path)}
        atomic_json(root / "v29_provenance.json", row)
        return row
    raw = json.loads(path.read_text(encoding="utf-8"))
    result_rows = raw.get("results", {})
    compact = {
        name: {
            "validation_nll": float(row["final_validation"]["nll"]),
            "test_nll": float(row["final_test"]["nll"]),
            "tokens_per_second": float(row["tokens_per_second"]),
            "peak_gib": float(row["peak_gib"]),
        }
        for name, row in result_rows.items()
    }
    out = {
        "found": True,
        "path": str(path),
        "sha256": v29.sha256(path),
        "screen_tokens": int(raw.get("args", {}).get("screen_token_budget", 0)),
        "candidates": compact,
    }
    atomic_json(root / "v29_provenance.json", out)
    return out


def pcaf_diagnostic(model: torch.nn.Module, val_c, val: torch.Tensor,
                    test_c, test: torch.Tensor, args,
                    device: torch.device) -> Dict[str, object]:
    if not hasattr(model, "cache") or not hasattr(model.cache, "enabled"):
        return {"status": "missing_cache"}

    model.eval()
    old_enabled = bool(model.cache.enabled)
    try:
        model.cache.enabled = True
        val_on = v29.evaluate_streaming(
            model, val_c, val, args, device, args.pcaf_diagnostic_token_budget
        )
        test_on = v29.evaluate_streaming(
            model, test_c, test, args, device, args.pcaf_diagnostic_token_budget
        )
        model.cache.enabled = False
        val_off = v29.evaluate_streaming(
            model, val_c, val, args, device, args.pcaf_diagnostic_token_budget
        )
        test_off = v29.evaluate_streaming(
            model, test_c, test, args, device, args.pcaf_diagnostic_token_budget
        )
        model.cache.enabled = True

        max_start = max(1, len(val) - args.train_seq - 1)
        start = int((args.eval_seed * 7919 + VERSION * 101) % max_start)
        window = val[start:start + args.train_seq + 1].long().to(device)
        x, y = window[:-1][None], window[1:][None]
        with torch.no_grad(), v23.amp_ctx(device, args.amp):
            states, logits = model.states_logits(x)
            stats = v26.cache_diagnostics(model.cache, states, logits, x, y)
    finally:
        model.cache.enabled = old_enabled
    dval = float(val_off["nll"] - val_on["nll"])
    dtest = float(test_off["nll"] - test_on["nll"])
    mean_gain = 0.5 * (dval + dtest)
    if mean_gain >= 0.010:
        recommendation = "KEEP_CANONICAL"
    elif mean_gain >= 0.003:
        recommendation = "KEEP_OPTIONAL_PENDING_98M"
    elif mean_gain > -0.002:
        recommendation = "REMOVAL_CANDIDATE_PENDING_SPEED_ABLATION"
    else:
        recommendation = "DISABLE_CANDIDATE_PCAF_HURTS"
    return {
        "status": "ok",
        "validation_on": val_on,
        "validation_off": val_off,
        "validation_off_minus_on_nll": dval,
        "test_on": test_on,
        "test_off": test_off,
        "test_off_minus_on_nll": dtest,
        "mean_nll_gain": mean_gain,
        "recommendation": recommendation,
        "cache_stats": stats,
    }


def make_decision(results: Mapping[str, v29.ScreenResult],
                  targets: Mapping[str, object],
                  contexts: Mapping[str, Mapping[str, Dict[str, float]]],
                  pcaf: Mapping[str, Mapping[str, object]], args) -> Dict[str, object]:
    field_speed = float(targets["field_speed_reference"])
    field_memory = float(targets["field_memory_reference"])
    ranked = sorted(results.values(), key=lambda r: (r.final_validation["nll"], r.final_test["nll"]))
    rows: List[Dict[str, object]] = []
    for result in ranked:
        ctx = contexts.get(result.candidate, {})
        c2 = float(ctx.get("2048", {}).get("nll", float("nan")))
        c16 = float(ctx.get("16384", {}).get("nll", float("nan")))
        finite_ctx = math.isfinite(c2) and math.isfinite(c16)
        drift = c16 - c2 if finite_ctx else float("inf")
        checks = {
            "beats_transformer_validation": result.final_validation["nll"] < targets["screen"][TRANSFORMER]["validation_nll"],
            "beats_mamba_validation": result.final_validation["nll"] < targets["screen"][MAMBA2]["validation_nll"],
            "beats_transformer_test": result.final_test["nll"] < targets["screen"][TRANSFORMER]["test_nll"],
            "beats_mamba_test": result.final_test["nll"] < targets["screen"][MAMBA2]["test_nll"],
            "speed_guard": result.tokens_per_second >= field_speed * args.min_speed_ratio,
            "memory_guard": result.peak_gib <= field_memory * args.max_memory_ratio,
            "parameter_guard": abs(result.param_delta_pct) <= args.param_tolerance_pct,
            "context_finite": finite_ctx,
            "context_guard": finite_ctx and drift <= args.max_context_drift,
        }
        p = pcaf.get(result.candidate, {})
        rows.append({
            "candidate": result.candidate,
            "validation_nll": result.final_validation["nll"],
            "test_nll": result.final_test["nll"],
            "validation_minus_transformer": result.final_validation["nll"] - targets["screen"][TRANSFORMER]["validation_nll"],
            "validation_minus_mamba": result.final_validation["nll"] - targets["screen"][MAMBA2]["validation_nll"],
            "test_minus_transformer": result.final_test["nll"] - targets["screen"][TRANSFORMER]["test_nll"],
            "test_minus_mamba": result.final_test["nll"] - targets["screen"][MAMBA2]["test_nll"],
            "tokens_per_second": result.tokens_per_second,
            "peak_gib": result.peak_gib,
            "context_drift_2k_to_16k": drift,
            "pcaf_validation_off_minus_on_nll": p.get("validation_off_minus_on_nll"),
            "pcaf_test_off_minus_on_nll": p.get("test_off_minus_on_nll"),
            "checks": checks,
            "eligible": bool(all(checks.values())),
        })

    eligible = [row for row in rows if row["eligible"]]
    promoted: List[str] = []
    if eligible:
        quality = min(eligible, key=lambda x: (x["validation_nll"], x["test_nll"]))["candidate"]
        speed = max(eligible, key=lambda x: x["tokens_per_second"])["candidate"]
        promoted.append(str(quality))
        if speed != quality:
            promoted.append(str(speed))
        action = "PROMOTE_FINALISTS_TO_98M_NO_RIVALS"
        reason = (
            "At least one finalist beat both frozen 49.152M Transformer/Mamba targets "
            "and passed systems plus long-context guards. Continue only the quality and "
            "speed/Pareto winners to 98.304M; do not retrain rivals yet."
        )
    else:
        action = "REFINE_FINALIST_MECHANISM_AT_49M_NO_RIVALS"
        reason = "No finalist cleared every frozen 49.152M target and guard."

    return {
        "action": action,
        "reason": reason,
        "promoted_candidates": promoted,
        "eligible_candidates": [str(row["candidate"]) for row in eligible],
        "ranked": rows,
        "frozen_targets_49m": targets["screen"],
        "frozen_targets_98m": targets["final_98m"],
    }


def export_bf16(path: Path, spec: CandidateSpec, shape: v23.Shape,
                result: v29.ScreenResult, args, deps,
                device: torch.device) -> None:
    model = v29.load_model_from_result(spec, shape, result, args, deps, device)
    state = {}
    for key, value in model.state_dict().items():
        cpu = value.detach().cpu()
        state[key] = cpu.to(torch.bfloat16) if cpu.is_floating_point() else cpu
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "format": "field_fusion_v30_49m_bf16_finalist",
        "version": VERSION,
        "candidate": asdict(spec),
        "shape": asdict(shape),
        "train_tokens": int(result.train_tokens),
        "state_dict": state,
        "args": vars(args),
    }, tmp)
    os.replace(tmp, path)
    del model
    clear_cuda()


def summary(args, canonical_path, canonical_sha, targets, provenance,
            results, contexts, pcaf, decision, prefix) -> str:
    width = 236
    lines = [
        "=" * width,
        "FIELD-FUSION v30 — 49.152M FINALIST CONFIRMATION / FROZEN RIVALS",
        "=" * width,
        f"canonical={canonical_path} sha256={canonical_sha}",
        f"paired tokens/candidate={args.screen_token_budget:,} context={args.train_seq} batch={args.batch_size} WSD gateLR=2x",
        f"v28_scoreboard={targets['path']} sha256={targets['sha256']} prefix_equal={prefix['prefix_equal']}",
        f"v29_provenance_found={provenance.get('found')} sha256={provenance.get('sha256', 'n/a')}",
        "No Transformer or pure Mamba-2 model was retrained.",
        "Topology: 16 Field blocks + 4 localized Mamba-2 blocks + 4 attention refreshes for hybrid finalists; PCAF remains at readout.",
        "",
        "FROZEN v28 TARGETS AT MATCHED 49.152M TOKENS",
        f"Transformer val={targets['screen'][TRANSFORMER]['validation_nll']:.5f} test={targets['screen'][TRANSFORMER]['test_nll']:.5f}",
        f"Mamba-2    val={targets['screen'][MAMBA2]['validation_nll']:.5f} test={targets['screen'][MAMBA2]['test_nll']:.5f}",
        f"Field-v28  val={targets['screen'][FUSION]['validation_nll']:.5f} test={targets['screen'][FUSION]['test_nll']:.5f}",
        "",
        "FINALIST RESULTS",
        f"{'candidate':52s} {'params':>12s} {'d%':>7s} {'val NLL':>10s} {'test NLL':>10s} {'tok/s':>10s} {'peakGB':>8s} {'2K→16K':>10s} {'PCAF dVal':>10s}",
    ]
    for row in decision["ranked"]:
        r = results[row["candidate"]]
        pdelta = row.get("pcaf_validation_off_minus_on_nll")
        ptxt = f"{float(pdelta):+10.5f}" if pdelta is not None else f"{'n/a':>10s}"
        lines.append(
            f"{r.candidate:52s} {r.params:12,d} {r.param_delta_pct:+7.3f} "
            f"{r.final_validation['nll']:10.5f} {r.final_test['nll']:10.5f} "
            f"{r.tokens_per_second:10,.0f} {r.peak_gib:8.2f} "
            f"{row['context_drift_2k_to_16k']:+10.5f} {ptxt}"
        )
    lines += ["", "PCAF MARGINAL CONTRIBUTION (same checkpoint; positive means PCAF helps)"]
    for row in decision["ranked"]:
        p = pcaf.get(row["candidate"], {})
        s = p.get("cache_stats", {})
        lines.append(
            f"{row['candidate']:52s} dVal={p.get('validation_off_minus_on_nll', float('nan')):+.5f} "
            f"dTest={p.get('test_off_minus_on_nll', float('nan')):+.5f} "
            f"coverage={s.get('candidate_coverage', float('nan')):.3f} "
            f"gate={s.get('gate_mean', float('nan')):.3f} win={s.get('cache_win_rate', float('nan')):.3f} "
            f"recommendation={p.get('recommendation', 'n/a')}"
        )
    lines += ["", "PROMOTION CHECKS"]
    for row in decision["ranked"]:
        c = row["checks"]
        lines.append(
            f"{row['candidate']:52s} eligible={row['eligible']} "
            f"Tval={c['beats_transformer_validation']} Mval={c['beats_mamba_validation']} "
            f"Ttest={c['beats_transformer_test']} Mtest={c['beats_mamba_test']} "
            f"speed={c['speed_guard']} memory={c['memory_guard']} params={c['parameter_guard']} "
            f"finite={c['context_finite']} ctx={c['context_guard']}"
        )
    lines += [
        "",
        "AUTOMATIC NEXT STEP",
        f"action={decision['action']}",
        f"promoted={','.join(decision['promoted_candidates']) if decision['promoted_candidates'] else 'none'}",
        f"eligible={','.join(decision['eligible_candidates']) if decision['eligible_candidates'] else 'none'}",
        f"reason={decision['reason']}",
        "No rival or longer run is launched automatically.",
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
    # v29 helpers consult its module-global candidate set.
    v29.CANDIDATES = tuple(specs)
    base_shape, shapes, accounting = v29.solve_candidate_shapes(args, deps)
    atomic_json(root / "component_accounting.json", accounting)
    targets = v29.load_frozen_targets(args, root)
    provenance = load_v29_provenance(args, root)

    architecture = v29.architecture_audit(specs, shapes, args, deps, device, root)
    preflight = (
        v29.causality_and_backward_preflight(specs, shapes, args, deps, device, root)
        if args.run_preflight else {}
    )

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
    prefix = v29.audit_starts(starts, args, root)

    results: Dict[str, v29.ScreenResult] = {}
    for spec in specs:
        log("=" * 220)
        log(f"49M FINALIST: {spec.name} — {spec.description}")
        results[spec.name] = v29.train_candidate(
            spec, shapes[spec.name], args, deps,
            train, val_c, val, test_c, test, starts, root, device,
        )
        atomic_json(root / "candidate_results.json", {k: asdict(v) for k, v in results.items()})

    contexts: Dict[str, Dict[str, Dict[str, float]]] = {}
    pcaf: Dict[str, Dict[str, object]] = {}
    for spec in specs:
        model = v29.load_model_from_result(spec, shapes[spec.name], results[spec.name], args, deps, device)
        contexts[spec.name] = v29.long_context_eval(model, test, args, device)
        pcaf[spec.name] = pcaf_diagnostic(model, val_c, val, test_c, test, args, device)
        atomic_json(root / "long_context_results.json", contexts)
        atomic_json(root / "pcaf_diagnostics.json", pcaf)
        del model
        clear_cuda()

    decision = make_decision(results, targets, contexts, pcaf, args)
    atomic_json(root / "decision.json", decision)

    if args.export_promoted_bf16 and decision["promoted_candidates"]:
        export_dir = root / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        for name in decision["promoted_candidates"]:
            spec = next(x for x in specs if x.name == name)
            export_bf16(
                export_dir / f"{name}_step{results[name].updates}_BF16.pt",
                spec, shapes[name], results[name], args, deps, device,
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
        "v29_provenance": provenance,
        "architecture_audit": architecture,
        "candidate_preflight": preflight,
        "paired_prefix_audit": prefix,
        "results": {k: asdict(v) for k, v in results.items()},
        "long_contexts": contexts,
        "pcaf_diagnostics": pcaf,
        "decision": decision,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "mamba_ssm": v25.MAMBA_VERSION,
    }
    atomic_json(root / "results.json", payload)
    text = summary(
        args, canonical_path, canonical_sha, targets, provenance,
        results, contexts, pcaf, decision, prefix,
    )
    atomic_text(root / "summary.txt", text)
    log(text)


if __name__ == "__main__":
    main()
