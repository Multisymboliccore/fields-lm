#!/usr/bin/env python3
"""FIELD-FUSION v31 — PCAF retraining ablation, 98M confirmation and 64K inference.

Execution plan
--------------
A) Train exactly one 49.152M-token arm from scratch with PCAF disabled:
   Field + four localized Mamba-2 blocks + refreshes 256/512/1024/1024.
   Compare it with the frozen PCAF-on v30 checkpoint/result.  No rival is
   retrained.

B) Choose PCAF on/off using paired validation/test quality and systems cost,
   then train from scratch to 98.304M tokens:
     1. the selected hybrid;
     2. the pure Field refresh-1024x2 scientific control (optional, on by default).

C) Load the frozen 98M Mamba-2 checkpoint from v28 and benchmark long-sequence
   prefill inference at 16K/32K/64K.  For each model/context the program tries
   the largest batch allowed by --infer-max-batch-tokens and backs off on OOM.
   The timed operation is backbone prefill plus the parametric next-token head.
   This avoids the full-vocabulary projection at every sequence position and is
   the clean systems comparison for long-context processing.  It is not an
   incremental autoregressive decode benchmark.
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
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

import field_fusion_delta_quality_ablation_v29 as v29
import field_fusion_final_ablation_v26 as v26
import field_fusion_finalists_49m_v30 as v30
import field_fusion_recipe_memory_v27 as v27
import field_fusion_scaling_confirmation_v28 as v28
import field_fusion_wiki100_canonical_v23 as v23
import field_fusion_wiki100_mamba2_v25 as v25

VERSION = 31
FUSION = v29.FUSION
TRANSFORMER = v29.TRANSFORMER
MAMBA2 = v29.MAMBA2
EXPECTED_CANONICAL_SHA256 = v29.EXPECTED_CANONICAL_SHA256
CandidateSpec = v29.CandidateSpec
MAMBA4_INDICES = v29.MAMBA4_INDICES

PCAF_OFF_SCREEN = CandidateSpec(
    "field_mamba4_refresh1024x2_pcaf_off_49m",
    "Retraining ablation: localized Mamba-2 + duplicated 1024 refresh, PCAF disabled from initialization.",
    refresh_1024x2=True,
    mamba_replace=MAMBA4_INDICES,
)
HYBRID_LONG = CandidateSpec(
    "field_mamba4_refresh1024x2_selected_98m",
    "98M principal finalist using the PCAF state selected by the 49M retraining ablation.",
    refresh_1024x2=True,
    mamba_replace=MAMBA4_INDICES,
)
PURE_LONG = CandidateSpec(
    "field_refresh1024x2_selected_98m",
    "98M pure-Field scientific control using the selected PCAF state.",
    refresh_1024x2=True,
)
ALL_SPECS = (PCAF_OFF_SCREEN, HYBRID_LONG, PURE_LONG)

# v29's training/loading helpers call its module-global builder.  Keep the
# architecture identical and change only the cache's enabled flag before the
# first forward.  FastSuccessorCacheV5 exits immediately after parametric CE
# when disabled, so no PCAF candidates/router/projection are executed.
_ORIGINAL_BUILD_CANDIDATE = v29.build_candidate
_ORIGINAL_CHECKPOINT_SIGNATURE = v29.checkpoint_signature
_ORIGINAL_STREAMING_NLL = v27.streaming_nll
_PCAF_ENABLED: Dict[str, bool] = {
    PCAF_OFF_SCREEN.name: False,
    HYBRID_LONG.name: True,
    PURE_LONG.name: True,
}


def build_candidate_v31(spec: CandidateSpec, shape: v23.Shape, args, deps,
                         device: torch.device) -> nn.Module:
    model = _ORIGINAL_BUILD_CANDIDATE(spec, shape, args, deps, device)
    enabled = bool(_PCAF_ENABLED.get(spec.name, True))
    if not hasattr(model, "cache") or not hasattr(model.cache, "enabled"):
        raise RuntimeError(f"candidate {spec.name} has no switchable PCAF cache")
    model.cache.enabled = enabled
    model.pcaf_enabled_v31 = enabled
    return model


def checkpoint_signature_v31(args, spec: CandidateSpec, shape: v23.Shape,
                             total_sequences: int) -> Dict[str, object]:
    row = dict(_ORIGINAL_CHECKPOINT_SIGNATURE(args, spec, shape, total_sequences))
    row["pcaf_enabled"] = bool(_PCAF_ENABLED.get(spec.name, True))
    row["v31_version"] = VERSION
    return row


def streaming_nll_v31(name: str, model: nn.Module, x: torch.Tensor,
                      y: torch.Tensor, chunk: int, return_tokens: bool = False):
    if name == FUSION and not bool(getattr(getattr(model, "cache", None), "enabled", True)):
        _, hidden = v27.hidden_for_readout(name, model, x)
        return v27.generic_chunked_ce(model, hidden, y, chunk, return_tokens=return_tokens)
    return _ORIGINAL_STREAMING_NLL(name, model, x, y, chunk, return_tokens=return_tokens)


v29.build_candidate = build_candidate_v31
v29.checkpoint_signature = checkpoint_signature_v31
v27.streaming_nll = streaming_nll_v31


def log(x: object = "") -> None:
    print(str(x), flush=True)


def atomic_json(path: Path, obj: object) -> None:
    v29.atomic_json(path, obj)


def atomic_text(path: Path, text: str) -> None:
    v29.atomic_text(path, text)


def sha256(path: Path) -> str:
    return v29.sha256(path)


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_args() -> argparse.Namespace:
    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument("--pcaf-screen-token-budget", type=int, default=49_152_000)
    custom.add_argument("--long-token-budget", type=int, default=98_304_000)
    custom.add_argument(
        "--long-eval-fractions", nargs="+", type=float,
        default=[0.256, 0.50, 0.75, 1.0],
    )
    custom.add_argument(
        "--target-v30-results",
        default="/home/ubuntu/pcaf_runs/field_fusion_finalists_49m_v30_run/results.json",
    )
    custom.add_argument(
        "--target-v28-results",
        default="/home/ubuntu/pcaf_runs/field_fusion_scaling_confirmation_v28_run/results.json",
    )
    custom.add_argument(
        "--target-v28-starts",
        default="/home/ubuntu/pcaf_runs/field_fusion_scaling_confirmation_v28_run/paired_example_starts.npy",
    )
    custom.add_argument("--pcaf-remove-max-val-loss", type=float, default=0.010)
    custom.add_argument("--pcaf-remove-max-test-loss", type=float, default=0.010)
    custom.add_argument("--pcaf-remove-min-speed-ratio", type=float, default=0.98)
    custom.add_argument("--long-checkpoint-every-updates", type=int, default=1000)
    custom.add_argument("--run-pure-control", action=argparse.BooleanOptionalAction, default=True)
    custom.add_argument("--export-bf16", action=argparse.BooleanOptionalAction, default=True)
    custom.add_argument(
        "--final-contexts", nargs="+", type=int,
        default=[2048, 16384, 32768, 65536],
    )
    custom.add_argument("--final-context-score-tokens", type=int, default=128)
    custom.add_argument("--final-context-windows", type=int, default=2)
    custom.add_argument(
        "--infer-contexts", nargs="+", type=int,
        default=[16384, 32768, 65536],
    )
    custom.add_argument("--infer-max-batch-tokens", type=int, default=524_288)
    custom.add_argument("--infer-max-batch", type=int, default=64)
    custom.add_argument("--infer-warmup", type=int, default=1)
    custom.add_argument("--infer-steps", type=int, default=3)
    custom.add_argument("--infer-min-free-gib", type=float, default=1.0)
    custom.add_argument("--infer-seed", type=int, default=731_031)
    custom.add_argument("--benchmark-pure-control", action=argparse.BooleanOptionalAction, default=True)
    custom_args, remaining = custom.parse_known_args()

    old = sys.argv
    try:
        sys.argv = [old[0], *remaining]
        args = v29.parse_args()
    finally:
        sys.argv = old
    for key, value in vars(custom_args).items():
        setattr(args, key, value)
    args.screen_token_budget = int(args.pcaf_screen_token_budget)
    args.quality_token_budget = int(args.pcaf_screen_token_budget)
    args.target_v28_results = str(args.target_v28_results)
    args.target_v28_starts = str(args.target_v28_starts)
    args.export_winner_bf16 = False
    v25.add_mamba_defaults(args)
    return args


def configure(args) -> None:
    v29.VERSION = VERSION
    v26.VERSION = VERSION
    v29.CANDIDATES = ALL_SPECS
    v29.configure(args)


def validate_args(args) -> None:
    quantum = int(args.train_seq) * int(args.batch_size)
    for name, value in (
        ("pcaf-screen-token-budget", args.pcaf_screen_token_budget),
        ("long-token-budget", args.long_token_budget),
    ):
        if value <= 0 or value % quantum:
            raise ValueError(f"{name}={value} must be positive and divisible by {quantum}")
    if args.pcaf_screen_token_budget >= args.long_token_budget:
        raise ValueError("PCAF screen must be shorter than long confirmation")
    if args.infer_steps < 1 or args.infer_warmup < 0:
        raise ValueError("invalid inference warmup/steps")
    if any(c < 2048 for c in args.infer_contexts):
        raise ValueError("inference contexts must be >= 2048")


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def result_row_for_candidate(raw: Mapping[str, object], candidate: str) -> Mapping[str, object]:
    rows = raw.get("results", {})
    if candidate not in rows:
        raise KeyError(f"result {candidate!r} not found; available={sorted(rows)}")
    return rows[candidate]


def result_row_for_model(raw: Mapping[str, object], model_name: str) -> Mapping[str, object]:
    rows = raw.get("results", {})
    found = [row for row in rows.values() if row.get("model") == model_name]
    if len(found) != 1:
        raise RuntimeError(f"expected one {model_name} row, found {len(found)}")
    return found[0]


def load_frozen_scoreboards(args, root: Path) -> Dict[str, object]:
    v30_path = Path(args.target_v30_results)
    v28_path = Path(args.target_v28_results)
    for p in (v30_path, v28_path):
        if not p.is_file():
            raise FileNotFoundError(p)
    r30 = read_json(v30_path)
    r28 = read_json(v28_path)
    if r30.get("canonical_sha256") != EXPECTED_CANONICAL_SHA256:
        raise RuntimeError("v30 canonical SHA mismatch")
    if r28.get("canonical_sha256") != EXPECTED_CANONICAL_SHA256:
        raise RuntimeError("v28 canonical SHA mismatch")

    on = result_row_for_candidate(r30, "field_mamba4_refresh1024x2_49m")
    final = {}
    for model in (TRANSFORMER, MAMBA2):
        row = result_row_for_model(r28, model)
        final[model] = {
            "validation_nll": float(row["final_validation"]["nll"]),
            "test_nll": float(row["final_test"]["nll"]),
            "tokens_per_second": float(row["tokens_per_second"]),
            "peak_gib": float(row["peak_gib"]),
            "checkpoint": str(row["checkpoint"]),
            "bf16_export": str(row.get("bf16_export", "")),
        }
    out = {
        "v30_path": str(v30_path),
        "v30_sha256": sha256(v30_path),
        "v28_path": str(v28_path),
        "v28_sha256": sha256(v28_path),
        "pcaf_on_49m": {
            "candidate": "field_mamba4_refresh1024x2_49m",
            "train_tokens": int(on["train_tokens"]),
            "validation_nll": float(on["final_validation"]["nll"]),
            "test_nll": float(on["final_test"]["nll"]),
            "tokens_per_second": float(on["tokens_per_second"]),
            "peak_gib": float(on["peak_gib"]),
            "checkpoint": str(on["checkpoint"]),
        },
        "frozen_98m": final,
        "v28_shapes": r28.get("shapes", {}),
    }
    if out["pcaf_on_49m"]["train_tokens"] != args.pcaf_screen_token_budget:
        raise RuntimeError("v30 PCAF-on row is not at the requested token budget")
    atomic_json(root / "frozen_scoreboards.json", out)
    return out


def make_starts(count: int, train_len: int, seed: int, path: Path) -> np.ndarray:
    return v26.make_example_starts(count, train_len, seed, path)


def audit_prefix(starts: np.ndarray, args, root: Path, label: str) -> Dict[str, object]:
    old_path = Path(args.target_v28_starts)
    if not old_path.is_file():
        raise FileNotFoundError(old_path)
    old = np.load(old_path)
    if len(old) < len(starts):
        raise RuntimeError(f"frozen starts shorter than {label}: {len(old)} < {len(starts)}")
    equal = bool(np.array_equal(starts, old[: len(starts)]))
    row = {
        "label": label,
        "count": int(len(starts)),
        "prefix_equal": equal,
        "new_sha256": hashlib.sha256(starts.tobytes()).hexdigest(),
        "v28_sha256": hashlib.sha256(old[: len(starts)].tobytes()).hexdigest(),
        "v28_path": str(old_path),
    }
    if not equal:
        raise AssertionError(f"{label} windows do not equal v28 prefix")
    atomic_json(root / f"paired_prefix_{label}.json", row)
    return row


def pcaf_retraining_decision(off: v29.ScreenResult, frozen: Mapping[str, object],
                             args) -> Dict[str, object]:
    on = frozen["pcaf_on_49m"]
    dval = float(off.final_validation["nll"] - on["validation_nll"])
    dtest = float(off.final_test["nll"] - on["test_nll"])
    speed_ratio = float(off.tokens_per_second / max(on["tokens_per_second"], 1e-9))
    memory_delta = float(off.peak_gib - on["peak_gib"])
    quality_pass = (
        dval <= args.pcaf_remove_max_val_loss
        and dtest <= args.pcaf_remove_max_test_loss
    )
    systems_pass = speed_ratio >= args.pcaf_remove_min_speed_ratio
    remove = bool(quality_pass and systems_pass)
    if dval < 0 and dtest < 0:
        rationale = "PCAF-off improves both validation and test; remove it."
    elif remove:
        rationale = (
            "PCAF-off stays within the predeclared quality tolerance and preserves/improves systems throughput; "
            "remove PCAF from the long candidate."
        )
    else:
        rationale = (
            "PCAF-off exceeds the allowed quality loss or fails the speed guard; keep PCAF for the 98M confirmation."
        )
    return {
        "selected_pcaf_enabled": not remove,
        "action": "REMOVE_PCAF" if remove else "KEEP_PCAF",
        "rationale": rationale,
        "pcaf_on": dict(on),
        "pcaf_off": asdict(off),
        "off_minus_on_validation_nll": dval,
        "off_minus_on_test_nll": dtest,
        "off_over_on_speed_ratio": speed_ratio,
        "off_minus_on_peak_gib": memory_delta,
        "quality_pass": quality_pass,
        "systems_pass": systems_pass,
        "thresholds": {
            "max_validation_loss": args.pcaf_remove_max_val_loss,
            "max_test_loss": args.pcaf_remove_max_test_loss,
            "min_speed_ratio": args.pcaf_remove_min_speed_ratio,
        },
    }


def export_bf16(path: Path, spec: CandidateSpec, shape: v23.Shape,
                result: v29.ScreenResult, args, deps,
                device: torch.device, pcaf_enabled: bool) -> None:
    _PCAF_ENABLED[spec.name] = bool(pcaf_enabled)
    model = v29.load_model_from_result(spec, shape, result, args, deps, device)
    state = {}
    for key, value in model.state_dict().items():
        cpu = value.detach().cpu()
        state[key] = cpu.to(torch.bfloat16) if cpu.is_floating_point() else cpu
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "format": "field_fusion_v31_bf16_inference",
        "version": VERSION,
        "candidate": asdict(spec),
        "shape": asdict(shape),
        "pcaf_enabled": bool(pcaf_enabled),
        "train_tokens": int(result.train_tokens),
        "state_dict": state,
        "args": vars(args),
    }, tmp)
    os.replace(tmp, path)
    del model
    clear_cuda()


def load_frozen_mamba(frozen: Mapping[str, object], args, deps,
                       device: torch.device) -> Tuple[nn.Module, v23.Shape, str]:
    raw28 = read_json(Path(frozen["v28_path"]))
    shapes = raw28.get("shapes", {})
    if MAMBA2 not in shapes:
        raise KeyError(f"v28 results have no shape for {MAMBA2}")
    shape = v23.Shape(**shapes[MAMBA2])
    model = v25.build_model_v25(MAMBA2, shape, args, deps, device).eval()
    row = result_row_for_model(raw28, MAMBA2)
    candidates = [str(row.get("checkpoint", "")), str(row.get("bf16_export", ""))]
    loaded = ""
    errors: List[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.is_file():
            errors.append(f"missing:{path}")
            continue
        obj = torch.load(path, map_location="cpu", weights_only=False)
        state = obj.get("model") if isinstance(obj, dict) else None
        if state is None and isinstance(obj, dict):
            state = obj.get("state_dict")
        if state is None:
            errors.append(f"no_state:{path}")
            continue
        model.load_state_dict(state, strict=True)
        loaded = str(path)
        break
    if not loaded:
        raise FileNotFoundError("could not load frozen v28 Mamba checkpoint: " + "; ".join(errors))
    return model, shape, loaded


def standard_batch_candidates(max_batch: int) -> List[int]:
    standards = [64, 48, 40, 32, 24, 20, 16, 12, 10, 8, 6, 5, 4, 3, 2, 1]
    out = [x for x in standards if x <= max_batch]
    if max_batch not in out:
        out.insert(0, max_batch)
    return sorted(set(out), reverse=True)


def make_infer_batch(data: torch.Tensor, batch: int, context: int,
                     seed: int, device: torch.device) -> torch.Tensor:
    x, _ = v23.batch_for_step(data, batch, context, seed + context * 17 + batch, 1, 0, device)
    return x


@torch.inference_mode()
def prefill_once(name: str, model: nn.Module, x: torch.Tensor,
                 device: torch.device, amp: str) -> torch.Tensor:
    with v23.amp_ctx(device, amp):
        if name == FUSION:
            h = model.emb(x)
            model._patch_aux = h.new_zeros(())
            for i, block in enumerate(model.blocks):
                h = block(h)
                if i == model.patch_position and model.softpatch is not None:
                    h = model.softpatch(h, x)
                    model._patch_aux = model.softpatch.last_aux
            h = model.final_norm(h)
        elif name == MAMBA2:
            h = model.emb(x).to(model.activation_dtype)
            for block in model.blocks:
                h = block(h)
            h = model.norm(h)
        else:
            raise KeyError(name)
        logits = model.lm_head(h[:, -1:, :])
        # Force all kernels, including the final projection, to complete.
        marker = logits.float().square().mean()
    return marker


@torch.inference_mode()
def try_prefill_benchmark(name: str, model: nn.Module, data: torch.Tensor,
                          context: int, batch: int, args,
                          device: torch.device) -> Dict[str, object]:
    clear_cuda()
    status, error = "ok", ""
    elapsed = tps = seqps = peak = baseline = free_after = None
    try:
        x = make_infer_batch(data, batch, context, args.infer_seed, device)
        for _ in range(args.infer_warmup):
            marker = prefill_once(name, model, x, device, args.amp)
            _ = float(marker.detach().cpu())
        sync(device)
        baseline = torch.cuda.memory_allocated(device) / 2**30
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(args.infer_steps):
            marker = prefill_once(name, model, x, device, args.amp)
        sync(device)
        elapsed = time.perf_counter() - started
        tps = args.infer_steps * batch * context / max(elapsed, 1e-9)
        seqps = args.infer_steps * batch / max(elapsed, 1e-9)
        peak = torch.cuda.max_memory_allocated(device) / 2**30
        free_after = torch.cuda.mem_get_info(device)[0] / 2**30
        if free_after < args.infer_min_free_gib:
            status = "low_headroom"
        del marker, x
    except torch.cuda.OutOfMemoryError as exc:
        status, error = "oom", str(exc).splitlines()[0]
    except Exception as exc:
        status, error = "error", repr(exc)
    clear_cuda()
    return {
        "model": name,
        "mode": "prefill_plus_last_token_head",
        "context": int(context),
        "batch": int(batch),
        "tokens_per_step": int(context * batch),
        "steps": int(args.infer_steps),
        "status": status,
        "seconds": elapsed,
        "tokens_per_second": tps,
        "sequences_per_second": seqps,
        "peak_gib": peak,
        "baseline_gib": baseline,
        "free_after_gib": free_after,
        "error": error,
    }


def benchmark_model_prefill(label: str, name: str, model: nn.Module,
                            data: torch.Tensor, args,
                            device: torch.device) -> List[Dict[str, object]]:
    model.eval()
    rows: List[Dict[str, object]] = []
    for context in map(int, args.infer_contexts):
        max_by_tokens = max(1, args.infer_max_batch_tokens // context)
        max_batch = min(int(args.infer_max_batch), int(max_by_tokens))
        selected: Optional[Dict[str, object]] = None
        for batch in standard_batch_candidates(max_batch):
            log(f"[infer/search] {label} ctx={context} batch={batch}")
            row = try_prefill_benchmark(name, model, data, context, batch, args, device)
            row["label"] = label
            row["comparison"] = "max_batch_search"
            rows.append(row)
            if row["status"] == "ok":
                selected = row
                break
            if row["status"] == "low_headroom":
                # Valid but intentionally back off once for a safer production batch.
                continue
        if selected is None:
            valid = [r for r in rows if r["label"] == label and r["context"] == context
                     and r["status"] in {"ok", "low_headroom"}]
            if valid:
                selected = valid[-1]
        if selected is None:
            log(f"[infer/result] {label} ctx={context} no viable batch")
        else:
            selected["selected"] = True
            log(
                f"[infer/result] {label} ctx={context} batch={selected['batch']} "
                f"tok/s={selected['tokens_per_second']:,.0f} peak={selected['peak_gib']:.2f}G"
            )
    return rows



def benchmark_model_fixed_batches(label: str, name: str, model: nn.Module,
                                  data: torch.Tensor, batches: Mapping[str, int],
                                  args, device: torch.device) -> List[Dict[str, object]]:
    model.eval()
    rows: List[Dict[str, object]] = []
    for context in map(int, args.infer_contexts):
        batch = int(batches[str(context)])
        log(f"[infer/matched] {label} ctx={context} batch={batch}")
        row = try_prefill_benchmark(name, model, data, context, batch, args, device)
        row["label"] = label
        row["comparison"] = "matched_batch"
        row["selected_matched"] = row["status"] in {"ok", "low_headroom"}
        rows.append(row)
        if row["selected_matched"]:
            log(
                f"[infer/matched-result] {label} ctx={context} batch={batch} "
                f"tok/s={row['tokens_per_second']:,.0f} peak={row['peak_gib']:.2f}G"
            )
    return rows


def common_infer_batches(rows: Sequence[Mapping[str, object]],
                         field_label: str) -> Dict[str, int]:
    selected = selected_infer_rows(rows)
    field = selected.get(field_label, {})
    mamba = selected.get("mamba2_frozen_v28", {})
    out: Dict[str, int] = {}
    for context in sorted(set(field) & set(mamba), key=int):
        out[str(context)] = min(int(field[context]["batch"]), int(mamba[context]["batch"]))
    return out

def selected_infer_rows(rows: Sequence[Mapping[str, object]]) -> Dict[str, Dict[str, Mapping[str, object]]]:
    out: Dict[str, Dict[str, Mapping[str, object]]] = {}
    for row in rows:
        if not row.get("selected"):
            continue
        out.setdefault(str(row["label"]), {})[str(row["context"])] = row
    return out


def inference_decision(rows: Sequence[Mapping[str, object]], hybrid_label: str) -> Dict[str, object]:
    selected = selected_infer_rows(rows)
    max_field = selected.get(hybrid_label, {})
    max_mamba = selected.get("mamba2_frozen_v28", {})
    matched_field = {str(r["context"]): r for r in rows
                     if r.get("label") == hybrid_label and r.get("selected_matched")}
    matched_mamba = {str(r["context"]): r for r in rows
                     if r.get("label") == "mamba2_frozen_v28" and r.get("selected_matched")}
    field = matched_field or max_field
    mamba = matched_mamba or max_mamba
    comparisons = []
    for context in sorted(set(field) & set(mamba), key=int):
        f, m = field[context], mamba[context]
        ftps, mtps = float(f["tokens_per_second"]), float(m["tokens_per_second"])
        fmem, mmem = float(f["peak_gib"]), float(m["peak_gib"])
        comparisons.append({
            "context": int(context),
            "field_batch": int(f["batch"]),
            "mamba_batch": int(m["batch"]),
            "field_tokens_per_second": ftps,
            "mamba_tokens_per_second": mtps,
            "field_over_mamba_speed_ratio": ftps / max(mtps, 1e-9),
            "field_peak_gib": fmem,
            "mamba_peak_gib": mmem,
            "field_minus_mamba_peak_gib": fmem - mmem,
            "field_speed_win": ftps > mtps,
            "field_memory_win": fmem < mmem,
            "field_pareto_win": ftps >= mtps and fmem <= mmem,
        })
    all_speed = bool(comparisons) and all(x["field_speed_win"] for x in comparisons)
    all_pareto = bool(comparisons) and all(x["field_pareto_win"] for x in comparisons)
    return {
        "mode": "prefill_plus_last_token_head_matched_batch",
        "note": "Backbone prefill plus one next-token vocabulary projection at matched batch; not incremental autoregressive decode.",
        "max_batch_capacity": {"field": max_field, "mamba": max_mamba},
        "hybrid_label": hybrid_label,
        "comparisons": comparisons,
        "field_wins_speed_at_all_contexts": all_speed,
        "field_pareto_wins_all_contexts": all_pareto,
    }


def final_quality_decision(long_results: Mapping[str, v29.ScreenResult],
                           frozen: Mapping[str, object]) -> Dict[str, object]:
    targets = frozen["frozen_98m"]
    rows = []
    for name, result in long_results.items():
        checks = {
            "beats_transformer_validation": result.final_validation["nll"] < targets[TRANSFORMER]["validation_nll"],
            "beats_mamba_validation": result.final_validation["nll"] < targets[MAMBA2]["validation_nll"],
            "beats_transformer_test": result.final_test["nll"] < targets[TRANSFORMER]["test_nll"],
            "beats_mamba_test": result.final_test["nll"] < targets[MAMBA2]["test_nll"],
        }
        rows.append({
            "candidate": name,
            "validation_nll": result.final_validation["nll"],
            "test_nll": result.final_test["nll"],
            "tokens_per_second": result.tokens_per_second,
            "peak_gib": result.peak_gib,
            "validation_minus_transformer": result.final_validation["nll"] - targets[TRANSFORMER]["validation_nll"],
            "validation_minus_mamba": result.final_validation["nll"] - targets[MAMBA2]["validation_nll"],
            "test_minus_transformer": result.final_test["nll"] - targets[TRANSFORMER]["test_nll"],
            "test_minus_mamba": result.final_test["nll"] - targets[MAMBA2]["test_nll"],
            "checks": checks,
            "beats_both": bool(all(checks.values())),
        })
    ranked = sorted(rows, key=lambda x: (x["validation_nll"], x["test_nll"]))
    winner = ranked[0] if ranked else None
    if winner and winner["beats_both"]:
        action = "RUN_FULL_THREE_SEED_ROUND_AND_CANONIZE"
        reason = "The 98M hybrid beat both frozen fair rivals on validation and test."
    elif winner and abs(winner["validation_minus_mamba"]) <= 0.020:
        action = "RUN_THREE_SEEDS_EQUIVALENCE"
        reason = "The best candidate is within the predefined 0.02-NLL equivalence band."
    else:
        action = "REFINE_BEFORE_FULL_RIVAL_RERUN"
        reason = "The frozen 98M quality targets were not both cleared."
    return {
        "action": action,
        "reason": reason,
        "winner": None if winner is None else winner["candidate"],
        "ranked": ranked,
        "frozen_targets": targets,
    }


def make_summary(args, frozen, pcaf_decision, off_result, long_results,
                 contexts, quality_decision, infer_decision, mamba_checkpoint,
                 audits) -> str:
    width = 230
    lines = [
        "=" * width,
        "FIELD-FUSION v31 — PCAF RETRAINING GATE / 98M CONFIRMATION / 16K–64K INFERENCE",
        "=" * width,
        f"PCAF screen={args.pcaf_screen_token_budget:,} tokens | long={args.long_token_budget:,} tokens/model | batch={args.batch_size} | context={args.train_seq}",
        f"paired_screen_prefix={audits['screen']['prefix_equal']} paired_long_prefix={audits['long']['prefix_equal']}",
        "No Transformer or pure Mamba model was retrained. Frozen Mamba weights are loaded only for inference benchmarking.",
        "",
        "PCAF RETRAINING ABLATION AT 49.152M",
        f"PCAF on  val={frozen['pcaf_on_49m']['validation_nll']:.5f} test={frozen['pcaf_on_49m']['test_nll']:.5f} tok/s={frozen['pcaf_on_49m']['tokens_per_second']:,.0f} peak={frozen['pcaf_on_49m']['peak_gib']:.2f}G",
        f"PCAF off val={off_result.final_validation['nll']:.5f} test={off_result.final_test['nll']:.5f} tok/s={off_result.tokens_per_second:,.0f} peak={off_result.peak_gib:.2f}G",
        f"off-on dVal={pcaf_decision['off_minus_on_validation_nll']:+.5f} dTest={pcaf_decision['off_minus_on_test_nll']:+.5f} speedRatio={pcaf_decision['off_over_on_speed_ratio']:.3f}",
        f"decision={pcaf_decision['action']} selected_pcaf_enabled={pcaf_decision['selected_pcaf_enabled']}",
        f"reason={pcaf_decision['rationale']}",
        "",
        "98.304M FINAL RESULTS",
        f"{'candidate':54s} {'val NLL':>10s} {'test NLL':>10s} {'tok/s':>10s} {'peakGB':>8s} {'2K→64K':>10s}",
    ]
    for row in quality_decision["ranked"]:
        ctx = contexts.get(row["candidate"], {})
        c2 = float(ctx.get("2048", {}).get("nll", float("nan")))
        c64 = float(ctx.get("65536", {}).get("nll", float("nan")))
        drift = c64 - c2 if math.isfinite(c2) and math.isfinite(c64) else float("inf")
        lines.append(
            f"{row['candidate']:54s} {row['validation_nll']:10.5f} {row['test_nll']:10.5f} "
            f"{row['tokens_per_second']:10,.0f} {row['peak_gib']:8.2f} {drift:+10.5f}"
        )
    lines += [
        "",
        "FROZEN 98M QUALITY TARGETS",
        f"Transformer val={frozen['frozen_98m'][TRANSFORMER]['validation_nll']:.5f} test={frozen['frozen_98m'][TRANSFORMER]['test_nll']:.5f}",
        f"Mamba-2    val={frozen['frozen_98m'][MAMBA2]['validation_nll']:.5f} test={frozen['frozen_98m'][MAMBA2]['test_nll']:.5f}",
        "",
        "LONG-CONTEXT PREFILL INFERENCE — HIGHEST SAFE BATCH",
        f"frozen_mamba_checkpoint={mamba_checkpoint}",
        "mode=backbone prefill + parametric last-token head; this is not incremental autoregressive decode",
        f"{'ctx':>8s} {'Field b':>8s} {'Mamba b':>8s} {'Field tok/s':>14s} {'Mamba tok/s':>14s} {'speed':>9s} {'Field GB':>10s} {'Mamba GB':>10s} {'Pareto':>8s}",
    ]
    for row in infer_decision["comparisons"]:
        lines.append(
            f"{row['context']:8,d} {row['field_batch']:8d} {row['mamba_batch']:8d} "
            f"{row['field_tokens_per_second']:14,.0f} {row['mamba_tokens_per_second']:14,.0f} "
            f"{row['field_over_mamba_speed_ratio']:9.3f} {row['field_peak_gib']:10.2f} "
            f"{row['mamba_peak_gib']:10.2f} {str(row['field_pareto_win']):>8s}"
        )
    lines += [
        "",
        "AUTOMATIC NEXT STEP",
        f"action={quality_decision['action']}",
        f"winner={quality_decision['winner']}",
        f"reason={quality_decision['reason']}",
        f"field_wins_mamba_prefill_speed_all={infer_decision['field_wins_speed_at_all_contexts']}",
        f"field_pareto_wins_mamba_all={infer_decision['field_pareto_wins_all_contexts']}",
        "No rival training or follow-up run is launched automatically.",
        "=" * width,
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required")
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
    if canonical_sha != EXPECTED_CANONICAL_SHA256:
        raise RuntimeError(f"canonical SHA mismatch: {canonical_sha}")
    frozen = load_frozen_scoreboards(args, root)

    # Solve all candidate shapes once.  PCAF on/off retains the same parameter
    # inventory, so any difference is purely training/computation, not width.
    v29.CANDIDATES = ALL_SPECS
    args.candidate = []
    base_shape, shapes, accounting = v29.solve_candidate_shapes(args, deps)
    atomic_json(root / "component_accounting.json", accounting)

    raw_rows = v23.core.load_raw_rows(args.cache_dir, args.data_frac)
    tokenizer = v23.base.copy_or_train_tokenizer(
        root, raw_rows[0], args.vocab_size,
        args.tokenizer_min_frequency, args.tokenizer_source,
    )
    train_c, val_c, test_c = v23.core.save_or_load_corpora(root, tokenizer, raw_rows)
    train = v23.core.place_tokens(train_c.tokens, device, args.data_device, "train")
    val = v23.core.place_tokens(val_c.tokens, device, args.data_device, "validation")
    test = v23.core.place_tokens(test_c.tokens, device, args.data_device, "test")

    # Phase A — PCAF-off retraining ablation at 49M.
    args.screen_token_budget = int(args.pcaf_screen_token_budget)
    args.quality_token_budget = int(args.pcaf_screen_token_budget)
    args.eval_fractions = [0.512, 1.0]
    args.checkpoint_every_updates = min(int(args.checkpoint_every_updates), 500)
    screen_sequences = args.pcaf_screen_token_budget // args.train_seq
    screen_starts = make_starts(
        screen_sequences, len(train) - args.train_seq - 1,
        args.data_seed, root / "pcaf_screen_starts.npy",
    )
    audit_screen = audit_prefix(screen_starts, args, root, "screen49m")
    _PCAF_ENABLED[PCAF_OFF_SCREEN.name] = False
    log("=" * 220)
    log("PHASE A — PCAF-OFF RETRAINING ABLATION AT 49.152M")
    off_result = v29.train_candidate(
        PCAF_OFF_SCREEN, shapes[PCAF_OFF_SCREEN.name], args, deps,
        train, val_c, val, test_c, test, screen_starts, root, device,
    )
    pcaf_decision = pcaf_retraining_decision(off_result, frozen, args)
    atomic_json(root / "pcaf_retraining_decision.json", pcaf_decision)
    selected_pcaf = bool(pcaf_decision["selected_pcaf_enabled"])
    state_path = root / "selected_pcaf_state.json"
    state_row = {"selected_pcaf_enabled": selected_pcaf, "decision": pcaf_decision["action"]}
    if state_path.is_file():
        prior = read_json(state_path)
        if bool(prior.get("selected_pcaf_enabled")) != selected_pcaf:
            raise RuntimeError(
                "PCAF decision changed while resumable long checkpoints exist. "
                "Use a fresh outdir before changing PCAF thresholds."
            )
    atomic_json(state_path, state_row)
    _PCAF_ENABLED[HYBRID_LONG.name] = selected_pcaf
    _PCAF_ENABLED[PURE_LONG.name] = selected_pcaf

    # Phase B — from-scratch 98M confirmation using the selected cache state.
    args.screen_token_budget = int(args.long_token_budget)
    args.quality_token_budget = int(args.long_token_budget)
    args.eval_fractions = list(map(float, args.long_eval_fractions))
    args.checkpoint_every_updates = int(args.long_checkpoint_every_updates)
    long_sequences = args.long_token_budget // args.train_seq
    long_starts = make_starts(
        long_sequences, len(train) - args.train_seq - 1,
        args.data_seed, root / "paired_example_starts.npy",
    )
    audit_long = audit_prefix(long_starts, args, root, "long98m")
    long_specs = [HYBRID_LONG] + ([PURE_LONG] if args.run_pure_control else [])
    long_results: Dict[str, v29.ScreenResult] = {}
    for spec in long_specs:
        log("=" * 220)
        log(
            f"PHASE B — 98M: {spec.name} pcaf={'on' if _PCAF_ENABLED[spec.name] else 'off'}"
        )
        long_results[spec.name] = v29.train_candidate(
            spec, shapes[spec.name], args, deps,
            train, val_c, val, test_c, test, long_starts, root, device,
        )
        atomic_json(root / "long_results.json", {k: asdict(v) for k, v in long_results.items()})

    contexts: Dict[str, Dict[str, Dict[str, float]]] = {}
    args.long_contexts = list(map(int, args.final_contexts))
    args.long_context_score_tokens = int(args.final_context_score_tokens)
    args.long_context_windows = int(args.final_context_windows)
    for spec in long_specs:
        _PCAF_ENABLED[spec.name] = selected_pcaf
        model = v29.load_model_from_result(spec, shapes[spec.name], long_results[spec.name], args, deps, device)
        contexts[spec.name] = v29.long_context_eval(model, test, args, device)
        del model
        clear_cuda()
    atomic_json(root / "long_context_results.json", contexts)

    quality_decision = final_quality_decision(long_results, frozen)
    atomic_json(root / "quality_decision.json", quality_decision)

    # Phase C — high-batch prefill inference.  Load only one model at a time.
    infer_rows: List[Dict[str, object]] = []
    hybrid_label = HYBRID_LONG.name
    hybrid_model = v29.load_model_from_result(
        HYBRID_LONG, shapes[HYBRID_LONG.name], long_results[HYBRID_LONG.name], args, deps, device
    )
    infer_rows.extend(benchmark_model_prefill(
        hybrid_label, FUSION, hybrid_model, test, args, device
    ))
    del hybrid_model
    clear_cuda()

    if args.benchmark_pure_control and PURE_LONG.name in long_results:
        pure_model = v29.load_model_from_result(
            PURE_LONG, shapes[PURE_LONG.name], long_results[PURE_LONG.name], args, deps, device
        )
        infer_rows.extend(benchmark_model_prefill(
            PURE_LONG.name, FUSION, pure_model, test, args, device
        ))
        del pure_model
        clear_cuda()

    mamba_model, mamba_shape, mamba_checkpoint = load_frozen_mamba(frozen, args, deps, device)
    infer_rows.extend(benchmark_model_prefill(
        "mamba2_frozen_v28", MAMBA2, mamba_model, test, args, device
    ))
    del mamba_model
    clear_cuda()

    # Re-run the principal Field and Mamba at the same batch per context.  The
    # first pass measures each model's maximum safe batch; this second pass is
    # the apples-to-apples speed and memory comparison.
    common_batches = common_infer_batches(infer_rows, hybrid_label)
    if set(common_batches) != {str(int(c)) for c in args.infer_contexts}:
        raise RuntimeError(f"could not establish common inference batches: {common_batches}")
    hybrid_model = v29.load_model_from_result(
        HYBRID_LONG, shapes[HYBRID_LONG.name], long_results[HYBRID_LONG.name], args, deps, device
    )
    infer_rows.extend(benchmark_model_fixed_batches(
        hybrid_label, FUSION, hybrid_model, test, common_batches, args, device
    ))
    del hybrid_model
    clear_cuda()
    mamba_model, _, _ = load_frozen_mamba(frozen, args, deps, device)
    infer_rows.extend(benchmark_model_fixed_batches(
        "mamba2_frozen_v28", MAMBA2, mamba_model, test, common_batches, args, device
    ))
    del mamba_model
    clear_cuda()

    expected_contexts = {str(int(c)) for c in args.infer_contexts}
    for label in (hybrid_label, "mamba2_frozen_v28"):
        completed = {str(r["context"]) for r in infer_rows
                     if r.get("label") == label and r.get("selected_matched")}
        if completed != expected_contexts:
            raise RuntimeError(f"matched-batch inference incomplete for {label}: {completed}")

    atomic_json(root / "inference_rows.json", infer_rows)
    infer_decision = inference_decision(infer_rows, hybrid_label)
    infer_decision["mamba_checkpoint"] = mamba_checkpoint
    infer_decision["mamba_shape"] = asdict(mamba_shape)
    atomic_json(root / "inference_decision.json", infer_decision)

    if args.export_bf16:
        export_dir = root / "exports"
        for spec in long_specs:
            export_bf16(
                export_dir / f"{spec.name}_step{long_results[spec.name].updates}_BF16.pt",
                spec, shapes[spec.name], long_results[spec.name], args, deps,
                device, selected_pcaf,
            )

    audits = {"screen": audit_screen, "long": audit_long}
    payload = {
        "version": VERSION,
        "args": vars(args),
        "canonical_source": str(canonical_path),
        "canonical_sha256": canonical_sha,
        "base_shape": asdict(base_shape),
        "candidate_shapes": {k: asdict(v) for k, v in shapes.items()},
        "component_accounting": accounting,
        "frozen_scoreboards": frozen,
        "paired_audits": audits,
        "pcaf_off_screen": asdict(off_result),
        "pcaf_retraining_decision": pcaf_decision,
        "long_results": {k: asdict(v) for k, v in long_results.items()},
        "long_contexts": contexts,
        "quality_decision": quality_decision,
        "inference_rows": infer_rows,
        "inference_decision": infer_decision,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "mamba_ssm": v25.MAMBA_VERSION,
    }
    atomic_json(root / "results.json", payload)
    text = make_summary(
        args, frozen, pcaf_decision, off_result, long_results,
        contexts, quality_decision, infer_decision, mamba_checkpoint,
        audits,
    )
    atomic_text(root / "summary.txt", text)
    log(text)


if __name__ == "__main__":
    main()
