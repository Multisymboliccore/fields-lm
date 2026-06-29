#!/usr/bin/env python3
"""FIELD-FUSION v33 — reliable gap-closure ablation with the exact v32 kernel patch.

This follow-up deliberately does NOT rerun v32 Phase A (16K/32K/64K prefill)
or Phase B (kernel sweep).  It validates and reuses their persisted JSON results,
promotes the exact BLOCK_C=32 / CHUNK_T=64 runtime geometry, and runs only the
paired 25.165824M-token quality ablation that v32 never started.

Reliability safeguards
----------------------
* paired training windows are copied directly from the frozen v28 starts file;
  no dependency on a helper from an older module and no RNG regeneration;
* the exact prefix is checked twice and hashed before any model is trained;
* the persisted v32 kernel winner must be exact, >=2% faster, and within 5% VRAM;
* the promoted kernel geometry is included in every checkpoint signature;
* tokenizer/tokenized corpora may be reused by symlink from v32;
* each arm remains independently resumable through the inherited v29 trainer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch

import field_fusion_gap_kernel_v32 as v32

VERSION = 33
EXPECTED_KERNEL_WINNER = "blockc32_t64"
EXPECTED_BLOCK_C = 32
EXPECTED_CHUNK_T = 64
EXPECTED_FIELD_CHUNK = 32


def log(value: object = "") -> None:
    print(str(value), flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument(
        "--source-v32-root",
        default="/home/ubuntu/pcaf_runs/field_fusion_gap_kernel_v32_run",
    )
    custom.add_argument(
        "--expected-kernel-winner",
        default=EXPECTED_KERNEL_WINNER,
    )
    custom.add_argument(
        "--package-selftest",
        action="store_true",
        help="Run CPU-only integrity tests and exit before CUDA/data loading.",
    )
    custom_args, remaining = custom.parse_known_args()

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining]
        args = v32.parse_args()
    finally:
        sys.argv = old_argv

    for key, value in vars(custom_args).items():
        setattr(args, key, value)

    # v33 runs only the missing quality phase.  Persisted A/B results are loaded.
    args.run_inference = False
    args.run_kernel_sweep = False
    args.run_gap_ablation = True

    # Exact winner from v32 Phase B.
    args.field_chunk = EXPECTED_FIELD_CHUNK
    args.triton_block_c = EXPECTED_BLOCK_C
    args.triton_chunk_t = EXPECTED_CHUNK_T

    # Keep the frozen paired-screen protocol.
    args.screen_token_budget = int(args.ablation_token_budget)
    args.quality_token_budget = int(args.ablation_token_budget)
    args.eval_fractions = list(map(float, args.ablation_eval_fractions))
    args.checkpoint_every_updates = int(args.ablation_checkpoint_every)
    args.profile_log_every_updates = int(args.ablation_log_every)
    args.screen_validation_token_budget = int(args.ablation_validation_token_budget)
    args.screen_test_token_budget = int(args.ablation_test_token_budget)
    args.export_winner_bf16 = True
    return args


def validate_reused_phases(args: argparse.Namespace, out_root: Path) -> tuple[Dict[str, object], Dict[str, object]]:
    source = Path(args.source_v32_root)
    inference_path = source / "inference_decision.json"
    kernel_path = source / "kernel_decision.json"
    if not inference_path.is_file():
        raise FileNotFoundError(inference_path)
    if not kernel_path.is_file():
        raise FileNotFoundError(kernel_path)

    inference = read_json(inference_path)
    kernel = read_json(kernel_path)

    comparisons = inference.get("comparisons", [])
    contexts = [int(row.get("context", -1)) for row in comparisons]
    if contexts != [16384, 32768, 65536]:
        raise AssertionError(f"unexpected reused inference contexts: {contexts}")
    for row in comparisons:
        required = (
            "field_tokens_per_second",
            "transformer_tokens_per_second",
            "mamba_tokens_per_second",
            "field_peak_gib",
            "transformer_peak_gib",
            "mamba_peak_gib",
        )
        if any(not math.isfinite(float(row.get(key, float("nan")))) for key in required):
            raise AssertionError(f"non-finite v32 inference row: {row}")

    winner = str(kernel.get("winner"))
    if winner != str(args.expected_kernel_winner):
        raise AssertionError(f"kernel winner mismatch: {winner!r}")
    if not bool(kernel.get("promote_runtime_patch")):
        raise AssertionError("v32 did not promote its runtime patch")
    if float(kernel.get("winner_speed_ratio", 0.0)) < 1.02:
        raise AssertionError("kernel speedup is below 2%")
    if float(kernel.get("winner_memory_ratio", float("inf"))) > 1.05:
        raise AssertionError("kernel memory ratio exceeds 1.05")

    winner_rows = [row for row in kernel.get("rows", []) if row.get("variant") == winner]
    if len(winner_rows) != 1:
        raise AssertionError(f"expected one kernel winner row, got {len(winner_rows)}")
    winner_row = winner_rows[0]
    if not bool(winner_row.get("exact")):
        raise AssertionError("promoted kernel is not exact")
    if float(winner_row.get("max_abs_logit_sample", float("inf"))) != 0.0:
        raise AssertionError("promoted kernel has a non-zero sampled logit difference")
    expected_geometry = {
        "field_chunk": EXPECTED_FIELD_CHUNK,
        "triton_block_c": EXPECTED_BLOCK_C,
        "triton_chunk_t": EXPECTED_CHUNK_T,
    }
    actual_geometry = {key: int(winner_row.get(key, -1)) for key in expected_geometry}
    if actual_geometry != expected_geometry:
        raise AssertionError(f"kernel geometry mismatch: {actual_geometry}")

    audit = {
        "source_v32_root": str(source),
        "inference_decision": str(inference_path),
        "inference_sha256": sha256(inference_path),
        "kernel_decision": str(kernel_path),
        "kernel_sha256": sha256(kernel_path),
        "contexts": contexts,
        "kernel_winner": winner,
        "kernel_speed_ratio": float(kernel["winner_speed_ratio"]),
        "kernel_memory_ratio": float(kernel["winner_memory_ratio"]),
        "kernel_geometry": actual_geometry,
        "kernel_exact": True,
    }
    atomic_json(out_root / "reused_v32_audit.json", audit)
    atomic_json(out_root / "reused_inference_decision.json", inference)
    atomic_json(out_root / "reused_kernel_decision.json", kernel)
    return inference, kernel


def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def make_starts_from_v28(count: int, upper: int, seed: int, path: Path) -> np.ndarray:
    """Return the exact frozen v28 prefix; never regenerate paired windows."""
    del seed
    source_path = Path(_ACTIVE_TARGET_V28_STARTS)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    frozen = np.load(source_path, allow_pickle=False)
    if frozen.ndim != 1 or len(frozen) < int(count):
        raise RuntimeError(f"invalid/short v28 starts: shape={frozen.shape}, need={count}")
    prefix = np.asarray(frozen[: int(count)], dtype=np.int64).copy()
    if prefix.size:
        if int(prefix.min()) < 0 or int(prefix.max()) > int(upper):
            raise RuntimeError(
                f"v28 starts out of range: min={int(prefix.min())} "
                f"max={int(prefix.max())} upper={int(upper)}"
            )

    if path.is_file():
        existing = np.load(path, allow_pickle=False)
        if existing.dtype != prefix.dtype or not np.array_equal(existing, prefix):
            raise AssertionError(f"existing paired starts do not match frozen v28 prefix: {path}")
    else:
        _save_npy_atomic(path, prefix)
    return prefix


_ACTIVE_TARGET_V28_STARTS = ""


def paired_starts_selftest(args: argparse.Namespace, out_root: Path) -> Dict[str, object]:
    source = Path(args.target_v28_starts)
    frozen = np.load(source, allow_pickle=False)
    count = int(args.ablation_token_budget) // int(args.train_seq)
    if len(frozen) < count:
        raise RuntimeError(f"frozen v28 starts too short: {len(frozen)} < {count}")
    upper = int(np.max(frozen[:count])) + 1
    with tempfile.TemporaryDirectory(prefix="field_v33_starts_") as temp_dir:
        target = Path(temp_dir) / "paired.npy"
        first = make_starts_from_v28(count, upper, int(args.data_seed), target)
        second = make_starts_from_v28(count, upper, int(args.data_seed) + 999, target)
        expected = np.asarray(frozen[:count], dtype=np.int64)
        if not np.array_equal(first, expected) or not np.array_equal(second, expected):
            raise AssertionError("paired-start selftest mismatch")
        if sha256(target) != sha256(target):
            raise AssertionError("paired-start hash instability")
    row = {
        "source": str(source),
        "source_sha256": sha256(source),
        "count": count,
        "prefix_bytes_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
        "dtype": str(expected.dtype),
        "min": int(expected.min()) if expected.size else None,
        "max": int(expected.max()) if expected.size else None,
        "repeat_equal": True,
        "seed_independent": True,
    }
    atomic_json(out_root / "paired_starts_selftest.json", row)
    return row


def install_v33_hooks(args: argparse.Namespace) -> None:
    global _ACTIVE_TARGET_V28_STARTS
    _ACTIVE_TARGET_V28_STARTS = str(args.target_v28_starts)
    v32.VERSION = VERSION
    v32.v29.VERSION = VERSION
    v32.v26.VERSION = VERSION
    v32.make_starts = make_starts_from_v28

    # Extend v32's checkpoint signature so an old geometry can never resume into
    # the exact BLOCK_C=32 run silently.
    prior_signature = v32.v29.checkpoint_signature

    def signature(args_, spec, shape, total_sequences):
        row = dict(prior_signature(args_, spec, shape, total_sequences))
        row["v33_version"] = VERSION
        row["v33_runtime_geometry"] = {
            "field_chunk": int(args_.field_chunk),
            "triton_block_c": int(args_.triton_block_c),
            "triton_chunk_t": int(args_.triton_chunk_t),
        }
        row["v33_starts_source_sha256"] = sha256(Path(args_.target_v28_starts))
        return row

    v32.v29.checkpoint_signature = signature


def preflight_report(args: argparse.Namespace, out_root: Path, reused_a: Mapping[str, object], reused_b: Mapping[str, object], starts: Mapping[str, object]) -> Dict[str, object]:
    row = {
        "version": VERSION,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_available": torch.cuda.is_available(),
        "bf16": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        "runtime_geometry": {
            "field_chunk": int(args.field_chunk),
            "triton_block_c": int(args.triton_block_c),
            "triton_chunk_t": int(args.triton_chunk_t),
        },
        "reused_inference_contexts": [int(x["context"]) for x in reused_a.get("comparisons", [])],
        "reused_kernel_winner": reused_b.get("winner"),
        "paired_starts": dict(starts),
        "selected_arms": [arm.name for arm in v32.selected_arms(args)],
    }
    atomic_json(out_root / "preflight.json", row)
    return row


def make_summary(
    args: argparse.Namespace,
    reused_inference: Mapping[str, object],
    reused_kernel: Mapping[str, object],
    gap: Mapping[str, object],
) -> str:
    width = 220
    lines = [
        "=" * width,
        "FIELD-FUSION v33 — EXACT-KERNEL QUALITY GAP CLOSURE",
        "=" * width,
        "v32 Phase A/B reused after SHA/integrity validation; no rival was trained or benchmarked again.",
        f"kernel={reused_kernel.get('winner')} speed={float(reused_kernel.get('winner_speed_ratio', float('nan'))):.4f}x "
        f"memory={float(reused_kernel.get('winner_memory_ratio', float('nan'))):.4f}x geometry=field32/blockC32/chunkT64",
        "",
        "REUSED 16K/32K/64K MATCHED-BATCH PREFILL",
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
        "25.165824M PAIRED QUALITY GAP SCREEN",
        f"{'candidate':30s} {'val':>9s} {'test':>9s} {'dVal':>9s} {'tok/s':>11s} {'speed':>8s} {'GB':>7s} {'2K→64K':>10s} {'eligible':>9s}",
    ]
    for row in gap.get("rows", []):
        drift = row.get("context_2k_to_64k")
        drift_text = "n/a" if drift is None else f"{float(drift):+.5f}"
        lines.append(
            f"{str(row['candidate']):30s} {float(row['validation_nll']):9.5f} "
            f"{float(row['test_nll']):9.5f} {float(row['validation_gain_vs_baseline']):+9.5f} "
            f"{float(row['tokens_per_second']):11,.0f} {float(row['speed_ratio_vs_baseline']):8.3f} "
            f"{float(row['peak_gib']):7.2f} {drift_text:>10s} {str(bool(row['eligible'])):>9s}"
        )
    lines += [
        "",
        "AUTOMATIC NEXT STEP",
        f"action={gap.get('action')}",
        f"winner={gap.get('winner')}",
        "No long run is launched automatically.",
        "=" * width,
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    v32.validate_paths(args)
    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)

    reused_inference, reused_kernel = validate_reused_phases(args, root)
    install_v33_hooks(args)
    starts_audit = paired_starts_selftest(args, root)
    preflight_report(args, root, reused_inference, reused_kernel, starts_audit)
    atomic_json(root / "args.json", vars(args))

    log("=" * 220)
    log("FIELD-FUSION v33 — PRE-RUN AUDIT")
    log(f"reused_inference_contexts={[x['context'] for x in reused_inference['comparisons']]}")
    log(
        f"reused_kernel={reused_kernel['winner']} "
        f"speed={float(reused_kernel['winner_speed_ratio']):.4f}x "
        f"memory={float(reused_kernel['winner_memory_ratio']):.4f}x"
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

    log("=" * 220)
    log("PHASE C ONLY — PAIRED 25.165824M QUALITY GAP ABLATION")
    gap_decision = v32.run_gap_ablation(args, root, device)

    payload = {
        "version": VERSION,
        "args": vars(args),
        "reused_inference_decision": reused_inference,
        "reused_kernel_decision": reused_kernel,
        "gap_decision": gap_decision,
        "source_v32_root": str(args.source_v32_root),
        "source_v32_kernel_sha256": sha256(Path(args.source_v32_root) / "kernel_decision.json"),
        "source_v32_inference_sha256": sha256(Path(args.source_v32_root) / "inference_decision.json"),
        "target_v28_starts_sha256": sha256(Path(args.target_v28_starts)),
    }
    atomic_json(root / "results.json", payload)
    summary = make_summary(args, reused_inference, reused_kernel, gap_decision)
    atomic_text(root / "summary.txt", summary)
    log(summary)


if __name__ == "__main__":
    main()
