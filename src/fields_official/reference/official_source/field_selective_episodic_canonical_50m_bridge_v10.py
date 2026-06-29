#!/usr/bin/env python3
"""Canonical 50M bridge for selective episodic corrective memory.

This arena transplants the v9 structural winners onto the validated canonical
Field-v4 Triton recurrence and the exact v6 speed optimizations.  The purpose is
to determine whether the short portable gains survive at ~50M parameters,
8K training context, one 10%-WikiText-103 epoch, and strict parameter parity.

Primary comparisons
-------------------
* hybrid_ref_opt                     current best optimized hybrid baseline
* selective_write_opt                selective episodic writes only
* selective_residual_opt             selective writes + corrective residual
* selective_hierarchical_opt         recent/old corrective banks
* attentionfree_ref_opt              current optimized query-key-free baseline
* attentionfree_selective_residual_opt  structural winner without QK attention
* transformer_flash_sdpa             strong wide Flash-SDPA external reference

The primary promotion metric is full-window TEST BPB@8K.  Validation is used
only for progress tracking.  Every model receives fresh paired initialization,
identical byte windows, 16,384 bytes/update, BF16, and no activation checkpoint.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
V3_PATH = HERE / "field_hybrid_attentionfree_qualification_v3.py"
BRIDGE_PATH = HERE / "field_hybrid_canonical_50m_bridge_v4.py"
OPT_PATH = HERE / "field_hybrid_speed_optimization_v6.py"
V9_PATH = HERE / "field_selective_episodic_memory_ablation_v9.py"
JUDGE_PATH = HERE / "field_transformer_judge_repair_50m_8k_v5.py"
CANONICAL_NAME = "field_only_v4_chunked_triton_wiki100.py"
EXPECTED_CANONICAL_SHA256 = "0454647a7f667cdf73f45029eee0f525c58c0788ecd433ec1a4db8c3937ac848"
VOCAB = 256
LN2 = math.log(2.0)

HYBRID_REF = "hybrid_ref_opt"
SELECTIVE_WRITE = "selective_write_opt"
SELECTIVE_RESIDUAL = "selective_residual_opt"
SELECTIVE_HIERARCHICAL = "selective_hierarchical_opt"
AF_REF = "attentionfree_ref_opt"
AF_SELECTIVE_RESIDUAL = "attentionfree_selective_residual_opt"
TRANSFORMER = "transformer_flash_sdpa"
FIELD_NAMES = (
    HYBRID_REF,
    SELECTIVE_WRITE,
    SELECTIVE_RESIDUAL,
    SELECTIVE_HIERARCHICAL,
    AF_REF,
    AF_SELECTIVE_RESIDUAL,
)
MODEL_NAMES = (*FIELD_NAMES, TRANSFORMER)


def log(msg: object = "") -> None:
    print(str(msg), flush=True)


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def nparams(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def locate_canonical(explicit: str) -> Path:
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend([
        HERE / CANONICAL_NAME,
        Path("/home/ubuntu") / CANONICAL_NAME,
        Path("/home/ubuntu/field_hybrid_canonical_50m_bridge_v4") / CANONICAL_NAME,
        Path("/home/ubuntu/field_hybrid_300m_h2h_v7") / CANONICAL_NAME,
        Path("/home/ubuntu/field_pcaf_efficiency_v1") / CANONICAL_NAME,
    ])
    for p in candidates:
        if p.is_file():
            return p.resolve()
    tried = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Validated canonical source not found. Tried:\n  {tried}\n"
        f"Upload {CANONICAL_NAME} beside this script."
    )


def is_attention_free(name: str) -> bool:
    return name in (AF_REF, AF_SELECTIVE_RESIDUAL)


def bridge_arm(name: str) -> str:
    return "attentionfree_multiscale" if is_attention_free(name) else "hybrid_w256_conf_parity"


def cache_mode(name: str) -> Tuple[bool, bool, bool]:
    if name == SELECTIVE_WRITE:
        return True, False, False
    if name in (SELECTIVE_RESIDUAL, AF_SELECTIVE_RESIDUAL):
        return True, True, False
    if name == SELECTIVE_HIERARCHICAL:
        return True, True, True
    return False, False, False


def make_bridge_args(args, hidden: int) -> argparse.Namespace:
    return argparse.Namespace(
        model_seed=args.model_seed,
        field_dim=args.field_dim,
        field_layers=args.field_layers,
        field_heads=args.field_heads,
        field_ff_hidden=int(hidden),
        field_chunk=args.field_chunk,
        triton_block_c=args.triton_block_c,
        triton_chunk_t=args.triton_chunk_t,
        num_buckets=args.num_buckets,
        tf_dim=1,
        tf_heads=1,
        tf_layers=1,
        tf_ff_hidden=1,
    )


def hidden_for(name: str, args) -> int:
    return args.af_ff_hidden if is_attention_free(name) else args.hybrid_ff_hidden


def install_fast_candidate_route(epi, optmod) -> None:
    """Use the exact v6 int32 sort route inside the v9 episodic cache.

    Keep an immutable proxy to the original fallback so monkeypatching cannot
    recurse if an unusually large key range ever exceeds int32 safety.
    """
    original = epi.v3.causal_recent_candidates
    proxy = SimpleNamespace(
        causal_ngram_buckets=epi.v3.causal_ngram_buckets,
        causal_recent_candidates=original,
    )

    def fast(tokens, order, num_buckets, top_k):
        return optmod.causal_recent_candidates_i32(tokens, order, num_buckets, top_k, proxy)

    epi.v3.causal_recent_candidates = fast


def build_field(
    name: str,
    args,
    v3,
    canonical,
    bridge,
    optmod,
    epi,
    device: torch.device,
) -> nn.Module:
    seed_all(args.model_seed)
    hidden = hidden_for(name, args)
    bargs = make_bridge_args(args, hidden)
    model = bridge.build_field(bridge_arm(name), bargs, v3, canonical, hidden).to(device)

    # Freeze the exact v6 systems path before introducing the new memory.
    optmod.replace_softpatch(model, v3)
    optmod.replace_cache(model, v3, i32=True)
    # v9's wrapper reads the validated feature width from the base cache.
    # The exact v6 fast cache mirrors the module tree but did not expose this
    # class constant as an instance attribute.
    model.cache.FEATURE_DIM = int(v3.SuccessorCacheV5.FEATURE_DIM)
    if is_attention_free(name):
        optmod.replace_multiscale(model, v3, lite=False)
    else:
        optmod.replace_local(model, v3, "cached", args.local_chunk)

    selective, residual, hierarchical = cache_mode(name)
    if selective or residual or hierarchical:
        model.cache = epi.EpisodicCorrectiveCache(
            model.cache,
            selective=selective,
            residual=residual,
            hierarchical=hierarchical,
            salience_floor=args.salience_floor,
            residual_limit=args.residual_limit,
        ).to(device)
    return model


def build_transformer(args, judge, v3, device: torch.device) -> nn.Module:
    seed_all(args.model_seed)
    return judge.StrongFlashTransformerLM(
        args.tf_dim, args.tf_heads, args.tf_layers, args.tf_ff_hidden, v3
    ).to(device)


def build_model(
    name: str,
    args,
    v3,
    canonical,
    bridge,
    optmod,
    epi,
    judge,
    device: torch.device,
) -> nn.Module:
    if name == TRANSFORMER:
        return build_transformer(args, judge, v3, device)
    return build_field(name, args, v3, canonical, bridge, optmod, epi, device)


def resolve_shapes(args, v3, canonical, bridge, optmod, epi, judge, device):
    shapes: Dict[str, object] = {}
    for name in MODEL_NAMES:
        model = build_model(name, args, v3, canonical, bridge, optmod, epi, judge, torch.device("cpu"))
        p = nparams(model)
        if name == TRANSFORMER:
            dim, layers, heads, ff, af = (
                args.tf_dim, args.tf_layers, args.tf_heads, args.tf_ff_hidden, False
            )
        else:
            dim, layers, heads, ff, af = (
                args.field_dim, args.field_layers, args.field_heads, hidden_for(name, args), is_attention_free(name)
            )
        shapes[name] = bridge.ModelShape(
            name=name, params=p, dim=dim, layers=layers, heads=heads,
            ff_hidden=ff, attention_free=af,
        )
        del model
    gc.collect()
    target = shapes[HYBRID_REF].params
    for name, shape in shapes.items():
        delta = 100.0 * (shape.params - target) / target
        if abs(delta) > args.max_param_delta_pct:
            raise RuntimeError(
                f"parameter mismatch {name}: {shape.params:,} ({delta:+.3f}%) "
                f"exceeds {args.max_param_delta_pct:.3f}%"
            )
    return shapes


def set_distill_scale(name: str, model: nn.Module, step: int, args) -> float:
    if name == TRANSFORMER:
        return 0.0
    value = min(1.0, step / max(float(args.conf_distill_ramp), 1.0))
    cache = getattr(model, "cache", None)
    if cache is not None:
        cache.distill_scale = value
        base = getattr(cache, "base", None)
        if base is not None:
            base.distill_scale = value
    return value


def lr_for(name: str, args) -> float:
    return args.transformer_lr if name == TRANSFORMER else args.field_lr


def group_ref(name: str) -> Optional[str]:
    if name in (HYBRID_REF, AF_REF, TRANSFORMER):
        return None
    return AF_REF if is_attention_free(name) else HYBRID_REF


def configure_bridge_runtime(bridge, args, v3, canonical, optmod, epi, judge, device):
    """Patch v4's proven trainer to use the v10 model factory."""
    def factory(name, _args, _v3, _canonical, _shapes, _device):
        return build_model(name, args, v3, canonical, bridge, optmod, epi, judge, device)

    def distill(name, model, step):
        return set_distill_scale(name, model, step, args)

    bridge.build_model = factory
    bridge.set_distill_scale = distill


def load_trained_model(name: str, root: Path, args, v3, canonical, bridge, optmod, epi, judge, device):
    model = build_model(name, args, v3, canonical, bridge, optmod, epi, judge, device)
    ckpt = root / "models" / name / "latest.pt"
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    return model


def windows_for_budget(context: int, token_budget: int) -> int:
    return max(1, int(math.ceil(token_budget / context)))


def add_test_evaluation(
    name: str,
    result: dict,
    root: Path,
    test: torch.Tensor,
    args,
    v3,
    canonical,
    bridge,
    optmod,
    epi,
    judge,
    device,
) -> dict:
    model = load_trained_model(name, root, args, v3, canonical, bridge, optmod, epi, judge, device)
    rows = []
    for context in args.final_contexts:
        windows = windows_for_budget(context, args.final_eval_token_budget)
        row = bridge.evaluate(
            name, model, test, context, windows, args.eval_seed + 50_000,
            device, args.amp, v3,
        )
        rows.append(asdict(row))
        log(f"[{name}] TEST context={context} bpb={row.bpb:.5f}")
    result = dict(result)
    result["test_eval"] = rows
    cache = getattr(model, "cache", None)
    if cache is not None and hasattr(cache, "residual_gain_recent"):
        with torch.no_grad():
            result["residual_gain_recent"] = float(
                cache.residual_limit * torch.tanh(cache.residual_gain_recent.float())
            )
            long = getattr(cache, "residual_gain_long", None)
            result["residual_gain_long"] = (
                float(cache.residual_limit * torch.tanh(long.float())) if long is not None else 0.0
            )
            raw = getattr(cache, "salience_strength_raw", None)
            result["salience_strength"] = float(torch.nn.functional.softplus(raw.float())) if raw is not None else 0.0
    atomic_json(root / "models" / name / "result.json", result)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def get_eval(result: dict, context: int, split: str = "test_eval") -> dict:
    for row in result[split]:
        if int(row["context"]) == int(context):
            return row
    raise KeyError((result.get("model"), split, context))


def exact_candidate_selftest(v3, optmod, device):
    tok = torch.randint(0, VOCAB, (2, 257), device=device)
    ref = v3.causal_recent_candidates(tok, 4, 8192, 8)
    fast = optmod.causal_recent_candidates_i32(tok, 4, 8192, 8, v3)
    if not torch.equal(ref, fast):
        raise AssertionError("int32 candidate route mismatch")
    log("[selftest] exact int32 candidate route PASS")


def run_selftest(args, shapes, device, v3, canonical, bridge, optmod, epi, judge):
    log("[selftest] canonical Triton, exact optimized side paths, episodic causality")
    test_args = argparse.Namespace(**vars(args))
    test_args.dim = 64
    test_args.heads = 4
    test_args.layers = 2
    test_args.field_chunk = 8
    test_args.triton_block_c = 8
    test_args.triton_chunk_t = 16
    test_args.models = ["field_reference", "field_triton"]
    test_args.checkpoint_blocks = False
    test_args.max_param_delta_pct = 1.0
    test_args.selftest_forward_tol = 0.002
    test_args.selftest_grad_rel_tol = 0.02
    test_args.selftest_grad_abs_tol = 0.002
    test_args.selftest_causal_tol = 0.0002
    canonical.run_kernel_self_test(device, test_args)
    log("[selftest] canonical Field reference/Triton PASS")
    exact_candidate_selftest(v3, optmod, device)

    # Same-shape backbones must remain exactly paired before memory changes.
    ref = build_model(HYBRID_REF, args, v3, canonical, bridge, optmod, epi, judge, device)
    sel = build_model(SELECTIVE_RESIDUAL, args, v3, canonical, bridge, optmod, epi, judge, device)
    max_abs = 0.0
    for p, q in zip(ref.blocks.parameters(), sel.blocks.parameters()):
        max_abs = max(max_abs, float((p - q).abs().max()))
    log(f"[selftest] paired canonical backbone max_abs={max_abs:.3e}")
    if max_abs != 0.0:
        raise AssertionError("paired backbone mismatch")
    del ref, sel

    x = torch.randint(0, VOCAB, (1, 97), device=device)
    y = torch.randint(0, VOCAB, (1, 97), device=device)
    for name in MODEL_NAMES:
        model = build_model(name, args, v3, canonical, bridge, optmod, epi, judge, device).train()
        set_distill_scale(name, model, args.conf_distill_ramp, args)
        with bridge.amp_ctx(device, args.amp):
            loss, primary = bridge.loss_for(name, model, x, y)
        loss.backward()
        finite = bool(torch.isfinite(loss)) and all(
            p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()
        )
        log(
            f"[selftest] {name:<42} params={nparams(model):,} "
            f"loss={float(loss.detach()):.5f} primary={float(primary):.5f} finite={finite}"
        )
        if not finite:
            raise AssertionError(name)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Full-cache causal test: suffix changes may not alter earlier token NLL.
    for name in (SELECTIVE_RESIDUAL, SELECTIVE_HIERARCHICAL, AF_SELECTIVE_RESIDUAL):
        model = build_model(name, args, v3, canonical, bridge, optmod, epi, judge, device).eval()
        seq_a = torch.randint(0, VOCAB, (1, 130), device=device)
        seq_b = seq_a.clone()
        seq_b[:, 90:] = torch.randint(0, VOCAB, seq_b[:, 90:].shape, device=device)
        xa, ya = seq_a[:, :-1], seq_a[:, 1:]
        xb, yb = seq_b[:, :-1], seq_b[:, 1:]
        with torch.no_grad(), bridge.amp_ctx(device, args.amp):
            sa, la = model.states_logits(xa)
            sb, lb = model.states_logits(xb)
            na = model.cache.token_nll(sa, la, xa, ya)
            nb = model.cache.token_nll(sb, lb, xb, yb)
        err = float((na[:, :88] - nb[:, :88]).abs().max())
        log(f"[selftest] {name} causal_token_nll max_abs={err:.3e}")
        if err > args.causal_tol:
            raise AssertionError((name, err))
        del model
    log("[selftest] PASS")


def make_summary(args, canonical_path: Path, shapes, results, systems) -> str:
    width = 210
    lines = [
        "=" * width,
        "CANONICAL 50M SELECTIVE EPISODIC MEMORY BRIDGE v10",
        "=" * width,
        f"canonical_source={canonical_path} sha256={sha256(canonical_path)}",
        (
            f"protocol: WikiText-103 {args.data_frac:.1%} | train ctx={args.train_seq} | "
            f"bytes/update={args.batch_size*args.accum*args.train_seq:,} | epochs={args.epochs} | "
            f"BF16 | no activation checkpointing"
        ),
        "Primary score: full-window TEST BPB@8K. All structural variants use the optimized v6 exact path.",
        "",
        "MODEL SHAPES",
        f"{'model':44s} {'params':>13s} {'d%':>8s} {'dim':>6s} {'layers':>7s} {'ff':>7s} {'AF':>4s}",
    ]
    target = shapes[HYBRID_REF].params
    for name in MODEL_NAMES:
        s = shapes[name]
        delta = 100.0 * (s.params - target) / target
        lines.append(
            f"{name:44s} {s.params:13,d} {delta:+8.3f} {s.dim:6d} {s.layers:7d} "
            f"{s.ff_hidden:7d} {str(s.attention_free):>4s}"
        )

    lines.extend([
        "",
        "FINAL QUALITY — TEST SPLIT",
        (
            f"{'model':44s} {'LR':>10s} {'BPB4K':>9s} {'BPB8K':>9s} {'dGroup':>9s} "
            f"{'BPB16K':>9s} {'oracle8':>9s} {'cap8':>7s} {'sep8':>7s} "
            f"{'gainR':>8s} {'train B/s':>12s} {'peak':>7s}"
        ),
    ])
    for name in MODEL_NAMES:
        r = results[name]
        e4 = get_eval(r, 4096)
        e8 = get_eval(r, 8192)
        e16 = get_eval(r, 16384)
        ref_name = group_ref(name)
        dgroup = 0.0 if ref_name is None else e8["bpb"] - get_eval(results[ref_name], 8192)["bpb"]
        oracle = "-" if e8.get("oracle_bpb") is None else f"{e8['oracle_bpb']:.5f}"
        cap = "-" if e8.get("capture") is None else f"{e8['capture']:.3f}"
        sep = "-" if e8.get("gate_sep") is None else f"{e8['gate_sep']:+.3f}"
        gain = r.get("residual_gain_recent")
        gain_s = "-" if gain is None else f"{gain:+.3f}"
        lines.append(
            f"{name:44s} {r['lr']:10.3e} {e4['bpb']:9.5f} {e8['bpb']:9.5f} "
            f"{dgroup:+9.5f} {e16['bpb']:9.5f} {oracle:>9s} {cap:>7s} {sep:>7s} "
            f"{gain_s:>8s} {r['train_bytes_per_second']:12,.0f} {r['train_peak_gib']:7.2f}"
        )

    lines.extend([
        "",
        "EQUAL NO-CHECKPOINT SYSTEMS BENCHMARK",
        f"{'model':44s} {'ctx':>7s} {'batch':>6s} {'status':>8s} {'B/s':>12s} {'step ms':>10s} {'peak GB':>8s}",
    ])
    for row in systems:
        bps = "-" if row.get("bytes_per_second") is None else f"{row['bytes_per_second']:,.0f}"
        ms = "-" if row.get("step_ms") is None else f"{row['step_ms']:.2f}"
        peak = "-" if row.get("peak_gib") is None else f"{row['peak_gib']:.2f}"
        lines.append(
            f"{row['model']:44s} {row['context']:7d} {row['batch']:6d} {row['status']:>8s} "
            f"{bps:>12s} {ms:>10s} {peak:>8s}"
        )

    sys_map = {(x["model"], int(x["context"])): x for x in systems}
    h_ref8 = get_eval(results[HYBRID_REF], 8192)["bpb"]
    sr8 = get_eval(results[SELECTIVE_RESIDUAL], 8192)["bpb"]
    sr16 = get_eval(results[SELECTIVE_RESIDUAL], 16384)["bpb"]
    h_ref16 = get_eval(results[HYBRID_REF], 16384)["bpb"]
    af_ref8 = get_eval(results[AF_REF], 8192)["bpb"]
    af8 = get_eval(results[AF_SELECTIVE_RESIDUAL], 8192)["bpb"]
    af_ref16 = get_eval(results[AF_REF], 16384)["bpb"]
    af16 = get_eval(results[AF_SELECTIVE_RESIDUAL], 16384)["bpb"]
    sr_speed = (
        (sys_map[(SELECTIVE_RESIDUAL, 8192)].get("bytes_per_second") or 0.0) /
        max(sys_map[(HYBRID_REF, 8192)].get("bytes_per_second") or 1.0, 1.0)
    )
    af_speed = (
        (sys_map[(AF_SELECTIVE_RESIDUAL, 8192)].get("bytes_per_second") or 0.0) /
        max(sys_map[(AF_REF, 8192)].get("bytes_per_second") or 1.0, 1.0)
    )
    sr_pass = (
        sr8 - h_ref8 <= -args.promote_hybrid_8k and
        sr16 <= h_ref16 + args.long_tolerance and
        sr_speed >= args.promote_hybrid_speed
    )
    af_pass = (
        af8 - af_ref8 <= -args.promote_af_8k and
        af16 <= af_ref16 + args.long_tolerance and
        af_speed >= args.promote_af_speed
    )
    best_main = min(
        (SELECTIVE_WRITE, SELECTIVE_RESIDUAL, SELECTIVE_HIERARCHICAL),
        key=lambda n: get_eval(results[n], 8192)["bpb"],
    )
    tf8 = get_eval(results[TRANSFORMER], 8192)["bpb"]
    lines.extend([
        "",
        "PROMOTION VERDICT",
        (
            f"selective_residual: d8K={sr8-h_ref8:+.5f} d16K={sr16-h_ref16:+.5f} "
            f"speed8K={sr_speed:.3f}x -> {'PASS' if sr_pass else 'FAIL'}"
        ),
        (
            f"attentionfree_selective_residual: d8K={af8-af_ref8:+.5f} "
            f"d16K={af16-af_ref16:+.5f} speed8K={af_speed:.3f}x -> {'PASS' if af_pass else 'FAIL'}"
        ),
        f"best_structural_main={best_main} | best_main_BPB8K={get_eval(results[best_main],8192)['bpb']:.5f}",
        f"strong_transformer_reference_BPB8K={tf8:.5f}",
        (
            "VERDICT: READY FOR SELECTIVE-EPISODIC 300M RERUN"
            if sr_pass else
            "VERDICT: STRUCTURAL GAIN DID NOT SURVIVE THE CANONICAL 50M GATE"
        ),
        (
            "ATTENTION-FREE VERDICT: PROMOTE"
            if af_pass else
            "ATTENTION-FREE VERDICT: KEEP CURRENT BRANCH"
        ),
        "=" * width,
    ])
    return "\n".join(lines) + "\n"


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=("selftest", "train", "systems", "summary", "all"), default="all")
    p.add_argument("--outdir", default="/home/ubuntu/pcaf_runs/field_selective_episodic_canonical_50m_v10")
    p.add_argument("--canonical-source", default="")
    p.add_argument("--cache-dir", default="/home/ubuntu/field_lab/hf_cache")
    p.add_argument("--data-frac", type=float, default=0.10)
    p.add_argument("--data-device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--train-steps", type=int, default=0)
    p.add_argument("--train-seq", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--field-dim", type=int, default=704)
    p.add_argument("--field-layers", type=int, default=8)
    p.add_argument("--field-heads", type=int, default=8)
    p.add_argument("--hybrid-ff-hidden", type=int, default=1888)
    p.add_argument("--af-ff-hidden", type=int, default=1896)
    p.add_argument("--field-ff-hidden", type=int, default=1888, help="compatibility field used by v4 trainer")
    p.add_argument("--field-chunk", type=int, default=32)
    p.add_argument("--triton-block-c", type=int, default=16)
    p.add_argument("--triton-chunk-t", type=int, default=64)
    p.add_argument("--num-buckets", type=int, default=8192)
    p.add_argument("--local-chunk", type=int, default=2048)
    p.add_argument("--salience-floor", type=float, default=0.10)
    p.add_argument("--residual-limit", type=float, default=4.0)
    p.add_argument("--tf-dim", type=int, default=704)
    p.add_argument("--tf-heads", type=int, default=11)
    p.add_argument("--tf-layers", type=int, default=8)
    p.add_argument("--tf-ff-hidden", type=int, default=2048)
    p.add_argument("--amp", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--field-lr", type=float, default=5e-4)
    p.add_argument("--transformer-lr", type=float, default=6e-4)
    p.add_argument("--warmup", type=int, default=250)
    p.add_argument("--min-lr-ratio", type=float, default=0.10)
    p.add_argument("--weight-decay", type=float, default=0.10)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--conf-distill-ramp", type=int, default=200)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--quick-eval-windows", type=int, default=4)
    p.add_argument("--final-contexts", type=int, nargs="+", default=[4096, 8192, 16384])
    p.add_argument("--final-eval-windows", type=int, default=8, help="compatibility field used by v4 trainer")
    p.add_argument("--final-eval-token-budget", type=int, default=65536)
    p.add_argument("--system-contexts", type=int, nargs="+", default=[8192, 16384])
    p.add_argument("--system-tokens-per-step", type=int, default=16384)
    p.add_argument("--system-warmup", type=int, default=2)
    p.add_argument("--system-steps", type=int, default=5)
    p.add_argument("--system-lr", type=float, default=3e-4)
    p.add_argument("--model-seed", type=int, default=1234)
    p.add_argument("--data-seed", type=int, default=5678)
    p.add_argument("--eval-seed", type=int, default=9012)
    p.add_argument("--max-param-delta-pct", type=float, default=0.50)
    p.add_argument("--causal-tol", type=float, default=0.003)
    p.add_argument("--promote-hybrid-8k", type=float, default=0.010)
    p.add_argument("--promote-hybrid-speed", type=float, default=0.95)
    p.add_argument("--promote-af-8k", type=float, default=0.008)
    p.add_argument("--promote-af-speed", type=float, default=0.95)
    p.add_argument("--long-tolerance", type=float, default=0.001)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/H100 required")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.set_float32_matmul_precision("high")

    for path in (V3_PATH, BRIDGE_PATH, OPT_PATH, V9_PATH, JUDGE_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    canonical_path = locate_canonical(args.canonical_source)
    actual = sha256(canonical_path)
    if actual != EXPECTED_CANONICAL_SHA256:
        raise RuntimeError(
            f"canonical SHA mismatch expected={EXPECTED_CANONICAL_SHA256} actual={actual} path={canonical_path}"
        )

    v3 = import_module(V3_PATH, "v10_v3")
    bridge = import_module(BRIDGE_PATH, "v10_bridge")
    optmod = import_module(OPT_PATH, "v10_opt")
    epi = import_module(V9_PATH, "v10_epi")
    judge = import_module(JUDGE_PATH, "v10_judge")
    canonical = import_module(canonical_path, "v10_canonical")
    optmod.v3_global = v3
    install_fast_candidate_route(epi, optmod)

    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)
    shapes = resolve_shapes(args, v3, canonical, bridge, optmod, epi, judge, device)
    configure_bridge_runtime(bridge, args, v3, canonical, optmod, epi, judge, device)
    atomic_json(root / "config.json", {
        "args": vars(args),
        "canonical_source": str(canonical_path),
        "canonical_sha256": actual,
        "shapes": {k: asdict(v) for k, v in shapes.items()},
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    })

    log("=" * 180)
    log("CANONICAL 50M SELECTIVE EPISODIC MEMORY BRIDGE v10")
    log(f"gpu={torch.cuda.get_device_name(0)} torch={torch.__version__} cuda={torch.version.cuda}")
    log(f"canonical={canonical_path} sha={actual}")
    target = shapes[HYBRID_REF].params
    for name in MODEL_NAMES:
        s = shapes[name]
        d = 100.0 * (s.params - target) / target
        log(f"{name:44s} params={s.params:,} d={d:+.3f}% ff={s.ff_hidden} AF={s.attention_free}")
    log("=" * 180)

    if args.mode in ("selftest", "all"):
        run_selftest(args, shapes, device, v3, canonical, bridge, optmod, epi, judge)
        if args.mode == "selftest":
            return

    train = val = test = None
    if args.mode in ("train", "all"):
        train, val, test = v3.load_wikitext103_raw(args.cache_dir, args.data_frac)
        train = v3.place_data(train, device, args.data_device, "train")
        val = v3.place_data(val, device, args.data_device, "validation")
        test = v3.place_data(test, device, args.data_device, "test")

    results_path = root / "all_results.json"
    results: Dict[str, dict] = {}
    if args.mode in ("train", "all"):
        assert train is not None and val is not None and test is not None
        for name in MODEL_NAMES:
            lr = lr_for(name, args)
            result = bridge.full_train(
                name, lr, args, train, val, device, v3, canonical, shapes, root
            )
            result = add_test_evaluation(
                name, result, root, test, args, v3, canonical, bridge,
                optmod, epi, judge, device,
            )
            results[name] = result
            atomic_json(results_path, results)
        if args.mode == "train":
            return
    else:
        if not results_path.exists():
            raise FileNotFoundError(results_path)
        results = json.loads(results_path.read_text())

    systems_path = root / "systems.json"
    if args.mode in ("systems", "all"):
        systems = []
        for context in args.system_contexts:
            for name in MODEL_NAMES:
                row = bridge.system_benchmark(name, args, device, v3, canonical, shapes, context)
                systems.append(row)
                log(
                    f"[systems] {name:44s} ctx={context:5d} status={row['status']} "
                    f"B/s={row.get('bytes_per_second') or 0:,.0f}"
                )
        atomic_json(systems_path, systems)
        if args.mode == "systems":
            return
    else:
        if not systems_path.exists():
            raise FileNotFoundError(systems_path)
        systems = json.loads(systems_path.read_text())

    summary = make_summary(args, canonical_path, shapes, results, systems)
    atomic_text(root / "summary.txt", summary)
    log(summary)


if __name__ == "__main__":
    main()
