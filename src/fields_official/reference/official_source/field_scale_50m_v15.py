#!/usr/bin/env python3
"""Canonical 50M scale test for the distilled Field memory winners.

This compact runner compares only the validated production candidates:

  * selective_residual_opt
  * selective_span4_opt                (Pareto candidate)
  * surface_multiview_span4_opt        (quality candidate)
  * attention-free equivalents
  * a strong Flash-SDPA Transformer reference

The protocol is the same canonical 50M/Triton setup used in v13:
WikiText-103 10%, train context 8192, equal bytes/update, BF16, one epoch,
full-window test BPB at 4K/8K/16K, and equal no-checkpoint systems tests.

The visible bundle is deliberately small. Historical validated dependencies are
packed inside the executable .pyz and extracted automatically at runtime.
"""
from __future__ import annotations

import argparse
import gc
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "field_selective_episodic_canonical_50m_bridge_v10.py"
CLOUD_PATH = HERE / "field_delta_span_phase_ablation_v12.py"


def import_module(path: Path, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base = import_module(BASE_PATH, "field_selective_episodic_canonical_50m_v10_base")
cloud = import_module(CLOUD_PATH, "field_delta_span_phase_ablation_v12_for_v13")

SELECTIVE = "selective_residual_opt"
PARETO = "selective_span4_opt"
QUALITY = "surface_multiview_span4_opt"
AF_SELECTIVE = "attentionfree_selective_residual_opt"
AF_PARETO = "attentionfree_span4_opt"
AF_QUALITY = "attentionfree_surface_multiview_span4_opt"
TRANSFORMER = "transformer_flash_sdpa"
FIELD_NAMES = (SELECTIVE, PARETO, QUALITY, AF_SELECTIVE, AF_PARETO, AF_QUALITY)
MODEL_NAMES = (*FIELD_NAMES, TRANSFORMER)


def is_attention_free(name: str) -> bool:
    return name.startswith("attentionfree_")


def bridge_arm(name: str) -> str:
    return "attentionfree_multiscale" if is_attention_free(name) else "hybrid_w256_conf_parity"


def hidden_for(name: str, args) -> int:
    return args.af_ff_hidden if is_attention_free(name) else args.hybrid_ff_hidden


def _iter_cache_chain(cache) -> Iterable[object]:
    seen = set()
    while cache is not None and id(cache) not in seen:
        seen.add(id(cache))
        yield cache
        cache = getattr(cache, "base", None)


def _install_cloud_fast_route(optmod) -> None:
    """Install the exact int32 candidate route in the dependency copy used by v12."""
    v3c = cloud.v3
    if getattr(v3c, "_v13_fast_route_installed", False):
        return
    original = v3c.causal_recent_candidates

    class Proxy:
        causal_ngram_buckets = staticmethod(v3c.causal_ngram_buckets)
        causal_recent_candidates = staticmethod(original)

    def fast(tokens, order, num_buckets, top_k):
        return optmod.causal_recent_candidates_i32(tokens, order, num_buckets, top_k, Proxy)

    v3c.causal_recent_candidates = fast
    v3c._v13_fast_route_installed = True


def _wrap_cache(name: str, raw_cache, args):
    selective = cloud.make_v10_cache(raw_cache, args)
    if name in (SELECTIVE, AF_SELECTIVE):
        return selective
    if name in (PARETO, AF_PARETO):
        return cloud.CloudMechanismCache(
            selective,
            delta_rank=None,
            span_max=4,
            phase=False,
            args=args,
        )
    if name in (QUALITY, AF_QUALITY):
        surface = cloud.make_surface_multiview_cache(raw_cache, args)
        return cloud.CloudMechanismCache(
            surface,
            delta_rank=None,
            span_max=4,
            phase=False,
            args=args,
        )
    raise KeyError(name)


def build_field(name, args, v3, canonical, bridge, optmod, epi, device):
    del epi  # v12 owns the validated v9/v11 wrappers used here.
    base.seed_all(args.model_seed)
    hidden = hidden_for(name, args)
    bargs = base.make_bridge_args(args, hidden)
    model = bridge.build_field(bridge_arm(name), bargs, v3, canonical, hidden).to(device)

    # Exact v6 systems path before memory wrappers.
    optmod.replace_softpatch(model, v3)
    optmod.replace_cache(model, v3, i32=True)
    model.cache.FEATURE_DIM = int(v3.SuccessorCacheV5.FEATURE_DIM)
    if is_attention_free(name):
        optmod.replace_multiscale(model, v3, lite=False)
    else:
        optmod.replace_local(model, v3, "cached", args.local_chunk)

    _install_cloud_fast_route(optmod)
    model.cache = _wrap_cache(name, model.cache, args).to(device)
    return model


def build_model(name, args, v3, canonical, bridge, optmod, epi, judge, device):
    if name == TRANSFORMER:
        return base.build_transformer(args, judge, v3, device)
    return build_field(name, args, v3, canonical, bridge, optmod, epi, device)


def set_distill_scale(name: str, model, step: int, args) -> float:
    if name == TRANSFORMER:
        return 0.0
    value = min(1.0, step / max(float(args.conf_distill_ramp), 1.0))
    for cache in _iter_cache_chain(getattr(model, "cache", None)):
        if hasattr(cache, "distill_scale"):
            try:
                cache.distill_scale = value
            except Exception:
                pass
    return value


def group_ref(name: str) -> Optional[str]:
    if name in (SELECTIVE, AF_SELECTIVE, TRANSFORMER):
        return None
    return AF_SELECTIVE if is_attention_free(name) else SELECTIVE



def _scalar_from_chain(cache, attr: str, transform=None):
    for node in _iter_cache_chain(cache):
        value = getattr(node, attr, None)
        if value is not None:
            with torch.no_grad():
                x = value.float() if torch.is_tensor(value) else torch.tensor(float(value))
                return float(transform(x) if transform is not None else x)
    return None


def add_test_evaluation(name, result, root, test, args, v3, canonical,
                        bridge, optmod, epi, judge, device):
    model = base.load_trained_model(
        name, root, args, v3, canonical, bridge, optmod, epi, judge, device
    )
    rows = []
    for context in args.final_contexts:
        windows = base.windows_for_budget(context, args.final_eval_token_budget)
        row = bridge.evaluate(
            name, model, test, context, windows, args.eval_seed + 50_000,
            device, args.amp, v3,
        )
        rows.append(asdict(row))
        base.log(f"[{name}] TEST context={context} bpb={row.bpb:.5f}")

    result = dict(result)
    result["test_eval"] = rows
    cache = getattr(model, "cache", None)
    result["residual_gain_recent"] = _scalar_from_chain(
        cache, "residual_gain_recent",
        lambda x: args.residual_limit * torch.tanh(x),
    )
    result["asym_score_gain"] = _scalar_from_chain(
        cache, "asym_score_raw",
        lambda x: args.score_limit * torch.tanh(x),
    )
    span = getattr(cache, "span", None)
    if span is not None:
        result["span_conf_scale"] = float(span.conf_scale.detach().float())
        result["span_length_weights"] = torch.softmax(
            span.length_logits.detach().float(), dim=0
        ).cpu().tolist()
    base.atomic_json(root / "models" / name / "result.json", result)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_selftest(args, shapes, device, v3, canonical, bridge, optmod, epi, judge):
    base.log("[selftest] canonical Triton + v10/v11/v12 consolidation causality")
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
    base.log("[selftest] canonical Field reference/Triton PASS")
    base.exact_candidate_selftest(v3, optmod, device)

    # Same backbone and exact baseline identity at zero-init overlays.
    models = {
        name: build_model(name, args, v3, canonical, bridge, optmod, epi, judge, device)
        for name in (SELECTIVE, PARETO, QUALITY, AF_SELECTIVE, AF_PARETO, AF_QUALITY)
    }
    for names in ((SELECTIVE, PARETO, QUALITY), (AF_SELECTIVE, AF_PARETO, AF_QUALITY)):
        ref = models[names[0]]
        for name in names[1:]:
            max_abs = 0.0
            for p, q in zip(ref.blocks.parameters(), models[name].blocks.parameters()):
                max_abs = max(max_abs, float((p - q).abs().max()))
            base.log(f"[selftest] paired backbone {names[0]} vs {name} max_abs={max_abs:.3e}")
            if max_abs != 0.0:
                raise AssertionError("paired backbone mismatch")

    x = torch.randint(0, base.VOCAB, (1, 97), device=device)
    y = torch.randint(0, base.VOCAB, (1, 97), device=device)
    for name in MODEL_NAMES:
        model = build_model(name, args, v3, canonical, bridge, optmod, epi, judge, device).train()
        set_distill_scale(name, model, args.conf_distill_ramp, args)
        with bridge.amp_ctx(device, args.amp):
            loss, primary = bridge.loss_for(name, model, x, y)
        loss.backward()
        finite = bool(torch.isfinite(loss)) and all(
            p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()
        )
        base.log(
            f"[selftest] {name:<46} params={base.nparams(model):,} "
            f"loss={float(loss):.5f} primary={float(primary):.5f} finite={finite}"
        )
        if not finite:
            raise AssertionError(name)
        del model

    # Token-NLL causality for the complete memory path.
    prefix = 64
    a = torch.randint(0, base.VOCAB, (1, 96), device=device)
    b = a.clone()
    b[:, prefix:] = torch.randint(0, base.VOCAB, b[:, prefix:].shape, device=device)
    ta = torch.randint(0, base.VOCAB, (1, 96), device=device)
    tb = ta.clone()
    tb[:, prefix:] = torch.randint(0, base.VOCAB, tb[:, prefix:].shape, device=device)
    for name in (PARETO, QUALITY, AF_PARETO, AF_QUALITY):
        model = build_model(name, args, v3, canonical, bridge, optmod, epi, judge, device).eval()
        set_distill_scale(name, model, args.conf_distill_ramp, args)
        with torch.no_grad(), bridge.amp_ctx(device, args.amp):
            sa, la = model.states_logits(a)
            sb, lb = model.states_logits(b)
            na = model.cache.token_nll(sa, la, a, ta)
            nb = model.cache.token_nll(sb, lb, b, tb)
        err = float((na[:, :prefix] - nb[:, :prefix]).abs().max())
        base.log(f"[selftest] {name} causal_token_nll max_abs={err:.3e}")
        if err > args.causal_tol:
            raise AssertionError(f"causal failure {name}: {err}")
        del model
    base.log("[selftest] PASS")
    del models, x, y, a, b, ta, tb
    gc.collect()
    torch.cuda.empty_cache()


def _get_eval(results, name: str, context: int):
    return base.get_eval(results[name], context)


def make_summary(args, canonical_path: Path, shapes, results, systems) -> str:
    width = 220
    lines = [
        "=" * width,
        "CANONICAL 50M SCALE v15 — PARETO / QUALITY / ATTENTION-FREE",
        "=" * width,
        f"canonical_source={canonical_path} sha256={base.sha256(canonical_path)}",
        (
            f"protocol: WikiText-103 {args.data_frac:.1%} | train ctx={args.train_seq} | "
            f"bytes/update={args.batch_size*args.accum*args.train_seq:,} | epochs={args.epochs} | "
            "BF16 | no activation checkpointing"
        ),
        "Primary score: full-window TEST BPB@8K; final selection uses quality and equal-systems speed together.",
        "",
        "MODEL SHAPES",
        f"{'model':48s} {'params':>13s} {'d%':>8s} {'dim':>6s} {'layers':>7s} {'ff':>7s} {'AF':>4s}",
    ]
    target = shapes[SELECTIVE].params
    for name in MODEL_NAMES:
        s = shapes[name]
        delta = 100.0 * (s.params - target) / target
        lines.append(
            f"{name:48s} {s.params:13,d} {delta:+8.3f} {s.dim:6d} {s.layers:7d} "
            f"{s.ff_hidden:7d} {str(s.attention_free):>4s}"
        )

    lines.extend([
        "",
        "FINAL QUALITY — TEST SPLIT",
        (
            f"{'model':48s} {'BPB4K':>9s} {'BPB8K':>9s} {'dLane':>9s} {'BPB16K':>9s} "
            f"{'oracle8':>9s} {'cap8':>7s} {'sep8':>7s} {'gRes':>7s} {'gView':>7s} "
            f"{'train B/s':>12s} {'peak':>7s}"
        ),
    ])
    for name in MODEL_NAMES:
        r = results[name]
        e4, e8, e16 = (_get_eval(results, name, c) for c in (4096, 8192, 16384))
        ref = group_ref(name)
        dlane = 0.0 if ref is None else e8["bpb"] - _get_eval(results, ref, 8192)["bpb"]
        oracle = "-" if e8.get("oracle_bpb") is None else f"{e8['oracle_bpb']:.5f}"
        cap = "-" if e8.get("capture") is None else f"{e8['capture']:.3f}"
        sep = "-" if e8.get("gate_sep") is None else f"{e8['gate_sep']:+.3f}"
        gres = "-" if r.get("residual_gain_recent") is None else f"{r['residual_gain_recent']:+.3f}"
        gview = "-" if r.get("asym_score_gain") is None else f"{r['asym_score_gain']:+.3f}"
        lines.append(
            f"{name:48s} {e4['bpb']:9.5f} {e8['bpb']:9.5f} {dlane:+9.5f} "
            f"{e16['bpb']:9.5f} {oracle:>9s} {cap:>7s} {sep:>7s} {gres:>7s} "
            f"{gview:>7s} {r['train_bytes_per_second']:12,.0f} {r['train_peak_gib']:7.2f}"
        )

    lines.extend([
        "",
        "EQUAL NO-CHECKPOINT SYSTEMS BENCHMARK",
        f"{'model':48s} {'ctx':>7s} {'batch':>6s} {'status':>8s} {'B/s':>12s} {'step ms':>10s} {'peak GB':>8s}",
    ])
    for row in systems:
        bps = "-" if row.get("bytes_per_second") is None else f"{row['bytes_per_second']:,.0f}"
        ms = "-" if row.get("step_ms") is None else f"{row['step_ms']:.2f}"
        peak = "-" if row.get("peak_gib") is None else f"{row['peak_gib']:.2f}"
        lines.append(
            f"{row['model']:48s} {row['context']:7d} {row['batch']:6d} {row['status']:>8s} "
            f"{bps:>12s} {ms:>10s} {peak:>8s}"
        )

    sys_map = {(r["model"], int(r["context"])): r for r in systems}
    def bpb(name, ctx): return _get_eval(results, name, ctx)["bpb"]
    def speed(name, ref):
        return (sys_map[(name, 8192)].get("bytes_per_second") or 0.0) / max(
            sys_map[(ref, 8192)].get("bytes_per_second") or 1.0, 1.0
        )

    h_pareto_gain = bpb(SELECTIVE,8192) - bpb(PARETO,8192)
    h_quality_gain = bpb(SELECTIVE,8192) - bpb(QUALITY,8192)
    h_pareto_long = bpb(SELECTIVE,16384) - bpb(PARETO,16384)
    h_quality_long = bpb(SELECTIVE,16384) - bpb(QUALITY,16384)
    h_pareto_speed = speed(PARETO, SELECTIVE)
    h_quality_speed = speed(QUALITY, SELECTIVE)

    af_pareto_gain = bpb(AF_SELECTIVE,8192) - bpb(AF_PARETO,8192)
    af_quality_gain = bpb(AF_SELECTIVE,8192) - bpb(AF_QUALITY,8192)
    af_pareto_long = bpb(AF_SELECTIVE,16384) - bpb(AF_PARETO,16384)
    af_quality_long = bpb(AF_SELECTIVE,16384) - bpb(AF_QUALITY,16384)
    af_pareto_speed = speed(AF_PARETO, AF_SELECTIVE)
    af_quality_speed = speed(AF_QUALITY, AF_SELECTIVE)

    h_pareto_pass = (
        h_pareto_gain >= args.promote_pareto_8k and
        h_pareto_long >= -args.long_tolerance and
        h_pareto_speed >= args.promote_pareto_speed
    )
    h_quality_pass = (
        h_quality_gain >= args.promote_quality_8k and
        h_quality_long >= -args.long_tolerance and
        h_quality_speed >= args.promote_quality_speed
    )
    af_pareto_pass = (
        af_pareto_gain >= args.promote_af_pareto_8k and
        af_pareto_long >= -args.long_tolerance and
        af_pareto_speed >= args.promote_af_pareto_speed
    )
    af_quality_pass = (
        af_quality_gain >= args.promote_af_quality_8k and
        af_quality_long >= -args.long_tolerance and
        af_quality_speed >= args.promote_af_quality_speed
    )

    tf8 = bpb(TRANSFORMER, 8192)
    field_candidates = [
        (PARETO, bpb(PARETO,8192), h_pareto_gain, h_pareto_speed, h_pareto_pass),
        (QUALITY, bpb(QUALITY,8192), h_quality_gain, h_quality_speed, h_quality_pass),
    ]
    qualified = [x for x in field_candidates if x[4]]
    best_qualified = min(qualified, key=lambda x: x[1])[0] if qualified else "none"

    lines.extend([
        "",
        "SCALE VERDICT",
        (
            f"hybrid Pareto: gain8K={h_pareto_gain:+.5f} | gain16K={h_pareto_long:+.5f} | "
            f"speed8K={h_pareto_speed:.3f}x | pass={h_pareto_pass}"
        ),
        (
            f"hybrid Quality: gain8K={h_quality_gain:+.5f} | gain16K={h_quality_long:+.5f} | "
            f"speed8K={h_quality_speed:.3f}x | pass={h_quality_pass}"
        ),
        (
            f"attention-free Pareto: gain8K={af_pareto_gain:+.5f} | gain16K={af_pareto_long:+.5f} | "
            f"speed8K={af_pareto_speed:.3f}x | pass={af_pareto_pass}"
        ),
        (
            f"attention-free Quality: gain8K={af_quality_gain:+.5f} | gain16K={af_quality_long:+.5f} | "
            f"speed8K={af_quality_speed:.3f}x | pass={af_quality_pass}"
        ),
        (
            f"BPB8K: selective={bpb(SELECTIVE,8192):.5f} | pareto={bpb(PARETO,8192):.5f} | "
            f"quality={bpb(QUALITY,8192):.5f} | transformer_ref={tf8:.5f}"
        ),
        f"BEST QUALIFIED HYBRID={best_qualified}",
        (
            "VERDICT: FREEZE 50M PARETO AND QUALITY PROFILES FOR THE NEXT SCALE TEST"
            if (h_pareto_pass or h_quality_pass) else
            "VERDICT: RETAIN SELECTIVE RESIDUAL BASELINE; NO NEW 50M PROFILE CLEARED ITS GATE"
        ),
        "=" * width,
    ])
    return "\n".join(lines) + "\n"


_original_parse_args = base.parse_args


def parse_args():
    args = _original_parse_args()
    # v11/v12 structural settings; intentionally fixed unless this source is edited.
    defaults: Dict[str, object] = {
        "address_dim": 24,
        "latent_top_k": 4,
        "score_limit": 2.0,
        "span_top_k": 4,
        "sidecar_max_mix": 0.40,
        "sidecar_gate_bias": 1.0e-6,
        "sidecar_aux_weight": 0.01,
        "gate_grad_scale": 0.01,
        "delta_heads": 4,
        "delta_block": 16,
        "phase_bands": 8,
        "phase_rank": 16,
        "promote_pareto_8k": 0.008,
        "promote_pareto_speed": 0.94,
        "promote_quality_8k": 0.015,
        "promote_quality_speed": 0.90,
        "promote_af_pareto_8k": 0.008,
        "promote_af_pareto_speed": 0.94,
        "promote_af_quality_8k": 0.015,
        "promote_af_quality_speed": 0.88,
    }
    for key, value in defaults.items():
        setattr(args, key, value)
    return args


# Patch the proven canonical trainer dynamically.
base.HYBRID_REF = SELECTIVE
base.AF_REF = AF_SELECTIVE
base.TRANSFORMER = TRANSFORMER
base.FIELD_NAMES = FIELD_NAMES
base.MODEL_NAMES = MODEL_NAMES
base.is_attention_free = is_attention_free
base.bridge_arm = bridge_arm
base.hidden_for = hidden_for
base.build_field = build_field
base.build_model = build_model
base.set_distill_scale = set_distill_scale
base.group_ref = group_ref
base.add_test_evaluation = add_test_evaluation
base.run_selftest = run_selftest
base.make_summary = make_summary
base.parse_args = parse_args

_original_log = base.log

def _v15_log(msg: object = ""):
    text = str(msg)
    if text == "CANONICAL 50M SELECTIVE EPISODIC MEMORY BRIDGE v10":
        text = "CANONICAL 50M SCALE v15 — PARETO / QUALITY / ATTENTION-FREE"
    _original_log(text)
base.log = _v15_log


if __name__ == "__main__":
    base.main()
