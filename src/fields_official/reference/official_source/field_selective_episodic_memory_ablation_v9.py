#!/usr/bin/env python3
"""Selective episodic corrective-memory ablation for Field/PCAF.

This v9 arena tests a structural change rather than another router-only loss:
past matching contexts store not only their observed successor, but also the
parametric model's error at that past event. Retrieval can then add a causal
residual correction on top of the validated confidence-PCAF mixture.

The second structural axis is selective writing. Past events that were hard for
the parametric model and/or changed the hidden state more strongly receive more
memory weight. A hierarchical arm keeps recent and older corrective traces in
separate banks with independently learned residual gains.

All experimental residual gains are initialized to zero, so the non-selective
residual arms begin exactly at the current confidence-PCAF reference. The
attention-free lane uses the same memory experiments on the validated
softpatch + multiscale causal-convolution backbone and never introduces a
query-key attention path.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
V8_PATH = HERE / "field_oracle_gap_ablation_v8.py"


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


arena = import_module(V8_PATH, "field_oracle_gap_ablation_v8_dep")
v3 = arena.v3
VOCAB = arena.VOCAB
LN2 = arena.LN2

HYBRID_REF = "hybrid_ref"
AF_REF = "attentionfree_ref"
HYBRID_ARMS = (
    HYBRID_REF,
    "selective_write",
    "episodic_residual",
    "selective_residual",
    "hierarchical_residual",
    "selective_hierarchical",
)
AF_ARMS = (
    AF_REF,
    "attentionfree_residual",
    "attentionfree_selective_residual",
)
ALL_ARMS = (*HYBRID_ARMS, *AF_ARMS)


class EpisodicCorrectiveCache(nn.Module):
    """Confidence PCAF plus selective write and retrieved error correction.

    For a past candidate j, the episodic error vector is

        e_j = one_hot(y_j) - p_theta(. | context_j)

    where y_j is already observed by every later query. Retrieval averages e_j
    over causally valid matching contexts and adds it as a zero-initialized
    residual in log-probability space. This preserves the validated PCAF output
    at initialization while allowing memory to correct systematic parametric
    errors in similar contexts.
    """

    def __init__(
        self,
        base: nn.Module,
        *,
        selective: bool,
        residual: bool,
        hierarchical: bool,
        salience_floor: float = 0.10,
        residual_limit: float = 4.0,
    ):
        super().__init__()
        self.base = base
        self.state_dim = int(base.state_dim)
        self.memory_dim = int(base.memory_dim)
        self.num_buckets = int(base.num_buckets)
        self.order = int(base.order)
        self.top_k = 8 if hierarchical else 4
        self.router_mode = str(base.router_mode)
        self.selective = bool(selective)
        self.residual = bool(residual)
        self.hierarchical = bool(hierarchical)
        self.salience_floor = float(salience_floor)
        self.residual_limit = float(residual_limit)
        self.enabled = True
        self.distill_scale = 1.0
        self.last_aux: Dict[str, float] = {}

        # Zero initialization gives exact reference behavior for non-selective
        # residual arms at step zero. The main loss learns the sign and size.
        self.residual_gain_recent = nn.Parameter(torch.zeros(()))
        if hierarchical:
            self.residual_gain_long = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("residual_gain_long", None)

        # Selective-write weighting is mostly structural but the strength can
        # adapt. Softplus(0)=0.693 gives a meaningful yet conservative start.
        if selective:
            self.salience_strength_raw = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("salience_strength_raw", None)

    def _features(self, *args, **kwargs):
        return self.base._features(*args, **kwargs)

    @property
    def distill_temperature(self) -> float:
        return float(self.base.distill_temperature)

    @property
    def distill_weight(self) -> float:
        return float(self.base.distill_weight)

    def _bank_distribution(
        self,
        weights: torch.Tensor,
        valid: torch.Tensor,
        cand_tokens: torch.Tensor,
        past_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return successor distribution, past-belief distribution, coverage."""
        w = weights * valid.float()
        denom = w.sum(-1, keepdim=True)
        has = denom.squeeze(-1) > 1e-8
        w = w / denom.clamp_min(1e-8)
        b, t, _ = w.shape
        successor = torch.zeros((b, t, VOCAB), device=w.device, dtype=torch.float32)
        successor.scatter_add_(-1, cand_tokens.long(), w.float())
        past = (w[..., None].float() * past_probs.float()).sum(-2)
        return successor, past, has

    def _compute(
        self,
        states: torch.Tensor,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        targets: torch.Tensor,
        compute_metrics: bool,
    ):
        flat_logits = logits.reshape(-1, VOCAB).float()
        flat_targets = targets.reshape(-1)
        param_log_probs = F.log_softmax(flat_logits, dim=-1)
        param_probs_flat = param_log_probs.exp()
        param_nll = -param_log_probs.gather(1, flat_targets[:, None]).squeeze(1)
        param_target = param_nll.neg().exp()

        if not self.enabled:
            primary = param_nll.mean()
            stats = (
                v3.CacheStats(0, 0, 0, 0, float(primary / LN2), float(primary / LN2), 0, 0, 0, 0, 0)
                if compute_metrics else None
            )
            return primary, primary.detach(), stats, param_nll.view_as(targets)

        b, t, _ = states.shape
        param_probs = param_probs_flat.view(b, t, VOCAB)
        idx = v3.causal_recent_candidates(tokens, self.order, self.num_buckets, self.top_k)
        valid = idx >= 0
        safe = idx.clamp_min(0)
        has_any = valid.any(-1)
        batch_idx = torch.arange(b, device=states.device)[:, None, None]

        proj = v3.normalize_rows(F.linear(states.float(), self.base.shared_weight.float()))
        q = proj[:, :, None, :]
        ck = proj[batch_idx, safe]
        scores = (ck * q).sum(-1) * (self.memory_dim ** -0.5)
        recency = safe.float() / max(float(t - 1), 1.0)
        scores = scores + self.base.recency_scale.float() * recency

        cand_tokens = targets[batch_idx, safe]
        detached_probs = param_probs.detach()
        past_probs = detached_probs[batch_idx, safe]
        past_true_prob = past_probs.gather(-1, cand_tokens[..., None]).squeeze(-1)

        write_strength = scores.new_ones(scores.shape)
        novelty = scores.new_zeros(scores.shape)
        if self.selective:
            # Surprise is high where the past parametric model underpredicted the
            # observed successor. Hidden novelty is a causal proxy for a state
            # disturbance; both are known after the past event is observed.
            error = torch.sqrt((1.0 - past_true_prob).clamp(0.0, 1.0))
            detached_states = v3.normalize_rows(states.detach().float())
            prev_safe = (safe - 1).clamp_min(0)
            s_now = detached_states[batch_idx, safe]
            s_prev = detached_states[batch_idx, prev_safe]
            novelty = (0.5 * (1.0 - (s_now * s_prev).sum(-1))).clamp(0.0, 1.0)
            salience = 0.75 * error + 0.25 * novelty
            write_strength = self.salience_floor + (1.0 - self.salience_floor) * salience
            alpha = F.softplus(self.salience_strength_raw.float()).clamp(0.0, 4.0)
            scores = scores + alpha * torch.log(write_strength.clamp_min(1e-4))

        scores = scores.masked_fill(~valid, -1.0e9)
        weights = torch.softmax(scores.float(), dim=-1) * valid.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)

        # The validated PCAF mixture uses the four most recent matches. The
        # hierarchical arm reserves older matches for a separate correction
        # bank without silently changing the reference mixture itself.
        base_k = min(4, self.top_k)
        base_weights = weights[..., :base_k]
        base_valid = valid[..., :base_k]
        base_tokens = cand_tokens[..., :base_k]
        base_recency = recency[..., :base_k]
        base_scores = scores[..., :base_k]
        base_denom = base_weights.sum(-1, keepdim=True)
        base_weights = base_weights / base_denom.clamp_min(1e-8)
        base_has = base_valid.any(-1)

        cache_dist = torch.zeros((b, t, VOCAB), device=states.device, dtype=torch.float32)
        cache_dist.scatter_add_(-1, base_tokens.long(), base_weights.float())
        target_cache = cache_dist.gather(-1, targets[..., None]).squeeze(-1)

        active = base_has.reshape(-1)
        state_logit = self.base.state_gate(states).float().squeeze(-1)
        if bool(active.any()):
            features = self._features(
                base_scores.reshape(-1, base_k)[active],
                base_weights.reshape(-1, base_k)[active],
                base_valid.reshape(-1, base_k)[active],
                base_tokens.reshape(-1, base_k)[active],
                base_recency.reshape(-1, base_k)[active],
                flat_logits[active],
            )
            route = self.base.router(features)[:, 0]
        else:
            features = states.new_zeros((0, self.base.FEATURE_DIM), dtype=torch.float32)
            route = states.new_zeros((0,), dtype=torch.float32)

        flat_state_logit = state_logit.reshape(-1)
        gate_flat = torch.zeros_like(flat_state_logit)
        gate_logit_active = states.new_zeros((0,), dtype=torch.float32)
        if bool(active.any()):
            if self.base.router_mode == "v5":
                gate_logit_active = flat_state_logit[active] + route
            else:
                cache_conf = features[:, 5].clamp(1e-4, 1.0 - 1e-4)
                param_conf = features[:, 8].clamp(1e-4, 1.0 - 1e-4)
                evidence = torch.logit(cache_conf) - torch.logit(param_conf)
                evidence = evidence + 1.25 * features[:, 6] - 0.50 * features[:, 7]
                evidence = evidence + 0.35 * features[:, 10] + 0.25 * features[:, 3]
                state_term = 0.0 if self.base.router_mode == "confidence_nostate" else flat_state_logit[active]
                gate_logit_active = (
                    state_term + route + self.base.evidence_gain * evidence + self.base.evidence_bias
                )
            gate_flat[active] = torch.sigmoid(gate_logit_active).clamp(1e-5, 1.0 - 1e-5)
        gate = gate_flat.view(b, t)

        base_probs = (1.0 - gate[..., None]) * param_probs + gate[..., None] * cache_dist
        base_probs = base_probs / base_probs.sum(-1, keepdim=True).clamp_min(1e-8)
        base_target = base_probs.gather(-1, targets[..., None]).squeeze(-1)

        correction = torch.zeros_like(base_probs)
        residual_l1 = states.new_zeros((), dtype=torch.float32)
        gain_recent = self.residual_limit * torch.tanh(self.residual_gain_recent.float())
        gain_long = states.new_zeros((), dtype=torch.float32)

        if self.residual:
            if self.hierarchical:
                recent_slice = slice(0, min(2, self.top_k))
                long_slice = slice(min(2, self.top_k), self.top_k)
                q_recent, p_recent, has_recent = self._bank_distribution(
                    weights[..., recent_slice], valid[..., recent_slice],
                    cand_tokens[..., recent_slice], past_probs[..., recent_slice, :],
                )
                residual_recent = q_recent - p_recent
                correction = correction + (
                    gate * has_recent.float() * gain_recent
                )[..., None] * residual_recent

                if self.top_k > 2:
                    q_long, p_long, has_long = self._bank_distribution(
                        weights[..., long_slice], valid[..., long_slice],
                        cand_tokens[..., long_slice], past_probs[..., long_slice, :],
                    )
                    residual_long = q_long - p_long
                    gain_long = self.residual_limit * torch.tanh(self.residual_gain_long.float())
                    # Older evidence is deliberately conservative and is useful
                    # only where such a bank actually exists.
                    correction = correction + (
                        gate * has_long.float() * gain_long
                    )[..., None] * residual_long
                    residual_l1 = 0.5 * (
                        residual_recent.abs().mean() + residual_long.abs().mean()
                    )
                else:
                    residual_l1 = residual_recent.abs().mean()
            else:
                q_mem, p_mem, has_mem = self._bank_distribution(
                    weights, valid, cand_tokens, past_probs
                )
                residual = q_mem - p_mem
                correction = (
                    gate * has_mem.float() * gain_recent
                )[..., None] * residual
                residual_l1 = residual.abs().mean()

        final_log_probs = F.log_softmax(torch.log(base_probs.clamp_min(1e-8)) + correction, dim=-1)
        token_nll = -final_log_probs.gather(-1, targets[..., None]).squeeze(-1)
        primary = token_nll.mean()
        loss = primary

        if self.training and bool(active.any()) and self.distill_scale > 0:
            pa = param_target[active]
            ca = target_cache.reshape(-1)[active]
            log_adv = torch.log(ca.detach().clamp_min(1e-8)) - torch.log(pa.detach().clamp_min(1e-8))
            teacher = torch.sigmoid(log_adv / self.distill_temperature)
            weight = torch.tanh(log_adv.abs())
            aux = (
                F.binary_cross_entropy_with_logits(gate_logit_active, teacher, reduction="none") * weight
            ).sum() / weight.sum().clamp_min(1.0)
            loss = primary + self.distill_weight * float(self.distill_scale) * aux
            final_target_train = token_nll.detach().neg().exp().reshape(-1)
            self.last_aux = {
                "distill": float(aux.detach()),
                "teacher": float(teacher.mean()),
                "cache_win": float((log_adv > 0).float().mean()),
                "write": float(write_strength[valid].mean()) if bool(valid.any()) else 0.0,
                "gain_recent": float(gain_recent.detach()),
                "gain_long": float(gain_long.detach()),
                "residual_l1": float(residual_l1.detach()),
                "corr_win": float((final_target_train[active] > base_target.reshape(-1)[active]).float().mean()),
            }

        stats = None
        if compute_metrics:
            with torch.no_grad():
                param_loss = param_nll.mean()
                final_target = token_nll.neg().exp().reshape(-1)
                target_cache_flat = target_cache.reshape(-1)
                # Keep the original confidence-PCAF oracle definition so the
                # reported headroom is directly comparable to v8. A corrective
                # memory may legitimately exceed 1.0 capture if it beats the
                # old per-token max(parametric, successor-cache) oracle.
                oracle_target = torch.maximum(param_target, target_cache_flat)
                oracle_target = torch.where(active, oracle_target, param_target)
                oracle_loss = -torch.log(oracle_target.clamp_min(1e-8)).mean()
                denom_t = param_loss - oracle_loss
                capture = (
                    float((param_loss - primary) / denom_t)
                    if float(denom_t) > 1e-8 else 0.0
                )
                cache_win = active & (target_cache_flat > param_target)
                cache_lose = active & ~cache_win
                gate_win = float(gate_flat[cache_win].mean()) if bool(cache_win.any()) else 0.0
                gate_lose = float(gate_flat[cache_lose].mean()) if bool(cache_lose.any()) else 0.0
                stats = v3.CacheStats(
                    coverage=float(has_any.float().mean()),
                    gate=float(gate_flat[active].mean()) if bool(active.any()) else 0.0,
                    hit=float((target_cache_flat > 0).float().mean()),
                    cache_prob=float(target_cache.mean()),
                    param_bpb=float(param_loss / LN2),
                    oracle_bpb=float(oracle_loss / LN2),
                    capture=capture,
                    cache_win_rate=float(cache_win.float().sum() / active.float().sum().clamp_min(1.0)),
                    gate_when_cache_wins=gate_win,
                    gate_when_cache_loses=gate_lose,
                    gate_separation=gate_win - gate_lose,
                )
        return loss, primary.detach(), stats, token_nll

    def forward(self, states, logits, tokens, targets, compute_metrics=False):
        loss, primary, stats, _ = self._compute(states, logits, tokens, targets, compute_metrics)
        return loss, primary, stats

    def token_nll(self, states, logits, tokens, targets):
        return self._compute(states, logits, tokens, targets, False)[3]


def base_arm_for(arm: str) -> str:
    return "softpatch_multiscale_conf" if arm in AF_ARMS else "softpatch_local_w256_conf"


def attention_free(arm: str) -> bool:
    return arm in AF_ARMS


def cache_mode(arm: str) -> Tuple[bool, bool, bool]:
    selective = arm in {
        "selective_write", "selective_residual", "selective_hierarchical",
        "attentionfree_selective_residual",
    }
    residual = arm in {
        "episodic_residual", "selective_residual", "hierarchical_residual",
        "selective_hierarchical", "attentionfree_residual",
        "attentionfree_selective_residual",
    }
    hierarchical = arm in {"hierarchical_residual", "selective_hierarchical"}
    return selective, residual, hierarchical


def modify_model(model: nn.Module, arm: str, args) -> nn.Module:
    if arm in {HYBRID_REF, AF_REF}:
        return model
    selective, residual, hierarchical = cache_mode(arm)
    model.cache = EpisodicCorrectiveCache(
        model.cache,
        selective=selective,
        residual=residual,
        hierarchical=hierarchical,
        salience_floor=args.salience_floor,
        residual_limit=args.residual_limit,
    )
    return model


def raw_model(arm: str, hidden: int, args, device: torch.device) -> nn.Module:
    model = v3.FieldPCAFLM(
        base_arm_for(arm), args.dim, args.layers, args.heads, hidden,
        args.field_chunk, args.num_buckets,
    ).to(device)
    return modify_model(model, arm, args).to(device)


def group_for(arm: str) -> str:
    return "attention_free" if arm in AF_ARMS else "hybrid"


def selftest(args, device, shapes):
    arena.log("[selftest] finite backward, exact zero-residual identity, and full-cache causality")
    x = torch.randint(0, VOCAB, (1, 17), device=device)
    y = torch.randint(0, VOCAB, (1, 17), device=device)
    for arm in ALL_ARMS:
        m = arena.build_model(arm, 0, shapes[arm], args, device)
        with arena.amp_ctx(device, args.amp):
            loss, primary, _ = m.loss_and_stats(x, y, False)
        loss.backward()
        finite = bool(torch.isfinite(loss)) and all(
            p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters()
        )
        arena.log(
            f"[selftest] {arm:<34} params={arena.nparams(m):,} "
            f"d={shapes[arm]['delta_pct']:+.3f}% loss={float(loss.detach()):.5f} "
            f"primary={float(primary):.5f} finite={finite}"
        )
        if not finite:
            raise AssertionError(arm)
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # A zero-gain non-selective residual overlay must exactly reproduce the
    # original confidence-PCAF mixture for the same initialized model.
    arena.seed_all(773)
    ref = v3.FieldPCAFLM("softpatch_local_w256_conf", 32, 1, 4, 64, 8, 64).to(device).eval()
    arena.seed_all(773)
    epi = v3.FieldPCAFLM("softpatch_local_w256_conf", 32, 1, 4, 64, 8, 64).to(device).eval()
    epi.cache = EpisodicCorrectiveCache(
        epi.cache, selective=False, residual=True, hierarchical=False,
        salience_floor=args.salience_floor, residual_limit=args.residual_limit,
    ).to(device)
    with torch.no_grad(), arena.amp_ctx(device, args.amp):
        l0 = ref.loss_and_stats(x, y, False)[1]
        l1 = epi.loss_and_stats(x, y, False)[1]
    err = float((l0 - l1).abs())
    arena.log(f"[selftest] zero-residual reference identity abs_loss={err:.3e}")
    if err > 3e-5:
        raise AssertionError(("zero residual identity", err))

    # Full memory path causality: alter only the suffix of a same-shape stream
    # and compare token losses whose inputs and targets are unchanged.
    for arm in ("selective_residual", "selective_hierarchical", "attentionfree_selective_residual"):
        m = arena.build_model(arm, 11, shapes[arm], args, device).eval()
        full = torch.randint(0, VOCAB, (1, 19), device=device)
        full2 = full.clone()
        prefix_targets = 8
        full2[:, prefix_targets + 1:] = torch.randint(
            0, VOCAB, full2[:, prefix_targets + 1:].shape, device=device
        )
        xa, ya = full[:, :-1], full[:, 1:]
        xb, yb = full2[:, :-1], full2[:, 1:]
        with torch.no_grad(), arena.amp_ctx(device, args.amp):
            sa, la = m.states_logits(xa)
            sb, lb = m.states_logits(xb)
            na = m.cache.token_nll(sa, la, xa, ya)[:, :prefix_targets]
            nb = m.cache.token_nll(sb, lb, xb, yb)[:, :prefix_targets]
        cerr = float((na - nb).abs().max())
        arena.log(f"[selftest] {arm:<34} causal_token_nll max_abs={cerr:.3e}")
        if cerr > 4e-4:
            raise AssertionError((arm, cerr))
        del m
    arena.log("[selftest] PASS")


def make_summary(stage_a, stage_b, stage_c, shapes, selected_main, selected_af, args):
    lines = [
        "FIELD SELECTIVE EPISODIC MEMORY ABLATION v9 — STRUCTURAL CORRECTIVE-MEMORY TEST",
        "=" * 250,
        f"Protocol: WikiText-103 {args.data_frac:.1%} | train ctx={args.seq_len} | "
        f"eval={args.long_context}/{args.very_long_context} | bytes/update={args.batch_size*args.accum*args.seq_len:,}",
        "Reference: optimized softpatch + local-w256 + confidence PCAF. AF lane stays query-key-attention-free.",
        "New memory stores/retrieves past parametric error vectors; selective arms prioritize surprising/novel past events.",
        "Portable exact Field math; a winner must pass a canonical 50M bridge before any 300M rerun.",
        "",
        "PARAMETER MATCHING",
        f"{'arm':<38} {'group':>14} {'params':>13} {'d%':>8} {'ff':>6} {'AF':>4}",
    ]
    for arm in ALL_ARMS:
        s = shapes[arm]
        lines.append(
            f"{arm:<38} {group_for(arm):>14} {s['params']:>13,d} {s['delta_pct']:>+8.3f} "
            f"{s['hidden']:>6d} {('yes' if s['attention_free'] else 'no'):>4}"
        )

    def table(title, rows):
        lines.extend([
            "", title,
            f"{'arm':<38} {'seed':>6} {'BPB2K':>9} {'BPB8K':>9} {'dGroup':>9} {'BPB16K':>9} "
            f"{'oracle':>9} {'cap':>7} {'write':>7} {'gainR':>7} {'B/s':>11} {'speed':>7}",
        ])
        refs = {}
        for r in rows:
            ref_name = AF_REF if r["group"] == "attention_free" else HYBRID_REF
            refs[(r["seed"], ref_name)] = next(
                (z for z in rows if z["seed"] == r["seed"] and z["arm"] == ref_name), r
            )
        for r in sorted(rows, key=lambda z: (z["group"], z["seed"], z["bpb_8k"])):
            ref_name = AF_REF if r["group"] == "attention_free" else HYBRID_REF
            ref = refs[(r["seed"], ref_name)]
            aux = r.get("cache_aux", {}) or {}
            lines.append(
                f"{r['arm']:<38} {r['seed']:>6d} {r['bpb_train_context']:>9.5f} {r['bpb_8k']:>9.5f} "
                f"{r['bpb_8k']-ref['bpb_8k']:>+9.5f} {r['bpb_16k']:>9.5f} "
                f"{r['oracle_bpb_8k']:>9.5f} {r['capture_8k']:>7.3f} "
                f"{float(aux.get('write', 0.0)):>7.3f} {float(aux.get('gain_recent', 0.0)):>+7.3f} "
                f"{r['bytes_per_second']:>11,.0f} {r['bytes_per_second']/max(ref['bytes_per_second'],1):>7.2f}"
            )

    table("STAGE A — ALL-STRUCTURE SCREEN", stage_a)
    table("STAGE B — LONGER PAIRED FINALISTS", stage_b)
    table("STAGE C — TWO-SEED CONFIRMATION", stage_c)

    lines += ["", "SEED-PAIRED AGGREGATES"]
    promoted = []
    for arm in [*selected_main, *selected_af]:
        ref = AF_REF if arm in AF_ARMS else HYBRID_REF
        d8, sd8 = arena.mean_delta(stage_c, arm, ref, "bpb_8k")
        d16, sd16 = arena.mean_delta(stage_c, arm, ref, "bpb_16k")
        speed = np.mean([
            r["bytes_per_second"] / next(
                z for z in stage_c if z["seed"] == r["seed"] and z["arm"] == ref
            )["bytes_per_second"]
            for r in stage_c if r["arm"] == arm
        ])
        lines.append(
            f"{arm:<38} vs {ref:<22} d8K={d8:+.5f}±{sd8:.5f} "
            f"d16K={d16:+.5f}±{sd16:.5f} speed={speed:.3f}x"
        )
        if arm in AF_ARMS:
            passed = d8 <= -args.af_promotion_gain and d16 <= 0.0 and speed >= args.af_min_speed
        else:
            passed = d8 <= -args.main_promotion_gain and d16 <= 0.0 and speed >= args.main_min_speed
        if passed:
            promoted.append(arm)

    lines += ["", "DECISION"]
    if promoted:
        lines.append("PROMOTE TO CANONICAL 50M BRIDGE: " + ", ".join(promoted))
        lines.append("Promotion means structural validation only; kernel work and a 50M canonical bridge remain mandatory.")
    else:
        lines.append("NO STRUCTURAL MEMORY ARM CLEARED THE TWO-SEED PROMOTION GATE.")
        lines.append("Retain the current hybrid and attention-free baselines; do not combine failed arms post hoc.")
    lines.append(f"selected_main={selected_main} | selected_attentionfree={selected_af}")
    lines.append("=" * 250)
    return "\n".join(lines) + "\n"


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=("all", "selftest", "summary"), default="all")
    p.add_argument("--outdir", default="./field_selective_episodic_memory_ablation_v9")
    p.add_argument("--cache-dir", default="./hf_cache")
    p.add_argument("--data-frac", type=float, default=0.05)
    p.add_argument("--data-device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--long-context", type=int, default=8192)
    p.add_argument("--very-long-context", type=int, default=16384)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--field-chunk", type=int, default=32)
    p.add_argument("--num-buckets", type=int, default=8192)
    p.add_argument("--salience-floor", type=float, default=0.10)
    p.add_argument("--residual-limit", type=float, default=4.0)
    p.add_argument("--hidden-step", type=int, default=8)
    p.add_argument("--max-param-delta-pct", type=float, default=0.15)
    p.add_argument("--amp", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--min-lr-ratio", type=float, default=0.10)
    p.add_argument("--weight-decay", type=float, default=0.10)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--distill-ramp", type=int, default=100)
    p.add_argument("--stage-a-steps", type=int, default=800)
    p.add_argument("--stage-b-steps", type=int, default=2400)
    p.add_argument("--confirm-steps", type=int, default=1800)
    p.add_argument("--main-finalists", type=int, default=2)
    p.add_argument("--af-finalists", type=int, default=1)
    p.add_argument("--stage-a-min-speed", type=float, default=0.70)
    p.add_argument("--main-promotion-gain", type=float, default=0.010)
    p.add_argument("--af-promotion-gain", type=float, default=0.008)
    p.add_argument("--main-min-speed", type=float, default=0.80)
    p.add_argument("--af-min-speed", type=float, default=0.85)
    p.add_argument("--screen-seed", type=int, default=0)
    p.add_argument("--confirm-seeds", default="1000,2000")
    p.add_argument("--eval-every", type=int, default=400)
    p.add_argument("--eval-windows", type=int, default=4)
    p.add_argument("--final-windows", type=int, default=8)
    p.add_argument("--long-windows", type=int, default=4)
    p.add_argument("--very-long-windows", type=int, default=2)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=400)
    p.add_argument("--model-seed", type=int, default=1234)
    p.add_argument("--data-seed", type=int, default=5678)
    p.add_argument("--eval-seed", type=int, default=9012)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


# Patch the proven v8 training/selection harness with the v9 architecture.
arena.HYBRID_REF = HYBRID_REF
arena.AF_REF = AF_REF
arena.HYBRID_ARMS = HYBRID_ARMS
arena.AF_ARMS = AF_ARMS
arena.ALL_ARMS = ALL_ARMS
arena.base_arm_for = base_arm_for
arena.attention_free = attention_free
arena.modify_model = modify_model
arena.raw_model = raw_model
arena.group_for = group_for
arena.selftest = selftest
arena.make_summary = make_summary
arena.parse_args = parse_args


if __name__ == "__main__":
    arena.main()
