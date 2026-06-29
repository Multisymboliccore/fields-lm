#!/usr/bin/env python3
"""Selective latent-addressing ablation for Field/PCAF.

The validated v10 selective-episodic memory improved *what* is written.  This
v11 arena tests the next structural bottleneck: whether the useful episode is
actually present in the shortlist and whether the shortlist is ranked with the
right view of the state.

Every arm keeps the validated selective-write + episodic-residual mechanism.
The new hypotheses are:
  - surface_multiview: asymmetric query/key and state-transition views rerank
    the existing exact n-gram shortlist.
  - latent_coarse / latent_fine: a second causal bank is addressed by a compact
    discrete code derived from the learned Field/PCAF state embedding.  The
    bank can retrieve semantically similar states that do not share the exact
    byte n-gram.  Its correction gain starts at zero, preserving the reference.
  - latent_multiview: combines the latent bank with the asymmetric state and
    transition reranker.

The attention-free lane uses the same memory experiments on the validated
softpatch + multiscale causal-convolution backbone and introduces no local
query-key attention block.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
V9_PATH = HERE / "field_selective_episodic_memory_ablation_v9.py"


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


v9 = import_module(V9_PATH, "field_selective_episodic_memory_ablation_v9_dep")
arena = v9.arena
v3 = v9.v3
VOCAB = v9.VOCAB
LN2 = v9.LN2

HYBRID_REF = "selective_residual_ref"
AF_REF = "attentionfree_selective_residual_ref"
HYBRID_ARMS = (
    HYBRID_REF,
    "surface_multiview",
    "latent_coarse",
    "latent_fine",
    "latent_multiview",
)
AF_ARMS = (
    AF_REF,
    "attentionfree_latent_coarse",
    "attentionfree_latent_fine",
    "attentionfree_latent_multiview",
)
ALL_ARMS = (*HYBRID_ARMS, *AF_ARMS)


def causal_recent_candidates_from_buckets(
    buckets: torch.Tensor, num_buckets: int, top_k: int
) -> torch.Tensor:
    """Return the top-k most recent earlier positions sharing each bucket.

    This is the same exact causal segmented-sort construction used by PCAF,
    generalized to externally supplied bucket IDs.
    """
    if buckets.dtype != torch.long:
        buckets = buckets.long()
    b, t = buckets.shape
    pos = torch.arange(t, device=buckets.device, dtype=torch.long)[None, :].expand(b, -1)
    bid = torch.arange(b, device=buckets.device, dtype=torch.long)[:, None].expand(-1, t)
    group = bid * int(num_buckets) + buckets
    key = group * (t + 1) + pos
    perm = torch.argsort(key.reshape(-1), stable=True)
    sg = group.reshape(-1).index_select(0, perm)
    sp = pos.reshape(-1).index_select(0, perm)
    n = perm.numel()
    cand_sorted = torch.full((n, top_k), -1, device=buckets.device, dtype=torch.long)
    for k in range(1, top_k + 1):
        if k >= n:
            break
        same = sg[k:] == sg[:-k]
        prev = torch.where(same, sp[:-k], torch.full_like(sp[:-k], -1))
        cand_sorted[k:, k - 1] = prev
    out = torch.full_like(cand_sorted, -1)
    out[perm] = cand_sorted
    return out.view(b, t, top_k)


def deduplicate_candidates(extra: torch.Tensor, primary: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Remove duplicates within extra and any position already in primary."""
    valid = extra >= 0
    if primary.numel():
        dup_primary = (extra[..., :, None] == primary[..., None, :]) & (primary[..., None, :] >= 0)
        valid = valid & ~dup_primary.any(-1)
    k = extra.size(-1)
    if k > 1:
        same = extra[..., :, None] == extra[..., None, :]
        earlier = torch.tril(
            torch.ones((k, k), device=extra.device, dtype=torch.bool), diagonal=-1
        )
        valid = valid & ~(same & earlier).any(-1)
    return torch.where(valid, extra, torch.full_like(extra, -1)), valid


def latent_bucket_codes(proj: torch.Tensor, bits: int) -> torch.Tensor:
    """Discrete causal address from the learned PCAF state embedding.

    The code path is intentionally detached: the continuous scorer still trains
    normally, while the discrete shortlist remains a stable indexing operation.
    """
    bits = max(1, min(int(bits), int(proj.size(-1)), 20))
    signs = (proj.detach()[..., :bits] >= 0).long()
    powers = (1 << torch.arange(bits, device=proj.device, dtype=torch.long))
    return (signs * powers).sum(-1)


class SelectiveLatentAddressingCache(nn.Module):
    """Validated selective-residual PCAF plus optional latent-address bank.

    The surface bank exactly reproduces v10 at initialization.  A latent bank
    uses a compact code from the learned state embedding to produce a second
    causal shortlist.  Its retrieved error correction is zero-initialized.

    Optional multi-view score residuals use asymmetric state projections and
    state-transition projections.  Their scalar gains are zero-initialized, so
    the existing PCAF ranking is also preserved at step zero.
    """

    def __init__(
        self,
        base: nn.Module,
        *,
        latent_bits: int | None,
        multiview: bool,
        address_dim: int,
        latent_top_k: int,
        salience_floor: float,
        residual_limit: float,
        score_limit: float,
    ):
        super().__init__()
        self.base = base
        self.state_dim = int(base.state_dim)
        self.memory_dim = int(base.memory_dim)
        self.num_buckets = int(base.num_buckets)
        self.order = int(base.order)
        self.top_k = 4
        self.router_mode = str(base.router_mode)
        self.enabled = True
        self.distill_scale = 1.0
        self.last_aux: Dict[str, float] = {}

        self.latent_bits = None if latent_bits is None else int(latent_bits)
        self.latent_top_k = int(latent_top_k)
        self.multiview = bool(multiview)
        self.salience_floor = float(salience_floor)
        self.residual_limit = float(residual_limit)
        self.score_limit = float(score_limit)

        # Exact v10 selective-residual parameters.
        self.residual_gain_recent = nn.Parameter(torch.zeros(()))
        self.salience_strength_raw = nn.Parameter(torch.zeros(()))

        if self.latent_bits is not None:
            self.latent_residual_gain = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("latent_residual_gain", None)

        if self.multiview:
            ad = max(8, int(address_dim))
            self.query_view = nn.Linear(self.state_dim, ad, bias=False)
            self.key_view = nn.Linear(self.state_dim, ad, bias=False)
            self.delta_query_view = nn.Linear(self.state_dim, ad, bias=False)
            self.delta_key_view = nn.Linear(self.state_dim, ad, bias=False)
            self.asym_score_raw = nn.Parameter(torch.zeros(()))
            self.delta_score_raw = nn.Parameter(torch.zeros(()))
        else:
            self.query_view = None
            self.key_view = None
            self.delta_query_view = None
            self.delta_key_view = None
            self.register_parameter("asym_score_raw", None)
            self.register_parameter("delta_score_raw", None)

    @property
    def distill_temperature(self) -> float:
        return float(self.base.distill_temperature)

    @property
    def distill_weight(self) -> float:
        return float(self.base.distill_weight)

    def _features(self, *args, **kwargs):
        return self.base._features(*args, **kwargs)

    def _bank_distribution(
        self,
        weights: torch.Tensor,
        valid: torch.Tensor,
        cand_tokens: torch.Tensor,
        past_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w = weights * valid.float()
        denom = w.sum(-1, keepdim=True)
        has = denom.squeeze(-1) > 1e-8
        w = w / denom.clamp_min(1e-8)
        b, t, _ = w.shape
        successor = torch.zeros((b, t, VOCAB), device=w.device, dtype=torch.float32)
        successor.scatter_add_(-1, cand_tokens.long(), w.float())
        past = (w[..., None].float() * past_probs.float()).sum(-2)
        return successor, past, has

    def _multiview_scores(
        self,
        states: torch.Tensor,
        safe: torch.Tensor,
        valid: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.multiview:
            z = states.new_zeros(safe.shape, dtype=torch.float32)
            s0 = states.new_zeros((), dtype=torch.float32)
            return z, s0, s0

        qv = v3.normalize_rows(self.query_view(states.float()))
        kv = v3.normalize_rows(self.key_view(states.float()))
        kv_c = kv[batch_idx, safe]
        asym = (qv[:, :, None, :] * kv_c).sum(-1) * (qv.size(-1) ** -0.5)

        previous = torch.cat((torch.zeros_like(states[:, :1]), states[:, :-1]), dim=1)
        delta = states.float() - previous.float()
        qd = v3.normalize_rows(self.delta_query_view(delta))
        kd = v3.normalize_rows(self.delta_key_view(delta))
        kd_c = kd[batch_idx, safe]
        dscore = (qd[:, :, None, :] * kd_c).sum(-1) * (qd.size(-1) ** -0.5)

        asym_gain = self.score_limit * torch.tanh(self.asym_score_raw.float())
        delta_gain = self.score_limit * torch.tanh(self.delta_score_raw.float())
        extra = asym_gain * asym + delta_gain * dscore
        return extra.masked_fill(~valid, 0.0), asym_gain, delta_gain

    def _prepare_bank(
        self,
        states: torch.Tensor,
        targets: torch.Tensor,
        param_probs: torch.Tensor,
        proj: torch.Tensor,
        idx: torch.Tensor,
        *,
        use_multiview: bool,
    ) -> Dict[str, torch.Tensor]:
        b, t, _ = states.shape
        valid = idx >= 0
        safe = idx.clamp_min(0)
        batch_idx = torch.arange(b, device=states.device)[:, None, None]
        q = proj[:, :, None, :]
        ck = proj[batch_idx, safe]
        scores = (ck * q).sum(-1) * (self.memory_dim ** -0.5)
        recency = safe.float() / max(float(t - 1), 1.0)
        scores = scores + self.base.recency_scale.float() * recency

        asym_gain = states.new_zeros((), dtype=torch.float32)
        delta_gain = states.new_zeros((), dtype=torch.float32)
        if use_multiview:
            extra, asym_gain, delta_gain = self._multiview_scores(
                states, safe, valid, batch_idx
            )
            scores = scores + extra

        cand_tokens = targets[batch_idx, safe]
        past_probs = param_probs.detach()[batch_idx, safe]
        past_true_prob = past_probs.gather(-1, cand_tokens[..., None]).squeeze(-1)

        # Validated v10 write salience: surprise + local state novelty.
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
        return {
            "idx": idx,
            "valid": valid,
            "safe": safe,
            "scores": scores,
            "weights": weights,
            "recency": recency,
            "cand_tokens": cand_tokens,
            "past_probs": past_probs,
            "write_strength": write_strength,
            "has": valid.any(-1),
            "asym_gain": asym_gain,
            "delta_gain": delta_gain,
        }

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
        proj = v3.normalize_rows(F.linear(states.float(), self.base.shared_weight.float()))

        surface_idx = v3.causal_recent_candidates(
            tokens, self.order, self.num_buckets, self.top_k
        )
        surface = self._prepare_bank(
            states, targets, param_probs, proj, surface_idx,
            use_multiview=self.multiview,
        )
        valid = surface["valid"]
        safe = surface["safe"]
        weights = surface["weights"]
        scores = surface["scores"]
        recency = surface["recency"]
        cand_tokens = surface["cand_tokens"]
        past_probs = surface["past_probs"]
        has_any = surface["has"]
        batch_idx = torch.arange(b, device=states.device)[:, None, None]

        cache_dist = torch.zeros((b, t, VOCAB), device=states.device, dtype=torch.float32)
        cache_dist.scatter_add_(-1, cand_tokens.long(), weights.float())
        target_cache = cache_dist.gather(-1, targets[..., None]).squeeze(-1)

        active = has_any.reshape(-1)
        state_logit = self.base.state_gate(states).float().squeeze(-1)
        if bool(active.any()):
            features = self._features(
                scores.reshape(-1, self.top_k)[active],
                weights.reshape(-1, self.top_k)[active],
                valid.reshape(-1, self.top_k)[active],
                cand_tokens.reshape(-1, self.top_k)[active],
                recency.reshape(-1, self.top_k)[active],
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

        # Validated selective episodic residual on the exact surface shortlist.
        q_mem, p_mem, has_mem = self._bank_distribution(
            weights, valid, cand_tokens, past_probs
        )
        residual = q_mem - p_mem
        gain_recent = self.residual_limit * torch.tanh(self.residual_gain_recent.float())
        correction = (gate * has_mem.float() * gain_recent)[..., None] * residual

        latent_coverage = states.new_zeros((), dtype=torch.float32)
        latent_overlap = states.new_zeros((), dtype=torch.float32)
        latent_gain = states.new_zeros((), dtype=torch.float32)
        latent_conf_mean = states.new_zeros((), dtype=torch.float32)
        latent_write_mean = states.new_zeros((), dtype=torch.float32)

        if self.latent_bits is not None:
            codes = latent_bucket_codes(proj, self.latent_bits)
            latent_raw = causal_recent_candidates_from_buckets(
                codes, 1 << self.latent_bits, self.latent_top_k
            )
            raw_valid = latent_raw >= 0
            if bool(raw_valid.any()):
                overlap_mask = (
                    (latent_raw[..., :, None] == surface_idx[..., None, :])
                    & (surface_idx[..., None, :] >= 0)
                    & raw_valid[..., :, None]
                )
                latent_overlap = overlap_mask.any(-1).float().sum() / raw_valid.float().sum().clamp_min(1.0)
            latent_idx, latent_valid = deduplicate_candidates(latent_raw, surface_idx)
            latent = self._prepare_bank(
                states, targets, param_probs, proj, latent_idx,
                use_multiview=self.multiview,
            )
            q_lat, p_lat, has_lat = self._bank_distribution(
                latent["weights"], latent["valid"], latent["cand_tokens"], latent["past_probs"]
            )
            residual_lat = q_lat - p_lat
            token_mass = q_lat.max(-1).values
            count = latent["valid"].float().sum(-1)
            ent = -(latent["weights"] * torch.log(latent["weights"].clamp_min(1e-8))).sum(-1)
            ent = torch.where(
                count > 1,
                ent / torch.log(count.clamp_min(2.0)),
                torch.zeros_like(ent),
            )
            latent_conf = (0.65 * token_mass + 0.35 * (1.0 - ent)).clamp(0.0, 1.0)
            latent_gain = self.residual_limit * torch.tanh(self.latent_residual_gain.float())
            correction = correction + (
                has_lat.float() * latent_conf * latent_gain
            )[..., None] * residual_lat
            latent_coverage = has_lat.float().mean()
            latent_conf_mean = latent_conf[has_lat].mean() if bool(has_lat.any()) else latent_conf.mean()
            latent_write_mean = (
                latent["write_strength"][latent["valid"]].mean()
                if bool(latent["valid"].any()) else states.new_zeros((), dtype=torch.float32)
            )

        final_log_probs = F.log_softmax(
            torch.log(base_probs.clamp_min(1e-8)) + correction, dim=-1
        )
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
                F.binary_cross_entropy_with_logits(
                    gate_logit_active, teacher, reduction="none"
                ) * weight
            ).sum() / weight.sum().clamp_min(1.0)
            loss = primary + self.distill_weight * float(self.distill_scale) * aux
            self.last_aux = {
                "distill": float(aux.detach()),
                "teacher": float(teacher.mean()),
                "cache_win": float((log_adv > 0).float().mean()),
                "write": float(surface["write_strength"][valid].mean()) if bool(valid.any()) else 0.0,
                "gain_recent": float(gain_recent.detach()),
                "latent_gain": float(latent_gain.detach()),
                "latent_coverage": float(latent_coverage.detach()),
                "latent_overlap": float(latent_overlap.detach()),
                "latent_conf": float(latent_conf_mean.detach()),
                "latent_write": float(latent_write_mean.detach()),
                "asym_gain": float(surface["asym_gain"].detach()),
                "delta_gain": float(surface["delta_gain"].detach()),
            }

        stats = None
        if compute_metrics:
            with torch.no_grad():
                param_loss = param_nll.mean()
                target_cache_flat = target_cache.reshape(-1)
                oracle_target = torch.maximum(param_target, target_cache_flat)
                oracle_target = torch.where(active, oracle_target, param_target)
                oracle_loss = -torch.log(oracle_target.clamp_min(1e-8)).mean()
                denom = param_loss - oracle_loss
                capture = float((param_loss - primary) / denom) if float(denom) > 1e-8 else 0.0
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


def attention_free(arm: str) -> bool:
    return arm in AF_ARMS


def base_arm_for(arm: str) -> str:
    return "softpatch_multiscale_conf" if attention_free(arm) else "softpatch_local_w256_conf"


def mode_for(arm: str) -> Tuple[int | None, bool]:
    if arm in {HYBRID_REF, AF_REF}:
        return None, False
    multiview = arm.endswith("multiview") or arm == "surface_multiview"
    if arm == "surface_multiview":
        return None, True
    if "coarse" in arm:
        return 8, multiview
    if "fine" in arm:
        return 12, multiview
    if "latent_multiview" in arm:
        return 10, True
    raise ValueError(arm)


def modify_model(model: nn.Module, arm: str, args) -> nn.Module:
    bits, multiview = mode_for(arm)
    if arm in {HYBRID_REF, AF_REF}:
        model.cache = v9.EpisodicCorrectiveCache(
            model.cache,
            selective=True,
            residual=True,
            hierarchical=False,
            salience_floor=args.salience_floor,
            residual_limit=args.residual_limit,
        )
    else:
        model.cache = SelectiveLatentAddressingCache(
            model.cache,
            latent_bits=bits,
            multiview=multiview,
            address_dim=args.address_dim,
            latent_top_k=args.latent_top_k,
            salience_floor=args.salience_floor,
            residual_limit=args.residual_limit,
            score_limit=args.score_limit,
        )
    return model


def raw_model(arm: str, hidden: int, args, device: torch.device) -> nn.Module:
    model = v3.FieldPCAFLM(
        base_arm_for(arm), args.dim, args.layers, args.heads, hidden,
        args.field_chunk, args.num_buckets,
    ).to(device)
    return modify_model(model, arm, args).to(device)


def group_for(arm: str) -> str:
    return "attention_free" if attention_free(arm) else "hybrid"


def selftest(args, device, shapes):
    arena.log("[selftest] finite backward, exact v10 identity, latent causality, gradient reach")
    x = torch.randint(0, VOCAB, (1, 33), device=device)
    y = torch.randint(0, VOCAB, (1, 33), device=device)
    for arm in ALL_ARMS:
        m = arena.build_model(arm, 0, shapes[arm], args, device)
        with arena.amp_ctx(device, args.amp):
            loss, primary, _ = m.loss_and_stats(x, y, False)
        loss.backward()
        finite = bool(torch.isfinite(loss)) and all(
            p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters()
        )
        aux = getattr(m.cache, "last_aux", {})
        arena.log(
            f"[selftest] {arm:<42} params={arena.nparams(m):,} "
            f"d={shapes[arm]['delta_pct']:+.3f}% loss={float(loss.detach()):.5f} "
            f"primary={float(primary):.5f} finite={finite} aux={aux}"
        )
        if not finite:
            raise AssertionError(arm)
        if hasattr(m.cache, "latent_residual_gain") and m.cache.latent_residual_gain is not None:
            g = m.cache.latent_residual_gain.grad
            arena.log(f"[selftest] {arm:<42} latent_gain_grad={0.0 if g is None else float(g):+.3e}")
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Experimental arms must exactly reproduce v10 at their zero-init overlays.
    for arm in ("surface_multiview", "latent_coarse", "latent_fine", "latent_multiview"):
        arena.seed_all(991)
        ref = v3.FieldPCAFLM("softpatch_local_w256_conf", 32, 1, 4, 64, 8, 64).to(device).eval()
        ref.cache = v9.EpisodicCorrectiveCache(
            ref.cache, selective=True, residual=True, hierarchical=False,
            salience_floor=args.salience_floor, residual_limit=args.residual_limit,
        ).to(device)
        arena.seed_all(991)
        exp = v3.FieldPCAFLM("softpatch_local_w256_conf", 32, 1, 4, 64, 8, 64).to(device).eval()
        bits, mv = mode_for(arm)
        exp.cache = SelectiveLatentAddressingCache(
            exp.cache, latent_bits=bits, multiview=mv,
            address_dim=min(args.address_dim, 16), latent_top_k=args.latent_top_k,
            salience_floor=args.salience_floor, residual_limit=args.residual_limit,
            score_limit=args.score_limit,
        ).to(device)
        with torch.no_grad(), arena.amp_ctx(device, args.amp):
            l0 = ref.loss_and_stats(x, y, False)[1]
            l1 = exp.loss_and_stats(x, y, False)[1]
        err = float((l0 - l1).abs())
        arena.log(f"[selftest] {arm:<42} zero-overlay abs_loss={err:.3e}")
        if err > 4e-5:
            raise AssertionError((arm, "zero identity", err))
        del ref, exp

    for arm in ("surface_multiview", "latent_coarse", "latent_fine", "latent_multiview", "attentionfree_latent_multiview"):
        m = arena.build_model(arm, 11, shapes[arm], args, device).eval()
        full = torch.randint(0, VOCAB, (1, 41), device=device)
        full2 = full.clone()
        prefix_targets = 17
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
        arena.log(f"[selftest] {arm:<42} causal_token_nll max_abs={cerr:.3e}")
        if cerr > 5e-4:
            raise AssertionError((arm, cerr))
        del m
    arena.log("[selftest] PASS")


def make_summary(stage_a, stage_b, stage_c, shapes, selected_main, selected_af, args):
    lines = [
        "FIELD SELECTIVE LATENT ADDRESSING ABLATION v11 — SHORTLIST/ADDRESS VALIDATION",
        "=" * 250,
        f"Protocol: WikiText-103 {args.data_frac:.1%} | train ctx={args.seq_len} | "
        f"eval={args.long_context}/{args.very_long_context} | bytes/update={args.batch_size*args.accum*args.seq_len:,}",
        "Reference: v10 selective-write + episodic-residual memory on the best hybrid/attention-free backbones.",
        "Latent arms add a causal state-code shortlist; multiview arms add zero-init asymmetric state/transition reranking.",
        "Portable exact Field math; any winner must pass a canonical 50M bridge before a 300M rerun.",
        "",
        "PARAMETER MATCHING",
        f"{'arm':<44} {'group':>14} {'params':>13} {'d%':>8} {'ff':>6} {'AF':>4}",
    ]
    for arm in ALL_ARMS:
        s = shapes[arm]
        lines.append(
            f"{arm:<44} {group_for(arm):>14} {s['params']:>13,d} {s['delta_pct']:>+8.3f} "
            f"{s['hidden']:>6d} {('yes' if s['attention_free'] else 'no'):>4}"
        )

    def table(title, rows):
        lines.extend([
            "", title,
            f"{'arm':<44} {'seed':>6} {'BPB2K':>9} {'BPB8K':>9} {'dGroup':>9} {'BPB16K':>9} "
            f"{'oracle':>9} {'cap':>7} {'latCov':>7} {'gLat':>7} {'gAsym':>7} {'B/s':>11} {'speed':>7}",
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
                f"{r['arm']:<44} {r['seed']:>6d} {r['bpb_train_context']:>9.5f} {r['bpb_8k']:>9.5f} "
                f"{r['bpb_8k']-ref['bpb_8k']:>+9.5f} {r['bpb_16k']:>9.5f} "
                f"{r['oracle_bpb_8k']:>9.5f} {r['capture_8k']:>7.3f} "
                f"{float(aux.get('latent_coverage', 0.0)):>7.3f} {float(aux.get('latent_gain', 0.0)):>+7.3f} "
                f"{float(aux.get('asym_gain', 0.0)):>+7.3f} "
                f"{r['bytes_per_second']:>11,.0f} {r['bytes_per_second']/max(ref['bytes_per_second'],1):>7.2f}"
            )

    table("STAGE A — ALL ADDRESSING HYPOTHESES", stage_a)
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
            f"{arm:<44} vs {ref:<38} d8K={d8:+.5f}±{sd8:.5f} "
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
        lines.append("PROMOTE TO CANONICAL 50M ADDRESSING BRIDGE: " + ", ".join(promoted))
        lines.append("Promotion is structural only; the latent sort/index path still requires canonical optimization.")
    else:
        lines.append("NO LATENT-ADDRESSING ARM CLEARED THE TWO-SEED PROMOTION GATE.")
        lines.append("Freeze v10 and proceed to the selective-episodic 300M rerun without further small-arena tuning.")
    lines.append(f"selected_main={selected_main} | selected_attentionfree={selected_af}")
    lines.append("=" * 250)
    return "\n".join(lines) + "\n"


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=("all", "selftest", "summary"), default="all")
    p.add_argument("--outdir", default="./field_selective_latent_addressing_ablation_v11")
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
    p.add_argument("--address-dim", type=int, default=24)
    p.add_argument("--latent-top-k", type=int, default=4)
    p.add_argument("--score-limit", type=float, default=2.0)
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
    p.add_argument("--stage-a-min-speed", type=float, default=0.65)
    p.add_argument("--main-promotion-gain", type=float, default=0.010)
    p.add_argument("--af-promotion-gain", type=float, default=0.008)
    p.add_argument("--main-min-speed", type=float, default=0.85)
    p.add_argument("--af-min-speed", type=float, default=0.88)
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


# Patch the proven v8/v9 training harness with the v11 architecture.
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
