#!/usr/bin/env python3
"""Final short structural ablation before the canonical 50M bridge.

Baseline: the validated v10 selective episodic memory plus the v11 surface
multi-view reranker.  This arena tests the remaining Cloud hypotheses that
were not already evaluated directly:

  1) A small causal block-Delta fast-weight sidecar (rank 8 / rank 16).
  2) Verified multi-byte continuation copying (2 / 4 byte spans).
  3) Complex phase-interference recall, native to the complex Field framing.
  4) Conservative combinations of the best low-cost mechanisms.

The attention-free lane remains active.  It starts from the validated v10
attention-free selective-residual model and separately measures whether the
v11 surface multi-view reranker and the new mechanisms help without adding a
local/full query-key attention path.

This is a mechanism-ranking arena.  Any promoted arm must still pass the
canonical 50M Triton bridge before a new 300M run.
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
V11_PATH = HERE / "field_selective_latent_addressing_ablation_v11.py"


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


v11 = import_module(V11_PATH, "field_selective_latent_addressing_ablation_v11_dep")
v9 = v11.v9
arena = v11.arena
v3 = v11.v3
VOCAB = v11.VOCAB
LN2 = v11.LN2

HYBRID_REF = "surface_multiview_ref"
AF_REF = "attentionfree_selective_ref"
HYBRID_ARMS = (
    HYBRID_REF,
    "delta_rank8",
    "delta_rank16",
    "verified_span2",
    "verified_span4",
    "phase_interference",
    "delta8_span4",
    "delta8_phase",
)
AF_ARMS = (
    AF_REF,
    "attentionfree_surface_multiview",
    "attentionfree_delta8",
    "attentionfree_span4",
    "attentionfree_phase",
    "attentionfree_delta_span",
)
ALL_ARMS = (*HYBRID_ARMS, *AF_ARMS)


def attention_free(arm: str) -> bool:
    return arm in AF_ARMS


def base_arm_for(arm: str) -> str:
    return "softpatch_multiscale_conf" if attention_free(arm) else "softpatch_local_w256_conf"


def group_for(arm: str) -> str:
    return "attention_free" if attention_free(arm) else "hybrid"


def make_v10_cache(base: nn.Module, args) -> nn.Module:
    return v9.EpisodicCorrectiveCache(
        base,
        selective=True,
        residual=True,
        hierarchical=False,
        salience_floor=args.salience_floor,
        residual_limit=args.residual_limit,
    )


def make_surface_multiview_cache(base: nn.Module, args) -> nn.Module:
    return v11.SelectiveLatentAddressingCache(
        base,
        latent_bits=None,
        multiview=True,
        address_dim=args.address_dim,
        latent_top_k=args.latent_top_k,
        salience_floor=args.salience_floor,
        residual_limit=args.residual_limit,
        score_limit=args.score_limit,
    )


class BlockDeltaSidecar(nn.Module):
    """Small causal fast-weight memory updated with a gated delta rule.

    The matrix is updated only at completed block boundaries.  Tokens in block j
    can read a memory containing blocks < j, while their token-level queries are
    still causal Field states.  This is deliberately smaller and safer than a
    full per-token DeltaNet transplant, but it directly tests the delta-update
    principle without changing the Field recurrence.
    """

    def __init__(
        self,
        state_dim: int,
        rank: int,
        heads: int,
        block_size: int,
        max_mix: float,
        gate_bias: float,
        gate_grad_scale: float,
    ):
        super().__init__()
        if rank < 2 or heads < 1 or block_size < 2:
            raise ValueError((rank, heads, block_size))
        self.state_dim = int(state_dim)
        self.rank = int(rank)
        self.heads = int(heads)
        self.block_size = int(block_size)
        self.max_mix = float(max_mix)

        width = self.heads * self.rank
        self.norm = v3.RMSNorm(state_dim)
        self.q_proj = nn.Linear(state_dim, width, bias=False)
        self.k_proj = nn.Linear(state_dim, width, bias=False)
        self.v_proj = nn.Linear(state_dim, width, bias=False)
        self.beta_proj = nn.Linear(state_dim, self.heads, bias=True)
        self.out_proj = nn.Linear(width, VOCAB, bias=False)
        self.gate_proj = nn.Linear(state_dim, 1, bias=True)
        self.decay_logit = nn.Parameter(torch.full((self.heads,), math.log(0.96 / 0.04)))

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.35)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.constant_(self.beta_proj.bias, math.log(0.12 / 0.88))
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, float(gate_bias))
        for parameter in self.gate_proj.parameters():
            parameter.register_hook(lambda grad, scale=float(gate_grad_scale): grad * scale)

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.norm(states).float()
        b, t, _ = x.shape
        h, r = self.heads, self.rank
        q = v3.normalize_rows(self.q_proj(x).view(b, t, h, r))

        end_idx = torch.arange(
            self.block_size - 1, t, self.block_size, device=x.device, dtype=torch.long
        )
        if end_idx.numel() == 0 or int(end_idx[-1]) != t - 1:
            end_idx = torch.cat((end_idx, torch.tensor([t - 1], device=x.device)))
        summaries = x.index_select(1, end_idx)
        nb = summaries.size(1)
        k = v3.normalize_rows(self.k_proj(summaries).view(b, nb, h, r))
        v = torch.tanh(self.v_proj(summaries).view(b, nb, h, r))
        beta = torch.sigmoid(self.beta_proj(summaries).float()).clamp(0.001, 0.999)
        decay = torch.sigmoid(self.decay_logit.float()).view(1, h, 1, 1)

        mem = x.new_zeros((b, h, r, r), dtype=torch.float32)
        memories: List[torch.Tensor] = [mem]
        # Memory for block j receives only the completed summary of block j-1.
        for j in range(max(0, nb - 1)):
            kj = k[:, j]
            vj = v[:, j]
            pred = torch.einsum("bhij,bhj->bhi", mem, kj)
            err = vj - pred
            update = err[..., :, None] * kj[..., None, :]
            mem = decay * mem + beta[:, j, :, None, None] * update
            memories.append(mem)
        mem_stack = torch.stack(memories, dim=1)

        block_ids = torch.div(
            torch.arange(t, device=x.device, dtype=torch.long),
            self.block_size,
            rounding_mode="floor",
        ).clamp_max(mem_stack.size(1) - 1)
        token_mem = mem_stack.index_select(1, block_ids)
        read = torch.einsum("bthij,bthj->bthi", token_mem, q)
        read = read.reshape(b, t, h * r)
        probs = torch.softmax(self.out_proj(read).float(), dim=-1)
        gate = self.max_mix * torch.clamp(
            self.gate_proj(x).float().squeeze(-1), min=0.0, max=1.0
        )
        coverage = (block_ids > 0).float().view(1, t).expand(b, -1)
        gate = gate * coverage
        return probs, gate, coverage


class VerifiedSpanSidecar(nn.Module):
    """Causal continuation copier with byte-by-byte verification.

    A continuation candidate is admitted only when all already-observed bytes
    since an earlier exact PCAF match agree with that past continuation.  The
    mechanism predicts just the next byte, so it cannot leak or commit an
    unverified future span.
    """

    def __init__(
        self,
        state_dim: int,
        order: int,
        num_buckets: int,
        max_span: int,
        top_k: int,
        max_mix: float,
        gate_bias: float,
        gate_grad_scale: float,
    ):
        super().__init__()
        if max_span not in {2, 4}:
            raise ValueError(max_span)
        self.order = int(order)
        self.num_buckets = int(num_buckets)
        self.max_span = int(max_span)
        self.top_k = int(top_k)
        self.max_mix = float(max_mix)
        self.length_logits = nn.Parameter(torch.zeros(max_span - 1))
        self.gate_proj = nn.Linear(state_dim, 1, bias=True)
        self.conf_scale = nn.Parameter(torch.tensor(0.0))
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, float(gate_bias))
        for parameter in self.gate_proj.parameters():
            parameter.register_hook(lambda grad, scale=float(gate_grad_scale): grad * scale)
        self.conf_scale.register_hook(lambda grad, scale=float(gate_grad_scale): grad * scale)

    def forward(
        self, states: torch.Tensor, tokens: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t = tokens.shape
        surface = v3.causal_recent_candidates(
            tokens, self.order, self.num_buckets, self.top_k
        )
        all_tokens: List[torch.Tensor] = []
        all_scores: List[torch.Tensor] = []
        all_valid: List[torch.Tensor] = []
        batch_idx = torch.arange(b, device=tokens.device)[:, None, None]

        for length in range(1, self.max_span):
            src_short = surface[:, : t - length, :]
            valid_short = src_short >= 0
            safe = src_short.clamp_min(0)
            for j in range(length):
                observed = tokens[:, 1 + j : t - length + 1 + j, None]
                past_byte = targets[batch_idx, safe + j]
                valid_short = valid_short & (past_byte == observed)
            next_token = targets[batch_idx, safe + length]
            recency = safe.float() / max(float(t - 1), 1.0)
            score_short = recency + self.length_logits[length - 1].float()

            tok = torch.zeros((b, t, self.top_k), device=tokens.device, dtype=torch.long)
            sco = torch.full((b, t, self.top_k), -1.0e9, device=tokens.device)
            val = torch.zeros((b, t, self.top_k), device=tokens.device, dtype=torch.bool)
            tok[:, length:] = next_token
            sco[:, length:] = score_short
            val[:, length:] = valid_short
            all_tokens.append(tok)
            all_scores.append(sco)
            all_valid.append(val)

        cand_tokens = torch.cat(all_tokens, dim=-1)
        scores = torch.cat(all_scores, dim=-1)
        valid = torch.cat(all_valid, dim=-1)
        scores = scores.masked_fill(~valid, -1.0e9)
        weights = torch.softmax(scores.float(), dim=-1) * valid.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        dist = torch.zeros((b, t, VOCAB), device=tokens.device, dtype=torch.float32)
        dist.scatter_add_(-1, cand_tokens, weights)
        has = valid.any(-1)
        confidence = dist.max(-1).values
        raw_gate = self.gate_proj(states).float().squeeze(-1)
        raw_gate = raw_gate + self.conf_scale.float() * (confidence - 0.5)
        gate = self.max_mix * torch.clamp(raw_gate, min=0.0, max=1.0) * has.float()
        return dist, gate, has.float()


class PhaseInterferenceSidecar(nn.Module):
    """Complex holographic memory with constructive/destructive interference.

    Past target-value codes are written with conjugated state-derived phase.
    The current query phase reads the exclusive cumulative memory.  Similar
    phase codes add constructively; unrelated traces tend to cancel.
    """

    def __init__(
        self,
        state_dim: int,
        bands: int,
        value_rank: int,
        max_mix: float,
        gate_bias: float,
        salience_floor: float,
        gate_grad_scale: float,
    ):
        super().__init__()
        self.bands = int(bands)
        self.value_rank = int(value_rank)
        self.max_mix = float(max_mix)
        self.salience_floor = float(salience_floor)
        self.norm = v3.RMSNorm(state_dim)
        self.phase_proj = nn.Linear(state_dim, self.bands, bias=False)
        self.value_embed = nn.Embedding(VOCAB, self.value_rank)
        self.decode_bias = nn.Parameter(torch.zeros(VOCAB))
        self.gate_proj = nn.Linear(state_dim, 1, bias=True)
        nn.init.xavier_uniform_(self.phase_proj.weight, gain=0.5)
        nn.init.normal_(self.value_embed.weight, mean=0.0, std=self.value_rank ** -0.5)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, float(gate_bias))
        for parameter in self.gate_proj.parameters():
            parameter.register_hook(lambda grad, scale=float(gate_grad_scale): grad * scale)

    def forward(
        self,
        states: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.norm(states).float()
        angles = math.pi * torch.tanh(self.phase_proj(x).float())
        phase = torch.complex(torch.cos(angles), torch.sin(angles))

        with torch.no_grad():
            param_probs = torch.softmax(logits.float(), dim=-1)
            true_prob = param_probs.gather(-1, targets[..., None]).squeeze(-1)
            surprise = torch.sqrt((1.0 - true_prob).clamp(0.0, 1.0))
            prev = torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)
            xn = v3.normalize_rows(x)
            pn = v3.normalize_rows(prev)
            novelty = (0.5 * (1.0 - (xn * pn).sum(-1))).clamp(0.0, 1.0)
            salience = self.salience_floor + (1.0 - self.salience_floor) * (
                0.75 * surprise + 0.25 * novelty
            )

        values = self.value_embed(targets).float()
        write = phase.conj()[..., :, None] * values[..., None, :].to(torch.complex64)
        write = write * salience[..., None, None]
        inclusive = torch.cumsum(write, dim=1)
        memory = inclusive - write
        mass_inclusive = torch.cumsum(salience, dim=1)
        mass = (mass_inclusive - salience).clamp_min(1.0)
        read = (phase[..., :, None] * memory).real.sum(-2)
        read = read / torch.sqrt(mass[..., None] * float(self.bands))
        logits_phase = F.linear(read.float(), self.value_embed.weight.float(), self.decode_bias.float())
        probs = torch.softmax(logits_phase, dim=-1)
        coverage = (mass_inclusive - salience > 0).float()
        gate = self.max_mix * torch.clamp(
            self.gate_proj(x).float().squeeze(-1), min=0.0, max=1.0
        )
        gate = gate * coverage
        return probs, gate, coverage


class CloudMechanismCache(nn.Module):
    """Post-memory mixture wrapper for delta/span/phase hypotheses.

    Each sidecar produces a proper next-byte distribution.  The final target
    probability is a causal convex mixture with the validated base cache.  This
    lets the experiment add the new mechanisms without duplicating or changing
    the proven v10/v11 PCAF equations.
    """

    def __init__(
        self,
        base: nn.Module,
        *,
        delta_rank: int | None,
        span_max: int | None,
        phase: bool,
        args,
    ):
        super().__init__()
        self.base = base
        self.state_dim = int(base.state_dim)
        self.memory_dim = int(base.memory_dim)
        self.num_buckets = int(base.num_buckets)
        self.order = int(base.order)
        self.router_mode = str(base.router_mode)
        self.last_aux: Dict[str, float] = {}
        self.aux_weight = float(args.sidecar_aux_weight)

        self.delta = None
        if delta_rank is not None:
            self.delta = BlockDeltaSidecar(
                self.state_dim,
                rank=int(delta_rank),
                heads=args.delta_heads,
                block_size=args.delta_block,
                max_mix=args.sidecar_max_mix,
                gate_bias=args.sidecar_gate_bias,
                gate_grad_scale=args.gate_grad_scale,
            )
        self.span = None
        if span_max is not None:
            self.span = VerifiedSpanSidecar(
                self.state_dim,
                order=self.order,
                num_buckets=self.num_buckets,
                max_span=int(span_max),
                top_k=args.span_top_k,
                max_mix=args.sidecar_max_mix,
                gate_bias=args.sidecar_gate_bias,
                gate_grad_scale=args.gate_grad_scale,
            )
        self.phase = None
        if phase:
            self.phase = PhaseInterferenceSidecar(
                self.state_dim,
                bands=args.phase_bands,
                value_rank=args.phase_rank,
                max_mix=args.sidecar_max_mix,
                gate_bias=args.sidecar_gate_bias,
                salience_floor=args.salience_floor,
                gate_grad_scale=args.gate_grad_scale,
            )

    @property
    def enabled(self):
        return self.base.enabled

    @enabled.setter
    def enabled(self, value):
        self.base.enabled = value

    @property
    def distill_scale(self):
        return self.base.distill_scale

    @distill_scale.setter
    def distill_scale(self, value):
        self.base.distill_scale = value

    def _compute(self, states, logits, tokens, targets, compute_metrics):
        base_loss, base_primary, base_stats, base_nll = self.base._compute(
            states, logits, tokens, targets, compute_metrics
        )
        target_prob = torch.exp(-base_nll.float()).clamp(1e-8, 1.0)
        aux_loss = states.new_zeros((), dtype=torch.float32)
        aux: Dict[str, float] = dict(getattr(self.base, "last_aux", {}) or {})

        mechanisms = []
        if self.delta is not None:
            p, g, cov = self.delta(states)
            mechanisms.append(("delta", p, g, cov))
        if self.span is not None:
            p, g, cov = self.span(states, tokens, targets)
            mechanisms.append(("span", p, g, cov))
        if self.phase is not None:
            p, g, cov = self.phase(states, logits, targets)
            mechanisms.append(("phase", p, g, cov))

        for name, probs, gate, coverage in mechanisms:
            p_target = probs.gather(-1, targets[..., None]).squeeze(-1).clamp(1e-8, 1.0)
            target_prob = (1.0 - gate) * target_prob + gate * p_target
            covered = coverage > 0
            eligible = covered if name != "span" else (covered & (p_target > 1.0e-7))
            if self.training and self.aux_weight > 0 and bool(eligible.any()):
                aux_loss = aux_loss + self.aux_weight * (-torch.log(p_target[eligible])).mean()
            aux[f"{name}_gate"] = float(gate.detach().mean())
            aux[f"{name}_coverage"] = float(coverage.detach().mean())
            aux[f"{name}_target"] = float(p_target.detach().mean())

        token_nll = -torch.log(target_prob.clamp_min(1e-8))
        primary = token_nll.mean()
        loss = base_loss + (primary - base_primary) + aux_loss
        aux["sidecar_aux"] = float(aux_loss.detach())
        self.last_aux = aux

        stats = base_stats
        if compute_metrics and base_stats is not None:
            final_bpb = float(primary.detach() / LN2)
            denom = base_stats.param_bpb - base_stats.oracle_bpb
            capture = (
                (base_stats.param_bpb - final_bpb) / denom if denom > 1e-8 else 0.0
            )
            stats = v3.CacheStats(
                coverage=base_stats.coverage,
                gate=base_stats.gate,
                hit=base_stats.hit,
                cache_prob=base_stats.cache_prob,
                param_bpb=base_stats.param_bpb,
                oracle_bpb=base_stats.oracle_bpb,
                capture=float(capture),
                cache_win_rate=base_stats.cache_win_rate,
                gate_when_cache_wins=base_stats.gate_when_cache_wins,
                gate_when_cache_loses=base_stats.gate_when_cache_loses,
                gate_separation=base_stats.gate_separation,
            )
        return loss, primary.detach(), stats, token_nll

    def forward(self, states, logits, tokens, targets, compute_metrics=False):
        loss, primary, stats, _ = self._compute(
            states, logits, tokens, targets, compute_metrics
        )
        return loss, primary, stats

    def token_nll(self, states, logits, tokens, targets):
        return self._compute(states, logits, tokens, targets, False)[3]


def mode_for(arm: str) -> Tuple[int | None, int | None, bool, bool]:
    """Return delta_rank, span_max, phase, use_surface_multiview."""
    if arm == HYBRID_REF:
        return None, None, False, True
    if arm == AF_REF:
        return None, None, False, False
    if arm == "attentionfree_surface_multiview":
        return None, None, False, True
    delta_rank = 16 if "rank16" in arm else (8 if "delta" in arm else None)
    span_max = 2 if "span2" in arm else (4 if "span4" in arm or "span" in arm else None)
    phase = "phase" in arm
    return delta_rank, span_max, phase, True


def modify_model(model: nn.Module, arm: str, args) -> nn.Module:
    delta_rank, span_max, phase, use_surface = mode_for(arm)
    if use_surface:
        base = make_surface_multiview_cache(model.cache, args)
    else:
        base = make_v10_cache(model.cache, args)
    if delta_rank is None and span_max is None and not phase:
        model.cache = base
    else:
        model.cache = CloudMechanismCache(
            base,
            delta_rank=delta_rank,
            span_max=span_max,
            phase=phase,
            args=args,
        )
    return model


def raw_model(arm: str, hidden: int, args, device: torch.device) -> nn.Module:
    model = v3.FieldPCAFLM(
        base_arm_for(arm),
        args.dim,
        args.layers,
        args.heads,
        hidden,
        args.field_chunk,
        args.num_buckets,
    ).to(device)
    return modify_model(model, arm, args).to(device)


def selftest(args, device, shapes):
    arena.log("[selftest] finite backward, near-identity, delta/span/phase causality")
    x = torch.randint(0, VOCAB, (1, 41), device=device)
    y = torch.randint(0, VOCAB, (1, 41), device=device)
    for arm in ALL_ARMS:
        m = arena.build_model(arm, 0, shapes[arm], args, device)
        with arena.amp_ctx(device, args.amp):
            loss, primary, _ = m.loss_and_stats(x, y, False)
        loss.backward()
        finite = bool(torch.isfinite(loss)) and all(
            p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters()
        )
        arena.log(
            f"[selftest] {arm:<38} params={arena.nparams(m):,} "
            f"d={shapes[arm]['delta_pct']:+.3f}% loss={float(loss.detach()):.5f} "
            f"primary={float(primary):.5f} finite={finite} aux={getattr(m.cache,'last_aux',{})}"
        )
        if not finite:
            raise AssertionError(arm)
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # New wrappers must begin very close to their validated surface reference.
    for arm in ("delta_rank8", "verified_span4", "phase_interference", "delta8_span4"):
        arena.seed_all(7781)
        ref = v3.FieldPCAFLM(
            "softpatch_local_w256_conf", 32, 1, 4, 64, 8, 64
        ).to(device).eval()
        ref.cache = make_surface_multiview_cache(ref.cache, args).to(device)
        arena.seed_all(7781)
        exp = v3.FieldPCAFLM(
            "softpatch_local_w256_conf", 32, 1, 4, 64, 8, 64
        ).to(device).eval()
        exp = modify_model(exp, arm, args).to(device).eval()
        with torch.no_grad(), arena.amp_ctx(device, args.amp):
            l0 = ref.loss_and_stats(x, y, False)[1]
            l1 = exp.loss_and_stats(x, y, False)[1]
        err = float((l0 - l1).abs())
        arena.log(f"[selftest] {arm:<38} initial abs_loss={err:.3e}")
        if err > args.initial_identity_tol:
            raise AssertionError((arm, "initial identity", err))
        del ref, exp

    # Equal-shape suffix perturbation tests full token NLL causality.
    invasive = (
        "delta_rank8",
        "verified_span4",
        "phase_interference",
        "delta8_span4",
        "attentionfree_delta_span",
    )
    for arm in invasive:
        m = arena.build_model(arm, 11, shapes[arm], args, device).eval()
        full = torch.randint(0, VOCAB, (1, 49), device=device)
        full2 = full.clone()
        prefix_targets = 21
        full2[:, prefix_targets + 1 :] = torch.randint(
            0, VOCAB,
            full2[:, prefix_targets + 1 :].shape,
            device=device,
        )
        xa, ya = full[:, :-1], full[:, 1:]
        xb, yb = full2[:, :-1], full2[:, 1:]
        with torch.no_grad(), arena.amp_ctx(device, args.amp):
            sa, la = m.states_logits(xa)
            sb, lb = m.states_logits(xb)
            na = m.cache.token_nll(sa, la, xa, ya)[:, :prefix_targets]
            nb = m.cache.token_nll(sb, lb, xb, yb)[:, :prefix_targets]
        cerr = float((na - nb).abs().max())
        arena.log(f"[selftest] {arm:<38} causal_token_nll max_abs={cerr:.3e}")
        if cerr > 8e-4:
            raise AssertionError((arm, cerr))
        del m
    arena.log("[selftest] PASS")


def make_summary(stage_a, stage_b, stage_c, shapes, selected_main, selected_af, args):
    lines = [
        "FIELD DELTA / VERIFIED-SPAN / PHASE-INTERFERENCE ABLATION v12",
        "=" * 255,
        f"Protocol: WikiText-103 {args.data_frac:.1%} | train ctx={args.seq_len} | "
        f"eval={args.long_context}/{args.very_long_context} | bytes/update={args.batch_size*args.accum*args.seq_len:,}",
        "Hybrid reference: v10 selective episodic residual + v11 surface multiview reranking.",
        "Attention-free reference: validated v10 selective episodic residual; surface multiview is measured as an arm.",
        "New mechanisms: block-Delta fast weights, verified continuation copying, and complex phase interference.",
        "Portable exact Field math; promoted mechanisms must pass one canonical 50M consolidation bridge.",
        "",
        "PARAMETER MATCHING",
        f"{'arm':<40} {'group':>14} {'params':>13} {'d%':>8} {'ff':>6} {'AF':>4}",
    ]
    for arm in ALL_ARMS:
        s = shapes[arm]
        lines.append(
            f"{arm:<40} {group_for(arm):>14} {s['params']:>13,d} {s['delta_pct']:>+8.3f} "
            f"{s['hidden']:>6d} {('yes' if s['attention_free'] else 'no'):>4}"
        )

    def table(title, rows):
        lines.extend([
            "", title,
            f"{'arm':<40} {'seed':>6} {'BPB2K':>9} {'BPB8K':>9} {'dGroup':>9} {'BPB16K':>9} "
            f"{'oracle':>9} {'cap':>7} {'dGate':>7} {'sGate':>7} {'pGate':>7} {'B/s':>11} {'speed':>7}",
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
                f"{r['arm']:<40} {r['seed']:>6d} {r['bpb_train_context']:>9.5f} {r['bpb_8k']:>9.5f} "
                f"{r['bpb_8k']-ref['bpb_8k']:>+9.5f} {r['bpb_16k']:>9.5f} "
                f"{r['oracle_bpb_8k']:>9.5f} {r['capture_8k']:>7.3f} "
                f"{float(aux.get('delta_gate', 0.0)):>7.3f} "
                f"{float(aux.get('span_gate', 0.0)):>7.3f} "
                f"{float(aux.get('phase_gate', 0.0)):>7.3f} "
                f"{r['bytes_per_second']:>11,.0f} {r['bytes_per_second']/max(ref['bytes_per_second'],1):>7.2f}"
            )

    table("STAGE A — ALL CLOUD HYPOTHESES", stage_a)
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
            f"{arm:<40} vs {ref:<34} d8K={d8:+.5f}±{sd8:.5f} "
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
        lines.append("PROMOTE TO THE CANONICAL 50M CONSOLIDATION BRIDGE: " + ", ".join(promoted))
        lines.append("Only promoted mechanisms should be combined with v10/v11; failed arms are not added post hoc.")
    else:
        lines.append("NO CLOUD HYPOTHESIS CLEARED THE TWO-SEED PROMOTION GATE.")
        lines.append("Freeze selective_residual + surface_multiview and proceed to the canonical 50M bridge.")
    lines.append(f"selected_main={selected_main} | selected_attentionfree={selected_af}")
    lines.append("=" * 255)
    return "\n".join(lines) + "\n"


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=("all", "selftest", "summary"), default="all")
    p.add_argument("--outdir", default="./field_delta_span_phase_ablation_v12")
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
    p.add_argument("--delta-heads", type=int, default=4)
    p.add_argument("--delta-block", type=int, default=16)
    p.add_argument("--span-top-k", type=int, default=4)
    p.add_argument("--phase-bands", type=int, default=8)
    p.add_argument("--phase-rank", type=int, default=16)
    p.add_argument("--sidecar-max-mix", type=float, default=0.40)
    p.add_argument("--sidecar-gate-bias", type=float, default=1.0e-6)
    p.add_argument("--sidecar-aux-weight", type=float, default=0.01)
    p.add_argument("--gate-grad-scale", type=float, default=0.01)
    p.add_argument("--initial-identity-tol", type=float, default=0.0002)
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
    p.add_argument("--stage-a-min-speed", type=float, default=0.60)
    p.add_argument("--main-promotion-gain", type=float, default=0.008)
    p.add_argument("--af-promotion-gain", type=float, default=0.008)
    p.add_argument("--main-min-speed", type=float, default=0.90)
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


# Reuse the proven paired training/evaluation harness.
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
