#!/usr/bin/env python3
"""FIELD-FUSION SYSTEM SMOKE v21.

Focused H100 systems audit for the 300M Field-Fusion architecture.

The v20 quality screen proved the architectural direction.  This smoke does not
attempt another long language-model run.  It isolates and fixes the two biggest
systems problems before WikiText-103 100%:

1. Refresh-path overhead
   * combine Q + latent projections into one GEMM;
   * combine K + V expansion into one GEMM;
   * use effective_window=min(configured_window, sequence_length), avoiding the
     2048-token padded attention launched for 1024-token training sequences;
   * optionally replace the D x D dynamic residual gate with a channel gate.

2. Byte-era dense episodic PCAF on a 16,384-token vocabulary
   * v20's dense corrective cache materializes BxTxV tensors and a BxTxKxV
     retrieved probability bank;
   * v21 supplies a proper sparse selective successor mixture whose memory is
     O(B*T*K) beyond the normal logits rather than O(B*T*K*V).

The smoke compares:

  fusion_v20_dense          exact v20 reference
  fusion_v21_exact_dense    mathematically equivalent refresh optimization,
                            same dense v20 cache
  fusion_v21_fast_pcaf      exact refresh + validated target-only Fast PCAF
  fusion_v21_sparse_pcaf    exact refresh + scalable surprise-weighted PCAF
  fusion_v21_light_sparse   sparse PCAF + cheap per-channel attention gate
  transformer_flash_300m    same PyTorch Flash-SDPA baseline as v20

It reports:
  * fixed-token context sweep for train and full-path inference;
  * a 1024-token microbatch/accumulation sweep at constant 8192 tokens/update;
  * CE-only versus complete PCAF loss-path cost;
  * allocated/reserved/activation-like peak memory;
  * optional CUDA profiler tables and traces.

No result from this script is a final quality claim.  The exact-dense variant
must match v20 numerically.  The sparse variants are systems candidates and need
at least a short paired quality confirmation before a final canonical run.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import field_fusion_300m_screen_v20 as v20

base = v20.base
core = v20.core
Shape = core.Shape

HERE = Path(__file__).resolve().parent
EXPECTED_CANONICAL_SHA256 = v20.EXPECTED_CANONICAL_SHA256

V20_DENSE = "fusion_v20_dense"
V21_EXACT_DENSE = "fusion_v21_exact_dense"
V21_FAST = "fusion_v21_fast_pcaf"
V21_SPARSE = "fusion_v21_sparse_pcaf"
V21_LIGHT = "fusion_v21_light_sparse"
TRANSFORMER = "transformer_flash_300m"
MODEL_NAMES = (
    V20_DENSE,
    V21_EXACT_DENSE,
    V21_FAST,
    V21_SPARSE,
    V21_LIGHT,
    TRANSFORMER,
)
FIELD_NAMES = tuple(x for x in MODEL_NAMES if x != TRANSFORMER)


def log(x: object = "") -> None:
    print(str(x), flush=True)


def atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
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


def nparams(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class OptimizedLatentGQALandmarkAttention(nn.Module):
    """Exact v20 attention math with fewer projections and no oversized pad."""

    def __init__(self, dim: int, q_heads: int, kv_heads: int, latent_dim: int,
                 local_window: int, landmark_chunk: int) -> None:
        super().__init__()
        if dim % q_heads:
            raise ValueError("dim must divide q_heads")
        if q_heads % kv_heads:
            raise ValueError("q_heads must be divisible by kv_heads")
        self.dim = int(dim)
        self.q_heads = int(q_heads)
        self.kv_heads = int(kv_heads)
        self.head_dim = dim // q_heads
        self.latent_dim = int(latent_dim)
        self.local_window = int(local_window)
        self.landmark_chunk = int(landmark_chunk)

        # Exact concatenations of the two separate v20 projections.
        self.q_latent_proj = nn.Linear(
            dim, q_heads * self.head_dim + latent_dim, bias=False
        )
        self.kv_up = nn.Linear(
            latent_dim, 2 * kv_heads * self.head_dim, bias=False
        )
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.global_mix_logit = nn.Parameter(torch.tensor(-0.7))

    @property
    def q_width(self) -> int:
        return self.q_heads * self.head_dim

    def _reshape_q(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.q_heads, self.head_dim).transpose(1, 2)

    def _reshape_kv(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)

    def _local_attention(self, q: torch.Tensor, k: torch.Tensor,
                         val: torch.Tensor) -> torch.Tensor:
        batch, q_heads, length, head_dim = q.shape
        window = min(self.local_window, length)
        if window <= 0:
            raise ValueError("empty sequence")

        # Most important v20 fix: a configured 2048 window no longer pads a
        # 1024-token training sequence to 2048 and launches 4x the score work.
        if length <= window:
            return v20.sdpa_gqa(q, k, val, causal=True)

        pad = (-length) % window
        if pad:
            q = F.pad(q, (0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, pad))
            val = F.pad(val, (0, 0, 0, pad))
        groups = q.shape[2] // window
        qg = q.reshape(batch, q_heads, groups, window, head_dim).permute(
            0, 2, 1, 3, 4
        ).reshape(batch * groups, q_heads, window, head_dim)
        kg = k.reshape(batch, self.kv_heads, groups, window, head_dim).permute(
            0, 2, 1, 3, 4
        ).reshape(batch * groups, self.kv_heads, window, head_dim)
        vg = val.reshape(batch, self.kv_heads, groups, window, head_dim).permute(
            0, 2, 1, 3, 4
        ).reshape(batch * groups, self.kv_heads, window, head_dim)
        yg = v20.sdpa_gqa(qg, kg, vg, causal=True)
        y = yg.reshape(batch, groups, q_heads, window, head_dim).permute(
            0, 2, 1, 3, 4
        ).reshape(batch, q_heads, groups * window, head_dim)
        return y[:, :, :length]

    def _landmark_attention(self, q: torch.Tensor, latent: torch.Tensor,
                            cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, _, length, _ = q.shape
        chunk = self.landmark_chunk
        full_chunks = length // chunk
        if full_chunks == 0:
            return q.new_zeros((b, self.q_heads, length, self.head_dim))

        landmark_latent = latent[:, : full_chunks * chunk].reshape(
            b, full_chunks, chunk, self.latent_dim
        ).mean(dim=2)
        kup = self.kv_up(landmark_latent)
        lk_raw, lv_raw = kup.split(
            self.kv_heads * self.head_dim, dim=-1
        )
        lk = self._reshape_kv(lk_raw)
        lv = self._reshape_kv(lv_raw)
        positions = torch.arange(
            chunk - 1, full_chunks * chunk, chunk,
            device=q.device, dtype=torch.long,
        )
        lk = v20.apply_rope(lk, cos[:, :, positions], sin[:, :, positions])

        null_k = lk.new_zeros((b, self.kv_heads, 1, self.head_dim))
        null_v = lv.new_zeros((b, self.kv_heads, 1, self.head_dim))
        lk = torch.cat((null_k, lk), dim=2)
        lv = torch.cat((null_v, lv), dim=2)
        token_chunks = torch.arange(length, device=q.device) // chunk
        landmark_ids = torch.arange(full_chunks, device=q.device)
        allowed_real = landmark_ids[None, :] < token_chunks[:, None]
        allowed = torch.cat((
            torch.ones((length, 1), device=q.device, dtype=torch.bool),
            allowed_real,
        ), dim=1)
        return v20.sdpa_gqa(
            q, lk, lv, causal=False, attn_mask=allowed[None, None]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, length, _ = x.shape
        qlatent = self.q_latent_proj(x)
        qraw, latent = qlatent.split((self.q_width, self.latent_dim), dim=-1)
        kup = self.kv_up(latent)
        kraw, vraw = kup.split(self.kv_heads * self.head_dim, dim=-1)
        q = self._reshape_q(qraw)
        k = self._reshape_kv(kraw)
        val = self._reshape_kv(vraw)
        cos, sin = v20.rope_cos_sin(x.device, x.dtype, length, self.head_dim)
        q = v20.apply_rope(q, cos, sin)
        k = v20.apply_rope(k, cos, sin)
        local = self._local_attention(q, k, val)
        global_summary = self._landmark_attention(q, latent, cos, sin)
        mix = torch.sigmoid(self.global_mix_logit).to(dtype=local.dtype)
        y = local + mix * global_summary
        y = y.transpose(1, 2).reshape(b, length, self.dim)
        return self.out_proj(y)


class FusionRefreshBlockV21(nn.Module):
    def __init__(self, dim: int, q_heads: int, kv_heads: int, latent_dim: int,
                 local_window: int, landmark_chunk: int, ff_hidden: int,
                 *, light_gate: bool) -> None:
        super().__init__()
        self.norm1 = v20.NativeRMSNorm(dim)
        self.attn = OptimizedLatentGQALandmarkAttention(
            dim, q_heads, kv_heads, latent_dim, local_window, landmark_chunk
        )
        self.light_gate = bool(light_gate)
        init = math.log(0.25 / 0.75)
        if self.light_gate:
            self.residual_gate_logit = nn.Parameter(torch.full((dim,), init))
            self.residual_gate = None
        else:
            self.residual_gate = nn.Linear(dim, dim, bias=True)
            nn.init.zeros_(self.residual_gate.weight)
            nn.init.constant_(self.residual_gate.bias, init)
            self.register_parameter("residual_gate_logit", None)
        self.norm2 = v20.NativeRMSNorm(dim)
        self.ff = v20.PackedSwiGLU(dim, ff_hidden)
        self.window = int(local_window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm1(x)
        a = self.attn(z)
        if self.light_gate:
            gate = torch.sigmoid(self.residual_gate_logit).to(z.dtype)
        else:
            assert self.residual_gate is not None
            gate = torch.sigmoid(self.residual_gate(z))
        x = x + gate * a
        return x + self.ff(self.norm2(x))


class SparseSelectivePCAF(nn.Module):
    """Vocabulary-scalable successor memory.

    This is a proper convex distribution, not a target-only unnormalised hack.
    The cache branch mixes two sparse successor distributions:

      q_recent  : normal retrieved successor weights
      q_surprise: the same candidates reweighted by how surprising their
                  observed successor was to the past parametric model

    Only target probabilities are needed for NLL, so no BxTxV cache tensor and
    no BxTxKxV past-probability bank are materialized.  Surprise statistics are
    detached, matching the selective-write semantics of the dense v10 cache.
    """

    def __init__(self, base_cache: nn.Module, optmod, *, salience_floor: float,
                 surprise_mix_init: float = 0.25) -> None:
        super().__init__()
        self.base = base_cache
        self.state_dim = int(base_cache.state_dim)
        self.memory_dim = int(base_cache.memory_dim)
        self.num_buckets = int(base_cache.num_buckets)
        self.order = int(base_cache.order)
        self.top_k = int(base_cache.top_k)
        self.router_mode = str(base_cache.router_mode)
        self.salience_floor = float(salience_floor)
        self.surprise_mix_logit = nn.Parameter(torch.tensor(math.log(
            surprise_mix_init / max(1.0 - surprise_mix_init, 1e-8)
        )))
        self.salience_strength_raw = nn.Parameter(torch.zeros(()))
        self.last_aux: Dict[str, torch.Tensor] = {}
        self._optmod = optmod
        self._v3 = base_cache._v3

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

    @property
    def distill_temperature(self) -> float:
        return float(self.base.distill_temperature)

    @property
    def distill_weight(self) -> float:
        return float(self.base.distill_weight)

    def _candidate_features(
        self,
        scores: torch.Tensor,
        weights: torch.Tensor,
        valid: torch.Tensor,
        cand_tokens: torch.Tensor,
        recency: torch.Tensor,
        flat_logits: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        """Equivalent confidence features without copying all active logit rows."""
        k = self.top_k
        masked = scores.masked_fill(~valid, -1.0e9)
        top2 = torch.topk(masked, k=min(2, k), dim=-1).values
        top1 = top2[..., 0]
        count = valid.float().sum(-1)
        if top2.shape[-1] > 1:
            margin = torch.where(
                count >= 2, top2[..., 0] - top2[..., 1],
                torch.zeros_like(top1),
            )
        else:
            margin = torch.zeros_like(top1)
        cand_ent = -(weights * torch.log(weights.clamp_min(1e-8))).sum(-1)
        cand_ent = torch.where(
            count > 1,
            cand_ent / torch.log(count.clamp_min(2.0)),
            torch.zeros_like(cand_ent),
        )
        wrec = (weights * recency).sum(-1)
        cl = cand_tokens.long()
        same = cl[..., :, None] == cl[..., None, :]
        token_mass = (same.float() * weights[..., None, :]).sum(-1)
        earlier = torch.tril(
            torch.ones((k, k), device=valid.device, dtype=torch.bool), diagonal=-1
        )
        unique = valid & ~(same & earlier).any(-1)
        unique_mass = token_mass.masked_fill(~unique, 0.0)
        mass2, massidx = torch.topk(unique_mass, k=min(2, k), dim=-1)
        cache_conf = mass2[..., 0]
        cache_second = mass2[..., 1] if mass2.shape[-1] > 1 else torch.zeros_like(cache_conf)
        cache_margin = cache_conf - cache_second
        cache_ent = -(unique_mass * torch.log(unique_mass.clamp_min(1e-8))).sum(-1)
        cache_ent = cache_ent / math.log(max(k, 2))
        cache_top = cl.gather(-1, massidx[..., :1]).squeeze(-1)

        # Parametric confidence is diagnostic/router input only.  Compute it
        # under no_grad in BF16, avoiding a persistent FP32 B*T*V copy.
        with torch.no_grad():
            topv, ptok = torch.topk(flat_logits, 2, dim=-1)
            logz = torch.logsumexp(flat_logits, dim=-1)
            topv = topv.float()
            logz = logz.float()
            pconf_all = torch.exp(topv[:, 0] - logz)
            psecond_all = torch.exp(topv[:, 1] - logz)
            pmargin_all = pconf_all - psecond_all

        active_idx = active.nonzero(as_tuple=False).squeeze(-1)
        cache_top_a = cache_top.reshape(-1)[active]
        ptoken_a = ptok[:, 0].reshape(-1)[active]
        # Paired scalar indexing; does not materialize active x vocabulary.
        p_cache_top = torch.exp(
            flat_logits[active_idx, cache_top_a].float() - logz[active_idx]
        )
        cl_a = cl.reshape(-1, k)[active]
        w_a = weights.reshape(-1, k)[active]
        cache_mass_ptop = (w_a * (cl_a == ptoken_a[:, None]).float()).sum(-1)
        cache_conf_a = cache_conf.reshape(-1)[active]
        pconf_a = pconf_all[active_idx]
        agree = (cache_top_a == ptoken_a).float()
        delta = cache_conf_a - pconf_a

        f = torch.stack((
            torch.tanh(top1.reshape(-1)[active]),
            torch.tanh(margin.reshape(-1)[active]),
            cand_ent.reshape(-1)[active].clamp(0, 1),
            (count.reshape(-1)[active] / float(k)).clamp(0, 1),
            wrec.reshape(-1)[active].clamp(0, 1),
            cache_conf_a.clamp(0, 1),
            cache_margin.reshape(-1)[active].clamp(0, 1),
            cache_ent.reshape(-1)[active].clamp(0, 1),
            pconf_a.clamp(0, 1),
            pmargin_all[active_idx].clamp(0, 1),
            agree,
            p_cache_top.clamp(0, 1),
            cache_mass_ptop.clamp(0, 1),
            delta.clamp(-1, 1),
            delta.abs().clamp(0, 1),
        ), dim=-1)
        return f.detach()

    def _compute(self, states: torch.Tensor, logits: torch.Tensor,
                 tokens: torch.Tensor, targets: torch.Tensor,
                 compute_metrics: bool):
        del compute_metrics
        b, t, _ = states.shape
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = targets.reshape(-1)
        # Let the fused CE implementation choose its stable accumulator.  The
        # returned token NLL is promoted for probability arithmetic.
        param_nll = F.cross_entropy(
            flat_logits, flat_targets, reduction="none"
        ).float()
        param_target = torch.exp(-param_nll)
        if not self.enabled:
            primary = param_nll.mean()
            return primary, primary.detach(), None, param_nll.view_as(targets)

        idx = self._optmod.causal_recent_candidates_i32(
            tokens, self.order, self.num_buckets, self.top_k, self._v3
        )
        valid = idx >= 0
        safe = idx.clamp_min(0)
        has = valid.any(-1)
        batch_idx = torch.arange(b, device=states.device)[:, None, None]

        proj = self._v3.normalize_rows(
            F.linear(states.float(), self.base.shared_weight.float())
        )
        q = proj[:, :, None, :]
        ck = proj[batch_idx, safe]
        scores = (ck * q).sum(-1) * (self.memory_dim ** -0.5)
        recency = safe.float() / max(float(t - 1), 1.0)
        scores = scores + self.base.recency_scale.float() * recency
        cand_tokens = targets[batch_idx, safe]

        # Past-model confidence is used only to reweight sparse candidates.
        # No B*T*K*V retrieval is performed.
        with torch.no_grad():
            logz = torch.logsumexp(logits, dim=-1).float()
            past_observed_logit = logits[batch_idx, safe, cand_tokens].float()
            past_logz = logz[batch_idx, safe]
            past_true = torch.exp(past_observed_logit - past_logz).clamp(0.0, 1.0)
            surprise = torch.sqrt((1.0 - past_true).clamp(0.0, 1.0))

        alpha = F.softplus(self.salience_strength_raw.float()).clamp(0.0, 4.0)
        salience = self.salience_floor + (1.0 - self.salience_floor) * surprise
        scores = scores + alpha * torch.log(salience.clamp_min(1e-4))
        scores = scores.masked_fill(~valid, -1.0e9)
        weights = torch.softmax(scores.float(), dim=-1) * valid.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)

        target_match = (cand_tokens == targets[:, :, None]).float()
        target_cache = (weights * target_match).sum(-1)
        surprise_weights = weights * salience
        surprise_weights = surprise_weights / surprise_weights.sum(
            -1, keepdim=True
        ).clamp_min(1e-6)
        surprise_target = (surprise_weights * target_match).sum(-1)
        surprise_mix = torch.sigmoid(self.surprise_mix_logit).float()
        sparse_cache_target = (
            (1.0 - surprise_mix) * target_cache
            + surprise_mix * surprise_target
        )

        active = has.reshape(-1)
        state_logit = self.base.state_gate(states).float().squeeze(-1)
        features = self._candidate_features(
            scores, weights, valid, cand_tokens, recency, flat_logits, active
        )
        route = self.base.router(features)
        flat_state_logit = state_logit.reshape(-1)
        gate_flat = torch.zeros_like(flat_state_logit)
        if self.router_mode == "v5":
            gate_logit_active = flat_state_logit[active] + route[:, 0]
        else:
            cache_conf = features[:, 5].clamp(1e-4, 1.0 - 1e-4)
            param_conf = features[:, 8].clamp(1e-4, 1.0 - 1e-4)
            evidence = torch.logit(cache_conf) - torch.logit(param_conf)
            evidence = evidence + 1.25 * features[:, 6] - 0.50 * features[:, 7]
            evidence = evidence + 0.35 * features[:, 10] + 0.25 * features[:, 3]
            state_term = (
                0.0 if self.router_mode == "confidence_nostate"
                else flat_state_logit[active]
            )
            gate_logit_active = (
                state_term + route[:, 0]
                + self.base.evidence_gain * evidence
                + self.base.evidence_bias
            )
        ga = torch.sigmoid(gate_logit_active).clamp(1e-5, 1.0 - 1e-5)
        gate_flat[active] = ga
        mixed = (
            (1.0 - gate_flat) * param_target
            + gate_flat * sparse_cache_target.reshape(-1)
        )
        token_nll = -torch.log(mixed.clamp_min(1e-8))
        primary = token_nll.mean()
        loss = primary

        if self.training and self.distill_scale > 0 and gate_logit_active.numel() > 0:
            pa = param_target[active]
            ca = sparse_cache_target.reshape(-1)[active]
            log_adv = (
                torch.log(ca.detach().clamp_min(1e-8))
                - torch.log(pa.detach().clamp_min(1e-8))
            )
            teacher = torch.sigmoid(log_adv / self.distill_temperature)
            weight = torch.tanh(log_adv.abs())
            gate_logit = torch.logit(gate_flat[active].clamp(1e-5, 1.0 - 1e-5))
            aux = (
                F.binary_cross_entropy_with_logits(
                    gate_logit, teacher, reduction="none"
                ) * weight
            ).sum() / weight.sum().clamp_min(1.0)
            loss = primary + self.distill_weight * float(self.distill_scale) * aux
            self.last_aux = {
                "distill": aux.detach(),
                "teacher": teacher.mean().detach(),
                "cache_win": (log_adv > 0).float().mean().detach(),
                "surprise_mix": surprise_mix.detach(),
                "surprise": surprise.mean().detach(),
            }
        return loss, primary.detach(), None, token_nll.view_as(targets)

    def forward(self, states, logits, tokens, targets, compute_metrics=False):
        loss, primary, stats, _ = self._compute(
            states, logits, tokens, targets, compute_metrics
        )
        return loss, primary, stats

    def token_nll(self, states, logits, tokens, targets):
        return self._compute(states, logits, tokens, targets, False)[3]


class FieldTokenSystemV21(nn.Module):
    def __init__(self, *, vocab: int, dim: int, layers: int, ff_hidden: int,
                 q_heads: int, kv_heads: int, latent_dim: int,
                 refresh_windows: Sequence[int], landmark_chunk: int,
                 field_chunk: int, triton_block_c: int, triton_chunk_t: int,
                 num_buckets: int, v3, canonical, light_gate: bool) -> None:
        super().__init__()
        if layers != len(refresh_windows) * 6:
            raise ValueError("layers must equal 6 * number of refresh windows")
        self.fusion = True
        self.emb = nn.Embedding(vocab, dim)
        modules: List[nn.Module] = []
        for window in refresh_windows:
            for _ in range(5):
                modules.append(canonical.FieldBlock(
                    dim, "triton", field_chunk, triton_block_c,
                    triton_chunk_t, ff_hidden,
                ))
            modules.append(FusionRefreshBlockV21(
                dim, q_heads, kv_heads, latent_dim, int(window),
                landmark_chunk, ff_hidden, light_gate=light_gate,
            ))
        self.blocks = nn.ModuleList(modules)
        self.final_norm = v3.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.cache = v3.SuccessorCacheV5(
            dim, memory_dim=64, num_buckets=num_buckets, order=4, top_k=4,
            router_mode="confidence",
        )
        self.softpatch = v3.BoundaryStateMixer(dim, rank=64, learned=True)
        self.patch_position = max(0, layers // 2 - 1)
        self.locals = nn.ModuleDict()
        self.multiscales = nn.ModuleDict()
        self._patch_aux: Optional[torch.Tensor] = None

    @property
    def attention_free(self) -> bool:
        return False

    def states_logits(self, tokens: torch.Tensor):
        x = self.emb(tokens)
        self._patch_aux = x.new_zeros(())
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == self.patch_position and self.softpatch is not None:
                x = self.softpatch(x, tokens)
                self._patch_aux = self.softpatch.last_aux
        return x, self.lm_head(self.final_norm(x))

    def loss_and_stats(self, tokens: torch.Tensor, targets: torch.Tensor,
                       compute_metrics: bool = False):
        states, logits = self.states_logits(tokens)
        loss, primary, stats = self.cache(
            states, logits, tokens, targets, compute_metrics
        )
        if self.training and self._patch_aux is not None:
            loss = loss + self._patch_aux
        return loss, primary, stats


def tied_embedding_init(model: nn.Module, seed: int, std: float = 0.02) -> None:
    v20.tied_embedding_init(model, seed, std)


def install_field_optimizations(model: FieldTokenSystemV21, args, arena, v3,
                                optmod, device: torch.device,
                                cache_mode: str) -> nn.Module:
    optmod.replace_softpatch(model, v3)
    optmod.replace_cache(model, v3, i32=True)
    model.cache.FEATURE_DIM = int(v3.SuccessorCacheV5.FEATURE_DIM)
    arena._install_cloud_fast_route(optmod)
    if cache_mode == "dense":
        model.cache = arena.cloud.make_v10_cache(model.cache, args).to(device)
    elif cache_mode == "fast":
        model.cache = model.cache.to(device)
    elif cache_mode == "sparse":
        model.cache = SparseSelectivePCAF(
            model.cache, optmod,
            salience_floor=args.salience_floor,
            surprise_mix_init=args.sparse_surprise_mix,
        ).to(device)
    else:
        raise ValueError(cache_mode)
    return model


def build_model(name: str, shape: Shape, args, arena, v3, canonical, bridge,
                optmod, epi, judge, device: torch.device) -> nn.Module:
    del bridge, epi, judge
    core.seed_all(args.model_seed)
    if name == TRANSFORMER:
        model = v20.StrongFlashTransformer300M(
            args.vocab_size, shape.dim, shape.heads, shape.layers, shape.ff_hidden
        ).to(device)
    elif name == V20_DENSE:
        # Build exactly the v20 reference under its original name.
        model = v20.FieldTokenSystem300M(
            fusion=True, vocab=args.vocab_size, dim=shape.dim,
            layers=shape.layers, ff_hidden=shape.ff_hidden,
            q_heads=args.fusion_q_heads, kv_heads=args.fusion_kv_heads,
            latent_dim=args.fusion_latent_dim,
            refresh_windows=args.refresh_windows,
            landmark_chunk=args.landmark_chunk,
            field_chunk=args.field_chunk,
            triton_block_c=args.triton_block_c,
            triton_chunk_t=args.triton_chunk_t,
            num_buckets=args.num_buckets,
            v3=v3, canonical=canonical,
        ).to(device)
        model = v20.optimize_field_model(model, args, arena, v3, optmod, device)
    elif name in {V21_EXACT_DENSE, V21_FAST, V21_SPARSE, V21_LIGHT}:
        model = FieldTokenSystemV21(
            vocab=args.vocab_size, dim=shape.dim, layers=shape.layers,
            ff_hidden=shape.ff_hidden, q_heads=args.fusion_q_heads,
            kv_heads=args.fusion_kv_heads,
            latent_dim=args.fusion_latent_dim,
            refresh_windows=args.refresh_windows,
            landmark_chunk=args.landmark_chunk,
            field_chunk=args.field_chunk,
            triton_block_c=args.triton_block_c,
            triton_chunk_t=args.triton_chunk_t,
            num_buckets=args.num_buckets,
            v3=v3, canonical=canonical,
            light_gate=(name == V21_LIGHT),
        ).to(device)
        mode = "dense" if name == V21_EXACT_DENSE else (
            "fast" if name == V21_FAST else "sparse"
        )
        model = install_field_optimizations(
            model, args, arena, v3, optmod, device, mode
        )
    else:
        raise KeyError(name)
    tied_embedding_init(model, args.embedding_seed)
    return model


def count_model(name: str, hidden: int, args, arena, v3, canonical, bridge,
                optmod, epi, judge) -> int:
    shape = Shape(name, 0, args.dim, args.layers, args.heads, hidden)
    model = build_model(
        name, shape, args, arena, v3, canonical, bridge, optmod, epi, judge,
        torch.device("cpu"),
    )
    value = nparams(model)
    del model
    gc.collect()
    return value


def solve_shape(name: str, args, arena, v3, canonical, bridge, optmod,
                epi, judge) -> Shape:
    probe = args.min_ff_hidden
    p0 = count_model(name, probe, args, arena, v3, canonical, bridge,
                     optmod, epi, judge)
    slope = 3 * args.dim * args.layers
    intercept = p0 - slope * probe
    raw = (args.target_params - intercept) / max(slope, 1)
    aligned = int(round(raw / args.ff_multiple) * args.ff_multiple)
    candidates = sorted(set(
        max(args.min_ff_hidden, min(args.max_ff_hidden,
            aligned + d * args.ff_multiple)) for d in range(-4, 5)
    ))
    rows = [(h, int(intercept + slope * h)) for h in candidates]
    hidden, estimated = min(rows, key=lambda hp: abs(hp[1] - args.target_params))
    # Count once to protect against any non-linear hidden-size parameter path.
    actual = count_model(name, hidden, args, arena, v3, canonical, bridge,
                         optmod, epi, judge)
    return Shape(name, actual, args.dim, args.layers, args.heads, hidden)


def set_distill(model: nn.Module, value: float = 1.0) -> None:
    cache = getattr(model, "cache", None)
    seen = set()
    while cache is not None and id(cache) not in seen:
        seen.add(id(cache))
        if hasattr(cache, "distill_scale"):
            try:
                cache.distill_scale = float(value)
            except Exception:
                pass
        cache = getattr(cache, "base", None)


def amp_ctx(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_optimizer(model: nn.Module, args):
    return core.make_optimizer(model, args.lr, args.weight_decay)


def fixed_batch(data: torch.Tensor, batch: int, context: int,
                seed: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    return core.batch_for_step(data, batch, context, seed, 1, 0, device)


def loss_call(name: str, model: nn.Module, x: torch.Tensor, y: torch.Tensor,
              *, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if name == TRANSFORMER:
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
        )
        return loss, loss.detach()
    if mode == "ce":
        states, logits = model.states_logits(x)
        del states
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
        )
        return loss, loss.detach()
    return model.loss_and_stats(x, y, compute_metrics=False)[:2]


@dataclass
class BenchRow:
    model: str
    kind: str
    context: int
    batch: int
    accum: int
    tokens_per_update: int
    status: str
    tokens_per_second: Optional[float]
    bytes_per_second_est: Optional[float]
    update_ms: Optional[float]
    baseline_alloc_gib: Optional[float]
    peak_alloc_gib: Optional[float]
    peak_reserved_gib: Optional[float]
    activation_like_gib: Optional[float]
    error: str = ""


def train_update_benchmark(
    name: str,
    shape: Shape,
    args,
    deps,
    data: torch.Tensor,
    device: torch.device,
    *,
    context: int,
    batch: int,
    accum: int,
    loss_mode: str,
    warmup: int,
    steps: int,
    bytes_per_token: float,
) -> BenchRow:
    arena, v3, canonical, bridge, optmod, epi, judge = deps
    model = build_model(name, shape, args, arena, v3, canonical, bridge,
                        optmod, epi, judge, device).train()
    set_distill(model, 1.0)
    optimizer = make_optimizer(model, args)
    micro_batches = [
        fixed_batch(data, batch, context,
                    args.eval_seed + context * 101 + m * 1009, device)
        for m in range(accum)
    ]

    def update_once() -> None:
        optimizer.zero_grad(set_to_none=True)
        for x, y in micro_batches:
            with amp_ctx(device, args.amp):
                loss, _ = loss_call(name, model, x, y, mode=loss_mode)
                loss = loss / accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

    status, error = "ok", ""
    tps = bps = update_ms = baseline = peak = reserved = activation = None
    try:
        for _ in range(warmup):
            update_once()
        sync(device)
        baseline = torch.cuda.memory_allocated(device) / 2**30
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(steps):
            update_once()
        sync(device)
        elapsed = time.perf_counter() - started
        tokens = steps * batch * accum * context
        tps = tokens / max(elapsed, 1e-9)
        bps = tps * bytes_per_token
        update_ms = elapsed * 1000.0 / steps
        peak = torch.cuda.max_memory_allocated(device) / 2**30
        reserved = torch.cuda.max_memory_reserved(device) / 2**30
        activation = max(0.0, peak - baseline)
    except torch.cuda.OutOfMemoryError as exc:
        status, error = "oom", str(exc)
    except Exception as exc:
        status, error = "error", repr(exc)
    row = BenchRow(
        model=name, kind=f"train_{loss_mode}", context=context,
        batch=batch, accum=accum,
        tokens_per_update=batch * accum * context,
        status=status, tokens_per_second=tps,
        bytes_per_second_est=bps, update_ms=update_ms,
        baseline_alloc_gib=baseline, peak_alloc_gib=peak,
        peak_reserved_gib=reserved, activation_like_gib=activation,
        error=error,
    )
    del optimizer, model, micro_batches
    clear_cuda()
    return row


def inference_benchmark(
    name: str,
    shape: Shape,
    args,
    deps,
    data: torch.Tensor,
    device: torch.device,
    *,
    context: int,
    batch: int,
    loss_mode: str,
    warmup: int,
    steps: int,
    bytes_per_token: float,
) -> BenchRow:
    arena, v3, canonical, bridge, optmod, epi, judge = deps
    model = build_model(name, shape, args, arena, v3, canonical, bridge,
                        optmod, epi, judge, device).eval()
    set_distill(model, 1.0)
    x, y = fixed_batch(data, batch, context,
                       args.eval_seed + context * 313, device)

    def call_once() -> None:
        with torch.inference_mode(), amp_ctx(device, args.amp):
            loss, _ = loss_call(name, model, x, y, mode=loss_mode)
            # Keep the reduction live without synchronizing it to the CPU.
            _ = loss + 0.0

    status, error = "ok", ""
    tps = bps = latency = baseline = peak = reserved = activation = None
    try:
        for _ in range(warmup):
            call_once()
        sync(device)
        baseline = torch.cuda.memory_allocated(device) / 2**30
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(steps):
            call_once()
        sync(device)
        elapsed = time.perf_counter() - started
        tokens = steps * batch * context
        tps = tokens / max(elapsed, 1e-9)
        bps = tps * bytes_per_token
        latency = elapsed * 1000.0 / steps
        peak = torch.cuda.max_memory_allocated(device) / 2**30
        reserved = torch.cuda.max_memory_reserved(device) / 2**30
        activation = max(0.0, peak - baseline)
    except torch.cuda.OutOfMemoryError as exc:
        status, error = "oom", str(exc)
    except Exception as exc:
        status, error = "error", repr(exc)
    row = BenchRow(
        model=name, kind=f"infer_{loss_mode}", context=context,
        batch=batch, accum=1, tokens_per_update=batch * context,
        status=status, tokens_per_second=tps,
        bytes_per_second_est=bps, update_ms=latency,
        baseline_alloc_gib=baseline, peak_alloc_gib=peak,
        peak_reserved_gib=reserved, activation_like_gib=activation,
        error=error,
    )
    del model, x, y
    clear_cuda()
    return row


def copy_attention_v20_to_v21(src: v20.LatentGQALandmarkAttention,
                              dst: OptimizedLatentGQALandmarkAttention) -> None:
    with torch.no_grad():
        dst.q_latent_proj.weight.copy_(torch.cat((
            src.q_proj.weight, src.kv_down.weight
        ), dim=0))
        dst.kv_up.weight.copy_(torch.cat((
            src.k_up.weight, src.v_up.weight
        ), dim=0))
        dst.out_proj.weight.copy_(src.out_proj.weight)
        dst.global_mix_logit.copy_(src.global_mix_logit)


def exact_refresh_selftest(args, device: torch.device) -> Dict[str, float]:
    torch.manual_seed(123)
    dim = 128
    qh, kvh, latent = 4, 2, 32
    rows = {}
    for length, window in ((97, 64), (128, 256), (257, 512)):
        old = v20.LatentGQALandmarkAttention(
            dim, qh, kvh, latent, window, 64
        ).to(device=device, dtype=torch.bfloat16).eval()
        new = OptimizedLatentGQALandmarkAttention(
            dim, qh, kvh, latent, window, 64
        ).to(device=device, dtype=torch.bfloat16).eval()
        copy_attention_v20_to_v21(old, new)
        x = torch.randn(2, length, dim, device=device, dtype=torch.bfloat16)
        with torch.inference_mode():
            yo = old(x)
            yn = new(x)
        diff = float((yo.float() - yn.float()).abs().max())
        rows[f"T{length}_W{window}"] = diff
        if diff > args.exact_tol:
            raise AssertionError(
                f"exact refresh mismatch T={length} W={window}: {diff}"
            )
        del old, new, x, yo, yn
        clear_cuda()
    return rows


def profile_one(name: str, shape: Shape, args, deps, data: torch.Tensor,
                device: torch.device, outdir: Path) -> Dict[str, str]:
    from torch.profiler import ProfilerActivity, profile

    arena, v3, canonical, bridge, optmod, epi, judge = deps
    model = build_model(name, shape, args, arena, v3, canonical, bridge,
                        optmod, epi, judge, device).train()
    set_distill(model, 1.0)
    optimizer = make_optimizer(model, args)
    x, y = fixed_batch(
        data, args.profile_batch, args.profile_context,
        args.eval_seed + 424242, device,
    )

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with amp_ctx(device, args.amp):
            loss, _ = loss_call(name, model, x, y, mode="full")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

    step()
    sync(device)
    trace = outdir / f"profile_{name}.json"
    table_path = outdir / f"profile_{name}.txt"
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False, profile_memory=True, with_stack=False,
    ) as prof:
        step()
    sync(device)
    prof.export_chrome_trace(str(trace))
    table = prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=args.profile_rows
    )
    atomic_text(table_path, table + "\n")
    del model, optimizer, x, y
    clear_cuda()
    return {"trace": str(trace), "table": str(table_path)}


def fnum(x: Optional[float], digits: int = 0) -> str:
    if x is None:
        return "-"
    return f"{x:,.{digits}f}"


def render_summary(args, shapes: Mapping[str, Shape], exact: Mapping[str, float],
                   context_rows: Sequence[BenchRow], update_rows: Sequence[BenchRow],
                   breakdown_rows: Sequence[BenchRow], profiles: Mapping[str, object],
                   bytes_per_token: float) -> str:
    lines: List[str] = []
    lines.append("=" * 196)
    lines.append("FIELD-FUSION SYSTEM SMOKE v21 — 300M H100 OPTIMIZATION AUDIT")
    lines.append("=" * 196)
    lines.append(
        "Purpose: isolate v20 protocol overhead, refresh overhead, and the byte-era dense PCAF memory path before WikiText-103 100%."
    )
    lines.append(
        f"fixed bytes/token estimate={bytes_per_token:.4f} | update target={args.tokens_per_update:,} tokens | BF16"
    )
    lines.append("")
    lines.append("MODEL SHAPES")
    lines.append(f"{'model':30s} {'params':>14s} {'dTarget%':>10s} {'ff':>7s}")
    for name in args.models:
        s = shapes[name]
        delta = 100.0 * (s.params - args.target_params) / args.target_params
        lines.append(f"{name:30s} {s.params:14,d} {delta:+10.3f} {s.ff_hidden:7d}")
    lines.append("")
    lines.append("EXACT REFRESH SELFTEST — v20 separate projections vs v21 packed projections/effective window")
    for key, value in exact.items():
        lines.append(f"{key:18s} max_abs={value:.6g}")

    lines.append("")
    lines.append("FIXED-TOKEN CONTEXT SWEEP — COMPLETE LOSS PATH")
    lines.append(
        f"{'kind':12s} {'model':30s} {'ctx':>6s} {'batch':>5s} {'tok/s':>12s} {'MB/s est':>11s} {'ms':>9s} {'base GB':>8s} {'peak GB':>8s} {'act~ GB':>8s} {'status':>8s}"
    )
    for r in context_rows:
        mbps = None if r.bytes_per_second_est is None else r.bytes_per_second_est / 1e6
        lines.append(
            f"{r.kind:12s} {r.model:30s} {r.context:6d} {r.batch:5d} "
            f"{fnum(r.tokens_per_second):>12s} {fnum(mbps,1):>11s} {fnum(r.update_ms,2):>9s} "
            f"{fnum(r.baseline_alloc_gib,2):>8s} {fnum(r.peak_alloc_gib,2):>8s} "
            f"{fnum(r.activation_like_gib,2):>8s} {r.status:>8s}"
        )

    lines.append("")
    lines.append("MICROBATCH / ACCUMULATION SWEEP @ ctx=1024 — CONSTANT TOKENS PER UPDATE")
    lines.append(
        f"{'model':30s} {'batch':>5s} {'accum':>5s} {'tokens':>8s} {'tok/s':>12s} {'MB/s est':>11s} {'ms/update':>11s} {'peak GB':>8s} {'act~ GB':>8s} {'status':>8s}"
    )
    for r in update_rows:
        mbps = None if r.bytes_per_second_est is None else r.bytes_per_second_est / 1e6
        lines.append(
            f"{r.model:30s} {r.batch:5d} {r.accum:5d} {r.tokens_per_update:8d} "
            f"{fnum(r.tokens_per_second):>12s} {fnum(mbps,1):>11s} {fnum(r.update_ms,2):>11s} "
            f"{fnum(r.peak_alloc_gib,2):>8s} {fnum(r.activation_like_gib,2):>8s} {r.status:>8s}"
        )

    lines.append("")
    lines.append("LOSS-PATH BREAKDOWN — SAME BACKBONE, CE ONLY VS COMPLETE CACHE")
    lines.append(
        f"{'kind':12s} {'model':30s} {'ctx':>6s} {'batch':>5s} {'tok/s':>12s} {'peak GB':>8s} {'act~ GB':>8s} {'status':>8s}"
    )
    for r in breakdown_rows:
        lines.append(
            f"{r.kind:12s} {r.model:30s} {r.context:6d} {r.batch:5d} "
            f"{fnum(r.tokens_per_second):>12s} {fnum(r.peak_alloc_gib,2):>8s} "
            f"{fnum(r.activation_like_gib,2):>8s} {r.status:>8s}"
        )

    # Automatic recommendations from the 1024 b8/a1 update rows and longest context.
    lines.append("")
    lines.append("AUTOMATIC DIAGNOSIS")
    by_key = {(r.model, r.kind, r.context, r.batch, r.accum): r for r in context_rows}
    upd = {(r.model, r.batch, r.accum): r for r in update_rows}
    reference = upd.get((V20_DENSE, 8, 1))
    for candidate in (V21_EXACT_DENSE, V21_FAST, V21_SPARSE, V21_LIGHT):
        row = upd.get((candidate, 8, 1))
        if reference and row and reference.status == row.status == "ok":
            speed = row.tokens_per_second / reference.tokens_per_second
            peak = row.peak_alloc_gib / reference.peak_alloc_gib
            lines.append(
                f"{candidate} vs v20 dense @1024 b8/a1: speed={speed:.3f}x peak={peak:.3f}x"
            )
    # Best microbatch per model.
    for name in args.models:
        valid = [r for r in update_rows if r.model == name and r.status == "ok"]
        if valid:
            best = max(valid, key=lambda r: r.tokens_per_second or 0.0)
            lines.append(
                f"best update config {name}: batch={best.batch} accum={best.accum} "
                f"tok/s={best.tokens_per_second:,.0f} peak={best.peak_alloc_gib:.2f}G"
            )

    lines.append("")
    lines.append("INTERPRETATION RULES")
    lines.append("1. exact_dense must preserve v20 outputs; its gain is safe to carry into the canonical run.")
    lines.append("2. fast_pcaf is the lower-cost validated PCAF path and isolates the dense episodic wrapper.")
    lines.append("3. sparse_pcaf/light_sparse are scalable redesign candidates; they need a short paired quality guard before final scaling.")
    lines.append("4. if b8/a1 strongly beats b1/a8, the old train tok/s was primarily microbatch/synchronization overhead, not architecture throughput.")
    lines.append("5. if sparse cache sharply cuts peak memory, B*T*K*V dense retrieval—not the Field recurrence—is the main VRAM culprit.")
    if profiles:
        lines.append("")
        lines.append("PROFILES")
        for name, info in profiles.items():
            lines.append(f"{name}: {info}")
    lines.append("=" * 196)
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", required=True)
    p.add_argument("--canonical-source", default="")
    p.add_argument("--cache-dir", default="/home/ubuntu/field_lab/hf_cache")
    p.add_argument("--tokenizer-source", default="")
    p.add_argument("--data-frac", type=float, default=0.05)
    p.add_argument("--data-device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--tokenizer-min-frequency", type=int, default=2)

    p.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    p.add_argument("--target-params", type=int, default=300_000_000)
    p.add_argument("--max-param-delta-pct", type=float, default=0.75)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--layers", type=int, default=24)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--min-ff-hidden", type=int, default=1024)
    p.add_argument("--max-ff-hidden", type=int, default=4096)
    p.add_argument("--ff-multiple", type=int, default=64)

    p.add_argument("--field-chunk", type=int, default=32)
    p.add_argument("--triton-block-c", type=int, default=16)
    p.add_argument("--triton-chunk-t", type=int, default=64)
    p.add_argument("--num-buckets", type=int, default=16384)
    p.add_argument("--salience-floor", type=float, default=0.10)
    p.add_argument("--residual-limit", type=float, default=4.0)
    # Compatibility fields consumed by imported v10 constructors.
    p.add_argument("--address-dim", type=int, default=64)
    p.add_argument("--latent-top-k", type=int, default=4)
    p.add_argument("--score-limit", type=float, default=8.0)
    p.add_argument("--sidecar-aux-weight", type=float, default=0.0)
    p.add_argument("--sidecar-max-mix", type=float, default=0.25)
    p.add_argument("--sidecar-gate-bias", type=float, default=-2.0)
    p.add_argument("--gate-grad-scale", type=float, default=1.0)
    p.add_argument("--delta-heads", type=int, default=4)
    p.add_argument("--delta-block", type=int, default=64)
    p.add_argument("--span-top-k", type=int, default=4)
    p.add_argument("--phase-bands", type=int, default=4)
    p.add_argument("--phase-rank", type=int, default=32)

    p.add_argument("--fusion-q-heads", type=int, default=16)
    p.add_argument("--fusion-kv-heads", type=int, default=4)
    p.add_argument("--fusion-latent-dim", type=int, default=256)
    p.add_argument("--refresh-windows", nargs="+", type=int,
                   default=[256, 512, 1024, 2048])
    p.add_argument("--landmark-chunk", type=int, default=256)
    p.add_argument("--sparse-surprise-mix", type=float, default=0.25)

    p.add_argument("--model-seed", type=int, default=1234)
    p.add_argument("--embedding-seed", type=int, default=314159)
    p.add_argument("--eval-seed", type=int, default=9012)
    p.add_argument("--amp", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.10)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--contexts", nargs="+", type=int,
                   default=[1024, 4096, 8192, 16384])
    p.add_argument("--tokens-per-call", type=int, default=8192)
    p.add_argument("--tokens-per-update", type=int, default=8192)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--inference-warmup", type=int, default=3)
    p.add_argument("--inference-steps", type=int, default=10)
    p.add_argument("--breakdown-contexts", nargs="+", type=int,
                   default=[1024, 4096])
    p.add_argument("--exact-tol", type=float, default=0.03)

    p.add_argument("--profile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--profile-models", nargs="+", choices=MODEL_NAMES,
                   default=[V20_DENSE, V21_LIGHT, TRANSFORMER])
    p.add_argument("--profile-context", type=int, default=1024)
    p.add_argument("--profile-batch", type=int, default=4)
    p.add_argument("--profile-rows", type=int, default=25)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.models = [m for m in MODEL_NAMES if m in set(args.models)]
    if not args.models:
        raise ValueError("no models selected")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA H100 required")
    if args.layers != len(args.refresh_windows) * 6:
        raise ValueError("layers must equal 6 * len(refresh_windows)")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.set_float32_matmul_precision("high")

    arena = base.import_module(base.V15_PATH, "field_scale_50m_v15_for_v21")
    canonical_path = base.locate_canonical(args.canonical_source)
    actual_sha = sha256(canonical_path)
    if actual_sha != EXPECTED_CANONICAL_SHA256:
        raise RuntimeError(
            f"canonical SHA mismatch expected={EXPECTED_CANONICAL_SHA256} actual={actual_sha}"
        )
    v3 = arena.base.import_module(arena.base.V3_PATH, "v21_v3")
    bridge = arena.base.import_module(arena.base.BRIDGE_PATH, "v21_bridge")
    optmod = arena.base.import_module(arena.base.OPT_PATH, "v21_opt")
    epi = arena.base.import_module(arena.base.V9_PATH, "v21_epi")
    judge = arena.base.import_module(arena.base.JUDGE_PATH, "v21_judge")
    canonical = arena.base.import_module(canonical_path, "v21_canonical")
    optmod.v3_global = v3
    arena.base.install_fast_candidate_route(epi, optmod)
    changed = core.patch_vocab(args.vocab_size, HERE, canonical_path)
    log(f"[vocab] patched VOCAB={args.vocab_size:,} in {len(changed)} modules")

    # Compatibility widths used by imported helper constructors.
    args.field_dim = args.dim
    args.field_layers = args.layers
    args.field_heads = args.heads
    args.field_ff_hidden = args.min_ff_hidden
    args.hybrid_ff_hidden = args.min_ff_hidden
    args.af_ff_hidden = args.min_ff_hidden
    args.tf_dim = args.dim
    args.tf_layers = args.layers
    args.tf_heads = args.heads
    args.tf_ff_hidden = args.min_ff_hidden
    args.conf_distill_ramp = 1

    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)
    raw_rows = core.load_raw_rows(args.cache_dir, args.data_frac)
    tokenizer = base.copy_or_train_tokenizer(
        root, raw_rows[0], args.vocab_size, args.tokenizer_min_frequency,
        args.tokenizer_source,
    )
    train_c, val_c, test_c = core.save_or_load_corpora(root, tokenizer, raw_rows)
    del val_c, test_c
    train = core.place_tokens(
        train_c.tokens, device, args.data_device, "train"
    )
    bytes_per_token = train_c.bytes_per_token

    deps = (arena, v3, canonical, bridge, optmod, epi, judge)
    shapes: Dict[str, Shape] = {}
    for name in args.models:
        log(f"[shape] solving {name}")
        shapes[name] = solve_shape(
            name, args, arena, v3, canonical, bridge, optmod, epi, judge
        )
        delta = 100.0 * (shapes[name].params - args.target_params) / args.target_params
        if abs(delta) > args.max_param_delta_pct:
            raise RuntimeError(f"parameter mismatch {name}: {delta:+.3f}%")
        log(
            f"[shape] {name} params={shapes[name].params:,} "
            f"dTarget={delta:+.3f}% ff={shapes[name].ff_hidden}"
        )

    exact = exact_refresh_selftest(args, device)
    log(f"[selftest] exact refresh: {exact}")

    context_rows: List[BenchRow] = []
    for context in args.contexts:
        batch = max(1, args.tokens_per_call // context)
        for name in args.models:
            log(f"[context/train] {name} ctx={context} batch={batch}")
            row = train_update_benchmark(
                name, shapes[name], args, deps, train, device,
                context=context, batch=batch, accum=1, loss_mode="full",
                warmup=args.warmup, steps=args.steps,
                bytes_per_token=bytes_per_token,
            )
            context_rows.append(row)
            log(asdict(row))
            log(f"[context/infer] {name} ctx={context} batch={batch}")
            row = inference_benchmark(
                name, shapes[name], args, deps, train, device,
                context=context, batch=batch, loss_mode="full",
                warmup=args.inference_warmup, steps=args.inference_steps,
                bytes_per_token=bytes_per_token,
            )
            context_rows.append(row)
            log(asdict(row))
            atomic_json(root / "context_rows.json", [asdict(x) for x in context_rows])

    update_rows: List[BenchRow] = []
    context = 1024
    micro_configs = [(1, 8), (2, 4), (4, 2), (8, 1)]
    # This sweep directly diagnoses the v20 launcher (batch=1, accum=8).
    for name in args.models:
        for batch, accum in micro_configs:
            if batch * accum * context != args.tokens_per_update:
                continue
            log(f"[microbatch] {name} batch={batch} accum={accum}")
            row = train_update_benchmark(
                name, shapes[name], args, deps, train, device,
                context=context, batch=batch, accum=accum,
                loss_mode="full", warmup=args.warmup, steps=args.steps,
                bytes_per_token=bytes_per_token,
            )
            update_rows.append(row)
            log(asdict(row))
            atomic_json(root / "update_rows.json", [asdict(x) for x in update_rows])

    breakdown_rows: List[BenchRow] = []
    for context in args.breakdown_contexts:
        batch = max(1, args.tokens_per_call // context)
        for name in args.models:
            if name == TRANSFORMER:
                continue
            for mode in ("ce", "full"):
                log(f"[breakdown] {name} mode={mode} ctx={context} batch={batch}")
                row = train_update_benchmark(
                    name, shapes[name], args, deps, train, device,
                    context=context, batch=batch, accum=1,
                    loss_mode=mode, warmup=args.warmup, steps=args.steps,
                    bytes_per_token=bytes_per_token,
                )
                breakdown_rows.append(row)
                log(asdict(row))
                atomic_json(
                    root / "breakdown_rows.json",
                    [asdict(x) for x in breakdown_rows],
                )

    profiles: Dict[str, object] = {}
    if args.profile:
        for name in args.profile_models:
            if name not in shapes:
                continue
            log(f"[profile] {name}")
            try:
                profiles[name] = profile_one(
                    name, shapes[name], args, deps, train, device, root
                )
            except Exception as exc:
                profiles[name] = {"error": repr(exc)}
                log(f"[profile] {name} failed: {exc!r}")
            atomic_json(root / "profiles.json", profiles)

    summary = render_summary(
        args, shapes, exact, context_rows, update_rows, breakdown_rows,
        profiles, bytes_per_token,
    )
    atomic_text(root / "summary.txt", summary)
    result = {
        "config": vars(args),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "canonical": str(canonical_path),
        "canonical_sha256": actual_sha,
        "bytes_per_token": bytes_per_token,
        "shapes": {k: asdict(v) for k, v in shapes.items()},
        "exact_refresh": exact,
        "context_rows": [asdict(x) for x in context_rows],
        "update_rows": [asdict(x) for x in update_rows],
        "breakdown_rows": [asdict(x) for x in breakdown_rows],
        "profiles": profiles,
    }
    atomic_json(root / "results.json", result)
    log(summary)


if __name__ == "__main__":
    main()
