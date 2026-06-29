#!/usr/bin/env python3
"""Short, paired ablation arena for closing the 300M Field/PCAF oracle gap.

This is a mechanism-ranking experiment, not a production-kernel benchmark.
It uses the exact portable Field equations from the validated v3 arena and
compares every hypothesis against the current best hybrid backbone:
softpatch + one local w256 path + confidence-aware PCAF.

Two research lanes are kept alive:
  1) Main hybrid lane: better router supervision/calibration, more candidates,
     and conservative zero-init data-dependent carrier residuals.
  2) Strictly attention-free lane: current multiscale branch, learned branch
     fusion, and the strongest router-supervision hypothesis.

Stage A screens all mechanisms with one paired seed. Stage B continues the two
best hybrid hypotheses and best attention-free hypothesis. Stage C confirms
with two new paired seeds. All arms are parameter-matched to the hybrid
reference by adjusting only SwiGLU width.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
V3_PATH = HERE / "field_hybrid_attentionfree_qualification_v3.py"
LN2 = math.log(2.0)
VOCAB = 256

HYBRID_REF = "hybrid_ref"
AF_REF = "attentionfree_ref"
HYBRID_ARMS = (
    HYBRID_REF,
    "router_regret",
    "router_hard_focal",
    "calibrated_mix",
    "topk8",
    "dynamic_radius",
    "dynamic_phase",
    "dynamic_both",
)
AF_ARMS = (
    AF_REF,
    "attentionfree_concat",
    "attentionfree_regret",
)
ALL_ARMS = (*HYBRID_ARMS, *AF_ARMS)


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


v3 = import_module(V3_PATH, "field_hybrid_attentionfree_qualification_v3_dep")


def log(msg: object = "") -> None:
    print(str(msg), flush=True)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def nparams(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


def amp_ctx(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_optimizer(model: nn.Module, lr: float, weight_decay: float):
    kwargs = dict(lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay)
    try:
        return torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(model.parameters(), **kwargs)


def lr_at(step: int, total: int, warmup: int, peak: float, min_ratio: float) -> float:
    if step <= warmup:
        return peak * step / max(1, warmup)
    p = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    c = 0.5 * (1.0 + math.cos(math.pi * p))
    return peak * (min_ratio + (1.0 - min_ratio) * c)


# ======================================================================================
# Router / cache hypotheses
# ======================================================================================


class ExperimentalSuccessorCache(nn.Module):
    """Successor cache with identical causal features and selectable training hypothesis."""

    FEATURE_DIM = 15

    def __init__(
        self,
        state_dim: int,
        memory_dim: int,
        num_buckets: int,
        order: int,
        top_k: int,
        aux_mode: str,
    ):
        super().__init__()
        if aux_mode not in {"soft", "regret", "hard_focal", "calibrated"}:
            raise ValueError(aux_mode)
        self.state_dim = state_dim
        self.memory_dim = memory_dim
        self.num_buckets = num_buckets
        self.order = order
        self.top_k = top_k
        self.aux_mode = aux_mode
        self.router_mode = "confidence"
        self.shared_weight = nn.Parameter(torch.empty(memory_dim, state_dim))
        nn.init.kaiming_uniform_(self.shared_weight, a=math.sqrt(5))
        self.state_gate = nn.Sequential(v3.RMSNorm(state_dim), nn.Linear(state_dim, 1))
        self.router = nn.Sequential(
            nn.LayerNorm(self.FEATURE_DIM),
            nn.Linear(self.FEATURE_DIM, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self.evidence_gain = nn.Parameter(torch.tensor(0.50))
        self.evidence_bias = nn.Parameter(torch.tensor(-0.75))
        self.recency_scale = nn.Parameter(torch.tensor(1.0))
        self.distill_temperature = 0.5
        self.distill_scale = 1.0
        self.distill_weight = {
            "soft": 0.05,
            "regret": 0.035,
            "hard_focal": 0.035,
            "calibrated": 0.05,
        }[aux_mode]
        # Exact identity at initialization. Only the calibrated arm learns these.
        if aux_mode == "calibrated":
            self.log_param_temperature = nn.Parameter(torch.zeros(()))
            self.log_cache_temperature = nn.Parameter(torch.zeros(()))
        else:
            self.register_buffer("log_param_temperature", torch.zeros(()), persistent=False)
            self.register_buffer("log_cache_temperature", torch.zeros(()), persistent=False)
        self.enabled = True
        self.last_aux: Dict[str, float] = {}

    def _features(self, scores, weights, valid, cand_tokens, recency, logits):
        n, k = valid.shape
        masked = scores.masked_fill(~valid, -1.0e9)
        top2 = torch.topk(masked, k=min(2, k), dim=-1).values
        top1 = top2[:, 0]
        count = valid.float().sum(-1)
        margin = (
            torch.where(count >= 2, top2[:, 0] - top2[:, 1], torch.zeros_like(top1))
            if top2.size(-1) > 1 else torch.zeros_like(top1)
        )
        cand_ent = -(weights * torch.log(weights.clamp_min(1e-8))).sum(-1)
        cand_ent = torch.where(
            count > 1,
            cand_ent / torch.log(count.clamp_min(2.0)),
            torch.zeros_like(cand_ent),
        )
        wrec = (weights * recency).sum(-1)
        cl = cand_tokens.long()
        same = cl[:, :, None] == cl[:, None, :]
        token_mass = (same.float() * weights[:, None, :]).sum(-1)
        earlier = torch.tril(
            torch.ones((k, k), device=valid.device, dtype=torch.bool), diagonal=-1
        )
        unique = valid & ~(same & earlier[None]).any(-1)
        unique_mass = token_mass.masked_fill(~unique, 0.0)
        mass2, massidx = torch.topk(unique_mass, k=min(2, k), dim=-1)
        cache_conf = mass2[:, 0]
        cache_second = mass2[:, 1] if mass2.size(-1) > 1 else torch.zeros_like(cache_conf)
        cache_margin = cache_conf - cache_second
        cache_ent = -(unique_mass * torch.log(unique_mass.clamp_min(1e-8))).sum(-1) / math.log(max(k, 2))
        cache_top = cl.gather(1, massidx[:, :1]).squeeze(1)
        l32 = logits.float()
        logz = torch.logsumexp(l32, -1)
        ptop, ptok = torch.topk(l32, 2, dim=-1)
        pconf = torch.exp(ptop[:, 0] - logz)
        psecond = torch.exp(ptop[:, 1] - logz)
        pmargin = pconf - psecond
        ptoken = ptok[:, 0]
        p_cache_top = torch.exp(l32.gather(1, cache_top[:, None]).squeeze(1) - logz)
        cache_mass_ptop = (weights * (cl == ptoken[:, None]).float()).sum(-1)
        agree = (cache_top == ptoken).float()
        delta = cache_conf - pconf
        return torch.stack((
            torch.tanh(top1), torch.tanh(margin), cand_ent.clamp(0, 1),
            (count / float(k)).clamp(0, 1), wrec.clamp(0, 1),
            cache_conf.clamp(0, 1), cache_margin.clamp(0, 1), cache_ent.clamp(0, 1),
            pconf.clamp(0, 1), pmargin.clamp(0, 1), agree,
            p_cache_top.clamp(0, 1), cache_mass_ptop.clamp(0, 1),
            delta.clamp(-1, 1), delta.abs().clamp(0, 1),
        ), dim=-1).detach()

    def _auxiliary(self, gate_logit, log_adv):
        weight = torch.tanh(log_adv.abs()).detach()
        denom = weight.sum().clamp_min(1.0)
        if self.aux_mode in {"soft", "calibrated"}:
            teacher = torch.sigmoid(log_adv.detach() / self.distill_temperature)
            raw = F.binary_cross_entropy_with_logits(gate_logit, teacher, reduction="none")
        elif self.aux_mode == "regret":
            # Directly teach the causal router the signed future loss advantage.
            target = (log_adv.detach() / self.distill_temperature).clamp(-6.0, 6.0)
            raw = F.smooth_l1_loss(gate_logit, target, reduction="none", beta=1.0)
            teacher = torch.sigmoid(target)
        else:
            teacher = (log_adv.detach() > 0).float()
            bce = F.binary_cross_entropy_with_logits(gate_logit, teacher, reduction="none")
            prob = torch.sigmoid(gate_logit)
            pt = torch.where(teacher > 0.5, prob, 1.0 - prob)
            raw = bce * (1.0 - pt).square()
        aux = (raw * weight).sum() / denom
        return aux, teacher

    def forward(self, states, logits, tokens, targets, compute_metrics=False):
        flat_logits_raw = logits.reshape(-1, VOCAB).float()
        flat_targets = targets.reshape(-1)
        param_temp = self.log_param_temperature.exp().clamp(0.50, 2.0)
        cache_temp = self.log_cache_temperature.exp().clamp(0.50, 2.0)
        flat_logits = flat_logits_raw / param_temp
        param_log_probs = F.log_softmax(flat_logits, dim=-1)
        param_nll = -param_log_probs.gather(1, flat_targets[:, None]).squeeze(1)
        param_target = torch.exp(-param_nll)
        if not self.enabled:
            primary = param_nll.mean()
            stats = v3.CacheStats(0, 0, 0, 0, float(primary/LN2), float(primary/LN2), 0, 0, 0, 0, 0) if compute_metrics else None
            return primary, primary.detach(), stats

        b, t, _ = states.shape
        idx = v3.causal_recent_candidates(tokens, self.order, self.num_buckets, self.top_k)
        valid = idx >= 0
        has = valid.any(-1)
        safe = idx.clamp_min(0)
        batch_idx = torch.arange(b, device=states.device)[:, None, None]
        proj = v3.normalize_rows(F.linear(states.float(), self.shared_weight.float()))
        q = proj[:, :, None, :]
        ck = proj[batch_idx, safe]
        scores = (ck * q).sum(-1) * (self.memory_dim ** -0.5)
        recency = safe.float() / max(float(t - 1), 1.0)
        scores = scores + self.recency_scale.float() * recency
        scores = scores.masked_fill(~valid, -1.0e9)
        weights = torch.softmax(scores.float() / cache_temp, dim=-1) * valid.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        cand_tokens = targets[batch_idx, safe]
        target_cache = (weights * (cand_tokens == targets[:, :, None]).float()).sum(-1)

        active = has.reshape(-1)
        state_logit = self.state_gate(states).float().squeeze(-1)
        if bool(active.any()):
            features = self._features(
                scores.reshape(-1, self.top_k)[active] / cache_temp,
                weights.reshape(-1, self.top_k)[active],
                valid.reshape(-1, self.top_k)[active],
                cand_tokens.reshape(-1, self.top_k)[active],
                recency.reshape(-1, self.top_k)[active],
                flat_logits[active],
            )
            route = self.router(features)[:, 0]
        else:
            features = states.new_zeros((0, self.FEATURE_DIM))
            route = states.new_zeros((0,))
        flat_state_logit = state_logit.reshape(-1)
        gate_flat = torch.zeros_like(flat_state_logit)
        gate_logit_active = states.new_zeros((0,), dtype=torch.float32)
        if bool(active.any()):
            cache_conf = features[:, 5].clamp(1e-4, 1.0 - 1e-4)
            param_conf = features[:, 8].clamp(1e-4, 1.0 - 1e-4)
            evidence = torch.logit(cache_conf) - torch.logit(param_conf)
            evidence = evidence + 1.25 * features[:, 6] - 0.50 * features[:, 7]
            evidence = evidence + 0.35 * features[:, 10] + 0.25 * features[:, 3]
            gate_logit_active = (
                flat_state_logit[active] + route + self.evidence_gain * evidence + self.evidence_bias
            )
            gate_flat[active] = torch.sigmoid(gate_logit_active).clamp(1e-5, 1.0 - 1e-5)
        gate = gate_flat.view(b, t)
        mixed = (1.0 - gate.reshape(-1)) * param_target + gate.reshape(-1) * target_cache.reshape(-1)
        primary = -torch.log(mixed.clamp_min(1e-8)).mean()
        loss = primary

        if self.training and bool(active.any()) and self.distill_scale > 0:
            pa = param_target[active]
            ca = target_cache.reshape(-1)[active]
            log_adv = torch.log(ca.detach().clamp_min(1e-8)) - torch.log(pa.detach().clamp_min(1e-8))
            aux, teacher = self._auxiliary(gate_logit_active, log_adv)
            loss = primary + self.distill_weight * float(self.distill_scale) * aux
            self.last_aux = {
                "aux": float(aux.detach()),
                "teacher": float(teacher.mean()),
                "cache_win": float((log_adv > 0).float().mean()),
                "param_temperature": float(param_temp.detach()),
                "cache_temperature": float(cache_temp.detach()),
            }

        stats = None
        if compute_metrics:
            with torch.no_grad():
                param_loss = param_nll.mean()
                oracle_target = torch.maximum(param_target, target_cache.reshape(-1))
                oracle_target = torch.where(active, oracle_target, param_target)
                oracle_loss = -torch.log(oracle_target.clamp_min(1e-8)).mean()
                denom = float((param_loss - oracle_loss).clamp_min(1e-12))
                capture = float((param_loss - primary) / max(denom, 1e-12))
                cache_target_flat = target_cache.reshape(-1)
                cache_win = active & (cache_target_flat > param_target)
                cache_lose = active & ~cache_win
                gate_win = float(gate_flat[cache_win].mean()) if bool(cache_win.any()) else 0.0
                gate_lose = float(gate_flat[cache_lose].mean()) if bool(cache_lose.any()) else 0.0
                stats = v3.CacheStats(
                    coverage=float(has.float().mean()),
                    gate=float(gate_flat[active].mean()) if bool(active.any()) else 0.0,
                    hit=float((cache_target_flat > 0).float().mean()),
                    cache_prob=float(target_cache.mean()),
                    param_bpb=float(param_loss / LN2),
                    oracle_bpb=float(oracle_loss / LN2),
                    capture=capture,
                    cache_win_rate=float(cache_win.float().sum() / active.float().sum().clamp_min(1.0)),
                    gate_when_cache_wins=gate_win,
                    gate_when_cache_loses=gate_lose,
                    gate_separation=gate_win - gate_lose,
                )
        return loss, primary.detach(), stats


# ======================================================================================
# Conservative Field and attention-free hypotheses
# ======================================================================================


def dynamic_field_read(raw, transition_r, transition_i, gamma, field_chunk):
    """Reference Field read with token-dependent diagonal complex carrier."""
    channels = raw.shape[-1] // 3
    inj_r = torch.tanh(raw[..., :channels])
    inj_i = torch.tanh(raw[..., channels:2*channels])
    vacancy = torch.sigmoid(raw[..., 2*channels:]) * v3.VAC_MAX
    injection = torch.complex(inj_r, inj_i)
    transition = torch.complex(transition_r, transition_i)
    a = (1.0 - vacancy).to(torch.complex64) * transition
    b = gamma.to(torch.complex64) * vacancy.to(torch.complex64) * injection
    states = v3.hierarchical_scan(a, b, field_chunk)
    previous = torch.cat((torch.zeros_like(states[:, :1]), states[:, :-1]), dim=1)
    moved = transition * previous
    displaced = vacancy.to(torch.complex64) * moved
    return torch.cat((states.real, states.imag, displaced.real, displaced.imag), dim=-1)



class DynamicCarrierField(nn.Module):
    """Zero-init low-rank residual modulation of radius and/or phase per token."""

    def __init__(self, base: nn.Module, mode: str, rank: int = 32, scale: float = 0.25):
        super().__init__()
        if mode not in {"radius", "phase", "both"}:
            raise ValueError(mode)
        self.base = base
        self.mode = mode
        self.scale = float(scale)
        dim = int(base.dim)
        channels = int(base.channels)
        out_dim = channels * (2 if mode == "both" else 1)
        self.transition_norm = v3.RMSNorm(dim)
        self.transition_down = nn.Linear(dim, rank, bias=False)
        self.transition_up = nn.Linear(rank, out_dim, bias=False)
        nn.init.zeros_(self.transition_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.base.write_proj(x).contiguous().float()
        delta = self.transition_up(F.silu(self.transition_down(self.transition_norm(x)))).float()
        if self.mode == "radius":
            dr, dt = delta, None
        elif self.mode == "phase":
            dr, dt = None, delta
        else:
            dr, dt = delta.chunk(2, dim=-1)
        radius_logit = self.base.radius_logit.float()[None, None, :]
        theta = self.base.theta.float()[None, None, :]
        if dr is not None:
            radius_logit = radius_logit + self.scale * torch.tanh(dr)
        if dt is not None:
            theta = theta + self.scale * torch.tanh(dt)
        radius = torch.sigmoid(radius_logit).clamp(0.50, 0.99995)
        tr = radius * torch.cos(theta)
        ti = radius * torch.sin(theta)
        gamma = torch.sqrt((1.0 - radius.square()).clamp_min(1e-4))
        read = dynamic_field_read(raw, tr, ti, gamma, self.base.field_chunk)
        out = self.base.out_proj(self.base.read_norm(read))
        gate = torch.sigmoid(self.base.gate_proj(x))
        return x + (out * gate).to(x.dtype)


class ConcatProjectMultiScale(nn.Module):
    """Learned full branch fusion, initialized to the old uniform average."""

    def __init__(self, old: nn.Module):
        super().__init__()
        self.dim = old.dim
        self.inner = old.inner
        self.dilations = old.dilations
        self.norm = old.norm
        self.in_proj = old.in_proj
        self.convs = old.convs
        self.gate = old.gate
        self.out = old.out
        branches = len(self.dilations)
        self.blend = nn.Linear(branches * self.inner, self.inner, bias=False)
        with torch.no_grad():
            self.blend.weight.zero_()
            for j in range(branches):
                idx = torch.arange(self.inner)
                self.blend.weight[idx, j * self.inner + idx] = 1.0 / branches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        z = self.in_proj(h).transpose(1, 2)
        branches = []
        for conv, dilation in zip(self.convs, self.dilations):
            branches.append(F.silu(conv(F.pad(z, (2 * dilation, 0)))).transpose(1, 2))
        y = self.blend(torch.cat(branches, dim=-1))
        y = y * torch.sigmoid(self.gate(h))
        return x + self.out(y)


def base_arm_for(arm: str) -> str:
    return "softpatch_multiscale_conf" if arm in AF_ARMS else "softpatch_local_w256_conf"


def attention_free(arm: str) -> bool:
    return arm in AF_ARMS


def replace_cache(model: nn.Module, aux_mode: str, top_k: int) -> None:
    old = model.cache
    new = ExperimentalSuccessorCache(
        old.state_dim, old.memory_dim, old.num_buckets, old.order, top_k, aux_mode
    ).to(next(model.parameters()).device)
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    allowed_missing = {"log_param_temperature", "log_cache_temperature"}
    if unexpected or any(k not in allowed_missing for k in missing):
        raise RuntimeError((missing, unexpected))
    model.cache = new


def modify_model(model: nn.Module, arm: str, args) -> nn.Module:
    if arm == "router_regret":
        replace_cache(model, "regret", 4)
    elif arm == "router_hard_focal":
        replace_cache(model, "hard_focal", 4)
    elif arm == "calibrated_mix":
        replace_cache(model, "calibrated", 4)
    elif arm == "topk8":
        replace_cache(model, "soft", 8)
    elif arm in {"dynamic_radius", "dynamic_phase", "dynamic_both"}:
        mode = arm.removeprefix("dynamic_")
        for block in model.blocks:
            block.mixer = DynamicCarrierField(
                block.mixer, mode, rank=args.transition_rank, scale=args.transition_scale
            )
    elif arm == "attentionfree_concat":
        for key in list(model.multiscales.keys()):
            model.multiscales[key] = ConcatProjectMultiScale(model.multiscales[key])
    elif arm == "attentionfree_regret":
        replace_cache(model, "regret", 4)
    elif arm in {HYBRID_REF, AF_REF}:
        pass
    else:
        raise KeyError(arm)
    return model


def default_hidden(dim: int) -> int:
    return ((int(8 * dim / 3) + 63) // 64) * 64


def raw_model(arm: str, hidden: int, args, device: torch.device) -> nn.Module:
    model = v3.FieldPCAFLM(
        base_arm_for(arm), args.dim, args.layers, args.heads, hidden,
        args.field_chunk, args.num_buckets,
    ).to(device)
    # modify_model may insert freshly constructed modules (dynamic carrier or
    # concat/project mixer). New nn.Modules are created on CPU by default, even
    # when the base model already lives on CUDA. Move the complete modified
    # graph once more so every newly inserted parameter/buffer follows device.
    model = modify_model(model, arm, args)
    return model.to(device)


def resolve_shapes(args, device: torch.device) -> Dict[str, dict]:
    base_hidden = default_hidden(args.dim)
    seed_all(args.model_seed)
    base = raw_model(HYBRID_REF, base_hidden, args, device)
    target = nparams(base)
    del base
    gc.collect()
    slope = 3 * args.dim * args.layers
    shapes: Dict[str, dict] = {}
    for arm in ALL_ARMS:
        seed_all(args.model_seed)
        probe = raw_model(arm, base_hidden, args, device)
        p0 = nparams(probe)
        del probe
        ideal = base_hidden + (target - p0) / max(1, slope)
        step = args.hidden_step
        hidden = max(64, int(round(ideal / step) * step))
        seed_all(args.model_seed)
        check = raw_model(arm, hidden, args, device)
        params = nparams(check)
        del check
        delta = 100.0 * (params - target) / target
        if abs(delta) > args.max_param_delta_pct:
            raise RuntimeError(f"{arm}: parameter delta {delta:+.3f}%")
        shapes[arm] = {
            "arm": arm,
            "hidden": hidden,
            "params": params,
            "delta_pct": delta,
            "attention_free": attention_free(arm),
        }
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return shapes


def build_model(arm: str, seed: int, shape: dict, args, device: torch.device) -> nn.Module:
    seed_all(args.model_seed + seed)
    model = raw_model(arm, int(shape["hidden"]), args, device)
    actual = nparams(model)
    if actual != int(shape["params"]):
        raise RuntimeError((arm, actual, shape["params"]))
    return model


# ======================================================================================
# Train / evaluate
# ======================================================================================


@dataclass
class Result:
    arm: str
    group: str
    seed: int
    params: int
    delta_pct: float
    hidden: int
    attention_free: bool
    steps: int
    bpb_train_context: float
    bpb_8k: float
    bpb_16k: float
    param_bpb_8k: float
    oracle_bpb_8k: float
    capture_8k: float
    coverage_8k: float
    gate_8k: float
    gate_sep_8k: float
    bytes_per_second: float
    peak_gib: float
    checkpoint: str


def group_for(arm: str) -> str:
    return "attention_free" if arm in AF_ARMS else "hybrid"


def eval_model(model, data, seq, windows, seed, args, device):
    starts = v3.fixed_starts(len(data), seq, windows, seed)
    return v3.evaluate(model, data, device, args.amp, seq, starts, batch_size=1)


def train_arm(
    arm: str,
    seed: int,
    target_steps: int,
    schedule_steps: int,
    shapes: Dict[str, dict],
    args,
    train,
    val,
    device,
    outroot: Path,
) -> dict:
    run = outroot / f"{arm}_seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    result_path = run / f"result_step{target_steps}.json"
    latest = run / "latest.pt"
    if args.resume and result_path.exists():
        return json.loads(result_path.read_text())

    model = build_model(arm, seed, shapes[arm], args, device)
    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    gen_device = train.device.type if train.device.type == "cuda" else "cpu"
    gen = torch.Generator(device=gen_device).manual_seed(args.data_seed + seed)
    start = 0
    history: List[dict] = []
    if args.resume and latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        gen.set_state(state["rng"].cpu())
        start = int(state["step"])
        history = list(state.get("history", []))
        log(f"[resume] {arm} seed={seed} {start}->{target_steps}")

    log("\n" + "=" * 180)
    log(
        f"TRAIN {arm} group={group_for(arm)} seed={seed} params={nparams(model):,} "
        f"d={shapes[arm]['delta_pct']:+.3f}% hidden={shapes[arm]['hidden']} "
        f"steps={start}->{target_steps} schedule={schedule_steps}"
    )
    log("=" * 180)
    model.train()
    torch.cuda.empty_cache() if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    wall = time.perf_counter()
    excluded = 0.0
    processed = 0

    for step in range(start + 1, target_steps + 1):
        model.cache.distill_scale = min(1.0, step / max(1.0, args.distill_ramp))
        lr = lr_at(step, schedule_steps, args.warmup, args.lr, args.min_lr_ratio)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        primary_sum = 0.0
        for micro in range(args.accum):
            x, y = v3.random_batch(train, args.batch_size, args.seq_len, gen, device)
            with amp_ctx(device, args.amp):
                loss, primary, _ = model.loss_and_stats(x, y, compute_metrics=False)
                scaled = loss / args.accum
            if not torch.isfinite(scaled):
                raise FloatingPointError((arm, seed, step, float(scaled)))
            scaled.backward()
            primary_sum += float(primary) / args.accum
            processed += x.numel()
            del x, y, loss, primary, scaled
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(grad):
            raise FloatingPointError((arm, seed, step, "grad"))
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == target_steps:
            sync(device)
            active = max(1e-9, time.perf_counter() - wall - excluded)
            bps = processed / active
            peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
            aux = getattr(model.cache, "last_aux", {})
            log(
                f"[{arm}] step={step:04d}/{target_steps} bpb={primary_sum/LN2:.4f} "
                f"grad={float(grad):.3f} lr={lr:.3e} B/s={bps:,.0f} peak={peak:.2f}G aux={aux}"
            )
            history.append({"step": step, "train_bpb": primary_sum/LN2, "bps": bps, "peak": peak})

        if step % args.eval_every == 0 or step == target_steps:
            t0 = time.perf_counter()
            ev = eval_model(model, val, args.seq_len, args.eval_windows, args.eval_seed + step, args, device)
            excluded += time.perf_counter() - t0
            log(
                f"[{arm}] EVAL step={step:04d} bpb={ev['bpb']:.5f} param={ev['param_bpb']:.5f} "
                f"oracle={ev['oracle_bpb']:.5f} cap={ev['capture']:.3f} "
                f"gate={ev['gate']:.3f} sep={ev['gate_sep']:+.3f}"
            )
            history.append({"step": step, "eval": ev})

        if step % args.save_every == 0 or step == target_steps:
            tmp = latest.with_suffix(".tmp")
            torch.save({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "history": history, "rng": gen.get_state().cpu(),
            }, tmp)
            os.replace(tmp, latest)

    sync(device)
    active = max(1e-9, time.perf_counter() - wall - excluded)
    bps = processed / active
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
    e0 = eval_model(model, val, args.seq_len, args.final_windows, args.eval_seed + 100, args, device)
    e8 = eval_model(model, val, args.long_context, args.long_windows, args.eval_seed + 200, args, device)
    e16 = eval_model(model, val, args.very_long_context, args.very_long_windows, args.eval_seed + 300, args, device)
    result = asdict(Result(
        arm=arm,
        group=group_for(arm),
        seed=seed,
        params=nparams(model),
        delta_pct=float(shapes[arm]["delta_pct"]),
        hidden=int(shapes[arm]["hidden"]),
        attention_free=attention_free(arm),
        steps=target_steps,
        bpb_train_context=e0["bpb"],
        bpb_8k=e8["bpb"],
        bpb_16k=e16["bpb"],
        param_bpb_8k=e8["param_bpb"],
        oracle_bpb_8k=e8["oracle_bpb"],
        capture_8k=e8["capture"],
        coverage_8k=e8["coverage"],
        gate_8k=e8["gate"],
        gate_sep_8k=e8["gate_sep"],
        bytes_per_second=bps,
        peak_gib=peak,
        checkpoint=str(latest),
    ))
    result["history"] = history
    result["cache_aux"] = getattr(model.cache, "last_aux", {})
    atomic_json(result_path, result)
    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


# ======================================================================================
# Self-test and selection
# ======================================================================================


def selftest(args, device, shapes):
    log("[selftest] finite backward, causal prefix, parameter parity, zero-init invariants")
    x = torch.randint(0, VOCAB, (1, 17), device=device)
    y = torch.randint(0, VOCAB, (1, 17), device=device)
    for arm in ALL_ARMS:
        m = build_model(arm, 0, shapes[arm], args, device)
        with amp_ctx(device, args.amp):
            loss, primary, _ = m.loss_and_stats(x, y, False)
        loss.backward()
        finite = bool(torch.isfinite(loss)) and all(
            p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters()
        )
        log(
            f"[selftest] {arm:<28} params={nparams(m):,} d={shapes[arm]['delta_pct']:+.3f}% "
            f"loss={float(loss.detach()):.5f} primary={float(primary):.5f} finite={finite}"
        )
        if not finite:
            raise AssertionError(arm)
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Same-shape zero-init dynamic carrier must reproduce the exact baseline logits.
    seed_all(777)
    ref = v3.FieldPCAFLM("softpatch_local_w256_conf", 32, 1, 4, 64, 8, 64).to(device).eval()
    seed_all(777)
    dyn = v3.FieldPCAFLM("softpatch_local_w256_conf", 32, 1, 4, 64, 8, 64).to(device).eval()
    dyn.blocks[0].mixer = DynamicCarrierField(dyn.blocks[0].mixer, "both", rank=8, scale=0.25).to(device)
    with torch.no_grad(), amp_ctx(device, args.amp):
        er = float((ref.states_logits(x)[1] - dyn.states_logits(x)[1]).abs().max())
    log(f"[selftest] dynamic zero-init equivalence max_abs={er:.3e}")
    if er > 2e-5:
        raise AssertionError("dynamic zero-init")

    seed_all(778)
    af0 = v3.FieldPCAFLM("softpatch_multiscale_conf", 32, 1, 4, 64, 8, 64).to(device).eval()
    seed_all(778)
    af1 = v3.FieldPCAFLM("softpatch_multiscale_conf", 32, 1, 4, 64, 8, 64).to(device).eval()
    for key in list(af1.multiscales.keys()):
        af1.multiscales[key] = ConcatProjectMultiScale(af1.multiscales[key]).to(device)
    with torch.no_grad(), amp_ctx(device, args.amp):
        ea = float((af0.states_logits(x)[1] - af1.states_logits(x)[1]).abs().max())
    log(f"[selftest] AF concat zero-init equivalence max_abs={ea:.3e}")
    if ea > 2e-5:
        raise AssertionError("AF concat zero-init")

    # Prefix causality on the two most invasive mechanisms.
    for arm in ("dynamic_both", "attentionfree_concat", "topk8"):
        m = build_model(arm, 11, shapes[arm], args, device).eval()
        p = 8
        x2 = x.clone()
        x2[:, p:] = torch.randint(0, VOCAB, x2[:, p:].shape, device=device)
        with torch.no_grad(), amp_ctx(device, args.amp):
            a = m.states_logits(x)[1][:, :p]
            b = m.states_logits(x2)[1][:, :p]
        err = float((a - b).abs().max())
        log(f"[selftest] {arm:<28} causal_prefix max_abs={err:.3e}")
        if err > 3e-4:
            raise AssertionError((arm, err))
        del m
    log("[selftest] PASS")


def rank_group(rows: List[dict], reference: str, candidates: Sequence[str], n: int, min_speed: float):
    ref = next(r for r in rows if r["arm"] == reference)
    valid = [
        r for r in rows
        if r["arm"] in candidates and r["bytes_per_second"] >= ref["bytes_per_second"] * min_speed
    ]
    return [r["arm"] for r in sorted(valid, key=lambda z: (z["bpb_8k"], z["bpb_16k"]))[:n]]


def mean_delta(rows: List[dict], arm: str, ref: str, key: str) -> Tuple[float, float]:
    by_seed = {r["seed"]: r for r in rows if r["arm"] == ref}
    vals = [r[key] - by_seed[r["seed"]][key] for r in rows if r["arm"] == arm and r["seed"] in by_seed]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def make_summary(stage_a, stage_b, stage_c, shapes, selected_main, selected_af, args):
    lines = [
        "FIELD ORACLE-GAP ABLATION v8 — SHORT PAIRED MECHANISM VALIDATION",
        "=" * 245,
        f"Protocol: WikiText-103 {args.data_frac:.1%} | train ctx={args.seq_len} | "
        f"eval={args.long_context}/{args.very_long_context} | bytes/update={args.batch_size*args.accum*args.seq_len:,}",
        "Reference: softpatch + local-w256 + confidence PCAF. AF lane remains strictly query-key-attention-free.",
        "Portable exact Field math; any winner must later be transplanted to the canonical Triton implementation.",
        "",
        "PARAMETER MATCHING",
        f"{'arm':<30} {'group':>14} {'params':>13} {'d%':>8} {'ff':>6} {'AF':>4}",
    ]
    for arm in ALL_ARMS:
        s = shapes[arm]
        lines.append(
            f"{arm:<30} {group_for(arm):>14} {s['params']:>13,d} {s['delta_pct']:>+8.3f} "
            f"{s['hidden']:>6d} {('yes' if s['attention_free'] else 'no'):>4}"
        )

    def table(title, rows):
        lines.extend([
            "", title,
            f"{'arm':<30} {'seed':>6} {'BPB2K':>9} {'BPB8K':>9} {'dGroup':>9} {'BPB16K':>9} "
            f"{'oracle':>9} {'cap':>7} {'sep':>7} {'B/s':>11} {'speed':>7}",
        ])
        ref_map = {}
        for r in rows:
            ref_name = AF_REF if r["group"] == "attention_free" else HYBRID_REF
            key = (r["seed"], ref_name)
            ref_map[key] = next((z for z in rows if z["seed"] == r["seed"] and z["arm"] == ref_name), r)
        for r in sorted(rows, key=lambda z: (z["group"], z["seed"], z["bpb_8k"])):
            ref_name = AF_REF if r["group"] == "attention_free" else HYBRID_REF
            ref = ref_map[(r["seed"], ref_name)]
            lines.append(
                f"{r['arm']:<30} {r['seed']:>6d} {r['bpb_train_context']:>9.5f} {r['bpb_8k']:>9.5f} "
                f"{r['bpb_8k']-ref['bpb_8k']:>+9.5f} {r['bpb_16k']:>9.5f} "
                f"{r['oracle_bpb_8k']:>9.5f} {r['capture_8k']:>7.3f} {r['gate_sep_8k']:>+7.3f} "
                f"{r['bytes_per_second']:>11,.0f} {r['bytes_per_second']/max(ref['bytes_per_second'],1):>7.2f}"
            )

    table("STAGE A — ALL-HYPOTHESIS SCREEN", stage_a)
    table("STAGE B — LONGER PAIRED FINALISTS", stage_b)
    table("STAGE C — TWO-SEED CONFIRMATION", stage_c)

    lines += ["", "SEED-PAIRED AGGREGATES"]
    promoted = []
    for arm in [*selected_main, *selected_af]:
        ref = AF_REF if arm in AF_ARMS else HYBRID_REF
        d8, sd8 = mean_delta(stage_c, arm, ref, "bpb_8k")
        d16, sd16 = mean_delta(stage_c, arm, ref, "bpb_16k")
        speed = np.mean([
            r["bytes_per_second"] / next(z for z in stage_c if z["seed"] == r["seed"] and z["arm"] == ref)["bytes_per_second"]
            for r in stage_c if r["arm"] == arm
        ])
        lines.append(
            f"{arm:<30} vs {ref:<20} d8K={d8:+.5f}±{sd8:.5f} "
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
        lines.append("Only promoted mechanisms should be combined; do not combine failed arms post hoc.")
    else:
        lines.append("NO HYPOTHESIS CLEARED THE TWO-SEED PROMOTION GATE.")
        lines.append("Freeze the current 300M hybrid/attention-free implementations rather than overfitting the small arena.")
    lines.append(f"selected_main={selected_main} | selected_attentionfree={selected_af}")
    lines.append("=" * 245)
    return "\n".join(lines) + "\n"


# ======================================================================================
# CLI / main
# ======================================================================================


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=("all", "selftest", "summary"), default="all")
    p.add_argument("--outdir", default="./field_oracle_gap_ablation_v8")
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
    p.add_argument("--transition-rank", type=int, default=32)
    p.add_argument("--transition-scale", type=float, default=0.25)
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
    p.add_argument("--stage-a-min-speed", type=float, default=0.75)
    p.add_argument("--main-promotion-gain", type=float, default=0.012)
    p.add_argument("--af-promotion-gain", type=float, default=0.010)
    p.add_argument("--main-min-speed", type=float, default=0.85)
    p.add_argument("--af-min-speed", type=float, default=0.90)
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


def parse_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    args = parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(json.dumps({
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "args": vars(args),
    }, indent=2))
    if device.type == "cuda" and args.amp == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unavailable")

    shapes_path = out / "shapes.json"
    if shapes_path.exists() and args.resume:
        shapes = json.loads(shapes_path.read_text())
    else:
        shapes = resolve_shapes(args, device)
        atomic_json(shapes_path, shapes)

    if args.mode in {"all", "selftest"}:
        selftest(args, device, shapes)
        if args.mode == "selftest":
            return

    if args.mode == "summary":
        stage_a = json.loads((out / "stage_a.json").read_text())
        stage_b = json.loads((out / "stage_b.json").read_text())
        stage_c = json.loads((out / "stage_c.json").read_text())
        selection = json.loads((out / "selection.json").read_text())
        summary = make_summary(
            stage_a, stage_b, stage_c, shapes,
            selection["main"], selection["attention_free"], args,
        )
        atomic_text(out / "summary.txt", summary)
        log(summary)
        return

    train, val, _ = v3.load_wikitext103_raw(args.cache_dir, args.data_frac)
    train = v3.place_data(train, device, args.data_device, "train")
    val = v3.place_data(val, device, args.data_device, "validation")
    log(
        f"[protocol] arms={ALL_ARMS} stageA={args.stage_a_steps} stageB={args.stage_b_steps} "
        f"confirm={args.confirm_steps} bytes/update={args.batch_size*args.accum*args.seq_len:,}"
    )

    stage_a_path = out / "stage_a.json"
    if stage_a_path.exists() and args.resume:
        stage_a = json.loads(stage_a_path.read_text())
    else:
        stage_a = [
            train_arm(
                arm, args.screen_seed, args.stage_a_steps, args.stage_b_steps,
                shapes, args, train, val, device, out / "screen_runs",
            ) for arm in ALL_ARMS
        ]
        atomic_json(stage_a_path, stage_a)

    main_selected = rank_group(
        stage_a, HYBRID_REF, [a for a in HYBRID_ARMS if a != HYBRID_REF],
        args.main_finalists, args.stage_a_min_speed,
    )
    af_selected = rank_group(
        stage_a, AF_REF, [a for a in AF_ARMS if a != AF_REF],
        args.af_finalists, args.stage_a_min_speed,
    )
    selection = {"main": main_selected, "attention_free": af_selected}
    atomic_json(out / "selection.json", selection)
    log(f"[selection] main={main_selected} attention_free={af_selected}")

    stage_b_arms = [HYBRID_REF, *main_selected, AF_REF, *af_selected]
    stage_b_path = out / "stage_b.json"
    if stage_b_path.exists() and args.resume:
        stage_b = json.loads(stage_b_path.read_text())
    else:
        stage_b = [
            train_arm(
                arm, args.screen_seed, args.stage_b_steps, args.stage_b_steps,
                shapes, args, train, val, device, out / "screen_runs",
            ) for arm in stage_b_arms
        ]
        atomic_json(stage_b_path, stage_b)

    # Re-rank after the longer run; confirmation includes both best main hypotheses
    # and best attention-free hypothesis, always against same-seed references.
    main_selected = rank_group(
        stage_b, HYBRID_REF, main_selected, args.main_finalists, args.stage_a_min_speed
    )
    af_selected = rank_group(
        stage_b, AF_REF, af_selected, args.af_finalists, args.stage_a_min_speed
    )
    selection = {"main": main_selected, "attention_free": af_selected}
    atomic_json(out / "selection.json", selection)

    stage_c_path = out / "stage_c.json"
    if stage_c_path.exists() and args.resume:
        stage_c = json.loads(stage_c_path.read_text())
    else:
        stage_c = []
        for seed in parse_ints(args.confirm_seeds):
            for arm in [HYBRID_REF, *main_selected, AF_REF, *af_selected]:
                stage_c.append(train_arm(
                    arm, seed, args.confirm_steps, args.confirm_steps,
                    shapes, args, train, val, device, out / "confirm_runs",
                ))
        atomic_json(stage_c_path, stage_c)

    summary = make_summary(
        stage_a, stage_b, stage_c, shapes, main_selected, af_selected, args
    )
    atomic_text(out / "summary.txt", summary)
    log("\n" + summary)

    rows = []
    for stage_name, stage in (("A", stage_a), ("B", stage_b), ("C", stage_c)):
        for r in stage:
            rows.append({"stage": stage_name, **{k: v for k, v in r.items() if k != "history"}})
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as f:
        fields = sorted({k for r in rows for k in r})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
