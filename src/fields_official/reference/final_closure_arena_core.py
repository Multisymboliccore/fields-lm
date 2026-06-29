#!/usr/bin/env python3
"""Final paper-closure arena extensions.

Adds four scientifically explicit controls around the promoted 18F/2M/4R model:
1) identical Fields with PCAF enabled;
2) identical Fields trained from update zero with PCAF disabled;
3) the historical "pure PCAF" interpretation: a causal-convolution backbone plus
   the same validated successor-memory PCAF, parameter matched at BPE-16K scale;
4) a parameter-identical Flash-SDPA Transformer using Hugging Face's YaRN
   frequency construction (factor 32, 2K -> 64K);
5) official Mamba-2 from mamba-ssm.

The module wraps the frozen official arena instead of editing its promoted source.
"""
from __future__ import annotations
import math, gc
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import official_arena_core as base

# Re-export stable helpers, then replace only the model/control surface.
for _name, _value in vars(base).items():
    if not _name.startswith('__'):
        globals()[_name] = _value

FIELD_PCAF_ON = 'field_official_18f2m4r_pcaf_on'
FIELD_PCAF_OFF = 'field_official_18f2m4r_pcaf_off'
PCAF_CONV = 'pcaf_conv_300m_bpe16k'
TRANSFORMER_YARN = 'transformer_flash_yarn64k_300m'
MAMBA2 = 'mamba2_official_300m'
OFFICIAL_FIELD = FIELD_PCAF_ON
TRANSFORMER = TRANSFORMER_YARN
MODELS = (FIELD_PCAF_ON, FIELD_PCAF_OFF, PCAF_CONV, TRANSFORMER_YARN, MAMBA2)
DISPLAY_NAMES = {
    FIELD_PCAF_ON: 'Fields 18F/2M/4R + PCAF',
    FIELD_PCAF_OFF: 'Fields 18F/2M/4R — PCAF off from init',
    PCAF_CONV: 'PCAF-Conv matched 300M (BPE-16K)',
    TRANSFORMER_YARN: 'Transformer Flash + YaRN 2K→64K',
    MAMBA2: 'Official Mamba-2',
}
BACKEND_FIELD_ON = 'final_field_pcaf_on'
BACKEND_FIELD_OFF = 'final_field_pcaf_off'
BACKEND_PCAF_CONV = 'final_pcaf_conv'
BACKEND_YARN = 'final_transformer_yarn'
BACKEND_MAMBA2 = 'final_mamba2'

# Patch constants used by helpers whose globals live in official_arena_core.
base.OFFICIAL_FIELD = OFFICIAL_FIELD
base.TRANSFORMER = TRANSFORMER
base.MAMBA2 = MAMBA2
base.DISPLAY_NAMES = DISPLAY_NAMES
base.MODELS = MODELS


def create_base_args(args, stack):
    out = base.create_base_args(args, stack)
    out.yarn_factor = float(getattr(args, 'yarn_factor', 32.0))
    out.yarn_original_context = int(getattr(args, 'yarn_original_context', 2048))
    out.yarn_beta_fast = float(getattr(args, 'yarn_beta_fast', 32.0))
    out.yarn_beta_slow = float(getattr(args, 'yarn_beta_slow', 1.0))
    out.yarn_rope_theta = float(getattr(args, 'yarn_rope_theta', 10000.0))
    out.yarn_truncate = bool(getattr(args, 'yarn_truncate', True))
    out.yarn_gradient_checkpointing = bool(getattr(args, 'yarn_gradient_checkpointing', False))
    return out


def comparator_shapes(stack: Mapping[str, Any], base_args, deps) -> Dict[str, Any]:
    v25 = stack['v25']
    shapes = v25.solve_shapes_v25(base_args, deps)
    return {
        TRANSFORMER_YARN: shapes[v25.TRANSFORMER],
        MAMBA2: shapes[v25.MAMBA2],
        PCAF_CONV: SimpleNamespace(
            name=PCAF_CONV,
            params=int(base_args.target_params),
            dim=int(shapes[v25.TRANSFORMER].dim),
            layers=int(shapes[v25.TRANSFORMER].layers),
            heads=int(shapes[v25.TRANSFORMER].heads),
            ff_hidden=0,
        ),
    }


class CausalConvPCAFBlock(nn.Module):
    """Historical PCAF-Conv block: causal depthwise conv + GELU pointwise MLP."""
    def __init__(self, dim: int, hidden: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size, groups=dim, bias=True)
        self.pointwise = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        z = F.pad(z.transpose(1, 2), (self.kernel_size - 1, 0))
        z = self.depthwise(z).transpose(1, 2)
        return x + self.pointwise(z)


class PCAFConvLM(nn.Module):
    """Causal-conv parametric LM plus the exact promoted FastSuccessorCacheV5."""
    def __init__(self, *, vocab: int, dim: int, layers: int, hidden: int,
                 kernel_size: int, cache: nn.Module, v20) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([
            CausalConvPCAFBlock(dim, hidden, kernel_size)
            for _ in range(layers)
        ])
        self.final_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.cache = cache
        self.patch_position = -1
        self.softpatch = None
        self._patch_aux = None
        self.pcaf_conv_hidden = int(hidden)
        self.pcaf_conv_kernel = int(kernel_size)

    def states_logits(self, tokens: torch.Tensor):
        x = self.emb(tokens)
        self._patch_aux = x.new_zeros(())
        for block in self.blocks:
            x = block(x)
        return x, self.lm_head(self.final_norm(x))

    def forward(self, tokens: torch.Tensor, targets=None, compute_metrics: bool = False):
        states, logits = self.states_logits(tokens)
        if targets is None:
            return logits
        loss, primary, stats = self.cache(states, logits, tokens, targets, compute_metrics)
        return loss + self._patch_aux, primary, stats


def _pcaf_conv_param_formula(vocab: int, dim: int, layers: int, hidden: int,
                             kernel: int, cache_params: int) -> int:
    # Exact historical block count: LayerNorm(2d), depthwise(d*k+d),
    # Linear d->h (d*h+h), Linear h->d (h*d+d); final LayerNorm(2d).
    per_block = 2 * dim * hidden + hidden + dim * kernel + 4 * dim
    return vocab * dim + 2 * dim + layers * per_block + cache_params


def _solve_pcaf_hidden(target: int, vocab: int, dim: int, layers: int,
                       kernel: int, cache_params: int) -> Tuple[int, int]:
    approx = (target - vocab * dim - 2 * dim - cache_params - layers * (dim * kernel + 4 * dim)) / (layers * (2 * dim + 1))
    centre = max(64, int(round(approx / 16.0)) * 16)
    candidates = []
    for h in range(max(64, centre - 256), centre + 257, 16):
        p = _pcaf_conv_param_formula(vocab, dim, layers, h, kernel, cache_params)
        candidates.append((abs(p - target), h, p))
    _, hidden, params = min(candidates)
    return int(hidden), int(params)


_YARN_CACHE = {}

def yarn_inv_freq(*, device: torch.device, head_dim: int, factor: float,
                  original_context: int, beta_fast: float, beta_slow: float,
                  rope_theta: float, truncate: bool) -> Tuple[torch.Tensor, float]:
    """HF-compatible YaRN inverse frequencies and attention scaling."""
    if head_dim % 2:
        raise ValueError('YaRN head_dim must be even')
    def get_mscale(scale: float) -> float:
        return 1.0 if scale <= 1.0 else 0.1 * math.log(scale) + 1.0
    attention_factor = get_mscale(factor)
    def correction_dim(rotations: float) -> float:
        return (head_dim * math.log(original_context / (rotations * 2.0 * math.pi))) / (2.0 * math.log(rope_theta))
    low = correction_dim(beta_fast)
    high = correction_dim(beta_slow)
    if truncate:
        low, high = math.floor(low), math.ceil(high)
    low = max(low, 0.0)
    high = min(high, float(head_dim - 1))
    pos_freqs = rope_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    inv_extrap = 1.0 / pos_freqs
    inv_interp = 1.0 / (factor * pos_freqs)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(head_dim // 2, device=device, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
    extrap_factor = 1.0 - ramp
    inv = inv_interp * (1.0 - extrap_factor) + inv_extrap * extrap_factor
    return inv, float(attention_factor)


def yarn_cos_sin(device: torch.device, dtype: torch.dtype, length: int,
                  head_dim: int, factor: float, original_context: int,
                  beta_fast: float, beta_slow: float, rope_theta: float,
                  truncate: bool):
    key = (str(device), dtype, int(length), int(head_dim), float(factor),
           int(original_context), float(beta_fast), float(beta_slow),
           float(rope_theta), bool(truncate))
    cached = _YARN_CACHE.get(key)
    if cached is not None:
        return cached
    inv, attention_factor = yarn_inv_freq(
        device=device, head_dim=head_dim, factor=factor,
        original_context=original_context, beta_fast=beta_fast,
        beta_slow=beta_slow, rope_theta=rope_theta, truncate=truncate,
    )
    pos = torch.arange(length, device=device, dtype=torch.float32)
    freq = torch.outer(pos, inv)
    cos = (freq.cos() * attention_factor).to(dtype)[None, None, :, :]
    sin = (freq.sin() * attention_factor).to(dtype)[None, None, :, :]
    _YARN_CACHE[key] = (cos, sin)
    return cos, sin


class YarnFlashBlock300M(nn.Module):
    def __init__(self, dim: int, heads: int, ff_hidden: int, *, yarn: Mapping[str, Any], v20) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError('dim must divide heads')
        self.dim = int(dim); self.heads = int(heads); self.head_dim = dim // heads
        self.norm1 = v20.NativeRMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = v20.NativeRMSNorm(dim)
        self.ff = v20.PackedSwiGLU(dim, ff_hidden)
        self.yarn = dict(yarn)
        self._apply_rope = v20.apply_rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.qkv(self.norm1(x)).view(b, t, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2).contiguous(); k = k.transpose(1, 2).contiguous(); v = v.transpose(1, 2).contiguous()
        cos, sin = yarn_cos_sin(x.device, x.dtype, t, self.head_dim, **self.yarn)
        q = self._apply_rope(q, cos, sin); k = self._apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(b, t, self.dim)
        x = x + self.proj(y)
        return x + self.ff(self.norm2(x))


class YarnFlashTransformer300M(nn.Module):
    def __init__(self, *, vocab: int, dim: int, heads: int, layers: int,
                 ff_hidden: int, yarn: Mapping[str, Any], v20,
                 gradient_checkpointing: bool) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([
            YarnFlashBlock300M(dim, heads, ff_hidden, yarn=yarn, v20=v20)
            for _ in range(layers)
        ])
        self.final_norm = v20.NativeRMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.yarn_config = dict(yarn)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.emb(tokens)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, use_reentrant=False, preserve_rng_state=False)
            else:
                x = block(x)
        return self.lm_head(self.final_norm(x))


def _build_yarn(stack, shape, args, device):
    v20 = stack['v23'].v20
    seed_all(args.model_seed)
    yarn = dict(
        factor=float(args.yarn_factor),
        original_context=int(args.yarn_original_context),
        beta_fast=float(args.yarn_beta_fast),
        beta_slow=float(args.yarn_beta_slow),
        rope_theta=float(args.yarn_rope_theta),
        truncate=bool(args.yarn_truncate),
    )
    model = YarnFlashTransformer300M(
        vocab=args.vocab_size, dim=shape.dim, heads=shape.heads,
        layers=shape.layers, ff_hidden=shape.ff_hidden, yarn=yarn, v20=v20,
        gradient_checkpointing=bool(args.yarn_gradient_checkpointing),
    ).to(device)
    v20.tied_embedding_init(model, args.embedding_seed)
    return model


def _build_pcaf_conv(stack, control_spec, field_shape, base_args, deps, device):
    # Build the exact cache through the promoted constructor; no checkpoint is loaded.
    template = base.build_official_field(stack, control_spec, field_shape, base_args, deps, device)
    # Match the complete promoted Fields model, including its PCAF parameters.
    # The cache object is then transplanted unchanged into the causal-conv control.
    target = nparams(template)
    cache = template.cache
    template.cache = nn.Identity()
    v20 = stack['v23'].v20
    dim = int(field_shape.dim)
    layers = int(field_shape.layers)
    kernel = 5
    cache_params = nparams(cache)
    hidden, formula_params = _solve_pcaf_hidden(target, base_args.vocab_size, dim, layers, kernel, cache_params)
    model = PCAFConvLM(
        vocab=base_args.vocab_size, dim=dim, layers=layers, hidden=hidden,
        kernel_size=kernel, cache=cache, v20=v20,
    ).to(device)
    v20.tied_embedding_init(model, base_args.embedding_seed)
    model.cache.enabled = True
    actual = nparams(model)
    delta_pct = 100.0 * (actual - target) / target
    model.parameter_audit = {
        'target': target, 'actual': actual, 'formula': formula_params,
        'delta_pct': delta_pct,
        'hidden': hidden, 'layers': layers, 'kernel': kernel,
        'cache_params': cache_params,
    }
    if abs(delta_pct) > float(base_args.max_param_delta_pct):
        raise RuntimeError(
            f'PCAF-Conv parameter mismatch {delta_pct:+.4f}% exceeds '
            f'{float(base_args.max_param_delta_pct):.4f}%: target={target:,} actual={actual:,}'
        )
    del template
    gc.collect()
    return model


def build_model(model_name: str, stack: Mapping[str, Any], control_spec: Any,
                field_shape: Any, comp_shapes: Mapping[str, Any], base_args,
                deps: Any, device: torch.device):
    if model_name in {FIELD_PCAF_ON, FIELD_PCAF_OFF}:
        model = base.build_official_field(stack, control_spec, field_shape, base_args, deps, device)
        enabled = model_name == FIELD_PCAF_ON
        if not hasattr(model, 'cache') or not hasattr(model.cache, 'enabled'):
            raise RuntimeError('promoted Fields model has no switchable PCAF cache')
        model.cache.enabled = enabled
        model._final_arena_model_name = model_name
        model._final_arena_pcaf_enabled = enabled
        return model, BACKEND_FIELD_ON if enabled else BACKEND_FIELD_OFF
    if model_name == PCAF_CONV:
        model = _build_pcaf_conv(stack, control_spec, field_shape, base_args, deps, device)
        model._final_arena_model_name = model_name
        return model, BACKEND_PCAF_CONV
    if model_name == TRANSFORMER_YARN:
        model = _build_yarn(stack, comp_shapes[TRANSFORMER_YARN], base_args, device)
        model._final_arena_model_name = model_name
        return model, BACKEND_YARN
    if model_name == MAMBA2:
        v25 = stack['v25']
        model = v25.build_model_v25(v25.MAMBA2, comp_shapes[MAMBA2], base_args, deps, device)
        model._final_arena_model_name = model_name
        return model, BACKEND_MAMBA2
    raise KeyError(model_name)


def topology_audit(model: nn.Module):
    return base.topology_audit(model)


def loss_call(stack: Mapping[str, Any], backend_name: str, model: nn.Module,
              x: torch.Tensor, y: torch.Tensor):
    """Dispatch loss through each model's native public API.

    The promoted FieldTokenSystemV21 intentionally exposes ``loss_and_stats``
    rather than ``forward``.  Calling it as a regular nn.Module therefore raises
    ``NotImplementedError`` even though the model itself is valid.  Keep the
    arena wrapper API-aware instead of modifying the frozen promoted model.
    """
    if backend_name in {BACKEND_FIELD_ON, BACKEND_FIELD_OFF, BACKEND_PCAF_CONV}:
        if hasattr(model, 'loss_and_stats'):
            result = model.loss_and_stats(x, y, compute_metrics=False)
        else:
            result = model(x, y, False)
        if not isinstance(result, (tuple, list)) or len(result) < 1:
            raise TypeError(
                f'field-like loss API returned an invalid value: '
                f'backend={backend_name} type={type(result).__name__}'
            )
        return result[0]
    logits = model(x)
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))


def _hidden_field_like(model: nn.Module, tokens: torch.Tensor):
    x = model.emb(tokens)
    if hasattr(model, '_patch_aux'):
        model._patch_aux = x.new_zeros(())
    patch_position = int(getattr(model, 'patch_position', -1))
    softpatch = getattr(model, 'softpatch', None)
    for i, block in enumerate(model.blocks):
        x = block(x)
        if i == patch_position and softpatch is not None:
            x = softpatch(x, tokens)
            if hasattr(model, '_patch_aux') and hasattr(softpatch, 'last_aux'):
                model._patch_aux = softpatch.last_aux
    return x, model.final_norm(x)


def streaming_token_nll(stack: Mapping[str, Any], backend_name: str,
                        model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                        chunk: int, return_tokens: bool):
    v27 = stack['v27']
    if backend_name == BACKEND_FIELD_OFF:
        _, hidden = _hidden_field_like(model, x)
        return v27.generic_chunked_ce(model, hidden, y, chunk, return_tokens=return_tokens)
    if backend_name in {BACKEND_FIELD_ON, BACKEND_PCAF_CONV}:
        states, hidden = _hidden_field_like(model, x)
        return v27.field_chunked_pcaf_nll(model, states, hidden, x, y, chunk, return_tokens=return_tokens)
    if backend_name == BACKEND_YARN:
        hidden = model.emb(x)
        for block in model.blocks:
            hidden = block(hidden)
        hidden = model.final_norm(hidden)
        return v27.generic_chunked_ce(model, hidden, y, chunk, return_tokens=return_tokens)
    if backend_name == BACKEND_MAMBA2:
        hidden = model.emb(x).to(model.activation_dtype)
        for block in model.blocks:
            hidden = block(hidden)
        hidden = model.norm(hidden)
        return v27.generic_chunked_ce(model, hidden, y, chunk, return_tokens=return_tokens)
    raise KeyError(backend_name)


def model_audit(model_name: str, model: nn.Module) -> Dict[str, Any]:
    row = {'model': model_name, 'parameters': nparams(model)}
    if model_name in {FIELD_PCAF_ON, FIELD_PCAF_OFF}:
        row['topology'] = topology_audit(model)
        row['pcaf_enabled'] = bool(model.cache.enabled)
    elif model_name == PCAF_CONV:
        row.update(dict(model.parameter_audit))
        row['pcaf_enabled'] = bool(model.cache.enabled)
    elif model_name == TRANSFORMER_YARN:
        row['yarn'] = dict(model.yarn_config)
        row['gradient_checkpointing'] = bool(model.gradient_checkpointing)
    return row
