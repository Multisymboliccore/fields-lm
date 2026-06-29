#!/usr/bin/env python3
"""Token-level canonical perplexity pilot for the best validated Field.

Purpose
-------
This is deliberately a short bridge from the byte-level research harness to a
more standard token-LM protocol. It trains one shared GPT-2-style byte-level BPE
(16k by default) on the training split, then compares:

  * field_selective_token       — validated selective-residual Field control
  * field_quality_span2_token   — surface multiview + verified two-token span
  * transformer_flash_token    — strong Flash-SDPA Transformer reference

All models share the exact tokenizer, token windows, update budget, parameter
target, tied input/output embeddings, optimizer policy and evaluation targets.
Primary quality is TEST perplexity at the training context. A normalized BPB is
also reported so this run remains connected to the byte-level line. Long-context
rows use the same suffix targets while only the available history changes.

The validated Field recurrence and memory dependencies are packed beside this
file. The canonical Triton Field source is still verified by SHA at runtime.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
V15_PATH = HERE / "field_scale_50m_v15.py"
CANONICAL_NAME = "field_only_v4_chunked_triton_wiki100.py"
EXPECTED_CANONICAL_SHA256 = "0454647a7f667cdf73f45029eee0f525c58c0788ecd433ec1a4db8c3937ac848"
LN2 = math.log(2.0)

FIELD_CONTROL = "field_selective_token"
FIELD_QUALITY = "field_quality_span2_token"
TRANSFORMER = "transformer_flash_token"
MODEL_NAMES = (FIELD_CONTROL, FIELD_QUALITY, TRANSFORMER)


def log(msg: object = "") -> None:
    print(str(msg), flush=True)


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


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


def locate_canonical(explicit: str) -> Path:
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend([
        HERE / CANONICAL_NAME,
        Path("/home/ubuntu") / CANONICAL_NAME,
        Path("/home/ubuntu/field_memory_consolidation_canonical_50m_v13") / CANONICAL_NAME,
        Path("/home/ubuntu/field_hybrid_canonical_50m_bridge_v4") / CANONICAL_NAME,
        Path("/home/ubuntu/field_hybrid_300m_h2h_v7") / CANONICAL_NAME,
        Path("/home/ubuntu/field_pcaf_efficiency_v1") / CANONICAL_NAME,
    ])
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError("validated canonical Field source not found")


def patch_vocab(vocab_size: int, runtime_root: Path, canonical_path: Path) -> List[str]:
    """Patch module-level VOCAB constants before any model is instantiated."""
    changed: List[str] = []
    roots = (str(runtime_root.resolve()), str(canonical_path.resolve()))
    for name, module in list(sys.modules.items()):
        if module is None or not hasattr(module, "VOCAB"):
            continue
        file = getattr(module, "__file__", "") or ""
        if file and (file.startswith(roots[0]) or file == roots[1]):
            setattr(module, "VOCAB", int(vocab_size))
            changed.append(name)
    return sorted(set(changed))


def tie_embeddings(model: nn.Module, initialize: bool = False) -> None:
    emb = getattr(model, "emb", None)
    if emb is None:
        raise AttributeError("model has no emb")
    if initialize:
        # nn.Embedding defaults to N(0, 1), which is far too large when the
        # same matrix is reused as the language-model head.  With RMS-normalized
        # hidden states it creates logits with O(sqrt(dim)) scale and an initial
        # NLL far above log(vocab).  GPT-style tied token embeddings use a small
        # standard deviation so the initial distribution is near-uniform.
        nn.init.normal_(emb.weight, mean=0.0, std=0.02)
    if hasattr(model, "lm_head"):
        model.lm_head.weight = emb.weight
    elif hasattr(model, "head"):
        model.head.weight = emb.weight
    else:
        raise AttributeError("model has no output head")


def embeddings_tied(model: nn.Module) -> bool:
    head = getattr(model, "lm_head", None) or getattr(model, "head", None)
    return bool(head is not None and head.weight.data_ptr() == model.emb.weight.data_ptr())


@dataclass(frozen=True)
class Shape:
    name: str
    params: int
    dim: int
    layers: int
    heads: int
    ff_hidden: int


@dataclass
class Corpus:
    tokens: torch.Tensor
    raw_bytes: int
    rows: int

    @property
    def bytes_per_token(self) -> float:
        return self.raw_bytes / max(int(self.tokens.numel()), 1)


def load_raw_rows(cache_dir: str, data_frac: float) -> Tuple[List[str], List[str], List[str]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Missing datasets package in the validated environment") from exc
    log("[data] loading Salesforce/wikitext wikitext-103-raw-v1")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", cache_dir=cache_dir)

    def nonempty(split) -> List[str]:
        return [str(row.get("text", "")) + "\n" for row in split if row.get("text", "")]

    all_train = nonempty(ds["train"])
    val = nonempty(ds["validation"])
    test = nonempty(ds["test"])
    if not 0.0 < data_frac <= 1.0:
        raise ValueError("data_frac must be in (0,1]")
    total_bytes = sum(len(x.encode("utf-8", errors="replace")) for x in all_train)
    limit = max(2, int(total_bytes * data_frac))
    train: List[str] = []
    seen = 0
    for text in all_train:
        train.append(text)
        seen += len(text.encode("utf-8", errors="replace"))
        if seen >= limit:
            break
    log(f"[data] rows train={len(train):,} val={len(val):,} test={len(test):,} train_bytes={seen:,}")
    return train, val, test


def build_or_load_tokenizer(root: Path, train_rows: Sequence[str], vocab_size: int, min_frequency: int):
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    except ImportError as exc:
        raise RuntimeError("Missing tokenizers package in the validated environment") from exc

    tok_path = root / "tokenizer" / "tokenizer.json"
    tok_path.parent.mkdir(parents=True, exist_ok=True)
    if tok_path.is_file():
        tok = Tokenizer.from_file(str(tok_path))
        if tok.get_vocab_size() != vocab_size:
            raise RuntimeError(
                f"tokenizer vocab mismatch cached={tok.get_vocab_size()} requested={vocab_size}"
            )
        log(f"[tokenizer] loaded {tok_path} vocab={tok.get_vocab_size():,}")
        return tok

    log(f"[tokenizer] training GPT-2-style byte-level BPE vocab={vocab_size:,}")
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
        special_tokens=["<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(iter(train_rows), trainer=trainer, length=len(train_rows))
    if tok.get_vocab_size() != vocab_size:
        raise RuntimeError(f"tokenizer produced vocab={tok.get_vocab_size()} expected={vocab_size}")
    tok.save(str(tok_path))
    sample = "Field token-level roundtrip: café, ação, 123.\n"
    decoded = tok.decode(tok.encode(sample).ids)
    if decoded != sample:
        raise AssertionError(f"tokenizer roundtrip mismatch: {decoded!r}")
    log(f"[tokenizer] saved {tok_path}")
    return tok


def encode_rows(tok, rows: Sequence[str], batch_rows: int = 256) -> Corpus:
    ids: List[int] = []
    raw_bytes = 0
    for i in range(0, len(rows), batch_rows):
        chunk = list(rows[i : i + batch_rows])
        enc = tok.encode_batch(chunk, add_special_tokens=False)
        for text, item in zip(chunk, enc):
            ids.extend(item.ids)
            raw_bytes += len(text.encode("utf-8", errors="replace"))
    if len(ids) < 2:
        raise RuntimeError("encoded corpus is empty")
    return Corpus(torch.tensor(ids, dtype=torch.int32), raw_bytes, len(rows))


def save_or_load_corpora(root: Path, tok, rows: Tuple[Sequence[str], Sequence[str], Sequence[str]]) -> Tuple[Corpus, Corpus, Corpus]:
    data_dir = root / "tokenized"
    data_dir.mkdir(parents=True, exist_ok=True)
    out: List[Corpus] = []
    for name, split_rows in zip(("train", "validation", "test"), rows):
        pt = data_dir / f"{name}.pt"
        meta = data_dir / f"{name}.json"
        if pt.is_file() and meta.is_file():
            tokens = torch.load(pt, map_location="cpu", weights_only=True)
            info = json.loads(meta.read_text(encoding="utf-8"))
            corpus = Corpus(tokens.to(torch.int32).contiguous(), int(info["raw_bytes"]), int(info["rows"]))
        else:
            log(f"[tokenizer] encoding {name}")
            corpus = encode_rows(tok, split_rows)
            torch.save(corpus.tokens, pt)
            atomic_json(meta, {"raw_bytes": corpus.raw_bytes, "rows": corpus.rows, "tokens": int(corpus.tokens.numel())})
        log(
            f"[tokenizer] {name}: tokens={corpus.tokens.numel():,} raw_bytes={corpus.raw_bytes:,} "
            f"bytes/token={corpus.bytes_per_token:.3f}"
        )
        out.append(corpus)
    return out[0], out[1], out[2]


def place_tokens(tokens: torch.Tensor, device: torch.device, mode: str, name: str) -> torch.Tensor:
    if mode == "cuda" or (mode == "auto" and device.type == "cuda"):
        out = tokens.to(device=device, dtype=torch.int32)
        log(f"[data] {name} tokens -> GPU ({out.numel()*out.element_size()/2**20:.1f} MiB)")
        return out
    out = tokens.contiguous()
    if device.type == "cuda":
        try:
            out = out.pin_memory()
        except RuntimeError:
            pass
    log(f"[data] {name} tokens kept on CPU")
    return out


def make_runtime_args(args, field_hidden: int, tf_hidden: int) -> argparse.Namespace:
    ns = argparse.Namespace(**vars(args))
    ns.hybrid_ff_hidden = int(field_hidden)
    ns.field_ff_hidden = int(field_hidden)
    ns.af_ff_hidden = int(field_hidden)
    ns.tf_ff_hidden = int(tf_hidden)
    return ns


def build_quality_field(arena, args, v3, canonical, bridge, optmod, epi, device, hidden: int):
    del epi
    arena.base.seed_all(args.model_seed)
    bargs = arena.base.make_bridge_args(args, hidden)
    model = bridge.build_field("hybrid_w256_conf_parity", bargs, v3, canonical, hidden).to(device)
    optmod.replace_softpatch(model, v3)
    optmod.replace_cache(model, v3, i32=True)
    model.cache.FEATURE_DIM = int(v3.SuccessorCacheV5.FEATURE_DIM)
    optmod.replace_local(model, v3, "cached", args.local_chunk)
    arena._install_cloud_fast_route(optmod)
    surface = arena.cloud.make_surface_multiview_cache(model.cache, args)
    model.cache = arena.cloud.CloudMechanismCache(
        surface,
        delta_rank=None,
        span_max=int(args.span_tokens),
        phase=False,
        args=args,
    ).to(device)
    return model


def build_model(name: str, shape: Shape, args, arena, v3, canonical, bridge, optmod, epi, judge, device):
    run_args = make_runtime_args(args, shape.ff_hidden if name != TRANSFORMER else args.field_ff_hidden, shape.ff_hidden if name == TRANSFORMER else args.tf_ff_hidden)
    seed_all(args.model_seed)
    if name == FIELD_CONTROL:
        model = arena.build_field(arena.SELECTIVE, run_args, v3, canonical, bridge, optmod, epi, device)
    elif name == FIELD_QUALITY:
        model = build_quality_field(arena, run_args, v3, canonical, bridge, optmod, epi, device, shape.ff_hidden)
    elif name == TRANSFORMER:
        run_args.tf_dim = shape.dim
        run_args.tf_heads = shape.heads
        run_args.tf_layers = shape.layers
        run_args.tf_ff_hidden = shape.ff_hidden
        model = arena.base.build_transformer(run_args, judge, v3, device)
    else:
        raise KeyError(name)
    tie_embeddings(model, initialize=True)
    return model


def count_model(name: str, hidden: int, args, arena, v3, canonical, bridge, optmod, epi, judge) -> int:
    if name == TRANSFORMER:
        shape = Shape(name, 0, args.tf_dim, args.tf_layers, args.tf_heads, hidden)
    else:
        shape = Shape(name, 0, args.field_dim, args.field_layers, args.field_heads, hidden)
    model = build_model(name, shape, args, arena, v3, canonical, bridge, optmod, epi, judge, torch.device("cpu"))
    value = nparams(model)
    del model
    gc.collect()
    return value


def solve_hidden(name: str, target: int, args, arena, v3, canonical, bridge, optmod, epi, judge) -> Tuple[int, int]:
    lo = 512
    hi = 4096
    p_lo = count_model(name, lo, args, arena, v3, canonical, bridge, optmod, epi, judge)
    p_next = count_model(name, lo + 16, args, arena, v3, canonical, bridge, optmod, epi, judge)
    slope = (p_next - p_lo) / 16.0
    if slope <= 0:
        raise RuntimeError(f"non-positive parameter slope for {name}")
    guess = int(round((lo + (target - p_lo) / slope) / 16.0) * 16)
    guess = max(lo, min(hi, guess))
    candidates = sorted(set(max(lo, min(hi, guess + 16 * d)) for d in range(-4, 5)))
    rows = [(h, count_model(name, h, args, arena, v3, canonical, bridge, optmod, epi, judge)) for h in candidates]
    return min(rows, key=lambda hp: abs(hp[1] - target))


def resolve_shapes(args, arena, v3, canonical, bridge, optmod, epi, judge) -> Dict[str, Shape]:
    field_hidden, field_params = solve_hidden(FIELD_CONTROL, args.target_params, args, arena, v3, canonical, bridge, optmod, epi, judge)
    quality_same = count_model(FIELD_QUALITY, field_hidden, args, arena, v3, canonical, bridge, optmod, epi, judge)
    if abs(quality_same - args.target_params) / args.target_params <= args.max_param_delta_pct / 100.0:
        quality_hidden, quality_params = field_hidden, quality_same
    else:
        quality_hidden, quality_params = solve_hidden(FIELD_QUALITY, args.target_params, args, arena, v3, canonical, bridge, optmod, epi, judge)
    tf_hidden, tf_params = solve_hidden(TRANSFORMER, args.target_params, args, arena, v3, canonical, bridge, optmod, epi, judge)
    shapes = {
        FIELD_CONTROL: Shape(FIELD_CONTROL, field_params, args.field_dim, args.field_layers, args.field_heads, field_hidden),
        FIELD_QUALITY: Shape(FIELD_QUALITY, quality_params, args.field_dim, args.field_layers, args.field_heads, quality_hidden),
        TRANSFORMER: Shape(TRANSFORMER, tf_params, args.tf_dim, args.tf_layers, args.tf_heads, tf_hidden),
    }
    for shape in shapes.values():
        delta = 100.0 * (shape.params - args.target_params) / args.target_params
        if abs(delta) > args.max_param_delta_pct:
            raise RuntimeError(f"parameter mismatch {shape.name}: {delta:+.3f}%")
    return shapes


def set_distill(name: str, model: nn.Module, step: int, args) -> None:
    if name == TRANSFORMER:
        return
    value = min(1.0, step / max(float(args.conf_distill_ramp), 1.0))
    cache = getattr(model, "cache", None)
    seen = set()
    while cache is not None and id(cache) not in seen:
        seen.add(id(cache))
        if hasattr(cache, "distill_scale"):
            try:
                cache.distill_scale = value
            except Exception:
                pass
        cache = getattr(cache, "base", None)


def token_nll(name: str, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if name == TRANSFORMER:
        logits = model(x)
        return F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none"
        ).view_as(y)
    states, logits = model.states_logits(x)
    return model.cache.token_nll(states, logits, x, y).float()


def training_loss(name: str, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if name == TRANSFORMER:
        logits = model(x)
        primary = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))
        return primary, primary.detach()
    loss, primary, _ = model.loss_and_stats(x, y, compute_metrics=False)
    return loss, primary.detach()


def amp_ctx(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16 if amp == "bf16" else torch.float16)


def make_optimizer(model: nn.Module, lr: float, weight_decay: float):
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    kwargs = {"lr": lr, "betas": (0.9, 0.95), "eps": 1e-8}
    if torch.cuda.is_available():
        kwargs["fused"] = True
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], **kwargs)


def lr_at(step: int, total: int, warmup: int, peak: float, min_ratio: float) -> float:
    if step <= warmup:
        return peak * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return peak * (min_ratio + (1.0 - min_ratio) * cosine)


def batch_for_step(data: torch.Tensor, batch: int, seq: int, seed: int, step: int, micro: int, device: torch.device):
    gen_device = data.device if data.device.type == "cuda" else torch.device("cpu")
    gen = torch.Generator(device=gen_device)
    gen.manual_seed(seed + step * 1_000_003 + micro * 9_176)
    starts = torch.randint(0, len(data) - seq - 1, (batch,), generator=gen, device=data.device)
    offsets = torch.arange(seq + 1, device=data.device, dtype=torch.long)
    win = data[starts[:, None] + offsets[None, :]].long()
    if win.device != device:
        win = win.to(device, non_blocking=True)
    return win[:, :-1], win[:, 1:]


@torch.no_grad()
def evaluate_windows(name: str, model: nn.Module, data: torch.Tensor, context: int, windows: int, seed: int, device: torch.device, amp: str) -> Dict[str, float]:
    model.eval()
    rng = np.random.default_rng(seed + context * 1009)
    starts = rng.integers(0, len(data) - context - 1, size=windows).tolist()
    total_nll = 0.0
    total_tokens = 0
    for start in starts:
        win = data[start : start + context + 1].long()
        if win.device != device:
            win = win.to(device, non_blocking=True)
        x, y = win[:-1][None, :], win[1:][None, :]
        with amp_ctx(device, amp):
            nll = token_nll(name, model, x, y)
        total_nll += float(nll.sum())
        total_tokens += int(nll.numel())
    mean = total_nll / max(total_tokens, 1)
    return {"context": context, "nll": mean, "ppl": math.exp(min(mean, 20.0)), "bits_per_token": mean / LN2, "tokens": total_tokens}


@torch.no_grad()
def evaluate_test_stream(name: str, model: nn.Module, corpus: Corpus, data: torch.Tensor, context: int, token_budget: int, device: torch.device, amp: str) -> Dict[str, float]:
    model.eval()
    usable = min(len(data) - 1, token_budget if token_budget > 0 else len(data) - 1)
    total_nll = 0.0
    total_tokens = 0
    for start in range(0, usable, context):
        length = min(context, usable - start)
        if length < 8:
            break
        win = data[start : start + length + 1].long()
        if win.device != device:
            win = win.to(device, non_blocking=True)
        x, y = win[:-1][None, :], win[1:][None, :]
        with amp_ctx(device, amp):
            nll = token_nll(name, model, x, y)
        total_nll += float(nll.sum())
        total_tokens += int(nll.numel())
    mean = total_nll / max(total_tokens, 1)
    bytes_est = total_tokens * corpus.bytes_per_token
    return {
        "context": context,
        "nll": mean,
        "ppl": math.exp(min(mean, 20.0)),
        "bits_per_token": mean / LN2,
        "bpb_norm": (total_nll / LN2) / max(bytes_est, 1e-9),
        "tokens": total_tokens,
        "bytes_est": bytes_est,
    }


@torch.no_grad()
def evaluate_matched_suffix(name: str, model: nn.Module, data: torch.Tensor, contexts: Sequence[int], score_tokens: int, windows: int, seed: int, device: torch.device, amp: str) -> List[Dict[str, float]]:
    max_ctx = max(contexts)
    if score_tokens >= min(contexts):
        raise ValueError("score_tokens must be smaller than every context")
    rng = np.random.default_rng(seed)
    ends = rng.integers(max_ctx + 1, len(data) - 1, size=windows).tolist()
    rows: List[Dict[str, float]] = []
    model.eval()
    for context in contexts:
        total_nll = 0.0
        total = 0
        for end in ends:
            start = end - context
            win = data[start : end + 1].long()
            if win.device != device:
                win = win.to(device, non_blocking=True)
            x, y = win[:-1][None, :], win[1:][None, :]
            with amp_ctx(device, amp):
                nll = token_nll(name, model, x, y)[:, -score_tokens:]
            total_nll += float(nll.sum())
            total += int(nll.numel())
        mean = total_nll / max(total, 1)
        rows.append({
            "context": int(context),
            "score_tokens": int(score_tokens),
            "windows": int(windows),
            "nll": mean,
            "ppl": math.exp(min(mean, 20.0)),
            "bits_per_token": mean / LN2,
        })
    return rows


def checkpoint_signature(args, name: str, shape: Shape) -> Dict[str, object]:
    return {
        "name": name,
        "shape": asdict(shape),
        "vocab_size": args.vocab_size,
        "train_seq": args.train_seq,
        "batch": args.batch_size,
        "accum": args.accum,
        "train_steps": args.train_steps,
        "model_seed": args.model_seed,
        "data_seed": args.data_seed,
        "span_tokens": args.span_tokens,
    }


def train_one(name: str, shape: Shape, args, arena, v3, canonical, bridge, optmod, epi, judge, train: torch.Tensor, val: torch.Tensor, test_corpus: Corpus, test: torch.Tensor, root: Path, device: torch.device) -> Dict[str, object]:
    out = root / "models" / name
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "result.json"
    if result_path.is_file() and args.resume:
        return json.loads(result_path.read_text(encoding="utf-8"))

    model = build_model(name, shape, args, arena, v3, canonical, bridge, optmod, epi, judge, device)
    lr = args.transformer_lr if name == TRANSFORMER else args.field_lr
    opt = make_optimizer(model, lr, args.weight_decay)
    signature = checkpoint_signature(args, name, shape)
    ckpt = out / "latest.pt"
    start_step = 0
    if ckpt.is_file() and args.resume:
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        if payload.get("signature") != signature:
            raise RuntimeError(f"checkpoint signature mismatch: {name}")
        model.load_state_dict(payload["model"], strict=True)
        tie_embeddings(model)
        opt.load_state_dict(payload["optimizer"])
        start_step = int(payload["step"])
        log(f"[{name}] resume {start_step}/{args.train_steps}")

    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sync(device)
    started = time.perf_counter()
    excluded = 0.0
    processed = 0
    best = float("inf")

    for step in range(start_step + 1, args.train_steps + 1):
        set_distill(name, model, step, args)
        opt.zero_grad(set_to_none=True)
        primary_sum = 0.0
        for micro in range(args.accum):
            x, y = batch_for_step(train, args.batch_size, args.train_seq, args.data_seed, step, micro, device)
            with amp_ctx(device, args.amp):
                loss, primary = training_loss(name, model, x, y)
                scaled = loss / args.accum
            scaled.backward()
            primary_sum += float(primary)
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
        now_lr = lr_at(step, args.train_steps, args.warmup, lr, args.min_lr_ratio)
        for group in opt.param_groups:
            group["lr"] = now_lr
        opt.step()
        processed += args.batch_size * args.accum * args.train_seq

        if step % args.log_every == 0 or step == args.train_steps:
            sync(device)
            elapsed = time.perf_counter() - started - excluded
            tps = processed / max(elapsed, 1e-9)
            nll = primary_sum / args.accum
            peak = torch.cuda.max_memory_allocated() / 2**30
            log(
                f"[{name}] step={step:05d}/{args.train_steps} nll={nll:.4f} "
                f"ppl={math.exp(min(nll,20.0)):.3f} grad={grad:.3f} lr={now_lr:.3e} "
                f"tok/s={tps:,.0f} peak={peak:.2f}G"
            )

        if step % args.eval_every == 0 or step == args.train_steps:
            t0 = time.perf_counter()
            row = evaluate_windows(name, model, val, args.train_seq, args.quick_eval_windows, args.eval_seed, device, args.amp)
            best = min(best, row["ppl"])
            log(f"[{name}] VAL step={step:05d} ppl={row['ppl']:.4f} bpt={row['bits_per_token']:.4f} best={best:.4f}")
            model.train()
            excluded += time.perf_counter() - t0

        if step % args.save_every == 0 or step == args.train_steps:
            tmp = ckpt.with_suffix(".tmp")
            torch.save({
                "signature": signature,
                "step": step,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
            }, tmp)
            os.replace(tmp, ckpt)

    sync(device)
    elapsed = time.perf_counter() - started - excluded
    test_row = evaluate_test_stream(name, model, test_corpus, test, args.train_seq, args.test_token_budget, device, args.amp)
    matched = evaluate_matched_suffix(name, model, test, args.matched_contexts, args.matched_score_tokens, args.matched_windows, args.eval_seed + 70000, device, args.amp)
    result: Dict[str, object] = {
        "model": name,
        "shape": asdict(shape),
        "lr": lr,
        "steps": args.train_steps,
        "best_validation_ppl": best,
        "train_tokens_per_second": processed / max(elapsed, 1e-9),
        "train_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "test": test_row,
        "matched_suffix": matched,
    }
    atomic_json(result_path, result)
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return result


def systems_benchmark(name: str, shape: Shape, args, arena, v3, canonical, bridge, optmod, epi, judge, context: int, bytes_per_token: float, device: torch.device) -> Dict[str, object]:
    model = build_model(name, shape, args, arena, v3, canonical, bridge, optmod, epi, judge, device).train()
    set_distill(name, model, args.conf_distill_ramp, args)
    lr = args.transformer_lr if name == TRANSFORMER else args.field_lr
    opt = make_optimizer(model, lr, args.weight_decay)
    batch = max(1, args.system_tokens_per_step // context)
    x = torch.randint(0, args.vocab_size, (batch, context), device=device)
    y = torch.randint(0, args.vocab_size, (batch, context), device=device)

    def step_once():
        opt.zero_grad(set_to_none=True)
        with amp_ctx(device, args.amp):
            loss, _ = training_loss(name, model, x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

    status, error = "ok", ""
    tps = step_ms = peak = None
    try:
        for _ in range(args.system_warmup):
            step_once()
        sync(device)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(args.system_steps):
            step_once()
        sync(device)
        elapsed = time.perf_counter() - start
        tps = args.system_steps * batch * context / max(elapsed, 1e-9)
        step_ms = elapsed * 1000.0 / args.system_steps
        peak = torch.cuda.max_memory_allocated() / 2**30
    except torch.cuda.OutOfMemoryError as exc:
        status, error = "oom", str(exc)
    except Exception as exc:
        status, error = "error", repr(exc)
    row = {
        "model": name,
        "context": context,
        "batch": batch,
        "status": status,
        "tokens_per_second": tps,
        "bytes_per_second_est": None if tps is None else tps * bytes_per_token,
        "step_ms": step_ms,
        "peak_gib": peak,
        "error": error,
    }
    del model, opt, x, y
    gc.collect()
    torch.cuda.empty_cache()
    return row


def run_selftest(args, shapes, arena, v3, canonical, bridge, optmod, epi, judge, device):
    log("[selftest] token vocabulary, tied heads, finite backward and causality")
    for name in MODEL_NAMES:
        model = build_model(name, shapes[name], args, arena, v3, canonical, bridge, optmod, epi, judge, device).train()
        softpatch = getattr(model, "softpatch", None)
        if softpatch is not None and hasattr(softpatch, "_boundary_lut"):
            lut_n = int(softpatch._boundary_lut.numel())
            lut_sum = float(softpatch._boundary_lut.sum().detach().cpu())
            log(f"[selftest] {name:<32} token_boundary_lut={lut_n} prior_sum={lut_sum:.1f}")
            if lut_n != args.vocab_size:
                raise AssertionError(f"token boundary LUT mismatch {name}: {lut_n} != {args.vocab_size}")
        if not embeddings_tied(model):
            raise AssertionError(f"untied embeddings: {name}")
        x = torch.randint(0, args.vocab_size, (1, 65), device=device)
        y = torch.randint(0, args.vocab_size, (1, 65), device=device)
        set_distill(name, model, args.conf_distill_ramp, args)
        with amp_ctx(device, args.amp):
            loss, primary = training_loss(name, model, x, y)
        loss.backward()
        finite = bool(torch.isfinite(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        uniform_nll = math.log(args.vocab_size)
        emb_std = float(model.emb.weight.detach().float().std().cpu())
        log(
            f"[selftest] {name:<32} params={nparams(model):,} loss={float(loss):.5f} "
            f"primary={float(primary):.5f} uniform_nll={uniform_nll:.5f} "
            f"emb_std={emb_std:.5f} finite={finite}"
        )
        if not finite:
            raise AssertionError(name)
        if not (0.015 <= emb_std <= 0.025):
            raise AssertionError(f"bad tied embedding scale {name}: {emb_std}")
        if float(primary) > uniform_nll + 3.0:
            raise AssertionError(
                f"initial token NLL is implausibly high for {name}: "
                f"{float(primary):.5f} vs log(vocab)={uniform_nll:.5f}"
            )
        model.eval()
        a = torch.randint(0, args.vocab_size, (1, 65), device=device)
        b = a.clone()
        b[:, 48:] = torch.randint(0, args.vocab_size, b[:, 48:].shape, device=device)
        ya = torch.randint(0, args.vocab_size, (1, 65), device=device)
        yb = ya.clone()
        yb[:, 48:] = torch.randint(0, args.vocab_size, yb[:, 48:].shape, device=device)
        with torch.no_grad(), amp_ctx(device, args.amp):
            na = token_nll(name, model, a, ya)
            nb = token_nll(name, model, b, yb)
        err = float((na[:, :46] - nb[:, :46]).abs().max())
        log(f"[selftest] {name:<32} causal_nll_max_abs={err:.3e}")
        if err > args.causal_tol:
            raise AssertionError(f"causality {name}: {err}")
        del model, x, y, a, b, ya, yb, na, nb
        gc.collect()
        torch.cuda.empty_cache()
    log("[selftest] PASS")


def make_summary(args, canonical_path: Path, tokenizer_path: Path, shapes: Dict[str, Shape], corpora: Dict[str, Corpus], results: Dict[str, Dict[str, object]], systems: List[Dict[str, object]]) -> str:
    width = 190
    lines = [
        "=" * width,
        "FIELD TOKEN-PPL PILOT v18 — SHARED BYTE-LEVEL BPE / TIED EMBEDDINGS",
        "=" * width,
        f"canonical_source={canonical_path} sha256={sha256(canonical_path)}",
        f"tokenizer={tokenizer_path} vocab={args.vocab_size:,} type=GPT-2-style byte-level BPE",
        (
            f"protocol: WikiText-103 {args.data_frac:.1%} | train ctx={args.train_seq} tokens | "
            f"steps={args.train_steps:,} | tokens/update={args.batch_size*args.accum*args.train_seq:,} | BF16"
        ),
        "Primary metric: TEST token perplexity at train context. BPB_norm uses the shared corpus bytes/token ratio.",
        "Long-context rows score identical suffix targets while only available history changes.",
        "",
        "TOKENIZED CORPORA",
    ]
    for name in ("train", "validation", "test"):
        c = corpora[name]
        lines.append(f"{name:12s} tokens={c.tokens.numel():12,d} raw_bytes={c.raw_bytes:12,d} bytes/token={c.bytes_per_token:.4f}")

    lines.extend([
        "",
        "MODEL SHAPES",
        f"{'model':34s} {'params':>13s} {'dTarget%':>10s} {'dim':>6s} {'layers':>7s} {'heads':>6s} {'ff':>7s}",
    ])
    for name in MODEL_NAMES:
        s = shapes[name]
        d = 100.0 * (s.params - args.target_params) / args.target_params
        lines.append(f"{name:34s} {s.params:13,d} {d:+10.3f} {s.dim:6d} {s.layers:7d} {s.heads:6d} {s.ff_hidden:7d}")

    lines.extend([
        "",
        "FINAL TEST QUALITY",
        f"{'model':34s} {'PPL@ctx':>10s} {'NLL':>9s} {'bits/tok':>10s} {'BPB_norm':>10s} {'tok/s train':>13s} {'peak GB':>8s}",
    ])
    for name in MODEL_NAMES:
        r = results[name]
        t = r["test"]
        lines.append(
            f"{name:34s} {t['ppl']:10.4f} {t['nll']:9.5f} {t['bits_per_token']:10.5f} "
            f"{t['bpb_norm']:10.5f} {r['train_tokens_per_second']:13,.0f} {r['train_peak_gib']:8.2f}"
        )

    lines.extend(["", "MATCHED-SUFFIX CONTEXT GENERALIZATION — SAME TARGET TOKENS"])
    header = f"{'model':34s}" + "".join(f" {'PPL@'+str(c):>11s}" for c in args.matched_contexts)
    lines.append(header)
    for name in MODEL_NAMES:
        by_ctx = {int(row["context"]): row for row in results[name]["matched_suffix"]}
        lines.append(f"{name:34s}" + "".join(f" {by_ctx[c]['ppl']:11.4f}" for c in args.matched_contexts))

    lines.extend([
        "",
        "EQUAL NO-CHECKPOINT TRAINING SYSTEMS",
        f"{'model':34s} {'ctx':>6s} {'batch':>6s} {'status':>8s} {'tok/s':>12s} {'byte/s est':>13s} {'step ms':>10s} {'peak GB':>8s}",
    ])
    for row in systems:
        tps = "-" if row["tokens_per_second"] is None else f"{row['tokens_per_second']:,.0f}"
        bps = "-" if row["bytes_per_second_est"] is None else f"{row['bytes_per_second_est']:,.0f}"
        ms = "-" if row["step_ms"] is None else f"{row['step_ms']:.2f}"
        peak = "-" if row["peak_gib"] is None else f"{row['peak_gib']:.2f}"
        lines.append(f"{row['model']:34s} {row['context']:6d} {row['batch']:6d} {row['status']:>8s} {tps:>12s} {bps:>13s} {ms:>10s} {peak:>8s}")

    field_ppl = float(results[FIELD_QUALITY]["test"]["ppl"])
    control_ppl = float(results[FIELD_CONTROL]["test"]["ppl"])
    tf_ppl = float(results[TRANSFORMER]["test"]["ppl"])
    quality_gain = control_ppl - field_ppl
    tf_gap = field_ppl - tf_ppl
    lines.extend([
        "",
        "PILOT VERDICT",
        f"quality Field vs selective control: dPPL={field_ppl-control_ppl:+.4f} ({quality_gain/control_ppl*100:+.2f}% relative improvement)",
        f"quality Field vs Transformer: dPPL={tf_gap:+.4f}",
        (
            "VERDICT: TOKEN-LEVEL TRANSFER IS POSITIVE; PROCEED TO A LONGER CANONICAL TOKEN RUN"
            if field_ppl < control_ppl else
            "VERDICT: THE BYTE-LEVEL QUALITY OVERLAY DID NOT TRANSFER CLEANLY; RETUNE TOKEN MEMORY BEFORE SCALING"
        ),
        "=" * width,
    ])
    return "\n".join(lines) + "\n"


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=("selftest", "train", "systems", "summary", "all"), default="all")
    p.add_argument("--outdir", default="/home/ubuntu/pcaf_runs/field_token_ppl_pilot_v18")
    p.add_argument("--canonical-source", default="")
    p.add_argument("--cache-dir", default="/home/ubuntu/field_lab/hf_cache")
    p.add_argument("--data-frac", type=float, default=0.05)
    p.add_argument("--data-device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--tokenizer-min-frequency", type=int, default=2)
    p.add_argument("--target-params", type=int, default=50_000_000)
    p.add_argument("--max-param-delta-pct", type=float, default=0.75)
    p.add_argument("--train-seq", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--train-steps", type=int, default=1200)
    p.add_argument("--field-dim", type=int, default=704)
    p.add_argument("--field-layers", type=int, default=8)
    p.add_argument("--field-heads", type=int, default=8)
    p.add_argument("--field-ff-hidden", type=int, default=1200)
    p.add_argument("--hybrid-ff-hidden", type=int, default=1200)
    p.add_argument("--af-ff-hidden", type=int, default=1200)
    p.add_argument("--field-chunk", type=int, default=32)
    p.add_argument("--triton-block-c", type=int, default=16)
    p.add_argument("--triton-chunk-t", type=int, default=64)
    p.add_argument("--num-buckets", type=int, default=16384)
    p.add_argument("--local-chunk", type=int, default=1024)
    p.add_argument("--salience-floor", type=float, default=0.10)
    p.add_argument("--residual-limit", type=float, default=4.0)
    p.add_argument("--span-tokens", type=int, default=2)
    p.add_argument("--address-dim", type=int, default=24)
    p.add_argument("--latent-top-k", type=int, default=4)
    p.add_argument("--score-limit", type=float, default=2.0)
    p.add_argument("--span-top-k", type=int, default=4)
    p.add_argument("--sidecar-max-mix", type=float, default=0.40)
    p.add_argument("--sidecar-gate-bias", type=float, default=1e-6)
    p.add_argument("--sidecar-aux-weight", type=float, default=0.01)
    p.add_argument("--gate-grad-scale", type=float, default=0.01)
    p.add_argument("--delta-heads", type=int, default=4)
    p.add_argument("--delta-block", type=int, default=16)
    p.add_argument("--phase-bands", type=int, default=8)
    p.add_argument("--phase-rank", type=int, default=16)
    p.add_argument("--tf-dim", type=int, default=704)
    p.add_argument("--tf-heads", type=int, default=11)
    p.add_argument("--tf-layers", type=int, default=8)
    p.add_argument("--tf-ff-hidden", type=int, default=1408)
    p.add_argument("--amp", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--field-lr", type=float, default=4e-4)
    p.add_argument("--transformer-lr", type=float, default=5e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--min-lr-ratio", type=float, default=0.10)
    p.add_argument("--weight-decay", type=float, default=0.10)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--conf-distill-ramp", type=int, default=150)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=300)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--quick-eval-windows", type=int, default=8)
    p.add_argument("--test-token-budget", type=int, default=262144)
    p.add_argument("--matched-contexts", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    p.add_argument("--matched-score-tokens", type=int, default=128)
    p.add_argument("--matched-windows", type=int, default=8)
    p.add_argument("--system-contexts", type=int, nargs="+", default=[1024, 2048, 4096])
    p.add_argument("--system-tokens-per-step", type=int, default=4096)
    p.add_argument("--system-warmup", type=int, default=2)
    p.add_argument("--system-steps", type=int, default=5)
    p.add_argument("--model-seed", type=int, default=1234)
    p.add_argument("--data-seed", type=int, default=5678)
    p.add_argument("--eval-seed", type=int, default=9012)
    p.add_argument("--causal-tol", type=float, default=0.005)
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

    arena = import_module(V15_PATH, "field_scale_50m_v15_token_bridge")
    canonical_path = locate_canonical(args.canonical_source)
    actual = sha256(canonical_path)
    if actual != EXPECTED_CANONICAL_SHA256:
        raise RuntimeError(f"canonical SHA mismatch expected={EXPECTED_CANONICAL_SHA256} actual={actual}")

    base = arena.base
    v3 = base.import_module(base.V3_PATH, "v18_v3")
    bridge = base.import_module(base.BRIDGE_PATH, "v18_bridge")
    optmod = base.import_module(base.OPT_PATH, "v18_opt")
    epi = base.import_module(base.V9_PATH, "v18_epi")
    judge = base.import_module(base.JUDGE_PATH, "v18_judge")
    canonical = base.import_module(canonical_path, "v18_canonical")
    optmod.v3_global = v3
    base.install_fast_candidate_route(epi, optmod)

    changed = patch_vocab(args.vocab_size, HERE, canonical_path)
    log(f"[vocab] patched VOCAB={args.vocab_size:,} in {len(changed)} modules")
    if len(changed) < 8:
        log(f"[vocab] patched modules={changed}")

    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)

    rows = load_raw_rows(args.cache_dir, args.data_frac)
    tokenizer = build_or_load_tokenizer(root, rows[0], args.vocab_size, args.tokenizer_min_frequency)
    train_c, val_c, test_c = save_or_load_corpora(root, tokenizer, rows)
    corpora = {"train": train_c, "validation": val_c, "test": test_c}

    train = place_tokens(train_c.tokens, device, args.data_device, "train")
    val = place_tokens(val_c.tokens, device, args.data_device, "validation")
    test = place_tokens(test_c.tokens, device, args.data_device, "test")

    shapes = resolve_shapes(args, arena, v3, canonical, bridge, optmod, epi, judge)
    atomic_json(root / "config.json", {
        "args": vars(args),
        "canonical_source": str(canonical_path),
        "canonical_sha256": actual,
        "tokenizer": str(root / "tokenizer" / "tokenizer.json"),
        "shapes": {name: asdict(shape) for name, shape in shapes.items()},
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    })

    log("=" * 160)
    log("FIELD TOKEN-PPL PILOT v18 — SHARED BYTE-LEVEL BPE / TIED EMBEDDINGS")
    log(f"gpu={torch.cuda.get_device_name(0)} torch={torch.__version__} cuda={torch.version.cuda}")
    log(f"canonical={canonical_path} sha={actual}")
    for name in MODEL_NAMES:
        s = shapes[name]
        d = 100.0 * (s.params - args.target_params) / args.target_params
        log(f"{name:34s} params={s.params:,} dTarget={d:+.3f}% dim={s.dim} layers={s.layers} heads={s.heads} ff={s.ff_hidden}")
    log("=" * 160)

    if args.mode in ("selftest", "all"):
        run_selftest(args, shapes, arena, v3, canonical, bridge, optmod, epi, judge, device)
        if args.mode == "selftest":
            return

    results_path = root / "all_results.json"
    if args.mode in ("train", "all"):
        results: Dict[str, Dict[str, object]] = {}
        for name in MODEL_NAMES:
            results[name] = train_one(
                name, shapes[name], args, arena, v3, canonical, bridge, optmod, epi, judge,
                train, val, test_c, test, root, device,
            )
            atomic_json(results_path, results)
        if args.mode == "train":
            return
    else:
        if not results_path.is_file():
            raise FileNotFoundError(results_path)
        results = json.loads(results_path.read_text(encoding="utf-8"))

    systems_path = root / "systems.json"
    if args.mode in ("systems", "all"):
        systems: List[Dict[str, object]] = []
        for context in args.system_contexts:
            for name in MODEL_NAMES:
                row = systems_benchmark(
                    name, shapes[name], args, arena, v3, canonical, bridge, optmod, epi, judge,
                    context, train_c.bytes_per_token, device,
                )
                systems.append(row)
                log(
                    f"[systems] {name:34s} ctx={context:5d} batch={row['batch']:2d} "
                    f"status={row['status']:>5s} tok/s={row.get('tokens_per_second') or 0:,.0f} "
                    f"peak={row.get('peak_gib') or 0:.2f}G"
                )
        atomic_json(systems_path, systems)
        if args.mode == "systems":
            return
    else:
        if not systems_path.is_file():
            raise FileNotFoundError(systems_path)
        systems = json.loads(systems_path.read_text(encoding="utf-8"))

    summary = make_summary(
        args, canonical_path, root / "tokenizer" / "tokenizer.json",
        shapes, corpora, results, systems,
    )
    atomic_text(root / "summary.txt", summary)
    log(summary)


if __name__ == "__main__":
    main()
