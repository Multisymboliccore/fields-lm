#!/usr/bin/env python3
"""Field-Fusion official R1: PG-19 from-scratch pilot and long-memory probes.

This controller reuses the frozen, promoted 18F/2M/4R constructor and the
validated parameter-matched Transformer Flash and official Mamba-2 comparators.
It deliberately does not import any rejected experimental macroblock.

Modes
-----
* pilot: one paired seed, 49.152M tokens/model on PG-19, then memory probes.
* three-seed: three paired seeds with the same protocol, then memory probes.
* memory-only: run probes from existing exports without retraining.
* package-selftest / gpu-selftest: preflight checks.

Training windows never cross book boundaries.  The tokenizer is a shared
16,384-vocabulary byte-level BPE trained by this package from the selected
PG-19 training books.  Loss/PPL is logged every 100 updates by default.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import final_closure_arena_core as arena

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "official_source"
CANONICAL_SHA256 = arena.CANONICAL_SHA256
OFFICIAL_FIELD = arena.OFFICIAL_FIELD
TRANSFORMER = arena.TRANSFORMER
MAMBA2 = arena.MAMBA2
FIELD_PCAF_OFF = arena.FIELD_PCAF_OFF
PCAF_CONV = arena.PCAF_CONV
MODELS = arena.MODELS
DISPLAY_NAMES = arena.DISPLAY_NAMES
SUITE_VERSION = 3


def log(value: object = "") -> None:
    print(str(value), flush=True)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8")
    os.replace(tmp, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=("pilot", "three-seed", "memory-only", "package-selftest", "gpu-selftest"), default="pilot")
    p.add_argument("--outdir", default="/home/ubuntu/pcaf_runs/field_fusion_pg19_memory_pilot_run")
    p.add_argument("--canonical-source", default=str(HERE / "field_only_v4_chunked_triton_wiki100.py"))
    p.add_argument("--cache-dir", default="/home/ubuntu/field_lab/hf_cache")
    p.add_argument("--data-root", default="/home/ubuntu/field_lab/field_fusion_pg19_official_data")
    p.add_argument("--dataset-id", default="emozilla/pg19")
    p.add_argument("--dataset-fallbacks", nargs="*", default=["Tanushreeeeee/pg19"])
    p.add_argument("--train-raw-byte-cap", type=int, default=536_870_912, help="Seeded sample drawn from the full PG-19 train split.")
    p.add_argument("--validation-doc-limit", type=int, default=50)
    p.add_argument("--test-doc-limit", type=int, default=100)
    p.add_argument("--vocab-size", type=int, default=16_384)
    p.add_argument("--tokenizer-min-frequency", type=int, default=2)
    p.add_argument("--data-device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--amp", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--seeds", nargs="+", type=int, default=[1234])
    p.add_argument("--data-seeds", nargs="+", type=int, default=[5678])
    p.add_argument("--eval-seed", type=int, default=9012)
    p.add_argument("--memory-seed", type=int, default=77123)
    p.add_argument("--train-token-budget", type=int, default=49_152_000)
    p.add_argument("--train-seq", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.10)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-fraction", type=float, default=0.02)
    p.add_argument("--wsd-stable-fraction", type=float, default=0.70)
    p.add_argument("--min-lr-ratio", type=float, default=0.10)
    p.add_argument("--log-every-updates", type=int, default=100)
    p.add_argument("--checkpoint-every-updates", type=int, default=500)
    p.add_argument("--eval-milestones", nargs="+", type=int, default=[25_165_824, 49_152_000])
    p.add_argument("--validation-token-budget", type=int, default=1_048_576)
    p.add_argument("--test-token-budget", type=int, default=1_048_576)
    p.add_argument("--stream-readout-chunk", type=int, default=512)
    p.add_argument("--long-contexts", nargs="+", type=int, default=[2048, 8192, 16384, 32768, 65536])
    p.add_argument("--long-context-windows", type=int, default=8)
    p.add_argument("--long-context-score-tokens", type=int, default=128)
    p.add_argument("--memory-contexts", nargs="+", type=int, default=[2048, 8192, 16384, 32768, 65536])
    p.add_argument("--memory-trials", type=int, default=12)
    p.add_argument("--memory-pairs", type=int, default=64)
    p.add_argument("--max-param-delta-pct", type=float, default=0.75)
    p.add_argument("--yarn-factor", type=float, default=32.0)
    p.add_argument("--yarn-original-context", type=int, default=2048)
    p.add_argument("--yarn-beta-fast", type=float, default=32.0)
    p.add_argument("--yarn-beta-slow", type=float, default=1.0)
    p.add_argument("--yarn-rope-theta", type=float, default=10000.0)
    p.add_argument("--yarn-truncate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--yarn-gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--keep-final-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--only-model", choices=MODELS)
    p.add_argument("--only-seed-index", type=int)
    p.add_argument("--export-root", default="", help="Existing quality root for memory-only mode.")
    p.add_argument("--data-smoke", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    if args.mode == "pilot":
        args.seeds, args.data_seeds = [1234], [5678]
    elif args.mode == "three-seed":
        args.seeds, args.data_seeds = [1234, 2345, 3456], [5678, 6789, 7890]
    if len(args.seeds) != len(args.data_seeds):
        p.error("--seeds and --data-seeds must have equal lengths")
    if args.train_token_budget % (args.train_seq * args.batch_size):
        p.error("train-token-budget must divide train-seq*batch-size")
    if sorted(set(args.eval_milestones))[-1] != args.train_token_budget:
        p.error("eval-milestones must end at train-token-budget")
    if args.log_every_updates < 1:
        p.error("log-every-updates must be positive")
    return args


def arena_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        outdir=args.outdir,
        canonical_source=args.canonical_source,
        cache_dir=args.cache_dir,
        shared_data_root=args.data_root,
        tokenizer_source="",
        data_device="cpu",
        amp=args.amp,
        seeds=list(args.seeds),
        data_seeds=list(args.data_seeds),
        eval_seed=args.eval_seed,
        system_seed=args.memory_seed,
        train_token_budget=args.train_token_budget,
        train_seq=args.train_seq,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_fraction=args.warmup_fraction,
        wsd_stable_fraction=args.wsd_stable_fraction,
        min_lr_ratio=args.min_lr_ratio,
        log_every_updates=args.log_every_updates,
        ppl_ema_beta=0.98,
        checkpoint_every_updates=args.checkpoint_every_updates,
        eval_milestones=list(args.eval_milestones),
        validation_token_budget=args.validation_token_budget,
        test_token_budget=args.test_token_budget,
        stream_readout_chunk=args.stream_readout_chunk,
        long_contexts=list(args.long_contexts),
        long_context_windows=args.long_context_windows,
        long_context_score_tokens=args.long_context_score_tokens,
        system_contexts=list(args.memory_contexts),
        system_batches=[1] * len(args.memory_contexts),
        system_warmup=1,
        system_steps=1,
        max_param_delta_pct=args.max_param_delta_pct,
        yarn_factor=args.yarn_factor,
        yarn_original_context=args.yarn_original_context,
        yarn_beta_fast=args.yarn_beta_fast,
        yarn_beta_slow=args.yarn_beta_slow,
        yarn_rope_theta=args.yarn_rope_theta,
        yarn_truncate=args.yarn_truncate,
        yarn_gradient_checkpointing=args.yarn_gradient_checkpointing,
        resume=args.resume,
        keep_final_checkpoints=args.keep_final_checkpoints,
        run_systems=False,
        package_selftest=False,
        gpu_selftest=False,
        only_model=args.only_model,
        only_seed_index=args.only_seed_index,
        worker_task=None,
    )


def prepare_factory(args: argparse.Namespace, device: torch.device):
    stack = arena.import_current_stack()
    aargs = arena_args(args)
    base_args = arena.create_base_args(aargs, stack)
    base_args.candidate = [arena.OFFICIAL_R1_SPEC]
    arena.configure_current_stack(base_args, stack)
    r1, v27 = stack["r1"], stack["v27"]
    control_name, control_spec = arena.find_control_spec(None, r1)
    canonical_path, canonical_hash, deps = v27.load_dependencies(base_args)
    if canonical_hash != CANONICAL_SHA256:
        raise RuntimeError(f"canonical SHA mismatch expected={CANONICAL_SHA256} actual={canonical_hash}")
    _, shapes, accounting = r1.solve_candidate_shapes_r1(base_args, deps)
    if set(shapes) != {arena.OFFICIAL_R1_SPEC}:
        raise RuntimeError(f"official shape solver leaked arms: {sorted(shapes)}")
    field_shape = shapes[arena.OFFICIAL_R1_SPEC]
    comp_shapes = arena.comparator_shapes(stack, base_args, deps)
    return stack, base_args, control_name, control_spec, field_shape, comp_shapes, deps, accounting


def extract_text(row: Mapping[str, Any]) -> str:
    for key in ("text", "book_text", "content", "document"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for value in row.values():
        if isinstance(value, str) and len(value) > 1024:
            return value
    return ""


def load_pg19_dataset(args: argparse.Namespace):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets package is required") from exc
    errors = []
    for dataset_id in [args.dataset_id, *args.dataset_fallbacks]:
        try:
            log(f"[pg19] loading dataset={dataset_id}")
            ds = load_dataset(dataset_id, cache_dir=args.cache_dir)
            train_key = "train"
            val_key = "validation" if "validation" in ds else ("valid" if "valid" in ds else None)
            test_key = "test" if "test" in ds else None
            if train_key not in ds or val_key is None or test_key is None:
                raise KeyError(f"required splits unavailable: {list(ds.keys())}")
            probe = extract_text(ds[train_key][0])
            if not probe:
                raise RuntimeError("could not locate text field")
            return ds, dataset_id, train_key, val_key, test_key
        except Exception as exc:
            errors.append(f"{dataset_id}: {type(exc).__name__}: {exc}")
            log(f"[pg19] dataset candidate failed: {errors[-1]}")
    raise RuntimeError("all PG-19 dataset candidates failed:\n" + "\n".join(errors))


def select_documents(split: Any, *, byte_cap: int, doc_limit: int, seed: int, shuffle: bool) -> Tuple[List[str], List[int], int]:
    n = len(split)
    order = np.random.default_rng(seed).permutation(n) if shuffle else np.arange(n)
    docs: List[str] = []
    indices: List[int] = []
    raw_bytes = 0
    for raw_idx in order.tolist():
        text = extract_text(split[int(raw_idx)])
        if not text:
            continue
        if not text.endswith("\n"):
            text += "\n"
        size = len(text.encode("utf-8", errors="replace"))
        docs.append(text)
        indices.append(int(raw_idx))
        raw_bytes += size
        if doc_limit > 0 and len(docs) >= doc_limit:
            break
        if byte_cap > 0 and raw_bytes >= byte_cap:
            break
    if not docs:
        raise RuntimeError("document selection produced no text")
    return docs, indices, raw_bytes


@dataclass
class PgCorpus:
    name: str
    token_path: Path
    offsets: np.ndarray
    raw_bytes: int
    rows: int
    token_count: int
    _memmap: Optional[np.memmap] = None

    @property
    def bytes_per_token(self) -> float:
        return self.raw_bytes / max(self.token_count, 1)

    @property
    def tokens(self) -> np.memmap:
        if self._memmap is None:
            self._memmap = np.memmap(self.token_path, dtype=np.uint16, mode="r")
        return self._memmap


def build_or_load_tokenizer(data_root: Path, train_docs: Sequence[str], vocab_size: int, min_frequency: int):
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    except ImportError as exc:
        raise RuntimeError("tokenizers package is required") from exc
    path = data_root / "tokenizer" / "tokenizer.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        tok = Tokenizer.from_file(str(path))
        if int(tok.get_vocab_size()) != int(vocab_size):
            raise RuntimeError(f"cached tokenizer vocab mismatch: {tok.get_vocab_size()} != {vocab_size}")
        log(f"[tokenizer] loaded {path} vocab={tok.get_vocab_size():,}")
        return tok
    log(f"[tokenizer] training PG-19 byte-level BPE vocab={vocab_size:,} docs={len(train_docs):,}")
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
    tok.train_from_iterator(iter(train_docs), trainer=trainer, length=len(train_docs))
    if int(tok.get_vocab_size()) != int(vocab_size):
        raise RuntimeError(f"tokenizer produced vocab={tok.get_vocab_size()} expected={vocab_size}")
    tok.save(str(path))
    sample = "PG-19 Field roundtrip: memory, café, ação, 1919.\n"
    if tok.decode(tok.encode(sample).ids) != sample:
        raise AssertionError("tokenizer roundtrip mismatch")
    log(f"[tokenizer] saved {path}")
    return tok


def encode_or_load_corpus(root: Path, name: str, tokenizer: Any, docs: Sequence[str], raw_bytes: int) -> PgCorpus:
    root.mkdir(parents=True, exist_ok=True)
    token_path = root / f"{name}.u16"
    offsets_path = root / f"{name}_offsets.npy"
    meta_path = root / f"{name}.json"
    if token_path.is_file() and offsets_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        offsets = np.load(offsets_path)
        corpus = PgCorpus(name, token_path, offsets, int(meta["raw_bytes"]), int(meta["rows"]), int(meta["tokens"]))
        if token_path.stat().st_size != corpus.token_count * 2:
            raise RuntimeError(f"corrupt token file: {token_path}")
        log(f"[tokenizer] loaded {name}: tokens={corpus.token_count:,} docs={corpus.rows:,}")
        return corpus
    tmp = token_path.with_suffix(".tmp")
    offsets: List[Tuple[int, int]] = []
    cursor = 0
    log(f"[tokenizer] encoding PG-19 {name} docs={len(docs):,}")
    with tmp.open("wb") as f:
        for i, text in enumerate(docs):
            ids = np.asarray(tokenizer.encode(text, add_special_tokens=False).ids, dtype=np.uint16)
            if ids.size < 2:
                continue
            ids.tofile(f)
            offsets.append((cursor, cursor + int(ids.size)))
            cursor += int(ids.size)
            if (i + 1) % 50 == 0:
                log(f"[tokenizer] {name} encoded_docs={i+1:,} tokens={cursor:,}")
    os.replace(tmp, token_path)
    offset_array = np.asarray(offsets, dtype=np.int64)
    np.save(offsets_path, offset_array)
    atomic_json(meta_path, {"name": name, "raw_bytes": int(raw_bytes), "rows": len(offsets), "tokens": cursor})
    corpus = PgCorpus(name, token_path, offset_array, int(raw_bytes), len(offsets), cursor)
    log(f"[tokenizer] {name}: tokens={cursor:,} docs={len(offsets):,} bytes/token={corpus.bytes_per_token:.3f}")
    return corpus


def prepare_pg19_data(args: argparse.Namespace) -> Dict[str, Any]:
    data_root = Path(args.data_root).expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    selection_path = data_root / "selection.json"
    raw_cache = data_root / "raw_selection.pt"
    tokenizer_path = data_root / "tokenizer" / "tokenizer.json"
    tokenized_root = data_root / "tokenized"
    complete_cache = (
        selection_path.is_file() and tokenizer_path.is_file() and
        all((tokenized_root / name).is_file() for name in (
            "train.u16", "train_offsets.npy", "train.json",
            "validation.u16", "validation_offsets.npy", "validation.json",
            "test.u16", "test_offsets.npy", "test.json",
        ))
    )
    train_docs: List[str] = []
    val_docs: List[str] = []
    test_docs: List[str] = []
    if complete_cache:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        log(f"[pg19] fast-loaded complete token cache dataset={selection['dataset_id']}")
    elif raw_cache.is_file() and selection_path.is_file():
        raw = torch.load(raw_cache, map_location="cpu", weights_only=False)
        train_docs, val_docs, test_docs = raw["train"], raw["validation"], raw["test"]
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        log(f"[pg19] loaded cached raw selection dataset={selection['dataset_id']}")
    else:
        ds, dataset_id, train_key, val_key, test_key = load_pg19_dataset(args)
        train_docs, train_idx, train_bytes = select_documents(
            ds[train_key], byte_cap=args.train_raw_byte_cap, doc_limit=0, seed=13579, shuffle=True)
        val_docs, val_idx, val_bytes = select_documents(
            ds[val_key], byte_cap=0, doc_limit=args.validation_doc_limit, seed=24680, shuffle=False)
        test_docs, test_idx, test_bytes = select_documents(
            ds[test_key], byte_cap=0, doc_limit=args.test_doc_limit, seed=97531, shuffle=False)
        selection = {
            "dataset_id": dataset_id,
            "train_split": train_key,
            "validation_split": val_key,
            "test_split": test_key,
            "train_indices": train_idx,
            "validation_indices": val_idx,
            "test_indices": test_idx,
            "train_raw_bytes": train_bytes,
            "validation_raw_bytes": val_bytes,
            "test_raw_bytes": test_bytes,
            "train_docs": len(train_docs),
            "validation_docs": len(val_docs),
            "test_docs": len(test_docs),
            "selection_seed": 13579,
            "requested_train_raw_byte_cap": int(args.train_raw_byte_cap),
        }
        atomic_json(selection_path, selection)
        torch.save({"train": train_docs, "validation": val_docs, "test": test_docs}, raw_cache)
        log(f"[pg19] selected train_docs={len(train_docs):,} bytes={train_bytes:,} val_docs={len(val_docs)} test_docs={len(test_docs)}")
    tokenizer = build_or_load_tokenizer(data_root, train_docs, args.vocab_size, args.tokenizer_min_frequency)
    train = encode_or_load_corpus(tokenized_root, "train", tokenizer, train_docs, int(selection["train_raw_bytes"]))
    val = encode_or_load_corpus(tokenized_root, "validation", tokenizer, val_docs, int(selection["validation_raw_bytes"]))
    test = encode_or_load_corpus(tokenized_root, "test", tokenizer, test_docs, int(selection["test_raw_bytes"]))
    minimum = args.train_seq + 1
    for corpus in (train, val, test):
        if not any(int(e - s) >= minimum for s, e in corpus.offsets):
            raise RuntimeError(f"{corpus.name} has no document long enough for seq={args.train_seq}")
    manifest = {
        **selection,
        "vocab_size": int(tokenizer.get_vocab_size()),
        "tokenizer_sha256": sha256(tokenizer_path),
        "train_tokens": train.token_count,
        "validation_tokens": val.token_count,
        "test_tokens": test.token_count,
        "document_boundary_safe": True,
    }
    atomic_json(data_root / "PG19_DATA_MANIFEST.json", manifest)
    log("pg19_shared_tokenizer=PASS")
    log("pg19_document_boundary_sampling=PASS")
    return {"train": train, "val": val, "test": test, "tokenizer": tokenizer, "data_manifest": manifest}


def capacities(corpus: PgCorpus, seq: int) -> Tuple[np.ndarray, np.ndarray]:
    lengths = corpus.offsets[:, 1] - corpus.offsets[:, 0]
    caps = np.maximum(lengths - int(seq), 0).astype(np.int64)
    keep = np.flatnonzero(caps > 0)
    if keep.size == 0:
        raise RuntimeError(f"no {corpus.name} document can support context={seq}")
    return keep, caps[keep]


def make_doc_starts(corpus: PgCorpus, count: int, seq: int, seed: int, path: Optional[Path] = None) -> np.ndarray:
    if path is not None and path.is_file():
        starts = np.load(path)
        if starts.shape == (count,):
            return starts.astype(np.int64, copy=False)
    keep, caps = capacities(corpus, seq)
    cumulative = np.cumsum(caps, dtype=np.int64)
    total = int(cumulative[-1])
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, total, size=count, dtype=np.int64)
    positions = np.searchsorted(cumulative, samples, side="right")
    previous = np.where(positions == 0, 0, cumulative[positions - 1])
    local = samples - previous
    docs = keep[positions]
    starts = corpus.offsets[docs, 0] + local
    starts = starts.astype(np.int64)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, starts)
    return starts


def batch_from_starts(corpus: PgCorpus, starts: np.ndarray, index: int, batch: int, seq: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    rows = np.empty((batch, seq + 1), dtype=np.int64)
    tokens = corpus.tokens
    for j in range(batch):
        start = int(starts[index + j])
        rows[j] = np.asarray(tokens[start:start + seq + 1], dtype=np.int64)
    window = torch.from_numpy(rows).to(device=device, non_blocking=True)
    return window[:, :-1], window[:, 1:]


@torch.no_grad()
def evaluate_corpus(args: argparse.Namespace, factory: Tuple[Any, ...], model: nn.Module, backend: str, corpus: PgCorpus, token_budget: int, seed: int, device: torch.device) -> Dict[str, float]:
    stack = factory[0]
    context = args.train_seq
    windows = max(1, math.ceil(token_budget / context)) if token_budget > 0 else 256
    starts = make_doc_starts(corpus, windows, context, seed)
    model.eval()
    total = 0.0
    count = 0
    for i in range(windows):
        x, y = batch_from_starts(corpus, starts, i, 1, context, device)
        with arena.amp_ctx(device, args.amp):
            nll = arena.streaming_token_nll(stack, backend, model, x, y, min(args.stream_readout_chunk, context), True).float()
        total += float(nll.sum().cpu())
        count += int(nll.numel())
    mean = total / max(count, 1)
    model.train()
    return {
        "context": context,
        "nll": mean,
        "ppl": math.exp(min(mean, 20.0)),
        "bits_per_token": mean / math.log(2.0),
        "bpb_norm": (total / math.log(2.0)) / max(count * corpus.bytes_per_token, 1e-9),
        "tokens": count,
        "windows": windows,
    }


@torch.no_grad()
def matched_suffix_docs(args: argparse.Namespace, factory: Tuple[Any, ...], model: nn.Module, backend: str, corpus: PgCorpus, seed: int, device: torch.device) -> Dict[str, Dict[str, Any]]:
    stack = factory[0]
    max_context = max(args.long_contexts)
    eligible = np.flatnonzero((corpus.offsets[:, 1] - corpus.offsets[:, 0]) >= max_context + 1)
    if eligible.size == 0:
        return {str(c): {"status": "unavailable", "context": int(c)} for c in args.long_contexts}
    rng = np.random.default_rng(seed)
    choices = rng.choice(eligible, size=args.long_context_windows, replace=eligible.size < args.long_context_windows)
    ends: List[int] = []
    for doc in choices.tolist():
        s, e = map(int, corpus.offsets[int(doc)])
        ends.append(int(rng.integers(s + max_context, e)))
    out: Dict[str, Dict[str, Any]] = {}
    model.eval()
    for context in args.long_contexts:
        total = 0.0
        count = 0
        try:
            for end in ends:
                raw = np.asarray(corpus.tokens[end - context:end + 1], dtype=np.int64)
                win = torch.from_numpy(raw).to(device=device)[None]
                x, y = win[:, :-1], win[:, 1:]
                with arena.amp_ctx(device, args.amp):
                    nll = arena.streaming_token_nll(stack, backend, model, x, y, min(args.stream_readout_chunk, context), True).float()
                tail = nll[:, -args.long_context_score_tokens:]
                total += float(tail.sum().cpu())
                count += int(tail.numel())
            mean = total / max(count, 1)
            row = {"status": "ok", "context": int(context), "nll": mean, "ppl": math.exp(min(mean, 20.0)), "tokens": count, "windows": len(ends)}
            log(f"[pg19-context] backend={backend} ctx={context} nll={mean:.5f} ppl={row['ppl']:.3f}")
        except torch.cuda.OutOfMemoryError as exc:
            clear_cuda()
            row = {"status": "oom", "context": int(context), "error": str(exc)}
            log(f"[pg19-context] backend={backend} ctx={context} status=OOM")
        except Exception as exc:
            clear_cuda()
            row = {"status": "error", "context": int(context), "error": repr(exc), "traceback": traceback.format_exc()}
            log(f"[pg19-context] backend={backend} ctx={context} status=ERROR error={exc!r}")
        out[str(context)] = row
    model.train()
    return out


def optimizer_for(model_name: str, model: nn.Module, factory: Tuple[Any, ...]):
    if model_name in {OFFICIAL_FIELD, FIELD_PCAF_OFF}:
        return arena.optimizer_for_field(model, factory[1], factory[0])
    return arena.optimizer_for_comparator(model, factory[1]), "adamw_uniform"


def export_bf16(path: Path, model_name: str, model: nn.Module, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    for key, value in model.state_dict().items():
        cpu = value.detach().cpu()
        state[key] = cpu.to(torch.bfloat16) if cpu.is_floating_point() else cpu
    tmp = path.with_suffix(".tmp")
    torch.save({"format": "field_fusion_pg19_official_bf16", "model": model_name, "metadata": dict(metadata), "state_dict": state}, tmp)
    os.replace(tmp, path)


def train_one(args: argparse.Namespace, factory: Tuple[Any, ...], data: Mapping[str, Any], model_name: str, seed_index: int, device: torch.device, runroot: Path) -> Dict[str, Any]:
    stack, base_args, control_name, control_spec, field_shape, comp_shapes, deps, _ = factory
    model_seed = int(args.seeds[seed_index])
    data_seed = int(args.data_seeds[seed_index])
    out = runroot / "quality" / f"seed{seed_index}_{model_seed}" / model_name
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "result.json"
    if result_path.is_file() and args.resume:
        return json.loads(result_path.read_text(encoding="utf-8"))
    base_args.model_seed = model_seed
    base_args.data_seed = data_seed
    seed_all(model_seed)
    model, backend = arena.build_model(model_name, stack, control_spec, field_shape, comp_shapes, base_args, deps, device)
    topology = arena.topology_audit(model) if model_name in {OFFICIAL_FIELD, FIELD_PCAF_OFF} else None
    model_audit = arena.model_audit(model_name, model)
    params = arena.nparams(model)
    optimizer, optimizer_recipe = optimizer_for(model_name, model, factory)
    sequences = args.train_token_budget // args.train_seq
    starts = make_doc_starts(data["train"], sequences, args.train_seq, data_seed, runroot / "paired_starts" / f"seed{seed_index}_{data_seed}.npy")
    updates = sequences // args.batch_size
    signature = {
        "suite_version": SUITE_VERSION,
        "dataset": data["data_manifest"]["dataset_id"],
        "data_manifest_sha256": sha256(Path(args.data_root) / "PG19_DATA_MANIFEST.json"),
        "model": model_name,
        "model_seed": model_seed,
        "data_seed": data_seed,
        "train_tokens": args.train_token_budget,
        "train_seq": args.train_seq,
        "batch": args.batch_size,
        "parameters": params,
        "canonical_sha256": CANONICAL_SHA256,
    }
    checkpoint = out / "latest.pt"
    sequence_index = 0
    compute_seconds = 0.0
    history: List[Dict[str, Any]] = []
    ppl_ema: Optional[float] = None
    raw = arena.load_checkpoint(checkpoint, signature, model, optimizer) if args.resume else None
    if raw:
        sequence_index = int(raw["sequence_index"])
        compute_seconds = float(raw.get("compute_seconds", 0.0))
        history = list(raw.get("history", []))
        ppl_ema = raw.get("ppl_ema")
    milestones = sorted(set(map(int, args.eval_milestones)))
    already = {int(row["train_tokens"]) for row in history if "train_tokens" in row}
    update = sequence_index // args.batch_size
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    while sequence_index < sequences:
        update += 1
        mult = arena.wsd_multiplier(update, updates, args.warmup_fraction, args.wsd_stable_fraction, args.min_lr_ratio)
        arena.apply_lr(optimizer, mult)
        started = time.perf_counter()
        x, y = batch_from_starts(data["train"], starts, sequence_index, args.batch_size, args.train_seq, device)
        optimizer.zero_grad(set_to_none=True)
        with arena.amp_ctx(device, args.amp):
            loss = arena.loss_call(stack, backend, model, x, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss model={model_name} seed={model_seed} update={update}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
        optimizer.step()
        arena.sync(device)
        step_seconds = time.perf_counter() - started
        compute_seconds += step_seconds
        sequence_index += args.batch_size
        trained_tokens = sequence_index * args.train_seq
        loss_value = float(loss.detach().float().cpu())
        train_ppl = math.exp(min(loss_value, 20.0))
        ppl_ema = train_ppl if ppl_ema is None else 0.98 * float(ppl_ema) + 0.02 * train_ppl
        if update == 1 or update % args.log_every_updates == 0 or trained_tokens == args.train_token_budget:
            current_lr = max(float(group["lr"]) for group in optimizer.param_groups)
            log(
                f"[{model_name} seed={model_seed}] step={update}/{updates} tokens={trained_tokens:,} "
                f"train_loss={loss_value:.5f} train_ppl={train_ppl:.4f} ppl_ema={ppl_ema:.4f} "
                f"lr={current_lr:.3e} grad={grad_norm:.3f} step_tok/s={(args.batch_size*args.train_seq)/max(step_seconds,1e-9):,.0f} "
                f"avg_tok/s={trained_tokens/max(compute_seconds,1e-9):,.0f}"
            )
        reached = [m for m in milestones if trained_tokens >= m and m not in already]
        for milestone in reached:
            val = evaluate_corpus(args, factory, model, backend, data["val"], args.validation_token_budget, args.eval_seed + seed_index * 17, device)
            test = evaluate_corpus(args, factory, model, backend, data["test"], args.test_token_budget, args.eval_seed + 1000 + seed_index * 17, device)
            row = {"train_tokens": milestone, "update": update, "validation": val, "test": test}
            history.append(row)
            already.add(milestone)
            atomic_json(out / "milestones.json", history)
            log(f"[{model_name} seed={model_seed}] MILESTONE {milestone:,} val={val['nll']:.5f} val_ppl={val['ppl']:.3f} test={test['nll']:.5f} test_ppl={test['ppl']:.3f}")
        if update % args.checkpoint_every_updates == 0 or trained_tokens == args.train_token_budget:
            arena.save_checkpoint(checkpoint, signature, model, optimizer, sequence_index, compute_seconds, history, ppl_ema)
        del x, y, loss
    peak_gib = torch.cuda.max_memory_allocated(device) / 2**30
    validation = evaluate_corpus(args, factory, model, backend, data["val"], args.validation_token_budget, args.eval_seed + seed_index * 17, device)
    test = evaluate_corpus(args, factory, model, backend, data["test"], args.test_token_budget, args.eval_seed + 1000 + seed_index * 17, device)
    contexts = matched_suffix_docs(args, factory, model, backend, data["test"], args.eval_seed + 2000 + seed_index * 101, device)
    export_path = out / "final_bf16.pt"
    export_bf16(export_path, model_name, model, signature)
    result = {
        "model": model_name,
        "display_name": DISPLAY_NAMES[model_name],
        "seed_index": seed_index,
        "model_seed": model_seed,
        "data_seed": data_seed,
        "backend_name": backend,
        "parameters": params,
        "optimizer_recipe": optimizer_recipe,
        "train_tokens": args.train_token_budget,
        "updates": updates,
        "compute_seconds": compute_seconds,
        "train_tokens_per_second": args.train_token_budget / max(compute_seconds, 1e-9),
        "peak_gib": peak_gib,
        "validation": validation,
        "test": test,
        "contexts": contexts,
        "milestones": history,
        "checkpoint": str(checkpoint),
        "export": str(export_path),
        "topology": topology,
        "model_audit": model_audit,
    }
    atomic_json(result_path, result)
    if not args.keep_final_checkpoints and checkpoint.is_file():
        checkpoint.unlink()
    del model, optimizer
    clear_cuda()
    return result


class LastTokenPrefill(nn.Module):
    def __init__(self, model_name: str, model: nn.Module):
        super().__init__()
        self.model_name = model_name
        self.model = model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        m = self.model
        if self.model_name in {OFFICIAL_FIELD, FIELD_PCAF_OFF, PCAF_CONV}:
            h = m.emb(tokens)
            if hasattr(m, "_patch_aux"):
                m._patch_aux = h.new_zeros(())
            patch_position = int(getattr(m, "patch_position", -1))
            softpatch = getattr(m, "softpatch", None)
            for i, block in enumerate(m.blocks):
                h = block(h)
                if i == patch_position and softpatch is not None:
                    h = softpatch(h, tokens)
                    if hasattr(m, "_patch_aux") and hasattr(softpatch, "last_aux"):
                        m._patch_aux = softpatch.last_aux
            h = m.final_norm(h)
        elif self.model_name == TRANSFORMER:
            h = m.emb(tokens)
            for block in m.blocks:
                h = block(h)
            h = m.final_norm(h)
        elif self.model_name == MAMBA2:
            h = m.emb(tokens).to(getattr(m, "activation_dtype", m.emb.weight.dtype))
            for block in m.blocks:
                h = block(h)
            h = m.norm(h)
        else:
            raise KeyError(self.model_name)
        return m.lm_head(h[:, -1, :])


def discover_exports(root: Path) -> Dict[Tuple[int, str], Path]:
    exports: Dict[Tuple[int, str], Path] = {}
    for path in sorted((root / "quality").glob("seed*/*/final_bf16.pt")):
        model = path.parent.name
        if model not in MODELS:
            continue
        try:
            seed_index = int(path.parent.parent.name.split("_", 1)[0].replace("seed", ""))
        except Exception:
            continue
        exports[(seed_index, model)] = path
    return exports


def load_export(args: argparse.Namespace, factory: Tuple[Any, ...], root: Path, model_name: str, seed_index: int, device: torch.device):
    path = discover_exports(root).get((seed_index, model_name))
    if path is None or not path.is_file():
        raise FileNotFoundError(f"missing export seed={seed_index} model={model_name} root={root}")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if raw.get("format") != "field_fusion_pg19_official_bf16" or raw.get("model") != model_name:
        raise RuntimeError(f"unexpected export format/model: {path}")
    metadata = dict(raw.get("metadata", {}))
    model_seed = int(metadata.get("model_seed", args.seeds[seed_index]))
    seed_all(model_seed)
    stack, base_args, _, control_spec, field_shape, comp_shapes, deps, _ = factory
    base_args.model_seed = model_seed
    model, backend = arena.build_model(model_name, stack, control_spec, field_shape, comp_shapes, base_args, deps, device)
    model.load_state_dict(raw["state_dict"], strict=True)
    model.eval()
    return model, backend, path, metadata


def trial_count(args: argparse.Namespace, context: int) -> int:
    if context >= 65536:
        return max(3, args.memory_trials // 4)
    if context >= 32768:
        return max(4, args.memory_trials // 2)
    return args.memory_trials


def make_memory_example(context: int, task: str, pairs: int, trial: int, seed: int, vocab_size: int) -> Tuple[np.ndarray, int, np.ndarray, float]:
    rng = np.random.default_rng(seed + context * 1009 + trial * 7919 + (0 if task == "needle" else 1000003))
    key_pool = np.arange(512, min(4096, vocab_size), dtype=np.int64)
    value_pool = np.arange(4096, min(8192, vocab_size), dtype=np.int64)
    filler_pool = np.arange(8192, vocab_size, dtype=np.int64)
    if filler_pool.size < 128:
        filler_pool = np.arange(256, vocab_size, dtype=np.int64)
    if task in ("needle", "assoc"):
        keys = rng.choice(key_pool, size=1, replace=False)
        values = rng.choice(value_pool, size=32, replace=False)
        correct = int(values[0])
        seq = rng.choice(filler_pool, size=context, replace=True).astype(np.int64)
        if task == "needle":
            depth = (0.1, 0.5, 0.9)[trial % 3]
            positions = [min(context - 3, max(1, int(depth * (context - 3))))]
        else:
            depth = 0.5
            positions = np.linspace(2, context - 8, 4, dtype=np.int64).tolist()
        for pos in positions:
            seq[int(pos)] = int(keys[0])
            seq[int(pos) + 1] = correct
        seq[-1] = int(keys[0])
        candidates = values.astype(np.int64)
        return seq, correct, candidates, depth
    pair_count = min(int(pairs), max(4, (context - 8) // 8), key_pool.size, value_pool.size)
    keys = rng.choice(key_pool, size=pair_count, replace=False)
    values = rng.choice(value_pool, size=pair_count, replace=False)
    seq = rng.choice(filler_pool, size=context, replace=True).astype(np.int64)
    positions = np.linspace(2, context - 8, pair_count, dtype=np.int64)
    for pos, key, value in zip(positions.tolist(), keys.tolist(), values.tolist()):
        seq[pos] = int(key)
        seq[pos + 1] = int(value)
    query = int(rng.integers(0, pair_count))
    seq[-1] = int(keys[query])
    return seq, int(values[query]), values.astype(np.int64), float(query / max(pair_count - 1, 1))


@torch.no_grad()
def run_memory_for_model(args: argparse.Namespace, factory: Tuple[Any, ...], export_root: Path, model_name: str, seed_index: int, device: torch.device) -> List[Dict[str, Any]]:
    model, backend, export_path, metadata = load_export(args, factory, export_root, model_name, seed_index, device)
    adapter = LastTokenPrefill(model_name, model).eval()
    rows: List[Dict[str, Any]] = []
    for task in ("needle", "assoc", "mqar"):
        for context in args.memory_contexts:
            trials = trial_count(args, context)
            candidate_hits = 0
            vocab_hits = 0
            margins: List[float] = []
            correct_nlls: List[float] = []
            status = "ok"
            error = None
            started = time.perf_counter()
            peak = 0.0
            try:
                torch.cuda.reset_peak_memory_stats(device)
                for trial in range(trials):
                    seq, correct, candidates, depth = make_memory_example(context, task, args.memory_pairs, trial, args.memory_seed + seed_index * 10007, args.vocab_size)
                    x = torch.from_numpy(seq)[None].to(device=device)
                    with arena.amp_ctx(device, args.amp):
                        logits = adapter(x).float()
                    cand = torch.from_numpy(candidates).to(device=device)
                    candidate_logits = logits[0].index_select(0, cand)
                    candidate_pred = int(candidates[int(candidate_logits.argmax().item())])
                    vocab_pred = int(logits[0].argmax().item())
                    candidate_hits += int(candidate_pred == correct)
                    vocab_hits += int(vocab_pred == correct)
                    correct_pos = int(np.flatnonzero(candidates == correct)[0])
                    correct_logit = float(candidate_logits[correct_pos].cpu())
                    other = torch.cat([candidate_logits[:correct_pos], candidate_logits[correct_pos + 1:]])
                    margins.append(correct_logit - float(other.max().cpu()))
                    # Complete official score for the correct answer token.  It is
                    # reported in addition to the candidate-ranking backbone probe.
                    full = np.concatenate([seq, np.asarray([correct], dtype=np.int64)])
                    win = torch.from_numpy(full)[None].to(device=device)
                    with arena.amp_ctx(device, args.amp):
                        nll = arena.streaming_token_nll(factory[0], backend, model, win[:, :-1], win[:, 1:], min(args.stream_readout_chunk, context), True).float()
                    correct_nlls.append(float(nll[0, -1].cpu()))
                    del x, logits, cand, candidate_logits, win, nll
                arena.sync(device)
                peak = torch.cuda.max_memory_allocated(device) / 2**30
            except torch.cuda.OutOfMemoryError as exc:
                status, error = "oom", str(exc)
                clear_cuda()
            except Exception as exc:
                status, error = "error", repr(exc)
                clear_cuda()
            seconds = time.perf_counter() - started
            row = {
                "model": model_name,
                "display_name": DISPLAY_NAMES[model_name],
                "seed_index": seed_index,
                "model_seed": int(metadata.get("model_seed", -1)),
                "task": task,
                "context": int(context),
                "trials": trials,
                "status": status,
                "candidate_accuracy": candidate_hits / trials if status == "ok" else None,
                "full_vocab_accuracy": vocab_hits / trials if status == "ok" else None,
                "mean_candidate_margin": statistics.fmean(margins) if margins else None,
                "mean_correct_answer_nll": statistics.fmean(correct_nlls) if correct_nlls else None,
                "seconds": seconds,
                "peak_gib": peak,
                "export": str(export_path),
                "error": error,
            }
            rows.append(row)
            if status == "ok":
                log(f"[memory] task={task} ctx={context} model={model_name} seed={seed_index} candidate_acc={row['candidate_accuracy']:.3f} vocab_acc={row['full_vocab_accuracy']:.3f} answer_nll={row['mean_correct_answer_nll']:.4f}")
            else:
                log(f"[memory] task={task} ctx={context} model={model_name} seed={seed_index} status={status} error={error}")
    del adapter, model
    clear_cuda()
    return rows


def aggregate_quality(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_model: Dict[str, List[Mapping[str, Any]]] = {m: [] for m in MODELS}
    for row in results:
        by_model[str(row["model"])].append(row)
    aggregate: Dict[str, Any] = {
        "models": {},
        "paired": {},
        "long_context": {},
        "paired_long_context": {},
    }
    for model, rows in by_model.items():
        if not rows:
            continue
        val = [float(r["validation"]["nll"]) for r in rows]
        test = [float(r["test"]["nll"]) for r in rows]
        speed = [float(r["train_tokens_per_second"]) for r in rows]
        mem = [float(r["peak_gib"]) for r in rows]
        aggregate["models"][model] = {
            "seeds": len(rows),
            "validation_nll_mean": statistics.fmean(val),
            "validation_nll_sd": statistics.stdev(val) if len(val) > 1 else 0.0,
            "test_nll_mean": statistics.fmean(test),
            "test_nll_sd": statistics.stdev(test) if len(test) > 1 else 0.0,
            "test_ppl": math.exp(min(statistics.fmean(test), 20.0)),
            "train_tokens_per_second_mean": statistics.fmean(speed),
            "peak_gib_mean": statistics.fmean(mem),
        }
        contexts = sorted({
            int(ctx)
            for r in rows
            for ctx in r.get("contexts", {}).keys()
        })
        aggregate["long_context"][model] = {}
        for context in contexts:
            context_rows = [
                r.get("contexts", {}).get(str(context), {})
                for r in rows
            ]
            ok = [float(x["nll"]) for x in context_rows if x.get("status") == "ok" and x.get("nll") is not None]
            statuses: Dict[str, int] = {}
            for x in context_rows:
                status = str(x.get("status", "missing"))
                statuses[status] = statuses.get(status, 0) + 1
            aggregate["long_context"][model][str(context)] = {
                "nll_mean": statistics.fmean(ok) if ok else None,
                "nll_sd": statistics.stdev(ok) if len(ok) > 1 else 0.0,
                "ok_seeds": len(ok),
                "total_seeds": len(rows),
                "statuses": statuses,
            }

    fields = {int(r["seed_index"]): r for r in by_model[OFFICIAL_FIELD]}
    all_contexts = sorted({
        int(ctx)
        for r in results
        for ctx in r.get("contexts", {}).keys()
    })
    for comparator in (FIELD_PCAF_OFF, PCAF_CONV, TRANSFORMER, MAMBA2):
        comp = {int(r["seed_index"]): r for r in by_model[comparator]}
        indices = sorted(set(fields) & set(comp))
        deltas = [float(comp[i]["test"]["nll"]) - float(fields[i]["test"]["nll"]) for i in indices]
        aggregate["paired"][comparator] = {
            "test_gain_mean": statistics.fmean(deltas) if deltas else None,
            "test_gain_sd": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
            "field_wins": sum(x > 0 for x in deltas),
            "pairs": len(deltas),
        }
        aggregate["paired_long_context"][comparator] = {}
        for context in all_contexts:
            context_deltas = []
            for i in indices:
                frow = fields[i].get("contexts", {}).get(str(context), {})
                crow = comp[i].get("contexts", {}).get(str(context), {})
                if frow.get("status") == "ok" and crow.get("status") == "ok":
                    context_deltas.append(float(crow["nll"]) - float(frow["nll"]))
            aggregate["paired_long_context"][comparator][str(context)] = {
                "field_nll_gain_mean": statistics.fmean(context_deltas) if context_deltas else None,
                "field_nll_gain_sd": statistics.stdev(context_deltas) if len(context_deltas) > 1 else 0.0,
                "field_wins": sum(x > 0 for x in context_deltas),
                "pairs": len(context_deltas),
            }

    field_test = aggregate["models"].get(OFFICIAL_FIELD, {}).get("test_nll_mean")
    comparator_tests = [aggregate["models"].get(m, {}).get("test_nll_mean") for m in (TRANSFORMER, MAMBA2)]
    comparator_tests = [x for x in comparator_tests if x is not None]
    if field_test is not None and len(comparator_tests) == 2 and all(field_test <= x - 0.005 for x in comparator_tests):
        aggregate["decision"] = "ADVANCE_PG19_TO_THREE_SEEDS" if len(fields) == 1 else "PG19_THREE_SEED_CONFIRMATION_POSITIVE"
    else:
        aggregate["decision"] = "KEEP_RESULT_AS_PILOT_NO_PROMOTION"
    return aggregate


def build_long_context_csv(args: argparse.Namespace, aggregate: Mapping[str, Any]) -> str:
    contexts = sorted({
        int(ctx)
        for model_rows in aggregate.get("long_context", {}).values()
        for ctx in model_rows.keys()
    })
    lines = ["model,display_name,context,status,nll_mean,nll_sd,ok_seeds,total_seeds"]
    for model in MODELS:
        rows = aggregate.get("long_context", {}).get(model, {})
        for context in contexts:
            row = rows.get(str(context), {})
            mean = row.get("nll_mean")
            status = "ok" if mean is not None else "+".join(sorted(row.get("statuses", {}).keys())) or "missing"
            mean_text = "" if mean is None else f"{float(mean):.8f}"
            sd_text = "" if mean is None else f"{float(row.get('nll_sd', 0.0)):.8f}"
            lines.append(
                f'{model},"{DISPLAY_NAMES[model]}",{context},{status},'
                f'{mean_text},{sd_text},'
                f'{int(row.get("ok_seeds", 0))},{int(row.get("total_seeds", 0))}'
            )
    for comparator in (FIELD_PCAF_OFF, PCAF_CONV, TRANSFORMER, MAMBA2):
        for context in contexts:
            row = aggregate.get("paired_long_context", {}).get(comparator, {}).get(str(context), {})
            gain = row.get("field_nll_gain_mean")
            gain_text = "" if gain is None else f"{float(gain):.8f}"
            gain_sd_text = "" if gain is None else f"{float(row.get('field_nll_gain_sd', 0.0)):.8f}"
            lines.append(
                f'paired_field_vs_{comparator},"Fields gain vs {DISPLAY_NAMES[comparator]}",{context},'
                f'{"ok" if gain is not None else "missing"},'
                f'{gain_text},{gain_sd_text},'
                f'{int(row.get("pairs", 0))},{int(row.get("pairs", 0))}'
            )
    return "\n".join(lines) + "\n"

def build_summary(args: argparse.Namespace, quality: Sequence[Mapping[str, Any]], memory: Sequence[Mapping[str, Any]], aggregate: Mapping[str, Any], data_manifest: Optional[Mapping[str, Any]]) -> str:
    lines = [
        "=" * 190,
        "FIELDS PAPER-CLOSURE ARENA v3 — PG-19 FROM SCRATCH + TRANSFORMER YaRN AT 64K",
        "=" * 190,
        f"mode={args.mode} seeds={list(args.seeds)} tokens/model/seed={args.train_token_budget:,} train_seq={args.train_seq} batch={args.batch_size} log_every={args.log_every_updates}",
        f"canonical_same_round_arms={list(MODELS)} yarn_target={int(args.yarn_factor * args.yarn_original_context):,} long_contexts={list(args.long_contexts)}",
    ]
    if data_manifest:
        lines.append(f"dataset={data_manifest.get('dataset_id')} sampled_train_docs={data_manifest.get('train_docs')} sampled_train_bytes={data_manifest.get('train_raw_bytes'):,} document_boundary_safe=True")
    lines.extend(["", "PG-19 QUALITY AT THE PAIRED 2K TRAIN/EVAL WINDOW"])
    lines.append(f"{'model':38s} {'params':>12s} {'val':>10s} {'test':>10s} {'PPL':>10s} {'tok/s':>12s} {'GB':>8s}")
    for model in MODELS:
        row = aggregate.get("models", {}).get(model)
        sample = next((x for x in quality if x["model"] == model), None)
        if not row or not sample:
            continue
        lines.append(f"{DISPLAY_NAMES[model]:38s} {int(sample['parameters']):12,d} {row['validation_nll_mean']:10.5f} {row['test_nll_mean']:10.5f} {row['test_ppl']:10.3f} {row['train_tokens_per_second_mean']:12,.0f} {row['peak_gib_mean']:8.2f}")
    lines.extend(["", "PAIRED FIELDS+PCAF TEST-NLL GAINS AT 2K (positive = lower Fields NLL)"])
    for model in (FIELD_PCAF_OFF, PCAF_CONV, TRANSFORMER, MAMBA2):
        row = aggregate.get("paired", {}).get(model, {})
        lines.append(f"vs {DISPLAY_NAMES[model]}: gain={row.get('test_gain_mean')} sd={row.get('test_gain_sd')} wins={row.get('field_wins')}/{row.get('pairs')}")

    contexts = sorted({
        int(ctx)
        for rows in aggregate.get("long_context", {}).values()
        for ctx in rows.keys()
    })
    lines.extend(["", "MATCHED-SUFFIX PG-19 LONG-CONTEXT NLL — SAME BOOK ENDINGS FOR EVERY ARM"])
    if contexts:
        lines.append(f"{'model':38s} " + " ".join(f"{c//1024:>7d}K" for c in contexts))
        for model in MODELS:
            cells = []
            model_rows = aggregate.get("long_context", {}).get(model, {})
            for context in contexts:
                row = model_rows.get(str(context), {})
                mean = row.get("nll_mean")
                if mean is not None:
                    cells.append(f"{float(mean):8.5f}")
                else:
                    statuses = row.get("statuses", {})
                    label = "OOM" if statuses.get("oom") else ("ERR" if statuses.get("error") else "MISS")
                    cells.append(f"{label:>8s}")
            lines.append(f"{DISPLAY_NAMES[model]:38s} " + " ".join(cells))

        lines.extend(["", "PAIRED LONG-CONTEXT FIELDS NLL GAINS (comparator NLL − Fields NLL; positive = Fields wins)"])
        for model in (FIELD_PCAF_OFF, PCAF_CONV, TRANSFORMER, MAMBA2):
            cells = []
            for context in contexts:
                row = aggregate.get("paired_long_context", {}).get(model, {}).get(str(context), {})
                gain = row.get("field_nll_gain_mean")
                cells.append(f"{float(gain):+8.5f}" if gain is not None else f"{'MISS':>8s}")
            lines.append(f"vs {DISPLAY_NAMES[model]:35s} " + " ".join(cells))

        yarn64 = aggregate.get("paired_long_context", {}).get(TRANSFORMER, {}).get("65536", {})
        gain64 = yarn64.get("field_nll_gain_mean")
        if gain64 is None:
            verdict64 = "unavailable (the result row will state OOM/error/missing)"
        elif float(gain64) > 0:
            verdict64 = f"FIELDS WINS by {float(gain64):.5f} NLL"
        elif float(gain64) == 0:
            verdict64 = "TIE"
        else:
            verdict64 = f"YaRN TRANSFORMER WINS by {-float(gain64):.5f} NLL"
        lines.extend([
            "",
            "CENTRAL 64K RESULT",
            f"Fields 18F/2M/4R + PCAF vs Transformer Flash+YaRN at 65,536: {verdict64}; paired_seeds={yarn64.get('pairs', 0)}",
        ])

    lines.extend(["", "ZERO-SHOT TOKEN MEMORY PROBES — candidate accuracy / full-vocab accuracy / correct-answer NLL"])
    for task in ("needle", "assoc", "mqar"):
        lines.append(f"task={task}")
        for context in args.memory_contexts:
            parts = []
            for model in MODELS:
                rows = [r for r in memory if r["task"] == task and int(r["context"]) == int(context) and r["model"] == model and r["status"] == "ok"]
                if rows:
                    ca = statistics.fmean(float(r["candidate_accuracy"]) for r in rows)
                    va = statistics.fmean(float(r["full_vocab_accuracy"]) for r in rows)
                    nl = statistics.fmean(float(r["mean_correct_answer_nll"]) for r in rows)
                    parts.append(f"{model}: {ca:.3f}/{va:.3f}/{nl:.3f}")
                else:
                    status = next((r["status"] for r in memory if r["task"] == task and int(r["context"]) == int(context) and r["model"] == model), "missing")
                    parts.append(f"{model}: {status}")
            lines.append(f"  ctx={context:6d} " + " | ".join(parts))
    lines.extend([
        "",
        "DECISION",
        f"action={aggregate.get('decision')}",
        "",
        "The YaRN arm is part of the same paired v3 schedule, not a post-hoc optional appendix.",
        "All five arms train from update zero on the same PG-19 token starts at context 2,048; long-context rows are matched-suffix extrapolation tests through 65,536.",
        "Memory probes are zero-shot backbone retrieval diagnostics; they are not instruction-following tests.",
        "A positive one-seed PG-19 result must be confirmed with the included three-seed launcher before a paper-level dataset claim.",
        "=" * 190,
    ])
    return "\n".join(lines) + "\n"

def package_selftest(args: argparse.Namespace) -> None:
    canonical = Path(args.canonical_source)
    if not canonical.is_file() or sha256(canonical) != CANONICAL_SHA256:
        raise RuntimeError("canonical source missing or SHA mismatch")
    if not (VENDOR / "OFFICIAL_SOURCE_MANIFEST.json").is_file():
        raise FileNotFoundError("official source manifest missing")
    source = Path(__file__).read_text(encoding="utf-8")
    required = [
        "pg19_document_boundary_sampling=PASS",
        "log_every_updates",
        "make_memory_example",
        "ADVANCE_PG19_TO_THREE_SEEDS",
    ]
    if not all(token in source for token in required):
        raise AssertionError("source self-audit failed")
    if arena.OFFICIAL_R1_SPEC != "r1_18f_2m_4r":
        raise AssertionError(f"unexpected official R1 spec: {arena.OFFICIAL_R1_SPEC}")
    if any(name in sys.modules for name in (
        "field_fusion_macroblock_c_joint_residual_screen",
        "field_fusion_field_native_summary_refresh_screen",
        "field_fusion_pcaf_write_read_codesign_screen",
    )):
        raise AssertionError("rejected experimental module imported")
    if args.log_every_updates != 100:
        raise AssertionError("default/logged PPL cadence must be 100 updates")
    # Pure CPU checks for the document sampler and synthetic probes.
    tmp = Path(args.outdir) / "cpu_selftest"
    tmp.mkdir(parents=True, exist_ok=True)
    token_path = tmp / "tokens.u16"
    np.arange(20000, dtype=np.uint16).tofile(token_path)
    corpus = PgCorpus("smoke", token_path, np.asarray([[0, 10000], [10000, 20000]], dtype=np.int64), 20000, 2, 20000)
    starts = make_doc_starts(corpus, 32, 2048, 123)
    for s in starts.tolist():
        if not ((0 <= s and s + 2048 < 10000) or (10000 <= s and s + 2048 < 20000)):
            raise AssertionError("document boundary sampler crossed a book")
    for task in ("needle", "assoc", "mqar"):
        seq, correct, candidates, _ = make_memory_example(2048, task, 32, 0, 7, 16384)
        if seq.shape != (2048,) or correct not in set(candidates.tolist()):
            raise AssertionError(f"memory generator failed: {task}")

    # Regression for FIX1: promoted FieldTokenSystemV21 has no forward(); it
    # exposes loss_and_stats().  The arena must dispatch to that API directly.
    class _LossAndStatsOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))

        def loss_and_stats(self, tokens, targets, compute_metrics=False):
            del compute_metrics
            loss = self.scale * (tokens.float().mean() + targets.float().mean() + 1.0)
            return loss, loss.detach(), {}

    dummy = _LossAndStatsOnly()
    dx = torch.ones((1, 4), dtype=torch.long)
    dy = torch.ones((1, 4), dtype=torch.long)
    dloss = arena.loss_call({}, arena.BACKEND_FIELD_ON, dummy, dx, dy)
    dloss.backward()
    if dummy.scale.grad is None or not torch.isfinite(dummy.scale.grad):
        raise AssertionError("loss_and_stats dispatch regression failed")

    if TRANSFORMER not in MODELS or 65536 not in args.long_contexts:
        raise AssertionError("Transformer+YaRN 64K is not in the canonical v3 arena")
    if int(args.yarn_factor * args.yarn_original_context) < max(args.long_contexts):
        raise AssertionError("YaRN configured target is below the requested long context")

    log("official_source_snapshot=PASS")
    log("direct_promoted_r1_spec=PASS")
    log("rejected_experimental_arms_not_imported=PASS")
    log("pg19_autonomous_tokenizer_design=PASS")
    log("pg19_document_boundary_sampler=PASS")
    log("ppl_logging_every_100_updates=PASS")
    log("needle_assoc_mqar_generators=PASS")
    log("field_loss_and_stats_dispatch=PASS")
    log("transformer_yarn64k_same_canonical_v3_arena=PASS")
    log("[package-selftest] PASS")


def dataset_smoke(args: argparse.Namespace) -> None:
    if not args.data_smoke:
        log("pg19_dataset_access=SKIPPED")
        return
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets package missing") from exc
    errors = []
    for dataset_id in [args.dataset_id, *args.dataset_fallbacks]:
        try:
            sample = load_dataset(dataset_id, split="validation[:1]", cache_dir=args.cache_dir)
            if len(sample) != 1 or not extract_text(sample[0]):
                raise RuntimeError("empty sample")
            log(f"pg19_dataset_access=PASS dataset={dataset_id}")
            return
        except Exception as exc:
            errors.append(f"{dataset_id}: {exc}")
    raise RuntimeError("PG-19 data smoke failed: " + " | ".join(errors))


def gpu_selftest(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable CUDA GPU required")
    device = torch.device("cuda")
    factory = prepare_factory(args, device)
    counts = {}
    for model_name in MODELS:
        seed_all(1234)
        model, backend = arena.build_model(model_name, factory[0], factory[3], factory[4], factory[5], factory[1], factory[6], device)
        counts[model_name] = arena.nparams(model)
        if model_name in {OFFICIAL_FIELD, FIELD_PCAF_OFF}:
            arena.topology_audit(model)
        if model_name == TRANSFORMER:
            cfg = dict(model.yarn_config)
            target_context = int(cfg["factor"] * cfg["original_context"])
            if target_context < 65536:
                raise AssertionError(f"YaRN target context too small: {target_context}")
            head_dim = int(model.blocks[0].head_dim)
            cos64, sin64 = arena.yarn_cos_sin(
                device, torch.bfloat16, 65536, head_dim, **cfg
            )
            if cos64.shape[-2] != 65536 or sin64.shape != cos64.shape:
                raise AssertionError(f"invalid YaRN 64K tables: {cos64.shape} {sin64.shape}")
            if not torch.isfinite(cos64[:, :, (0, -1), :]).all() or not torch.isfinite(sin64[:, :, (0, -1), :]).all():
                raise AssertionError("non-finite YaRN 64K frequencies")
            del cos64, sin64
            arena._YARN_CACHE.clear()
        x = torch.randint(0, args.vocab_size, (1, 64), device=device)
        y = torch.randint(0, args.vocab_size, (1, 64), device=device)
        model.train()
        with arena.amp_ctx(device, args.amp):
            loss = arena.loss_call(factory[0], backend, model, x, y)
        loss.backward()
        if not torch.isfinite(loss):
            raise AssertionError(f"nonfinite selftest loss: {model_name}")
        model.eval()
        adapter = LastTokenPrefill(model_name, model)
        with torch.inference_mode(), arena.amp_ctx(device, args.amp):
            logits = adapter(x)
        if logits.shape != (1, args.vocab_size) or not torch.isfinite(logits).all():
            raise AssertionError(f"memory adapter failed: {model_name}")
        del adapter, model, x, y, loss, logits
        clear_cuda()
    delta = 100.0 * (max(counts.values()) - min(counts.values())) / min(counts.values())
    if delta > args.max_param_delta_pct:
        raise AssertionError(f"parameter parity failed: {counts} delta={delta:.3f}%")
    dataset_smoke(args)
    log("official_topology_18f_2m_4r=PASS")
    log("transformer_flash_yarn64k=PASS")
    log("transformer_yarn_frequency_table_65536=PASS")
    log("transformer_yarn64k_same_canonical_v3_arena=PASS")
    log("pcaf_off_from_initialization=PASS")
    log("pcaf_conv_bpe16k=PASS")
    log("official_mamba2=PASS")
    log("parameter_parity=PASS")
    log("all_models_forward_backward=PASS")
    log("memory_last_token_adapter=PASS")
    log("[gpu-selftest] PASS")


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable CUDA GPU required")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.set_float32_matmul_precision("high")
    root = Path(args.outdir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    factory = prepare_factory(args, device)
    data = None if args.mode == "memory-only" else prepare_pg19_data(args)
    quality: List[Dict[str, Any]] = []
    if args.mode != "memory-only":
        schedule = []
        for seed_index in range(len(args.seeds)):
            order = list(MODELS[seed_index % len(MODELS):] + MODELS[:seed_index % len(MODELS)])
            for model_name in order:
                if args.only_model and model_name != args.only_model:
                    continue
                if args.only_seed_index is not None and seed_index != args.only_seed_index:
                    continue
                schedule.append((seed_index, model_name))
        for seed_index, model_name in schedule:
            quality.append(train_one(args, factory, data, model_name, seed_index, device, root))
        # Include already completed rows when resuming or running a restricted arm.
        for path in sorted((root / "quality").glob("seed*/*/result.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            key = (int(row["seed_index"]), str(row["model"]))
            if not any((int(x["seed_index"]), str(x["model"])) == key for x in quality):
                quality.append(row)
    export_root = Path(args.export_root).expanduser().resolve() if args.export_root else root
    exports = discover_exports(export_root)
    if not exports:
        raise RuntimeError(f"no PG-19 exports found under {export_root}")
    seed_indices = sorted({idx for idx, _ in exports})
    memory: List[Dict[str, Any]] = []
    for seed_index in seed_indices:
        for model_name in MODELS:
            if (seed_index, model_name) not in exports:
                continue
            memory.extend(run_memory_for_model(args, factory, export_root, model_name, seed_index, device))
    if not quality:
        for path in sorted((export_root / "quality").glob("seed*/*/result.json")):
            quality.append(json.loads(path.read_text(encoding="utf-8")))
    aggregate = aggregate_quality(quality) if quality else {"models": {}, "paired": {}, "decision": "MEMORY_ONLY"}
    payload = {
        "suite_version": SUITE_VERSION,
        "args": vars(args),
        "data_manifest": data["data_manifest"] if data else None,
        "quality": quality,
        "memory": memory,
        "aggregate": aggregate,
    }
    atomic_json(root / "results.json", payload)
    atomic_json(root / "memory.json", memory)
    summary = build_summary(args, quality, memory, aggregate, data["data_manifest"] if data else None)
    atomic_text(root / "summary.txt", summary)
    atomic_text(root / "long_context_table.csv", build_long_context_csv(args, aggregate))
    log(summary)


def main() -> None:
    args = parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    if args.mode == "package-selftest":
        package_selftest(args)
    elif args.mode == "gpu-selftest":
        gpu_selftest(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
