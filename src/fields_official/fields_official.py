#!/usr/bin/env python3
"""Official public API for the promoted Fields 18F/2M/4R + PCAF model.

This module is intentionally a *faithful facade*, not a clean-room rewrite.
The paper model was promoted through a chain of audited research modules.  A
premature monolithic rewrite would be easier to read, but it could silently
change initialization, recurrence, PCAF routing, optimizer grouping, or block
placement.  The public API below therefore does three things:

1. verifies the frozen source snapshot and canonical Field kernel by SHA-256;
2. reconstructs the exact promoted 18-Field / 2-Mamba-2 / 4-refresh topology;
3. exposes a conventional ``nn.Module.forward`` interface for third parties.

Architecture (zero-indexed block positions)
--------------------------------------------

    token ids
       │
       ▼
    BPE-16K embedding, d_model=1024
       │
       ├─  0..4   Native Field blocks
       ├─  5      Refresh-attention station
       ├─  6..9   Native Field blocks
       ├─ 10      Official Mamba-2 editor
       ├─ 11      Refresh-attention station
       ├─ 12..16  Native Field blocks
       ├─ 17      Refresh-attention station
       ├─ 18..21  Native Field blocks
       ├─ 22      Official Mamba-2 editor
       └─ 23      Refresh-attention station
       │
       ▼
    final normalization → tied language-model head
       │
       └─ PCAF successor memory (enabled in the paper model)

The native Field block retains the validated independent-vacancy complex
recurrence and displaced readout.  The CUDA path uses the exact chunk-parallel
Triton implementation from the frozen canonical source.

Release status
--------------
This file is suitable as the stable public entry point.  Before tagging a
GitHub v1.0 release, run the included GPU equivalence test against the frozen
paper checkpoint/source and record the resulting hashes in the release notes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn


# ======================================================================================
# 1. Frozen identity and promoted topology
# ======================================================================================

OFFICIAL_CANONICAL_SHA256 = (
    "0454647a7f667cdf73f45029eee0f525c58c0788ecd433ec1a4db8c3937ac848"
)

OFFICIAL_ARCHITECTURE: Mapping[str, Any] = {
    "name": "Fields 18F/2M/4R + PCAF",
    "vocabulary_size": 16_384,
    "target_parameters": 300_000_000,
    "model_dimension": 1_024,
    "block_count": 24,
    "native_field_positions": (
        0, 1, 2, 3, 4,
        6, 7, 8, 9,
        12, 13, 14, 15, 16,
        18, 19, 20, 21,
    ),
    "mamba2_positions": (10, 22),
    "refresh_positions": (5, 11, 17, 23),
    "refresh_windows": (256, 512, 1024, 1024),
    "pcaf_enabled": True,
    "field_chunk": 32,
    "triton_channel_block": 32,
    "triton_time_chunk": 64,
}


# ======================================================================================
# 2. User-facing configuration
# ======================================================================================

@dataclass(frozen=True)
class FieldsConfig:
    """Configuration for reconstructing the exact promoted paper model.

    The architecture-defining values are frozen.  The fields below control
    source location, initialization, precision, and runtime placement only.

    Parameters
    ----------
    source_root:
        Directory containing ``reference/``.  By default this is inferred from
        the installed package.
    canonical_source:
        Exact ``field_only_v4_chunked_triton_wiki100.py`` file.  Its SHA-256
        must equal :data:`OFFICIAL_CANONICAL_SHA256`.
    cache_dir:
        Hugging Face / tokenizer cache directory used by legacy dependency
        loaders.  Model construction itself does not download a dataset.
    model_seed:
        Parameter initialization seed.
    embedding_seed:
        Shared tied-embedding initialization seed used in the paper arena.
    pcaf_enabled:
        ``True`` reconstructs the paper model.  ``False`` is exposed solely for
        the registered ablation trained from update zero without PCAF.
    amp:
        Runtime precision declaration used by the frozen constructors.
    """

    source_root: Optional[Path] = None
    canonical_source: Optional[Path] = None
    cache_dir: Path = Path.home() / "field_lab" / "hf_cache"
    output_dir: Path = Path.home() / "pcaf_runs" / "fields_official_api"
    model_seed: int = 1234
    embedding_seed: int = 314159
    pcaf_enabled: bool = True
    amp: str = "bf16"

    def resolved_source_root(self) -> Path:
        if self.source_root is not None:
            return Path(self.source_root).expanduser().resolve()
        return (Path(__file__).resolve().parent / "reference").resolve()

    def resolved_canonical_source(self) -> Path:
        if self.canonical_source is not None:
            return Path(self.canonical_source).expanduser().resolve()
        return self.resolved_source_root() / "field_only_v4_chunked_triton_wiki100.py"


@dataclass
class FieldsForwardOutput:
    """Structured output returned by :class:`FieldsForCausalLM`.

    ``logits`` is available when labels are omitted.  During supervised
    training the exact native ``loss_and_stats`` path is used; this avoids a
    second forward pass and preserves the paper implementation byte-for-byte.
    """

    loss: Optional[torch.Tensor] = None
    primary_loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    states: Optional[torch.Tensor] = None
    stats: Optional[Mapping[str, Any]] = None


# ======================================================================================
# 3. Source-integrity verification
# ======================================================================================

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_source(config: FieldsConfig) -> Dict[str, Any]:
    """Verify every frozen source file before importing the architecture.

    Returns a serializable audit dictionary.  Any missing file or hash mismatch
    raises immediately, before model allocation or random initialization.
    """

    root = config.resolved_source_root()
    vendor = root / "official_source"
    manifest_path = vendor / "OFFICIAL_SOURCE_MANIFEST.json"
    canonical = config.resolved_canonical_source()

    if not manifest_path.is_file():
        raise FileNotFoundError(f"official source manifest not found: {manifest_path}")
    if not canonical.is_file():
        raise FileNotFoundError(
            "canonical Field kernel not found. Copy the exact paper source to "
            f"{canonical} (expected SHA-256 {OFFICIAL_CANONICAL_SHA256})."
        )

    canonical_hash = _sha256(canonical)
    if canonical_hash != OFFICIAL_CANONICAL_SHA256:
        raise RuntimeError(
            "canonical source SHA-256 mismatch: "
            f"expected={OFFICIAL_CANONICAL_SHA256} actual={canonical_hash} "
            f"path={canonical}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verified = []
    for row in manifest.get("files", []):
        path = vendor / str(row["path"])
        if not path.is_file():
            raise FileNotFoundError(f"frozen source file missing: {path}")
        actual = _sha256(path)
        expected = str(row["sha256"])
        if actual != expected:
            raise RuntimeError(
                f"frozen source SHA-256 mismatch: file={path.name} "
                f"expected={expected} actual={actual}"
            )
        verified.append({"path": path.name, "sha256": actual})

    return {
        "canonical_path": str(canonical),
        "canonical_sha256": canonical_hash,
        "manifest_path": str(manifest_path),
        "verified_snapshot_files": verified,
        "architecture": dict(OFFICIAL_ARCHITECTURE),
    }


# ======================================================================================
# 4. Conventional public nn.Module facade
# ======================================================================================

class FieldsForCausalLM(nn.Module):
    """Thin, lossless API wrapper around the frozen promoted implementation."""

    def __init__(
        self,
        core_model: nn.Module,
        *,
        backend_name: str,
        source_audit: Mapping[str, Any],
        topology_audit: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.core_model = core_model
        self.backend_name = str(backend_name)
        self.source_audit = dict(source_audit)
        self.topology_audit = dict(topology_audit)

    @property
    def pcaf_enabled(self) -> bool:
        cache = getattr(self.core_model, "cache", None)
        return bool(getattr(cache, "enabled", False))

    @property
    def config_dict(self) -> Dict[str, Any]:
        return {
            **dict(OFFICIAL_ARCHITECTURE),
            "pcaf_enabled": self.pcaf_enabled,
            "backend_name": self.backend_name,
            "topology_audit": self.topology_audit,
            "canonical_sha256": self.source_audit["canonical_sha256"],
        }

    def set_pcaf_enabled(self, enabled: bool) -> None:
        """Switch PCAF routing for controlled inference/ablation use.

        Paper-quality comparisons must not toggle this midway through training.
        The registered no-PCAF ablation initializes and trains with this flag
        disabled from update zero.
        """

        cache = getattr(self.core_model, "cache", None)
        if cache is None or not hasattr(cache, "enabled"):
            raise AttributeError("the frozen model does not expose cache.enabled")
        cache.enabled = bool(enabled)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        *,
        compute_metrics: bool = False,
        return_states: bool = False,
    ) -> FieldsForwardOutput:
        """Run causal language modeling through the exact native API.

        With ``labels=None``, the method returns logits and optionally hidden
        states.  With labels, it delegates to ``loss_and_stats`` because the
        promoted model intentionally does not define a legacy ``forward``.
        """

        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise TypeError("input_ids must be a rank-2 torch.long tensor [batch, time]")

        if labels is None:
            states, logits = self.core_model.states_logits(input_ids)
            return FieldsForwardOutput(
                logits=logits,
                states=states if return_states else None,
            )

        if labels.shape != input_ids.shape or labels.dtype != torch.long:
            raise TypeError("labels must match input_ids shape and use torch.long")
        if not hasattr(self.core_model, "loss_and_stats"):
            raise AttributeError("frozen promoted model is missing loss_and_stats")

        result = self.core_model.loss_and_stats(
            input_ids,
            labels,
            compute_metrics=bool(compute_metrics),
        )
        if not isinstance(result, (tuple, list)) or len(result) < 3:
            raise TypeError("loss_and_stats returned an unexpected value")
        loss, primary_loss, stats = result[:3]
        return FieldsForwardOutput(
            loss=loss,
            primary_loss=primary_loss,
            stats=stats,
        )


# ======================================================================================
# 5. Exact model reconstruction
# ======================================================================================

def _factory_namespace(config: FieldsConfig) -> argparse.Namespace:
    """Create the minimal namespace consumed by the frozen paper constructor."""

    return argparse.Namespace(
        outdir=str(config.output_dir.expanduser()),
        canonical_source=str(config.resolved_canonical_source()),
        cache_dir=str(config.cache_dir.expanduser()),
        data_root=str(config.output_dir.expanduser() / "data"),
        amp=config.amp,
        seeds=[int(config.model_seed)],
        data_seeds=[5678],
        eval_seed=9012,
        memory_seed=77123,
        train_token_budget=49_152_000,
        train_seq=2_048,
        batch_size=4,
        lr=3e-4,
        weight_decay=0.10,
        grad_clip=1.0,
        warmup_fraction=0.02,
        wsd_stable_fraction=0.70,
        min_lr_ratio=0.10,
        log_every_updates=100,
        checkpoint_every_updates=500,
        eval_milestones=[25_165_824, 49_152_000],
        validation_token_budget=1_048_576,
        test_token_budget=1_048_576,
        stream_readout_chunk=512,
        long_contexts=[2_048, 8_192, 16_384, 32_768, 65_536],
        long_context_windows=8,
        long_context_score_tokens=128,
        memory_contexts=[2_048, 8_192, 16_384, 32_768, 65_536],
        memory_trials=12,
        memory_pairs=64,
        max_param_delta_pct=0.75,
        yarn_factor=32.0,
        yarn_original_context=2_048,
        yarn_beta_fast=32.0,
        yarn_beta_slow=1.0,
        yarn_rope_theta=10_000.0,
        yarn_truncate=True,
        yarn_gradient_checkpointing=False,
        resume=False,
        keep_final_checkpoints=True,
        only_model=None,
        only_seed_index=None,
    )


def build_official_fields(
    config: Optional[FieldsConfig] = None,
    *,
    device: Optional[torch.device | str] = None,
) -> FieldsForCausalLM:
    """Reconstruct and return the exact promoted Fields paper model.

    This function does not load PG-19 or WikiText-103.  It imports the frozen
    architecture stack, solves the registered parameter-matched shape, creates
    the model from scratch, and performs the 18F/2M/4R topology audit.
    """

    cfg = config or FieldsConfig()
    audit = verify_frozen_source(cfg)
    root = cfg.resolved_source_root()
    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)

    suite = importlib.import_module("field_fusion_final_closure_suite")
    arena = importlib.import_module("final_closure_arena_core")

    target_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    args = _factory_namespace(cfg)
    factory = suite.prepare_factory(args, target_device)
    stack, base_args, _, control_spec, field_shape, _, deps, _ = factory

    base_args.model_seed = int(cfg.model_seed)
    base_args.embedding_seed = int(cfg.embedding_seed)
    model_name = arena.FIELD_PCAF_ON if cfg.pcaf_enabled else arena.FIELD_PCAF_OFF
    model, backend = arena.build_model(
        model_name,
        stack,
        control_spec,
        field_shape,
        {},
        base_args,
        deps,
        target_device,
    )
    topology = arena.topology_audit(model)
    return FieldsForCausalLM(
        model,
        backend_name=backend,
        source_audit=audit,
        topology_audit=topology,
    )
