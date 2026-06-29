"""Hugging Face Hub integration for the official Fields runtime.

This module deliberately separates *runtime code* from *checkpoint artifacts*:

- install the audited runtime from the official GitHub repository;
- download model weights and tokenizer files from a Hugging Face model repo;
- reconstruct the exact architecture locally and load safe tensor weights.

The integration uses ``ModelHubMixin`` with explicit safetensors serialization.
It does not rely on pickle and does not execute Python code downloaded from a
model repository.  The architecture code comes from the installed, versioned
``fields-official`` package instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn
from huggingface_hub import ModelHubMixin, snapshot_download
from safetensors.torch import load_model, save_model

from .fields_official import (
    FieldsConfig,
    FieldsForwardOutput,
    FieldsForCausalLM,
    build_official_fields,
)


class FieldsHubModel(
    nn.Module,
    ModelHubMixin,
    library_name="fields-lm",
    repo_url="https://github.com/Multisymboliccore/fields-lm",
    pipeline_tag="text-generation",
    tags=["pytorch", "causal-lm", "long-context", "state-space-model", "research"],
):
    """Serializable Hub-facing wrapper for Fields 18F/2M/4R + PCAF.

    Parameters are intentionally limited to JSON-serializable values because
    they are persisted in ``config.json`` alongside the checkpoint.
    """

    CONFIG_FILENAME = "config.json"
    WEIGHTS_FILENAME = "model.safetensors"

    def __init__(
        self,
        *,
        model_seed: int = 1234,
        embedding_seed: int = 314159,
        pcaf_enabled: bool = True,
        amp: str = "bf16",
    ) -> None:
        super().__init__()
        self.model_seed = int(model_seed)
        self.embedding_seed = int(embedding_seed)
        self.pcaf_enabled_at_build = bool(pcaf_enabled)
        self.amp = str(amp)

        public = build_official_fields(
            FieldsConfig(
                model_seed=self.model_seed,
                embedding_seed=self.embedding_seed,
                pcaf_enabled=self.pcaf_enabled_at_build,
                amp=self.amp,
            ),
            device="cpu",
        )
        self.core_model = public.core_model
        self.backend_name = public.backend_name
        self.source_audit = public.source_audit
        self.topology_audit = public.topology_audit

    @property
    def hub_config(self) -> Dict[str, Any]:
        return {
            "architectures": ["FieldsHubModel"],
            "model_type": "fields",
            "model_seed": self.model_seed,
            "embedding_seed": self.embedding_seed,
            "pcaf_enabled": self.pcaf_enabled_at_build,
            "amp": self.amp,
            "vocab_size": 16_384,
            "model_max_length": 65_536,
            "library_name": "fields-lm",
            "canonical_sha256": self.source_audit["canonical_sha256"],
            "backend_name": self.backend_name,
            "topology_audit": self.topology_audit,
            "safe_serialization": True,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        *,
        compute_metrics: bool = False,
        return_states: bool = False,
    ) -> FieldsForwardOutput:
        facade = FieldsForCausalLM(
            self.core_model,
            backend_name=self.backend_name,
            source_audit=self.source_audit,
            topology_audit=self.topology_audit,
        )
        return facade(
            input_ids,
            labels,
            compute_metrics=compute_metrics,
            return_states=return_states,
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        """Write ``config.json`` and ``model.safetensors`` atomically enough for release use."""

        target = Path(save_directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / self.CONFIG_FILENAME).write_text(
            json.dumps(self.hub_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        save_model(
            self,
            str(target / self.WEIGHTS_FILENAME),
            metadata={
                "format": "pt",
                "architecture": "Fields 18F/2M/4R + PCAF",
                "canonical_sha256": self.source_audit["canonical_sha256"],
            },
        )

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Path],
        force_download: bool,
        local_files_only: bool,
        token: Optional[str | bool],
        map_location: str | torch.device = "cpu",
        strict: bool = True,
        **model_kwargs: Any,
    ) -> "FieldsHubModel":
        """Download a snapshot, reconstruct Fields, and load safetensors weights."""

        candidate = Path(model_id).expanduser()
        if candidate.is_dir():
            root = candidate.resolve()
        else:
            root = Path(
                snapshot_download(
                    repo_id=model_id,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                    token=token,
                    allow_patterns=[
                        cls.CONFIG_FILENAME,
                        cls.WEIGHTS_FILENAME,
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "special_tokens_map.json",
                        "README.md",
                        "SOURCE_IDENTITY.txt",
                        "ENVIRONMENT.txt",
                    ],
                )
            )

        config_path = root / cls.CONFIG_FILENAME
        weights_path = root / cls.WEIGHTS_FILENAME
        if not config_path.is_file():
            raise FileNotFoundError(f"missing Hub config: {config_path}")
        if not weights_path.is_file():
            raise FileNotFoundError(f"missing safetensors weights: {weights_path}")

        config: Mapping[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
        init_kwargs = {
            "model_seed": int(config.get("model_seed", 1234)),
            "embedding_seed": int(config.get("embedding_seed", 314159)),
            "pcaf_enabled": bool(config.get("pcaf_enabled", True)),
            "amp": str(config.get("amp", "bf16")),
        }
        for key in tuple(init_kwargs):
            if key in model_kwargs:
                init_kwargs[key] = model_kwargs.pop(key)
        if model_kwargs:
            unexpected = ", ".join(sorted(model_kwargs))
            raise TypeError(f"unexpected FieldsHubModel load arguments: {unexpected}")

        model = cls(**init_kwargs)
        load_model(model, str(weights_path), strict=bool(strict), device=str(map_location))
        model.eval()
        return model
