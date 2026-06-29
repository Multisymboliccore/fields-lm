"""Public API for Fields LM."""

from .fields_official import (
    OFFICIAL_ARCHITECTURE,
    OFFICIAL_CANONICAL_SHA256,
    FieldsConfig,
    FieldsForwardOutput,
    FieldsForCausalLM,
    build_official_fields,
    verify_frozen_source,
)
from .hub import FieldsHubModel

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "OFFICIAL_ARCHITECTURE",
    "OFFICIAL_CANONICAL_SHA256",
    "FieldsConfig",
    "FieldsForwardOutput",
    "FieldsForCausalLM",
    "FieldsHubModel",
    "build_official_fields",
    "verify_frozen_source",
]
