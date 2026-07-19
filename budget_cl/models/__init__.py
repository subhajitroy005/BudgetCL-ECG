"""Model construction: tiny Transformer, LoRA, adapters, and the arm registry.

TensorFlow is imported lazily. The byte accounting, metrics, and manifest logic
must stay importable (and testable) without a TensorFlow install, so heavyweight
symbols are exposed through :func:`__getattr__` rather than at import time.
"""

from __future__ import annotations

from typing import Any

from .arm_specs import ARMS, BUDGET_BYTES, POSTPOOL_BYTES, RAW_BYTES
from .parameter_scope import EXPECTED_SCOPE, assert_trainable_scope, expected_scope

__all__ = [
    "ARMS",
    "BUDGET_BYTES",
    "EXPECTED_SCOPE",
    "POSTPOOL_BYTES",
    "RAW_BYTES",
    "LoRAAttention",
    "ResidualAdapter",
    "assert_trainable_scope",
    "build_above_pool_arm",
    "build_lora_backbone",
    "expected_scope",
]

_LAZY = {
    "LoRAAttention": ".lora",
    "ResidualAdapter": ".lora",
    "build_above_pool_arm": ".lora",
    "build_lora_backbone": ".lora",
}


def __getattr__(name: str) -> Any:
    """Import TensorFlow-dependent symbols only when actually requested."""
    if name in _LAZY:
        from importlib import import_module

        module = import_module(_LAZY[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
