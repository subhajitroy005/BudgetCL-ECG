"""INT8 replay quantization (storage format, not a compute format)."""

from __future__ import annotations

from .int8 import INT8_QMAX, compute_scale, dequantize, quantize, roundtrip_error

__all__ = ["INT8_QMAX", "compute_scale", "dequantize", "quantize", "roundtrip_error"]
