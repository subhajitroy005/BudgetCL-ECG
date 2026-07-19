"""Symmetric per-tensor INT8 quantization for replay values.

Replay is stored INT8, which is what makes the byte budget achievable. The
scheme is SYMMETRIC and PER-TENSOR:

    s     = max(|x|) / 127
    q     = clip(round(x / s), -127, 127)
    x_hat = s * q

The zero point is therefore structurally zero and is not stored per record;
only the scale is, in the fixed buffer metadata. The range is clipped to
[-127, 127] rather than [-128, 127] so the representation stays symmetric --
using -128 would make the negative range one step wider than the positive one.

Replay values are dequantized back to FP32 before entering the training graph;
quantization is a STORAGE format here, not a compute format. No claim is made
about integer-arithmetic training.
"""

from __future__ import annotations

import numpy as np

__all__ = ["INT8_QMAX", "compute_scale", "quantize", "dequantize", "roundtrip_error"]

INT8_QMAX = 127


def compute_scale(x: np.ndarray) -> float:
    """Per-tensor symmetric scale.

    Returns 1.0 for an all-zero tensor so quantization stays well defined
    instead of dividing by zero.
    """
    absmax = float(np.max(np.abs(np.asarray(x, dtype=np.float64)))) if np.size(x) else 0.0
    if absmax == 0.0:
        return 1.0
    return absmax / INT8_QMAX


def quantize(x: np.ndarray, scale: float | None = None) -> tuple[np.ndarray, float]:
    """Quantize to INT8.

    Args:
        x: Float tensor.
        scale: Reuse an existing scale (e.g. a calibrated one). When None the
            scale is computed from this tensor.

    Returns:
        ``(q, scale)`` with ``q`` of dtype int8.

    Raises:
        ValueError: if ``scale`` is not positive.
    """
    if scale is None:
        scale = compute_scale(x)
    if scale <= 0:
        raise ValueError(f"quantization scale must be positive, got {scale}")
    q = np.clip(np.round(np.asarray(x, dtype=np.float64) / scale), -INT8_QMAX, INT8_QMAX)
    return q.astype(np.int8), float(scale)


def dequantize(q: np.ndarray, scale: float) -> np.ndarray:
    """Reconstruct FP32 values from INT8 storage."""
    return (np.asarray(q, dtype=np.float32) * np.float32(scale)).astype(np.float32)


def roundtrip_error(x: np.ndarray, scale: float | None = None) -> float:
    """Maximum absolute quantize/dequantize error.

    Bounded by ``scale / 2`` for values inside the representable range; larger
    values indicate clipping.
    """
    q, s = quantize(x, scale)
    return float(np.max(np.abs(np.asarray(x, dtype=np.float32) - dequantize(q, s))))
