"""INT8 replay quantization tests."""

from __future__ import annotations

import numpy as np
import pytest

from budget_cl.quantization import INT8_QMAX, compute_scale, dequantize, quantize, roundtrip_error


def test_symmetric_range_excludes_minus_128():
    """The range is [-127, 127] so positive and negative sides match."""
    q, _ = quantize(np.array([-1000.0, 1000.0]))
    assert q.min() >= -INT8_QMAX
    assert q.max() <= INT8_QMAX


def test_scale_maps_absmax_to_127():
    x = np.array([-2.0, 1.0, 0.5])
    assert compute_scale(x) == pytest.approx(2.0 / 127)


def test_roundtrip_error_bounded_by_half_scale():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000)
    assert roundtrip_error(x) <= compute_scale(x) / 2 + 1e-9


def test_all_zero_tensor_is_safe():
    """An all-zero tensor must not divide by zero."""
    q, s = quantize(np.zeros(10))
    assert s == 1.0
    assert np.all(q == 0)
    assert np.all(dequantize(q, s) == 0)


def test_zero_point_is_structurally_zero():
    """Symmetric quantization maps 0.0 exactly to 0."""
    q, s = quantize(np.array([0.0, 1.0, -1.0]))
    assert q[0] == 0
    assert dequantize(q, s)[0] == 0.0


def test_non_positive_scale_rejected():
    with pytest.raises(ValueError):
        quantize(np.array([1.0]), scale=0.0)
