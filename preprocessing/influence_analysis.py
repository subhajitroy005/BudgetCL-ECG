"""Measure how far a filter's influence reaches backwards across a boundary.

Rather than assume the non-causal front end is "probably fine near the split",
this module measures it: truncate the signal at a boundary, refilter, and see
how far back the output actually changes.

Measured on the released pipeline at 360 Hz (relative to signal std):

    tolerance   backward reach
    1e-3         84 samples (233 ms)
    1e-4        113 samples (314 ms)
    1e-5        120 samples (333 ms)

That is shorter than one RR interval, so with a 198-sample beat window only the
last one or two adaptation beats of a record can be affected at all. The E17
split-first rerun uses a 720-sample guard -- about a sixfold margin.
"""

from __future__ import annotations

import numpy as np

from .filters import denoise

__all__ = ["measure_backward_influence"]


def measure_backward_influence(
    signal: np.ndarray,
    boundary: int,
    fs: int = 360,
    tolerances: tuple[float, ...] = (1e-3, 1e-4, 1e-5),
) -> dict[float, int]:
    """How many samples before ``boundary`` change when the tail is removed.

    Args:
        signal: Raw signal.
        boundary: Index to truncate at.
        fs: Sampling rate, used only for the millisecond conversion in logs.
        tolerances: Relative thresholds (multiples of the signal std).

    Returns:
        ``{tolerance: reach_in_samples}``.

    Raises:
        ValueError: if the boundary leaves no signal on one side.
    """
    signal = np.asarray(signal, dtype=float)
    if not 0 < boundary < len(signal):
        raise ValueError(f"boundary {boundary} must lie inside the signal (len {len(signal)})")

    full = denoise(signal, fs)[:boundary]
    truncated = denoise(signal[:boundary], fs)
    diff = np.abs(full - truncated)
    sd = float(np.std(signal)) or 1.0

    reach: dict[float, int] = {}
    for tol in tolerances:
        exceeded = np.where(diff > tol * sd)[0]
        reach[tol] = int(boundary - 1 - exceeded.min()) if len(exceeded) else 0
    return reach
