"""Per-patient help/harm accounting.

Adaptation is NOT uniformly beneficial, and cohort means hide that. The paper
therefore reports harm at two thresholds, because the choice changes the story
materially: counting any decline as harm makes the encoder arms look safer,
but at a meaningful threshold maximum head-only replay harms nobody while the
encoder arms each harm two patients.

The meaningful-change threshold ``epsilon = 0.02`` was fixed BEFORE analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["MEANINGFUL_CHANGE_EPSILON", "HarmProfile", "harm_profile"]

MEANINGFUL_CHANGE_EPSILON = 0.02


@dataclass(frozen=True)
class HarmProfile:
    """Threshold-free and thresholded help/harm counts for one arm."""

    n_patients: int
    improved: int
    worsened: int
    improved_meaningful: int
    harmed_meaningful: int
    harm_rate: float
    worst_change: float
    fifth_percentile: float
    max_gain: float


def harm_profile(gains: np.ndarray, epsilon: float = MEANINGFUL_CHANGE_EPSILON) -> HarmProfile:
    """Summarise per-patient gains against the frozen model.

    Args:
        gains: One signed gain per patient (seeds already averaged within
            patient), relative to the frozen baseline.
        epsilon: Meaningful-change threshold.

    Returns:
        A :class:`HarmProfile`. ``harm_rate`` is
        ``|{p : gain_p < -epsilon}| / P`` -- the quantity a deployment would
        use to decide whether a per-patient revert policy is needed.
    """
    g = np.asarray(gains, dtype=float)
    g = g[~np.isnan(g)]
    n = len(g)
    if n == 0:
        raise ValueError("harm_profile requires at least one patient gain")
    harmed = int(np.sum(g < -epsilon))
    return HarmProfile(
        n_patients=n,
        improved=int(np.sum(g > 0)),
        worsened=int(np.sum(g < 0)),
        improved_meaningful=int(np.sum(g > epsilon)),
        harmed_meaningful=harmed,
        harm_rate=float(harmed / n),
        worst_change=float(np.min(g)),
        fifth_percentile=float(np.percentile(g, 5)),
        max_gain=float(np.max(g)),
    )
