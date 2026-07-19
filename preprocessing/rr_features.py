"""RR interval features, and the decision latency they imply.

The model consumes BOTH the preceding and the FOLLOWING RR interval. The
following interval is only known once the next R peak has been detected, so a
classification cannot be emitted at the current R peak.

    The task evaluated here is DELAYED heartbeat classification, one beat
    behind the signal -- not zero-latency classification.

This is a property of the source architecture, not of the adaptation method,
and it applies equally to every arm including the frozen baseline, so it does
not affect any comparison in the paper. It does bound the deployment claim.

RR values are normalised with DS1 statistics ONLY and clipped to +/-2, so no
target-database statistic ever enters normalization.
"""

from __future__ import annotations

import numpy as np

__all__ = ["DS1_RR_MEAN", "DS1_RR_STD", "RR_CLIP", "compute_rr", "normalize_rr"]

DS1_RR_MEAN = 0.7785   # seconds
DS1_RR_STD = 0.4956    # seconds
RR_CLIP = 2.0


def compute_rr(prev_sample: int, sample: int, next_sample: int, fs: int) -> tuple[float, float]:
    """Pre-RR and post-RR in seconds for one beat.

    ``post_rr`` requires ``next_sample``; that dependency is the one-beat
    latency documented above.
    """
    return (sample - prev_sample) / fs, (next_sample - sample) / fs


def normalize_rr(
    rr: np.ndarray,
    mean: float = DS1_RR_MEAN,
    std: float = DS1_RR_STD,
    clip: float = RR_CLIP,
) -> np.ndarray:
    """Standardise RR features with SOURCE-domain statistics and clip.

    Defaults are the DS1 statistics. Passing target statistics here would leak
    target distribution information into the input representation.
    """
    z = (np.asarray(rr, dtype=np.float32) - np.float32(mean)) / np.float32(std)
    return np.clip(z, -clip, clip).astype(np.float32)
