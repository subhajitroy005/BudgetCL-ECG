"""Chronological adaptation/test splits.

The split is by TIME, never at random. A random split would let beats
neighbouring a test beat -- nearly identical in morphology -- appear in
adaptation, which inflates every arm and destroys the point of the protocol.

Primary protocol (E7):
    first 500 annotated beats  -> adaptation
    1 unlabelled guard beat    -> dropped, prevents centred-window overlap
    all remaining beats        -> test

Split-first protocol (E17):
    497 adaptation beats, because windows straddling the raw-signal cut are
    dropped. See :mod:`preprocessing.split_first_pipeline`.
"""

from __future__ import annotations

import numpy as np

__all__ = ["PRIMARY_ADAPTATION_BEATS", "PRIMARY_GUARD_BEATS", "SPLIT_FIRST_ADAPTATION_BEATS",
           "chronological_split"]

PRIMARY_ADAPTATION_BEATS = 500
PRIMARY_GUARD_BEATS = 1
SPLIT_FIRST_ADAPTATION_BEATS = 497


def chronological_split(
    n_beats: int,
    n_adapt: int = PRIMARY_ADAPTATION_BEATS,
    guard_beats: int = PRIMARY_GUARD_BEATS,
) -> tuple[np.ndarray, np.ndarray]:
    """Split beat indices chronologically into (adaptation, test).

    Args:
        n_beats: Total usable beats in the record, in annotation order.
        n_adapt: Labelled adaptation beats to take from the start.
        guard_beats: Unlabelled beats dropped between the two segments.

    Returns:
        ``(adapt_idx, test_idx)``. ``n_adapt`` is clamped so at least one test
        beat always remains.

    Raises:
        ValueError: on negative inputs.
    """
    if n_beats < 0 or n_adapt < 0 or guard_beats < 0:
        raise ValueError("n_beats, n_adapt and guard_beats must all be non-negative")
    guard_beats = int(guard_beats)
    n_adapt = min(int(n_adapt), max(0, n_beats - guard_beats - 1))
    test_start = min(n_beats, n_adapt + guard_beats)
    return np.arange(n_adapt), np.arange(test_start, n_beats)
