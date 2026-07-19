"""Preprocessing boundary tests.

The central test here perturbs FUTURE test samples and asserts that
adaptation-side features do not move. That is the property the E17 split-first
pipeline exists to guarantee, and the property the released whole-record
pipeline demonstrably does NOT have.

Both directions are asserted deliberately: showing that split-first is clean is
only meaningful alongside evidence that the original pipeline is not, since
that is what made the sensitivity rerun necessary.
"""

from __future__ import annotations

import numpy as np
import pytest

from preprocessing import denoise, measure_backward_influence, split_first_denoise
from preprocessing.split_first_pipeline import DEFAULT_GUARD_SAMPLES

BOUNDARY = 10_000


def test_split_first_adaptation_is_immune_to_future_samples(synthetic_ecg):
    """Changing test-segment samples must not alter adaptation features at all.

    This is the strong form of the leakage check: the assertion is bit-identity,
    not "small difference".
    """
    original = synthetic_ecg.copy()
    perturbed = synthetic_ecg.copy()
    # Corrupt everything after the boundary, far beyond any plausible influence.
    perturbed[BOUNDARY:] += 5.0

    a = split_first_denoise(original, BOUNDARY, guard_samples=DEFAULT_GUARD_SAMPLES)
    b = split_first_denoise(perturbed, BOUNDARY, guard_samples=DEFAULT_GUARD_SAMPLES)

    assert np.array_equal(a.adaptation, b.adaptation), (
        "split-first adaptation features changed when only FUTURE test samples "
        "were modified; the split is not isolating the two segments"
    )


def test_whole_record_pipeline_is_not_immune(synthetic_ecg):
    """The released pipeline DOES leak across the boundary.

    Documents the defect the split-first rerun addresses. If this ever starts
    passing, the front end has changed and the E17 rationale needs revisiting.
    """
    original = synthetic_ecg.copy()
    perturbed = synthetic_ecg.copy()
    perturbed[BOUNDARY:] += 5.0

    a = denoise(original, 360)[:BOUNDARY]
    b = denoise(perturbed, 360)[:BOUNDARY]
    assert not np.array_equal(a, b), (
        "expected the non-causal whole-record pipeline to leak across the split"
    )


def test_backward_influence_is_bounded_and_short(synthetic_ecg):
    """The leak exists but reaches under one RR interval.

    Measured reach on real records: 84 samples @1e-3, 120 @1e-5 (333 ms). The
    bound is what makes a 720-sample guard a ~6x margin.
    """
    reach = measure_backward_influence(synthetic_ecg, BOUNDARY, fs=360)
    assert reach[1e-3] <= reach[1e-5], "a looser tolerance cannot reach further"
    # Comfortably under one second at 360 Hz even at the tightest tolerance.
    assert reach[1e-5] < 360, f"backward influence unexpectedly long: {reach[1e-5]} samples"


def test_guard_exceeds_measured_influence(synthetic_ecg):
    """The E17 guard must dominate the measured reach by a wide margin."""
    reach = measure_backward_influence(synthetic_ecg, BOUNDARY, fs=360)
    assert DEFAULT_GUARD_SAMPLES > 4 * reach[1e-5], (
        f"guard {DEFAULT_GUARD_SAMPLES} is not a comfortable margin over a "
        f"measured reach of {reach[1e-5]} samples"
    )


def test_segments_do_not_overlap(synthetic_ecg):
    """The guard region belongs to neither segment."""
    seg = split_first_denoise(synthetic_ecg, BOUNDARY, guard_samples=DEFAULT_GUARD_SAMPLES)
    assert seg.adaptation_end == BOUNDARY - DEFAULT_GUARD_SAMPLES
    assert seg.test_start == BOUNDARY + DEFAULT_GUARD_SAMPLES
    assert seg.test_start - seg.adaptation_end == 2 * DEFAULT_GUARD_SAMPLES


def test_invalid_boundary_is_rejected(synthetic_ecg):
    """A boundary that leaves a segment empty must raise, not truncate."""
    with pytest.raises(ValueError):
        split_first_denoise(synthetic_ecg, 10, guard_samples=DEFAULT_GUARD_SAMPLES)
    with pytest.raises(ValueError):
        split_first_denoise(synthetic_ecg, len(synthetic_ecg) - 10,
                            guard_samples=DEFAULT_GUARD_SAMPLES)
