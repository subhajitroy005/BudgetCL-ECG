"""Shared pytest fixtures.

Every test here runs WITHOUT TensorFlow and without the PhysioNet recordings,
so correctness of the byte accounting, the metric, the cohort definition, and
the preprocessing boundary can be checked in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from budget_cl.replay import PostPoolReplayRecord, RawReplayRecord  # noqa: E402


@pytest.fixture
def example_raw_record() -> RawReplayRecord:
    """One raw replay exemplar with deterministic contents."""
    rng = np.random.default_rng(0)
    return RawReplayRecord(
        ecg=rng.integers(-127, 128, size=198).astype(np.int8),
        pre_rr=12, post_rr=-8, label=1, valid=1, class_id=1,
    )


@pytest.fixture
def example_postpool_record() -> PostPoolReplayRecord:
    """One post-pool replay exemplar with deterministic contents."""
    rng = np.random.default_rng(1)
    return PostPoolReplayRecord(
        pooled=rng.integers(-127, 128, size=16).astype(np.int8),
        pre_rr=3, post_rr=5, label=2, valid=1, class_id=2,
    )


@pytest.fixture
def synthetic_ecg() -> np.ndarray:
    """A synthetic 360 Hz signal long enough to split with a 2 s guard."""
    rng = np.random.default_rng(20260719)
    t = np.arange(20_000) / 360.0
    beats = np.sin(2 * np.pi * 1.2 * t) * 0.6           # ~72 bpm carrier
    noise = rng.standard_normal(t.size) * 0.05
    return (beats + noise).astype(np.float64)
