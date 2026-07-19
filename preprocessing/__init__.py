"""ECG preprocessing pipelines.

Two pipelines, deliberately kept separate:

    original_pipeline     non-causal, whole-record; behind every primary number
    split_first_pipeline  E17 sensitivity; no adaptation-side sample can depend
                          on a test-side raw sample

A fully causal streaming front end is future work and is NOT implemented here.
"""

from __future__ import annotations

from .filters import denoise, lowpass_filter, remove_baseline_wander
from .influence_analysis import measure_backward_influence
from .original_pipeline import preprocess_record
from .rr_features import compute_rr, normalize_rr
from .split_first_pipeline import (
    DEFAULT_GUARD_SAMPLES,
    SplitFirstSegments,
    split_first_denoise,
)

__all__ = [
    "DEFAULT_GUARD_SAMPLES",
    "SplitFirstSegments",
    "compute_rr",
    "denoise",
    "lowpass_filter",
    "measure_backward_influence",
    "normalize_rr",
    "preprocess_record",
    "remove_baseline_wander",
    "split_first_denoise",
]
