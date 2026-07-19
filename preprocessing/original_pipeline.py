"""Released preprocessing: OFFLINE, NON-CAUSAL, whole-record.

This is the pipeline behind every primary number in the paper. It is inherited
unchanged from the source architecture so that the adaptation results are
attributable to the adaptation method rather than to a different front end.

It is explicitly NOT deployable as-is:

* the median kernels are centred, so each output depends on both sides;
* the low-pass is zero-phase (``filtfilt``), so each output depends on FUTURE
  samples;
* filtering happens before the chronological split, so adaptation-side samples
  near the boundary depend on test-segment raw samples.

The honest description of the protocol is therefore "offline chronological
label partitioning with non-causal whole-record preprocessing", not "strictly
causal chronological adaptation". See :mod:`preprocessing.split_first_pipeline`
for the sensitivity rerun that removes the last point.
"""

from __future__ import annotations

import numpy as np

from .filters import denoise

__all__ = ["CAUSAL", "preprocess_record"]

#: This pipeline is not causal. Kept as a module constant so config validation
#: and the leakage audit can assert it rather than trusting documentation.
CAUSAL = False


def preprocess_record(signal: np.ndarray, fs: int = 360) -> np.ndarray:
    """Filter a whole record before beat extraction."""
    return denoise(np.asarray(signal), fs)
