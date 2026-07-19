"""Split-first preprocessing (experiment E17).

Why this exists
---------------
The released pipeline filters each record AS A WHOLE and only then splits it
chronologically. Both median kernels and the zero-phase low-pass are
non-causal, so an adaptation-side sample within the filters' backward reach of
the split depends on raw samples belonging to the TEST segment. Test *labels*
are never involved and no test beat enters adaptation, but the test *signal* is
not strictly unseen.

This pipeline removes the dependency outright instead of arguing about its size:

1. locate the chronological boundary in SAMPLES (R peak of the first test beat,
   minus the window's left half);
2. cut the RAW signal there;
3. discard ``guard`` samples on each side of the cut;
4. filter the two pieces INDEPENDENTLY;
5. drop any beat whose window would straddle the cut;
6. keep a uniform 497-beat adaptation block (rather than 500) so the protocol
   stays balanced across records.

The default 720-sample (2 s) guard is roughly a sixfold margin on the measured
120-sample influence reach (see :mod:`preprocessing.influence_analysis`).

Result
------
Every conclusion held. Arm means moved by at most 0.007 macro-F1
(A0 -0.000, A1 +0.007, A4 -0.006, A5 +0.000), both encoder-versus-frozen
contrasts stayed significant at p = 0.0012, and both encoder-versus-A1
contrasts stayed non-significant. The shifts have MIXED SIGN, which is not the
systematic drop that recovered leakage would produce.

This is a SENSITIVITY analysis, not a replacement primary analysis: the paper's
headline numbers remain those of the released pipeline so they correspond to
the pre-specified plan.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .filters import denoise

__all__ = [
    "DEFAULT_GUARD_SAMPLES",
    "SPLIT_FIRST_ADAPTATION_BEATS",
    "SplitFirstSegments",
    "split_first_denoise",
]

DEFAULT_GUARD_SAMPLES = 720          # 2 s at 360 Hz
SPLIT_FIRST_ADAPTATION_BEATS = 497


@dataclass(frozen=True)
class SplitFirstSegments:
    """Independently filtered adaptation and test segments.

    Attributes:
        adaptation: Filtered samples ``[0, boundary - guard)``.
        test: Filtered samples ``[boundary + guard, end)``.
        adaptation_end: Exclusive raw index where the adaptation piece stops.
        test_start: Raw index where the test piece begins; also the offset that
            maps a raw sample index into ``test``.
        boundary: The raw chronological boundary the cut was taken around.
        guard_samples: Samples discarded on each side.
    """

    adaptation: np.ndarray
    test: np.ndarray
    adaptation_end: int
    test_start: int
    boundary: int
    guard_samples: int


def split_first_denoise(
    signal: np.ndarray,
    boundary: int,
    fs: int = 360,
    guard_samples: int = DEFAULT_GUARD_SAMPLES,
) -> SplitFirstSegments:
    """Cut the raw signal at ``boundary`` and filter each side independently.

    The two calls to :func:`~preprocessing.filters.denoise` never see each
    other's samples, so no adaptation-side output can depend on a test-side
    input. That is the property :func:`tests.test_preprocessing_boundary`
    verifies by perturbing future test samples and asserting the adaptation
    features are bit-identical.

    Args:
        signal: Raw single-lead signal.
        boundary: Chronological split point, in samples.
        fs: Sampling rate.
        guard_samples: Samples discarded either side of the cut.

    Returns:
        A :class:`SplitFirstSegments`.

    Raises:
        ValueError: if the guard is negative, or the boundary and guard leave
            one of the two segments empty.
    """
    signal = np.asarray(signal)
    if guard_samples < 0:
        raise ValueError(f"guard_samples must be non-negative, got {guard_samples}")

    adaptation_end = int(boundary - guard_samples)
    test_start = int(boundary + guard_samples)
    if adaptation_end <= 0:
        raise ValueError(
            f"boundary {boundary} with guard {guard_samples} leaves no adaptation signal"
        )
    if test_start >= len(signal):
        raise ValueError(
            f"boundary {boundary} with guard {guard_samples} leaves no test signal "
            f"(signal length {len(signal)})"
        )

    return SplitFirstSegments(
        adaptation=denoise(signal[:adaptation_end], fs),
        test=denoise(signal[test_start:], fs),
        adaptation_end=adaptation_end,
        test_start=test_start,
        boundary=int(boundary),
        guard_samples=int(guard_samples),
    )
