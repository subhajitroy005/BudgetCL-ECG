"""Two one-sided tests (TOST) for the budget-sweep equivalence claim.

Used for E10, where the question is whether raising the adaptation-state budget
from 8 to 64 KiB CHANGES anything. A non-significant difference is not evidence
of equivalence, so equivalence is tested directly.

Margin
------
``delta = 0.05`` macro-F1, fixed before the analysis. It is roughly one third
of the A4-A0 effect (+0.136), so a change smaller than delta is a small
fraction of the effect the paper is about.

Multiplicity
------------
The 18 decisions are **not** multiplicity-adjusted, and the paper says so. The
bias runs toward OVER-claiming equivalence, so this is reported rather than
quietly corrected. ``multiplicity_adjusted`` is carried in the output so the
policy travels with the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import t as student_t

__all__ = ["EQUIVALENCE_MARGIN", "TostResult", "tost_paired"]

EQUIVALENCE_MARGIN = 0.05


@dataclass(frozen=True)
class TostResult:
    """Outcome of one paired equivalence test."""

    contrast: str
    n: int
    mean_difference: float
    delta: float
    lower_one_sided_p: float
    upper_one_sided_p: float
    max_tost_p: float
    ci90_low: float
    ci90_high: float
    equivalent: bool
    multiplicity_adjusted: bool = False


def tost_paired(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    delta: float = EQUIVALENCE_MARGIN,
    contrast: str = "",
    alpha: float = 0.05,
) -> TostResult:
    """Paired TOST for equivalence within +/- ``delta``.

    Equivalence is declared when BOTH one-sided tests reject, which is the same
    as the 90% CI lying entirely inside ``(-delta, +delta)``.

    Raises:
        ValueError: if the arrays mismatch, are empty, or ``delta <= 0``.
    """
    if delta <= 0:
        raise ValueError(f"delta must be positive, got {delta}")
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must have equal length, got {a.shape} and {b.shape}")
    d = (a - b)[~np.isnan(a - b)]
    n = len(d)
    if n < 2:
        raise ValueError("TOST requires at least two paired differences")

    mean = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(n))
    df = n - 1
    if se == 0:
        # Identical arms: equivalent iff the (zero) difference is inside the margin.
        equivalent = abs(mean) < delta
        return TostResult(contrast, n, mean, delta, 0.0, 0.0, 0.0, mean, mean, equivalent)

    p_lower = float(student_t.sf((mean + delta) / se, df))   # H0: diff <= -delta
    p_upper = float(student_t.cdf((mean - delta) / se, df))  # H0: diff >= +delta
    max_p = max(p_lower, p_upper)

    crit = float(student_t.ppf(1 - alpha, df))               # 90% CI for alpha=0.05
    return TostResult(
        contrast=contrast,
        n=n,
        mean_difference=mean,
        delta=delta,
        lower_one_sided_p=p_lower,
        upper_one_sided_p=p_upper,
        max_tost_p=max_p,
        ci90_low=mean - crit * se,
        ci90_high=mean + crit * se,
        equivalent=bool(max_p < alpha),
        multiplicity_adjusted=False,
    )
