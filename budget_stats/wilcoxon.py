"""Paired Wilcoxon signed-rank tests.

Conventions, stated because they change the answer:

* two-sided;
* zero differences handled by ``zero_method="wilcox"`` (dropped), the SciPy
  default and the convention the effect size matches;
* exact p-values for small samples where SciPy provides them, normal
  approximation otherwise -- SciPy chooses automatically at n = 21;
* the test evaluates signed RANKS, not the mean. It can therefore disagree
  with a bootstrap interval on the mean without either being wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import wilcoxon

from .effect_sizes import rank_biserial

__all__ = ["PairedTestResult", "paired_test"]


@dataclass(frozen=True)
class PairedTestResult:
    """Outcome of one paired comparison."""

    n: int
    mean_difference: float
    median_difference: float
    p_value: float
    effect_size: float
    improved: int
    worsened: int


def paired_test(scores_a: np.ndarray, scores_b: np.ndarray) -> PairedTestResult:
    """Paired Wilcoxon signed-rank test of ``a - b``.

    Args:
        scores_a: One score per patient for the first arm.
        scores_b: Matching scores for the second arm; must be aligned by patient.

    Returns:
        A :class:`PairedTestResult`. ``p_value`` is ``nan`` when every paired
        difference is zero, which SciPy cannot test.

    Raises:
        ValueError: if the arrays differ in length or are empty.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must have equal length, got {a.shape} and {b.shape}")
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) == 0:
        raise ValueError("paired_test requires at least one complete pair")

    d = a - b
    try:
        p = float(wilcoxon(d, zero_method="wilcox").pvalue)
    except ValueError:
        # SciPy raises when every difference is zero.
        p = float("nan")

    return PairedTestResult(
        n=int(len(d)),
        mean_difference=float(np.mean(d)),
        median_difference=float(np.median(d)),
        p_value=p,
        effect_size=rank_biserial(d),
        improved=int(np.sum(d > 0)),
        worsened=int(np.sum(d < 0)),
    )
