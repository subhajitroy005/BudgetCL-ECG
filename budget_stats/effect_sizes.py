"""Matched-pairs rank-biserial effect size.

Reported alongside every p-value: with 21 patients, a significant p-value with
a small effect and with a large one mean quite different things.

Convention
----------
The paper computes the effect size from the Wilcoxon test statistic::

    r_rb = 1 - 2W / (n(n+1)/2)

where ``W`` is the SciPy signed-rank statistic and ``n`` the number of pairs.
This is the form behind every published value, and :func:`rank_biserial`
implements it so the repository reproduces the manuscript tables exactly.

One wrinkle is documented rather than silently "fixed": SciPy drops zero
differences when forming ``W``, while ``n`` here counts ALL pairs. When ties
are present that makes the denominator marginally conservative (it shrinks
``|r_rb|`` slightly). Changing it would move published numbers without changing
any conclusion, so the published convention is kept and
:func:`rank_biserial_from_ranks` is offered for the zero-dropping variant.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, wilcoxon

__all__ = ["rank_biserial", "rank_biserial_from_ranks"]


def rank_biserial(differences: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation, paper convention.

    Args:
        differences: Paired differences, one per patient.

    Returns:
        Effect size in [-1, 1]; 0.0 when every difference is zero or the input
        is empty.
    """
    d = np.asarray(differences, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n == 0 or np.all(d == 0):
        return 0.0
    try:
        w = float(wilcoxon(d, zero_method="wilcox").statistic)
    except ValueError:
        return 0.0
    return float(1.0 - (2.0 * w) / (n * (n + 1) / 2.0))


def rank_biserial_from_ranks(differences: np.ndarray) -> float:
    """Zero-dropping variant computed directly from signed rank sums.

    Provided for comparison only; the published tables use
    :func:`rank_biserial`.
    """
    d = np.asarray(differences, dtype=float)
    d = d[~np.isnan(d)]
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = rankdata(np.abs(d))
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())
