"""Bias-corrected and accelerated (BCa) bootstrap confidence intervals.

The released pipeline reports *percentile* intervals (``statistics.bootstrap``).
The Paper 1 pre-registration (``preregistration/paper1_stat_plan.md``) mandates
**BCa** intervals for the equivalence (TOST) and effect-size analyses, because a
median paired difference and a rank-biserial correlation are skewed statistics
for which a percentile interval is biased. BCa corrects for both median bias
(``z0``) and skew (acceleration ``a``, estimated by jackknife).

This is a separate module so the percentile-interval reproduction of the
released tables is left byte-for-byte untouched.

Deterministic: fixed seed, fixed resample count. Statistic is any function of a
1-D sample vector (here, the per-patient paired-difference vector, n = 21).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import norm

__all__ = ["BCA_ITERATIONS", "BCA_SEED", "bca_ci"]

BCA_ITERATIONS = 10_000
BCA_SEED = 20260719  # matches statistics.bootstrap.BOOTSTRAP_SEED


def bca_ci(
    x: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    alpha: float = 0.05,
    iterations: int = BCA_ITERATIONS,
    seed: int = BCA_SEED,
) -> tuple[float, float]:
    """Two-sided BCa CI for ``statistic`` over the sample ``x``.

    Args:
        x: 1-D sample (e.g. 21 per-patient paired differences).
        statistic: maps a sample vector to a scalar (median, rank-biserial, ...).
        alpha: two-sided level; 0.05 -> 95% CI, 0.10 -> 90% CI.
        iterations: bootstrap resamples (patients resampled with replacement).
        seed: fixed RNG seed for reproducibility.

    Returns:
        (low, high). Falls back to the percentile interval when the acceleration
        or bias term is undefined (degenerate sample), which is stated rather
        than hidden.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        raise ValueError("bca_ci requires at least one observation")
    theta_hat = float(statistic(x))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(iterations, n))
    boot = np.array([statistic(x[i]) for i in idx], dtype=float)

    lo_p, hi_p = 100 * alpha / 2, 100 * (1 - alpha / 2)

    # Bias correction z0 from the fraction of boot stats below the point estimate.
    prop = float(np.mean(boot < theta_hat))
    if prop <= 0.0 or prop >= 1.0:
        # z0 undefined -> percentile interval.
        return float(np.percentile(boot, lo_p)), float(np.percentile(boot, hi_p))
    z0 = norm.ppf(prop)

    # Acceleration by jackknife (leave-one-out).
    jack = np.array([statistic(np.delete(x, i)) for i in range(n)], dtype=float)
    jbar = jack.mean()
    diff = jbar - jack
    denom = 6.0 * (np.sum(diff ** 2) ** 1.5)
    a = 0.0 if denom == 0 else float(np.sum(diff ** 3) / denom)

    def adjust(z_alpha: float) -> float:
        return norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))

    a1 = adjust(norm.ppf(alpha / 2))
    a2 = adjust(norm.ppf(1 - alpha / 2))
    return (float(np.percentile(boot, 100 * a1)),
            float(np.percentile(boot, 100 * a2)))
