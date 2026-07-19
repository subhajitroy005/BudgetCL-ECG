"""Patient bootstrap.

Naming matters here, so it is stated exactly. The procedure is:

    1. average the five seeds WITHIN each patient  -> one score per patient
    2. resample the 21 patients WITH replacement
    3. recompute the arm mean or the paired difference
    4. repeat 10,000 times
    5. report the 2.5th and 97.5th percentiles

Seeds are collapsed in step 1 and are NEVER resampled. This is therefore a
**patient bootstrap**, and we do not call it hierarchical. A hierarchical
patient-and-seed bootstrap would resample seeds *inside* each resampled
patient; that is a different procedure with different coverage, and conflating
the two misstates where the uncertainty comes from.

Intervals are percentile intervals (not BCa). The RNG seed is fixed so an
interval is reproducible to the digit.
"""

from __future__ import annotations

import numpy as np

__all__ = ["BOOTSTRAP_ITERATIONS", "BOOTSTRAP_SEED", "patient_bootstrap_ci", "paired_bootstrap_ci"]

BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260719


def patient_bootstrap_ci(
    patient_scores: np.ndarray,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI for an arm mean, resampling patients.

    Args:
        patient_scores: One score per patient (seeds already averaged).
        iterations: Bootstrap resamples.
        seed: RNG seed; fixed by default for reproducibility.
        alpha: Two-sided level (0.05 -> 95% interval).

    Raises:
        ValueError: if no patient scores are supplied.
    """
    x = np.asarray(patient_scores, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        raise ValueError("patient_bootstrap_ci requires at least one patient score")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(iterations, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(
        np.percentile(means, 100 * (1 - alpha / 2))
    )


def paired_bootstrap_ci(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI for the MEAN PAIRED DIFFERENCE ``a - b``.

    Patients are resampled as units, so the pairing is preserved.

    Note this interval estimates the mean paired difference, whereas the
    Wilcoxon signed-rank test evaluates signed RANKS. The two target different
    estimands, so an interval spanning zero alongside a significant signed-rank
    p-value is not a contradiction -- see ``docs/statistics.md``.

    Raises:
        ValueError: if the two arrays differ in length or are empty.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must have equal length, got {a.shape} and {b.shape}")
    d = a - b
    d = d[~np.isnan(d)]
    if len(d) == 0:
        raise ValueError("paired_bootstrap_ci requires at least one paired difference")
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(iterations, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(
        np.percentile(means, 100 * (1 - alpha / 2))
    )
