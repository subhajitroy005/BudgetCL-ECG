"""Holm-Bonferroni correction over the pre-specified comparison family.

The family was fixed in the written analysis plan BEFORE the final runs and has
six members. It is not expanded post hoc -- adding comparisons after seeing
results would invalidate the correction.
"""

from __future__ import annotations

import numpy as np

__all__ = ["PRESPECIFIED_FAMILY", "holm_correct"]

PRESPECIFIED_FAMILY: tuple[str, ...] = (
    "A4_vs_A1",
    "A5_vs_A1",
    "A4_vs_A2",
    "A4_vs_A3",
    "A4_vs_A0",
    "A5_vs_A0",
)


def holm_correct(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values, preserving monotonicity.

    Sorts ascending, scales the i-th smallest by ``(m - i)``, then enforces a
    running maximum so an adjusted value can never fall below an earlier one.

    Returns:
        ``{comparison: adjusted_p}``, each clipped to 1.0.
    """
    if not p_values:
        return {}
    names = list(p_values)
    raw = np.array([p_values[n] for n in names], dtype=float)
    order = np.argsort(raw)
    m = len(raw)

    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        scaled = (m - rank) * raw[idx]
        running = max(running, scaled)      # monotonicity
        adjusted[idx] = min(running, 1.0)
    return {n: float(adjusted[i]) for i, n in enumerate(names)}
