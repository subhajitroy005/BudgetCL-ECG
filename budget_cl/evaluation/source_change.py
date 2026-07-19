"""Source-domain change, with ONE sign convention.

    delta_src = source_after - source_before

Positive means the adapted model got BETTER on the source domain; negative
means source performance fell. Every source-domain number in the paper is
negative, so -0.027 is a SMALLER loss than -0.136.

We deliberately avoid the word "forgetting" for this quantity. The two common
conventions in the literature differ in sign, and mixing them is a standard
source of confusion; "source-domain macro-F1 change" also describes what is
measured (a change in a score) rather than asserting a mechanism.
"""

from __future__ import annotations

__all__ = ["source_domain_change"]


def source_domain_change(source_after: float, source_before: float) -> float:
    """Signed change in source-domain score. Negative means source loss."""
    return float(source_after) - float(source_before)
