"""Feasibility assertions for byte accounts.

Every experiment calls :func:`validate_report` before training so an
over-budget configuration fails immediately instead of producing results that
silently violate the constraint the paper is about.
"""

from __future__ import annotations

from .structures import MemoryReport

__all__ = ["BudgetExceededError", "validate_report"]


class BudgetExceededError(ValueError):
    """Raised when a configuration does not fit its effective budget."""


def validate_report(report: MemoryReport, *, strict: bool = True) -> bool:
    """Check a report against its ceiling.

    Args:
        report: The account to check.
        strict: Raise on overflow instead of returning False. Set False for the
            deliberately infeasible causal controls (e.g. B5 at 16 KiB), where
            infeasibility is the finding rather than an error.

    Returns:
        True when the configuration fits.

    Raises:
        BudgetExceededError: if it does not fit and ``strict`` is True.
    """
    if report.fits:
        return True
    if strict:
        raise BudgetExceededError(
            f"arm {report.arm} needs {report.used_bytes} B but the effective "
            f"budget is {report.effective_budget_bytes} B "
            f"(nominal {report.budget_bytes} B minus {report.reserve_bytes} B reserve); "
            f"over by {-report.remaining_bytes} B"
        )
    return False
