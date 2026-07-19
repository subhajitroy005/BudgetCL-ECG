"""Exception hierarchy.

Distinct types so a runner can tell a budget violation (the constraint the
paper is about) from a data-integrity failure (a reason to stop entirely).
"""

from __future__ import annotations

__all__ = [
    "BudgetCLError",
    "ConfigurationError",
    "BudgetExceededError",
    "DataIntegrityError",
    "SubjectLeakageError",
    "CheckpointError",
]


class BudgetCLError(Exception):
    """Base class for every error raised by this package."""


class ConfigurationError(BudgetCLError):
    """A configuration is missing, malformed, or internally inconsistent."""


class BudgetExceededError(BudgetCLError):
    """A configuration does not fit its persistent-state budget."""


class DataIntegrityError(BudgetCLError):
    """Dataset, manifest, or split validation failed."""


class SubjectLeakageError(DataIntegrityError):
    """Source and target cohorts share a subject.

    Raised for the 201/202 case and any future cross-split subject overlap.
    Always fatal: results computed under leakage are not interpretable.
    """


class CheckpointError(BudgetCLError):
    """A checkpoint is missing or fails its SHA-256 / parameter-count check."""
