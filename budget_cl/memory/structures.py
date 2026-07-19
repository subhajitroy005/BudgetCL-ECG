"""Typed containers for the byte-accounting report.

Keeping these as frozen dataclasses rather than bare dicts means a report can
be serialised, diffed against a previous release, and asserted in tests without
depending on dictionary key spelling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

__all__ = ["MemoryCategory", "MemoryReport"]


@dataclass(frozen=True)
class MemoryCategory:
    """One line of the byte account.

    Attributes:
        name: Human-readable category (e.g. ``"optimizer moments"``).
        bytes: Size in bytes.
        persistent: True when the quantity must survive between update steps.
            Gradients are the interesting case: they are ``persistent=False``
            but ``included_in_budget=True``, because a device must still
            reserve the region even though the values are transient.
        included_in_budget: Whether the category counts against the ceiling.
    """

    name: str
    bytes: int
    persistent: bool
    included_in_budget: bool


@dataclass(frozen=True)
class MemoryReport:
    """Complete byte account for one arm at one budget."""

    arm: str
    trainable_parameters: int
    trainable_weight_bytes: int
    gradient_bytes: int
    optimizer_bytes: int
    replay_items: int
    replay_payload_bytes: int
    replay_serialized_bytes: int
    buffer_metadata_bytes: int
    padding_bytes: int
    used_bytes: int
    budget_bytes: int
    reserve_bytes: int
    effective_budget_bytes: int
    categories: tuple[MemoryCategory, ...] = field(default_factory=tuple)

    @property
    def remaining_bytes(self) -> int:
        """Headroom left under the effective ceiling; negative means overflow."""
        return self.effective_budget_bytes - self.used_bytes

    @property
    def fits(self) -> bool:
        """Whether this configuration is feasible under its ceiling."""
        return self.used_bytes <= self.effective_budget_bytes

    def to_dict(self) -> dict:
        """Plain dict for JSON/CSV serialisation, including derived fields."""
        d = asdict(self)
        d["remaining_bytes"] = self.remaining_bytes
        d["fits"] = self.fits
        return d
