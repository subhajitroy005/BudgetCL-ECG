"""Analytical persistent-state memory accounting.

Public entry points::

    from budget_cl.memory import calculate_arm_memory, maximum_replay_items

    report = calculate_arm_memory("A4", budget_bytes=16384)
    assert report.fits
    print(report.replay_items, report.used_bytes)   # 62 16208

The numbers this package produces are the ones asserted in the manuscript;
``tests/test_memory_accounting.py`` pins them so a refactor cannot silently
move a published byte total.
"""

from __future__ import annotations

from .accounting import account_arm, account_state, params_for_arm
from .alignment import align_to, alignment_report
from .budget_solver import effective_budget, maximum_replay_items, per_parameter_state_bytes
from .structures import MemoryCategory, MemoryReport
from .validation import validate_report

__all__ = [
    "MemoryCategory",
    "MemoryReport",
    "account_arm",
    "account_state",
    "align_to",
    "alignment_report",
    "calculate_arm_memory",
    "effective_budget",
    "maximum_replay_items",
    "params_for_arm",
    "per_parameter_state_bytes",
    "validate_report",
]


def calculate_arm_memory(
    arm: str,
    budget_bytes: int = 16_384,
    reserve_bytes: int = 0,
    optimizer: str = "adam",
    alignment: int = 1,
    requested_items: int | None = None,
) -> MemoryReport:
    """Full byte account for one arm as a typed :class:`MemoryReport`.

    The reserve is applied by lowering the ceiling the replay solver sees, which
    is how the paper's 1 KiB-reserve replication is generated: the arms are
    re-solved against 15,360 B rather than re-labelled at 16,384 B.

    Args:
        arm: Arm identifier (``"A0"`` .. ``"A5"``).
        budget_bytes: Nominal ceiling before any reserve.
        reserve_bytes: Firmware reserve withheld from the budget.
        optimizer: Optimizer whose state is charged per parameter.
        alignment: Record alignment; 1 (packed) is what the paper reports.
        requested_items: Force an explicit replay count instead of solving for
            the maximum. Used by the fixed-count causal controls (B5, B6).

    Returns:
        A :class:`MemoryReport`. Check ``.fits`` rather than assuming success --
        some control arms are deliberately infeasible at 16 KiB.
    """
    effective = effective_budget(budget_bytes, reserve_bytes)
    row = account_arm(
        arm,
        budget_bytes=effective,
        optimizer=optimizer,
        alignment=alignment,
        requested_items=requested_items,
    )
    return MemoryReport(
        arm=arm,
        trainable_parameters=int(row["trainable_params"]),
        trainable_weight_bytes=int(row["trainable_weight_bytes"]),
        gradient_bytes=int(row["gradient_bytes"]),
        optimizer_bytes=int(row["optimizer_bytes"]),
        replay_items=int(row["prototypes"]),
        replay_payload_bytes=int(row["replay_value_bytes"]),
        replay_serialized_bytes=int(row["replay_buffer_bytes"]),
        buffer_metadata_bytes=int(row["fixed_metadata_bytes"]),
        padding_bytes=int(row["padding_bytes"]),
        used_bytes=int(row["total_bytes"]),
        budget_bytes=int(budget_bytes),
        reserve_bytes=int(reserve_bytes),
        effective_budget_bytes=int(effective),
        categories=(
            MemoryCategory("trainable weights", int(row["trainable_weight_bytes"]), True, True),
            # Gradients are update-time working state, but a device must still
            # reserve the region -- so persistent=False, in_budget=True.
            MemoryCategory("gradients", int(row["gradient_bytes"]), False, True),
            MemoryCategory("optimizer moments", int(row["optimizer_bytes"]), True, True),
            MemoryCategory("replay records", int(row["replay_buffer_bytes"]), True, True),
            MemoryCategory("buffer metadata", int(row["fixed_metadata_bytes"]), True, True),
        ),
    )
