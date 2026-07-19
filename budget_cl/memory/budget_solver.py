"""Derive replay capacity from a byte budget.

This is the constraint that makes the arms comparable. Every configuration in
the paper is generated from :func:`maximum_replay_items` so that the byte total
is bounded *by construction* rather than checked after the fact:

    N_max = floor((budget - reserve - S*P - metadata) / record_bytes)

where ``S`` is the per-parameter state cost (weight + gradient + optimizer
moments; 16 B/parameter for FP32 Adam) and ``P`` the trainable parameter count.

The consequence is the trade-off the paper studies: spending bytes on trainable
parameters removes bytes that could have held replay exemplars, and the
exchange rate differs by two orders of magnitude between the raw (203 B) and
post-pool (21 B) record formats.
"""

from __future__ import annotations

from .accounting import OPTIMIZER_SLOTS, PRECISION_BYTES

__all__ = ["per_parameter_state_bytes", "maximum_replay_items", "effective_budget"]


def per_parameter_state_bytes(optimizer: str = "adam", weight_precision: str = "fp32") -> int:
    """Bytes of persistent state each trainable parameter costs.

    FP32 Adam gives the 16 B/parameter figure used throughout the paper:
    4 (weight) + 4 (gradient) + 2 x 4 (Adam moments).

    Raises:
        ValueError: if the optimizer or precision is unknown.
    """
    if optimizer not in OPTIMIZER_SLOTS:
        raise ValueError(f"unknown optimizer: {optimizer!r}")
    if weight_precision not in PRECISION_BYTES:
        raise ValueError(f"unknown weight precision: {weight_precision!r}")
    width = PRECISION_BYTES[weight_precision]
    return int(width * (2 + OPTIMIZER_SLOTS[optimizer]))  # weight + gradient + moments


def effective_budget(budget_bytes: int, reserve_bytes: int = 0) -> int:
    """Budget actually available after holding back a firmware reserve.

    An arm consuming 16,383 of 16,384 bytes is not shippable, so the paper also
    reports a replication against a 1 KiB reserve (15,360 B effective ceiling).

    Raises:
        ValueError: if the reserve is negative or exceeds the budget.
    """
    if reserve_bytes < 0:
        raise ValueError(f"reserve_bytes must be non-negative, got {reserve_bytes}")
    if reserve_bytes >= budget_bytes:
        raise ValueError(
            f"reserve_bytes ({reserve_bytes}) must be smaller than "
            f"budget_bytes ({budget_bytes})"
        )
    return int(budget_bytes - reserve_bytes)


def maximum_replay_items(
    budget_bytes: int,
    trainable_parameters: int,
    bytes_per_record: int,
    buffer_metadata_bytes: int = 0,
    reserve_bytes: int = 0,
    optimizer: str = "adam",
    weight_precision: str = "fp32",
) -> int:
    """Largest replay count that fits the budget, or 0 if none does.

    Args:
        budget_bytes: Nominal ceiling (e.g. 16384).
        trainable_parameters: Count of parameters receiving gradients.
        bytes_per_record: SERIALIZED record size (203 raw, 21 post-pool) --
            not the value payload (200 / 18), which excludes label and flags.
        buffer_metadata_bytes: Fixed per-buffer overhead (54 raw, 49 post-pool).
        reserve_bytes: Firmware reserve held back from the budget.

    Returns:
        Replay item count, clamped at 0 when the trainable state alone already
        exhausts the budget.

    Raises:
        ValueError: if ``bytes_per_record`` is not positive.
    """
    if bytes_per_record <= 0:
        raise ValueError(f"bytes_per_record must be positive, got {bytes_per_record}")

    available = effective_budget(budget_bytes, reserve_bytes)
    state = per_parameter_state_bytes(optimizer, weight_precision) * int(trainable_parameters)
    remaining = available - state - int(buffer_metadata_bytes)
    if remaining <= 0:
        return 0
    return int(remaining // bytes_per_record)
