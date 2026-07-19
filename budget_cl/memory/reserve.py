"""Replay counts under the primary ceiling and the 1 KiB firmware reserve.

An arm consuming 16,383 of 16,384 bytes is not shippable. The paper therefore
re-solves every arm against a 15,360 B effective ceiling and re-runs it at the
reduced replay counts, rather than reporting old results against new byte
figures.

The reserve costs A1 49 exemplars and the encoder arms five each, and it
reshuffles the arm ordering -- which is what identifies rank-1 as the more
robust operating point.
"""

from __future__ import annotations

from . import calculate_arm_memory

__all__ = ["PRIMARY_BUDGET_BYTES", "FIRMWARE_RESERVE_BYTES", "reserve_comparison"]

PRIMARY_BUDGET_BYTES = 16_384
FIRMWARE_RESERVE_BYTES = 1_024


def reserve_comparison(arms: list[str] | None = None) -> list[dict]:
    """Replay counts and byte totals with and without the firmware reserve.

    Returns:
        One row per arm with ``items_primary`` / ``items_reserved`` and the
        corresponding byte totals.
    """
    arms = arms or ["A0", "A1", "A2", "A3", "A4", "A5"]
    rows = []
    for arm in arms:
        primary = calculate_arm_memory(arm, PRIMARY_BUDGET_BYTES)
        reserved = calculate_arm_memory(
            arm, PRIMARY_BUDGET_BYTES, reserve_bytes=FIRMWARE_RESERVE_BYTES
        )
        rows.append(
            {
                "arm": arm,
                "trainable_parameters": primary.trainable_parameters,
                "items_primary": primary.replay_items,
                "items_reserved": reserved.replay_items,
                "items_lost": primary.replay_items - reserved.replay_items,
                "bytes_primary": primary.used_bytes,
                "bytes_reserved": reserved.used_bytes,
                "fits_primary": primary.fits,
                "fits_reserved": reserved.fits,
            }
        )
    return rows
