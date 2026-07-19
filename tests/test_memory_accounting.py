"""Byte-accounting tests.

These pin the published byte totals. If a refactor moves any number here, it
has moved a number in the paper, and that must be a deliberate act.
"""

from __future__ import annotations

import pytest

from budget_cl.memory import calculate_arm_memory, maximum_replay_items
from budget_cl.memory.budget_solver import per_parameter_state_bytes
from budget_cl.memory.validation import BudgetExceededError, validate_report

PRIMARY_BUDGET = 16_384
RESERVED_BUDGET = 15_360

# (arm, trainable params, replay items @16 KiB, items @15,360 B) -- from the paper.
PUBLISHED = [
    ("A0", 0, 0, 0),
    ("A1", 95, 705, 656),
    ("A2", 131, 678, 629),
    ("A3", 167, 650, 601),
    ("A4", 223, 62, 57),
    ("A5", 351, 52, 47),
]


@pytest.mark.parametrize(("arm", "params", "items", "items_reserved"), PUBLISHED)
def test_published_arm_accounting(arm, params, items, items_reserved):
    """Each arm reproduces its published parameter and replay counts."""
    primary = calculate_arm_memory(arm, PRIMARY_BUDGET)
    reserved = calculate_arm_memory(arm, PRIMARY_BUDGET, reserve_bytes=1024)
    assert primary.trainable_parameters == params
    assert primary.replay_items == items
    assert reserved.replay_items == items_reserved


@pytest.mark.parametrize(("arm", "_p", "_i", "_r"), PUBLISHED)
def test_arms_fit_primary_budget(arm, _p, _i, _r):
    """No arm may exceed the 16 KiB ceiling."""
    report = calculate_arm_memory(arm, PRIMARY_BUDGET)
    assert report.used_bytes <= PRIMARY_BUDGET
    assert report.fits


@pytest.mark.parametrize(("arm", "_p", "_i", "_r"), PUBLISHED)
def test_arms_fit_reserved_budget(arm, _p, _i, _r):
    """Every arm must also fit once 1 KiB is held back for firmware."""
    report = calculate_arm_memory(arm, PRIMARY_BUDGET, reserve_bytes=1024)
    assert report.used_bytes <= RESERVED_BUDGET
    assert report.fits


def test_a4_fits_primary_budget():
    """Spec example: A4 at 16 KiB."""
    assert calculate_arm_memory("A4").used_bytes <= 16_384


def test_a4_fits_reserved_budget():
    """Spec example: A4 with the 1 KiB reserve."""
    assert calculate_arm_memory("A4", reserve_bytes=1024).used_bytes <= 15_360


def test_per_parameter_state_is_16_bytes_for_fp32_adam():
    """The 16 B/parameter figure the paper quotes: 4 + 4 + 2x4."""
    assert per_parameter_state_bytes("adam", "fp32") == 16


def test_replay_capacity_decreases_with_trainable_parameters():
    """The core trade-off: trainable parameters displace replay exemplars."""
    few = maximum_replay_items(16_384, 95, 21, buffer_metadata_bytes=49)
    many = maximum_replay_items(16_384, 351, 21, buffer_metadata_bytes=49)
    assert few > many


def test_reserve_never_increases_capacity():
    """Holding bytes back cannot buy more replay."""
    for arm, *_ in PUBLISHED:
        primary = calculate_arm_memory(arm, PRIMARY_BUDGET)
        reserved = calculate_arm_memory(arm, PRIMARY_BUDGET, reserve_bytes=1024)
        assert reserved.replay_items <= primary.replay_items


def test_validate_report_raises_on_overflow():
    """An infeasible configuration must fail loudly, not silently."""
    report = calculate_arm_memory("A5", budget_bytes=2_048)
    if not report.fits:
        with pytest.raises(BudgetExceededError):
            validate_report(report)
        assert validate_report(report, strict=False) is False


def test_raw_record_is_ten_times_costlier_than_postpool():
    """203 B vs 21 B is what forces the volume-versus-depth trade-off."""
    a1 = calculate_arm_memory("A1")   # post-pool replay
    a4 = calculate_arm_memory("A4")   # raw replay
    assert a1.replay_items > 10 * a4.replay_items
