"""Shared typed containers passed between subpackages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["ArmConfig", "PatientAdaptationData", "PatientTestData", "AdaptationResult"]


@dataclass(frozen=True)
class ArmConfig:
    """Everything that defines one experimental arm.

    Attributes:
        arm: Identifier (``"A4"``, ``"B3"``, ``"R1"`` ...).
        trainable_scope: ``"frozen"`` | ``"postpool"`` | ``"encoder_lora"``.
        replay_location: ``"raw"`` | ``"postpool"`` | None.
        rank: LoRA / adapter rank; 0 for head-only, None when frozen.
        budget_bytes: Nominal persistent-state ceiling.
        reserve_bytes: Firmware reserve withheld from the budget.
        anchor_l2: L2 coefficient on the LoRA B factor (arms R1/R2 only).
            Note this anchors B, not the functional update BA.
    """

    arm: str
    trainable_scope: str
    replay_location: str | None
    rank: int | None = None
    budget_bytes: int = 16_384
    reserve_bytes: int = 0
    optimizer: str = "adam"
    anchor_l2: float = 0.0
    requested_items: int | None = None


@dataclass
class PatientAdaptationData:
    """The labelled early segment for one patient."""

    record: str
    ecg: np.ndarray
    rr: np.ndarray
    labels: np.ndarray


@dataclass
class PatientTestData:
    """The held-out later segment for one patient."""

    record: str
    ecg: np.ndarray
    rr: np.ndarray
    labels: np.ndarray


@dataclass
class AdaptationResult:
    """Outcome of adapting one arm to one patient under one seed."""

    arm: str
    record: str
    seed: int
    confusion_matrix: np.ndarray
    macro_f1_present: float
    source_change: float
    used_bytes: int
    replay_items: int
    epochs_run: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
