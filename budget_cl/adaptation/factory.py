"""Map arm identifiers to adaptation implementations.

One lookup point, so an experiment cannot construct an arm in a way that
differs subtly from how another experiment constructs the same arm. That kind
of drift is invisible in results and fatal to a paired comparison.
"""

from __future__ import annotations

from budget_cl.exceptions import ConfigurationError
from budget_cl.models.arm_specs import ARMS
from budget_cl.types import ArmConfig

__all__ = ["arm_config", "trainable_scope_for"]


def trainable_scope_for(arm: str) -> str:
    """Declared trainable scope for an arm.

    Raises:
        ConfigurationError: on an unknown arm.
    """
    if arm not in ARMS:
        raise ConfigurationError(f"unknown arm {arm!r}; known arms: {sorted(ARMS)}")
    return str(ARMS[arm]["trainable_scope"])


def arm_config(
    arm: str,
    budget_bytes: int = 16_384,
    reserve_bytes: int = 0,
    optimizer: str = "adam",
) -> ArmConfig:
    """Build a typed :class:`~budget_cl.types.ArmConfig` from the registry.

    Raises:
        ConfigurationError: on an unknown arm.
    """
    if arm not in ARMS:
        raise ConfigurationError(f"unknown arm {arm!r}; known arms: {sorted(ARMS)}")
    spec = ARMS[arm]
    return ArmConfig(
        arm=arm,
        trainable_scope=str(spec["trainable_scope"]),
        replay_location=spec.get("replay"),
        rank=spec.get("rank"),
        budget_bytes=budget_bytes,
        reserve_bytes=reserve_bytes,
        optimizer=optimizer,
        anchor_l2=float(spec.get("anchor_l2", 0.0)),
        requested_items=spec.get("requested_items"),
    )
