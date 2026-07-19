"""Assert that each arm trains exactly what it claims to train.

A silent scope error -- an arm that accidentally unfreezes the encoder, or a
"frozen" baseline that quietly updates its head -- would invalidate every
comparison while still producing plausible numbers. These assertions run before
adaptation so that failure is loud and immediate.

Expected trainable scope per arm:

    A0        nothing (frozen reference)
    A1        classifier head only
    A2 / A3   post-pool residual adapter (rank 1 / 2) + head
    A4 / A5   encoder-attention LoRA (rank 1 / 2) + head
    R1 / R2   rank-1 encoder LoRA + head, plus a B-factor anchor
"""

from __future__ import annotations

from .arm_specs import ARMS

__all__ = ["EXPECTED_SCOPE", "expected_scope", "assert_trainable_scope"]

EXPECTED_SCOPE: dict[str, str] = {
    "A0": "frozen",
    "A1": "postpool",
    "A2": "postpool",
    "A3": "postpool",
    "A4": "encoder_lora",
    "A5": "encoder_lora",
    "R1": "encoder_lora",
    "R2": "encoder_lora",
}


def expected_scope(arm: str) -> str:
    """Declared trainable scope for an arm.

    Raises:
        KeyError: on an unknown arm.
    """
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}; known arms: {sorted(ARMS)}")
    return str(ARMS[arm]["trainable_scope"])


def assert_trainable_scope(arm: str, trainable_parameter_count: int) -> None:
    """Check a built model's trainable count against its arm.

    Only the frozen case can be verified from the count alone -- A0 must have
    exactly zero trainable parameters, and any non-zero count means the freeze
    did not take. Other arms are verified structurally when built.

    Raises:
        AssertionError: if a frozen arm has trainable parameters, or a
            non-frozen arm has none.
    """
    scope = expected_scope(arm)
    if scope == "frozen" and trainable_parameter_count != 0:
        raise AssertionError(
            f"arm {arm} is declared frozen but has {trainable_parameter_count} "
            "trainable parameters; the freeze did not take"
        )
    if scope != "frozen" and trainable_parameter_count == 0:
        raise AssertionError(
            f"arm {arm} is declared trainable ({scope}) but has no trainable parameters"
        )
