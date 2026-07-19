"""Analytical persistent-state byte accounting.

This module is the single source of truth for every byte total reported in the
paper. Replay capacity is *derived* from the budget rather than chosen: given a
trainable parameter count and a per-record size, :func:`account_state` returns
the largest replay count that fits, and every experiment asserts the resulting
total against its ceiling.

What "persistent state" means here
----------------------------------
A statically reserved writable region the adaptation procedure needs *across*
update steps. It is **not** total SRAM and does **not** imply non-volatility.
Counted:  trainable weights, gradients, optimizer moments, replay values,
          replay labels, buffer metadata, alignment padding.
Excluded: forward/backward activations, stack and workspace, allocator
          overhead -- these are peak-SRAM quantities and are not measured
          anywhere in this work.

Gradients are charged even though they are update-time working state, because
an implementation that materialises a full gradient vector must reserve that
region. Excluding them would understate what a device has to set aside.

    B_persistent = B_weights + B_gradients + B_optimizer
                 + B_replay + B_labels + B_metadata + B_padding

See ``docs/memory_accounting.md`` for a field-by-field walkthrough.
"""


from __future__ import annotations

from dataclasses import dataclass
from math import floor

PRECISION_BYTES = {
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "fp16": 2,
    "float16": 2,
    "fp32": 4,
    "float32": 4,
}

OPTIMIZER_SLOTS = {
    "adam": 2,
    "sgd_momentum": 1,
    "sgdm": 1,
    "sgd": 0,
}


@dataclass(frozen=True)
class ReplaySpec:
    name: str
    latent_values: int
    ecg_values: int
    rr_values: int
    quantized_tensors: int

    @property
    def values(self) -> int:
        return self.latent_values + self.ecg_values + self.rr_values


REPLAY_SPECS = {
    "postpool": ReplaySpec(
        name="postpool",
        latent_values=16,
        ecg_values=0,
        rr_values=2,
        quantized_tensors=1,
    ),
    "raw": ReplaySpec(
        name="raw",
        latent_values=0,
        ecg_values=198,
        rr_values=2,
        quantized_tensors=2,
    ),
    "none": ReplaySpec(
        name="none",
        latent_values=0,
        ecg_values=0,
        rr_values=0,
        quantized_tensors=0,
    ),
}


def align_to(n: int, alignment: int) -> int:
    """Round ``n`` up to the requested byte alignment."""
    if alignment <= 1:
        return int(n)
    return int(((n + alignment - 1) // alignment) * alignment)


def account_state(
    trainable_params: int,
    replay: str,
    budget_bytes: int = 16_384,
    optimizer: str = "adam",
    weight_precision: str = "fp32",
    replay_precision: str = "int8",
    alignment: int = 1,
    num_classes: int = 5,
    requested_items: int | None = None,
) -> dict:
    """Return a complete persistent-state byte account for one configuration.

    ``requested_items=None`` asks for the maximum number of replay entries that
    fits. Supplying a count instead reports whether that explicit allocation
    fits under the same accounting rule.
    """
    if replay not in REPLAY_SPECS:
        raise ValueError(f"unknown replay representation: {replay}")
    if optimizer not in OPTIMIZER_SLOTS:
        raise ValueError(f"unknown optimizer: {optimizer}")
    if weight_precision not in PRECISION_BYTES:
        raise ValueError(f"unknown weight precision: {weight_precision}")
    if replay_precision not in PRECISION_BYTES:
        raise ValueError(f"unknown replay precision: {replay_precision}")

    spec = REPLAY_SPECS[replay]
    weight_b = PRECISION_BYTES[weight_precision]
    replay_b = PRECISION_BYTES[replay_precision]

    trainable_weight_bytes = int(trainable_params * weight_b)
    gradient_bytes = int(trainable_params * weight_b)
    optimizer_bytes = int(trainable_params * OPTIMIZER_SLOTS[optimizer] * weight_b)
    trainable_state_bytes = trainable_weight_bytes + gradient_bytes + optimizer_bytes

    if replay == "none":
        item_value_bytes = 0
        item_metadata_bytes = 0
        item_unaligned_bytes = 0
        item_stride_bytes = 0
        fixed_unaligned = 0
        fixed_metadata_bytes = 0
        max_items = 0
        count = 0 if requested_items is None else int(requested_items)
    else:
        item_value_bytes = int(spec.values * replay_b)
        # Per-entry persistent fields required by the roadmap.
        label_bytes_per_item = 1
        valid_flag_bytes_per_item = 1
        class_id_bytes_per_item = 1
        item_metadata_bytes = (
            label_bytes_per_item + valid_flag_bytes_per_item + class_id_bytes_per_item
        )
        item_unaligned_bytes = item_value_bytes + item_metadata_bytes
        item_stride_bytes = align_to(item_unaligned_bytes, alignment)

        # Fixed buffer-level persistent fields. Use explicit fixed-width counts
        # rather than Python object sizes; this is serialized-state accounting.
        fixed_unaligned = (
            4  # buffer write index
            + 4  # replay count
            + 16  # reservoir/RNG state: seed plus seen counter
            + num_classes * 4  # class counters / selection metadata
            + spec.quantized_tensors * 4  # FP32 scale(s)
            + spec.quantized_tensors * 1  # INT8 zero point(s)
        )
        fixed_metadata_bytes = align_to(fixed_unaligned, alignment)

        remaining = budget_bytes - trainable_state_bytes - fixed_metadata_bytes
        max_items = max(0, floor(remaining / item_stride_bytes))
        count = max_items if requested_items is None else int(requested_items)

    replay_value_bytes = int(count * item_value_bytes)
    label_bytes = int(count if replay != "none" else 0)
    valid_flag_bytes = int(count if replay != "none" else 0)
    class_id_bytes = int(count if replay != "none" else 0)
    item_padding_bytes = int(count * max(0, item_stride_bytes - item_unaligned_bytes))
    fixed_padding_bytes = int(max(0, fixed_metadata_bytes - fixed_unaligned))
    padding_bytes = item_padding_bytes + fixed_padding_bytes
    metadata_bytes = valid_flag_bytes + class_id_bytes + fixed_metadata_bytes + item_padding_bytes

    replay_buffer_bytes = count * item_stride_bytes + fixed_metadata_bytes
    total_bytes = trainable_state_bytes + replay_buffer_bytes
    fits = total_bytes <= budget_bytes

    return {
        "budget_bytes": int(budget_bytes),
        "optimizer": optimizer,
        "weight_precision": weight_precision,
        "replay_precision": replay_precision,
        "alignment": int(alignment),
        "representation": replay,
        "trainable_params": int(trainable_params),
        "trainable_weight_bytes": trainable_weight_bytes,
        "gradient_bytes": gradient_bytes,
        "optimizer_bytes": optimizer_bytes,
        "trainable_state_bytes": trainable_state_bytes,
        "replay_item_value_bytes": int(item_value_bytes),
        "replay_item_metadata_bytes": int(item_metadata_bytes),
        "replay_item_unaligned_bytes": int(item_unaligned_bytes),
        "replay_item_stride_bytes": int(item_stride_bytes),
        "fixed_metadata_bytes": int(fixed_metadata_bytes),
        "prototypes": int(count),
        "max_prototypes": int(max_items),
        "replay_value_bytes": replay_value_bytes,
        "label_bytes": label_bytes,
        "valid_flag_bytes": valid_flag_bytes,
        "class_id_bytes": class_id_bytes,
        "padding_bytes": padding_bytes,
        "metadata_bytes": metadata_bytes,
        "replay_buffer_bytes": int(replay_buffer_bytes),
        "total_bytes": int(total_bytes),
        "headroom_bytes": int(budget_bytes - total_bytes),
        "fits": bool(fits),
    }


def params_for_arm(arm: str, prototype_dim: int = 18, num_classes: int = 5,
                   embed_dim: int = 16) -> tuple[int, str, int]:
    """Return (trainable_params, replay_representation, lora_or_adapter_rank)."""
    head_params = prototype_dim * num_classes + num_classes
    if arm == "A0":
        return 0, "none", 0
    if arm == "A1":
        return head_params, "postpool", 0
    if arm in {"A2", "A3"}:
        rank = 1 if arm == "A2" else 2
        return head_params + 2 * prototype_dim * rank, "postpool", rank
    if arm in {"A4", "A5"}:
        rank = 1 if arm == "A4" else 2
        return head_params + 8 * embed_dim * rank, "raw", rank
    raise ValueError(f"unknown arm: {arm}")


def account_arm(
    arm: str,
    budget_bytes: int = 16_384,
    optimizer: str = "adam",
    weight_precision: str = "fp32",
    replay_precision: str = "int8",
    alignment: int = 1,
    num_classes: int = 5,
    prototype_dim: int = 18,
    embed_dim: int = 16,
    requested_items: int | None = None,
) -> dict:
    """Complete accounting for one named A0-A5 arm."""
    params, replay, rank = params_for_arm(arm, prototype_dim, num_classes, embed_dim)
    row = account_state(
        params,
        replay,
        budget_bytes=budget_bytes,
        optimizer=optimizer,
        weight_precision=weight_precision,
        replay_precision=replay_precision,
        alignment=alignment,
        num_classes=num_classes,
        requested_items=requested_items,
    )
    row.update({"arm": arm, "rank": int(rank)})
    return row
