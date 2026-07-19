"""Logical replay record formats and their serialized sizes.

The paper distinguishes two quantities that are easy to conflate, and
conflating them understates a budget:

    value payload    the INT8 tensor alone      raw 200 B, post-pool  18 B
    serialized record  payload + label + flags  raw 203 B, post-pool  21 B

Every byte total in the paper uses the SERIALIZED size. The payload figures
appear only when the text is explicitly naming the tensor.

    b_record = b_payload + b_label + b_flags + b_pad

Both structs are all int8/uint8, so packed size, natural ABI size, and 1-byte
stride coincide. See :mod:`budget_cl.memory.alignment` for wider ABIs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "RAW_ECG_VALUES",
    "RAW_PAYLOAD_BYTES",
    "RAW_RECORD_BYTES",
    "POSTPOOL_VALUES",
    "POSTPOOL_PAYLOAD_BYTES",
    "POSTPOOL_RECORD_BYTES",
    "PER_RECORD_METADATA_BYTES",
    "RawReplayRecord",
    "PostPoolReplayRecord",
]

# --- raw replay: one R-peak-centred beat window plus its two RR intervals ----
RAW_ECG_VALUES = 198
RAW_RR_VALUES = 2
RAW_PAYLOAD_BYTES = RAW_ECG_VALUES + RAW_RR_VALUES          # 200

# --- post-pool replay: the fused 16-D pooled vector plus the two RR values ---
POSTPOOL_VALUES = 16
POSTPOOL_PAYLOAD_BYTES = POSTPOOL_VALUES + RAW_RR_VALUES    # 18

# label (1) + valid flag (1) + class id (1)
PER_RECORD_METADATA_BYTES = 3

RAW_RECORD_BYTES = RAW_PAYLOAD_BYTES + PER_RECORD_METADATA_BYTES        # 203
POSTPOOL_RECORD_BYTES = POSTPOOL_PAYLOAD_BYTES + PER_RECORD_METADATA_BYTES  # 21


@dataclass(frozen=True)
class RawReplayRecord:
    """One raw-beat replay exemplar (203 B serialized).

    Attributes:
        ecg: 198 INT8 samples, R-peak centred (99 left, 99 right).
        pre_rr: Quantised preceding RR interval.
        post_rr: Quantised following RR interval. Note this is what introduces
            the one-beat decision latency discussed in the paper.
        label: AAMI class index used as the training target.
        valid: Occupancy flag for the fixed-capacity buffer slot.
        class_id: Class tag used by the class-balanced selector.
    """

    ecg: np.ndarray
    pre_rr: int
    post_rr: int
    label: int
    valid: int = 1
    class_id: int = 0

    def __post_init__(self) -> None:
        if self.ecg.shape != (RAW_ECG_VALUES,):
            raise ValueError(
                f"raw replay ecg must have shape ({RAW_ECG_VALUES},), got {self.ecg.shape}"
            )


@dataclass(frozen=True)
class PostPoolReplayRecord:
    """One post-pool replay exemplar (21 B serialized).

    Stored *after* the encoder, so replaying it never re-enters the encoder.
    That is what makes post-pool replay storage-compressive and cheap to
    replay, and it is why the encoder cannot be adapted from it.
    """

    pooled: np.ndarray
    pre_rr: int
    post_rr: int
    label: int
    valid: int = 1
    class_id: int = 0

    def __post_init__(self) -> None:
        if self.pooled.shape != (POSTPOOL_VALUES,):
            raise ValueError(
                f"post-pool replay vector must have shape ({POSTPOOL_VALUES},), "
                f"got {self.pooled.shape}"
            )
