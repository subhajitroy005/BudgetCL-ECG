"""Byte-exact packing of replay records.

The paper's memory claim is only meaningful if the records actually pack to the
sizes it quotes, so serialization is implemented explicitly here rather than
inferred from a struct definition, and the sizes are asserted on every call.

Layout (little-endian, packed, no implicit padding)::

    raw record        203 B = 198 int8 ecg | int8 pre_rr | int8 post_rr
                              | uint8 label | uint8 valid | uint8 class_id
    post-pool record   21 B =  16 int8 pooled | int8 pre_rr | int8 post_rr
                              | uint8 label | uint8 valid | uint8 class_id
"""

from __future__ import annotations

import numpy as np

from .record_formats import (
    POSTPOOL_RECORD_BYTES,
    POSTPOOL_VALUES,
    RAW_ECG_VALUES,
    RAW_RECORD_BYTES,
    PostPoolReplayRecord,
    RawReplayRecord,
)

__all__ = [
    "serialize_raw_record",
    "serialize_postpool_record",
    "deserialize_raw_record",
    "deserialize_postpool_record",
    "record_size",
]


def _as_int8(values: np.ndarray, name: str) -> np.ndarray:
    """Cast to int8, refusing values that would wrap around silently."""
    arr = np.asarray(values)
    if arr.dtype != np.int8:
        if np.any(arr < -128) or np.any(arr > 127):
            raise ValueError(f"{name} contains values outside the int8 range")
        arr = arr.astype(np.int8)
    return arr


def serialize_raw_record(record: RawReplayRecord) -> bytes:
    """Pack a raw replay record into exactly 203 bytes."""
    payload = _as_int8(record.ecg, "ecg").tobytes()
    rr = np.array([record.pre_rr, record.post_rr], dtype=np.int8).tobytes()
    meta = np.array([record.label, record.valid, record.class_id], dtype=np.uint8).tobytes()
    out = payload + rr + meta
    # Guard the paper's headline byte figure at the point it is produced.
    if len(out) != RAW_RECORD_BYTES:
        raise AssertionError(f"raw record serialized to {len(out)} B, expected {RAW_RECORD_BYTES}")
    return out


def serialize_postpool_record(record: PostPoolReplayRecord) -> bytes:
    """Pack a post-pool replay record into exactly 21 bytes."""
    payload = _as_int8(record.pooled, "pooled").tobytes()
    rr = np.array([record.pre_rr, record.post_rr], dtype=np.int8).tobytes()
    meta = np.array([record.label, record.valid, record.class_id], dtype=np.uint8).tobytes()
    out = payload + rr + meta
    if len(out) != POSTPOOL_RECORD_BYTES:
        raise AssertionError(
            f"post-pool record serialized to {len(out)} B, expected {POSTPOOL_RECORD_BYTES}"
        )
    return out


def deserialize_raw_record(blob: bytes) -> RawReplayRecord:
    """Inverse of :func:`serialize_raw_record`.

    Raises:
        ValueError: if the blob is not exactly 203 bytes.
    """
    if len(blob) != RAW_RECORD_BYTES:
        raise ValueError(f"expected {RAW_RECORD_BYTES} B, got {len(blob)}")
    ecg = np.frombuffer(blob[:RAW_ECG_VALUES], dtype=np.int8).copy()
    pre_rr, post_rr = np.frombuffer(blob[RAW_ECG_VALUES:RAW_ECG_VALUES + 2], dtype=np.int8)
    label, valid, class_id = np.frombuffer(blob[RAW_ECG_VALUES + 2:], dtype=np.uint8)
    return RawReplayRecord(ecg, int(pre_rr), int(post_rr), int(label), int(valid), int(class_id))


def deserialize_postpool_record(blob: bytes) -> PostPoolReplayRecord:
    """Inverse of :func:`serialize_postpool_record`.

    Raises:
        ValueError: if the blob is not exactly 21 bytes.
    """
    if len(blob) != POSTPOOL_RECORD_BYTES:
        raise ValueError(f"expected {POSTPOOL_RECORD_BYTES} B, got {len(blob)}")
    pooled = np.frombuffer(blob[:POSTPOOL_VALUES], dtype=np.int8).copy()
    pre_rr, post_rr = np.frombuffer(blob[POSTPOOL_VALUES:POSTPOOL_VALUES + 2], dtype=np.int8)
    label, valid, class_id = np.frombuffer(blob[POSTPOOL_VALUES + 2:], dtype=np.uint8)
    return PostPoolReplayRecord(
        pooled, int(pre_rr), int(post_rr), int(label), int(valid), int(class_id)
    )


def record_size(location: str) -> int:
    """Serialized bytes for a replay location.

    Args:
        location: ``"raw"`` or ``"postpool"``.

    Raises:
        ValueError: on an unknown location.
    """
    sizes = {"raw": RAW_RECORD_BYTES, "postpool": POSTPOOL_RECORD_BYTES}
    if location not in sizes:
        raise ValueError(f"unknown replay location {location!r}; expected one of {sorted(sizes)}")
    return sizes[location]
