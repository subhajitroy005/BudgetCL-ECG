"""Replay records must pack to exactly the sizes the paper claims."""

from __future__ import annotations

import numpy as np
import pytest

from budget_cl.replay import (
    POSTPOOL_PAYLOAD_BYTES,
    POSTPOOL_RECORD_BYTES,
    RAW_PAYLOAD_BYTES,
    RAW_RECORD_BYTES,
    PostPoolReplayRecord,
    RawReplayRecord,
    deserialize_postpool_record,
    deserialize_raw_record,
    serialize_postpool_record,
    serialize_raw_record,
)


def test_raw_record_size(example_raw_record):
    """Raw replay record serializes to 203 B."""
    assert len(serialize_raw_record(example_raw_record)) == 203


def test_postpool_record_size(example_postpool_record):
    """Post-pool replay record serializes to 21 B."""
    assert len(serialize_postpool_record(example_postpool_record)) == 21


def test_payload_and_record_sizes_are_distinct():
    """Payload and serialized record are different quantities (200/203, 18/21).

    Conflating them understates a budget, which is why the paper reports both.
    """
    assert RAW_PAYLOAD_BYTES == 200
    assert RAW_RECORD_BYTES == 203
    assert POSTPOOL_PAYLOAD_BYTES == 18
    assert POSTPOOL_RECORD_BYTES == 21
    assert RAW_RECORD_BYTES - RAW_PAYLOAD_BYTES == 3      # label + valid + class_id
    assert POSTPOOL_RECORD_BYTES - POSTPOOL_PAYLOAD_BYTES == 3


def test_raw_roundtrip(example_raw_record):
    """Serialization is lossless."""
    back = deserialize_raw_record(serialize_raw_record(example_raw_record))
    assert np.array_equal(back.ecg, example_raw_record.ecg)
    assert back.pre_rr == example_raw_record.pre_rr
    assert back.post_rr == example_raw_record.post_rr
    assert back.label == example_raw_record.label
    assert back.class_id == example_raw_record.class_id


def test_postpool_roundtrip(example_postpool_record):
    """Serialization is lossless."""
    back = deserialize_postpool_record(serialize_postpool_record(example_postpool_record))
    assert np.array_equal(back.pooled, example_postpool_record.pooled)
    assert back.label == example_postpool_record.label


def test_wrong_shape_is_rejected():
    """A malformed record must fail at construction, not at pack time."""
    with pytest.raises(ValueError, match="shape"):
        RawReplayRecord(ecg=np.zeros(100, np.int8), pre_rr=0, post_rr=0, label=0)
    with pytest.raises(ValueError, match="shape"):
        PostPoolReplayRecord(pooled=np.zeros(8, np.int8), pre_rr=0, post_rr=0, label=0)


def test_deserialize_rejects_wrong_length():
    """Truncated blobs must raise rather than silently misparse."""
    with pytest.raises(ValueError):
        deserialize_raw_record(b"\x00" * 202)
    with pytest.raises(ValueError):
        deserialize_postpool_record(b"\x00" * 20)
