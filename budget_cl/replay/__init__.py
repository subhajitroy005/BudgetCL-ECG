"""Fixed-capacity source replay buffers and their byte-exact records.

The replay buffer holds SOURCE-domain (DS1) exemplars only. It is frozen at the
start of adaptation and never absorbs target-patient beats, which is what makes
it a fixed persistent-state cost rather than a growing one.
"""

from __future__ import annotations

from .record_formats import (
    POSTPOOL_PAYLOAD_BYTES,
    POSTPOOL_RECORD_BYTES,
    RAW_PAYLOAD_BYTES,
    RAW_RECORD_BYTES,
    PostPoolReplayRecord,
    RawReplayRecord,
)
from .selectors import class_balanced_selection, random_selection
from .serialization import (
    deserialize_postpool_record,
    deserialize_raw_record,
    record_size,
    serialize_postpool_record,
    serialize_raw_record,
)

__all__ = [
    "POSTPOOL_PAYLOAD_BYTES",
    "POSTPOOL_RECORD_BYTES",
    "PostPoolReplayRecord",
    "RAW_PAYLOAD_BYTES",
    "RAW_RECORD_BYTES",
    "RawReplayRecord",
    "class_balanced_selection",
    "deserialize_postpool_record",
    "deserialize_raw_record",
    "random_selection",
    "record_size",
    "serialize_postpool_record",
    "serialize_raw_record",
]
