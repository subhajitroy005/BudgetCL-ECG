"""Content hashing for release integrity.

Every released artifact carries a SHA-256 so a reviewer can prove the code, the
checkpoint, and the results they have are the ones the paper describes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = ["sha256_file", "sha256_bytes", "config_hash"]

_CHUNK = 1 << 20  # 1 MiB, so large checkpoints stream rather than load


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's contents, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """SHA-256 of a byte string."""
    return hashlib.sha256(data).hexdigest()


def config_hash(config: dict[str, Any]) -> str:
    """Stable hash of a resolved configuration.

    Keys are sorted so logically identical configs hash identically regardless
    of YAML key order; this is what lets a result row claim which config
    produced it.
    """
    return sha256_bytes(json.dumps(config, sort_keys=True, default=str).encode())
