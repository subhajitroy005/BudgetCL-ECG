"""Package version, kept in one place.

Must match ``pyproject.toml`` and the release tag; ``tests/test_release_integrity.py``
asserts that.
"""

from __future__ import annotations

__all__ = ["__version__", "RELEASE_TAG"]

__version__ = "1.0.0"
RELEASE_TAG = "v1.0.0-arxiv"
