#!/usr/bin/env python3
"""Verify the source checkpoint SHA-256 against the release manifest.

Every result adapts from ONE frozen DS1 checkpoint. A reproduction that starts
from a different checkpoint moves every number while reporting no error, so
this check runs before any experiment.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from budget_cl.exceptions import CheckpointError  # noqa: E402
from budget_cl.models.checkpoint_loader import (  # noqa: E402
    EXPECTED_SOURCE_SHA256,
    verify_checkpoint,
)
from budget_cl.utils import repo_root  # noqa: E402

DEFAULT = repo_root() / "checkpoints" / "source" / "source_model.keras"


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    try:
        digest = verify_checkpoint(path)
    except CheckpointError as exc:
        print(f"FAIL  {exc}", file=sys.stderr)
        return 1
    print(f"PASS  {path.name} matches the pinned hash")
    print(f"      sha256 {digest}")
    print(f"      expected {EXPECTED_SOURCE_SHA256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
