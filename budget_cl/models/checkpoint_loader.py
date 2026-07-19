"""Source checkpoint loading with integrity verification.

Every result in the paper adapts from ONE frozen DS1 checkpoint. If a
reproduction silently starts from a different checkpoint, every number moves
and nothing reports an error -- so the hash and the parameter count are checked
before the weights are used.
"""

from __future__ import annotations

import json
from pathlib import Path

from budget_cl.exceptions import CheckpointError
from budget_cl.utils.hashing import sha256_file
from budget_cl.utils.paths import repo_root

__all__ = ["EXPECTED_SOURCE_SHA256", "EXPECTED_PARAMETERS", "verify_checkpoint"]

# Pinned in configs/release/arxiv_v1.yaml and manifests/source_checkpoint_manifest.json.
EXPECTED_SOURCE_SHA256 = "a6d4eff14caa4404eb57dc5eb9ecfcb9e9d3f1a1bf907d6d147fb5611eb79fce"
EXPECTED_PARAMETERS = 6_643


def verify_checkpoint(
    path: str | Path,
    expected_sha256: str = EXPECTED_SOURCE_SHA256,
) -> str:
    """Verify a checkpoint's SHA-256.

    Args:
        path: Checkpoint file.
        expected_sha256: Hash the release manifest pins.

    Returns:
        The computed hash.

    Raises:
        CheckpointError: if the file is missing or the hash does not match.
    """
    p = Path(path)
    if not p.is_absolute():
        p = repo_root() / p
    if not p.exists():
        raise CheckpointError(
            f"source checkpoint not found: {p}\n"
            "Model binaries are not committed to Git. Fetch it from the GitHub "
            "Release or Zenodo archive listed in checkpoints/README.md."
        )
    digest = sha256_file(p)
    if digest != expected_sha256:
        raise CheckpointError(
            f"checkpoint hash mismatch for {p}\n"
            f"  expected {expected_sha256}\n"
            f"  found    {digest}\n"
            "Results from a different checkpoint are not comparable with the paper."
        )
    return digest


def load_manifest(path: str | Path | None = None) -> dict:
    """Load the source-checkpoint manifest."""
    p = Path(path) if path else repo_root() / "manifests" / "source_checkpoint_manifest.json"
    if not p.exists():
        raise CheckpointError(f"checkpoint manifest not found: {p}")
    return json.loads(p.read_text())
