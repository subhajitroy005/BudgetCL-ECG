#!/usr/bin/env python3
"""Fetch the DS1 source checkpoint and verify it against the pinned hash.

A published checksum alone is not enough to reproduce anything -- the file has
to be obtainable. This script downloads ``source_model.keras`` from the GitHub
release asset for the current tag and verifies its SHA-256 before writing it
into place.

    python scripts/download_checkpoint.py
    python scripts/verify_checkpoint_hash.py

The download is verified BEFORE the file is moved to its final location, so a
truncated or tampered download can never leave a plausible-looking checkpoint
on disk.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from budget_cl.models.checkpoint_loader import EXPECTED_SOURCE_SHA256  # noqa: E402
from budget_cl.utils import get_logger, repo_root  # noqa: E402
from budget_cl.utils.hashing import sha256_file  # noqa: E402
from budget_cl.version import RELEASE_TAG  # noqa: E402

LOG = get_logger("download_checkpoint")

REPO_URL = "https://github.com/subhajitroy005/BudgetCL-ECG"
ASSET = "source_model.keras"
DEFAULT_URL = f"{REPO_URL}/releases/download/{RELEASE_TAG}/{ASSET}"
DEST = repo_root() / "checkpoints" / "source" / ASSET


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="Override the download URL.")
    parser.add_argument("--dest", default=str(DEST), help="Destination path.")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if a valid checkpoint is present.")
    args = parser.parse_args()

    dest = Path(args.dest)
    if dest.exists() and not args.force:
        digest = sha256_file(dest)
        if digest == EXPECTED_SOURCE_SHA256:
            LOG.info("checkpoint already present and verified: %s", dest)
            return 0
        LOG.warning("existing checkpoint has the WRONG hash (%s); re-downloading", digest[:16])

    LOG.info("downloading %s", args.url)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Download to a temporary file and verify BEFORE moving it into place.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".partial") as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(args.url, tmp_path)  # noqa: S310 - fixed https release URL
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        tmp_path.unlink(missing_ok=True)
        LOG.error("download failed: %s", exc)
        LOG.error(
            "If the release asset is not published yet, obtain %s from the "
            "archived snapshot listed in checkpoints/README.md and place it at %s",
            ASSET, dest,
        )
        return 1

    digest = sha256_file(tmp_path)
    if digest != EXPECTED_SOURCE_SHA256:
        tmp_path.unlink(missing_ok=True)
        LOG.error("hash mismatch -- refusing to install the download")
        LOG.error("  expected %s", EXPECTED_SOURCE_SHA256)
        LOG.error("  found    %s", digest)
        return 1

    shutil.move(str(tmp_path), dest)
    LOG.info("verified and installed: %s", dest)
    LOG.info("sha256 %s", digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
