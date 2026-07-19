#!/usr/bin/env python3
"""Download the PhysioNet databases into datasets/raw/.

These recordings are NOT redistributed by this repository. This script fetches
them from PhysioNet, where they remain under their own terms.

    python scripts/download_datasets.py              # all three
    python scripts/download_datasets.py --db mitdb   # just MIT-BIH
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from budget_cl.utils import get_logger, repo_root  # noqa: E402

LOG = get_logger("download_datasets")

DATABASES = {
    "mitdb": "MIT-BIH Arrhythmia Database (source DS1 + target DS2)",
    "incartdb": "St. Petersburg INCART 12-lead Arrhythmia Database (external)",
    "svdb": "MIT-BIH Supraventricular Arrhythmia Database (external)",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", choices=sorted(DATABASES), action="append",
                        help="Download only this database (repeatable).")
    parser.add_argument("--dest", default=str(repo_root() / "datasets" / "raw"))
    args = parser.parse_args()

    try:
        import wfdb
    except ImportError:
        LOG.error("wfdb is not installed. Run: pip install -r requirements.txt")
        return 1

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    for slug in args.db or sorted(DATABASES):
        target = dest / slug
        if target.exists() and any(target.glob("*.hea")):
            LOG.info("%s already present at %s -- skipping", slug, target)
            continue
        LOG.info("downloading %s: %s", slug, DATABASES[slug])
        target.mkdir(parents=True, exist_ok=True)
        try:
            wfdb.dl_database(slug, str(target))
        except Exception as exc:  # noqa: BLE001 - report and continue to the next db
            LOG.error("failed to download %s: %s", slug, exc)
            return 1

    LOG.info("done. Verify the layout with: make verify-data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
