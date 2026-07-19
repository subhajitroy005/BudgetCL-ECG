#!/usr/bin/env python3
"""Check the local dataset layout against the released manifests.

Reports what is present, what is missing, and which records are deliberately
excluded, so a reproduction fails early and legibly instead of halfway through
a multi-hour run.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from budget_cl.utils import repo_root  # noqa: E402

RAW = repo_root() / "datasets" / "raw"


def check(manifest: Path, subdir: str) -> tuple[int, int, int]:
    """Return (present, missing, excluded) counts for one manifest."""
    if not manifest.exists():
        print(f"  SKIP  {manifest.name} not found")
        return 0, 0, 0

    present = missing = excluded = 0
    with manifest.open() as fh:
        for row in csv.DictReader(fh):
            record = row["record_id"]
            if row.get("included", "true") == "false":
                excluded += 1
                print(f"  EXCL  {record}: {row.get('reason', '')}")
                continue
            if (RAW / subdir / f"{record}.hea").exists():
                present += 1
            else:
                missing += 1
    return present, missing, excluded


def main() -> int:
    manifests = repo_root() / "manifests"
    print("Dataset manifest verification")
    print(f"  raw data root: {RAW}")
    if not RAW.exists():
        print("\n  raw data not downloaded. Run:  make download-data")
        print("  PhysioNet recordings are NOT redistributed with this repository.")
        return 1

    total_missing = 0
    for name, subdir in (
        ("mitbih_ds1_records.csv", "mitdb"),
        ("mitbih_ds2_primary_21.csv", "mitdb"),
    ):
        print(f"\n{name}")
        present, missing, excluded = check(manifests / name, subdir)
        print(f"  present {present} / missing {missing} / excluded {excluded}")
        total_missing += missing

    if total_missing:
        print(f"\nFAIL  {total_missing} configured record(s) missing")
        return 1
    print("\nPASS  every configured record is present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
