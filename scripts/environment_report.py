#!/usr/bin/env python3
"""Print (and optionally save) the environment report for a run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from budget_cl.utils import environment_report  # noqa: E402


def main() -> int:
    report = environment_report()
    print(json.dumps(report, indent=2))
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {sys.argv[1]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
