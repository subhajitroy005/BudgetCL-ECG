#!/usr/bin/env python3
"""Run every figure script in figures/scripts/.

Step 5 of the reproduction chain. Each script regenerates ONE figure from
released CSVs, so a figure can never silently go stale relative to the table
beside it -- a failure mode this project has hit before.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "figures" / "scripts"


def main() -> int:
    scripts = sorted(p for p in SCRIPTS.glob("figure_*.py"))
    if not scripts:
        print("no figure scripts found", file=sys.stderr)
        return 1

    failed = []
    for script in scripts:
        print(f"--- {script.name}")
        result = subprocess.run([sys.executable, str(script)], cwd=REPO)
        if result.returncode != 0:
            failed.append(script.name)

    if failed:
        print(f"\nFAILED: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"\nregenerated {len(scripts)} figure(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
