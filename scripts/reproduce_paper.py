#!/usr/bin/env python3
"""One-command reproduction of every published artifact.

Runs the chain end to end::

    released per-cell CSVs
      -> statistics  -> tables -> figures -> LaTeX -> number verification

By design this does NOT re-train. Training the full grid takes hours on a GPU
and needs the PhysioNet recordings; the released per-cell CSVs are the
reproducibility boundary. Use ``make run-primary`` to regenerate those from raw
data.

Exit code is non-zero if any stage fails, so CI can gate on it.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(label: str, cmd: list[str], cwd: Path | None = None, optional: bool = False) -> bool:
    """Run one stage, printing a banner and reporting success."""
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    result = subprocess.run(cmd, cwd=cwd or REPO)
    if result.returncode != 0:
        print(f"  -> {'SKIPPED' if optional else 'FAILED'}: {label}", file=sys.stderr)
        return optional
    print(f"  -> ok: {label}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-paper", action="store_true",
                        help="Skip the LaTeX build (useful without a TeX install).")
    args = parser.parse_args()

    py = sys.executable
    stages: list[tuple[str, list[str], Path | None, bool]] = [
        ("1/6  audit: leakage and reproducibility",
         [py, "scripts/leakage_audit.py"], None, False),
        ("2/6  statistics: bootstrap, Wilcoxon, Holm",
         [py, "scripts/run_statistics.py"], None, False),
        ("3/6  tables: LaTeX from released CSVs",
         [py, "scripts/make_tables.py"], None, False),
        ("4/6  figures: PDFs from released CSVs",
         [py, "scripts/make_figures.py"], None, False),
    ]

    # The LaTeX build is optional: a reader may reproduce every number without
    # a TeX installation, and failing the whole run for that would be wrong.
    if not args.skip_paper:
        engine = None
        for candidate in ("latexmk", "tectonic"):
            if shutil.which(candidate):
                engine = candidate
                break
        if engine == "latexmk":
            stages.append(("5/6  paper: compile manuscript",
                           ["latexmk", "-pdf", "main.tex"], REPO / "manuscript", True))
        elif engine == "tectonic":
            stages.append(("5/6  paper: compile manuscript",
                           ["tectonic", "-X", "compile", "main.tex"], REPO / "manuscript", True))
        else:
            print("note: no latexmk or tectonic found; skipping the paper build")

    stages.append(("6/6  verify: manuscript numbers vs released artifacts",
                   [py, "scripts/verify_manuscript_numbers.py"], None, False))

    results = [(label, run(label, cmd, cwd, optional)) for label, cmd, cwd, optional in stages]

    print(f"\n{'=' * 78}\nREPRODUCTION SUMMARY\n{'=' * 78}")
    for label, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")

    failed = [label for label, ok in results if not ok]
    if failed:
        print(f"\n{len(failed)} stage(s) failed")
        return 1
    print("\nAll stages passed. Every released number regenerated from saved artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
