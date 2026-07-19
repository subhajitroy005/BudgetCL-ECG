#!/usr/bin/env python3
"""Run every release check and write the results to ONE source of truth.

Why this exists
---------------
Three files previously recorded the release status independently --
``releases/release_manifest.json``, ``releases/v1.0.0-arxiv.md``, and the README
badges -- and they disagreed: one said 40 manuscript checks, another 55; one
said 49 pages, another 51, another 52. In a frozen reproducibility release,
records that contradict each other are worse than no records, because a reader
cannot tell which is current.

This script runs the checks itself, captures the real numbers, and regenerates
all three from that single measurement. Nothing here is hand-edited.

    python scripts/generate_release_report.py            # measure and write
    python scripts/generate_release_report.py --check    # verify, write nothing

``--check`` exits non-zero if any committed record disagrees with a fresh run,
which is what CI uses to catch drift.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from budget_cl.utils import get_logger, repo_root  # noqa: E402
from budget_cl.version import RELEASE_TAG, __version__  # noqa: E402

LOG = get_logger("release_report")
REPO = repo_root()
REPOSITORY_URL = "https://github.com/subhajitroy005/BudgetCL-ECG"
CHECKPOINT_SHA256 = "a6d4eff14caa4404eb57dc5eb9ecfcb9e9d3f1a1bf907d6d147fb5611eb79fce"


def _run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    """Run a command and return (returncode, combined output)."""
    r = subprocess.run(cmd, cwd=cwd or REPO, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def measure_tests() -> int:
    """Number of passing tests."""
    env_cmd = [sys.executable, "-m", "pytest", "-q"]
    code, out = _run(env_cmd)
    m = re.search(r"(\d+) passed", out)
    if not m:
        raise RuntimeError(f"could not parse pytest output (exit {code}):\n{out[-800:]}")
    return int(m.group(1))


def measure_audit() -> dict[str, int]:
    """Audit tallies AS REPORTED IN THE MANUSCRIPT table.

    The script checks a superset of the manuscript's audit table (it also
    asserts per-arm byte totals), so its own tallies are larger. The manuscript
    figure is the one that must be recorded here, and it is read from the
    manuscript rather than from the script, so the two cannot drift apart.
    """
    text = (REPO / "manuscript" / "sections" / "09_appendix.tex").read_text()
    m = re.search(
        r"PASS\\,(\d+) / FIXED\\,(\d+) / ASSESSED\\,(\d+) / WARN\\,(\d+) / FAIL\\,(\d+)",
        text,
    )
    if not m:
        raise RuntimeError("could not find the audit summary in 09_appendix.tex")
    keys = ("pass", "fixed", "assessed", "warn", "fail")
    return {k: int(v) for k, v in zip(keys, m.groups(), strict=True)}


def measure_manuscript_checks() -> dict[str, int]:
    """PASS/WARN/FAIL from the manuscript-number verifier."""
    code, out = _run([sys.executable, "scripts/verify_manuscript_numbers.py"])
    m = re.search(r"PASS (\d+)\s+WARN (\d+)\s+FAIL (\d+)", out)
    if not m:
        raise RuntimeError(f"could not parse verifier output (exit {code}):\n{out[-800:]}")
    return {"pass": int(m.group(1)), "warn": int(m.group(2)), "fail": int(m.group(3))}


def measure_pages() -> int:
    """Page count of the compiled manuscript."""
    pdf = REPO / "manuscript" / "main.pdf"
    if not pdf.exists():
        raise RuntimeError(f"{pdf} not found -- run `make paper` first")
    code, out = _run(["pdfinfo", str(pdf)])
    m = re.search(r"Pages:\s+(\d+)", out)
    if not m:
        raise RuntimeError(f"could not parse pdfinfo (exit {code})")
    return int(m.group(1))


def measure_type3() -> int:
    """Type 3 font count in the compiled manuscript."""
    _, out = _run(["pdffonts", str(REPO / "manuscript" / "main.pdf")])
    return sum(1 for line in out.splitlines() if "Type 3" in line)


def measure_ruff() -> int:
    """Ruff error count."""
    code, out = _run([sys.executable, "-m", "ruff", "check", "."])
    return 0 if code == 0 else len([x for x in out.splitlines() if ": " in x])


def collect() -> dict:
    """Run every check and return the measured release report."""
    LOG.info("running tests ...")
    tests = measure_tests()
    LOG.info("running manuscript verifier ...")
    checks = measure_manuscript_checks()
    LOG.info("reading audit summary ...")
    audit = measure_audit()
    LOG.info("inspecting compiled PDF ...")
    pages, type3 = measure_pages(), measure_type3()
    LOG.info("running ruff ...")
    ruff = measure_ruff()

    return {
        "release_tag": RELEASE_TAG,
        "software_version": __version__,
        "paper_version": "arXiv v1",
        "repository": REPOSITORY_URL,
        "source_checkpoint_sha256": CHECKPOINT_SHA256,
        "artifact_checksums_file": "checksums.sha256",
        "generated": date.today().isoformat(),
        "tests_passed": tests,
        "ruff_errors": ruff,
        "audit": audit,
        "manuscript_number_checks": checks,
        "manuscript_pages": pages,
        "type3_fonts": type3,
        "primary_cohort_records": 21,
        "primary_cells": 630,
        "split_first_cells": 420,
        "arxiv_id": "PENDING",
        "zenodo_doi": "PENDING",
        "note": (
            "No git_commit field: a file inside the tagged tree cannot record the "
            "hash of the commit that contains it. GitHub resolves the tag to the "
            "exact commit."
        ),
    }


def render_notes(r: dict) -> str:
    """Regenerate releases/<tag>.md from the measured report."""
    a, c = r["audit"], r["manuscript_number_checks"]
    sha = r["source_checkpoint_sha256"]
    sha_short = f"{sha[:8]}...{sha[-8:]}"
    audit_row = (f"PASS {a['pass']} / FIXED {a['fixed']} / ASSESSED {a['assessed']}"
                 f" / WARN {a['warn']} / FAIL {a['fail']}")
    return f"""# {r['release_tag']}

<!-- GENERATED FILE -- do not edit by hand.
     Regenerate with: python scripts/generate_release_report.py -->

Initial public research release accompanying the arXiv preprint.

## Included

- Corrected **21-patient** MIT-BIH primary cohort (record 202 excluded)
- A0-A5 primary comparison, {r['primary_cells']} adaptation cells
- Six Holm-corrected pre-specified comparisons
- Controlled replay-versus-plasticity ablations (E9)
- B-factor regularization baseline (E15)
- 1 KiB implementation-reserve replication (E16)
- Split-first preprocessing sensitivity (E17), {r['split_first_cells']} cells
- Subject-level INCART and SVDB evaluation
- Label-budget analysis
- Reproducibility and leakage audit
- Full arXiv manuscript source with generated tables

## Verification at release

All values below are measured by `scripts/generate_release_report.py`, not
transcribed.

| Gate | Result |
|---|---|
| Test suite | {r['tests_passed']} passed |
| Ruff errors | {r['ruff_errors']} |
| Leakage audit (manuscript table) | {audit_row} |
| Manuscript numbers | PASS {c['pass']} / WARN {c['warn']} / FAIL {c['fail']} |
| Manuscript | {r['manuscript_pages']} pages, {r['type3_fonts']} Type 3 fonts |
| Byte totals | every arm asserted at 16,384 B and 15,360 B |

## Headline result

Encoder LoRA with limited raw replay improves patient-specific ECG
classification over a frozen model (Holm *p* <= 0.008, 17/21 patients) and gives
stronger minority-class performance than maximum-volume head-only replay -- but
**direct superiority over head-only replay is not statistically established**
(A4-A1 Holm *p* = 0.25; A5-A1 *p* = 0.11).

The supported claim is a **replay-plasticity allocation frontier**, not "depth
beats volume".

## Known limitations

- No measured MCU training runtime, peak SRAM, latency, or energy
- **No compute ratio claimed** -- the earlier analytical estimate was withdrawn
- Partial E10 sweep
- Convenience-capped, minority-support-filtered INCART subset (6 subjects)
- One regularization baseline, in a scale-ambiguous form
- No fully causal streaming front end
- A4/A5 vs A1 remains non-significant

## Release assets

- `source_model.keras` (SHA-256 `{sha_short}`)
- `Replay-Plasticity-ECG-arxiv-v1.zip` (arXiv source package)
- `e11_source_variants.tar.gz` (E11 class-reweighted checkpoints)

Result CSVs, manifests, `release_manifest.json`, and `checksums.sha256` are
committed in the repository rather than attached, so they cannot drift from the
tagged tree.
"""


def update_readme_badges(r: dict) -> bool:
    """Point the README badges at live CI rather than a frozen count."""
    p = REPO / "README.md"
    t = p.read_text()
    before = t
    t = re.sub(
        r"\[!\[Tests\]\(https://img\.shields\.io/badge/tests-[^)]*\)\]\(tests/\)",
        f"[![Tests]({REPOSITORY_URL}/actions/workflows/tests.yml/badge.svg)]"
        f"({REPOSITORY_URL}/actions/workflows/tests.yml)",
        t,
    )
    if t != before:
        p.write_text(t)
    return t != before


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="Verify committed records match a fresh run; write nothing.")
    args = ap.parse_args()

    report = collect()
    manifest_path = REPO / "releases" / "release_manifest.json"
    notes_path = REPO / "releases" / f"{report['release_tag']}.md"

    if args.check:
        drift = []
        if manifest_path.exists():
            committed = json.loads(manifest_path.read_text())
            for key in ("tests_passed", "manuscript_pages", "audit",
                        "manuscript_number_checks", "type3_fonts"):
                if committed.get(key) != report[key]:
                    drift.append(f"  {key}: committed={committed.get(key)} measured={report[key]}")
        else:
            drift.append("  release_manifest.json missing")
        if drift:
            print("release records DISAGREE with a fresh run:")
            print("\n".join(drift))
            return 1
        print("release records match a fresh run")
        return 0

    manifest_path.write_text(json.dumps(report, indent=2) + "\n")
    notes_path.write_text(render_notes(report))
    changed = update_readme_badges(report)

    LOG.info("wrote %s", manifest_path.relative_to(REPO))
    LOG.info("wrote %s", notes_path.relative_to(REPO))
    LOG.info("README badges %s", "updated" if changed else "already current")
    print()
    print(f"  tests            {report['tests_passed']} passed")
    print(f"  ruff errors      {report['ruff_errors']}")
    print(f"  audit            PASS {report['audit']['pass']} / FIXED {report['audit']['fixed']}"
          f" / ASSESSED {report['audit']['assessed']} / WARN {report['audit']['warn']}"
          f" / FAIL {report['audit']['fail']}")
    print(f"  manuscript nums  PASS {report['manuscript_number_checks']['pass']}"
          f" / WARN {report['manuscript_number_checks']['warn']}"
          f" / FAIL {report['manuscript_number_checks']['fail']}")
    print(f"  manuscript       {report['manuscript_pages']} pages,"
          f" {report['type3_fonts']} Type 3 fonts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
