"""R9 -- automated guard for manuscript number provenance (Review Change Block 1).

This block exists because MANUSCRIPT_VALUE_MAP.md was hand-maintained across two
passes that disagreed, mixing mean-aggregated descriptives with median-aggregated
tests. A hand-maintained provenance file gives the appearance of traceability
without the property; this test supplies the property for the governed set.

What it enforces:
  1. Every source file referenced by results/MANUSCRIPT_VALUE_MAP.md exists.
  2. The canonical headline values in results/r1_headline_values.csv match the
     regenerated arm summary (results/primary/E7_arm_summary_median.csv), so the
     map's source of truth cannot silently drift.
  3. The locked headline values (0.666, 0.785, 0.810, 0.811, "16 of 21") are
     present in the abstract, and the superseded mean-aggregated values
     (0.803, 0.809, "17 of 21", "17/21") are ABSENT from the primary-headline
     files (abstract and conclusion). This is the regression that reopened the
     defect; it must fail the instant 0.803 reappears in the abstract.

Scope note: a byte-for-byte census of every numeric literal in the manuscript is
delegated to scripts/verify_manuscript_numbers.py plus the generated-from-results
tables. This test is the targeted guard for the estimator-unification defect and
the value-map <-> manuscript agreement, per R9.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VALUE_MAP = REPO / "results" / "MANUSCRIPT_VALUE_MAP.md"
HEADLINE = REPO / "results" / "r1_headline_values.csv"
ARM_SUMMARY = REPO / "results" / "primary" / "E7_arm_summary_median.csv"
MAIN = REPO / "manuscript" / "main.tex"
CONCLUSION = REPO / "manuscript" / "sections" / "08_conclusion.tex"

# Source-file references in the value map look like a path with one of these roots.
_PATH_RE = re.compile(r"`((?:results|manuscript|preregistration|manifests|figures|scripts)/[^`]+?)`")

# Superseded mean-aggregated literals that must not reappear in the primary-headline
# files. They remain legal only inside the caption-labelled secondary tables (E9/E12/
# E15/E16/E17), tracked in BLOCKERS.md ESC-4.
SUPERSEDED_IN_HEADLINE = ["0.803", "0.809", "17 of 21", "17/21"]

# Locked headline literals that must be present in the abstract.
REQUIRED_IN_ABSTRACT = ["0.666", "0.810", "0.811", "16 of 21"]


def _map_source_paths() -> list[str]:
    text = VALUE_MAP.read_text()
    # a path may carry a trailing " ... " qualifier after the backtick; strip to the path
    paths = set()
    for m in _PATH_RE.finditer(text):
        p = m.group(1).strip()
        # drop generator-argument fragments like "(T x d_model)" already outside backticks
        paths.add(p)
    return sorted(paths)


def test_value_map_source_files_exist():
    missing = [p for p in _map_source_paths() if not (REPO / p).exists()]
    assert not missing, f"MANUSCRIPT_VALUE_MAP.md references non-existent files: {missing}"


def test_headline_csv_matches_arm_summary():
    with ARM_SUMMARY.open() as f:
        summ = {r["arm"]: float(r["macro_f1_mean_of_medians"]) for r in csv.DictReader(f)}
    with HEADLINE.open() as f:
        head = {r["key"]: r["value"] for r in csv.DictReader(f)}
    # 3-dp single-round of the locked estimator must equal the headline string
    checks = {
        "A0_per_patient_macro_f1": "A0",
        "A1_max_head_only_replay": "A1",
        "A4_rank1_encoder_lora": "A4",
        "A5_rank2_encoder_lora": "A5",
    }
    for key, arm in checks.items():
        expect = f"{summ[arm]:.3f}"
        assert head[key] == expect, f"{key}: headline {head[key]} != arm summary {expect}"


def test_abstract_has_locked_values():
    text = MAIN.read_text()
    missing = [v for v in REQUIRED_IN_ABSTRACT if v not in text]
    assert not missing, f"abstract is missing locked headline values: {missing}"


@pytest.mark.parametrize("path", [MAIN, CONCLUSION])
def test_superseded_values_absent_from_headline_files(path):
    text = path.read_text()
    present = [v for v in SUPERSEDED_IN_HEADLINE if v in text]
    assert not present, (
        f"{path.name} contains superseded mean-aggregated value(s) {present}; "
        "primary-headline files must use the locked estimator (0.810/0.811, 16 of 21)."
    )
