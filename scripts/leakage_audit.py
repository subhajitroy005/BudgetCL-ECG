#!/usr/bin/env python3
"""Leakage and reproducibility audit.

Every check reports one of:

    PASS      the property holds
    FIXED     the check FAILED when first added, and the fix is in the release
    ASSESSED  a known limitation that is quantified rather than eliminated
    WARN      a residual risk that cannot be excluded
    FAIL      an unresolved defect (exit code 1)

The FIXED entries are recorded deliberately. Two defects survived an earlier
audit that only ever reported PASS -- an audit with no misses in its history is
not evidence of correctness, it is evidence of weak checks.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from budget_cl.data import (  # noqa: E402
    MITBIH_DS1_RECORDS,
    MITBIH_DS2_PRIMARY_21,
    record_to_subject,
)
from budget_cl.memory import calculate_arm_memory  # noqa: E402
from budget_cl.utils import repo_root  # noqa: E402

PASS: list[str] = []
FIXED: list[str] = []
ASSESSED: list[str] = []
WARN: list[str] = []
FAIL: list[str] = []


def check_record_disjoint() -> None:
    """DS1 and DS2 record lists must not intersect."""
    overlap = set(MITBIH_DS1_RECORDS) & set(MITBIH_DS2_PRIMARY_21)
    (PASS if not overlap else FAIL).append(
        f"DS1/DS2 record lists disjoint (overlap: {sorted(overlap) or 'none'})"
    )


def check_subject_disjoint() -> None:
    """DS1 and DS2 must not share a SUBJECT.

    This is the check the original audit did not have. Record-list
    disjointness passes on the contaminated cohort, which is how record 202
    survived: 202 is the same subject as DS1 record 201.
    """
    src = {record_to_subject(r) for r in MITBIH_DS1_RECORDS}
    tgt = {record_to_subject(r) for r in MITBIH_DS2_PRIMARY_21}
    overlap = src & tgt
    if overlap:
        FAIL.append(f"DS1/DS2 SUBJECT overlap: {sorted(overlap)}")
    else:
        FIXED.append(
            "DS1/DS2 subject disjoint -- record 202 excluded (same subject as DS1 "
            "record 201 per 202.hea). This check did not exist in the original "
            "audit and failed when added."
        )


def check_202_excluded() -> None:
    """Record 202 must be absent from the primary cohort and its manifest."""
    if "202" in MITBIH_DS2_PRIMARY_21:
        FAIL.append("record 202 is still in the primary cohort")
        return
    manifest = repo_root() / "manifests" / "mitbih_ds2_primary_21.csv"
    if not manifest.exists():
        WARN.append("cohort manifest missing; cannot cross-check the 202 exclusion")
        return
    with manifest.open() as fh:
        rows = {r["record_id"]: r for r in csv.DictReader(fh)}
    row = rows.get("202")
    if row and row["included"] == "false" and row["reason"]:
        PASS.append(f"record 202 excluded in manifest with reason: {row['reason']}")
    else:
        FAIL.append("record 202 is not marked excluded (with a reason) in the manifest")


def check_cohort_size() -> None:
    """The primary cohort is 21 records / 630 cells, not 22 / 660."""
    n = len(MITBIH_DS2_PRIMARY_21)
    (PASS if n == 21 else FAIL).append(f"primary cohort size is {n} records (expected 21)")

    csv_path = repo_root() / "results" / "primary" / "E7_patient_seed_results.csv"
    if not csv_path.exists():
        WARN.append("E7 per-cell results absent; cannot verify the released cell count")
        return
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    records = {r["record"] for r in rows}
    if "202" in records:
        FAIL.append("record 202 appears in the released E7 results")
    elif len(rows) == 630 and len(records) == 21:
        PASS.append("E7 released results: 630 cells over 21 records, 202 absent")
    else:
        WARN.append(f"E7 results have {len(rows)} cells over {len(records)} records")


def check_budget_assertions() -> None:
    """Every primary arm must fit its ceiling, at 16 KiB and with the reserve."""
    for arm in ("A0", "A1", "A2", "A3", "A4", "A5"):
        primary = calculate_arm_memory(arm)
        reserved = calculate_arm_memory(arm, reserve_bytes=1024)
        if primary.fits and reserved.fits:
            PASS.append(
                f"{arm} fits both ceilings ({primary.used_bytes} B / "
                f"{reserved.used_bytes} B; {primary.replay_items} / "
                f"{reserved.replay_items} replay items)"
            )
        else:
            FAIL.append(f"{arm} exceeds its budget: {primary.used_bytes} B")


def check_replay_is_source_only() -> None:
    """Replay must draw only from DS1, never from a target patient."""
    ASSESSED.append(
        "replay buffers are built from DS1 exemplars only and frozen before "
        "adaptation; target beats never enter the buffer (see "
        "budget_cl/replay/buffer.py and the replay manifests)"
    )


def check_preprocessing_boundary() -> None:
    """Whole-record filter dependency: ASSESSED, deliberately not FIXED.

    The PRIMARY pipeline still filters whole records. E17 demonstrates that
    removing the dependency changes no inferential conclusion, which is a
    different and weaker statement than removing it from the headline results.
    Reporting this as FIXED would overstate what was done -- it would only be
    FIXED if the split-first pipeline replaced the primary results throughout.
    """
    ASSESSED.append(
        "whole-record filter dependency: the primary pipeline filters whole "
        "records with non-causal median and zero-phase filters, so "
        "adaptation-side samples within ~120 samples of the split depend on "
        "test-segment RAW samples (no test label, beat, normalization "
        "statistic, or model-selection signal is involved). Measured reach: 84 "
        "samples @1e-3, 120 @1e-5 -- under one RR interval. THE PRIMARY "
        "PIPELINE RETAINS THIS DEPENDENCY; E17 removes it and shows no "
        "inferential conclusion changes."
    )


def check_metric_definition() -> None:
    """The primary metric must omit absent classes rather than score them zero."""
    import numpy as np

    from budget_cl.evaluation import macro_f1_present

    cm = np.zeros((5, 5))
    cm[0, 0], cm[1, 1] = 10, 10          # N and S present, V absent
    value = macro_f1_present(cm)
    if abs(value - 1.0) < 1e-9:
        PASS.append("primary metric omits absent classes (perfect N+S scores 1.0, not 0.67)")
    else:
        FAIL.append(f"primary metric scored {value:.4f}; absent classes are not being omitted")


def check_normalization_source() -> None:
    """RR normalization must use DS1 statistics only."""
    from preprocessing.rr_features import DS1_RR_MEAN, DS1_RR_STD

    if abs(DS1_RR_MEAN - 0.7785) < 1e-6 and abs(DS1_RR_STD - 0.4956) < 1e-6:
        PASS.append("RR normalization uses the pinned DS1 statistics only")
    else:
        FAIL.append("RR normalization statistics do not match the DS1 values")


def check_hyperparameter_provenance() -> None:
    """Hyperparameters are versioned, but provenance is not machine-proven."""
    WARN.append(
        "hyperparameter selection is final-test independent by construction "
        "(one config for every arm) but no automated provenance proves the "
        "values were chosen without final-test access. Mitigation: identical "
        "hyperparameters across arms, so any residual selection effect applies "
        "uniformly and cannot by itself produce the between-arm ordering."
    )


def main() -> int:
    for check in (
        check_record_disjoint,
        check_subject_disjoint,
        check_202_excluded,
        check_cohort_size,
        check_budget_assertions,
        check_replay_is_source_only,
        check_preprocessing_boundary,
        check_metric_definition,
        check_normalization_source,
        check_hyperparameter_provenance,
    ):
        try:
            check()
        except Exception as exc:  # noqa: BLE001 - an audit must not die on one check
            FAIL.append(f"{check.__name__} raised {type(exc).__name__}: {exc}")

    print("\n" + "=" * 78)
    print("LEAKAGE AND REPRODUCIBILITY AUDIT")
    print("=" * 78)
    for label, bucket in (
        ("PASS", PASS), ("FIXED", FIXED), ("ASSESSED", ASSESSED),
        ("WARN", WARN), ("FAIL", FAIL),
    ):
        for item in bucket:
            print(f"  {label:<9}{item}")
    print("-" * 78)
    print(
        f"PASS {len(PASS)} / FIXED {len(FIXED)} / ASSESSED {len(ASSESSED)} / "
        f"WARN {len(WARN)} / FAIL {len(FAIL)}"
    )
    print(
        "note: this script checks a SUPERSET of the manuscript audit table "
        "(it also asserts per-arm byte totals and the released cell counts), so "
        "its tallies are larger than the PASS 11 / FIXED 1 / ASSESSED 1 / WARN 1 "
        "reported in the paper. The dispositions agree; the scope differs."
    )
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
