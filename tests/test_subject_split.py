"""Cohort integrity: the 201/202 correction must stay in force."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from budget_cl.data import (
    EXCLUDED_RECORDS,
    MITBIH_DS1_RECORDS,
    MITBIH_DS2_PRIMARY_21,
    MITBIH_DS2_PUBLISHED,
    assert_subject_disjoint,
    record_to_subject,
)

REPO = Path(__file__).resolve().parents[1]


def test_record_202_is_excluded():
    """Record 202 must not appear in the primary target cohort."""
    assert "202" not in MITBIH_DS2_PRIMARY_21


def test_primary_cohort_is_21_records():
    """21 patient-disjoint records, not the published 22."""
    assert len(MITBIH_DS2_PRIMARY_21) == 21
    assert len(MITBIH_DS2_PUBLISHED) == 22


def test_source_target_subjects_are_disjoint():
    """No SUBJECT may appear on both sides of the split."""
    source = {record_to_subject(r) for r in MITBIH_DS1_RECORDS}
    target = {record_to_subject(r) for r in MITBIH_DS2_PRIMARY_21}
    assert source.isdisjoint(target)
    assert_subject_disjoint()


def test_record_lists_alone_do_not_catch_the_defect():
    """Record-list disjointness PASSES on the contaminated cohort.

    This is why the original audit missed record 202, and why the subject-level
    check has to exist separately.
    """
    assert set(MITBIH_DS1_RECORDS).isdisjoint(set(MITBIH_DS2_PUBLISHED))
    contaminated = {record_to_subject(r) for r in MITBIH_DS2_PUBLISHED}
    source = {record_to_subject(r) for r in MITBIH_DS1_RECORDS}
    assert not source.isdisjoint(contaminated)  # subject check DOES catch it


def test_202_maps_to_subject_201():
    """202 and 201 are the same subject per 202.hea."""
    assert record_to_subject("202") == "201"
    assert record_to_subject("201") == "201"


def test_exclusion_has_a_documented_reason():
    """Every exclusion must carry a reason, not just a flag."""
    assert "202" in EXCLUDED_RECORDS
    assert "201" in EXCLUDED_RECORDS["202"]


def test_manifest_matches_the_code():
    """The released manifest must agree with the in-code cohort definition."""
    manifest = REPO / "manifests" / "mitbih_ds2_primary_21.csv"
    if not manifest.exists():
        pytest.skip("cohort manifest not present")
    with manifest.open() as fh:
        rows = list(csv.DictReader(fh))
    included = {r["record_id"] for r in rows if r["included"] == "true"}
    assert included == set(MITBIH_DS2_PRIMARY_21)
    row_202 = next(r for r in rows if r["record_id"] == "202")
    assert row_202["included"] == "false"
    assert row_202["subject_id"] == "201"
