"""Subject identity across the DS1/DS2 split.

The cohort trap this module exists to prevent
---------------------------------------------
The de Chazal DS1/DS2 partition is RECORD-disjoint, not PATIENT-disjoint.
MIT-BIH record 202's own WFDB header states it "was taken from the same analog
tape as record 201", and the two carry identical demographics. de Chazal places
**201 in DS1 and 202 in DS2**, so a source model trained on DS1 has already
seen that subject: evaluating adaptation on record 202 as though it were an
unseen patient measures partly memorization rather than personalization.

All 48 MIT-BIH records were audited for this failure mode; 201/202 is the only
cross-split pair. Record 202 is excluded, leaving **21 genuinely
patient-disjoint records / 630 primary cells**, not 22 / 660.

This is not cosmetic. On the contaminated 22-record cohort the A4-vs-A3
comparison was significant (Holm p = 0.047); on the corrected cohort it is not
(p = 0.075). Any 22-record number is not strictly patient-disjoint and may be
optimistically biased.
"""

from __future__ import annotations

__all__ = [
    "MITBIH_DS1_RECORDS",
    "MITBIH_DS2_PUBLISHED",
    "MITBIH_DS2_PRIMARY_21",
    "EXCLUDED_RECORDS",
    "SAME_SUBJECT_PAIRS",
    "record_to_subject",
    "assert_subject_disjoint",
]

# de Chazal et al. (2004) published partition.
MITBIH_DS1_RECORDS: tuple[str, ...] = (
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
    "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
    "223", "230",
)

MITBIH_DS2_PUBLISHED: tuple[str, ...] = (
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
    "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
    "233", "234",
)

# Records sharing a subject across the split: {record: canonical subject}.
SAME_SUBJECT_PAIRS: dict[str, str] = {"202": "201"}

EXCLUDED_RECORDS: dict[str, str] = {
    "202": "Same subject as source record 201 (202.hea: same analog tape)",
}

# The corrected primary cohort actually used for every headline number.
MITBIH_DS2_PRIMARY_21: tuple[str, ...] = tuple(
    r for r in MITBIH_DS2_PUBLISHED if r not in EXCLUDED_RECORDS
)


def record_to_subject(record: str) -> str:
    """Canonical subject for a record.

    Identity for every record except the known same-subject pair, where 202
    maps to subject 201.
    """
    return SAME_SUBJECT_PAIRS.get(str(record), str(record))


def assert_subject_disjoint(
    source_records: tuple[str, ...] = MITBIH_DS1_RECORDS,
    target_records: tuple[str, ...] = MITBIH_DS2_PRIMARY_21,
) -> None:
    """Fail if source and target cohorts share a SUBJECT.

    Record-list disjointness is not enough -- that check passes on the
    contaminated cohort, which is how the defect survived the original audit.

    Raises:
        AssertionError: listing the offending subjects.
    """
    source_subjects = {record_to_subject(r) for r in source_records}
    target_subjects = {record_to_subject(r) for r in target_records}
    overlap = source_subjects & target_subjects
    if overlap:
        raise AssertionError(
            f"source and target cohorts share subject(s) {sorted(overlap)}; "
            "the evaluation is not patient-disjoint"
        )
