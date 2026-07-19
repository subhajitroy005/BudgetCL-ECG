"""Aggregation to the correct statistical unit.

Two collapses happen, in this order, and both are load-bearing:

1. SEEDS within a patient  -- five seeds become one score per patient, so the
   bootstrap resamples patients rather than treating seeds as independent.
2. RECORDINGS within a SUBJECT (external databases only) -- INCART's 11
   evaluated recordings come from only 6 unique subjects, so a record-level
   analysis would treat non-independent recordings as independent samples.

Aggregating to subjects fixes the INDEPENDENCE defect in the statistics. It
does NOT make the protocol longitudinal: adaptation is still record-wise with
the model reset between recordings.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

__all__ = ["mean_over_seeds", "aggregate_records_to_subjects"]


def mean_over_seeds(scores_by_patient: dict[int, list[float]]) -> dict[int, float]:
    """Collapse per-seed scores to one score per patient."""
    return {p: float(np.mean(v)) for p, v in scores_by_patient.items() if len(v) > 0}


def aggregate_records_to_subjects(
    scores_by_record: dict[str, float],
    record_to_subject: dict[str, str],
) -> dict[str, float]:
    """Average recording scores within each unique subject.

    Raises:
        KeyError: if a record has no subject mapping. Failing loudly is
            deliberate -- a silently unmapped record would be counted as its
            own subject and reinflate the sample size, which is exactly the
            defect this function exists to fix.
    """
    grouped: dict[str, list[float]] = defaultdict(list)
    for record, score in scores_by_record.items():
        if record not in record_to_subject:
            raise KeyError(f"record {record!r} has no subject mapping")
        grouped[record_to_subject[record]].append(score)
    return {s: float(np.mean(v)) for s, v in grouped.items()}
