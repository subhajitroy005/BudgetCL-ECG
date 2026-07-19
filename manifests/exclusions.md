# Exclusions

Every excluded record carries a deterministic, documented reason. Nothing is
dropped for being inconvenient or for producing an unwelcome result.

## MIT-BIH: record 202

**Excluded from the evaluation cohort.**

Record 202's WFDB header states it was "taken from the same analog tape as
record 201", and both carry identical demographics. de Chazal places **201 in
DS1** and **202 in DS2**, so a model trained on DS1 has already seen that
subject.

All 48 MIT-BIH records were audited for this failure mode. 201/202 is the only
cross-split pair. Record 201 remains a legitimate DS1 *training* record; only the
evaluation set changes.

**This is not cosmetic.** On the contaminated 22-record cohort the A4-vs-A3
comparison was significant (Holm *p* = 0.047); on the corrected cohort it is not
(*p* = 0.075). Every arm mean rose slightly. Any 22-record number is not strictly
patient-disjoint and may be optimistically biased.

Detected by a subject-level disjointness check that did **not** exist in the
original audit — record-list disjointness passes on the contaminated cohort,
which is exactly why it survived.

## External databases

| Reason | INCART | SVDB |
|---|---:|---:|
| Outside configured record cap | 62 | 24 |
| Preprocessing failure | 1 | 0 |
| Insufficient S+V test support (<30 beats) | 1 | 8 |
| **Evaluated** | **11** (6 subjects) | **46** |

Two disclosures:

1. The record cap is **computational convenience**, not a protocol criterion.
   INCART is the first 13 of 75 records.
2. The minority-support rule uses **test-window class labels**, so external
   eligibility is outcome-adjacent and enriches the cohort for recordings where
   minority-class adaptation can be evaluated. No claim of representative
   full-database validation is made from these cohorts.
