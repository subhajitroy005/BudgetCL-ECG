# Datasets

**No PhysioNet recording is redistributed in this repository.** All three
databases are public; download them yourself and comply with the terms attached
to each.

| Database | PhysioNet slug | Role | Records used |
|---|---|---|---|
| MIT-BIH Arrhythmia | `mitdb` | source (DS1) + target (DS2) | 22 source / **21** target |
| St. Petersburg INCART | `incartdb` | external validation | 13 attempted, 11 evaluated (**6 subjects**) |
| MIT-BIH Supraventricular | `svdb` | external validation | 54 attempted, 46 evaluated |

## Download

```bash
make download-data     # or: python scripts/download_datasets.py
make verify-data       # checks the layout against manifests/
```

## Expected layout

```
datasets/
├── raw/               # git-ignored: WFDB files as downloaded
│   ├── mitdb/         #   100.dat 100.hea 100.atr ...
│   ├── incartdb/
│   └── svdb/
├── processed/         # git-ignored: extracted beats and RR features
└── cache/             # git-ignored: optional feature caches
```

Roughly 2 GB for all three databases.

## Cohort notes that matter

**MIT-BIH DS2 is 21 records, not 22.** Record 202's header states it was "taken
from the same analog tape as record 201", and de Chazal places 201 in DS1.
Evaluating on 202 measures partly memorization. All 48 records were audited;
201/202 is the only cross-split pair. See
[`manifests/exclusions.md`](../manifests/exclusions.md).

**External cohorts are convenience samples.** INCART was capped at the first 13
records for tractability, and eligibility additionally requires ≥30 S+V beats in
the *test* window — so the rule uses test-window labels and enriches the cohort
for recordings where minority-class adaptation can be evaluated at all. This does
not invalidate the external study, but it rules out any claim of representative
full-database validation.

**INCART recordings are not independent.** The 11 evaluated recordings come from
only 6 unique subjects, so all INCART statistics aggregate to subjects.
