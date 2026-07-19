# Released results

Machine-readable outputs. Every file here is either a direct experiment output
or is regenerated from one by `scripts/run_statistics.py`.

## Per-cell results (experiment output)

| File | Cells | Contents |
|---|---:|---|
| `primary/E7_patient_seed_results.csv` | 630 | 6 arms × 21 records × 5 seeds |
| `reserve/E16_patient_seed_results.csv` | 630 | same grid at the 15,360 B ceiling |
| `preprocessing_sensitivity/E17_patient_seed_results.csv` | 420 | 4 arms, split-first pipeline |
| `regularization/E15_patient_seed_results.csv` | 210 | R1/R2 B-factor anchor |

Columns: `experiment_id, arm, record, seed, macro_f1_present, source_change,
used_bytes, replay_items`.

`macro_f1_present` is macro-F1 over the N/S/V classes **present** in that
patient's test segment, recomputed from the saved confusion matrix. Absent
classes are omitted, not scored zero — the difference is about 0.15 macro-F1.

`source_change` is signed: `source_after − source_before`. Negative means source
performance fell, so −0.027 is a *smaller* loss than −0.136.

## Derived summaries (regenerate with `make statistics`)

| File | Feeds |
|---|---|
| `primary/E7_patient_summary.csv` | seeds averaged within patient |
| `primary/E7_arm_summary.csv` | Table: primary arms |
| `primary/E8_paired_tests.csv` | Table: six pre-specified comparisons |
| `preprocessing_sensitivity/E17_arm_summary.csv` | Table + figure: sensitivity |

## What is not here

- Raw TensorFlow logs and per-cell checkpoints (large, not needed to reproduce
  any published number).
- PhysioNet recordings.
- Adapted model binaries.
