# Experiment matrix (E6–E17)

"Primary" means the corrected all-DS2 cohort of **21 patient-disjoint records ×
5 seeds**. Secondary experiments use reduced panels or other databases and are
**never pooled** with primary results.

| ID | Status | Contents |
|---|---|---|
| E6 | complete | Persistent-state accounting for 8/16/32/64 KiB, Adam/SGD, INT8 replay; every arm asserted ≤ its ceiling. |
| **E7** | complete (primary) | Primary A0–A5 evaluation. 21 records × 5 seeds, `gap_beats=1`, **630 cells**. Record 202 excluded. |
| **E8** | complete (primary) | Six pre-specified paired comparisons, Holm correction, patient-bootstrap CIs, rank-biserial effects. |
| E9 | complete (primary) | Controlled replay-versus-plasticity matrix B0–B7 (B5 infeasible at 16 KiB) plus fixed-count and fixed-scope controls. |
| E10 | **partial** (secondary) | Budget sweep. 49 of 140 configurations measured, 5 infeasible, 86 unrun; Adam ranks 1–2 only, on a reduced 12-record/3-seed panel. Supports the saturation claim only — no optimizer or rank-4/8 claim. |
| E11 | **partial** | Five class-reweighted DS1 source checkpoints trained; adaptation re-run from **two** of them. Robustness claim covers two sources, not five. |
| E12 | complete | Label-budget curve (class-aware oracle and random) with coverage and help/harm counts. |
| E13 | complete (subject level) | Cross-database INCART and SVDB, aggregated to **unique subjects** and bootstrapped over subjects. |
| E14 | complete | Leakage/reproducibility audit, split manifests, checkpoint hash, config package. |
| E15 | complete (primary) | Memory-matched regularization baseline: rank-1 LoRA + L2 anchor on the **B factor**, with (R2) and without (R1) replay. Zero persistent bytes. |
| E16 | complete (primary) | 1 KiB implementation-reserve replication. Counts re-solved against 15,360 B and the grid re-run. |
| **E17** | complete (sensitivity) | Split-first preprocessing sensitivity. A0/A1/A4/A5, **420 cells**, 497 adaptation beats, 720-sample guard. |

## Things a reader should not over-read

- **E9** — only the *plasticity* contrast is statistically supported (B4−B2,
  p = 0.005). The replay contrast is positive in mean but not supported
  (B4−B3, p = 0.473), and replay *volume* shows no benefit at all (B1−B6,
  5 improved / 15 worsened). Arm means alone are misleading here.
- **E10** — the only defensible claim is budget saturation *within the measured
  panel*. 86 of 140 cells are unrun.
- **E11** — the alternative checkpoints were never shown to be **stronger** on
  unseen-patient DS2. The claim is robustness to source *objective*, not to a
  stronger source *model*.
- **E15** — one regularizer, in a scale-ambiguous form (`BA = (cB)(A/c)`). It
  does not license any claim about regularization in general.
- **E16** — every gain stays significant, but the exact **ordering is not
  preserved**. That is the finding, not a footnote.
- **E17** — a sensitivity analysis, not a replacement primary analysis.
