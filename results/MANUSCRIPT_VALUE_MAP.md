# Manuscript value map

Every numeric claim in the manuscript → its source file under `results/` (or the
generator that writes it). Numbers are the **pre-registered median-primary** values
(spec §4.2.2); the mean values remain in `e8_sensitivity_mean_aggregation.csv`.

## Abstract / headline

| Manuscript number | Meaning | Source file | Column / row |
|---|---|---|---|
| 6,643 | model parameters | `manifests/source_checkpoint_manifest.json` | `parameters` |
| 1,056 / 8,448 / 16 | token / FFN / pooled elements | `scripts/gen_e6_accounting.py` (T×d_model, T×d_ff, d_model) | — |
| 0.666 | A0 mean macro-F1 | `results/primary/E7_arm_summary.csv` | A0 `mean` (median-agg identical: 0.6664) |
| 0.810 | A4 mean (median-agg) | recomputed by `run_equivalence_mde.py` | A4 arm central |
| 0.812 | A5 mean (median-agg) | recomputed by `run_equivalence_mde.py` | A5 arm central |
| 16 of 21 | A4/A5 patients improved vs A0 (median) | `run_equivalence_mde.py` improved count | A4/A5–A0 |
| 12 | A1 patients improved vs A0 (median) | `run_equivalence_mde.py` improved count | A1–A0 |
| Holm p ≤ 0.017 | A4/A5 vs A0 significance | `results/e8_paired_tests.csv` | `p_holm` A4–A0 (0.007), A5–A0 (0.017) |
| p = 0.005 | plasticity ablation (B4−B2) | `results/ablations/E9_paired_contrasts.csv` | `p_wilcoxon` B4_minus_B2 |
| δ = 0.02 | equivalence margin | `preregistration/paper1_stat_plan.md` | §1 |
| MDE ≈ 0.01–0.04 | minimum detectable effect | `results/e8_mde.csv` | `mde_power80` |

## §3 Method — two-axis accounting (T1)

| Number | Meaning | Source |
|---|---|---|
| 0.1 KiB | A1–A3 peak transient (batch-1 FP16) | `results/e6_two_axis_accounting.csv` (A1/A2/A3, R2/R3) |
| 154.6 / 77.3 / 41.3 KiB | A4/A5 transient R1 / R2 / R3 | `results/e6_two_axis_accounting.csv` (A4/A5) |
| 1.99× | R3 compute multiplier | `results/e6_two_axis_accounting.csv` `compute_multiplier` |
| ~300× | A1 vs A4 transient asymmetry (R3) | 41.3 KiB ÷ 0.1 KiB, from the CSV |
| 33.0 KiB | A0 inference arena (FP16) | `results/e6_two_axis_accounting.csv` A0 R2/R3 |
| 16,208 / 16,226 / 16,374 … | per-arm persistent totals | `results/e6_two_axis_accounting.csv` `persistent_total_bytes` |

## §4 Protocol / §5 Results — paired comparisons (T2, median primary)

| Number | Meaning | Source (`results/e8_paired_tests.csv` unless noted) |
|---|---|---|
| n = 21 | statistical unit | `n` column (all rows) |
| median Δ +0.0007 / +0.0034 | A4–A1 / A5–A1 | `median_difference` |
| Holm p = 0.079 | A4–A1, A5–A1 | `p_holm` |
| rank-biserial + BCa CI | all comparisons | `rank_biserial`, `rb_ci_low/high` |
| 90% TOST CI, equivalence bool | all comparisons | `results/e8_tost_equivalence.csv` |
| A4–A2 equivalent | within δ | `results/e8_tost_equivalence.csv` (equivalent=True) |
| record-232 reversal | A4–A1 equivalence flips | `results/e8_sensitivity_record232.csv` vs primary |
| mean-agg p = 0.25 / 0.11 | A4/A5–A1 (sensitivity b) | `results/e8_sensitivity_mean_aggregation.csv` |
| Table V | regenerated | `manuscript/tables/table_V_paired_comparisons.tex` ← `run_equivalence_mde.py` |
| Fig 5 | paired diffs + ±δ band | `figures/paper/fig05_paired_differences.pdf` |
| Fig 13 | two-axis Pareto | `figures/paper/fig13_two_axis_pareto.pdf` |
| Fig 3 | persistent composition | `figures/paper/fig03_budget_composition.pdf` |

## §6 Discussion — survey refutation

| Claim | Source |
|---|---|
| survey quote "backpropagation is unlikely to constitute a bottleneck relative to the buffer" | `\cite{pavan2026odlsurvey}` (arXiv:2605.31226, verified line 1840 of the PDF) |
| refutation: 41.3 KiB ≈ 2.5× the 16 KiB buffer | `results/e6_two_axis_accounting.csv` A4 R3 |

## Orphan-number check

No manuscript number in the edited sections lacks a `results/` source. **Remaining
sync pass (flagged in PAPER1_STATUS.md):** §5 results *prose* still contains some
mean-aggregation figures from the pre-pre-registration draft; these must be
reconciled to the median-primary values above during the compile-verified pass in
the author's TeX environment (no TeX Live on the analysis host).
