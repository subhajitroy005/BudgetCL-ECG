# Paper 1 — Pre-registered Statistical Analysis Plan

**Status:** locked before any equivalence (TOST) or MDE test is executed.
**Manuscript:** *The Replay-Point Cliff …* (Paper 1), primary comparison A4/A5 vs A1.
**Data source (fixed):** `results/primary/E7_patient_seed_results.csv` — 6 arms ×
21 patient-disjoint MIT-BIH DS2 records × 5 seeds = 630 chronological adaptation
cells, column `macro_f1_present`.

This plan is written and committed **before** the equivalence and MDE analyses
are run. The lock is recorded in `preregistration/PREREG_LOCK.txt` (git SHA +
SHA-256 of this file + ISO-8601 UTC timestamp). Every analysis script asserts the
lock exists and that this file's SHA-256 still matches before it computes a single
test statistic; on mismatch it aborts.

No number below references an observed difference. The equivalence margin, the
tests, the family, the effect size, the MDE procedure, and the sensitivity
analyses are all fixed here in advance.

---

## 1. Equivalence margin δ

**δ = 0.02 macro-F1**, fixed in advance.

Justification (written before any equivalence test; does **not** reference
observed differences): for a screening-tier wearable ECG classifier whose role
is to flag beats for downstream review, a per-patient macro-F1 difference below
0.02 does not change which patients are triaged or the clinician's review burden
in a way that is clinically actionable. 0.02 is also well below the arm-vs-frozen
effect the paper is built on. A difference smaller than δ is therefore treated as
"practically the same operating point."

If this margin is ever revised, the new justification must be written before any
test is run and must not cite observed differences.

## 2. Statistical unit and aggregation

- **Unit of analysis: the patient. n = 21.** Never the seed-level cell (n ≠ 105)
  and never the arm×patient×seed cell (n ≠ 630).
- **Seed aggregation: median across the 5 seeds, within patient, BEFORE any
  test.** This yields one score per patient per arm; the 21 per-patient scores
  are the observations. (The existing released pipeline used the *mean* across
  seeds; mean aggregation is retained as pre-declared sensitivity analysis (b),
  §8. Median is the pre-registered primary.)

## 3. Primary comparison family (family of 6)

`A4−A1, A5−A1, A4−A2, A4−A3, A4−A0, A5−A0`.
Paired differences are per-patient `left − right` on the median-aggregated scores.

## 4. Superiority test

Two-sided **Wilcoxon signed-rank** on the 21 per-patient paired differences,
**Holm-corrected within the family of 6**. Zero differences handled by SciPy's
`zero_method="wilcox"` (dropped), matching the reported effect-size convention.

## 5. Equivalence test (TOST)

**Nonparametric TOST.** For each comparison, compute the **90% BCa bootstrap CI**
(bias-corrected and accelerated; **10,000 resamples; fixed seed 20260719**) of the
**median** paired difference, resampling the 21 patients with replacement.
**Equivalence at α = 0.05 is declared iff that 90% CI is fully contained in
[−δ, +δ] = [−0.02, +0.02].** (A 90% two-sided CI implements the two one-sided
tests at α = 0.05 each.)

## 6. Effect size

**Matched-pairs rank-biserial correlation** (paper convention
`r_rb = 1 − 2W / (n(n+1)/2)`), reported with a **BCa bootstrap CI** (10,000
resamples, fixed seed 20260719) resampling patients.

## 7. Minimum detectable effect (MDE)

**Monte Carlo.** Take the observed per-patient paired-difference vector for a
comparison, **centre it** (subtract its median so the null holds), add a candidate
constant shift Δ, resample n = 21 patients with replacement, run the two-sided
Wilcoxon test at the **Holm-adjusted α for that comparison's position in the
family of 6**. Sweep Δ from 0 upward; report the **smallest Δ achieving power ≥
0.80**. **10,000 iterations per Δ, fixed seed 20260719.** Δ grid: 0.00 to 0.20 in
steps of 0.005.

## 8. Sensitivity analyses (declared in advance)

- **(a) Record 232:** repeat the primary family with and without record 232.
  Conclusions must be reported as unchanged, or the reversal reported prominently.
- **(b) Aggregation:** repeat the primary family with **mean** instead of median
  seed aggregation.

## 9. Family membership of the ablation test

The B-arm plasticity ablation (**B4 − B2**, "plasticity beyond raw replay",
reported at p = 0.005 in `results/ablations/E9_paired_contrasts.csv`) is declared
as a **separate family** from the family of 6, **Holm-corrected within its own E9
causal-ablation family** (B4−B3, B4−B2, B1−B6, B4−B7), **not** pooled with the
primary comparisons.

Why separate: the family of 6 tests *which A-arm operating point wins*; the E9
family tests *the causal mechanism* (does encoder plasticity, or replay volume,
carry the gain?) via matched controls that hold one factor fixed. These are
different hypotheses over different arm sets; pooling them would both mis-state
the correction and conflate an operating-point question with a mechanism question.

---

## Frozen parameters

| Parameter | Value |
|---|---|
| δ (equivalence margin) | 0.02 macro-F1 |
| Statistical unit / n | patient / 21 |
| Seed aggregation (primary) | median of 5 |
| Primary family | A4−A1, A5−A1, A4−A2, A4−A3, A4−A0, A5−A0 (6) |
| Superiority | two-sided Wilcoxon, Holm over 6 |
| Equivalence | nonparametric TOST via 90% BCa CI ⊂ [−δ,+δ] |
| Bootstrap | BCa, 10,000 resamples, seed 20260719 |
| Effect size | rank-biserial + BCa CI |
| MDE | Monte Carlo, 10,000 iters/Δ, power ≥ 0.80, seed 20260719 |
| Sensitivities | (a) drop record 232; (b) mean aggregation |
| Ablation family | E9 (B4−B3, B4−B2, B1−B6, B4−B7), separate, Holm within |
