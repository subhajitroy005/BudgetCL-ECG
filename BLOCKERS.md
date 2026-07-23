# Paper 1 — Experiment Block 1: Blockers & Escalations

Append-only. Each entry: ID, task, severity, what, why it blocks, recommended action.

---

## ESC-1 (T1) — A4/A5 R3 peak transient exceeds the ~24 KiB escalation threshold

- **Task:** T1 (two-axis accounting), spec §3.6 failure criterion 2 / §8 escalation 2.
- **Severity:** framing (NOT a hard stop; does not invalidate any result or change arm sizes).
- **Finding:** With FP16 + block-granularity rematerialization (regime R3), the
  encoder-LoRA arms A4/A5 have a peak transient working set of **42,312 B = 41.3 KiB**,
  above the ~24 KiB the spec flags. Driver: the FFN hidden tensor (66×128 = 8,448
  elements) and its gradient. `results/e6_two_axis_accounting.csv`, `results/e6_accounting_report.md`.
- **Why the spec flags it:** if adaptation transient is large, the T3 argument
  "peak transient is aliasable onto the inference arena, so excluding it from the
  16 KiB persistent budget is defensible" weakens.
- **Mitigating analysis (do not treat as a resolution — owner decision):** the
  frozen-model inference forward arena (A0) is itself **33.0 KiB at FP16** (it holds
  the same 8,448-element FFN tensor). A4/A5 R3 adaptation transient (41.3 KiB) is
  therefore only **≈1.25× the inference arena the device already provisions**. The
  marginal transient beyond inference is the FFN-hidden gradient (~8,448 elems ≈
  16.5 KiB FP16, or ~8.3 KiB if the gradient aliases the activation via in-place).
  The aliasing argument survives *with this caveat stated explicitly*, but it is no
  longer "negligible."
- **Recommended action (owner):** keep T3's §5.7 limitation framing as "peak
  transient reported under three regimes; R3 is ≈1.25–2× the inference arena and
  partially aliasable — argued, not assumed," rather than claiming full aliasing.
  Do NOT soften the R3 number. Confirm n_heads=1/key_dim=16 (assumed from the spec's
  own 17 KiB arithmetic and invariant in the param count) if the source `.keras`
  becomes available — a larger n_heads raises the attention (not FFN) working set.
- **Status:** RECORDED. T1 outputs produced in full. T2 is independent and proceeds.
  T3 framing paragraph held for owner sign-off.

---

## ESC-2 (T2) — Pre-registered median aggregation changes the abstract's headline numbers

- **Task:** T2, spec §8 escalation 4 ("re-derived statistics disagree materially with the abstract").
- **Severity:** framing/reporting (no result invalidated; conclusion structure unchanged).
- **Finding:** the locked plan fixes **median** seed aggregation as primary. Under it:
  A4–A1 Holm p 0.25→**0.079** (median Δ **+0.0007**), A5–A1 0.11→0.079; A4/A5 arm means
  0.803/0.809→**0.810/0.812**; vs-A0 stays significant (Holm 0.007/0.017). The **mean**-aggregation
  sensitivity reproduces the released E8 exactly (0.2517/0.1097/0.0071/0.0080), so the old abstract
  numbers are the mean values — correct, but now sensitivity (b), not primary.
- **Why it matters:** T3 will rewrite the abstract's numbers. Median-primary yields a cleaner frontier
  story (median A4–A1 difference ≈ 0) but different figures than Subhajit has been circulating.
- **Also (spec §4.6 mandatory report):** dropping record 232 flips the A4–A1 **equivalence** verdict
  (not-equivalent → equivalent). The A4–A1 equivalence is 232-fragile.
- **Owner decision:** (1) confirm **median** as the manuscript primary (per the locked plan), or elect
  mean; (2) confirm the A4–A1 headline as "inconclusive (bounded, not equivalent within 0.02)" per
  spec §4.7 row 2. Do not soften either. The lock is already committed to median.
- **Status:** RECORDED. T2 outputs produced in full under median primary + both sensitivities.

---

## ESC-3 (R1/R2) — A5 headline: traceable value is 0.811, review specifies 0.812

- **Task:** Review Change Block 1, R1/R2 (headline number sweep).
- **Severity:** low (0.001 cosmetic; no conclusion affected).
- **Finding:** A5 (rank-2 encoder LoRA) `mean_of_medians` = **0.811468** exactly
  (`results/primary/E7_arm_summary_median.csv`, `results/r0_estimator_matrix.csv`).
  At 3 dp this single-rounds to **0.811**. The review's R1 headline table and R2
  defect table specify **0.812**, which is a double-round of the review's own 4 dp
  checksum (0.8115 -> 0.812). Our 4 dp value (0.8115) matches the review's checksum
  exactly, so we agree on the underlying number and differ only on 3 dp rendering.
- **Decision (rule 8 trust the repository; rule 3 traceability):** manuscript uses
  **0.811**. A4 = 0.8104 -> 0.810 is unambiguous and matches the review. Every
  headline traces to `E7_arm_summary_median.csv` / `r0_estimator_matrix.csv`.
- **Owner note:** if you prefer the review's 0.812 for external consistency with
  circulated drafts, it is a deliberate double-round of 0.8115; say so and I will
  switch, but the single-round-correct figure is 0.811.
- **Status:** RECORDED. Proceeding with 0.811.

---

## ESC-4 (R1/R2) — Secondary-experiment descriptives remain seeds-mean-averaged

- **Task:** Review Change Block 1, R1/R2. Escalation trigger 5 (orphan numbers R2
  did not anticipate) + rule 8 (trust repo, record discrepancy).
- **Severity:** consistency/disclosure. No conclusion inverts; primary cohort is
  fully unified to the locked median estimator.
- **Finding:** the PRIMARY (E7) cohort is fully unified to `mean_of_medians`
  (arm summary, harm table, all RQ2 prose, abstract, conclusion, §6). The
  SECONDARY / sensitivity experiments still report seeds-mean-averaged descriptive
  statistics. Occurrences (all now caption-labelled "seeds mean-averaged"):
  - `manuscript/tables/table_reserve.tex` (E16) — unreserved column imports primary
    means 0.803/0.809; reserved column is E16 mean. Median values exist
    (`results/reserve/E16_patient_seed_results.csv`) but regeneration flips the
    reserved ordering (A2/A5 swap: median A2 0.791 > A5 0.787) and touches the
    prose "$p\le0.010$" test reference.
  - `manuscript/tables/table_splitfirst.tex` (E17) + §5:568 — A4/A5 0.803/0.809.
    Median: A4 0.805 (16/21), A5 0.816 (17/21) — no longer an identical count.
  - `manuscript/tables/table_regularization.tex` (E15) — embeds test p-values
    (p=0.0012, 0.0011); regenerating under median re-runs those tests (rule 2).
  - `manuscript/tables/table_e12_label_curve.tex` (E12) — §15 marks E12 out of scope.
  - `manuscript/tables/table_causal_ablation.tex` (E9) + §5:216 — **hard-blocked**:
    ablation-specific arms B2/B3/B6/B7 have NO per-seed file under `results/`
    (only `results/ablations/E9_paired_contrasts.csv` exists), so their arm means
    cannot be median-regenerated from current data. The confirmatory contrasts
    already use the locked median-of-differences statistic.
- **Why not done in this block:** (1) rule 2 forbids re-running hypothesis tests,
  and E15/E16 tables embed secondary test statistics; (2) E9 per-seed data is
  missing; (3) §15 places E12 (and budget-sweep) out of scope. Doing it partially
  would splice median primary baselines onto mean secondary variants inside single
  comparison arrows.
- **Owner decision:** authorize a follow-up unification block that (a) regenerates
  E15/E16/E17/E12 descriptives + re-runs their sensitivity test families under
  median, and (b) recovers or re-runs E9 per-seed records for B2/B3/B6/B7; OR
  accept the secondary tables as explicitly-labelled seeds-mean-averaged
  sensitivity analyses (current state). A global disclosure footnote is in §6.
- **Status:** RECORDED. Primary unified; secondary labelled and deferred.

---

## ESC-6 (R8) — Minority-class-support correlation lacks a source file

- **Task:** Review Change Block 1, R8 (exploratory responder structure).
- **Severity:** low (exploratory analysis; does not touch any pre-registered result).
- **Finding:** R8 asks for the correlation between per-patient A4-A1 gain and
  per-patient minority-class (S+V+F) test-window support. No per-record class-support
  file exists under `results/`, and the repository contains no raw PhysioNet beat
  annotations (`.atr/.dat/.hea`) from which support could be derived offline. Rule 3
  (never invent a number) forbids fabricating support counts.
- **Delivered instead:** `results/r8_responder_exploratory.csv` and the §6
  exploratory subsection report the fully traceable half of the responder
  structure -- the per-patient A4-A1 gain distribution and its mass concentration
  (14 better / 3 tied / 4 worse; top-3 patients hold 61% of positive mass) -- all
  from `results/primary/E7_patient_seed_results.csv`.
- **Owner decision:** to complete the support correlation, provide (or authorize
  regenerating) a per-record S+V+F test-window support table from the PhysioNet
  annotations; then the correlation can be added to the same exploratory subsection.
- **Status:** RECORDED. Responder structure delivered; support correlation deferred.
