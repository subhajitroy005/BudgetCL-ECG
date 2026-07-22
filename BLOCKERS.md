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
