# Paper 1 — Experiment Block 1 Status

Executable spec: `~/Documents/Subhajit_Projects/Research/Paper_1/Context/Paper_1_Experement_1.md`
Repo: `BudgetCL-ECG` (branch `development`)

Progress log is append-only. Newest entries at the bottom of each task.

---

## T0 — Repository discovery  ✅ COMPLETE

### T0.1 Path inventory

| Artifact | Path |
|---|---|
| Model architecture (source builder) | **Not committed.** `training/models/tiny_transformer.py` is imported by `budget_cl/models/lora.py:231` but is not in the repo. Architecture reconstructed from `budget_cl/models/lora.py` (graph rebuild), `budget_cl/memory/accounting.py`, `manifests/source_checkpoint_manifest.json`, `checkpoints/README.md`, `docs/memory_accounting.md`. |
| LoRA / adapter graph rebuild | `budget_cl/models/lora.py` (`build_lora_backbone`, `build_above_pool_arm`) |
| Arm/config definitions (A0–A5, B0–B7, R1/R2) | `budget_cl/models/arm_specs.py`; per-arm configs `configs/model/a0_frozen.yaml` … `a5_lora_r2.yaml` |
| Persistent-state accounting | `budget_cl/memory/accounting.py` (`account_state`, `params_for_arm`, `account_arm`) |
| Adaptation training loop | `budget_cl/adaptation/base.py`, `budget_cl/adaptation/factory.py` |
| Per-patient results (macro-F1 × patient × seed × arm) | `results/primary/E7_patient_seed_results.csv` (630 rows) |
| Per-arm summary | `results/primary/E7_arm_summary.csv` |
| Per-patient summary | `results/primary/E7_patient_summary.csv` |
| Existing paired tests (E8) | `results/primary/E8_paired_tests.csv` |
| Ablation contrasts (E9) | `results/ablations/E9_paired_contrasts.csv` |
| Split manifests (DS1 / DS2) | `manifests/mitbih_ds1_records.csv`, `manifests/mitbih_ds2_primary_21.csv`; identity map `manifests/mitbih_subject_identity_map.csv` |
| Source checkpoint manifest | `manifests/source_checkpoint_manifest.json` (SHA-256 `a6d4eff1…9fce`, 6,643 params) |
| Manuscript sections | `manuscript/sections/01…11_*.tex` |
| Manuscript main / refs | `manuscript/main.tex`, `manuscript/references.tex` |
| Tables / figures | `manuscript/tables/*.tex`, `manuscript/figures/*.pdf`, `figures/paper/*.pdf`, `figures/scripts/*.py` |

### T0.2 / T0.3 / T0.4 Architecture (from code, param-count verified)

Source model: `conv tokenizer → 1 encoder block → mean pool → RR fusion → 5-way head`
(`manifests/source_checkpoint_manifest.json`). Confirmed against `docs/memory_accounting.md`
and the graph reconstructed in `budget_cl/models/lora.py`.

| Hyperparameter | Value | Source / confirmation |
|---|---:|---|
| Input beat length (samples) | **198** | `accounting.py` `RAW_ECG_BYTES=198`; `REPLAY_SPECS["raw"].ecg_values=198` |
| RR features | 2 | `REPLAY_SPECS` `rr_values=2` |
| Tokens `T` | **66** | 198-sample beat, Conv1D k=3 s=3 valid → (198−3)/3+1 = 66; matches "1,056 token elements" = 66×16 (README:53, abstract) |
| `d_model` (embed_dim) | **16** | `accounting.py` `embed_dim=16`; `patch_embed.filters`; pooled vector = 16 |
| `d_ff` | **128** | ffn hidden; "8,448-element FFN tensor" = 66×128 (abstract); LoRA count 128r ⇒ attention dim 16 |
| `n_layers` (encoder blocks) | **1** | source manifest "1 encoder block"; single `encoder1_*` block in `lora.py` |
| `n_heads × key_dim` | **= 16** (= d_model) | MHA param count 1,088 requires heads×key_dim=16. **Split (heads, key_dim) not identifiable from param count**; the committed `.keras` file would disambiguate. Common config: 2 heads × 8. Not needed for T1 arithmetic. |
| Classifier head | Dense(18 → 5), softmax | fused = pooled(16) + RR(2) = 18 → 5; 18×5+5 = **95** params (`params_for_arm`, A1) |

**T0.4 — confirmation of the prior derivation (T=66, d_model=16, d_ff=128): CONFIRMED against code.**

### T0.3 Parameter-count verification — sums EXACTLY to 6,643

| Layer | Shape | Params |
|---|---|---:|
| patch_embed | Conv1D(filters=16, k=3, s=3, valid), 1 in-ch | 3·1·16 + 16 = **64** |
| pos_embed | additive table 66×16 | **1,056** |
| rr_proj | Dense(2, no bias), in=2 | 2·2 = **4** |
| encoder1_ln1 | LayerNorm(16) | 2·16 = **32** |
| encoder1_mha | MHA, Q/K/V/O, dim 16 | 4·(16·16 + 16) = **1,088** |
| encoder1_ln2 | LayerNorm(16) | **32** |
| encoder1_ffn1 | Dense(16 → 128) | 16·128 + 128 = **2,176** |
| encoder1_ffn2 | Dense(128 → 16) | 128·16 + 16 = **2,064** |
| final_ln | LayerNorm(16) | **32** |
| class_probabilities | Dense(18 → 5), softmax | 18·5 + 5 = **95** |
| **Total** | | **6,643** ✅ |

LoRA on the encoder (A4/A5) targets the **4 attention projections** (Q,K,V,O), each 16×16,
rank r ⇒ r(16+16)=32r per matrix, ×4 = **128r** params — exactly `8·embed_dim·rank`
(`params_for_arm`, `accounting.py:228`). A4 = 95 + 128 = **223**; A5 = 95 + 256 = **351**.

### T0.5 Results inventory — 6 arms × 21 patients × 5 seeds = 630 cells

`results/primary/E7_patient_seed_results.csv`: **630 data rows, complete. No missing cells.**
- Arms: A0 A1 A2 A3 A4 A5 (6)
- Seeds: 42 43 44 45 46 (5)
- Records (21): 100 103 105 111 113 117 121 123 200 210 212 213 214 219 221 222 228 231 232 233 234
- Record **232 present** (needed for the T2 sensitivity analysis).
- Columns: `experiment_id, arm, record, seed, macro_f1_present, source_change, used_bytes, replay_items`

### T0 — abstract-vs-results cross-check (rule #3)

Every headline number in the current abstract traces to a result file. **No material disagreement.**

| Abstract number | Meaning | Result file | Value | Match |
|---|---|---|---|---|
| 0.666 | A0 (frozen) mean macro-F1 | `E7_arm_summary.csv` | 0.6664 | ✓ |
| 0.803 | A4 mean | `E7_arm_summary.csv` | 0.8028 | ✓ |
| 0.809 | A5 mean | `E7_arm_summary.csv` | 0.8089 | ✓ |
| p=0.25 | A4 vs A1 Holm | `E8_paired_tests.csv` | 0.2517 | ✓ |
| p=0.11 | A5 vs A1 Holm | `E8_paired_tests.csv` | 0.1097 | ✓ |
| p=0.005 | plasticity ablation (B4−B2) | `E9_paired_contrasts.csv` | 0.00511 | ✓ |
| p=0.008 | A5 vs A0 Holm | `E8_paired_tests.csv` | 0.008001 | ✓ |
| 17/21, 12 | patients improved (A4/A5 vs A1) | `E8_paired_tests.csv` | 17 (vs A0), 14 (vs A1) | ⚠ see note |

⚠ **Discrepancy DISC-1:** abstract says A4/A5 improve "17 of 21 patients against 12 for maximum
head-only replay." `E8_paired_tests.csv` shows A4/A5 vs **A0** improved=17; A4/A5 vs **A1**
improved=14/16. The "17 of 21" is the vs-A0 count. The "12 for head-only replay" is not directly
in E8 (needs the per-patient A1-vs-A0 improved count). Flag for verification during T2/T3 table
regeneration — the sentence conflates two baselines.

### T0 state of block-specific deliverables (do NOT yet exist)

- `preregistration/` — absent (T2 must create + gate).
- `results/e6_two_axis_accounting.csv`, `results/e6_accounting_report.md`, `scripts/gen_e6_accounting.py` — absent (T1).
- `figures/…/fig13_two_axis_pareto.pdf`, `fig03_budget_composition.pdf` (regenerated) — absent/legacy names differ (`fig_budget_composition.pdf` exists).
- `results/e8_tost_equivalence.csv`, `results/e8_mde.csv`, `results/e8_sensitivity_record232.csv`, `results/e8_sensitivity_mean_aggregation.csv` — absent (T2).
- `results/SUPERSEDED.md`, `results/MANUSCRIPT_VALUE_MAP.md` — absent.

Prior partial work present: `E8_paired_tests.csv` (n=21, 6-family, Holm) and TOST-related manuscript
tables (`table_pairwise_tests.tex`), per commit history ("TOST count and table fix"). These will be
reconciled, not blindly recreated (rule #7).

**T0 ACCEPTANCE: met.** Architecture table complete and param-count verified; results inventory
complete with no unknowns.

---

## T1 — Two-axis memory accounting  ✅ COMPLETE (with escalation ESC-1)

Generator: `scripts/gen_e6_accounting.py` (deterministic, stdlib for accounting; matplotlib for figures).
Run with `PYTHONNOUSERSITE=1` (system numpy 1.26.4 + matplotlib 3.6.3; user site has a conflicting numpy 2.x).

### Outputs produced
- `results/e6_two_axis_accounting.csv` — 18 rows (6 arms × 3 regimes), persistent components + transient peak.
- `results/e6_accounting_report.md` — narrative, hand-verification of A1 & A4, assumptions, fit-check.
- `figures/paper/fig03_budget_composition.pdf` — persistent-state stacked bar per arm.
- `figures/paper/fig13_two_axis_pareto.pdf` — new: persistent (x) vs peak transient (y, log), per (arm,regime).

### Persistent axis (Step 1) — all arms FIT 16,384 B
A0 0 | A1 16,374 | A2 16,383 | A3 16,371 | A4 16,208 | A5 16,226. Matches configs & `docs/memory_accounting.md`.
Basis: Adam (2 moments), weights FP32, replay INT8, packed 1-B alignment. **No `N_replay` correction needed.**

### Transient axis (Step 2) — the excluded axis, per regime
| Arm | R1 naive FP32 | R2 FP16 | R3 FP16+remat | note |
|---|---:|---:|---:|---|
| A0 | 66.0 KiB | 33.0 KiB | 33.0 KiB | inference arena; no adaptation pass |
| A1 | 0.2 KiB | 0.1 KiB | 0.1 KiB | encoder never runs (post-pool replay) |
| A2 | 0.3 KiB | 0.1 KiB | 0.1 KiB | |
| A3 | 0.3 KiB | 0.1 KiB | 0.1 KiB | |
| A4 | 154.6 KiB | 77.3 KiB | **41.3 KiB** | full fwd+bwd (raw replay); R3 mult 1.99× |
| A5 | 154.6 KiB | 77.3 KiB | **41.3 KiB** | LoRA rank doesn't change activation sizes |

The **≈300× asymmetry** (A1 0.1 KiB vs A4 41.3 KiB at R3) on the axis the 16 KiB budget excludes is a
direct consequence of the replay-point cliff: post-pool replay means no encoder graph is live during
adaptation; raw replay means the full encoder forward+backward is.

### Step 3 reconciliation — hand totals agree to the byte
A1: persistent 16,374 B; transient R1 = (18+5+5+18)×4 = 184 B ✓. A4: persistent 16,208 B; transient
R1 = 39,582×4 = 158,328 B ✓; R3 = 21,156×2 = 42,312 B ✓. Full itemization in `e6_accounting_report.md`.

### Acceptance criteria (§3.5)
- [x] Six arms, three regimes present.
- [x] Automated & hand totals agree exactly (A1, A4).
- [x] Every arm persistent ≤ 16,384 B (no non-fitting arm).
- [x] R3 A4/A5 peak transient reported with compute multiplier (1.99×).
- [x] Zero use of "measured" in T1 outputs (verified).

### ⚠ ESCALATION ESC-1 (§3.6 failure criterion 2 / §8 escalation 2)
A4/A5 R3 peak transient = **41.3 KiB > ~24 KiB threshold** (FFN-hidden gradient driver). Recorded in
`BLOCKERS.md`. **Not a hard stop**: no result invalidated, no arm resized. It affects only the T3
limitation framing. Mitigating fact: the A0 inference arena is 33.0 KiB FP16, so A4/A5 adaptation
transient is only ≈1.25× the arena the device already provisions — the aliasing argument survives
*with the caveat stated*. **T3's §5.7 paragraph is held for owner sign-off; the R3 number will not
be softened.** T2 (independent) proceeds.

**T1 ACCEPTANCE: met.** Escalation ESC-1 logged for owner decision on T3 framing only.

---

---

## T2 — Pre-registration, MDE, equivalence testing  ✅ COMPLETE (escalation ESC-2)

### Gate (Step 2) — satisfied
- `preregistration/paper1_stat_plan.md` committed (391bc14), then `PREREG_LOCK.txt` committed (8dd4967).
- Lock UTC **2026-07-22T11:06:30Z** strictly precedes every T2 result file (all 11:14Z). ✅
- `scripts/run_equivalence_mde.py` asserts `sha256(plan)==lock` and aborts on mismatch (verified: "lock OK").

### Pseudoreplication audit (Step 3) — PASS
Every arm = 21 patients × exactly 5 seeds (630 cells). Paired tests operate on the 21-patient
intersection → **n=21 everywhere** (record-232 drop = 20, as declared). Only `scipy.stats` calls
(wilcoxon, rankdata, Student-t) take per-patient arrays of length 21. **No test uses 105 or 630.**
No violation; nothing to correct.

### Outputs (Step 4)
`results/e8_paired_tests.csv`, `e8_tost_equivalence.csv`, `e8_mde.csv`, `e8_mde_power_curves.csv`,
`e8_sensitivity_record232.csv`, `e8_sensitivity_mean_aggregation.csv`;
`manuscript/tables/table_V_paired_comparisons.tex`; `figures/paper/fig05_paired_differences.pdf`.
New primitive: `statistics/bca.py` (BCa; the released `statistics/bootstrap.py` is percentile-only).

### Headline results (pre-registered **median** aggregation, δ=0.02)
| Comparison | median Δ | Holm p | rank-biserial | 90% BCa CI | Equivalent | MDE@80% |
|---|---:|---:|---:|---|:--:|---:|
| A4–A1 | +0.0007 | 0.079 | 0.72 | [0.000, 0.024] | **No** (inconclusive) | 0.010 |
| A5–A1 | +0.0034 | 0.079 | 0.74 | [0.000, 0.029] | No | 0.010 |
| A4–A2 | +0.0044 | 0.079 | 0.63 | [0.000, 0.016] | **Yes** | 0.025 |
| A4–A3 | +0.0043 | 0.079 | 0.67 | [0.000, 0.037] | No | 0.015 |
| A4–A0 | +0.0401 | **0.007** | 0.91 | [0.001, 0.137] | No (superior) | 0.040 |
| A5–A0 | +0.0377 | **0.017** | 0.81 | [0.001, 0.140] | No (superior) | 0.040 |

**Interpretation (spec §4.7):** A4/A5 **beat the frozen model A0** (significant after Holm, large
effect) — headline superiority claim restricted to these. A4/A5 vs **A1** is **inconclusive**: not
superior after Holm (p=0.079) and not equivalent within δ=0.02 (CI upper 0.024/0.029 just exceeds the
margin). The 21-patient design can detect a median shift ≥ ~0.010–0.040; the observed A4–A1 median
(0.0007) is below its MDE. **A4 vs A2 is statistically equivalent within δ.**

### Sensitivities
- **(a) Record 232 — REVERSAL, reported prominently (spec §4.6):** dropping 232 flips **A4–A1** from
  *not equivalent* (CI upper 0.0240) to *equivalent* (CI upper 0.0198 ⊂ ±0.02). The A4–A1 equivalence
  verdict is fragile and 232-dependent; A5–A1 stays not-equivalent both ways; A4/A5-vs-A0 superiority
  is robust. This reversal MUST be stated in T3.
- **(b) Mean aggregation — reproduces the released E8 exactly** (Holm 0.2517 / 0.1097 / 0.0071 /
  0.0080). Confirms the pipeline reproduces published numbers to the digit; the old abstract's
  p=0.25 / 0.11 are the *mean-aggregation* values, now demoted to sensitivity (b).

### ⚠ ESCALATION ESC-2 (§8 escalation 4): re-derived stats differ from the abstract
Pre-registered **median** aggregation changes the headline: A4–A1 goes from mean-based Holm p=0.25 to
median-based Holm p=0.079 with a practically-zero median Δ (+0.0007), and the arm central tendency
shifts (A4 0.803→0.810, A5 0.809→0.812). **The conclusion structure is unchanged** (vs-A0 significant;
vs-A1 not superior after correction) and the tiny median Δ *supports* the frontier thesis, but the
specific numbers move. This is expected — the abstract predates the pre-registration and used mean
aggregation (now sensitivity b), reproduced exactly. **Owner decision needed:** confirm median as the
manuscript's primary (per the locked plan) before T3 rewrites the abstract numbers. See BLOCKERS.md ESC-2.

**T2 ACCEPTANCE: met.** All criteria satisfied; ESC-2 logged for owner confirmation of median-primary.

---

## T3 — Retitle and re-order the contribution  🟡 SUBSTANTIALLY COMPLETE

Defaults applied (all reversible; logged in BLOCKERS.md): **median** primary (spec §4.2.2, locked),
**A4–A1 = inconclusive/bounded** (spec §4.7 row 2). Owner may override ESC-1/ESC-2.

### Done
- **Title + subtitle + pdftitle + pdfkeywords** → cliff-led ("The Replay-Point Cliff: Why Pooled Tiny
  Transformers Break Latent-Replay Assumptions at 16 KiB"). `main.tex`.
- **Abstract rewritten** to §5.2 ordering (cliff → two-axis → plasticity → bounded frontier); null moved
  to position 4 as "inconclusive/bounded"; median numbers (0.666→0.810/0.812, 16/21 vs 12, Holm ≤0.017);
  software-only sentence preserved verbatim.
- **Citations — all 7 groups verified as REAL records (no ESC-5).** Already present: Pellegrini
  (`pellegrini2020latent`), Ravaglia (`ravaglia2021tinyml`), Busia (`busia2025tiny`). **Added 4** to
  `references.tex`: EpiSMART (Shahbazinia et al., arXiv:2509.13974), binary-network (Basso-Bert et al.,
  2503.07107), edge-anomaly benchmark (Barusco et al., 2604.06435), 2026 on-device-learning survey
  (Pavan et al., 2605.31226). Survey's backprop-memory quote verified verbatim (PDF line 1840).
  **Zero undefined citations** (checked cite-keys vs bibitems).
- **§2 related work:** EpiSMART biosignal-neighbour differentiation + binary/anomaly "frontier-in-vision"
  differentiation added.
- **§3 method:** new argued "Persistent versus transient state" paragraph with T1 R1/R2/R3 numbers,
  1.99× multiplier, ~300× asymmetry, 33 KiB inference-arena aliasing (ESC-1 caveat stated, not softened).
- **§4 protocol:** pre-registration paragraph (n=21, median primary, δ=0.02 + lock SHA-256, TOST 90% BCa,
  MDE, ablation as separate E9 family).
- **§6 discussion:** survey-refutation subsection (verified Pavan quote + T1 41.3 KiB ≈ 2.5× buffer);
  "17/21"→"16/21" count fixed to median.
- **§7 limitations:** memory limitation rewritten — transient now reported analytically under 3 regimes,
  aliasable-with-caveat (argued), while hardware remains unmeasured.
- **Tables/figures generated:** `table_V_paired_comparisons.tex` (median, TOST+equiv cols), `fig03`,
  `fig05` (±δ band), `fig13` (two-axis Pareto).
- **`results/MANUSCRIPT_VALUE_MAP.md`** built — every headline number → source file.

### Also done (second T3 pass)
- **§5 results:** wired in `table_V_paired_comparisons` (now carries `\label{tab:pairwise_tests}`,
  drop-in for the mean-based table); paired-tests paragraph rewritten to median primary with TOST +
  MDE + record-232 fragility; `fig05` (±δ band) replaces the old forest plot.
- **§10 appendix:** added "Two-axis byte accounting" subsection + `table_two_axis` (generated) + `fig13`.
- Figures `fig03/fig05/fig13` copied into `manuscript/figures/` (graphicspath resolves them).
- **ESC-2 resolution applied (reversible):** median governs the pre-registered **tests** (Table V,
  §5 tests, §6, §8 significance → Holm ≤0.017); **descriptive** arm means stay mean-of-means
  (0.803/0.809, 17/21) to match every released auto-generated table. Superseded Holm-p mentions
  (0.008) updated/deferred to Table V; no stray primary `p=0.25/0.11` remain in prose.
- **Final consistency sweep (this pass):** §6 bullet 2 still carried the stale mean-based frozen-model
  bound `Holm p≤0.008`; corrected to **≤0.017** to match §5/§8 and the median primary
  `e8_paired_tests.csv` (A4–A0 0.007, A5–A0 0.017). This was the last primary-headline straggler.
- **§7 leak-narrative `0.047 → 0.075` deliberately NOT converted to median.** It is a mean-based
  before/after of the 202-exclusion data-cleaning decision: `0.075` traces to
  `e8_sensitivity_mean_aggregation.csv` (A4–A3 Holm 0.0747) and legacy `primary/E8_paired_tests.csv`;
  `0.047` is the historical 22-record (leaked) value with no released file. Half-converting to
  `0.047 → 0.079` would splice a 22-record mean value onto a 21-record median value in one arrow —
  less honest than the coherent mean pair. The qualitative claim (excluding 202 removes A4–A3
  significance) holds under median too, since median A4–A3 Holm = 0.079 (>0.05). Left as-is by design.
- **Static checks pass:** all `\ref` resolve; `\begin`/`\end` balanced (46/46); all `\includegraphics`
  and `\input` targets exist; zero undefined citations; software-only sentence intact verbatim;
  abstract leads with the cliff; title cliff-led.

### Remaining (genuinely blocked)
1. **Compile clean** (`pdflatex`/`latexmk`): **no TeX Live on this host** — cannot run. Static checks
   above substitute for structure; a real compile is still required to confirm zero undefined refs and
   no new overfull hboxes.
2. **Owner sign-off** on ESC-1 (T3 §5.7 framing) and ESC-2 (median-governs-tests resolution).
3. **Optional (ESC-2 option B):** if the owner wants median as primary for **descriptive** stats too,
   regenerate `E7_arm_summary`/forest/effect sizes under median (0.810/0.812) — a wide cascade over
   released CSVs, deliberately NOT done unilaterally.

**T3 ACCEPTANCE: substantially met.** Cliff-led reordering, argued two-axis distinction, pre-registered
equivalence/MDE under a predating lock, and every edited number traced to `results/` are all in place
and internally consistent. Only a TeX-environment compile and owner ESC sign-off remain.

---

## Discrepancy log
- **DISC-1** (T0): abstract "17 of 21 patients against 12 for maximum head-only replay" conflates
  vs-A0 (17 improved) with vs-A1 (14/16 improved). Verify per-patient A1-vs-A0 improved count during
  T3 table regeneration; correct the sentence if 12 ≠ the A1-vs-A0 count.
- **DISC-2** (T2): released pipeline aggregates seeds by **mean**; pre-registration mandates **median**
  primary (mean → sensitivity b). Resolved: both computed; mean reproduces released E8 exactly.
- **DISC-3** (T2): released `statistics/bootstrap.py` is **percentile**; spec mandates **BCa** →
  implemented `statistics/bca.py`. Percentile reproduction left untouched.
- **DISC-4** (T2): abstract numbers (mean-based) differ from pre-registered median primary → see ESC-2.
- **ESC-1** (T1): A4/A5 R3 transient 41.3 KiB > 24 KiB → T3 framing. BLOCKERS.md.
- **ESC-2** (T2): median-primary changes headline numbers → owner confirms primary. BLOCKERS.md.

---

# Review Change Block 1 — Estimator Unification and Disclosure Fixes

Baseline `e52c5b5`. Locked estimator: `mean_of_medians` (median over seeds within
patient, then mean across patients). Escalations in `BLOCKERS.md` (ESC-3..ESC-6).

## R0 — Verification gate ✅
`scripts/r0_estimator_matrix.py` → `results/r0_estimator_matrix.csv`. Matrix matches
the review checksum to 4 dp (A0 0.6664 / A1 0.7852 / A4 0.8104 / A5 0.8115 under
`mean_of_medians`); improved counts 17/17/12/14 mean-within, **16/16/12/14**
median-within. **PASS** — review and repository agree on the raw data.

## R1 — Descriptive unification [BLOCKING] ✅
`scripts/r1_arm_summary_median.py` → `results/primary/E7_arm_summary_median.csv`
(supersedes `E7_arm_summary.csv`, logged in `SUPERSEDED.md`) + `results/r1_headline_values.csv`.
Root cause confirmed: `run_statistics.py:71` reduces seeds by **mean**; locked plan
uses **median**. `results/e8_*.csv` **unmodified** (verified via git). CI on the mean
of per-patient medians reuses `patient_bootstrap_ci`.

## R2 — Manuscript number sweep [BLOCKING] ✅ (primary) + ESC-4 (secondary)
- Abstract, §5 RQ2, §6, §8 → A4 **0.810**, A5 **0.811** (ESC-3: traceable single-round
  of 0.81147; review's 0.812 is a double-round), A0 0.666, A1 0.785; improved counts
  **16/21** (A1 stays 12/21). `table_primary_arms.tex` + `table_harm.tex` regenerated
  under median (`scripts/r1_regen_primary_tables.py`); harm counts updated
  (A2 harm 1→2, A3 2→3, A1 thresholded-improved 11→10, etc.).
- Secondary experiments (E9/E12/E15/E16/E17) still seeds-mean-averaged; every
  occurrence caption-labelled "seeds mean-averaged" + global §6 footnote. Full
  unification deferred (ESC-4): blocked by rule 2 (embedded secondary test p-values)
  and E9 missing per-seed data for B2/B3/B6/B7.
- `MANUSCRIPT_VALUE_MAP.md` rebuilt from `r1_headline_values.csv`.

## R3 — Skew reconciliation ✅
§5 paragraph after the direct-tests discussion: median paired diff +0.0007 vs arm gap
+0.025 (~36×) explained by right-skew (14 better / 3 tied / 4 worse; top-3 hold 61%);
BCa lower bound = 0.000 explained as 3 zeros + 4 negatives, not a clipping bug.

## R4 — Two-axis precision disclosure ✅ + R4.3 check recorded
Three ratios in §3, each labelled by denominator: survey refutation 2.58×
(42,312÷16,384, §6); aliasing same-precision **1.25×** (21,156÷16,896 elems);
aliasing deployed reality **2.50×** (42,312 B FP16 ÷ 16,896 B INT8).
**R4.3 deployed-precision check:** the accounting computes the A0 inference arena at
FP16 (33,792 B); the repo carries no explicit inference-precision config field, but
`replay_precision="int8"` (`gen_e6_accounting.py:220`, `e6_accounting_report.md:9`)
establishes an INT8 deployment pipeline, consistent with 16 KiB-class TinyML
convention. The 2.50× ratio names the FP16-adaptation-vs-INT8-inference assumption
explicitly. Escalation trigger 3 (deployed inference is NOT INT8) does **not** fire —
no evidence contradicts INT8; the FP16 arena figure is the matched-precision basis for
the 1.25× ratio.

## R5 — Rematerialization trade ✅
§6: block rematerialisation buys 1.87× peak-transient (79,164→42,312 B) at 1.993×
compute — value-neutral at this scale; FFN-hidden tensor+gradient dominate, so finer
granularity or topology change is needed, not block checkpointing. §7: on an
energy-constrained wearable this is a losing trade unless memory is the binding
constraint.

## R6 — Seed-stability asymmetry ✅
§5 descriptive paragraph: A1 mean↔median gap 0.043 (least stable), A4 0.026, A5 0.004
(invariant); framed descriptively, no test (source `r0_estimator_matrix.csv`).

## R7 — Record-232 fragility ✅
§5 (prominent, not footnote) + §7: full-cohort 90% upper 0.0240 (not equiv.) vs
drop-232 0.0198 (equiv.), margin **0.0002**, record 232 = +0.0428 (6th-largest);
verdict not robust either direction. Source `e8_sensitivity_record232.csv`.

## R8 — Responder structure [EXPLORATORY] ✅ + ESC-6
`scripts/r8_responder_exploratory.py` → `results/r8_responder_exploratory.csv`.
§6 subsection explicitly headed exploratory/post-hoc: 14/3/4 split, top-3 hold 61%;
no p-value, no hypothesis-testing vocabulary, absent from abstract. The minority-class
support correlation is **escalated (ESC-6)**: no per-record class-support file and no
raw `.atr` annotations in the repo; rule 3 forbids inventing it.

## R9 — Automated value-map guard ✅
`tests/test_manuscript_values.py` (5 tests): value-map source files exist; headline CSV
matches arm summary; abstract carries locked values; superseded values absent from
primary-headline files. **Acceptance verified**: passes now; fails when 0.803 is
reintroduced into the abstract. Wired into `Makefile` (`verify-values`, `verify-paper`)
and CI (`tests.yml` runs `pytest -v`).

## R10 — Repository hygiene ✅
`git mv statistics budget_stats` (stdlib-shadowing removed). Importers updated:
`run_statistics.py`, `run_equivalence_mde.py` (2), `r1_arm_summary_median.py`,
`tests/test_statistics.py`; `README.md` note added. `run_equivalence_mde.py` still
asserts the pre-registration hash (`assert_lock()`), prereg SHA-256 still matches
`PREREG_LOCK.txt`. Full suite: **91 passed**.

## Definition of done — status
One estimator (`mean_of_medians`) governs the primary cohort and all pre-registered
tests; every primary number traces to a file and `tests/test_manuscript_values.py`
enforces it. Skew, memory ratios, record-232, and responder structure disclosed at
true strength. **Open:** secondary-experiment unification (ESC-4), support correlation
(ESC-6), A5 3-dp convention (ESC-3), and a clean TeX compile (no TeX Live on host).
All edits uncommitted per "commit only when asked."
