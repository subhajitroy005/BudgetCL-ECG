# Metrics

## Primary metric: per-patient macro-F1 over present classes

```
C_p       = { c ∈ {N,S,V} : n_(p,c) ≥ 1 }
macroF1_p = (1/|C_p|) Σ_{c ∈ C_p} F1_(p,c)
```

Rules, all of which change the number:

- classes **absent** from a patient's test segment are **omitted**, not scored
  zero. Scoring them zero reads ~0.15 lower (A0 = 0.509 instead of 0.666);
- F is excluded from the primary patient metric (reported separately, pooled);
- Q is excluded entirely;
- empty-denominator F1 uses `zero_division=0`;
- recomputed from the saved 5×5 **confusion matrix**, never read from a stored
  scalar field.

This is the single easiest thing to get wrong in this project. Use
`budget_cl.evaluation.macro_f1_present`.

## Unit of analysis

The **patient**, not the beat and not the seed. Five seeds are averaged within a
patient *before* any test, so the bootstrap resamples patients.

For external databases the unit is the **subject**: INCART's 11 evaluated
recordings are only 6 unique subjects, and treating recordings as independent
would inflate the sample size.

Aggregating to subjects fixes an *independence* defect. It does **not** make the
protocol longitudinal — adaptation is still record-wise with the model reset
between recordings.

## Source-domain change

```
Δ_src = source_after − source_before
```

Positive means improvement; negative means source loss. Every source-domain
number in the paper is negative, so −0.027 is a **smaller** loss than −0.136.

The word "forgetting" is avoided for this quantity: the two common conventions
differ in sign, and "source-domain macro-F1 change" describes what is measured
(a change in a score) without asserting a mechanism.

## Harm analysis

Adaptation is not uniformly beneficial, and cohort means hide that. Both
thresholds are reported because the choice changes the story:

- **threshold-free:** encoder arms harm 4/21, A1 harms 8/21 — encoder looks safer;
- **ε = 0.02** (fixed before analysis): A1 harms **0**, encoder arms harm 2 each.

Most of the apparent safety advantage of the encoder arms is sub-threshold noise.
What actually separates them is the size of the tail loss, not its frequency.
