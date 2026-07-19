# Statistics

## Patient bootstrap

```
1. average the five seeds WITHIN each patient   -> one score per patient
2. resample the 21 patients WITH replacement
3. recompute the arm mean or paired difference
4. repeat 10,000 times
5. report the 2.5th and 97.5th percentiles
```

Seeds are collapsed in step 1 and are **never resampled**. This is a
**patient bootstrap**. It is *not* called hierarchical — a hierarchical
patient-and-seed bootstrap would resample seeds *inside* each resampled patient,
which is a different procedure with different coverage.

Percentile intervals (not BCa). RNG seed fixed at 20260719, so intervals
reproduce to the digit.

## Paired tests

Two-sided Wilcoxon signed-rank, zeros dropped (`zero_method="wilcox"`).

Effect size is the matched-pairs rank-biserial correlation, computed from the
test statistic as `r_rb = 1 − 2W/(n(n+1)/2)`. That convention is what produced
every published value.

## Why a CI can span zero while the test is significant

Two rows in the paper do exactly this. It is not an error — the procedures target
**different estimands**:

- the bootstrap interval estimates the **mean** paired difference, which a few
  large negative patients can drag toward zero;
- the signed-rank test evaluates the distribution of signed **ranks**, and is
  sensitive to a consistent *direction* even when magnitudes are small.

Both affected rows show that pattern: median +0.003 to +0.008 with 16 of 20
patients improving, but a heavy left tail. Neither is claimed as a positive
result, since neither survives Holm.

## Multiplicity

Holm-Bonferroni over a **pre-specified six-member family**, fixed in the written
analysis plan before the final runs:

```
A4_vs_A1  A5_vs_A1  A4_vs_A2  A4_vs_A3  A4_vs_A0  A5_vs_A0
```

**Only `A4_vs_A0` and `A5_vs_A0` survive.** The family is not expanded post hoc.

One departure from a strict reading of "pre-specified" is disclosed: record 202
was excluded *after* a subject-identity audit and every affected analysis rerun.
That changed the cohort, not the methods.

## Equivalence testing (TOST)

Used for the E10 budget sweep, where the question is whether more budget changes
anything — and a non-significant difference is not evidence of equivalence.

Margin **δ = 0.05** macro-F1, fixed before analysis: roughly one third of the
A4−A0 effect (+0.136).

**The 18 TOST decisions are NOT multiplicity-adjusted**, and the bias runs toward
*over*-claiming equivalence. This is reported rather than quietly corrected;
`multiplicity_adjusted` is carried in the output so the policy travels with the
numbers.
