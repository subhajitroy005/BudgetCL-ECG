#!/usr/bin/env python3
"""R0 -- Verification baseline (Review Change Block 1).

Reproduce the four aggregation orders from the raw per-(record,seed,arm) macro-F1
so the estimator defect is independently verifiable before it is fixed.

  mean_of_means     : mean over seeds within patient, then mean over patients
  mean_of_medians   : median over seeds within patient, then mean over patients  [LOCKED PLAN]
  median_of_medians : median over seeds, then median over patients
  median_of_means   : mean over seeds, then median over patients

Also emits patients-improved counts (mean- and median-within-patient) for the
paired contrasts A4-A0, A5-A0, A1-A0, A4-A1.

Writes results/r0_estimator_matrix.csv and prints a checksum comparison.
"""
import csv
import statistics as stdlib_stats
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "results" / "primary" / "E7_patient_seed_results.csv"
OUT = REPO / "results" / "r0_estimator_matrix.csv"

# checksum table from the review (treated as a checksum, not as input)
CHECKSUM = {
    "A0": (0.6664, 0.6664, 0.6665, 0.6665),
    "A1": (0.7858, 0.7852, 0.7432, 0.7470),
    "A4": (0.8028, 0.8104, 0.7767, 0.7597),
    "A5": (0.8089, 0.8115, 0.8126, 0.7860),
}
COUNT_CHECKSUM = {  # (mean_within, median_within)
    "A4_vs_A0": (17, 16),
    "A5_vs_A0": (17, 16),
    "A1_vs_A0": (12, 12),
    "A4_vs_A1": (14, 14),
}


def load():
    # per (arm, record) -> list of per-seed macro_f1
    d = defaultdict(lambda: defaultdict(list))
    with SRC.open() as f:
        for row in csv.DictReader(f):
            d[row["arm"]][row["record"]].append(float(row["macro_f1_present"]))
    return d


def mean(xs):
    return stdlib_stats.fmean(xs)


def median(xs):
    return stdlib_stats.median(xs)


def estimators(per_record):
    records = sorted(per_record)
    per_patient_mean = [mean(per_record[r]) for r in records]
    per_patient_median = [median(per_record[r]) for r in records]
    return {
        "mean_of_means": mean(per_patient_mean),
        "mean_of_medians": mean(per_patient_median),
        "median_of_medians": median(per_patient_median),
        "median_of_means": median(per_patient_mean),
    }


def improved_counts(data, hi, lo):
    """count patients where hi > lo, aggregating seeds within patient by mean and by median."""
    records = sorted(set(data[hi]) & set(data[lo]))
    mean_ct = median_ct = 0
    for r in records:
        if mean(data[hi][r]) > mean(data[lo][r]):
            mean_ct += 1
        if median(data[hi][r]) > median(data[lo][r]):
            median_ct += 1
    return mean_ct, median_ct, len(records)


def main():
    data = load()
    rows = []
    ok = True
    print(f"{'arm':<4} {'mean_of_means':>14} {'mean_of_medians':>16} "
          f"{'median_of_medians':>18} {'median_of_means':>16}")
    for arm in ["A0", "A1", "A2", "A3", "A4", "A5"]:
        est = estimators(data[arm])
        rows.append({"arm": arm, **{k: f"{v:.6f}" for k, v in est.items()}})
        vals = (est["mean_of_means"], est["mean_of_medians"],
                est["median_of_medians"], est["median_of_means"])
        print(f"{arm:<4} {vals[0]:>14.4f} {vals[1]:>16.4f} {vals[2]:>18.4f} {vals[3]:>16.4f}")
        if arm in CHECKSUM:
            for got, exp in zip(vals, CHECKSUM[arm]):
                if round(got, 4) != exp:
                    ok = False
                    print(f"   MISMATCH {arm}: got {got:.4f} expected {exp:.4f}")

    OUT.write_text("")
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "mean_of_means", "mean_of_medians",
                                          "median_of_medians", "median_of_means"])
        w.writeheader()
        w.writerows(rows)

    print("\nPatients improved (mean_within / median_within / n):")
    for label, (hi, lo) in [("A4_vs_A0", ("A4", "A0")), ("A5_vs_A0", ("A5", "A0")),
                            ("A1_vs_A0", ("A1", "A0")), ("A4_vs_A1", ("A4", "A1"))]:
        mc, mdc, n = improved_counts(data, hi, lo)
        exp = COUNT_CHECKSUM[label]
        flag = "" if (mc, mdc) == exp else f"  MISMATCH expected {exp}"
        print(f"  {label}: mean {mc}/{n}  median {mdc}/{n}{flag}")
        if (mc, mdc) != exp:
            ok = False

    print(f"\nR0 checksum: {'PASS' if ok else 'FAIL'}")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
