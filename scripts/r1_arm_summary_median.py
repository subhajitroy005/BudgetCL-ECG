#!/usr/bin/env python3
"""R1 -- Unify descriptive statistics on the locked estimator (Review Change Block 1).

The released results/primary/E7_arm_summary.csv reduces seeds within patient by
MEAN (run_statistics.py:71). The locked plan (paper1_stat_plan.md, SHA-256
3fddf1...54f5) reduces seeds within patient by MEDIAN. This regenerates the
DESCRIPTIVE arm summary under mean_of_medians so it matches the aggregation the
pre-registered tests already use. It does NOT touch results/e8_*.csv.

Emits:
  results/primary/E7_arm_summary_median.csv  -- central value = mean_of_medians
  results/r1_headline_values.csv             -- canonical headline set with sources
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from budget_stats.bootstrap import patient_bootstrap_ci  # noqa: E402

SRC = REPO / "results" / "primary" / "E7_patient_seed_results.csv"
ARM_OUT = REPO / "results" / "primary" / "E7_arm_summary_median.csv"
HEAD_OUT = REPO / "results" / "r1_headline_values.csv"
PRIMARY_ARMS = ["A0", "A1", "A2", "A3", "A4", "A5"]


def load_median_within_patient():
    """{arm: {record: median-over-seeds macro_f1}}."""
    per = defaultdict(lambda: defaultdict(list))
    with SRC.open() as f:
        for row in csv.DictReader(f):
            per[row["arm"]][row["record"]].append(float(row["macro_f1_present"]))
    return {a: {r: float(np.median(v)) for r, v in d.items()} for a, d in per.items()}


def improved_count(scores, hi, lo):
    recs = sorted(set(scores[hi]) & set(scores[lo]))
    return sum(1 for r in recs if scores[hi][r] > scores[lo][r]), len(recs)


def main():
    scores = load_median_within_patient()

    rows = []
    for arm in PRIMARY_ARMS:
        vals = np.array([scores[arm][r] for r in sorted(scores[arm])], dtype=float)
        lo, hi = patient_bootstrap_ci(vals)  # percentile CI on the mean, resampling patients
        rows.append({
            "arm": arm,
            "n_patients": len(vals),
            "macro_f1_mean_of_medians": round(float(vals.mean()), 4),
            "sd": round(float(vals.std(ddof=1)), 4),
            "median_of_medians": round(float(np.median(vals)), 4),
            "ci_low": round(lo, 4),
            "ci_high": round(hi, 4),
        })
    with ARM_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {ARM_OUT.relative_to(REPO)}")
    for r in rows:
        print(f"  {r['arm']}: {r['macro_f1_mean_of_medians']:.4f}  "
              f"CI [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")

    central = {r["arm"]: r["macro_f1_mean_of_medians"] for r in rows}
    a4a0, _ = improved_count(scores, "A4", "A0")
    a5a0, _ = improved_count(scores, "A5", "A0")
    a1a0, _ = improved_count(scores, "A1", "A0")
    a4a1, n = improved_count(scores, "A4", "A1")

    headline = [
        ("A0_per_patient_macro_f1", f"{central['A0']:.3f}", "results/primary/E7_arm_summary_median.csv", "A0 macro_f1_mean_of_medians"),
        ("A1_max_head_only_replay", f"{central['A1']:.3f}", "results/primary/E7_arm_summary_median.csv", "A1 macro_f1_mean_of_medians"),
        ("A4_rank1_encoder_lora", f"{central['A4']:.3f}", "results/primary/E7_arm_summary_median.csv", "A4 macro_f1_mean_of_medians"),
        ("A5_rank2_encoder_lora", f"{central['A5']:.3f}", "results/primary/E7_arm_summary_median.csv", "A5 macro_f1_mean_of_medians"),
        ("A4_patients_improved_vs_A0", f"{a4a0} of {n}", "results/r0_estimator_matrix.csv", "median-within improved count"),
        ("A5_patients_improved_vs_A0", f"{a5a0} of {n}", "results/r0_estimator_matrix.csv", "median-within improved count"),
        ("A1_patients_improved_vs_A0", f"{a1a0} of {n}", "results/r0_estimator_matrix.csv", "median-within improved count"),
        ("A4_patients_improved_vs_A1", f"{a4a1} of {n}", "results/r0_estimator_matrix.csv", "median-within improved count"),
    ]
    with HEAD_OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value", "source_file", "source_locator"])
        w.writerows(headline)
    print(f"\nwrote {HEAD_OUT.relative_to(REPO)}")
    for k, v, sf, _ in headline:
        print(f"  {k} = {v}   [{sf}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
