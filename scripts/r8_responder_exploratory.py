#!/usr/bin/env python3
r"""R8 (EXPLORATORY / post-hoc) -- responder structure of the A4-A1 contrast.

NOT pre-registered. Hypothesis-generating only. No p-value is computed.

Writes results/r8_responder_exploratory.csv: per-patient median-within A4-A1
macro-F1 difference, sorted, with each patient's share of the total positive
mass and the running cumulative share. This is the traceable half of R8.

The minority-class-support correlation requested in the change spec (per-patient
A4-A1 gain vs per-patient S+V+F test-window support) is NOT computed here: no
per-record class-support file exists under results/, and deriving it requires the
raw PhysioNet beat annotations (.atr), which are absent from the repository. That
correlation is escalated in BLOCKERS.md (ESC-6); rule 3 forbids inventing it.
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "results" / "primary" / "E7_patient_seed_results.csv"
OUT = REPO / "results" / "r8_responder_exploratory.csv"


def main():
    per = defaultdict(lambda: defaultdict(list))
    with RAW.open() as f:
        for row in csv.DictReader(f):
            per[row["arm"]][row["record"]].append(float(row["macro_f1_present"]))
    med = {a: {r: float(np.median(v)) for r, v in d.items()} for a, d in per.items()}
    recs = sorted(med["A0"])
    diffs = {r: med["A4"][r] - med["A1"][r] for r in recs}

    pos_total = sum(d for d in diffs.values() if d > 0)
    ordered = sorted(diffs.items(), key=lambda x: -x[1])
    rows, cum = [], 0.0
    for rank, (r, d) in enumerate(ordered, 1):
        share = (d / pos_total) if d > 0 else 0.0
        cum += max(share, 0.0)
        rows.append({
            "record": r,
            "a4_minus_a1_median_within": round(d, 4),
            "sign": "better" if d > 0 else ("tied" if d == 0 else "worse"),
            "rank_from_top": rank,
            "share_of_positive_mass": round(share, 4),
            "cumulative_positive_share": round(cum, 4),
        })
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_better = sum(1 for d in diffs.values() if d > 0)
    n_tied = sum(1 for d in diffs.values() if d == 0)
    n_worse = sum(1 for d in diffs.values() if d < 0)
    top3 = sum(d for _, d in ordered[:3])
    print(f"wrote {OUT.relative_to(REPO)}")
    print(f"  better/tied/worse: {n_better}/{n_tied}/{n_worse}")
    print(f"  top-3 share of positive mass: {100*top3/pos_total:.1f}%")
    print(f"  range: {min(diffs.values()):+.4f} to {max(diffs.values()):+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
