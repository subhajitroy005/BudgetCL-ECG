#!/usr/bin/env python3
"""R1/R2 -- regenerate the PRIMARY-cohort descriptive tables under the locked
estimator (median over seeds within patient, then across patients).

Emits, in the exact LaTeX format of the originals:
  manuscript/tables/table_primary_arms.tex   (arm summary; supersedes mean version)
  manuscript/tables/table_harm.tex           (harm profile; supersedes mean version)

Central values and per-patient gains G_p use median-within-patient reduction,
identical to the aggregation the pre-registered tests (results/e8_*.csv) already
use. This script does NOT touch any e8_*.csv or any secondary-experiment table.
Source of truth: results/primary/E7_patient_seed_results.csv (raw per-seed) and
results/primary/E7_arm_summary_median.csv (arm summary + bootstrap CI, from R1).
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "results" / "primary" / "E7_patient_seed_results.csv"
SUMMARY = REPO / "results" / "primary" / "E7_arm_summary_median.csv"
TAB = REPO / "manuscript" / "tables"
EPS = 0.02
ARMS = ["A0", "A1", "A2", "A3", "A4", "A5"]
ARM_LABELS = {
    "A0": "A0 (frozen)", "A1": r"A1 (head $+$ max replay)",
    "A2": r"A2 (post-pool $r{=}1$)", "A3": r"A3 (post-pool $r{=}2$)",
    "A4": r"A4 (LoRA $r{=}1$)", "A5": r"A5 (LoRA $r{=}2$)",
}


def load_median():
    per = defaultdict(lambda: defaultdict(list))
    with RAW.open() as f:
        for row in csv.DictReader(f):
            per[row["arm"]][row["record"]].append(float(row["macro_f1_present"]))
    return {a: {r: float(np.median(v)) for r, v in d.items()} for a, d in per.items()}


def primary_arms(summary_rows):
    cap = (r"\caption{Per-patient \macrof{} over the 21-record patient-disjoint "
           r"cohort, five seeds \textbf{median-aggregated within patient} (the "
           r"pre-registered locked estimator). Intervals are patient bootstraps "
           r"($10{,}000$ resamples).}")
    lines = [r"% GENERATED FILE -- do not edit by hand.",
             r"% Regenerate with: python3 scripts/r1_regen_primary_tables.py",
             r"\begin{table}[H]", r"\centering", r"\small", cap,
             r"\label{tab:primary_arms}",
             r"\begin{tabular}{lrrrl}", r"\toprule",
             r"Arm & Mean & SD & Median & 95\% CI \\", r"\midrule"]
    for r in summary_rows:
        lines.append(
            f"{ARM_LABELS[r['arm']]} & ${float(r['macro_f1_mean_of_medians']):.3f}$ & "
            f"${float(r['sd']):.3f}$ & ${float(r['median_of_medians']):.3f}$ & "
            f"$[{float(r['ci_low']):.3f}, {float(r['ci_high']):.3f}]$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def harm(med):
    recs = sorted(med["A0"])

    def bold(s):
        return rf"\textbf{{{s}}}"

    cap = (r"\caption{Patient-harm profile on the 21-record patient-disjoint cohort, "
           r"seeds \textbf{median-aggregated within patient} (the pre-registered "
           r"locked estimator). $G_p$ is the per-patient change in the primary "
           r"metric against the frozen model A0. Improved / unchanged / harmed use a "
           r"meaningful-change threshold $\epsilon = 0.02$ stated before analysis, and "
           r"$R_{\mathrm{harm}} = |\{p : G_p < -\epsilon\}| / P$. ``Impr.\ ($>0$)'' "
           r"repeats the threshold-free count from Table~\ref{tab:primary_arms}.}")
    lines = [r"\begin{table}[H]", r"\centering", r"\small", cap,
             r"\label{tab:harm}", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
             (r"Arm & Mean $G_p$ & Med.\ $G_p$ & Impr. & Unch. & Harmed & "
              r"$R_{\mathrm{harm}}$ & Min $G_p$ & 5th pct. & Max $G_p$ & "
              r"Impr.\ ($>0$) \\"),
             r"\midrule"]
    for a in ["A1", "A2", "A3", "A4", "A5"]:
        g = np.array([med[a][r] - med["A0"][r] for r in recs])
        impr_e = int((g > EPS).sum())
        unch = int((np.abs(g) <= EPS).sum())
        harm_e = int((g < -EPS).sum())
        impr_0 = int((g > 0).sum())
        rharm = harm_e / len(g)
        cells = [
            a,
            f"${g.mean():+.3f}$",
            f"${np.median(g):+.3f}$",
            bold(impr_e) if a in ("A4", "A5") else str(impr_e),
            str(unch),
            bold(harm_e) if a == "A1" else str(harm_e),
            bold(f"{rharm:.3f}") if a == "A1" else f"{rharm:.3f}",
            f"${g.min():+.3f}$" if a != "A4" else rf"$\mathbf{{{g.min():+.3f}}}$",
            f"${np.percentile(g, 5):+.3f}$",
            f"${g.max():+.3f}$",
            bold(impr_0) if a in ("A4", "A5") else str(impr_0),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main():
    with SUMMARY.open() as f:
        summary_rows = list(csv.DictReader(f))
    med = load_median()

    (TAB / "table_primary_arms.tex").write_text(primary_arms(summary_rows))
    (TAB / "table_harm.tex").write_text(harm(med))
    print("regenerated table_primary_arms.tex and table_harm.tex under median")
    for r in summary_rows:
        print(f"  {r['arm']}: mean_of_medians={float(r['macro_f1_mean_of_medians']):.3f} "
              f"median_of_medians={float(r['median_of_medians']):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
