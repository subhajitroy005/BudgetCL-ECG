#!/usr/bin/env python3
"""Regenerate every statistical output from the released per-cell results.

This is step 3 of the reproduction chain::

    frozen YAML -> experiment runner -> patient/seed CSV
                                        ^^^^^^^^^^^^^^^^ (input here)
      -> STATISTICS  -> table/figure generator -> LaTeX manuscript

It reads only ``results/**/E*_patient_seed_results.csv`` and writes summary and
inferential CSVs. It never re-trains, so it runs in seconds on any machine and
needs neither a GPU nor the PhysioNet recordings.

Outputs:
    results/primary/E7_patient_summary.csv       per-arm, per-patient scores
    results/primary/E7_arm_summary.csv           arm means and bootstrap CIs
    results/primary/E8_paired_tests.csv          six pre-specified comparisons
    results/preprocessing_sensitivity/E17_arm_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from budget_stats import (  # noqa: E402
    PRESPECIFIED_FAMILY,
    holm_correct,
    paired_bootstrap_ci,
    paired_test,
    patient_bootstrap_ci,
)

from budget_cl.utils import get_logger, repo_root  # noqa: E402

LOG = get_logger("run_statistics")

PRIMARY_ARMS = ["A0", "A1", "A2", "A3", "A4", "A5"]
SPLIT_FIRST_ARMS = ["A0", "A1", "A4", "A5"]

# E17 sensitivity contrasts. These are a SEPARATE four-member family from the
# six pre-specified primary comparisons and are Holm-corrected within
# themselves; they are not pooled with the primary family.
SPLIT_FIRST_CONTRASTS = [("A4", "A0"), ("A5", "A0"), ("A4", "A1"), ("A5", "A1")]


def read_patient_seed_csv(path: Path) -> dict[str, dict[str, float]]:
    """Load a patient/seed CSV and average seeds within each patient.

    Seed averaging happens HERE, once, so every downstream statistic operates
    on one score per patient and the bootstrap resamples patients rather than
    treating seeds as independent observations.

    Returns:
        ``{arm: {record: mean_macro_f1_present}}``.
    """
    per: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open() as fh:
        for row in csv.DictReader(fh):
            value = row["macro_f1_present"]
            if value == "":
                continue
            per[row["arm"]][row["record"]].append(float(value))
    return {a: {r: float(np.mean(v)) for r, v in d.items()} for a, d in per.items()}


def write_csv(path: Path, rows: list[dict], header: list[str]) -> None:
    """Write rows to CSV, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        # csv defaults to CRLF; force LF so released CSVs are byte-stable.
        writer = csv.DictWriter(fh, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    LOG.info("wrote %s (%d rows)", path.relative_to(repo_root()), len(rows))


def arm_summary(scores: dict[str, dict[str, float]], arms: list[str]) -> list[dict]:
    """Per-arm mean, SD, median, and patient-bootstrap CI."""
    rows = []
    for arm in arms:
        if arm not in scores:
            continue
        records = sorted(scores[arm])
        values = np.array([scores[arm][r] for r in records])
        lo, hi = patient_bootstrap_ci(values)
        rows.append(
            {
                "arm": arm,
                "n_patients": len(values),
                "mean": round(float(values.mean()), 4),
                "sd": round(float(values.std(ddof=1)), 4),
                "median": round(float(np.median(values)), 4),
                "ci_low": round(lo, 4),
                "ci_high": round(hi, 4),
            }
        )
    return rows


def paired_family(scores: dict[str, dict[str, float]]) -> list[dict]:
    """Run the six pre-specified comparisons and apply Holm correction."""
    records = sorted(set.intersection(*(set(scores[a]) for a in PRIMARY_ARMS if a in scores)))
    rows, raw_p = [], {}

    for name in PRESPECIFIED_FAMILY:
        left, right = name.split("_vs_")
        if left not in scores or right not in scores:
            LOG.warning("skipping %s: missing arm", name)
            continue
        a = np.array([scores[left][r] for r in records])
        b = np.array([scores[right][r] for r in records])
        result = paired_test(a, b)
        lo, hi = paired_bootstrap_ci(a, b)
        raw_p[name] = result.p_value
        rows.append(
            {
                "comparison": name,
                "n_pairs": result.n,
                "mean_difference": round(result.mean_difference, 4),
                "median_difference": round(result.median_difference, 4),
                "ci_low": round(lo, 4),
                "ci_high": round(hi, 4),
                "p_wilcoxon": round(result.p_value, 6),
                "p_holm": None,  # filled below
                "rank_biserial": round(result.effect_size, 3),
                "improved": result.improved,
                "worsened": result.worsened,
            }
        )

    adjusted = holm_correct(raw_p)
    for row in rows:
        row["p_holm"] = round(adjusted[row["comparison"]], 6)
    return rows


def split_first_contrasts(scores: dict[str, dict[str, float]]) -> list[dict]:
    """Paired contrasts on the split-first cohort.

    Both raw and Holm-adjusted p-values are reported. The Holm correction is
    applied across these FOUR sensitivity contrasts only -- they are not part
    of the pre-specified primary family and are not pooled with it.
    """
    available = [a for a in SPLIT_FIRST_ARMS if a in scores]
    if len(available) < 2:
        return []
    records = sorted(set.intersection(*(set(scores[a]) for a in available)))
    rows, raw_p = [], {}

    for left, right in SPLIT_FIRST_CONTRASTS:
        if left not in scores or right not in scores:
            continue
        a = np.array([scores[left][r] for r in records])
        b = np.array([scores[right][r] for r in records])
        result = paired_test(a, b)
        lo, hi = paired_bootstrap_ci(a, b)
        name = f"{left}_vs_{right}"
        raw_p[name] = result.p_value
        rows.append(
            {
                "comparison": name,
                "n_pairs": result.n,
                "mean_difference": round(result.mean_difference, 4),
                "median_difference": round(result.median_difference, 4),
                "ci_low": round(lo, 4),
                "ci_high": round(hi, 4),
                "p_wilcoxon_raw": round(result.p_value, 6),
                "p_holm_within_sensitivity_family": None,
                "rank_biserial": round(result.effect_size, 3),
                "improved": result.improved,
                "worsened": result.worsened,
            }
        )

    adjusted = holm_correct(raw_p)
    for row in rows:
        row["p_holm_within_sensitivity_family"] = round(adjusted[row["comparison"]], 6)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default=str(repo_root() / "results"),
        help="Directory holding the released result CSVs.",
    )
    args = parser.parse_args()
    results = Path(args.results_dir)

    primary_csv = results / "primary" / "E7_patient_seed_results.csv"
    if not primary_csv.exists():
        LOG.error("missing %s -- run `make run-primary` or restore the released CSVs", primary_csv)
        return 1

    scores = read_patient_seed_csv(primary_csv)

    write_csv(
        results / "primary" / "E7_patient_summary.csv",
        [
            {"arm": a, "record": r, "macro_f1_present": round(v, 6)}
            for a in PRIMARY_ARMS
            if a in scores
            for r, v in sorted(scores[a].items())
        ],
        ["arm", "record", "macro_f1_present"],
    )
    write_csv(
        results / "primary" / "E7_arm_summary.csv",
        arm_summary(scores, PRIMARY_ARMS),
        ["arm", "n_patients", "mean", "sd", "median", "ci_low", "ci_high"],
    )
    write_csv(
        results / "primary" / "E8_paired_tests.csv",
        paired_family(scores),
        [
            "comparison", "n_pairs", "mean_difference", "median_difference",
            "ci_low", "ci_high", "p_wilcoxon", "p_holm", "rank_biserial",
            "improved", "worsened",
        ],
    )

    split_csv = results / "preprocessing_sensitivity" / "E17_patient_seed_results.csv"
    if split_csv.exists():
        split_scores = read_patient_seed_csv(split_csv)
        write_csv(
            results / "preprocessing_sensitivity" / "E17_arm_summary.csv",
            arm_summary(split_scores, SPLIT_FIRST_ARMS),
            ["arm", "n_patients", "mean", "sd", "median", "ci_low", "ci_high"],
        )
        write_csv(
            results / "preprocessing_sensitivity" / "E17_paired_tests.csv",
            split_first_contrasts(split_scores),
            [
                "comparison", "n_pairs", "mean_difference", "median_difference",
                "ci_low", "ci_high", "p_wilcoxon_raw",
                "p_holm_within_sensitivity_family", "rank_biserial",
                "improved", "worsened",
            ],
        )
    else:
        LOG.warning("E17 results not found at %s; skipping sensitivity summary", split_csv)

    LOG.info("statistics complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
