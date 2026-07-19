#!/usr/bin/env python3
"""Figure 4: per-patient macro-F1 distribution across arms.

Regenerates one figure from released CSVs only -- no re-training, no raw data.
Each point is ONE PATIENT (five seeds already averaged), which is the unit the
statistics use.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import apply_style, save_figure  # noqa: E402

ARMS = ["A0", "A1", "A2", "A3", "A4", "A5"]
LABELS = {
    "A0": "A0\nfrozen", "A1": "A1\nhead", "A2": "A2\nadapter r1",
    "A3": "A3\nadapter r2", "A4": "A4\nLoRA r1", "A5": "A5\nLoRA r2",
}


def load(path: Path) -> dict[str, dict[str, float]]:
    """Per-arm, per-patient scores from the released summary CSV."""
    per: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open() as fh:
        for row in csv.DictReader(fh):
            per[row["arm"]][row["record"]] = float(row["macro_f1_present"])
    return per


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default=str(repo / "results" / "primary" / "E7_patient_summary.csv")
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"error: {path} not found -- run `make statistics` first", file=sys.stderr)
        return 1

    scores = load(path)
    apply_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    records = sorted(scores["A0"])
    data = [[scores[a][r] for r in records] for a in ARMS]

    ax.boxplot(data, tick_labels=[LABELS[a] for a in ARMS], showfliers=False,
               medianprops={"color": "black"})
    # Grey lines follow one patient across arms: the spread is patient-driven,
    # and the mean is carried by a minority of patients.
    for r in records:
        ax.plot(range(1, len(ARMS) + 1), [scores[a][r] for a in ARMS],
                color="0.7", linewidth=0.5, alpha=0.6, zorder=0)
    for i, a in enumerate(ARMS, start=1):
        ax.scatter([i] * len(records), [scores[a][r] for r in records],
                   s=10, color="#1f77b4", zorder=3)

    ax.set_ylabel("per-patient macro-F1 (N/S/V, present classes)")
    ax.set_title("Per-patient adaptation performance\n"
                 "(21-record patient-disjoint cohort, 5 seeds averaged per patient)")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.3)

    out = save_figure(fig, "fig_patient_distribution")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
