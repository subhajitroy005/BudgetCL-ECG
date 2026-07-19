#!/usr/bin/env python3
"""Figure 7: forest plot of the six pre-specified paired comparisons.

Regenerated from ``results/primary/E8_paired_tests.csv`` so the drawn intervals
cannot drift from the released statistics. The previous version of this figure
was copied from an earlier analysis directory, which meant nothing verified that
its interval positions still matched the CSV -- the same class of defect that
left a stale confidence interval in the prose.

Only the two frozen-model comparisons survive Holm correction, so those are the
only ones drawn filled; the rest are open markers to keep the plot from
implying more than the statistics support.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import apply_style, save_figure  # noqa: E402

LABELS = {
    "A4_vs_A1": "A4 vs A1",
    "A5_vs_A1": "A5 vs A1",
    "A4_vs_A2": "A4 vs A2",
    "A4_vs_A3": "A4 vs A3",
    "A4_vs_A0": "A4 vs A0",
    "A5_vs_A0": "A5 vs A0",
}
ORDER = list(LABELS)


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(repo / "results/primary/E8_paired_tests.csv"))
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"error: {path} not found -- run `make statistics` first", file=sys.stderr)
        return 1

    with path.open() as fh:
        rows = {r["comparison"]: r for r in csv.DictReader(fh)}

    apply_style()
    fig, ax = plt.subplots(figsize=(6.6, 3.4))

    for i, name in enumerate(ORDER):
        if name not in rows:
            continue
        r = rows[name]
        mean = float(r["mean_difference"])
        lo, hi = float(r["ci_low"]), float(r["ci_high"])
        survives = float(r["p_holm"]) < 0.05
        y = len(ORDER) - 1 - i

        ax.plot([lo, hi], [y, y], color="black", linewidth=1.4, zorder=2)
        ax.plot(
            [mean], [y],
            marker="o", markersize=6, zorder=3,
            color="#1f77b4" if survives else "white",
            markeredgecolor="#1f77b4", markeredgewidth=1.4,
        )
        ax.text(
            hi + 0.006, y,
            f"[{lo:+.3f}, {hi:+.3f}]  Holm $p$={float(r['p_holm']):.3f}",
            va="center", fontsize=7.2,
            color="black" if survives else "0.35",
        )

    ax.axvline(0, color="0.4", linewidth=0.9, linestyle="--", zorder=1)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([LABELS[n] for n in reversed(ORDER)])
    ax.set_xlabel("paired patient macro-F1 difference (95% patient-bootstrap CI)")
    ax.set_title(
        "Pre-specified paired comparisons\n"
        "(filled = survives Holm correction; 21 patients, seeds averaged)",
        fontsize=9,
    )
    ax.set_xlim(-0.06, 0.42)
    ax.grid(axis="x", alpha=0.3)

    out = save_figure(fig, "fig_pairwise_effects")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
