#!/usr/bin/env python3
"""Figure 13: boundary-leakage sensitivity (E17).

Whole-record versus split-first arm means. The point of the figure is that the
shifts are small AND MIXED IN SIGN -- recovered leakage would have produced a
systematic drop.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import apply_style, save_figure  # noqa: E402

ARMS = ["A0", "A1", "A4", "A5"]


def read_means(path: Path) -> dict[str, float]:
    with path.open() as fh:
        return {r["arm"]: float(r["mean"]) for r in csv.DictReader(fh)}


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", default=str(repo / "results/primary/E7_arm_summary.csv"))
    parser.add_argument(
        "--splitfirst",
        default=str(repo / "results/preprocessing_sensitivity/E17_arm_summary.csv"),
    )
    args = parser.parse_args()

    for p in (args.primary, args.splitfirst):
        if not Path(p).exists():
            print(f"error: {p} not found -- run `make statistics` first", file=sys.stderr)
            return 1

    base, split = read_means(Path(args.primary)), read_means(Path(args.splitfirst))
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.4), width_ratios=[1.4, 1])

    x = np.arange(len(ARMS))
    ax1.bar(x - 0.19, [base[a] for a in ARMS], 0.38, label="whole-record (released)")
    ax1.bar(x + 0.19, [split[a] for a in ARMS], 0.38, label="split-first (E17)")
    ax1.set_xticks(x, ARMS)
    ax1.set_ylabel("per-patient macro-F1")
    ax1.set_ylim(0, 1.0)
    ax1.legend(fontsize=7.5, frameon=False)
    ax1.set_title("Arm means under both pipelines")
    ax1.grid(axis="y", alpha=0.3)

    deltas = [split[a] - base[a] for a in ARMS]
    ax2.barh(ARMS, deltas, color=["#2ca02c" if d >= 0 else "#d62728" for d in deltas])
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel(r"$\Delta$ macro-F1 (split-first $-$ whole-record)")
    ax2.set_xlim(-0.02, 0.02)
    ax2.set_title("Shifts are small and mixed in sign")
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle("Removing the boundary filter dependency changes no conclusion", fontsize=10)
    out = save_figure(fig, "fig_splitfirst_sensitivity")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
