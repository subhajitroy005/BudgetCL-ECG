"""Shared Matplotlib style for all publication figures.

Type 3 fonts are rejected by many publishers and are the most common reason a
figure fails a submission check, so TrueType (type 42) is forced here in ONE
place rather than repeated per script.

Import this module before creating any figure::

    from _style import apply_style, save_figure
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

__all__ = ["apply_style", "save_figure", "PAPER_DIR"]

PAPER_DIR = Path(__file__).resolve().parents[1] / "paper"


def apply_style() -> None:
    """Set publication defaults, including TrueType font embedding."""
    mpl.rcParams["pdf.fonttype"] = 42   # TrueType, NOT Type 3
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.size"] = 9
    mpl.rcParams["axes.titlesize"] = 10
    mpl.rcParams["axes.labelsize"] = 9
    mpl.rcParams["figure.dpi"] = 150
    mpl.rcParams["savefig.bbox"] = "tight"
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False


def save_figure(fig, name: str, out_dir: Path | None = None) -> Path:
    """Save a figure as PDF into ``figures/paper/`` and return its path."""
    out_dir = out_dir or PAPER_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (name if name.endswith(".pdf") else f"{name}.pdf")
    fig.savefig(path)
    return path
