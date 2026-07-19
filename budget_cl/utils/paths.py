"""Repository-relative path helpers.

Scripts are run from several working directories (repo root, ``experiments/``,
CI). Resolving paths against the repository root rather than the CWD keeps
config references and output locations stable.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["repo_root", "results_dir", "configs_dir", "manifests_dir", "figures_dir"]

# This file is budget_cl/utils/paths.py, so the root is three levels up.
_ROOT = Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    """Absolute path to the repository root."""
    return _ROOT


def results_dir() -> Path:
    """Machine-readable released results."""
    return _ROOT / "results"


def configs_dir() -> Path:
    """Frozen experiment configurations."""
    return _ROOT / "configs"


def manifests_dir() -> Path:
    """Record, subject and replay manifests."""
    return _ROOT / "manifests"


def figures_dir() -> Path:
    """Figure scripts and released figure PDFs."""
    return _ROOT / "figures"
