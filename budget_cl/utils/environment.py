"""Environment capture for reproducibility.

Written beside every experiment run so a result can be attributed to a specific
interpreter, library set, and commit.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any

__all__ = ["environment_report", "git_commit"]


def git_commit(short: bool = False) -> str:
    """Current commit hash, or ``"unknown"`` outside a git checkout."""
    cmd = ["git", "rev-parse", "--short" if short else "HEAD"]
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def environment_report() -> dict[str, Any]:
    """Interpreter, platform, key library versions, and git commit."""
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_commit": git_commit(),
    }
    for module in ("numpy", "scipy", "pandas", "sklearn", "tensorflow", "wfdb", "matplotlib"):
        try:
            report[module] = __import__(module).__version__
        except Exception:  # noqa: BLE001 - a missing optional dep is not fatal here
            report[module] = "not installed"
    return report
