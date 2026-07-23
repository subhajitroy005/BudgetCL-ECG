"""Statistical analysis: bootstrap, Wilcoxon, Holm, TOST, effect sizes.

Deliberately separate from :mod:`budget_cl` so the inferential layer can be
audited on its own, and so re-running statistics never requires TensorFlow.

.. warning::

   This package is named ``statistics`` and therefore SHADOWS Python's standard
   library module of the same name for any process that puts the repository
   root on ``sys.path``. That is safe here because:

   * it is **not** installed as a distribution package (see
     ``[tool.setuptools.packages.find]`` in ``pyproject.toml``), so
     ``pip install -e .`` never shadows the stdlib environment-wide;
   * ``budget_cl`` and ``preprocessing`` never import it;
   * only repo-local scripts and tests use it.

   If you need the stdlib module in this repository, import it as
   ``importlib.import_module`` from a context without the repo root on the
   path, or use ``numpy``/``scipy`` equivalents -- which is what this code does.
"""

from __future__ import annotations

from .bootstrap import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    paired_bootstrap_ci,
    patient_bootstrap_ci,
)
from .effect_sizes import rank_biserial
from .holm import PRESPECIFIED_FAMILY, holm_correct
from .tost import EQUIVALENCE_MARGIN, TostResult, tost_paired
from .wilcoxon import PairedTestResult, paired_test

__all__ = [
    "BOOTSTRAP_ITERATIONS",
    "BOOTSTRAP_SEED",
    "EQUIVALENCE_MARGIN",
    "PRESPECIFIED_FAMILY",
    "PairedTestResult",
    "TostResult",
    "holm_correct",
    "paired_bootstrap_ci",
    "paired_test",
    "patient_bootstrap_ci",
    "rank_biserial",
    "tost_paired",
]
