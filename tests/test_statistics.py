"""Statistical-method tests, including the published-value regressions."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import (
    PRESPECIFIED_FAMILY,
    holm_correct,
    paired_bootstrap_ci,
    paired_test,
    patient_bootstrap_ci,
    tost_paired,
)

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def test_holm_family_has_six_prespecified_members():
    assert len(PRESPECIFIED_FAMILY) == 6


def test_holm_is_monotone_and_conservative():
    """Adjusted p-values never decrease and never fall below raw values."""
    raw = {"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.30, "e": 0.5, "f": 0.9}
    adj = holm_correct(raw)
    assert all(adj[k] >= raw[k] for k in raw)
    ordered = [adj[k] for k in sorted(raw, key=lambda k: raw[k])]
    assert ordered == sorted(ordered)


def test_holm_smallest_is_scaled_by_family_size():
    raw = {"a": 0.001, "b": 0.5}
    assert holm_correct(raw)["a"] == pytest.approx(0.002)


def test_bootstrap_is_reproducible():
    """A fixed seed must give a bit-identical interval."""
    x = np.random.default_rng(0).normal(size=21)
    assert patient_bootstrap_ci(x, iterations=1000) == patient_bootstrap_ci(x, iterations=1000)


def test_paired_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap_ci(np.zeros(5), np.zeros(6))


def test_tost_declares_equivalence_for_tiny_differences():
    rng = np.random.default_rng(1)
    a = rng.normal(0.8, 0.01, 20)
    result = tost_paired(a, a + 0.001, delta=0.05)
    assert result.equivalent
    assert not result.multiplicity_adjusted  # policy travels with the number


def test_tost_rejects_equivalence_for_large_differences():
    rng = np.random.default_rng(2)
    a = rng.normal(0.8, 0.01, 20)
    assert not tost_paired(a, a + 0.20, delta=0.05).equivalent


def test_published_paired_tests_reproduce():
    """Regression against the released E8 table.

    If this fails, either the statistics changed or the results did -- both are
    things a reader of the paper needs to know about.
    """
    path = REPO / "results" / "primary" / "E8_paired_tests.csv"
    summary = REPO / "results" / "primary" / "E7_patient_summary.csv"
    if not (path.exists() and summary.exists()):
        pytest.skip("released statistics not present")

    scores: dict[str, dict[str, float]] = {}
    with summary.open() as fh:
        for row in csv.DictReader(fh):
            scores.setdefault(row["arm"], {})[row["record"]] = float(row["macro_f1_present"])

    with path.open() as fh:
        published = {r["comparison"]: r for r in csv.DictReader(fh)}

    for name, row in published.items():
        left, right = name.split("_vs_")
        records = sorted(scores[left])
        a = np.array([scores[left][r] for r in records])
        b = np.array([scores[right][r] for r in records])
        result = paired_test(a, b)
        assert result.mean_difference == pytest.approx(float(row["mean_difference"]), abs=5e-4)
        assert result.p_value == pytest.approx(float(row["p_wilcoxon"]), abs=1e-4)
        assert result.effect_size == pytest.approx(float(row["rank_biserial"]), abs=2e-3)


def test_only_frozen_model_comparisons_survive_holm():
    """The paper's central negative result, pinned as a test."""
    path = REPO / "results" / "primary" / "E8_paired_tests.csv"
    if not path.exists():
        pytest.skip("released statistics not present")
    with path.open() as fh:
        rows = {r["comparison"]: float(r["p_holm"]) for r in csv.DictReader(fh)}
    survivors = {k for k, v in rows.items() if v < 0.05}
    assert survivors == {"A4_vs_A0", "A5_vs_A0"}, (
        "the set of Holm survivors changed; the paper claims only the two "
        "frozen-model comparisons survive"
    )
