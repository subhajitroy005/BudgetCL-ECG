"""Primary-metric tests.

The metric omits ABSENT classes rather than scoring them zero. That distinction
is worth ~0.15 macro-F1 and is the single easiest thing to get wrong here.
"""

from __future__ import annotations

import numpy as np
import pytest

from budget_cl.evaluation import (
    harm_profile,
    macro_f1_present,
    per_class_f1,
    present_classes,
    source_domain_change,
)


def _cm(cells: dict[tuple[int, int], float]) -> np.ndarray:
    """Build a 5x5 confusion matrix from {(true, pred): count}."""
    m = np.zeros((5, 5))
    for (true, pred), n in cells.items():
        m[true, pred] = n
    return m


def test_absent_classes_are_omitted_not_zeroed():
    """Perfect N and S with V absent scores 1.0, not 0.667."""
    cm = _cm({(0, 0): 100, (1, 1): 10})
    assert macro_f1_present(cm) == pytest.approx(1.0)


def test_present_classes_uses_true_support():
    """Presence is defined by the true row, not by predictions."""
    # V is never a true label but is predicted once -> still absent.
    cm = _cm({(0, 0): 100, (0, 2): 1, (1, 1): 10})
    assert present_classes(cm) == [0, 1]


def test_zero_division_is_zero_not_nan():
    """A present class the model never gets right scores 0.0."""
    cm = _cm({(0, 0): 50, (1, 0): 10})   # S present, never predicted
    assert per_class_f1(cm, 1) == 0.0
    assert macro_f1_present(cm) == pytest.approx(np.mean([per_class_f1(cm, 0), 0.0]))


def test_no_present_class_returns_nan():
    """An undefined mean is nan, not a misleading zero."""
    assert np.isnan(macro_f1_present(np.zeros((5, 5))))


def test_f_and_q_excluded_from_primary_metric():
    """Only N/S/V enter the primary patient metric."""
    with_f = _cm({(0, 0): 100, (1, 1): 10, (3, 3): 5})
    without_f = _cm({(0, 0): 100, (1, 1): 10})
    assert macro_f1_present(with_f) == pytest.approx(macro_f1_present(without_f))


def test_non_square_matrix_rejected():
    with pytest.raises(ValueError, match="square"):
        macro_f1_present(np.zeros((5, 3)))


def test_source_change_sign_convention():
    """Negative means source performance FELL."""
    assert source_domain_change(0.70, 0.75) == pytest.approx(-0.05)
    assert source_domain_change(0.80, 0.75) == pytest.approx(+0.05)
    # -0.027 is a SMALLER loss than -0.136
    assert source_domain_change(0.723, 0.75) > source_domain_change(0.614, 0.75)


def test_harm_profile_threshold_changes_the_story():
    """Sub-threshold declines are not counted as meaningful harm."""
    gains = np.array([0.30, 0.10, 0.01, -0.005, -0.01, -0.20])
    profile = harm_profile(gains, epsilon=0.02)
    assert profile.worsened == 3            # threshold-free
    assert profile.harmed_meaningful == 1   # only -0.20 clears the threshold
    assert profile.worst_change == pytest.approx(-0.20)
    assert profile.harm_rate == pytest.approx(1 / 6)


def test_harm_profile_requires_data():
    with pytest.raises(ValueError):
        harm_profile(np.array([]))
