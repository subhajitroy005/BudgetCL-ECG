"""The primary metric: per-patient macro-F1 over PRESENT classes.

.. warning::

   This is the single easiest thing to get wrong in this project, and getting
   it wrong once cost a full rework. Read this before computing any headline
   number.

Definition
----------
For patient ``p``, let ``C_p`` be the N/S/V classes that actually occur in that
patient's TEST segment::

    C_p        = { c in {N, S, V} : n_(p,c) >= 1 }
    macroF1_p  = (1 / |C_p|) * sum_{c in C_p} F1_(p,c)

Rules, all of which matter:

* Classes ABSENT from a patient's test segment are OMITTED from the mean.
  They are **not** scored as zero. Scoring absent classes as zero reads about
  0.15 lower and is a different metric (A0 = 0.509 instead of 0.666).
* F is excluded from the primary patient metric; fusion beats are reported
  separately in the pooled beat-level table.
* Q is excluded entirely (unsupported in this cohort).
* Empty-denominator F1 uses ``zero_division=0``.
* The metric is recomputed from the saved 5x5 CONFUSION MATRIX, never read
  from a stored scalar field.

Any hand analysis that reads a stored ``macro_f1_nsv`` field instead of calling
:func:`macro_f1_present` is computing something else.
"""

from __future__ import annotations

import numpy as np

__all__ = ["NSV_INDICES", "macro_f1_present", "per_class_f1", "present_classes"]

# Row/column indices of N, S, V in the 5x5 AAMI (N, S, V, F, Q) confusion matrix.
NSV_INDICES = (0, 1, 2)


def present_classes(
    confusion: np.ndarray,
    class_indices: tuple[int, ...] = NSV_INDICES,
) -> list[int]:
    """Classes with at least one TRUE beat in this patient's test segment.

    Presence is defined by the row sum (true support), not the column sum
    (predicted count): a class the model hallucinates but that never occurs
    is still absent.
    """
    cm = np.asarray(confusion, dtype=float)
    return [c for c in class_indices if cm[c].sum() > 0]


def per_class_f1(confusion: np.ndarray, class_index: int) -> float:
    """F1 for one class from the confusion matrix, 0.0 on an empty denominator."""
    cm = np.asarray(confusion, dtype=float)
    tp = cm[class_index, class_index]
    fn = cm[class_index].sum() - tp
    fp = cm[:, class_index].sum() - tp
    denom = 2.0 * tp + fp + fn
    if denom == 0:
        return 0.0  # zero_division=0
    return float(2.0 * tp / denom)


def macro_f1_present(
    confusion: np.ndarray,
    class_indices: tuple[int, ...] = NSV_INDICES,
) -> float:
    """Primary metric: macro-F1 over the classes present in the test segment.

    Args:
        confusion: 5x5 AAMI confusion matrix for one patient and one seed.
        class_indices: Classes eligible for the mean (N/S/V by default).

    Returns:
        Macro-F1 over present classes, or ``nan`` when no eligible class
        occurs at all (which would make the mean undefined rather than zero).
    """
    cm = np.asarray(confusion, dtype=float)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"confusion matrix must be square, got shape {cm.shape}")

    present = present_classes(cm, class_indices)
    if not present:
        return float("nan")
    return float(np.mean([per_class_f1(cm, c) for c in present]))
