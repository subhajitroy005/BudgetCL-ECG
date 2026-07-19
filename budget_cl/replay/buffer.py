"""Fixed-capacity source replay buffer.

Invariants this class enforces, each corresponding to a claim in the paper:

* **Fixed capacity.** The item count comes from the byte solver, so the buffer
  cannot grow past what the budget allows.
* **Source-only contents.** Exemplars are drawn from DS1. Target-patient beats
  never enter the buffer, so replay is a fixed persistent cost rather than a
  growing one, and the target test segment can never leak in through replay.
* **Frozen across patients.** The buffer is built once per (arm, seed) and
  reset identically for every patient. No adaptation state carries between
  patients.
* **Deterministic.** Given a seed, the selected exemplars are reproducible;
  :mod:`budget_cl.replay.manifests` records exactly which ones were chosen.
"""

from __future__ import annotations

import numpy as np

from .selectors import class_balanced_selection, random_selection

__all__ = ["ReplayBuffer"]

_SELECTORS = {
    "class_balanced": class_balanced_selection,
    "random": random_selection,
}


class ReplayBuffer:
    """A frozen buffer of source-domain exemplars.

    Args:
        capacity: Item count, as derived from the byte budget.
        selector: ``"class_balanced"`` (used in the paper) or ``"random"``.
        seed: Selection seed; identical seeds reproduce identical buffers.

    Raises:
        ValueError: on negative capacity or an unknown selector.
    """

    def __init__(self, capacity: int, selector: str = "class_balanced", seed: int = 42) -> None:
        if capacity < 0:
            raise ValueError(f"capacity must be non-negative, got {capacity}")
        if selector not in _SELECTORS:
            raise ValueError(f"unknown selector {selector!r}; expected one of {sorted(_SELECTORS)}")
        self.capacity = int(capacity)
        self.selector = selector
        self.seed = int(seed)
        self._indices: np.ndarray | None = None
        self._values: np.ndarray | None = None
        self._labels: np.ndarray | None = None

    def fill(self, source_values: np.ndarray, source_labels: np.ndarray) -> None:
        """Select and freeze the buffer contents from the SOURCE pool.

        Args:
            source_values: Candidate exemplars; must be DS1 only.
            source_labels: Matching class indices.

        Raises:
            ValueError: if values and labels disagree in length.
        """
        source_values = np.asarray(source_values)
        source_labels = np.asarray(source_labels)
        if len(source_values) != len(source_labels):
            raise ValueError(
                f"values ({len(source_values)}) and labels ({len(source_labels)}) "
                "must have the same length"
            )
        if self.capacity == 0:
            self._indices = np.array([], dtype=int)
            self._values = source_values[:0]
            self._labels = source_labels[:0]
            return

        idx = _SELECTORS[self.selector](source_labels, self.capacity, self.seed)
        self._indices = idx
        self._values = source_values[idx]
        self._labels = source_labels[idx]

    @property
    def indices(self) -> np.ndarray:
        """Indices selected from the source pool (for the replay manifest)."""
        if self._indices is None:
            raise RuntimeError("buffer has not been filled; call fill() first")
        return self._indices

    @property
    def values(self) -> np.ndarray:
        """Stored exemplar values."""
        if self._values is None:
            raise RuntimeError("buffer has not been filled; call fill() first")
        return self._values

    @property
    def labels(self) -> np.ndarray:
        """Stored exemplar labels."""
        if self._labels is None:
            raise RuntimeError("buffer has not been filled; call fill() first")
        return self._labels

    def __len__(self) -> int:
        return 0 if self._values is None else len(self._values)

    def class_counts(self, num_classes: int = 5) -> dict[int, int]:
        """Stored items per class, for verifying the balance the selector aimed at."""
        return {c: int(np.sum(self.labels == c)) for c in range(num_classes)}
