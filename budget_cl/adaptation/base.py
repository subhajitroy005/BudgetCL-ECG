"""The adaptation-method interface.

Every arm implements the SAME interface, which is what makes the comparison
fair. If one arm could reach into training differently from another, the
between-arm ordering would no longer be attributable to the allocation.

Fixed for every arm:

* each patient resets to the same frozen source checkpoint -- no adaptation
  state carries between patients;
* early stopping uses a slice of the ADAPTATION union only, never the test
  segment;
* replay is source-domain and frozen before adaptation begins;
* the byte total is asserted before training starts.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from budget_cl.types import AdaptationResult, ArmConfig, PatientAdaptationData, PatientTestData

__all__ = ["AdaptationMethod"]


@runtime_checkable
class AdaptationMethod(Protocol):
    """Protocol every arm implementation satisfies."""

    def prepare(self, model: Any, config: ArmConfig) -> None:
        """Set the trainable scope and verify it against the arm's declaration.

        Implementations must freeze everything outside their declared scope and
        call :func:`budget_cl.models.assert_trainable_scope`.
        """
        ...

    def adapt(
        self,
        patient_data: PatientAdaptationData,
        replay_buffer: Any | None,
    ) -> AdaptationResult:
        """Adapt to one patient from the frozen source checkpoint."""
        ...

    def evaluate(self, test_data: PatientTestData) -> Any:
        """Evaluate on the held-out later segment.

        Must return a confusion matrix; the primary metric is recomputed from
        it rather than accumulated during training.
        """
        ...
