"""BudgetCL-ECG: replay-plasticity co-design under a 16 KiB adaptation budget.

Reproducible implementation behind "Replay-Plasticity Co-Design at 16 KiB for
Patient-Specific Adaptation of a Tiny ECG Transformer".

The subpackages that carry the paper's claims:

    memory      analytical persistent-state byte accounting (replay capacity is
                DERIVED from the budget, never chosen)
    replay      fixed-capacity source replay buffers, byte-exact records
    evaluation  the primary metric -- per-patient macro-F1 over PRESENT classes
    data        cohort definition, including the 201/202 subject correction
    models      tiny Transformer, LoRA, post-pool adapters, arm registry

Importing this package does not import TensorFlow.
"""

from __future__ import annotations

from .version import RELEASE_TAG, __version__

__all__ = ["RELEASE_TAG", "__version__"]
