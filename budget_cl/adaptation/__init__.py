"""Patient-specific adaptation methods, one per arm.

All arms share the :class:`~budget_cl.adaptation.base.AdaptationMethod`
interface so the comparison stays fair; :mod:`budget_cl.adaptation.factory` is
the single construction point.
"""

from __future__ import annotations

from .base import AdaptationMethod
from .factory import arm_config, trainable_scope_for

__all__ = ["AdaptationMethod", "arm_config", "trainable_scope_for"]
