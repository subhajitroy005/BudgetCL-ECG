"""AAMI class mapping for MIT-BIH annotation symbols.

The AAMI EC57 convention collapses the raw beat symbols into five classes:

    N  normal and bundle-branch blocks
    S  supraventricular ectopic
    V  ventricular ectopic
    F  fusion of ventricular and normal
    Q  paced / unclassifiable

Non-beat annotations (rhythm marks, artefact) are dropped BEFORE RR intervals
are computed, so RR timing is never measured across a non-beat mark.
"""

from __future__ import annotations

__all__ = ["AAMI_CLASSES", "CLASS_TO_INDEX", "SYMBOL_TO_AAMI", "BEAT_SYMBOLS", "map_symbol"]

AAMI_CLASSES: tuple[str, ...] = ("N", "S", "V", "F", "Q")
CLASS_TO_INDEX: dict[str, int] = {c: i for i, c in enumerate(AAMI_CLASSES)}

SYMBOL_TO_AAMI: dict[str, str] = {
    # N: normal, bundle-branch block, atrial/nodal escape
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    # S: supraventricular ectopic
    "A": "S", "a": "S", "J": "S", "S": "S",
    # V: ventricular ectopic
    "V": "V", "E": "V",
    # F: fusion of ventricular and normal
    "F": "F",
    # Q: paced, fusion of paced and normal, unclassifiable
    "/": "Q", "f": "Q", "Q": "Q",
}

BEAT_SYMBOLS: frozenset[str] = frozenset(SYMBOL_TO_AAMI)


def map_symbol(symbol: str) -> str | None:
    """AAMI class for an annotation symbol, or None if it is not a beat."""
    return SYMBOL_TO_AAMI.get(symbol)
