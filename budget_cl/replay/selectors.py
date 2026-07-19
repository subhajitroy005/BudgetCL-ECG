"""Which source exemplars go into the replay buffer.

The paper holds the selection policy FIXED and varies only the allocation
between replay volume and trainable depth. Selector quality is a separate
question and is listed as future work; what matters here is that the policy is
deterministic given a seed, so a reviewer can reproduce the exact buffer.

The paper uses :func:`class_balanced_selection`.
"""

from __future__ import annotations

import numpy as np

__all__ = ["class_balanced_selection", "random_selection"]


def random_selection(labels: np.ndarray, n_items: int, seed: int) -> np.ndarray:
    """Uniformly sample ``n_items`` indices without replacement.

    Deployable comparator for the class-balanced policy.
    """
    rng = np.random.default_rng(seed)
    n_items = min(int(n_items), len(labels))
    return np.sort(rng.choice(len(labels), size=n_items, replace=False))


def class_balanced_selection(
    labels: np.ndarray,
    n_items: int,
    seed: int,
    num_classes: int = 5,
) -> np.ndarray:
    """Select an equal quota per present class, redistributing any shortfall.

    Minority classes (S, V, F) are precisely what patient adaptation has to
    repair, so a uniformly random buffer would mostly hold N beats and waste
    the budget. Quotas are therefore equal across classes that are actually
    present; when a class cannot fill its quota, the remainder is redistributed
    to classes that still have unused examples.

    Args:
        labels: Class index per candidate exemplar.
        n_items: Buffer capacity, as solved from the byte budget.
        seed: Seed for the per-class draw; identical seeds give identical
            buffers, which is what the replay manifests record.
        num_classes: Number of AAMI classes.

    Returns:
        Sorted array of selected indices, length <= ``n_items``.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    n_items = int(n_items)
    if n_items <= 0:
        return np.array([], dtype=int)

    pools = {c: np.where(labels == c)[0] for c in range(num_classes)}
    present = [c for c, idx in pools.items() if len(idx) > 0]
    if not present:
        return np.array([], dtype=int)

    # Shuffle once per class so the draw is deterministic under `seed`.
    for c in present:
        pools[c] = rng.permutation(pools[c])

    taken: dict[int, int] = dict.fromkeys(present, 0)
    remaining = n_items
    # Round-robin rather than a single floor division: this redistributes the
    # quota of any exhausted class instead of leaving buffer slots empty.
    while remaining > 0:
        progressed = False
        for c in present:
            if remaining == 0:
                break
            if taken[c] < len(pools[c]):
                taken[c] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break  # every class exhausted; buffer is smaller than capacity

    chosen = np.concatenate([pools[c][: taken[c]] for c in present if taken[c] > 0])
    return np.sort(chosen)
