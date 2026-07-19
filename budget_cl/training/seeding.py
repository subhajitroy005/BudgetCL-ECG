"""Seed every source of randomness in one call.

Six independent streams affect a result. Seeding some but not others produces
runs that look reproducible until the day they are not, so they are all set
together here.
"""

from __future__ import annotations

import os
import random

import numpy as np

__all__ = ["PRIMARY_SEEDS", "set_all_seeds"]

#: Seeds used for every primary result in the paper.
PRIMARY_SEEDS = (42, 43, 44, 45, 46)


def set_all_seeds(seed: int) -> None:
    """Seed Python, NumPy, hashing, and TensorFlow if it is importable.

    Replay selection, label selection, and data shuffling all derive from these
    streams, so one call fixes the whole run.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
    except ImportError:
        return  # statistics and accounting do not need TensorFlow
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
