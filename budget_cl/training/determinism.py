"""Enable deterministic TensorFlow kernels where supported.

Honest scope: this reduces non-determinism, it does not eliminate it. Exact
bit-reproducibility across different GPUs and driver versions is NOT claimed --
kernel selection can still vary. That is why the paper averages five seeds
within each patient before any statistical test, so conclusions do not rest on
a single run being bit-identical.
"""

from __future__ import annotations

import os

__all__ = ["enable_determinism"]


def enable_determinism() -> bool:
    """Request deterministic ops. Returns True if TensorFlow accepted the request.

    Must be called BEFORE TensorFlow builds any graph.
    """
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    try:
        import tensorflow as tf
    except ImportError:
        return False
    try:
        tf.config.experimental.enable_op_determinism()
        return True
    except (AttributeError, RuntimeError):
        # Older TF, or a graph already exists -- the env vars still apply.
        return False
