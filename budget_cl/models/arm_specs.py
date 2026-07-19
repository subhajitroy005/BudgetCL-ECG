"""Arm registry: the experimental grid, in one place.

Two properties are kept SEPARATE on purpose:

    replay           the representation stored in the byte budget
    trainable_scope  where gradients are allowed to flow

Separating them is what makes the E9 controls possible -- e.g. "head trained
from raw-stored replay" (B2) isolates replay location from trainable depth.
Collapsing them into a single "arm type" would make those contrasts
inexpressible.

Groups
------
A0-A5   primary arms (frozen, head-only, post-pool adapter r1/r2, encoder LoRA r1/r2)
B0-B7   controlled replay-versus-plasticity ablations (E9); B5 is infeasible at 16 KiB
R1/R2   memory-matched B-factor regularization baselines (E15), zero persistent bytes
"""


from __future__ import annotations

AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
S_IDX, V_IDX = 1, 2

BUDGET_BYTES = 16_384
POSTPOOL_BYTES = 18
RAW_ECG_BYTES = 198
RAW_RR_BYTES = 2
RAW_BYTES = RAW_ECG_BYTES + RAW_RR_BYTES


# The six A-arms are the E5 spec. The B/FC/FS arms are E9 controls. ``replay``
# is the persistent representation stored in the byte budget; ``trainable_scope``
# is where gradients are allowed to flow. Separating the two enables causal
# controls such as "head trained from raw-stored replay".
ARMS = {
    "A0": {"replay": None, "tap": None, "rank": None, "trainable_scope": "frozen",
           "trainable": "none (frozen)", "adapts_features": False},
    "A1": {"replay": "postpool", "tap": "postpool", "rank": 0, "trainable_scope": "postpool",
           "trainable": "head only", "adapts_features": False},
    "A2": {"replay": "postpool", "tap": "postpool", "rank": 1, "trainable_scope": "postpool",
           "trainable": "rank-1 above-pool adapter + head", "adapts_features": False},
    "A3": {"replay": "postpool", "tap": "postpool", "rank": 2, "trainable_scope": "postpool",
           "trainable": "rank-2 above-pool adapter + head", "adapts_features": False},
    "A4": {"replay": "raw", "tap": "raw", "rank": 1, "trainable_scope": "encoder_lora",
           "trainable": "rank-1 encoder LoRA + head", "adapts_features": True},
    "A5": {"replay": "raw", "tap": "raw", "rank": 2, "trainable_scope": "encoder_lora",
           "trainable": "rank-2 encoder LoRA + head", "adapts_features": True},

    # E9 causal matrix controls.
    "B0": {"replay": None, "tap": None, "rank": None, "trainable_scope": "frozen",
           "trainable": "frozen reference", "ablation_id": "B0", "adapts_features": False},
    "B1": {"replay": "postpool", "tap": "postpool", "rank": 0, "trainable_scope": "postpool",
           "trainable": "head + maximum post-pool replay", "ablation_id": "B1", "adapts_features": False},
    "B2": {"replay": "raw", "tap": "raw", "rank": 0, "trainable_scope": "postpool",
           "trainable": "head + maximum raw replay", "ablation_id": "B2", "adapts_features": False},
    "B3": {"replay": None, "tap": None, "rank": 1, "trainable_scope": "encoder_lora",
           "requested_items": 0, "trainable": "rank-1 encoder LoRA without replay",
           "ablation_id": "B3", "adapts_features": True},
    "B4": {"replay": "raw", "tap": "raw", "rank": 1, "trainable_scope": "encoder_lora",
           "trainable": "rank-1 encoder LoRA + maximum raw replay", "ablation_id": "B4",
           "adapts_features": True},
    "B5": {"replay": "raw", "tap": "raw", "rank": 1, "trainable_scope": "encoder_lora",
           "same_count_as": "B2", "trainable": "rank-1 encoder LoRA + B2 raw replay count",
           "ablation_id": "B5", "adapts_features": True},
    "B6": {"replay": "postpool", "tap": "postpool", "rank": 0, "trainable_scope": "postpool",
           "same_count_as": "B4", "trainable": "head + B4 post-pool replay count",
           "ablation_id": "B6", "adapts_features": False},
    "B7": {"replay": "postpool", "tap": "postpool", "rank": 1, "trainable_scope": "postpool",
           "trainable": "rank-1 post-pool adapter + maximum post-pool replay",
           "ablation_id": "B7", "adapts_features": False},

    # Memory-matched regularization baselines. LoRA B is zero-initialised and
    # the attention delta is (x @ A) @ B, so an L2 penalty on B anchors the
    # adapted model to the frozen source function. The anchor costs ZERO
    # persistent bytes because the source weights are the frozen base already
    # stored for inference -- unlike EWC, which would need a parameter copy and
    # a Fisher value per trainable weight. R1/R2 therefore isolate the question
    # "can regularization substitute for replay at the same byte cost?"
    "R1": {"replay": None, "tap": None, "rank": 1, "trainable_scope": "encoder_lora",
           "requested_items": 0, "anchor_l2": 1e-2,
           "trainable": "rank-1 encoder LoRA, source-anchor L2, no replay",
           "ablation_id": "R1", "adapts_features": True},
    "R2": {"replay": "raw", "tap": "raw", "rank": 1, "trainable_scope": "encoder_lora",
           "anchor_l2": 1e-2,
           "trainable": "rank-1 encoder LoRA, source-anchor L2 + maximum raw replay",
           "ablation_id": "R2", "adapts_features": True},

    # E9 fixed-count replay controls. All use raw-stored replay so replay count
    # is literally matched across trainable depths; above-pool models extract
    # frozen post-pool vectors from that raw buffer at training time.
    "FC_HEAD_RAW_32": {"replay": "raw", "tap": "raw", "rank": 0, "trainable_scope": "postpool",
                       "requested_items": 32, "fixed_count": 32, "trainable": "head, raw replay n=32"},
    "FC_ADAPTER_RAW_32": {"replay": "raw", "tap": "raw", "rank": 1, "trainable_scope": "postpool",
                          "requested_items": 32, "fixed_count": 32, "trainable": "post-pool adapter, raw replay n=32"},
    "FC_LORA_RAW_32": {"replay": "raw", "tap": "raw", "rank": 1, "trainable_scope": "encoder_lora",
                       "requested_items": 32, "fixed_count": 32, "trainable": "encoder LoRA, raw replay n=32"},
    "FC_HEAD_RAW_53": {"replay": "raw", "tap": "raw", "rank": 0, "trainable_scope": "postpool",
                       "requested_items": 53, "fixed_count": 53, "trainable": "head, raw replay n=53"},
    "FC_ADAPTER_RAW_53": {"replay": "raw", "tap": "raw", "rank": 1, "trainable_scope": "postpool",
                          "requested_items": 53, "fixed_count": 53, "trainable": "post-pool adapter, raw replay n=53"},
    "FC_LORA_RAW_53": {"replay": "raw", "tap": "raw", "rank": 1, "trainable_scope": "encoder_lora",
                       "requested_items": 53, "fixed_count": 53, "trainable": "encoder LoRA, raw replay n=53"},
    "FC_HEAD_RAW_64": {"replay": "raw", "tap": "raw", "rank": 0, "trainable_scope": "postpool",
                       "requested_items": 64, "fixed_count": 64, "trainable": "head, raw replay n=64"},
    "FC_ADAPTER_RAW_64": {"replay": "raw", "tap": "raw", "rank": 1, "trainable_scope": "postpool",
                          "requested_items": 64, "fixed_count": 64, "trainable": "post-pool adapter, raw replay n=64"},
    "FC_LORA_RAW_64": {"replay": "raw", "tap": "raw", "rank": 1, "trainable_scope": "encoder_lora",
                       "requested_items": 64, "fixed_count": 64, "trainable": "encoder LoRA, raw replay n=64"},

    # E9 fixed-scope replay-count controls for rank-1 encoder LoRA.
    "FS_LORA_RAW_0": {"replay": None, "tap": None, "rank": 1, "trainable_scope": "encoder_lora",
                      "requested_items": 0, "fixed_scope_count": 0, "trainable": "encoder LoRA, no replay"},
    "FS_LORA_RAW_8": {"replay": "raw", "tap": "raw", "rank": 1, "trainable_scope": "encoder_lora",
                      "requested_items": 8, "fixed_scope_count": 8, "trainable": "encoder LoRA, raw replay n=8"},
    "FS_LORA_RAW_16": {"replay": "raw", "tap": "raw", "rank": 1, "trainable_scope": "encoder_lora",
                       "requested_items": 16, "fixed_scope_count": 16, "trainable": "encoder LoRA, raw replay n=16"},
    "FS_LORA_RAW_32": {"replay": "raw", "tap": "raw", "rank": 1, "trainable_scope": "encoder_lora",
                       "requested_items": 32, "fixed_scope_count": 32, "trainable": "encoder LoRA, raw replay n=32"},
    "FS_LORA_RAW_64": {"replay": "raw", "tap": "raw", "rank": 1, "trainable_scope": "encoder_lora",
                       "requested_items": 64, "fixed_scope_count": 64, "trainable": "encoder LoRA, raw replay n=64"},
    "FS_LORA_RAW_128": {"replay": "raw", "tap": "raw", "rank": 1, "trainable_scope": "encoder_lora",
                        "requested_items": 128, "fixed_scope_count": 128, "trainable": "encoder LoRA, raw replay n=128"},
}
