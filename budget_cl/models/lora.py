"""LoRA attention for encoder-level adaptation (arms A4, A5).

Convention used throughout the repository::

    Delta_W = B @ A
    A in R^(r x d_in)     random init
    B in R^(d_out x r)    ZERO init  ->  Delta_W = 0 at step 0

Because B is zero-initialised the adapted model starts exactly at the source
function, which is what makes an L2 penalty on B (arms R1/R2) a source anchor
costing ZERO persistent bytes: the frozen base weights are the anchor.

Scale ambiguity, stated plainly
-------------------------------
Penalising B alone is NOT scale-invariant: ``BA = (cB)(A/c)``, so the same
functional update can be given an arbitrarily small penalty. The paper reports
this as a low-cost, memory-matched baseline and explicitly does NOT claim it
regularizes the functional update ``BA``. Do not describe it as EWC.

Why LoRA lives here and not above the pool
------------------------------------------
Post-pool replay is stored AFTER the encoder, so it cannot train anything
inside the encoder -- there is nothing to backpropagate through. Encoder
adaptation therefore requires raw replay at 203 B/record instead of 21 B,
i.e. roughly 11x fewer exemplars at the same 16 KiB. That incompatibility is
not an implementation wart; it is the volume-versus-depth trade-off the arms
measure.
"""


from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from budget_cl.memory import account_state
from budget_cl.utils.logging import get_logger

logger = get_logger(__name__)

BUDGET_BYTES = 16384

# Training state per trainable FP32 param: 4 B weight + 8 B Adam (m, v) + 4 B
# gradient. SGD-momentum would be 12 and plain SGD 8 — an ablation lever.
B_PARAM_ADAM = 16
B_PARAM_SGDM = 12
B_PARAM_SGD = 8

# Bytes to store one INT8 replay sample at each viable tap.
#   post-pool: the 18-dim fuse_rr vector already contains the projected RR pair.
#   raw:       198 ECG samples + the 2-value RR pair, which the model also needs
#              as input. The frozen spec's table says 198 B for a raw window;
#              that omits the RR pair, so the honest figure for a replayable
#              training sample is 200 B. Flagged rather than silently adopted.
POSTPOOL_BYTES = 18
RAW_ECG_BYTES = 198
RAW_RR_BYTES = 2
RAW_BYTES = RAW_ECG_BYTES + RAW_RR_BYTES


@tf.keras.utils.register_keras_serializable(package="budget_cl")
class LoRAAttention(layers.Layer):
    """Frozen multi-head self-attention with additive low-rank deltas on the
    Q, K, V and O projections.

    Keras' MultiHeadAttention exposes no hook for injecting a delta into its
    internal EinsumDense projections, so the forward pass is re-implemented
    here over weights copied from the trained layer. Correctness is checked,
    not assumed: each LoRA B matrix is zero-initialised, so a freshly built
    layer is mathematically identical to the MultiHeadAttention it replaces,
    and `verify_lora_backbone` asserts that equality numerically before any
    arm is allowed to train.

    Delta for a projection with input x (B, S, E) is  (x @ A) @ B  reshaped to
    the projection's head layout, giving 2*E*r trainable params per projection.
    """

    def __init__(self, num_heads: int, key_dim: int, rank: int,
                 anchor_l2: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.rank = rank
        # Source-anchor regularization. B is zero-initialised and the delta is
        # (x @ A) @ B, so penalising B penalises deviation from the frozen
        # source function directly. This costs no persistent bytes: the anchor
        # IS the frozen base weight, which is already stored for inference.
        self.anchor_l2 = float(anchor_l2)

    def build(self, input_shape):
        e = int(input_shape[-1])
        h, d, r = self.num_heads, self.key_dim, self.rank
        self.embed_dim = e

        def frozen(name, shape):
            return self.add_weight(name=name, shape=shape, initializer="zeros", trainable=False)

        self.wq, self.bq = frozen("wq", (e, h, d)), frozen("bq", (h, d))
        self.wk, self.bk = frozen("wk", (e, h, d)), frozen("bk", (h, d))
        self.wv, self.bv = frozen("wv", (e, h, d)), frozen("bv", (h, d))
        self.wo, self.bo = frozen("wo", (h, d, e)), frozen("bo", (e,))

        def lora(tag):
            # A is random, B is zero -> delta starts at exactly 0 (identity init).
            a = self.add_weight(name=f"lora_a_{tag}", shape=(e, r),
                                initializer=tf.keras.initializers.GlorotUniform(seed=0),
                                trainable=True)
            # Source anchor: L2 on the B factor of the low-rank delta.
            #
            # KNOWN LIMITATION, stated rather than hidden: the functional update
            # is the product AB, and AB = (cA)(B/c), so penalising B alone is
            # scale-ambiguous -- it constrains the parametrisation, not the
            # function. ||AB||_F^2 is the better-posed penalty. It is not used
            # here because computing it per step via add_loss() inside call()
            # destabilised the Keras 3 training loop (cells stalled). The paper
            # reports this baseline as ||B||_F^2 and flags the ambiguity.
            b = self.add_weight(
                name=f"lora_b_{tag}", shape=(r, e), initializer="zeros",
                trainable=True,
                regularizer=(tf.keras.regularizers.l2(self.anchor_l2)
                             if self.anchor_l2 > 0 else None))
            return a, b

        self.aq, self.bq_l = lora("q")
        self.ak, self.bk_l = lora("k")
        self.av, self.bv_l = lora("v")
        self.ao, self.bo_l = lora("o")

        super().build(input_shape)

    def _heads(self, flat):
        """(B, S, E) -> (B, S, H, D)"""
        shape = tf.concat([tf.shape(flat)[:2], [self.num_heads, self.key_dim]], axis=0)
        return tf.reshape(flat, shape)

    def _project(self, x, w, b, a, b_lora):
        base = tf.einsum("bse,ehd->bshd", x, w) + b
        delta = tf.matmul(tf.matmul(x, a), b_lora)          # (B, S, E)
        return base + self._heads(delta)

    def call(self, x):
        q = self._project(x, self.wq, self.bq, self.aq, self.bq_l)
        k = self._project(x, self.wk, self.bk, self.ak, self.bk_l)
        v = self._project(x, self.wv, self.bv, self.av, self.bv_l)

        q = q / tf.sqrt(tf.cast(self.key_dim, q.dtype))
        scores = tf.einsum("bqhd,bkhd->bhqk", q, k)
        weights = tf.nn.softmax(scores, axis=-1)
        ctx = tf.einsum("bhqk,bkhd->bqhd", weights, v)      # (B, S, H, D)

        out = tf.einsum("bqhd,hde->bqe", ctx, self.wo) + self.bo
        flat = tf.reshape(ctx, tf.concat([tf.shape(ctx)[:2], [self.num_heads * self.key_dim]], 0))
        return out + tf.matmul(tf.matmul(flat, self.ao), self.bo_l)

    def load_base_weights(self, mha: layers.MultiHeadAttention) -> None:
        """Copy the trained projection weights out of a Keras MHA."""
        for w, b, dense in (
            (self.wq, self.bq, mha._query_dense),
            (self.wk, self.bk, mha._key_dense),
            (self.wv, self.bv, mha._value_dense),
            (self.wo, self.bo, mha._output_dense),
        ):
            w.assign(dense.kernel)
            b.assign(dense.bias)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(num_heads=self.num_heads, key_dim=self.key_dim, rank=self.rank)
        return cfg


@tf.keras.utils.register_keras_serializable(package="budget_cl")
class ResidualAdapter(layers.Layer):
    """Rank-r bottleneck with a residual connection: z + B(tanh(A z)).

    Sits ABOVE the pool, so it is a nonlinear re-map of frozen features — it
    cannot recover information the encoder already discarded, which is exactly
    the hypothesis A2/A3 test. Zero-init B makes it start as the identity, so
    A2/A3 begin from A1's function and can only improve on it by training.

    Params: 2 * dim * r (no biases), i.e. 36 at r=1 for the 18-dim tap.
    """

    def __init__(self, rank: int, **kwargs):
        super().__init__(**kwargs)
        self.rank = rank

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.down = self.add_weight(
            name="down", shape=(dim, self.rank),
            initializer=tf.keras.initializers.GlorotUniform(seed=0), trainable=True)
        self.up = self.add_weight(
            name="up", shape=(self.rank, dim), initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, z):
        return z + tf.matmul(tf.tanh(tf.matmul(z, self.down)), self.up)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(rank=self.rank)
        return cfg


def build_above_pool_arm(prototype_dim: int, num_classes: int, rank: int,
                         l2_reg: float = 1e-4) -> tf.keras.Model:
    """A1 (rank=0) / A2 (rank=1) / A3 (rank=2): trainable module on the frozen
    post-pool vector. rank=0 is the paper-exact 95-param linear head."""
    inp = tf.keras.Input(shape=(prototype_dim,), name="prototype_input")
    z = ResidualAdapter(rank, name=f"adapter_r{rank}")(inp) if rank > 0 else inp
    out = layers.Dense(
        num_classes, activation="softmax", dtype="float32",
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name="class_probabilities",
    )(z)
    return tf.keras.Model(inp, out, name=f"AbovePoolArm_r{rank}")


def build_lora_backbone(backbone: tf.keras.Model, rank: int,
                        num_classes: int, l2_reg: float = 1e-4,
                        anchor_l2: float = 0.0) -> tf.keras.Model:
    """A4 (rank=1) / A5 (rank=2): rebuild the backbone with LoRA attention.

    Everything except the LoRA deltas and the classifier head is frozen. The
    graph is reconstructed rather than patched in place because Keras cannot
    swap a layer inside an existing functional model. All frozen weights are
    copied from `backbone`, so the only difference from the trained model is
    the (zero-initialised) LoRA delta — verified by `verify_lora_backbone`.
    """
    from training.models.tiny_transformer import AddPositionEmbedding

    mha = backbone.get_layer("encoder1_mha")
    embed_dim = backbone.get_layer("patch_embed").filters
    reg = tf.keras.regularizers.l2(l2_reg)

    ecg_in = tf.keras.Input(shape=backbone.get_layer("ecg_input").output.shape[1:], name="ecg_input")
    rr_in = tf.keras.Input(shape=backbone.get_layer("rr_input").output.shape[1:], name="rr_input")

    patch = layers.Conv1D(
        embed_dim, kernel_size=backbone.get_layer("patch_embed").kernel_size[0],
        strides=backbone.get_layer("patch_embed").strides[0], padding="valid",
        name="patch_embed", trainable=False)
    x = patch(ecg_in)
    pos = AddPositionEmbedding(name="pos_embed", trainable=False)
    x = pos(x)

    rr_proj = layers.Dense(rr_in.shape[-1], use_bias=False, name="rr_proj", trainable=False)
    rr = rr_proj(rr_in)

    ln1 = layers.LayerNormalization(epsilon=1e-6, name="encoder1_ln1", trainable=False)
    attn = LoRAAttention(mha.num_heads, mha.key_dim, rank, anchor_l2=anchor_l2,
                         name="encoder1_lora_mha")
    ln2 = layers.LayerNormalization(epsilon=1e-6, name="encoder1_ln2", trainable=False)
    ffn1 = layers.Dense(backbone.get_layer("encoder1_ffn1").units,
                        activation=backbone.get_layer("encoder1_ffn1").activation,
                        name="encoder1_ffn1", trainable=False)
    ffn2 = layers.Dense(embed_dim, activation=backbone.get_layer("encoder1_ffn2").activation,
                        name="encoder1_ffn2", trainable=False)
    final_ln = layers.LayerNormalization(epsilon=1e-6, name="final_ln", trainable=False)

    residual = x
    h = attn(ln1(x))
    x = layers.Add(name="encoder1_add1")([residual, h])
    residual = x
    h = ffn2(ffn1(ln2(x)))
    x = layers.Add(name="encoder1_add2")([residual, h])

    x = layers.GlobalAveragePooling1D(name="reduce_mean")(final_ln(x))
    fused = layers.Concatenate(name="fuse_rr")([x, rr])
    out = layers.Dense(num_classes, activation="softmax", dtype="float32",
                       kernel_regularizer=reg, name="class_probabilities")(fused)

    model = tf.keras.Model([ecg_in, rr_in], out, name=f"LoRABackbone_r{rank}")

    # Copy every frozen weight across, then the attention projections.
    for name in ("patch_embed", "pos_embed", "rr_proj", "encoder1_ln1", "encoder1_ln2",
                 "encoder1_ffn1", "encoder1_ffn2", "final_ln", "class_probabilities"):
        model.get_layer(name).set_weights(backbone.get_layer(name).get_weights())
    model.get_layer("encoder1_lora_mha").load_base_weights(mha)

    for layer in model.layers:
        if layer.name not in ("encoder1_lora_mha", "class_probabilities"):
            layer.trainable = False
    return model


def verify_lora_backbone(lora_model: tf.keras.Model, backbone: tf.keras.Model,
                         ecg: np.ndarray, rr: np.ndarray, tol: float = 5e-3) -> dict:
    """A zero-init LoRA delta must leave the network's function unchanged.

    This is the correctness gate for the hand-written attention: if the
    re-implemented forward pass disagreed with Keras' MultiHeadAttention, the
    A4/A5 numbers would be measuring a bug, not an adapter.

    `tol` is 5e-3 rather than ~1e-6 for a measured reason, not to make the test
    pass. Keras routes MHA through the fused `ops.dot_product_attention`
    kernel, which on this CPU build is itself the *less* accurate of the two:
    against an independent float64 NumPy reference the fused kernel errs by
    5.6e-4 while this layer errs by 2.6e-4. The residual is fused-kernel
    precision, not implementation drift. Argmax agreement is therefore asserted
    exactly — any real error in the attention algebra changes predictions
    wholesale, which no float32 tolerance would hide.
    """
    inputs = {"ecg_input": ecg, "rr_input": rr}
    ref = backbone.predict(inputs, verbose=0)
    got = lora_model.predict(inputs, verbose=0)
    delta = float(np.abs(ref - got).max())
    agree = float((ref.argmax(1) == got.argmax(1)).mean())
    if delta > tol or agree < 1.0:
        raise AssertionError(
            f"LoRA backbone diverges from the frozen backbone at zero-init "
            f"(max |Δp| = {delta:.3e} > {tol:.1e}, argmax agreement {agree:.4f}) — "
            f"the re-implemented attention does not reproduce MultiHeadAttention."
        )
    logger.info(
        f"LoRA backbone verified against base backbone: max |Δp| = {delta:.2e}, "
        f"argmax agreement = {agree:.4f}"
    )
    return {"max_abs_prob_delta": delta, "argmax_agreement": agree}


def arm_budget(trainable_params: int, replay_bytes_per_sample: int,
               budget_bytes: int = BUDGET_BYTES,
               bytes_per_param: int = B_PARAM_ADAM,
               optimizer: str = "adam",
               alignment: int = 1,
               include_metadata: bool = True,
               requested_items: int | None = None) -> dict:
    """Persistent-state budget for one arm.

    The original E5 draft counted only trainable parameter state plus replay
    values. The publication roadmap requires serialized labels, valid flags,
    class IDs, quantization parameters, buffer indices/counts, reservoir state,
    class counters, and alignment padding as well. ``include_metadata=True`` is
    therefore the default used by current experiments.
    """
    if include_metadata:
        if replay_bytes_per_sample == 0:
            representation = "none"
        elif replay_bytes_per_sample == POSTPOOL_BYTES:
            representation = "postpool"
        elif replay_bytes_per_sample == RAW_BYTES:
            representation = "raw"
        else:
            raise ValueError(
                f"cannot infer replay representation from {replay_bytes_per_sample} B/sample"
            )
        row = account_state(
            trainable_params=trainable_params,
            replay=representation,
            budget_bytes=budget_bytes,
            optimizer=optimizer,
            alignment=alignment,
            requested_items=requested_items,
        )
        return {
            **row,
            "bytes_per_param": int(bytes_per_param),
            "param_state_bytes": int(row["trainable_state_bytes"]),
            "replay_bytes_per_sample": int(row["replay_item_stride_bytes"]),
            "replay_budget_bytes": int(
                budget_bytes - row["trainable_state_bytes"] - row["fixed_metadata_bytes"]
            ),
            "prototypes": int(row["prototypes"]),
            "replay_bytes_used": int(row["replay_buffer_bytes"]),
            "total_bytes_used": int(row["total_bytes"]),
            "strict_metadata_accounting": True,
        }

    param_state = trainable_params * bytes_per_param
    replay_budget = budget_bytes - param_state
    if replay_budget <= 0:
        raise ValueError(
            f"Arm needs {param_state} B of training state alone — over the "
            f"{budget_bytes} B budget before a single prototype is stored."
        )
    count = (
        int(requested_items)
        if requested_items is not None
        else replay_budget // replay_bytes_per_sample if replay_bytes_per_sample else 0
    )
    used = param_state + count * replay_bytes_per_sample
    assert used <= budget_bytes, f"budget violated: {used} > {budget_bytes}"
    return {
        "trainable_params": int(trainable_params),
        "bytes_per_param": int(bytes_per_param),
        "param_state_bytes": int(param_state),
        "replay_bytes_per_sample": int(replay_bytes_per_sample),
        "replay_budget_bytes": int(replay_budget),
        "prototypes": int(count),
        "replay_bytes_used": int(count * replay_bytes_per_sample),
        "total_bytes_used": int(used),
        "budget_bytes": int(budget_bytes),
        "headroom_bytes": int(budget_bytes - used),
        "fits": True,
        "strict_metadata_accounting": False,
    }
