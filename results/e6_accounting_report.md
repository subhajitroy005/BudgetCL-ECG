# E6 Two-Axis Memory Accounting (Task T1)

Analytical byte accounting. **No hardware runtime, SRAM, latency, or energy quantity is reported here** -- every number is arithmetic over the statically known 6,643-parameter graph, not an observed device quantity.

## Assumptions

- Architecture: T=66, d_model=16, d_ff=128, n_heads=1, key_dim=16, 198-sample beat + 2 RR, 5 classes, 1 encoder block (PAPER1_STATUS.md).

- Optimizer Adam (2 moments); trainable weights FP32; replay INT8; packed 1-byte alignment (paper headline basis).

- Persistent axis from `budget_cl.memory.accounting.account_arm` (the paper's verified single source of truth).

- Transient axis, batch 1. R1: FP32, naive retain-all. R2: FP16 retain-all. R3: FP16 + checkpoint at attention/FFN granularity with recompute.

- Attention score tensor = n_heads*T^2 = 4,356 elements (n_heads=1); this reproduces the spec's ~17 KiB estimate exactly (4,356*4 = 17,424 B).


## Two-axis summary (persistent vs peak transient)

| Arm | Trainable | Replay items | Persistent (B) | Fits 16 KiB | Transient R1 (KiB) | R2 (KiB) | R3 (KiB) |
|---|---:|---:|---:|:--:|---:|---:|---:|
| A0 | 0 | 0 | 0 | yes | 66.0 | 33.0 | 33.0 |
| A1 | 95 | 705 | 16,374 | yes | 0.2 | 0.1 | 0.1 |
| A2 | 131 | 678 | 16,383 | yes | 0.3 | 0.1 | 0.1 |
| A3 | 167 | 650 | 16,371 | yes | 0.3 | 0.1 | 0.1 |
| A4 | 223 | 62 | 16,208 | yes | 154.6 | 77.3 | 41.3 |
| A5 | 351 | 52 | 16,226 | yes | 154.6 | 77.3 | 41.3 |

The asymmetry the 16 KiB budget excludes: post-pool arms (A1-A3) run no encoder pass during adaptation, so their peak transient is a fraction of a KiB; the raw-replay encoder arms (A4/A5) run the full forward+backward and pay tens of KiB. This is a direct consequence of the replay-point cliff -- replay location decides whether the encoder graph is live during adaptation at all.


**R3 compute multiplier (A4/A5): 1.993x** forward FLOPs (recomputes the encoder block once during backward).


## Hand verification (must agree with the CSV to the byte)

### A1 (head-only, post-pool replay)

Persistent: head = 18*5+5 = 95 params. weights 380=380 + grads 380 + Adam 760 = 1520 B; replay 705 items. CSV persistent total = **16,374 B**.

Transient (R1): stored 18-d vector -> logits(5) -> probs(5); forward = 18+5+5 = 28 elems, +18 backward temp = 46 elems * 4 B = **184 B** (< 1 KiB). CSV = **184 B**. Encoder never runs.


### A4 (rank-1 encoder LoRA, raw replay)

Persistent: 95 head + 128 LoRA (4 attn projections * 2*16*1) = 223 params. trainable-state 3568=3,568 B (16 B/param, Adam) + replay 62 raw records. CSV persistent total = **16,208 B**.

Transient R1 (retain-all, FP32): sum of forward activations = 39,582 elems. Dominant terms: ffn1 8,448 + relu 8,448 + scores 4,356 + attn_weights 4,356 + thirteen 1,056-tensors 13,728 + small 246. 39,582 * 4 B = **158,328 B** = 154.6 KiB. CSV = **158,328 B**.

Transient R3 (checkpoint + recompute, FP16): retained residual-stream boundaries pos+add1+add2 (3*1,056) + head path (36) = 3,204 elems; FFN backward working set ln2(1,056)+relu(8,448)+g_relu(8,448) = 17,952 elems dominates attention (12,936). Peak = (3,204+17,952) = 21,156 elems * 2 B = **42,312 B** = 41.3 KiB. CSV = **42,312 B**.


> **ESCALATION (spec failure criterion 2):** A4 R3 peak transient exceeds ~24 KiB (41.3 KiB). The FFN hidden gradient is the driver. See BLOCKERS.md; the T3 aliasing-onto-inference-arena argument must be stated with this caveat (the marginal transient beyond the inference arena is the FFN-hidden gradient, ~8,448 elems).


## Fit check (Step 4): persistent <= 16,384 B

| Arm | Persistent (B) | Headroom (B) | Verdict |
|---|---:|---:|:--:|
| A0 | 0 | 16,384 | PASS |
| A1 | 16,374 | 10 | PASS |
| A2 | 16,383 | 1 | PASS |
| A3 | 16,371 | 13 | PASS |
| A4 | 16,208 | 176 | PASS |
| A5 | 16,226 | 158 | PASS |

**All arms fit: True.**

