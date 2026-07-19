# Memory accounting

Every byte total in the paper comes from `budget_cl/memory/`. Replay capacity is
**derived** from the budget, never chosen:

```
N_max = floor((budget − reserve − 16P − metadata) / record_bytes)
```

## What "persistent state" means

A **statically reserved writable region the adaptation procedure needs across
update steps.** Not total SRAM. Not an implication of non-volatility.

| Category | Counted? | Persistent? | Note |
|---|---|---|---|
| Trainable weights | yes | yes | 4 B/param FP32 |
| Gradients | **yes** | no | update-time working state, but the region must be reserved |
| Optimizer moments | yes | yes | 2 × 4 B/param for Adam |
| Replay values | yes | yes | INT8 |
| Replay labels + flags | yes | yes | 3 B/record |
| Buffer metadata | yes | yes | 54 B raw / 49 B post-pool |
| Alignment padding | yes | yes | 0 B at 1-byte packing |
| Activations, stack, allocator | **no** | no | peak SRAM; never measured here |

Charging gradients is a deliberate conservative choice: excluding them would
understate what a device has to set aside.

## Per-parameter cost

```
weight 4 B + gradient 4 B + 2 Adam moments 8 B = 16 B/parameter
```

## Payload vs serialized record

Conflating these understates a budget, so they are named separately:

```
b_record = b_payload + b_label + b_flags + b_pad
```

| Location | Payload | Label+flags | **Record** |
|---|---:|---:|---:|
| Raw beat + RR | 200 B | 3 B | **203 B** |
| Post-pool + RR | 18 B | 3 B | **21 B** |

Every byte total uses the record size. The payload figures appear only when the
text is explicitly naming the tensor.

## Alignment

Both structs are all `int8`/`uint8`, so packed size, natural ABI size, and
1-byte stride coincide. Under wider alignment they do not:

| Record | Packed | 4-byte | 8-byte |
|---|---:|---:|---:|
| Raw | 203 B | 204 B | 208 B |
| Post-pool | 21 B | 24 B | 24 B |

The paper reports packed figures and states the assumption, because a
packed-state claim that silently depends on a favourable ABI is not reproducible.

## Resulting configurations

| Arm | Params | Record | Items @16 KiB | Items @15,360 B | Bytes used |
|---|---:|---|---:|---:|---:|
| A0 | 0 | — | 0 | 0 | 0 |
| A1 | 95 | 21 B | 705 | 656 | 16,374 |
| A2 | 131 | 21 B | 678 | 629 | 16,383 |
| A3 | 167 | 21 B | 650 | 601 | 16,371 |
| A4 | 223 | 203 B | 62 | 57 | 16,208 |
| A5 | 351 | 203 B | 52 | 47 | 16,226 |

The 11× gap between A1 and A4 replay counts *is* the volume-versus-depth
trade-off the paper measures.

## The firmware reserve

An arm using 16,383 of 16,384 bytes is not shippable. The paper re-solves every
arm against 15,360 B and **re-runs** the grid at the reduced counts, rather than
relabelling old results. That costs A1 49 exemplars and the encoder arms five
each — and it reshuffles the arm ordering, which is what identifies rank-1 as the
more robust operating point.

## Full-model regularization is infeasible here

Storing an FP32 parameter anchor plus a Fisher value for EWC needs
6,643 × 8 = 53,144 B — already over the ceiling before any replay. This
establishes infeasibility for *full-model* FP32 regularization only; restricted
or compressed variants remain untested.
