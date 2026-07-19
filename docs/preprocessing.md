# Preprocessing

## Original pipeline (behind every primary number)

Applied to the **whole record** before beat extraction, inherited unchanged from
the source architecture so adaptation results are not confounded by a front-end
change.

1. **Baseline-wander removal** — cascaded median filters, 200 ms then 600 ms.
2. **Noise suppression** — 4th-order Butterworth low-pass at 35 Hz, zero-phase
   (`filtfilt`).
3. **Resampling** — polyphase to 360 Hz, external databases only.
4. **Beat windows** — 198 samples centred on the annotated R peak (99 left,
   99 right). Windows running off a record edge are dropped, not padded.
5. **R peaks** — from WFDB `atr` annotations, never a detector, so detector error
   is excluded by construction.
6. **RR features** — pre-RR and post-RR, normalised with **DS1 statistics only**
   (μ = 0.7785 s, σ = 0.4956 s) and clipped to ±2.

All three filters are **non-causal**. The pipeline could not run unmodified in a
streaming device.

## Two disclosed consequences

### One-beat decision latency

The model consumes the **following** RR interval, which is only known after the
next R peak. The evaluated task is therefore *delayed* heartbeat classification,
one beat behind the signal — not zero-latency classification. This is a property
of the source architecture and applies to every arm including the frozen
baseline, so it affects no comparison, but it does bound the deployment claim.

### Filter support crosses the chronological split

Because filtering happens before the split and the filters are non-causal,
adaptation-side samples near the boundary depend on **test-segment raw samples**.
Test labels are never involved and no test beat enters adaptation, but the test
*signal* is not strictly unseen.

The honest description of the protocol is **"offline chronological label
partitioning with non-causal whole-record preprocessing"**, not "strictly causal
chronological adaptation".

Measured backward influence (`preprocessing/influence_analysis.py`):

| Tolerance | Backward reach |
|---|---:|
| 1e-3 | 84 samples (233 ms) |
| 1e-4 | 113 samples (314 ms) |
| 1e-5 | 120 samples (333 ms) |

Shorter than one RR interval, so with a 198-sample window only the last one or
two adaptation beats per record can be affected. Empirically, 18 of 21 records
showed numerically zero distortion; three showed up to ~10% on their single worst
boundary beat.

## Split-first pipeline (E17)

Removes the dependency instead of arguing about its size:

1. locate the chronological boundary in samples;
2. cut the **raw** signal there;
3. discard 720 samples (2 s) each side — ~6× the measured reach;
4. filter the two pieces **independently**;
5. drop beats whose window straddles the cut;
6. keep a uniform 497-beat adaptation block.

`tests/test_preprocessing_boundary.py` proves the property by perturbing future
test samples and asserting adaptation features are **bit-identical**. It also
asserts the original pipeline *does* leak, since showing split-first is clean is
only meaningful alongside evidence of what it fixes.

**Result:** arm means moved by ≤0.007, mixed in sign; both encoder-vs-frozen
contrasts stayed significant; both vs-A1 stayed non-significant. Every conclusion
held. This is a *sensitivity* analysis — the headline numbers remain those of the
released pipeline, matching the pre-specified plan.

## Future work

A fully causal streaming front end would need causal baseline removal, a causal
low-pass with state carried across the boundary, and either a pre-RR-only model
or an accepted one-beat delay. Not implemented here.
