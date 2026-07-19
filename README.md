# BudgetCL-ECG

**Replay–plasticity co-design for patient-specific adaptation of a tiny ECG Transformer under a 16 KiB analytical adaptation-state budget.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-86%20passing-brightgreen.svg)](tests/)
[![Release](https://img.shields.io/badge/release-v1.0.0--arxiv-blue.svg)](releases/v1.0.0-arxiv.md)

---

## Overview

A 6,643-parameter ECG Transformer is adapted to an individual patient on-device.
Everything adaptation needs — trainable weights, gradients, optimizer moments,
and replay exemplars — has to fit in **one 16 KiB budget**. This repository asks
how those bytes should be split, and provides the code, frozen configurations,
released results, and manuscript source behind the answer.

**This is a software and analytical study.** No hardware runtime, peak SRAM,
latency, or energy is measured anywhere. The 16 KiB figure is an *analytical
persistent-state* budget, not a measured footprint. See
[Known limitations](#known-limitations).

## Associated paper

> **Replay–Plasticity Co-Design at 16 KiB for Patient-Specific Adaptation of a Tiny ECG Transformer**
> Subhajit Roy, 2026.

| | |
|---|---|
| arXiv | *assigned on submission* |
| Software release | `v1.0.0-arxiv` |
| Source checkpoint SHA-256 | `a6d4eff14caa4404eb57dc5eb9ecfcb9e9d3f1a1bf907d6d147fb5611eb79fce` |
| Manuscript source | [`manuscript/`](manuscript/) |

## Research question

A beat window is 198 samples. Push it through the encoder and it *expands* to
1,056 token elements and an 8,448-element feed-forward tensor; only after global
pooling does it contract to 16. This is the **replay-point cliff**, and it means
only post-pool replay is storage-compressive:

| Replay tap | Payload | Serialized record | vs. raw |
|---|---:|---:|---:|
| Raw beat | 198 B | 203 B | 1.00× |
| Conv / encoder tokens | 1,056 B | — | 5.3× |
| FFN hidden | 8,448 B | — | 42.7× |
| **Post-pool + RR** | **18 B** | **21 B** | **0.09×** |

So the naive budget-optimal move is to store as many post-pool exemplars as
possible. But post-pool exemplars enter the graph *after* the encoder, so they
cannot train anything inside it. Buying encoder plasticity means raw replay at
203 B/record — roughly 11× fewer exemplars.

**Given a fixed number of bytes, how much should be spent on remembering the
source versus on the freedom to change the representation?**

## Main contribution

The *fixed-byte co-design*: measuring that replay cost is non-monotonic in depth
for this architecture, formulating a single persistent-state budget that replay
volume and trainable depth must share, generating budget-matched configurations
from that constraint, and evaluating the resulting allocation under
patient-sequential adaptation.

LoRA, replay, latent replay, patient-specific ECG classification, and the tiny
Transformer itself are all established prior work and none is claimed here.

## Key results

Over the corrected **21 patient-disjoint MIT-BIH DS2 records × 5 seeds
(630 adaptation cells)**:

| Arm | Trainable | Replay | Items | Per-patient macro-F1 |
|---|---:|---|---:|---:|
| A0 frozen | 0 | — | 0 | 0.666 |
| A1 head + max replay | 95 | post-pool | 705 | 0.786 |
| A2 adapter r1 | 131 | post-pool | 678 | 0.790 |
| A3 adapter r2 | 167 | post-pool | 650 | 0.775 |
| **A4 encoder LoRA r1** | 223 | raw | 62 | **0.803** |
| **A5 encoder LoRA r2** | 351 | raw | 52 | **0.809** |

Both encoder arms beat the frozen model (Holm *p* ≤ 0.008, 17/21 patients
improved). **Their direct comparison against maximum head-only replay is NOT
significant** (A4−A1 Holm *p* = 0.25; A5−A1 *p* = 0.11). Only the two
frozen-model comparisons survive Holm correction.

The honest reading is a **replay–plasticity allocation frontier with several
competitive operating points** — not "depth beats volume".

## Repository structure

```
budget_cl/        reusable implementation (models, adaptation, replay, memory,
                  training, evaluation, data, quantization, utils)
preprocessing/    ECG front ends: original (non-causal) and split-first (E17)
statistics/       bootstrap, Wilcoxon, Holm, TOST, effect sizes
configs/          frozen YAML: every experiment-defining value lives here
manifests/        exact records, subjects, and replay selections
experiments/      thin runners over budget_cl/
results/          released machine-readable per-cell and summary CSVs
figures/          plot scripts + released paper PDFs
scripts/          reproduction, verification, packaging
tests/            correctness and leakage tests
manuscript/       arXiv LaTeX source (tables/ are GENERATED)
docs/             detailed technical documentation
releases/         release notes and artifact manifests
```

### The traceability chain

Every published number follows this path, with **no manual copying**:

```
frozen YAML config → experiment runner → patient/seed CSV
    → statistics script → table/figure generator → LaTeX manuscript
```

`manuscript/tables/*.tex` carries a "GENERATED FILE — do not edit" banner, and
`scripts/verify_manuscript_numbers.py` fails if the paper and the artifacts
disagree.

## Installation

```bash
git clone https://github.com/subhajitroy005/BudgetCL-ECG.git
cd BudgetCL-ECG
make install          # pip install -r requirements.txt && pip install -e .
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate budgetcl-ecg
pip install -e .
```

**Requirements:** Python 3.12, TensorFlow 2.21.0. Reproducing the *statistics,
tables, and figures* needs neither a GPU nor TensorFlow — only NumPy, SciPy, and
Matplotlib. Retraining the full grid needs a GPU (results were produced on an
NVIDIA RTX 3050, 6 GiB) and takes several hours.

## Dataset preparation

**PhysioNet recordings are not redistributed here.** The MIT-BIH Arrhythmia
Database, INCART, and SVDB are public and remain under their own terms.

```bash
make download-data    # fetches into datasets/raw/ (git-ignored)
make verify-data      # checks the local layout against manifests/
```

See [`datasets/README.md`](datasets/README.md).

## Source checkpoint

Model binaries are not committed to Git. Fetch `source_model.keras` from the
GitHub Release or Zenodo archive into `checkpoints/source/`, then:

```bash
make verify-checkpoint
```

This is not optional ceremony: a reproduction that silently starts from a
different checkpoint moves every number and reports no error.

## Quick-start example

Byte accounting needs no data at all:

```python
from budget_cl.memory import calculate_arm_memory

report = calculate_arm_memory("A4")            # rank-1 encoder LoRA
print(report.trainable_parameters)             # 223
print(report.replay_items)                     # 62   <- DERIVED from the budget
print(report.used_bytes, report.fits)          # 16208 True

reserved = calculate_arm_memory("A4", reserve_bytes=1024)
print(reserved.replay_items)                   # 57   <- 1 KiB firmware reserve
```

Replay capacity is never chosen by hand; it is solved from the budget.

## Reproduce the primary experiment

```bash
make run-primary      # DATA + GPU: E7, 6 arms x 21 records x 5 seeds
```

## Reproduce all tables and figures

From the released CSVs, no raw data and no GPU:

```bash
make statistics       # bootstrap, Wilcoxon, Holm -> results/
make tables           # -> manuscript/tables/*.tex
make figures          # -> figures/paper/*.pdf
make verify-paper     # manuscript numbers vs released artifacts
make paper            # compile the PDF

make reproduce-paper  # all of the above, in order
```

| Result file | Produces |
|---|---|
| `results/primary/E7_patient_seed_results.csv` | per-cell primary results (630 cells) |
| `results/primary/E7_arm_summary.csv` | Table: primary arms |
| `results/primary/E8_paired_tests.csv` | Table: six pre-specified comparisons |
| `results/reserve/E16_patient_seed_results.csv` | reserve replication |
| `results/regularization/E15_patient_seed_results.csv` | B-factor anchor baseline |
| `results/preprocessing_sensitivity/E17_*.csv` | Table + figure: split-first sensitivity |

## Persistent-state accounting

```
B = B_weights + B_gradients + B_optimizer + B_replay + B_labels + B_metadata + B_padding
```

**"Persistent state" means a statically reserved writable region the adaptation
procedure needs across update steps.** It is *not* total SRAM and does *not*
imply non-volatility.

- **Counted:** trainable weights, gradients, optimizer moments, replay values
  and labels, buffer metadata, alignment padding.
- **Excluded:** forward/backward activations, stack and workspace, allocator
  overhead — peak-SRAM quantities, not measured anywhere in this work.

Gradients are charged even though they are update-time working state, because an
implementation that materialises a gradient vector must reserve the region.

Payload and serialized record are kept distinct: raw is 200 B payload / **203 B
record**; post-pool is 18 B / **21 B**. Every byte total uses the serialized
size. See [`docs/memory_accounting.md`](docs/memory_accounting.md).

## Preprocessing sensitivity

The released front end filters each record **as a whole** before the
chronological split, using centred median kernels and a zero-phase Butterworth
low-pass. All are non-causal, so adaptation-side samples near the split depend
on test-segment *raw samples*. Test labels are never involved and no test beat
enters adaptation, but the test *signal* is not strictly unseen.

Measured backward influence: **84 samples at 1e-3 tolerance, 120 samples
(333 ms) at 1e-5** — under one RR interval, so at most the last one or two
adaptation beats per record.

**E17** removes the dependency: cut the raw signal at the boundary, discard a 2 s
guard each side, filter the pieces independently, rerun A0/A1/A4/A5 (420 cells).

| Arm | Whole-record | Split-first | Δ |
|---|---:|---:|---:|
| A0 | 0.666 | 0.666 | −0.000 |
| A1 | 0.786 | 0.793 | +0.007 |
| A4 | 0.803 | 0.796 | −0.006 |
| A5 | 0.809 | 0.809 | +0.000 |

A4−A0 and A5−A0 stay significant (*p* = 0.0012); both vs-A1 stay
non-significant. The shifts are small and **mixed in sign** — not the systematic
drop recovered leakage would produce. Every conclusion holds.

## Reproducibility guarantees

```bash
make test     # 86 tests
make audit    # leakage and reproducibility audit
make lint     # ruff + mypy
```

Enforced automatically:

- record 202 excluded, and **subject** disjointness asserted (not just record
  disjointness — the record-level check passes on the contaminated cohort,
  which is how the defect originally survived);
- every arm's byte total asserted against its ceiling, at 16 KiB and 15,360 B;
- replay records serialize to exactly 203 B / 21 B;
- the primary metric omits absent classes rather than scoring them zero;
- split-first adaptation features are **bit-identical** when future test samples
  are perturbed;
- released statistics reproduce the published p-values and effect sizes.

The audit reports `PASS / FIXED / ASSESSED / WARN / FAIL`. The `FIXED` entries
are kept deliberately: an audit that only ever reports PASS is evidence of weak
checks, not of correctness.

## Known limitations

1. **No hardware measurements.** No MCU runtime, peak SRAM, latency, or energy.
   The budget is analytical persistent state.
2. **No compute ratio is claimed.** Raw replay re-forwards through the encoder
   while post-pool bypasses it, so encoder adaptation costs meaningfully more —
   but an earlier analytical estimate was withdrawn as not safely derivable, and
   no replacement number is quoted.
3. **A4/A5 vs A1 is not significant.** Needs a larger cohort.
4. **E10 budget sweep is partial** (49 of 140 configurations, Adam ranks 1–2 on
   a reduced 12-record panel). No optimizer or rank-4/8 claim is made.
5. **One regularization baseline** — a B-factor anchor, which is
   scale-ambiguous (`BA = (cB)(A/c)`) and does *not* directly regularize the
   functional update. Restricted EWC, distillation, and output-consistency
   regularization are untested.
6. **External cohorts are convenience-capped and minority-support-filtered.**
   Eligibility uses test-window class labels, so the cohort is enriched for
   recordings where minority-class adaptation can be evaluated. INCART's 11
   recordings are only **6 unique subjects**; that result is descriptive.
7. **One-beat decision latency.** The model uses the *following* RR interval, so
   classification is delayed by one beat. It applies to every arm equally.
8. **No causal streaming front end.** See the sensitivity section above.

## Citation

```bibtex
@software{roy2026budgetclecg,
  author  = {Roy, Subhajit},
  title   = {BudgetCL-ECG: Replay--Plasticity Co-Design at 16 KiB},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/subhajitroy005/BudgetCL-ECG}
}
```

See [`CITATION.cff`](CITATION.cff).

## License

[MIT](LICENSE) for the software. **Dataset terms are separate:** MIT-BIH,
INCART, and SVDB remain under their PhysioNet terms and are not redistributed
here.
