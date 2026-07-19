# Reproducibility

## One command

```bash
make reproduce-paper
```

Runs: audit → statistics → tables → figures → LaTeX → number verification.
Non-zero exit if any stage fails.

## What is and is not reproduced

| Stage | Needs raw data? | Needs GPU? | Runtime |
|---|---|---|---|
| statistics, tables, figures | no | no | seconds |
| manuscript build | no | no | ~1 min |
| number verification | no | no | seconds |
| **retraining the grid** | **yes** | **yes** | hours |

The released per-cell CSVs are the reproducibility boundary. Everything
downstream regenerates from them on any machine. Retraining regenerates the CSVs
themselves and needs the PhysioNet recordings plus a GPU.

## Determinism

Seeded: Python, NumPy, TensorFlow, replay selection, label selection, and data
shuffling. Primary seeds are `{42, 43, 44, 45, 46}`; the bootstrap seed is fixed
at 20260719.

Exact GPU-level bit-reproducibility is **not** claimed — TensorFlow kernel
selection can vary across hardware. Patient-level conclusions are stable across
the five seeds, which is why five are averaged before any test.

## Verification gates

```bash
make test               # 86 tests
make audit              # leakage and reproducibility audit
make verify-checkpoint  # source checkpoint SHA-256
make verify-data        # local dataset layout vs manifests
make verify-paper       # manuscript numbers vs released artifacts
```

## Known environment note

If `pytest` fails during collection with `ModuleNotFoundError: No module named
'lark'`, a ROS installation on the system path is injecting a `launch_testing`
plugin. Work around it with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

## What would break reproduction

1. A different source checkpoint — hence the hard hash gate.
2. Reading `macro_f1_nsv` instead of recomputing `macro_f1_present` from the
   confusion matrix (~0.15 macro-F1 difference).
3. Including record 202 (22-record cohort) — moves every arm mean and restores a
   significance that the corrected cohort does not support.
4. Hand-editing `manuscript/tables/*.tex`, which is generated output.
