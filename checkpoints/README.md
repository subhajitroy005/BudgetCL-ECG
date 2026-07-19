# Checkpoints

Model binaries are **not committed to Git** (see `.gitignore`). Fetch them from
the GitHub Release or the Zenodo archive for `v1.0.0-arxiv`.

```
checkpoints/
├── source/              # the DS1 checkpoint every result adapts from
│   └── source_model.keras
├── source_variants/     # E11 class-reweighted alternatives
└── adapted/             # git-ignored; not released per patient/seed
```

## The source checkpoint

| | |
|---|---|
| File | `source_model.keras` |
| SHA-256 | `a6d4eff14caa4404eb57dc5eb9ecfcb9e9d3f1a1bf907d6d147fb5611eb79fce` |
| Parameters | 6,643 |
| Trained on | MIT-BIH DS1 (de Chazal partition) |

Verify before running anything:

```bash
make verify-checkpoint
```

Every result in the paper adapts from this one checkpoint. A reproduction that
starts from a different one moves every number while reporting no error, which
is why the hash check is a hard gate rather than a warning.

## Adapted checkpoints

630 primary cells would mean 630 adapted models. These are not released
individually; the per-cell **results** are, in `results/`. The adaptation is
deterministic given the checkpoint, config, and seed.
