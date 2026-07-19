# Contributing

This repository is a **research artifact** accompanying a specific paper. Its
first obligation is to reproduce that paper's numbers exactly, so contributions
are evaluated a little differently from a general-purpose library.

## The rule that matters most

**No published number may be typed by hand.** Every value follows:

```
frozen YAML config → experiment runner → patient/seed CSV
    → statistics script → table/figure generator → LaTeX manuscript
```

If you find yourself editing `manuscript/tables/*.tex`, stop — those are
generated. Change the generator or the data instead.

## Before opening a PR

```bash
make test        # 86 tests
make lint        # ruff + mypy
make audit       # leakage and reproducibility audit
make verify-paper
```

CI additionally fails if regenerated artifacts differ from the committed ones.

## Changing a published number

If a change moves a value that appears in the paper, say so explicitly in the PR
description, with the before/after. That is not a blocker — it is the thing
reviewers most need to see.

## Code style

- Ruff, 100-column lines, Python 3.12.
- Docstrings explain **why**, not just what. The interesting comments in this
  codebase are the ones recording a trap (absent-class metric handling, the
  201/202 subject overlap, payload vs serialized record size). Keep writing
  those.
- New scientific claims need a test. New *limitations* need a sentence in the
  README or `docs/` — understating a limitation is a worse defect here than a
  slow function.

## What not to do

- Do not commit PhysioNet recordings or model binaries.
- Do not widen a claim beyond what the statistics support. Several phrases are
  retired on purpose and `scripts/verify_manuscript_numbers.py` will flag them.
- Do not remove a `FIXED` or `ASSESSED` entry from the audit to make the output
  look cleaner. Those entries are the audit's value.
