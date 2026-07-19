# Manuscript

LaTeX source for the arXiv preprint.

```bash
make paper          # from the repo root
# or:
cd manuscript && latexmk -pdf main.tex
```

Builds to **49 pages**, 0 errors, 0 undefined references, 0 Type 3 fonts.

## `tables/` is GENERATED

Every file matching `table_primary_arms`, `table_pairwise_tests`,
`table_arm_budget`, `table_splitfirst` is produced by `scripts/make_tables.py`
from the released CSVs and carries a do-not-edit banner.

**Do not type result values into `main.tex` or edit generated tables.** Change
the data or the generator, then:

```bash
make statistics tables verify-paper
```

`scripts/verify_manuscript_numbers.py` fails if the manuscript and the released
artifacts disagree — it has already caught a case where a copy operation
silently clobbered generated tables with stale hand-maintained ones.

## arXiv packaging

```bash
make arxiv-package    # -> releases/arxiv_v1_source.tar.gz + .sha256
```

Stages only what is needed to compile: `main.tex`, `references.tex`,
`sections/`, the tables the manuscript actually inputs, and the figures it
actually includes. Refuses to ship build artifacts and warns on absolute private
paths.

Verified by extracting the archive into a clean directory and compiling it there
— which is the only test that catches a missing include.
