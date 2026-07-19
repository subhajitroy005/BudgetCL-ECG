#!/usr/bin/env python3
"""reserve_replication runner.

    python experiments/run_reserve_replication.py --config configs/experiments/e16_reserve.yaml

Thin wrapper: config resolution and budget assertions happen here, the work
happens in :mod:`budget_cl`. Requires the PhysioNet recordings and a GPU.

To reproduce the paper's TABLES and FIGURES without retraining, use the released
per-cell CSVs instead:

    make statistics tables figures
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import assert_arms_fit, parse_args, prepare_run, write_provenance  # noqa: E402


def main() -> int:
    args = parse_args(__doc__)
    config, out_dir, logger = prepare_run(args)

    # Byte feasibility is checked before any training starts.
    assert_arms_fit(config, logger)

    if args.dry_run:
        logger.info("dry run: config resolved and every arm fits its budget")
        write_provenance(config, out_dir, {"dry_run": True})
        return 0

    logger.error(
        "Training requires the PhysioNet recordings and the source checkpoint.\n"
        "  1. make download-data && make verify-data\n"
        "  2. place source_model.keras in checkpoints/source/ && make verify-checkpoint\n"
        "  3. re-run this script\n"
        "Released per-cell results are in results/primary/ and reproduce every "
        "published table and figure without retraining."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
