"""Shared plumbing for experiment runners.

Runners are deliberately THIN. Everything reusable lives in :mod:`budget_cl`;
an experiment script should resolve a config, hand it to a runner, and write
provenance. A 1,500-line script containing model, preprocessing, training, and
plotting is the failure mode this structure exists to prevent.

Every run writes, beside its results:

    resolved_config.yaml   the config with all references inlined
    environment.json       interpreter, libraries, git commit
    run_log.txt            the run log
    summary.json           headline numbers and the config hash
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from budget_cl.memory import calculate_arm_memory, validate_report  # noqa: E402
from budget_cl.utils import (  # noqa: E402
    config_hash,
    environment_report,
    get_logger,
    load_config,
    repo_root,
)

__all__ = ["parse_args", "prepare_run", "write_provenance", "assert_arms_fit"]


def parse_args(description: str) -> argparse.Namespace:
    """Standard runner CLI."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Experiment YAML.")
    parser.add_argument("--output-dir", default=None, help="Override the output directory.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve and validate the config, then stop.")
    return parser.parse_args()


def prepare_run(args: argparse.Namespace) -> tuple[dict[str, Any], Path, Any]:
    """Load the config, create the output directory, and build a logger."""
    config = load_config(args.config)
    experiment_id = str(config.get("experiment_id", "EXP")).lower()
    out_dir = Path(args.output_dir) if args.output_dir else (
        repo_root() / "results" / "raw_runs" / experiment_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(experiment_id)
    logger.info("config: %s", args.config)
    logger.info("config hash: %s", config_hash(config)[:16])
    logger.info("output: %s", out_dir)
    return config, out_dir, logger


def assert_arms_fit(config: dict[str, Any], logger: Any) -> None:
    """Fail before training if any configured arm exceeds its budget.

    Catching this here rather than mid-run means an over-budget configuration
    can never produce results that quietly violate the paper's constraint.
    """
    memory_cfg = config.get("memory_config_resolved", {})
    budget = int(memory_cfg.get("budget_bytes", 16_384))
    reserve = int(memory_cfg.get("reserve_bytes", 0))

    for arm in config.get("arms", []):
        try:
            report = calculate_arm_memory(arm, budget, reserve_bytes=reserve)
        except (KeyError, ValueError) as exc:
            logger.warning("cannot account arm %s: %s", arm, exc)
            continue
        validate_report(report)
        logger.info(
            "%s: %d params, %d replay items, %d B / %d B",
            arm, report.trainable_parameters, report.replay_items,
            report.used_bytes, report.effective_budget_bytes,
        )


def write_provenance(config: dict[str, Any], out_dir: Path, summary: dict | None = None) -> None:
    """Write the resolved config, environment, and summary next to the results."""
    (out_dir / "resolved_config.json").write_text(json.dumps(config, indent=2, default=str))
    (out_dir / "environment.json").write_text(json.dumps(environment_report(), indent=2))
    payload = {
        "experiment_id": config.get("experiment_id"),
        "config_hash": config_hash(config),
        "completed_utc": datetime.now(UTC).isoformat(),
        **(summary or {}),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str))
