"""YAML configuration loading with reference resolution.

Experiment configs reference dataset, model, memory and preprocessing configs
by path. :func:`load_config` resolves those references so a runner receives one
fully resolved dictionary -- and so the resolved config can be hashed and
written beside the results, making every run self-describing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .paths import repo_root

__all__ = ["load_config", "load_yaml", "resolve_references"]

# Keys whose values are paths to further YAML files to inline.
_REFERENCE_KEYS = (
    "dataset_config",
    "model_config",
    "memory_config",
    "training_config",
    "preprocessing_config",
    "statistics_config",
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load one YAML file.

    Raises:
        FileNotFoundError: with the resolved path, since a mistyped config
            reference is the most common failure here.
    """
    # Imported lazily so that logging, path helpers, statistics, and the byte
    # accounting stay usable in an environment without a YAML parser.
    import yaml

    p = Path(path)
    if not p.is_absolute():
        p = repo_root() / p
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open() as fh:
        return yaml.safe_load(fh) or {}


def resolve_references(config: dict[str, Any]) -> dict[str, Any]:
    """Inline referenced config files under ``<key>_resolved``.

    The original path is kept so the provenance of each value stays visible in
    the resolved config written next to the results.
    """
    resolved = dict(config)
    for key in _REFERENCE_KEYS:
        ref = config.get(key)
        if isinstance(ref, str):
            resolved[f"{key}_resolved"] = load_yaml(ref)
    return resolved


def load_config(path: str | Path) -> dict[str, Any]:
    """Load an experiment config and inline everything it references."""
    return resolve_references(load_yaml(path))
