"""Configuration tests: configs must agree with the code that consumes them."""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from budget_cl.data import MITBIH_DS2_PRIMARY_21  # noqa: E402
from budget_cl.memory import calculate_arm_memory  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"


def load(rel: str) -> dict:
    return yaml.safe_load((CONFIGS / rel).read_text())


def test_every_config_parses():
    """No config may be malformed YAML."""
    files = list(CONFIGS.rglob("*.yaml"))
    assert files, "no configs found"
    for path in files:
        assert yaml.safe_load(path.read_text()) is not None, f"{path} is empty or invalid"


def test_primary_dataset_config_matches_code():
    """The cohort YAML and the in-code cohort must not drift apart."""
    cfg = load("datasets/mitbih_ds2_primary_21.yaml")
    assert tuple(cfg["records"]) == MITBIH_DS2_PRIMARY_21
    assert "202" in cfg["excluded_records"]
    assert cfg["selection"]["uses_test_burden"] is False


def test_arm_configs_match_the_byte_solver():
    """Replay counts in the arm configs are generated, so they must match."""
    for arm, fname in [
        ("A1", "a1_head"), ("A2", "a2_adapter_r1"), ("A3", "a3_adapter_r2"),
        ("A4", "a4_lora_r1"), ("A5", "a5_lora_r2"),
    ]:
        cfg = load(f"model/{fname}.yaml")
        primary = calculate_arm_memory(arm)
        reserved = calculate_arm_memory(arm, reserve_bytes=1024)
        assert cfg["arm"] == arm
        assert cfg["trainable_parameters"] == primary.trainable_parameters
        assert cfg["replay"]["items_primary"] == primary.replay_items
        assert cfg["replay"]["items_reserved"] == reserved.replay_items
        assert cfg["budget"]["used_bytes_primary"] == primary.used_bytes


def test_memory_configs_use_serialized_record_sizes():
    """Byte totals must use serialized sizes (203/21), not payloads (200/18)."""
    cfg = load("memory/budget_16k.yaml")
    assert cfg["replay"]["raw_record_bytes"] == 203
    assert cfg["replay"]["postpool_record_bytes"] == 21


def test_reserved_budget_is_consistent():
    cfg = load("memory/budget_16k_reserved.yaml")
    assert cfg["budget_bytes"] - cfg["reserve_bytes"] == cfg["effective_budget_bytes"] == 15_360


def test_e17_config_declares_the_guard_and_label_count():
    cfg = load("experiments/e17_split_first.yaml")
    assert cfg["boundary_guard_samples_each_side"] == 720
    assert cfg["adaptation"]["labels"] == 497
    assert cfg["arms"] == ["A0", "A1", "A4", "A5"]


def test_original_pipeline_is_marked_non_causal():
    """The released front end must never be advertised as causal."""
    cfg = load("preprocessing/original_whole_record.yaml")
    assert cfg["causal"] is False
    assert cfg["whole_record_filtering"] is True
    assert cfg["known_limitations"]["filter_support_crosses_split"] is True
