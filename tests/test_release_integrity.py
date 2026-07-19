"""Release-integrity tests: the repository must describe itself consistently."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

from budget_cl.models.checkpoint_loader import EXPECTED_PARAMETERS, EXPECTED_SOURCE_SHA256
from budget_cl.version import RELEASE_TAG, __version__

REPO = Path(__file__).resolve().parents[1]


def test_version_matches_pyproject():
    """budget_cl.version and pyproject.toml must agree."""
    text = (REPO / "pyproject.toml").read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match and match.group(1) == __version__


def test_release_tag_embeds_the_version():
    assert __version__ in RELEASE_TAG


def test_citation_file_is_consistent():
    """CITATION.cff must carry the same version and license as the package."""
    yaml = pytest.importorskip("yaml")
    cff = yaml.safe_load((REPO / "CITATION.cff").read_text())
    assert str(cff["version"]) == __version__
    assert cff["license"] == "MIT"


def test_license_is_mit_everywhere():
    """One license, stated identically in LICENSE, pyproject, and CITATION."""
    assert "MIT License" in (REPO / "LICENSE").read_text()
    assert 'license = { text = "MIT" }' in (REPO / "pyproject.toml").read_text()


def test_license_disclaims_dataset_terms():
    """The software license must not appear to cover PhysioNet recordings."""
    text = (REPO / "LICENSE").read_text()
    assert "PhysioNet" in text
    assert "NOT redistributed" in text


def test_checkpoint_manifest_matches_code():
    """The pinned hash must be identical in code and in the manifest."""
    manifest = json.loads((REPO / "manifests" / "source_checkpoint_manifest.json").read_text())
    assert manifest["sha256"] == EXPECTED_SOURCE_SHA256
    assert manifest["parameters"] == EXPECTED_PARAMETERS


def test_release_config_matches_checkpoint_hash():
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load((REPO / "configs" / "release" / "arxiv_v1.yaml").read_text())
    assert cfg["checkpoint_sha256"] == EXPECTED_SOURCE_SHA256
    assert cfg["primary_cohort_records"] == 21
    assert cfg["primary_cells"] == 630


def test_gitignore_does_not_exclude_released_artifacts():
    """Result CSVs, figures and manifests are part of the release."""
    ignored = (REPO / ".gitignore").read_text()
    for pattern in ("results/primary/", "manifests/", "figures/paper/", "manuscript/tables/"):
        assert pattern not in ignored.split("\n"), f"{pattern} must not be git-ignored"


def test_raw_datasets_are_git_ignored():
    """PhysioNet recordings must never be committed."""
    ignored = (REPO / ".gitignore").read_text()
    for pattern in ("datasets/raw/", "datasets/processed/", "datasets/cache/"):
        assert pattern in ignored


def test_released_results_exclude_record_202():
    """No released result file may contain the contaminated record."""
    for csv_path in (REPO / "results").rglob("*_patient_seed_results.csv"):
        with csv_path.open() as fh:
            records = {row["record"] for row in csv.DictReader(fh)}
        assert "202" not in records, f"record 202 present in {csv_path.name}"
