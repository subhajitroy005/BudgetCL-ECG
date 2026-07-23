# BudgetCL-ECG reproduction entry points.
#
# The dependency chain every published number follows:
#   frozen YAML config -> experiment runner -> patient/seed CSV
#     -> statistics -> table/figure generator -> LaTeX manuscript
#
# Targets that need PhysioNet recordings are marked DATA. Everything else runs
# from the released CSVs in results/ and needs no raw signal data.
# Prefer an active virtualenv's python, then `python`, then `python3`.
# Debian-family systems ship python3 without a `python` alias, so hardcoding
# `python` makes every target fail with exit 127 on a stock install.
PYTHON ?= $(shell command -v python 2>/dev/null || command -v python3)
TAG := v1.0.0-arxiv

.PHONY: help install install-dev download-data verify-data verify-checkpoint \
        download-checkpoint run-primary run-ablations run-reserve run-split-first statistics \
        figures tables verify-paper paper reproduce-paper test lint \
        arxiv-package release-report check-release-report release-tag check-tag audit clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies and the package (editable)
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

install-dev:  ## Install development and test dependencies
	$(PYTHON) -m pip install -r requirements-dev.txt
	pre-commit install

download-data:  ## DATA: fetch MIT-BIH, INCART and SVDB from PhysioNet
	$(PYTHON) scripts/download_datasets.py

verify-data:  ## Check the local dataset layout against the manifests
	$(PYTHON) scripts/verify_dataset_manifest.py

download-checkpoint:  ## Fetch the source checkpoint and verify its hash
	$(PYTHON) scripts/download_checkpoint.py

verify-checkpoint:  ## Verify the source checkpoint SHA-256
	$(PYTHON) scripts/verify_checkpoint_hash.py

run-primary:  ## DATA: E7 primary A0-A5 evaluation (21 records x 5 seeds)
	$(PYTHON) experiments/run_primary.py --config configs/experiments/e7_primary.yaml

run-ablations:  ## DATA: E9 controlled replay-versus-plasticity ablations
	$(PYTHON) experiments/run_ablation.py --config configs/experiments/e9_ablation.yaml

run-reserve:  ## DATA: E16 1 KiB implementation-reserve replication
	$(PYTHON) experiments/run_reserve_replication.py --config configs/experiments/e16_reserve.yaml

run-split-first:  ## DATA: E17 split-first preprocessing sensitivity
	$(PYTHON) experiments/run_split_first_sensitivity.py --config configs/experiments/e17_split_first.yaml

statistics:  ## Recompute bootstrap, Wilcoxon, Holm and TOST from result CSVs
	$(PYTHON) scripts/run_statistics.py

figures:  ## Regenerate every paper figure from released CSVs
	$(PYTHON) scripts/make_figures.py

tables:  ## Regenerate every LaTeX table from released CSVs
	$(PYTHON) scripts/make_tables.py

verify-paper:  ## Check every manuscript number against the released artifacts
	$(PYTHON) scripts/verify_manuscript_numbers.py
	pytest -q tests/test_manuscript_values.py

verify-values:  ## Guard: locked-estimator headline values vs value map (R9)
	pytest -q tests/test_manuscript_values.py

audit:  ## Run the leakage and reproducibility audit
	$(PYTHON) scripts/leakage_audit.py

paper:  ## Compile the manuscript
	cd manuscript && latexmk -pdf main.tex

reproduce-paper:  ## Full chain: verify -> statistics -> tables -> figures -> paper
	$(PYTHON) scripts/reproduce_paper.py

test:  ## Run the test suite
	pytest -v

lint:  ## Ruff and mypy
	ruff check .
	mypy budget_cl

arxiv-package:  ## Build the arXiv source archive
	bash scripts/package_arxiv.sh

release-report:  ## Regenerate release_manifest.json, release notes, README badges
	$(PYTHON) scripts/generate_release_report.py

check-release-report:  ## Fail if the committed release records drift from a fresh run
	$(PYTHON) scripts/generate_release_report.py --check

release-tag:  ## Create the release tag (REFUSES if it already exists)
	@test -z "$$(git status --porcelain)" || { \
	    echo "refusing: working tree is dirty"; exit 1; \
	}
	@if git rev-parse "$(TAG)" >/dev/null 2>&1; then \
	    echo "refusing: tag $(TAG) already exists"; \
	    echo "A published release tag is immutable. Create a new version"; \
	    echo "instead of moving it: v1.0.1-arxiv, v1.1.0, v2.0.0-journal."; \
	    exit 1; \
	fi
	git tag -a "$(TAG)" \
	    -m "Initial research and reproducibility release corresponding to arXiv v1"
	git push origin main
	git push origin "$(TAG)"

check-tag:  ## Fail if the tag does not point at HEAD
	@test "$$(git rev-parse HEAD)" = "$$(git rev-list -n 1 $(TAG) 2>/dev/null)" \
	  && echo "OK: $(TAG) points at HEAD" \
	  || { echo "STALE: $(TAG) -> $$(git rev-list -n 1 $(TAG) 2>/dev/null), HEAD -> $$(git rev-parse HEAD)"; \
	       echo "A published tag is immutable: cut a new version rather than moving it."; exit 1; }

clean:  ## Remove build and LaTeX artifacts
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	cd manuscript && latexmk -C 2>/dev/null || true
