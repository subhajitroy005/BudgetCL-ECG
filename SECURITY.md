# Security policy

This repository contains research code for offline analysis of public,
de-identified physiological databases. It is **not** a medical device, not a
diagnostic system, and not intended for clinical use.

## Reporting

For a security-relevant issue, open a GitHub issue, or contact the author
directly if the report should not be public initially.

## Scope

Relevant:

- code execution risks in the analysis or download scripts;
- unsafe deserialization of configs, checkpoints, or result files;
- accidental inclusion of credentials or private paths in the repository.

Not applicable:

- clinical safety of model predictions — this is not a clinical system;
- adversarial robustness of the classifier, which is out of scope for this work.

## Data handling

No patient-identifiable data is present. The databases used are public and
de-identified, and are **not redistributed** here. `datasets/raw/` is
git-ignored.
