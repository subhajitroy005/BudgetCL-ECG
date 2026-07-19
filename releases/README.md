# Releases

| File | Contents |
|---|---|
| `v1.0.0-arxiv.md` | release notes for the initial public release |
| `release_manifest.json` | tag, commit, checkpoint hash, cohort sizes, verification results |
| `checksums.sha256` | SHA-256 for every released CSV, manifest, figure, and generated table |
| `arxiv_v1_source.tar.gz` | arXiv submission package (built by `make arxiv-package`) |

## Verify a download

```bash
sha256sum -c releases/checksums.sha256
```

## Large artifacts

Model binaries and raw datasets are **not** in Git. Attach them to the GitHub
Release or archive them on Zenodo, and record their hashes here.
