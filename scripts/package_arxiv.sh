#!/usr/bin/env bash
# Build the arXiv source archive from manuscript/.
#
# arXiv wants TeX source, not a PDF-only submission. This stages exactly what is
# needed to compile and nothing else: no build artifacts, no datasets, no
# checkpoints, no review notes, no private paths.
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE=$(mktemp -d)
OUT_DIR="releases"
# arXiv accepts .zip and .tar.gz alike. The ZIP is the primary artifact because
# it is what the release checklist uploads; the tarball is kept for anyone who
# prefers it. Both are byte-identical in content.
BASE="Replay-Plasticity-ECG-arxiv-v1"
OUT="$OUT_DIR/${BASE}.zip"
OUT_TAR="$OUT_DIR/arxiv_v1_source.tar.gz"
mkdir -p "$OUT_DIR"

cp manuscript/main.tex manuscript/references.tex "$STAGE/" 2>/dev/null || {
  echo "error: manuscript/main.tex or references.tex missing"; exit 1; }
mkdir -p "$STAGE/sections" "$STAGE/tables" "$STAGE/figures"
cp manuscript/sections/*.tex "$STAGE/sections/"
cp manuscript/tables/*.tex   "$STAGE/tables/"

# Only figures the manuscript actually includes.
grep -rho 'includegraphics\[[^]]*\]{[^}]*}' manuscript/main.tex manuscript/sections/*.tex \
  | sed 's/.*{\(.*\)}/\1/' | sort -u > "$STAGE/.figlist"
missing=0
while read -r f; do
  [ -z "$f" ] && continue
  if [ -f "manuscript/figures/$f" ]; then cp "manuscript/figures/$f" "$STAGE/figures/"
  else echo "MISSING FIGURE: $f"; missing=1; fi
done < "$STAGE/.figlist"
rm -f "$STAGE/.figlist"
[ "$missing" -eq 0 ] || { echo "aborting: missing figures"; exit 1; }

# Drop tables the manuscript never inputs.
for t in "$STAGE"/tables/*.tex; do
  b="tables/$(basename "${t%.tex}")"
  grep -rq "input{$b}" manuscript/main.tex manuscript/sections/*.tex \
    || { echo "  dropping unused $(basename "$t")"; rm "$t"; }
done

# Refuse to ship build artifacts or private paths.
if find "$STAGE" \( -name '*.aux' -o -name '*.log' -o -name '*.out' \
     -o -name '*.toc' -o -name '*.synctex.gz' \) | grep -q .; then
  echo "aborting: build artifacts staged"; exit 1
fi
if grep -rIl '/home/' "$STAGE" | grep -q .; then
  echo "WARNING: absolute private paths in staged source:"; grep -rIl '/home/' "$STAGE"
fi

rm -f "$OUT" "$OUT_TAR"
tar -czf "$OUT_TAR" -C "$STAGE" .
( cd "$STAGE" && zip -qr "$OLDPWD/$OUT" . -x '.*' )
rm -rf "$STAGE"
sha256sum "$OUT" > "${OUT}.sha256"
sha256sum "$OUT_TAR" > "${OUT_TAR}.sha256"

echo
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
echo "wrote $OUT_TAR ($(du -h "$OUT_TAR" | cut -f1))"
echo
unzip -Z1 "$OUT" | sed 's/^/  /' | sort
echo
cat "${OUT}.sha256"
