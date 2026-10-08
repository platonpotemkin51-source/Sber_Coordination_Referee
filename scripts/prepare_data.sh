#!/usr/bin/env bash
# Unpack the shipped dataset (data/*.csv.gz -> data/*.csv).
set -euo pipefail
cd "$(dirname "$0")/.."
for f in data/*.csv.gz; do
    out="${f%.gz}"
    if [ -f "$out" ]; then
        echo "skip $out (already unpacked)"
    else
        gunzip -kf "$f"
        echo "unpacked $out"
    fi
done
ls -la data
