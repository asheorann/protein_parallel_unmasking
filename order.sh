#!/usr/bin/env bash
# Ablation: does generation ORDER matter with the partition held fixed?
# Same blocks, four orderings. Separates "which positions share a block" from
# "which block goes first".
set -euo pipefail
cd "$(dirname "$0")/../.."

for ORDER in natural burial size random; do
  python -m blocks.cli bound \
    --pdb-dir "${PDB_DIR:-blocks/data/pdb}" \
    --device  "${DEVICE:-cuda}" \
    --strategies burial anticluster contiguous \
    --order-by "$ORDER" \
    --limit "${LIMIT:-50}" \
    --out "blocks/results/bound_order_${ORDER}.jsonl"
done
