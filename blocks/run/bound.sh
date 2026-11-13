#!/usr/bin/env bash
# Experiment 1: block-decomposed NLL bound vs partition.
# Cheap. No generation, no folding. Run this first; it is the theory result.
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> repo root

python -m blocks.cli bound \
  --pdb-dir   "${PDB_DIR:-blocks/data/pdb}" \
  --model     "${MODEL:-facebook/esm2_t30_150M_UR50D}" \
  --device    "${DEVICE:-cuda}" \
  --limit     "${LIMIT:-100}" \
  --max-length "${MAX_LEN:-256}" \
  --block-size "${BLOCK:-16}" \
  --n-t       "${NT:-8}" \
  --max-blocks "${MAXB:-16}" \
  --batch-size "${BS:-8}" \
  --check \
  --out "${OUT:-blocks/results/bound.jsonl}"
