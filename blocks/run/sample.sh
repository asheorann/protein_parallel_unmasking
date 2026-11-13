#!/usr/bin/env bash
# Experiment 2: generation quality vs NFE, one curve per partition.
# Much more expensive than bound.sh. FOLD=0 skips ESMFold.
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> repo root

python -m blocks.cli sample \
  --pdb-dir   "${PDB_DIR:-blocks/data/pdb}" \
  --model     "${MODEL:-facebook/esm2_t30_150M_UR50D}" \
  --device    "${DEVICE:-cuda}" \
  --limit     "${LIMIT:-30}" \
  --max-length "${MAX_LEN:-200}" \
  --block-size "${BLOCK:-16}" \
  --steps     ${STEPS:-8 16 32 64 128} \
  --n-seeds   "${SEEDS:-5}" \
  --out blocks/results/samples.jsonl

if [ "${FOLD:-1}" = "1" ]; then
  python -m blocks.cli fold \
    --samples blocks/results/samples.jsonl \
    --device "${DEVICE:-cuda}" \
    --chunk-size "${CHUNK:-64}" \
    --out blocks/results/folded.jsonl
else
  cp blocks/results/samples.jsonl blocks/results/folded.jsonl
fi
