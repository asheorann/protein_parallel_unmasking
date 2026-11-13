#!/bin/bash
#SBATCH -J p2_reftm
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 6:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
# Run from anywhere: resolve this folder's root from the script path.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python}"        # override: PY="${PY:-python}"

# ===================================================================
# Ref-TM: does a conditioned sequence adopt its SOURCE protein's fold?
#
# This is the check the conditioning result has never had. +19.9 pLDDT says
# ESMFold is confident; it does not say the structure has anything to do with
# the protein whose chemical pattern the model was handed. The CATH arm makes
# the question answerable because its pattern sources ARE structural domains
# with deposited coordinates (127 of 128 downloaded).
#
# Read the four arms against two reference points:
#   TM >= 0.5   conventional same-fold threshold
#   ~0.17-0.30  what unrelated domains of this size score (measured 0.29 on a
#               random pair from this very set)
# and against the `source` control, which folds the source's own slice and
# aligns it to its own crystal structure -- the ceiling ESMFold can reach
# here, which bounds every other row.
#
# Run by path with -I: the structure files are downloaded data, so the
# download directory must never be on the import path.
# ===================================================================

[ -s data/cath_domains/index.json ] || {
  echo "FATAL: run conditioning/patterns/fetch_cath_domains.py first" >&2; exit 1; }
$PY -c "import tmtools; print('tmtools ok')"

echo "########## K_cath_purged (the quotable row) ##########"
$PY -I conditioning/experiments/ref_tm.py \
    --scored conditioning/results/K_cath_purged_scored.json \
    --patterns data/clustered_cath/heldout_patterns.json \
    --domains data/cath_domains \
    --out conditioning/results/M_reftm_K_purged.json

echo "########## K_cath_orig ##########"
$PY -I conditioning/experiments/ref_tm.py \
    --scored conditioning/results/K_cath_orig_scored.json \
    --patterns data/clustered_cath/heldout_patterns.json \
    --domains data/cath_domains \
    --skip-source-control \
    --out conditioning/results/M_reftm_K_orig.json

echo "########## done ##########"
ls -la conditioning/results/M_reftm_*.json
