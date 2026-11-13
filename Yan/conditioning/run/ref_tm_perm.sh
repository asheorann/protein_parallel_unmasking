#!/bin/bash
#SBATCH -J p2_reftmperm
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 4:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
# Run from anywhere: resolve this folder's root from the script path.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python}"        # override: PY="${PY:-python}"

# The permuted-pairing control for the Ref-TM result.
#
# Paired Ref-TM says a conditioned sequence folds like the domain whose
# pattern it was given: 0.596, with 64.6 % above the same-fold threshold,
# against 0.299 for the bit-matched random partition. One alternative remains:
# that the conditioned arm folds into some generically domain-like shape that
# would score well against ANY reference. Aligning each generated structure to
# a DIFFERENT source settles it -- that mean must fall to the unrelated-pair
# background of ~0.29.
#
# Only the two conditioned arms are re-folded; the unconditional arms are
# already at background and have nothing to distinguish.

$PY -I conditioning/experiments/ref_tm.py \
    --scored conditioning/results/K_cath_purged_scored.json \
    --patterns data/clustered_cath/heldout_patterns.json \
    --domains data/cath_domains \
    --cells "dms6|conditioned,rand6|conditioned" \
    --skip-source-control \
    --out conditioning/results/M_reftm_K_purged_perm.json
echo "########## done ##########"
