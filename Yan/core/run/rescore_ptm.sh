#!/bin/bash
#SBATCH -J p2_rescore
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
# Run from anywhere: resolve this folder's root from the script path.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python}"        # override: PY="${PY:-python}"

# HF's ESMFold `ptm` is a batch-level maximum (openfold compute_tm ends in an
# argmax over the whole batch), so every sequence in a fold batch received the
# same value and the foldability flag that thresholds on pTM was corrupted.
# metrics.py now recomputes pTM per sequence from ptm_logits. Re-score the
# decisive files; outputs go to *_scored2.json so the originals survive.
echo "### step 1: verify the fix is batch-invariant ###"
$PY core/tools/verify_ptm.py

for F in C_revised_self A_oracle_pdb C_reparam_w0.0 C_reparam_w0.75; do
  echo "### re-scoring $F ###"
  $PY -m core.run_score --gen idea_a_contacts/results/${F}.json \
      --out idea_a_contacts/results/${F}_scored2.json --fold-batch 16 --pppl-n 8
done
echo DONE
