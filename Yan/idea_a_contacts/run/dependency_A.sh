#!/bin/bash
#SBATCH -J p2_depA
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 4:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
# Run from anywhere: resolve this folder's root from the script path.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python}"        # override: PY="${PY:-python}"
# Part A across the full reveal range, with probe positions forced masked in
# every member of a chunk. Both oracle sets, so the answer does not depend on
# the ad-hoc selection.
$PY -m idea_a_contacts.experiments.run_dependency --part A --reveals 0.10 0.25 0.50 0.75 \
    --n-probe 40 --batch 8 --states 6 \
    --oracle-file idea_a_contacts/results/A_oracle_pdb.npz --out idea_a_contacts/results/E_depA_pdb.json
$PY -m idea_a_contacts.experiments.run_dependency --part A --reveals 0.10 0.25 0.50 0.75 \
    --n-probe 40 --batch 8 --states 6 \
    --oracle-file idea_a_contacts/results/A_oracle_cath.npz --out idea_a_contacts/results/E_depA_cath.json
echo DONE
