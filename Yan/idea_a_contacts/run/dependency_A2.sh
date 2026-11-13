#!/bin/bash
#SBATCH -J p2_depA2
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
# Rerun Part A keeping per-chain rows, so AUROC(dependency | true contact)
# can be split by CATH class. This is the linchpin: the oracle helps on
# beta-rich folds (+7.47) and hurts on alpha-rich ones (-4.59), and the
# prediction is that contacts predict the model's own coupling BETTER in
# beta folds. If they do, the whole account is coherent.
$PY -m idea_a_contacts.experiments.run_dependency --part A --reveals 0.10 0.25 0.50 0.75 \
    --n-probe 40 --batch 8 --states 6 \
    --oracle-file idea_a_contacts/results/A_oracle_cath.npz --out idea_a_contacts/results/E_depA_cath2.json
echo DONE
