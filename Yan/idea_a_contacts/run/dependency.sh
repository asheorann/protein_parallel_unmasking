#!/bin/bash
#SBATCH -J p2_dep
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 8:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
# Run from anywhere: resolve this folder's root from the script path.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python}"        # override: PY="${PY:-python}"

echo "### smoke ###"
$PY -m idea_a_contacts.experiments.run_dependency --states 2 --reveals 0.5 --group 8 \
    --n-probe 24 --batch 4 --out idea_a_contacts/results/E_dep_smoke.json

echo "### full ###"
$PY -m idea_a_contacts.experiments.run_dependency --states 12 --reveals 0.25 0.50 0.75 \
    --group 8 16 --n-probe 40 --batch 8 \
    --out idea_a_contacts/results/E_dependency.json
echo DONE
