#!/bin/bash
#SBATCH -J p2_sc
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
# Run from anywhere: resolve this folder's root from the script path.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python}"        # override: PY="${PY:-python}"

# Is the absorbing baseline a strawman? DPLM's own decoder scores the
# published 83.6 in our harness, but every absorbing arm lands near 55.
# Two candidate causes: (1) we rank commits by DPLM's Gumbel-perturbed
# score, which is its RE-MASKING criterion and may be noise-dominated as a
# commit ranking; (2) irrevocable commitment is simply much worse than
# re-predicting everything each step. This separates them.
echo "########## clean confidence vs gumbel-perturbed, absorbing ##########"
$PY -m idea_a_contacts.experiments.run_generate --lengths 200 --steps 25 200 \
    --selectors confidence random --n 48 --batch 16 \
    --score-mode clean --out idea_a_contacts/results/A_clean.json
$PY -m core.run_score --gen idea_a_contacts/results/A_clean.json \
    --out idea_a_contacts/results/A_clean_scored.json --fold-batch 12 --pppl-n 8
echo DONE
