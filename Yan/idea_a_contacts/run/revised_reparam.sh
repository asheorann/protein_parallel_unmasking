#!/bin/bash
#SBATCH -J p2_rep
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
N=${N:-48}

# The decisive test for the revised method. The absorbing decoder sits about
# 30 pLDDT below the one DPLM ships, so a rule that helps only there cannot
# matter for a real system. Here the warm-up runs inside the reparameterised
# remasking decoder, where the baseline reaches 80+, using the FREE
# self-attention signal that the deposited-structure measurement showed is
# the only one with information while masked.
for W in 0.0 0.35 0.5 0.75; do
  echo "########## reparam, self signal, warmup=$W ##########"
  $PY -m idea_a_contacts.experiments.run_generate --lengths 200 --steps 25 50 100 200 \
      --selectors reparam_confidence reparam_contact reparam_contact_shuffled \
      --n $N --batch 16 --contact-mode self --contact-size 150 \
      --tau-mode quantile --tau-q 0.90 --warmup $W \
      --out idea_a_contacts/results/C_reparam_w${W}.json
  $PY -m core.run_score --gen idea_a_contacts/results/C_reparam_w${W}.json \
      --out idea_a_contacts/results/C_reparam_w${W}_scored.json --fold-batch 12 --pppl-n 8
done
echo DONE
