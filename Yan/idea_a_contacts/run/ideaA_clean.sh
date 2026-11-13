#!/bin/bash
#SBATCH -J p2_Ac
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 16:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
# Run from anywhere: resolve this folder's root from the script path.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python}"        # override: PY="${PY:-python}"
N=${N:-48}

# The main grid, with the correct commit ranking.
#
# Our first grid ranked commits by DPLM's Gumbel-perturbed score. That is
# DPLM's RE-MASKING criterion, and as a commit ranking it is noise
# dominated: confidence ordering scored within a few points of random. With
# the clean probability of the sampled token instead, confidence ordering
# scores 73.9 against 47.4 for random at 25 passes, which reproduces the
# published spread. Every selector below shares that ranking, so the
# comparison isolates WHICH positions are committed together.
echo "########## main grid, clean commit ranking, quantile threshold ##########"
$PY -m idea_a_contacts.experiments.run_generate --lengths 200 --steps 6 12 25 50 100 200 \
    --selectors confidence random entropy contact contact_shuffled \
                anti_contact distance punt \
    --n $N --batch 16 --score-mode clean \
    --contact-mode self --tau-mode quantile --tau-q 0.99 \
    --out idea_a_contacts/results/Ac_grid.json

echo "########## threshold sensitivity ##########"
$PY -m idea_a_contacts.experiments.run_generate --lengths 200 --steps 25 \
    --selectors contact contact_shuffled --n $N --batch 16 \
    --score-mode clean --contact-mode self --tau-mode quantile --tau-q 0.90 \
    --out idea_a_contacts/results/Ac_q90.json

echo "########## oracle, clean ranking ##########"
$PY -m idea_a_contacts.experiments.run_generate --lengths 200 --steps 6 12 25 50 \
    --selectors contact_oracle contact_oracle_shuffled anti_contact_oracle \
    --n $N --batch 16 --tau 0.5 --score-mode clean \
    --oracle-file idea_a_contacts/results/A_oracle_maps.npz --out idea_a_contacts/results/Ac_oracle.json

echo "########## score ##########"
for f in Ac_grid Ac_oracle Ac_q90; do
  $PY -m core.run_score --gen idea_a_contacts/results/$f.json \
      --out idea_a_contacts/results/${f}_scored.json --fold-batch 12 --pppl-n 8
done
echo DONE
