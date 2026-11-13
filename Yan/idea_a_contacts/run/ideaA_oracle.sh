#!/bin/bash
#SBATCH -J p2_Ao
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
# Run from anywhere: resolve this folder's root from the script path.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python}"        # override: PY="${PY:-python}"
N=${N:-48}

# The decisive test for Idea A. Hand the selector a CORRECT contact map and
# see whether the mechanism helps at all. If a correct map does not beat
# confidence ordering at matched forward passes, then no improvement to
# contact prediction can rescue the idea, and the measured collapse of the
# contact signal under masking stops being the binding constraint.
echo "########## oracle: true contact maps of natural proteins ##########"
$PY -m idea_a_contacts.experiments.run_generate --lengths 200 --steps 6 12 25 50 \
    --selectors contact_oracle contact_oracle_shuffled anti_contact_oracle \
    --n $N --batch 16 --tau 0.5 \
    --oracle-file idea_a_contacts/results/A_oracle_maps.npz --out idea_a_contacts/results/A_oracle.json
$PY -m core.run_score --gen idea_a_contacts/results/A_oracle.json \
    --out idea_a_contacts/results/A_oracle_scored.json --fold-batch 12 --pppl-n 8

echo "########## self-oracle: the map of what the baseline itself makes ####"
$PY -m idea_a_contacts.oracles.self_oracle --length 200 --steps 12 25 50 --n $N \
    --batch 16 --out idea_a_contacts/results/A_self_oracle.json
$PY -m core.run_score --gen idea_a_contacts/results/A_self_oracle.json \
    --out idea_a_contacts/results/A_self_oracle_scored.json --fold-batch 12 --pppl-n 8
echo DONE
