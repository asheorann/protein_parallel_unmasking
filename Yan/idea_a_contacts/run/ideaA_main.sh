#!/bin/bash
#SBATCH -J p2_Am
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

# The main grid. Two settings matter and both are now right:
#   ranking  DPLM's Gumbel-perturbed score. Ranking by raw confidence and
#            committing irrevocably collapses to polyleucine (131 of 200
#            residues identical), so the perturbation is not optional.
#   threshold a quantile of each sequence's own coupling values. A fixed
#            0.3 rejected 0.05 percent of pairs, so the rule never bound.
echo "########## main grid: correct ranking, binding threshold ##########"
$PY -m idea_a_contacts.experiments.run_generate --lengths 200 --steps 12 25 50 100 \
    --selectors confidence contact contact_shuffled anti_contact distance \
    --n $N --batch 16 --contact-mode self \
    --tau-mode quantile --tau-q 0.99 --out idea_a_contacts/results/Am_q99.json

echo "########## stronger threshold: reject the top 10 percent of pairs ####"
$PY -m idea_a_contacts.experiments.run_generate --lengths 200 --steps 12 25 50 \
    --selectors contact contact_shuffled --n $N --batch 16 \
    --contact-mode self --tau-mode quantile --tau-q 0.90 \
    --out idea_a_contacts/results/Am_q90.json

echo "########## the proposal as written: separate ESM-2 650M pass ##########"
$PY -m idea_a_contacts.experiments.run_generate --lengths 200 --steps 12 25 50 \
    --selectors contact contact_shuffled --n $N --batch 8 \
    --contact-mode external --contact-size 650 --contact-every 1 \
    --tau-mode quantile --tau-q 0.99 --out idea_a_contacts/results/Am_ext.json

echo "########## score ##########"
for f in Am_q99 Am_q90 Am_ext; do
  $PY -m core.run_score --gen idea_a_contacts/results/$f.json \
      --out idea_a_contacts/results/${f}_scored.json --fold-batch 12 --pppl-n 8
done
echo DONE
