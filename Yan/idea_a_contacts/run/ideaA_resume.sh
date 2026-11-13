#!/bin/bash
#SBATCH -J p2_Ar
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

# Resume: stages 0, 0b, 1 and 2 already completed and their files are on
# disk. Scoring runs FIRST here, so the headline numbers exist even if a
# later generation stage fails again.
echo "########## score what is already generated ##########"
for f in A_fidelity A_self; do
  [ -f idea_a_contacts/results/${f}_scored.json ] && { echo "skip $f"; continue; }
  echo "--- scoring $f ---"
  $PY -m core.run_score --gen idea_a_contacts/results/$f.json \
      --out idea_a_contacts/results/${f}_scored.json --fold-batch 12 --pppl-n 10
done

echo "########## stage 3: raw-attention coupling ##########"
$PY -m idea_a_contacts.experiments.run_generate --lengths 200 --steps 12 25 50 100 \
    --selectors contact contact_shuffled --n $N --batch 16 \
    --contact-mode rawattn --tau-mode quantile --tau-q 0.99 \
    --out idea_a_contacts/results/A_rawattn.json

echo "########## stage 5: oracle contact maps ##########"
$PY -m idea_a_contacts.experiments.run_generate --lengths 200 --steps 6 12 25 50 \
    --selectors contact_oracle contact_oracle_shuffled anti_contact_oracle \
    --n $N --batch 16 --tau 0.5 \
    --oracle-file idea_a_contacts/results/A_oracle_maps.npz --out idea_a_contacts/results/A_oracle.json

echo "########## stage 6: quantile threshold on predicted contacts ##########"
$PY -m idea_a_contacts.experiments.run_generate --lengths 200 --steps 12 25 50 100 \
    --selectors contact contact_shuffled anti_contact --n $N --batch 16 \
    --contact-mode self --tau-mode quantile --tau-q 0.99 \
    --out idea_a_contacts/results/A_q99.json

echo "########## stage 8: gated to the late trajectory ##########"
$PY -m idea_a_contacts.experiments.run_generate --lengths 200 --steps 12 25 50 100 \
    --selectors contact contact_shuffled --n $N --batch 16 \
    --contact-mode self --tau-mode quantile --tau-q 0.99 \
    --contact-min-revealed 0.5 --out idea_a_contacts/results/A_gated.json

echo "########## stage 4: external ESM-2 650M contacts (the proposal) #######"
$PY -m idea_a_contacts.experiments.run_generate --lengths 200 --steps 12 25 50 \
    --selectors contact contact_shuffled --n $N --batch 8 \
    --contact-mode external --contact-size 650 --contact-every 1 \
    --tau-mode quantile --tau-q 0.99 --out idea_a_contacts/results/A_ext.json

echo "########## stage 10: self-oracle ##########"
$PY -m idea_a_contacts.oracles.self_oracle --length 200 --steps 12 25 50 --n $N \
    --batch 16 --out idea_a_contacts/results/A_self_oracle.json

echo "########## score the rest ##########"
for f in A_oracle A_q99 A_rawattn A_gated A_ext A_self_oracle; do
  [ -f idea_a_contacts/results/$f.json ] || continue
  echo "--- scoring $f ---"
  $PY -m core.run_score --gen idea_a_contacts/results/$f.json \
      --out idea_a_contacts/results/${f}_scored.json --fold-batch 12 --pppl-n 10
done
echo DONE
