#!/bin/bash
#SBATCH -J p2_Arp
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 10:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
# Run from anywhere: resolve this folder's root from the script path.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python}"        # override: PY="${PY:-python}"
N=${N:-48}

# Idea A inside the decoder DPLM actually ships. The absorbing sampler the
# proposal implies scores about 55 pLDDT in our harness; DPLM's own
# reparameterised decoder scores 83.6, matching its published 84.3. The
# rule therefore has to be tested where the quality is, not only where the
# proposal's framing puts it. Forward passes and the survivor count per
# step are unchanged; only WHICH positions survive changes.
echo "########## contact rule inside the reparameterised decoder ##########"
$PY -m idea_a_contacts.experiments.run_generate --lengths 200 --steps 25 50 100 200 \
    --selectors reparam_confidence reparam_contact reparam_contact_shuffled \
                reparam_anti_contact \
    --n $N --batch 12 --contact-mode self --tau-mode quantile --tau-q 0.99 \
    --out idea_a_contacts/results/A_reparam.json
$PY -m core.run_score --gen idea_a_contacts/results/A_reparam.json \
    --out idea_a_contacts/results/A_reparam_scored.json --fold-batch 12 --pppl-n 10
echo DONE
