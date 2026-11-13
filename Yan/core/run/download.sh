#!/bin/bash
#SBATCH -J p2_dl
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_ENABLE_HXET=0
# Run from anywhere: resolve this folder's root from the script path.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY_BIN="${PY:-python}"        # override: PY=./venv2/bin/python
"$PY_BIN" - <<'PY'
from huggingface_hub import snapshot_download
for r in ["facebook/esm2_t30_150M_UR50D", "facebook/esmfold_v1", "airkingbd/dplm_650m"]:
    print("downloading", r, flush=True)
    p = snapshot_download(r, allow_patterns=["*.json","*.bin","*.txt","*.model","*.safetensors"])
    print("OK", r, p, flush=True)
PY
du -sh "$HF_HOME/hub/models--facebook--esmfold_v1" || true
echo DONE
