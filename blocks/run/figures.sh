#!/usr/bin/env bash
# Figures and paste-ready markdown tables.
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> repo root

ARGS=()
[ -f blocks/results/bound.jsonl ]  && ARGS+=(--bound  blocks/results/bound.jsonl)
[ -f blocks/results/folded.jsonl ] && ARGS+=(--folded blocks/results/folded.jsonl)

python -m blocks.cli figures "${ARGS[@]}" --outdir blocks/figures
