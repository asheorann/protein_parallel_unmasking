#!/usr/bin/env bash
# Fast self-test. No data, no downloads unless WITH_MODEL=1.
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> repo root

python -m blocks.cli smoke
echo
echo "set WITH_MODEL=1 to also exercise the model path (downloads ESM-2 150M)"
if [ "${WITH_MODEL:-0}" = "1" ]; then
  python -m blocks.cli smoke --with-model --device "${DEVICE:-cuda}"
fi
