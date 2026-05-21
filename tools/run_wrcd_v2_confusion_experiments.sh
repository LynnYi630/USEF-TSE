#!/usr/bin/env bash
# Run the three WRCD-V2 confusion-margin fine-tuning experiments on a training server.
#
# Usage examples:
#   bash tools/run_wrcd_v2_confusion_experiments.sh
#   DEVICE=1 MEMORY=20 bash tools/run_wrcd_v2_confusion_experiments.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-0}"
MEMORY="${MEMORY:-0}"

CONFIGS=(
  "config/config-USEF-TCN-WRCD-V2-CONF005.yaml"
  "config/config-USEF-TCN-WRCD-V2-CONF010.yaml"
  "config/config-USEF-TCN-WRCD-V2-CONF005-AUXAUG.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo "[INFO] training ${config}"
  "$PYTHON" train.py \
    --config "$config" \
    --device "$DEVICE" \
    --memory "$MEMORY"
done
