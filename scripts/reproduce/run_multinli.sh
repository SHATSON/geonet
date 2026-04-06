#!/usr/bin/env bash
# scripts/reproduce/run_multinli.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "$(dirname "$0")/../..")"
echo "=== MultiNLI ==="
for SEED in {0..9}; do
    python train.py --config configs/geonet_snli.yaml --seed "${SEED}" dataset.name=multinli 2>&1 | tail -3
done
python evaluate.py --config configs/geonet_snli.yaml --all_seeds dataset.name=multinli
