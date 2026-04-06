#!/usr/bin/env bash
# scripts/reproduce/run_cifar100.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "$(dirname "$0")/../..")"
echo "=== CIFAR-100 ==="
for SEED in {0..9}; do
    python train.py --config configs/geonet_cifar100.yaml --seed "${SEED}" 2>&1 | tail -3
    # ResNet-101 baseline
    python train.py --config configs/geonet_cifar100.yaml --seed "${SEED}" model.name=resnet101 2>&1 | tail -3
done
python evaluate.py --config configs/geonet_cifar100.yaml --all_seeds
