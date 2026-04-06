#!/usr/bin/env bash
# scripts/reproduce/run_imagenet.sh
# ImageNet-1K requires the dataset to be downloaded manually.
# Set IMAGENET_ROOT to your ImageNet directory before running.
# Expected structure: $IMAGENET_ROOT/{train,val}/{synset_id}/*.JPEG
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "$(dirname "$0")/../..")"

IMAGENET_ROOT="${IMAGENET_ROOT:-data/imagenet}"

if [ ! -d "${IMAGENET_ROOT}/train" ]; then
    echo "ERROR: ImageNet not found at ${IMAGENET_ROOT}."
    echo "  Download from https://image-net.org and set IMAGENET_ROOT."
    exit 1
fi

echo "=== ImageNet-1K (IMAGENET_ROOT=${IMAGENET_ROOT}) ==="
for SEED in {0..9}; do
    python train.py \
        --config configs/geonet_cifar100.yaml \
        --seed   "${SEED}" \
        dataset.name=imagenet \
        dataset.root="${IMAGENET_ROOT}" \
        model.num_classes=1000 \
        model.num_superclass=0 \
        2>&1 | tail -3
done
python evaluate.py \
    --config configs/geonet_cifar100.yaml \
    --all_seeds \
    dataset.name=imagenet \
    dataset.root="${IMAGENET_ROOT}" \
    model.num_classes=1000
