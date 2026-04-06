#!/usr/bin/env bash
# scripts/reproduce/run_ogbn.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "$(dirname "$0")/../..")"
MODELS=(geonet gcn gatv2 graphsage_v2 hgcn_plus kappa_gcn_v2)
echo "=== ogbn-arxiv ==="
for MODEL in "${MODELS[@]}"; do
    for SEED in {0..9}; do
        echo "  → ${MODEL} seed=${SEED}"
        python train.py --config configs/geonet_ogbn_arxiv.yaml --seed "${SEED}" model.name="${MODEL}" 2>&1 | tail -3
    done
done
for MODEL in "${MODELS[@]}"; do
    python evaluate.py --config configs/geonet_ogbn_arxiv.yaml --all_seeds model.name="${MODEL}"
done
