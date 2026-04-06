#!/usr/bin/env bash
# scripts/reproduce/run_wordnet.sh
# Reproduce all WordNet-Mammals results (Table 2, Section 7.1).
# Runs GeoNet + 5 baselines × seeds 0–9.

set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "$(dirname "$0")/../..")"

MODELS=(geonet gcn gatv2 graphsage_v2 hgcn_plus kappa_gcn_v2)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

echo "=== WordNet-Mammals: ${#MODELS[@]} models × ${#SEEDS[@]} seeds ==="

for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "  → ${MODEL} seed=${SEED}"
        python train.py \
            --config configs/geonet_wordnet.yaml \
            --seed   "${SEED}" \
            model.name="${MODEL}" \
            2>&1 | tail -5
    done
done

echo ""
echo "Aggregating results..."
for MODEL in "${MODELS[@]}"; do
    echo "--- ${MODEL} ---"
    python evaluate.py \
        --config     configs/geonet_wordnet.yaml \
        --all_seeds \
        model.name="${MODEL}"
done
