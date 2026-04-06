#!/usr/bin/env bash
# scripts/reproduce/run_ablations.sh
# Reproduce Table 3 ablation study on WordNet-Mammals (Section 7.4).

set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "$(dirname "$0")/../..")"

VARIANTS=(geonet geonet_no_hel geonet_no_gaa geonet_no_rom geonet_fixed_c geonet_eucl)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

echo "=== Ablations: ${#VARIANTS[@]} variants × ${#SEEDS[@]} seeds ==="

for VARIANT in "${VARIANTS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "  → ${VARIANT} seed=${SEED}"
        python train.py \
            --config configs/ablations.yaml \
            --seed   "${SEED}" \
            model.name="${VARIANT}" \
            2>&1 | tail -3
    done
done

echo ""
echo "Aggregating ablation results..."
for VARIANT in "${VARIANTS[@]}"; do
    echo "--- ${VARIANT} ---"
    python evaluate.py \
        --config configs/ablations.yaml \
        --all_seeds \
        model.name="${VARIANT}"
done
