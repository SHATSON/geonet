#!/usr/bin/env bash
# scripts/reproduce/reproduce_all.sh
# ─────────────────────────────────────────────────────────────────────────────
# Master script to reproduce ALL results reported in the paper.
# Runs all models × datasets × seeds 0–9 and generates all tables/figures.
#
# Estimated total compute: ~3,100 GPU-hours on 4 × NVIDIA A100 80GB.
# Per-experiment runtimes: see Appendix C, Table C2.
#
# Usage:
#   # Full reproduction (all experiments):
#   bash scripts/reproduce/reproduce_all.sh
#
#   # Single experiment only (e.g., GeoNet on WordNet, seed 0):
#   python train.py --config configs/geonet_wordnet.yaml --seed 0
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

SEEDS=(0 1 2 3 4 5 6 7 8 9)

echo "=========================================================="
echo "GeoNet — Full Paper Reproduction Script"
echo "Paper: Hyperbolic and Riemannian Geometry as Inductive"
echo "       Biases for Deep Neural Network Architecture Design"
echo "DOI:   10.5281/zenodo.XXXXXXX"
echo "=========================================================="
echo ""

# ── Step 0: Verify environment ───────────────────────────────────────────────
echo "[Step 0] Verifying environment and checksums..."
python -c "
from geonet.utils.reproducibility import log_environment, verify_checksums
log_environment()
verify_checksums('data/checksums.txt')
print('Environment OK.')
"

# ── Step 1: Table 2 — Graph benchmarks ───────────────────────────────────────
echo ""
echo "[Step 1] Table 2: WordNet-Mammals (GeoNet + all baselines, seeds 0–9)"
bash scripts/reproduce/run_wordnet.sh

echo ""
echo "[Step 1] Table 2: ogbn-arxiv (GeoNet + all baselines, seeds 0–9)"
bash scripts/reproduce/run_ogbn.sh

# ── Step 2: Section 7.2 — NLI ────────────────────────────────────────────────
echo ""
echo "[Step 2] Section 7.2: SNLI (GeoNet + baselines, seeds 0–9)"
bash scripts/reproduce/run_snli.sh

echo ""
echo "[Step 2] Section 7.2: MultiNLI (GeoNet + baselines, seeds 0–9)"
bash scripts/reproduce/run_multinli.sh

# ── Step 3: Section 7.3 — Image classification ───────────────────────────────
echo ""
echo "[Step 3] Section 7.3: CIFAR-100 (GeoNet + ResNet-101, seeds 0–9)"
bash scripts/reproduce/run_cifar100.sh

echo ""
echo "[Step 3] Section 7.3: ImageNet-1K (GeoNet + ResNet-101, seeds 0–9)"
bash scripts/reproduce/run_imagenet.sh

# ── Step 4: Table 3 — Ablation studies ───────────────────────────────────────
echo ""
echo "[Step 4] Table 3: Ablation studies on WordNet-Mammals (seeds 0–9)"
bash scripts/reproduce/run_ablations.sh

# ── Step 5: Generate all tables and figures ───────────────────────────────────
echo ""
echo "[Step 5] Generating paper tables and figures..."
jupyter nbconvert --to notebook --execute \
    notebooks/01_reproduce_table2.ipynb \
    --output notebooks/01_reproduce_table2_executed.ipynb

jupyter nbconvert --to notebook --execute \
    notebooks/02_reproduce_table3_ablations.ipynb \
    --output notebooks/02_reproduce_table3_ablations_executed.ipynb

jupyter nbconvert --to notebook --execute \
    notebooks/03_curvature_sensitivity.ipynb \
    --output notebooks/03_curvature_sensitivity_executed.ipynb

echo ""
echo "=========================================================="
echo "Reproduction complete."
echo "Results are in: outputs/"
echo "Tables/figures: notebooks/*_executed.ipynb"
echo "=========================================================="
