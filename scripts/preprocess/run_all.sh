#!/usr/bin/env bash
# scripts/preprocess/run_all.sh
# ─────────────────────────────────────────────────────────────────────────────
# Preprocess all six datasets used in the paper (Appendix B).
# Run this before any training scripts.
#
# Usage:  bash scripts/preprocess/run_all.sh
#
# Expected runtime: ~15–30 minutes (network-dependent; downloads ~8 GB).
# All outputs are written to data/; checksums are written to data/checksums.txt.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

echo "=============================================="
echo "GeoNet — Dataset Preprocessing (Appendix B)"
echo "=============================================="

# ── 1. WordNet-Mammals ────────────────────────────────────────────────────────
echo ""
echo "[1/5] Preprocessing WordNet-Mammals..."
python scripts/preprocess/wordnet.py \
    --output_dir data/wordnet \
    --seed 42
echo "  Done."

# ── 2. ogbn-arxiv ─────────────────────────────────────────────────────────────
echo ""
echo "[2/5] Downloading ogbn-arxiv via OGB..."
python scripts/preprocess/ogbn_arxiv.py
echo "  Done."

# ── 3. SNLI ───────────────────────────────────────────────────────────────────
echo ""
echo "[3/5] Downloading SNLI..."
python scripts/preprocess/snli.py --dataset snli
echo "  Done."

# ── 4. MultiNLI ──────────────────────────────────────────────────────────────
echo ""
echo "[4/5] Downloading MultiNLI..."
python scripts/preprocess/snli.py --dataset multinli
echo "  Done."

# ── 5. CIFAR-100 ──────────────────────────────────────────────────────────────
echo ""
echo "[5/5] Downloading CIFAR-100..."
python scripts/preprocess/cifar100.py
echo "  Done."

# ── Generate checksums ────────────────────────────────────────────────────────
echo ""
echo "Generating SHA-256 checksums → data/checksums.txt"
python - <<'EOF'
from geonet.utils.reproducibility import write_checksums
from pathlib import Path

files = sorted(Path("data").rglob("*.pt")) + \
        sorted(Path("data").rglob("*.pkl")) + \
        sorted(Path("data").rglob("*.csv"))

write_checksums([str(f) for f in files], "data/checksums.txt", data_root="data")
print(f"  Wrote checksums for {len(files)} files.")
EOF

echo ""
echo "=============================================="
echo "All datasets preprocessed successfully."
echo "Verify checksums:  python -c \"from geonet.utils.reproducibility import verify_checksums; verify_checksums('data/checksums.txt')\""
echo "=============================================="
