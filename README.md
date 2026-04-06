# GeoNet: Hyperbolic and Riemannian Geometry as Inductive Biases for Deep Neural Network Architecture Design

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch 2.3](https://img.shields.io/badge/PyTorch-2.3-orange.svg)](https://pytorch.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.19444522)

**Paper:** Hyperbolic and Riemannian Geometry as Inductive Biases for Deep Neural Network Architecture Design  
**Author:** Pamba Shatson Fasco, PhD Candidate, Department of Computer Science, Kampala International University
**Submitted to:** Journal of Machine Learning Research (JMLR), 2026  
**Repository DOI:** 10.5281/zenodo.19444522

---

## Overview

This repository contains the complete source code, pre-trained checkpoints, dataset preprocessing scripts, experiment configuration files, and Jupyter notebooks referenced in the paper. All reported results are reproducible by following the instructions below.

**GeoNet** introduces three architectural innovations:
- **HEL** — Hyperbolic Embedding Layers (Poincaré ball model, learnable curvature)
- **ROM** — Riemannian Optimization Module (curvature-adaptive RiemannianAdam)
- **GAA** — Geometry-Aware Attention (geodesic-distance attention logits)

---

## Repository Structure

```
geonet/
├── geonet/                    # Core library
│   ├── __init__.py
│   ├── layers/                # HEL, hyperbolic linear, activations
│   │   ├── __init__.py
│   │   ├── hyperbolic_embedding.py
│   │   ├── hyperbolic_linear.py
│   │   └── activations.py
│   ├── models/                # GeoNet graph, NLP, vision models
│   │   ├── __init__.py
│   │   ├── geonet_graph.py
│   │   ├── geonet_nlp.py
│   │   ├── geonet_vision.py
│   │   └── baselines.py
│   ├── optim/                 # ROM: Riemannian optimization
│   │   ├── __init__.py
│   │   └── riemannian_adam.py
│   ├── attention/             # GAA: Geometry-Aware Attention
│   │   ├── __init__.py
│   │   └── geodesic_attention.py
│   └── utils/                 # Manifold math, metrics, logging
│       ├── __init__.py
│       ├── manifold.py
│       ├── metrics.py
│       └── reproducibility.py
├── scripts/
│   ├── preprocess/            # Dataset preprocessing
│   │   ├── wordnet.py
│   │   ├── ogbn_arxiv.py
│   │   ├── snli.py
│   │   ├── cifar100.py
│   │   └── run_all.sh
│   └── reproduce/             # Reproduce all paper results
│       ├── run_wordnet.sh
│       ├── run_ogbn.sh
│       ├── run_snli.sh
│       ├── run_multinli.sh
│       ├── run_cifar100.sh
│       ├── run_imagenet.sh
│       ├── run_ablations.sh
│       └── reproduce_all.sh
├── configs/                   # YAML experiment configurations
│   ├── geonet_wordnet.yaml
│   ├── geonet_ogbn_arxiv.yaml
│   ├── geonet_snli.yaml
│   ├── geonet_cifar100.yaml
│   └── ablations.yaml
├── notebooks/
│   ├── 01_reproduce_table2.ipynb
│   ├── 02_reproduce_table3_ablations.ipynb
│   └── 03_curvature_sensitivity.ipynb
├── tests/
│   ├── test_manifold.py
│   ├── test_hel.py
│   ├── test_gaa.py
│   └── test_rom.py
├── data/
│   └── checksums.txt
├── train.py                   # Main training entry point
├── evaluate.py                # Evaluation entry point
├── setup.py
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/[author]/geonet.git
cd geonet
pip install -e .
```

### 2. Using Docker (recommended for full reproducibility)

```bash
docker build -t geonet:v1.0 .
docker run --gpus all -v $(pwd)/data:/workspace/data geonet:v1.0 \
    bash scripts/reproduce/reproduce_all.sh
```

### 3. Preprocess all datasets

```bash
bash scripts/preprocess/run_all.sh
# Verify checksums
python -c "import geonet.utils.reproducibility as r; r.verify_checksums('data/checksums.txt')"
```

### 4. Reproduce a single experiment (e.g., WordNet)

```bash
python train.py --config configs/geonet_wordnet.yaml --seed 0
```

### 5. Reproduce all seeds and generate Table 2

```bash
bash scripts/reproduce/run_wordnet.sh      # runs seeds 0-9
jupyter nbconvert --to notebook --execute notebooks/01_reproduce_table2.ipynb
```

---

## Reproducing All Paper Results

```bash
bash scripts/reproduce/reproduce_all.sh
```

**Estimated compute:** ~3,100 GPU-hours on 4 × NVIDIA A100 80GB.  
Per-experiment budgets are documented in `configs/` and Appendix C of the paper.

---

## Pre-trained Checkpoints

Pre-trained checkpoints for all 480 model-seed combinations (48 model-dataset pairs × 10 seeds) are archived at:

```
https://doi.org/10.5281/zenodo.19444522
```

Download and verify:

```bash
python scripts/reproduce/download_checkpoints.py --verify
```

---

## Citation

```bibtex
@article{[author]2026geonet,
  title   = {Hyperbolic and Riemannian Geometry as Inductive Biases for
             Deep Neural Network Architecture Design},
  author  = {Pamba Shatson Fasco},
  journal = {Journal of Machine Learning Research},
  year    = {2026},
  volume  = {XX},
  pages   = {1--XX},
  doi     = {10.5281/zenodo.19444522}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
