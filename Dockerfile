# GeoNet Docker Image — v1.0
# Reproduces the exact software environment described in paper Section 6.4.
# CUDA 12.1, Python 3.11, PyTorch 2.3.0, Geoopt 0.5.0
#
# Build:  docker build -t geonet:v1.0 .
# Run:    docker run --gpus all -v $(pwd)/data:/workspace/data geonet:v1.0 \
#             bash scripts/reproduce/reproduce_all.sh

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

LABEL maintainer="[author.email@university.edu]"
LABEL version="1.0"
LABEL description="GeoNet: Geometric Deep Learning Framework (JMLR 2026)"

# ── System dependencies ──────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    git \
    wget \
    curl \
    unzip \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /workspace

# ── Python dependencies (pinned for reproducibility) ─────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
        --index-url https://download.pytorch.org/whl/cu121 && \
    pip install torch-scatter torch-sparse torch-geometric \
        -f https://data.pyg.org/whl/torch-2.3.0+cu121.html && \
    pip install -r requirements.txt

# ── Copy source code ──────────────────────────────────────────────────────────
COPY . .
RUN pip install -e .

# ── Environment variables ─────────────────────────────────────────────────────
ENV PYTHONPATH=/workspace
ENV PYTHONHASHSEED=0
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

# ── Entrypoint ────────────────────────────────────────────────────────────────
ENTRYPOINT ["/bin/bash"]
CMD ["-c", "echo 'GeoNet v1.0 ready. Run: bash scripts/reproduce/reproduce_all.sh'"]
