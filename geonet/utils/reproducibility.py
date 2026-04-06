"""
geonet/utils/reproducibility.py
────────────────────────────────
Utilities ensuring full reproducibility of all paper experiments.

Implements the NeurIPS 2024 Reproducibility Checklist requirements
(Pineau et al., 2022) referenced in paper Appendix D.

Usage
-----
    from geonet.utils.reproducibility import seed_everything, verify_checksums

    seed_everything(42)          # call before any data loading or model init
    verify_checksums("data/checksums.txt")
"""

import os
import sys
import random
import hashlib
import json
import logging
import platform
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Seed management
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    """Set all random seeds for full reproducibility.

    Covers Python, NumPy, PyTorch CPU/GPU, and CUDA determinism flags.
    Must be called before any dataset loading or model initialisation.

    Parameters
    ----------
    seed : int
        The random seed. Paper experiments use seeds 0–9.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Enforce deterministic CUDA algorithms (may reduce throughput slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)

    # Required for torch.use_deterministic_algorithms — see PyTorch docs
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info(f"[Reproducibility] All seeds set to {seed}.")


# ─────────────────────────────────────────────────────────────────────────────
# Checksum verification
# ─────────────────────────────────────────────────────────────────────────────

def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 of a file in chunks (handles large dataset files)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksums(checksums_path: str, data_root: str = "data") -> bool:
    """Verify SHA-256 checksums of all preprocessed dataset files.

    Reads checksums from `data/checksums.txt` (format: <hash>  <relative_path>)
    and raises AssertionError if any file does not match.

    Parameters
    ----------
    checksums_path : str  — path to checksums file
    data_root      : str  — root directory for relative paths

    Returns
    -------
    bool — True if all checksums pass.
    """
    checksums_path = Path(checksums_path)
    if not checksums_path.exists():
        logger.warning(f"Checksums file not found: {checksums_path}. Skipping.")
        return True

    data_root = Path(data_root)
    passed, failed = 0, []

    with open(checksums_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            expected_hash, rel_path = line.split(None, 1)
            file_path = data_root / rel_path.strip()
            if not file_path.exists():
                logger.warning(f"  MISSING: {file_path}")
                failed.append(str(file_path))
                continue
            actual_hash = _sha256(file_path)
            if actual_hash == expected_hash:
                passed += 1
                logger.debug(f"  OK: {rel_path}")
            else:
                failed.append(str(file_path))
                logger.error(
                    f"  CHECKSUM MISMATCH: {rel_path}\n"
                    f"    expected: {expected_hash}\n"
                    f"    actual:   {actual_hash}"
                )

    if failed:
        raise AssertionError(
            f"Checksum verification failed for {len(failed)} file(s):\n"
            + "\n".join(f"  {p}" for p in failed)
        )

    logger.info(f"[Checksums] All {passed} files verified successfully.")
    return True


def write_checksums(file_paths: list, output_path: str, data_root: str = "data") -> None:
    """Compute and write SHA-256 checksums for a list of files.

    Call this after preprocessing to generate data/checksums.txt.
    """
    output_path = Path(output_path)
    data_root = Path(data_root)
    with open(output_path, "w") as f:
        f.write(f"# GeoNet dataset checksums — generated {datetime.now().isoformat()}\n")
        f.write("# Format: <sha256>  <relative_path>\n\n")
        for fp in file_paths:
            fp = Path(fp)
            h = _sha256(fp)
            rel = fp.relative_to(data_root) if fp.is_relative_to(data_root) else fp
            f.write(f"{h}  {rel}\n")
    logger.info(f"Checksums written to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Environment snapshot
# ─────────────────────────────────────────────────────────────────────────────

def get_environment_info() -> Dict:
    """Capture full software environment for reproducibility logging.

    Returns a dict with Python, PyTorch, CUDA, and key library versions
    matching the paper's Section 6.4 specification.
    """
    info = {
        "timestamp": datetime.now().isoformat(),
        "python":    sys.version,
        "platform":  platform.platform(),
        "torch":     torch.__version__,
        "cuda":      torch.version.cuda,
        "cudnn":     str(torch.backends.cudnn.version()),
        "gpu_count": torch.cuda.device_count(),
        "gpus": [
            torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        ],
    }
    # Optional library versions
    for lib in ["numpy", "geoopt", "torch_geometric", "transformers", "sklearn"]:
        try:
            mod = __import__(lib.replace("-", "_"))
            info[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[lib] = "not installed"
    return info


def log_environment(output_path: Optional[str] = None) -> Dict:
    """Log environment info to console and optionally to a JSON file."""
    env = get_environment_info()
    logger.info("=== Environment ===")
    for k, v in env.items():
        logger.info(f"  {k}: {v}")
    if output_path:
        with open(output_path, "w") as f:
            json.dump(env, f, indent=2)
        logger.info(f"Environment info saved to {output_path}")
    return env


# ─────────────────────────────────────────────────────────────────────────────
# Results persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: Dict, output_path: str) -> None:
    """Save experiment results dict to JSON (used by train.py and evaluate.py)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


def load_results(path: str) -> Dict:
    """Load results JSON produced by save_results."""
    with open(path) as f:
        return json.load(f)
